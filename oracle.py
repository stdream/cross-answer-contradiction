"""
SLM Oracle — Ollama LLM-as-Expert
==================================
FCA 속성탐색의 oracle로서 SLM을 사용.
- Ollama API로 yes/no 질문
- JSON 구조화 프롬프트
- self-correction loop (모순 감지 시 재질의)
"""
from __future__ import annotations

import json
import logging
import requests
from dataclasses import dataclass, field

from fca_engine import FormalContext, Implication, check_consistency

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"


@dataclass
class OracleConfig:
    model: str = "qwen2.5:7b"
    temperature: float = 0.1
    max_retries: int = 3
    self_correction: bool = True
    consistency_check: bool = True      # Ablation: 모순 감지 on/off
    structured_query: bool = True       # Ablation: suggest-then-verify on/off
    suggest_count: int = 5
    ollama_url: str = OLLAMA_URL


@dataclass
class QueryLog:
    prompt: str
    response: str
    model: str


class OllamaOracle:
    """Ollama 기반 SLM oracle — FCA 속성탐색용."""

    def __init__(self, domain: dict, config: OracleConfig | None = None):
        self.domain = domain
        self.config = config or OracleConfig()
        self.query_log: list[QueryLog] = []
        self.total_queries: int = 0
        self.num_corrections: int = 0
        self.num_contradictions: int = 0
        self._confirmed: list[Implication] = []

    # ── Ollama API ───────────────────────────────────────────────────────

    def _call_ollama(self, prompt: str) -> str:
        """Ollama API 호출."""
        resp = requests.post(
            f"{self.config.ollama_url}/api/generate",
            json={
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self.config.temperature},
            },
            timeout=120,
        )
        resp.raise_for_status()
        text = resp.json()["response"].strip()
        self.total_queries += 1
        self.query_log.append(
            QueryLog(prompt=prompt, response=text, model=self.config.model)
        )
        return text

    # ── Parsing helpers ──────────────────────────────────────────────────

    def _parse_yes_no(self, response: str) -> bool | None:
        """응답에서 yes/no 추출."""
        low = response.lower().strip()
        if low.startswith("yes"):
            return True
        if low.startswith("no"):
            return False
        try:
            data = json.loads(response)
            ans = data.get("answer", "")
            return str(ans).lower().startswith("yes")
        except (json.JSONDecodeError, TypeError, AttributeError):
            return None

    @staticmethod
    def _clean_name(raw: str) -> str:
        """SLM 응답에서 객체 이름 정리."""
        cleaned = raw.strip().strip('"').strip("'").strip(".").strip()
        for prefix in ("a ", "an ", "the "):
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):]
        # 번호 prefix 제거 (e.g., "1. apple")
        parts = cleaned.split(".", 1)
        if len(parts) == 2 and parts[0].strip().isdigit():
            cleaned = parts[1]
        parts = cleaned.split(")", 1)
        if len(parts) == 2 and parts[0].strip().isdigit():
            cleaned = parts[1]
        return cleaned.strip().lower()

    # ── Atomic queries ───────────────────────────────────────────────────

    def ask_attribute(self, obj: str, attr: str) -> bool:
        """객체가 속성을 가지는지 질문. Classification 프롬프트 사용."""
        prompt = (
            f'We are classifying {self.domain["name"]} '
            f"by their common properties.\n"
            f"Answer YES if the property commonly applies "
            f"to the item, NO if not.\n\n"
            f"Item: {obj}\n"
            f"Property: {attr}\n"
            f"Answer:"
        )
        resp = self._call_ollama(prompt)
        result = self._parse_yes_no(resp)
        if result is None:
            logger.warning("Unparseable: %s / %s → %r", obj, attr, resp)
            return False
        return result

    def get_object_attributes(self, obj: str) -> set[str]:
        """객체의 모든 속성을 질문으로 확인."""
        return {
            attr for attr in self.domain["attributes"]
            if self.ask_attribute(obj, attr)
        }

    # ── Counterexample strategies ────────────────────────────────────────

    def suggest_objects(
        self, premise: frozenset[str], n: int | None = None,
    ) -> list[str]:
        """SLM에게 premise를 가진 well-known 객체 n개 제안 요청."""
        count = n or self.config.suggest_count
        domain_name = self.domain["name"]
        if not premise:
            prompt = (
                f"Name {count} well-known, common {domain_name} "
                f"that are very different from each other.\n"
                f"Reply with names only, comma-separated."
            )
        else:
            premise_str = ", ".join(sorted(premise))
            prompt = (
                f"Name {count} well-known {domain_name} "
                f"that have [{premise_str}].\n"
                f"Reply with names only, comma-separated."
            )
        resp = self._call_ollama(prompt)
        names: list[str] = []
        for part in resp.replace("\n", ",").split(","):
            name = self._clean_name(part)
            if name and len(name) < 50 and name not in names:
                names.append(name)
        return names[:count]

    def ask_counterexample(
        self,
        premise: frozenset[str],
        conclusion: frozenset[str],
    ) -> str | None:
        """직접 반례 이름 요청. 이름 반환 또는 None."""
        missing = sorted(conclusion - premise)
        missing_str = ", ".join(missing)

        if not premise:
            prompt = (
                f'About {self.domain["description"]}.\n'
                f"Consider typical, commonly known forms.\n"
                f"We claim ALL {self.domain['name']} have: [{missing_str}].\n"
                f"Name one that LACKS at least one of these properties.\n"
                f"Reply with just the name, or NONE if the claim is correct."
            )
        else:
            premise_str = ", ".join(sorted(premise))
            prompt = (
                f'About {self.domain["description"]}.\n'
                f"Consider typical, commonly known forms.\n"
                f"We claim: if something has [{premise_str}], "
                f"then it also has [{missing_str}].\n"
                f"Name a counterexample that has [{premise_str}] "
                f"but LACKS at least one of [{missing_str}].\n"
                f"Reply with just the name, or NONE if the rule is correct."
            )
        resp = self._call_ollama(prompt)
        cleaned = self._clean_name(resp)
        if not cleaned or "none" in cleaned[:15]:
            return None
        return cleaned

    # ── Validation ───────────────────────────────────────────────────────

    def _validate_counterexample(
        self,
        name: str,
        premise: frozenset[str],
        conclusion: frozenset[str],
    ) -> tuple[bool, set[str]]:
        """반례 후보를 검증. Returns (is_valid, attrs)."""
        attrs = self.get_object_attributes(name)

        if self.config.consistency_check and self._confirmed:
            violations = check_consistency(name, attrs, self._confirmed)
            if violations:
                self.num_contradictions += len(violations)
                if self.config.self_correction:
                    corrected = self._self_correct(name, attrs, violations)
                    if corrected != attrs:
                        self.num_corrections += 1
                        attrs = corrected

        is_valid = premise <= attrs and not conclusion <= attrs
        return is_valid, attrs

    # ── Main oracle protocol ─────────────────────────────────────────────

    def confirm_implication(
        self,
        premise: frozenset[str],
        conclusion: frozenset[str],
        context: FormalContext,
    ) -> tuple[bool, str | None, set[str] | None]:
        """함의 확인 (FCA oracle protocol).

        전략 1 (primary): 다양한 객체 제안 → 속성별 yes/no 검증
        전략 2 (fallback): 직접 반례 이름 요청 → 속성 검증

        structured_query=False → 전략 2만 사용 (ablation).

        Returns: (accepted, counterexample_name, counterexample_attrs)
        """
        # 전략 1: 객체 제안 후 속성별 yes/no 검증 (structured_query=True일 때만)
        if self.config.structured_query:
            candidates = self.suggest_objects(premise)
            for cand in candidates:
                if cand in context.objects:
                    continue
                is_valid, attrs = self._validate_counterexample(
                    cand, premise, conclusion,
                )
                if is_valid:
                    return (False, cand, attrs)

        # 전략 2: 직접 반례 요청
        name = self.ask_counterexample(premise, conclusion)
        if name is not None and name not in context.objects:
            is_valid, attrs = self._validate_counterexample(
                name, premise, conclusion,
            )
            if is_valid:
                return (False, name, attrs)

        # 반례 없음 → 수락
        self._confirmed.append(Implication(premise, conclusion))
        return (True, None, None)

    # ── Self-correction ──────────────────────────────────────────────────

    def _self_correct(
        self,
        obj: str,
        attrs: set[str],
        violations: list,
    ) -> set[str]:
        """모순 감지 시 재질의로 교정 시도."""
        corrected = set(attrs)
        for v in violations:
            premise_str = ", ".join(sorted(v.violated_implication.premise))
            for m in v.missing_attrs:
                prompt = (
                    f'You said "{obj}" has: [{premise_str}].\n'
                    f'We confirmed that things with [{premise_str}] '
                    f'also have "{m}".\n'
                    f'Does "{obj}" actually have "{m}"?\n'
                    f"Answer YES or NO only."
                )
                resp = self._call_ollama(prompt)
                if self._parse_yes_no(resp):
                    corrected.add(m)
                    logger.info("Self-corrected: %s +%s", obj, m)
        return corrected
