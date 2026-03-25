"""
FCA Attribute Exploration Engine — Algorithm 19 (Ganter & Obiedkov 2016 Ch.4)
=============================================================================
수학적 알고리즘. 구현+테스트 통과 후 절대 수정하지 않는다.
모든 실험 비교의 기반.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import logging

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Implication:
    """FCA 함의 A → B (premise ⊆ conclusion)."""
    premise: frozenset[str]
    conclusion: frozenset[str]

    def __post_init__(self):
        if not self.premise <= self.conclusion:
            raise ValueError(
                f"Premise must be ⊆ conclusion: {self.premise} ⊄ {self.conclusion}"
            )

    @property
    def added(self) -> frozenset[str]:
        """conclusion \\ premise."""
        return self.conclusion - self.premise

    def holds_for(self, attrs: set[str] | frozenset[str]) -> bool:
        """이 함의가 주어진 속성 집합에 대해 성립하는지 확인."""
        return not (self.premise <= attrs) or (self.conclusion <= attrs)

    def __repr__(self) -> str:
        p = ", ".join(sorted(self.premise)) or "∅"
        a = ", ".join(sorted(self.added))
        return f"{{{p}}} → {{{a}}}"


@dataclass
class ConsistencyViolation:
    """모순 감지: 객체가 확인된 함의를 위반."""
    object_name: str
    object_attrs: frozenset[str]
    violated_implication: Implication
    missing_attrs: frozenset[str]


@dataclass
class ExplorationResult:
    """속성탐색 결과."""
    implications: list[Implication]
    context: FormalContext
    exploration_log: list[dict]
    num_questions: int
    num_counterexamples: int

    @property
    def num_implications(self) -> int:
        return len(self.implications)


# ── Formal Context ───────────────────────────────────────────────────────────

class FormalContext:
    """형식 문맥 (G, M, I) — 객체 × 속성 교차표."""

    def __init__(
        self,
        attributes: list[str],
        objects: dict[str, set[str]] | None = None,
    ):
        self.attributes: list[str] = list(attributes)
        self.objects: dict[str, set[str]] = {}
        if objects:
            for name, attrs in objects.items():
                self.objects[name] = set(attrs)

    def add_object(self, name: str, attrs: set[str]) -> None:
        self.objects[name] = set(attrs)

    def extent(self, attrs: set[str] | frozenset[str]) -> frozenset[str]:
        """A' — attrs를 모두 가진 객체 집합."""
        if not attrs:
            return frozenset(self.objects.keys())
        return frozenset(
            obj
            for obj, obj_attrs in self.objects.items()
            if set(attrs) <= obj_attrs
        )

    def intent(self, objs: set[str] | frozenset[str]) -> frozenset[str]:
        """B' — 객체들이 공유하는 속성 집합."""
        if not objs:
            return frozenset(self.attributes)
        result: set[str] | None = None
        for obj in objs:
            if obj in self.objects:
                if result is None:
                    result = set(self.objects[obj])
                else:
                    result &= self.objects[obj]
        return frozenset(result) if result is not None else frozenset(self.attributes)

    def double_prime(self, attrs: set[str] | frozenset[str]) -> frozenset[str]:
        """A'' = (A')' — 문맥 내 속성 폐포."""
        return self.intent(self.extent(attrs))

    def __repr__(self) -> str:
        return f"FormalContext({len(self.objects)} objs × {len(self.attributes)} attrs)"


# ── Core Algorithms ──────────────────────────────────────────────────────────

def closure_under_implications(
    attrs: set[str] | frozenset[str],
    implications: list[Implication],
) -> frozenset[str]:
    """L(A) — 함의 집합 L 하에서 A의 폐포. 고정점까지 반복."""
    closed = set(attrs)
    changed = True
    while changed:
        changed = False
        for impl in implications:
            if impl.premise <= closed and not impl.conclusion <= closed:
                closed |= impl.conclusion
                changed = True
    return frozenset(closed)


def next_closure(
    current: frozenset[str],
    attributes: list[str],
    implications: list[Implication],
) -> frozenset[str] | None:
    """사전식(lecticographic) 순서에서 current 다음의 L-폐포 집합.

    Returns None when current is the last closed set.
    """
    n = len(attributes)
    attr_idx = {a: i for i, a in enumerate(attributes)}

    for i in range(n - 1, -1, -1):
        m_i = attributes[i]
        if m_i in current:
            continue

        # oplus(current, m_i) = (current ∩ {a : index(a) < i}) ∪ {m_i}
        prefix = frozenset(a for a in current if attr_idx[a] < i)
        closed = closure_under_implications(prefix | {m_i}, implications)

        # i 미만에서 새 원소가 추가되지 않았는지 확인
        if all(
            attributes[j] not in closed or attributes[j] in current
            for j in range(i)
        ):
            return closed

    return None


# ── Oracle Protocol ──────────────────────────────────────────────────────────

@runtime_checkable
class Oracle(Protocol):
    """탐색 oracle 인터페이스."""

    def confirm_implication(
        self,
        premise: frozenset[str],
        conclusion: frozenset[str],
        context: FormalContext,
    ) -> tuple[bool, str | None, set[str] | None]:
        """함의 확인. Returns (accepted, counterexample_name, counterexample_attrs)."""
        ...


# ── Attribute Exploration — Algorithm 19 ─────────────────────────────────────

def full_exploration(
    attributes: list[str],
    oracle: Oracle,
    initial_objects: dict[str, set[str]] | None = None,
    max_iterations: int = 10_000,
) -> ExplorationResult:
    """Algorithm 19 (Ganter & Obiedkov 2016 Ch.4) 실행.

    Args:
        attributes: 속성 리스트 M (순서 고정)
        oracle: 함의 확인 oracle
        initial_objects: 초기 객체들 (선택)
        max_iterations: 안전 한계
    """
    context = FormalContext(attributes, initial_objects)
    implications: list[Implication] = []
    log: list[dict] = []
    num_questions = 0
    num_counterexamples = 0

    A = closure_under_implications(frozenset(), implications)
    iteration = 0

    while A is not None and iteration < max_iterations:
        iteration += 1

        # A가 intent인지, 아니면 함의를 제안
        while True:
            A_pp = context.double_prime(A)

            if A == A_pp:
                log.append({"type": "intent", "set": sorted(A)})
                break

            # 함의 후보: A → A''
            num_questions += 1
            accepted, ce_name, ce_attrs = oracle.confirm_implication(
                A, A_pp, context
            )

            if accepted:
                impl = Implication(premise=A, conclusion=A_pp)
                implications.append(impl)
                log.append({
                    "type": "implication",
                    "premise": sorted(A),
                    "conclusion": sorted(A_pp),
                })
                logger.info("Accepted: %s", impl)
                break

            assert ce_name is not None and ce_attrs is not None, \
                "Oracle rejected but provided no counterexample"
            context.add_object(ce_name, ce_attrs)
            num_counterexamples += 1
            log.append({
                "type": "counterexample",
                "object": ce_name,
                "attrs": sorted(ce_attrs),
            })
            logger.info("Counterexample: %s %s", ce_name, sorted(ce_attrs))

        A = next_closure(A, attributes, implications)

    return ExplorationResult(
        implications=implications,
        context=context,
        exploration_log=log,
        num_questions=num_questions,
        num_counterexamples=num_counterexamples,
    )


# ── Consistency Checking (Hallucination Detector) ────────────────────────────

def check_consistency(
    object_name: str,
    object_attrs: set[str] | frozenset[str],
    implications: list[Implication],
) -> list[ConsistencyViolation]:
    """객체 속성이 확인된 함의와 일관되는지 검사.

    위반 = hallucination: SLM이 premise는 있다고 했는데 conclusion은 없다고 한 경우.
    """
    attrs_fs = frozenset(object_attrs)
    return [
        ConsistencyViolation(
            object_name=object_name,
            object_attrs=attrs_fs,
            violated_implication=impl,
            missing_attrs=impl.conclusion - attrs_fs,
        )
        for impl in implications
        if impl.premise <= attrs_fs and not impl.conclusion <= attrs_fs
    ]
