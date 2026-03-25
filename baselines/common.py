"""
Baseline 공통 유틸리티
====================
Ollama API, test set 생성, 공통 결과 포맷.
"""
from __future__ import annotations

import json
import random
import logging
import requests
import time
from pathlib import Path
from dataclasses import dataclass, field

from fca_engine import Implication, FormalContext, closure_under_implications

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"


# ── Ollama API ───────────────────────────────────────────────────────────────

def call_ollama(
    prompt: str,
    model: str = "qwen2.5:7b",
    temperature: float = 0.1,
    url: str = OLLAMA_URL,
) -> str:
    resp = requests.post(
        f"{url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def parse_yes_no(response: str) -> bool | None:
    low = response.lower().strip()
    if low.startswith("yes"):
        return True
    if low.startswith("no"):
        return False
    return None


# ── Gold standard loading ────────────────────────────────────────────────────

def load_gold(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def gold_to_context(gold: dict) -> FormalContext:
    """Gold standard JSON → FormalContext."""
    return FormalContext(
        gold["attributes"],
        {name: set(attrs) for name, attrs in gold["objects"].items()},
    )


def gold_to_implications(gold: dict) -> list[Implication]:
    return [
        Implication(frozenset(b["premise"]), frozenset(b["conclusion"]))
        for b in gold["canonical_basis"]
    ]


# ── Test set generation ──────────────────────────────────────────────────────

def generate_test_set(gold: dict, seed: int = 42) -> list[dict]:
    """Gold canonical basis (valid=True) + 동일 수 invalid implications (valid=False)."""
    rng = random.Random(seed)
    attributes = gold["attributes"]
    context = gold_to_context(gold)

    # Valid: gold canonical basis
    items: list[dict] = []
    for b in gold["canonical_basis"]:
        items.append({
            "premise": b["premise"],
            "conclusion": b["conclusion"],
            "valid": True,
        })
    n_valid = len(items)

    # Invalid: random implications that don't hold in context
    invalids: list[dict] = []
    seen: set[tuple] = set()
    attempts = 0
    while len(invalids) < n_valid and attempts < n_valid * 500:
        attempts += 1
        k = rng.randint(1, min(4, len(attributes)))
        premise = sorted(rng.sample(attributes, k))
        premise_fs = frozenset(premise)
        closure = context.double_prime(premise_fs)

        non_implied = [a for a in attributes if a not in closure]
        if not non_implied:
            continue
        extra = rng.choice(non_implied)
        conclusion = sorted(set(premise) | {extra})
        conclusion_fs = frozenset(conclusion)

        if closure >= conclusion_fs:
            continue

        key = (tuple(premise), tuple(conclusion))
        if key in seen:
            continue
        seen.add(key)

        invalids.append({
            "premise": premise,
            "conclusion": conclusion,
            "valid": False,
        })

    items.extend(invalids[:n_valid])
    rng.shuffle(items)
    return items


# ── Implication question prompt ──────────────────────────────────────────────

def format_implication_question(
    premise: list[str],
    conclusion: list[str],
    domain_desc: str,
    cot: bool = False,
) -> str:
    premise_str = ", ".join(premise) or "no specific properties"
    added = sorted(set(conclusion) - set(premise))
    added_str = ", ".join(added)
    prompt = (
        f"About {domain_desc}:\n"
        f"We claim: if something has [{premise_str}], "
        f"then it also has [{added_str}].\n"
        f"Does this rule hold for ALL items in this domain?\n"
    )
    if cot:
        prompt += "Think step by step, then answer YES or NO.\n"
    else:
        prompt += "Answer YES or NO.\n"
    return prompt


# ── Result saving ────────────────────────────────────────────────────────────

def save_result(result: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Saved: %s", path)


def compute_baseline_metrics(predictions: list[dict]) -> dict:
    """Baseline prediction list → P/R/F1/accuracy."""
    tp = sum(1 for p in predictions if p["predicted"] and p["valid"])
    fp = sum(1 for p in predictions if p["predicted"] and not p["valid"])
    fn = sum(1 for p in predictions if not p["predicted"] and p["valid"])
    tn = sum(1 for p in predictions if not p["predicted"] and not p["valid"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(predictions) if predictions else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
