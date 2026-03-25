"""
Evaluation — Metrics + Comparison Tables
========================================
- knowledge_accuracy: P/R/F1 (closure-based for FCA, direct for baselines)
- cross_answer_contradiction_rate: 논리적 모순 비율
- format_comparison_table: markdown 테이블 출력
"""
from __future__ import annotations

import json
from pathlib import Path
from fca_engine import Implication, FormalContext, closure_under_implications


# ── Knowledge Accuracy ───────────────────────────────────────────────────────

def knowledge_accuracy_fca(
    discovered_impls: list[dict],
    gold_impls: list[dict],
) -> dict:
    """FCA 결과 vs gold basis — closure-based P/R/F1.

    Precision: discovered 중 gold에서 derivable한 비율.
    Recall: gold 중 discovered에서 derivable한 비율.
    """
    discovered = [
        Implication(frozenset(d["premise"]), frozenset(d["conclusion"]))
        for d in discovered_impls
    ]
    gold = [
        Implication(frozenset(g["premise"]), frozenset(g["conclusion"]))
        for g in gold_impls
    ]

    # Precision: each discovered should follow from gold
    if discovered:
        correct = sum(
            1 for d in discovered
            if d.conclusion <= closure_under_implications(d.premise, gold)
        )
        precision = correct / len(discovered)
    else:
        precision = 0.0

    # Recall: each gold should follow from discovered
    if gold:
        recovered = sum(
            1 for g in gold
            if g.conclusion <= closure_under_implications(g.premise, discovered)
        )
        recall = recovered / len(gold)
    else:
        recall = 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "num_discovered": len(discovered),
        "num_gold": len(gold),
    }


# ── Cross-Answer Contradiction Rate ──────────────────────────────────────────

def cross_answer_contradiction_rate(
    accepted_impls: list[dict],
    gold_objects: dict[str, list[str]],
) -> dict:
    """Accepted implications vs gold objects — 모순 비율.

    CCR = gold objects 중 적어도 하나의 accepted implication을 위반하는 비율.
    """
    impls = [
        Implication(frozenset(i["premise"]), frozenset(i["conclusion"]))
        for i in accepted_impls
    ]

    violations = 0
    violation_details: list[dict] = []
    total = len(gold_objects)

    for name, attrs_list in gold_objects.items():
        attrs = frozenset(attrs_list)
        for impl in impls:
            if impl.premise <= attrs and not impl.conclusion <= attrs:
                violations += 1
                violation_details.append({
                    "object": name,
                    "implication": repr(impl),
                    "missing": sorted(impl.conclusion - attrs),
                })
                break

    ccr = violations / total if total > 0 else 0.0
    return {
        "ccr": round(ccr, 4),
        "violations": violations,
        "total_objects": total,
        "details": violation_details[:10],  # 상위 10개만
    }


def fca_exploration_ccr(exploration_log: list[dict]) -> dict:
    """FCA 탐색 로그에서 self-detected 모순 비율."""
    counterexamples = [e for e in exploration_log if e["type"] == "counterexample"]
    implications = [e for e in exploration_log if e["type"] == "implication"]
    intents = [e for e in exploration_log if e["type"] == "intent"]
    total_questions = len(counterexamples) + len(implications)

    return {
        "num_counterexamples": len(counterexamples),
        "num_implications": len(implications),
        "num_intents": len(intents),
        "total_questions": total_questions,
    }


# ── Comparison Table ─────────────────────────────────────────────────────────

def format_comparison_table(results: list[dict]) -> str:
    """결과 목록 → markdown 비교 테이블."""
    header = "| Method | P | R | F1 | CCR | Queries | Time(s) |"
    sep = "|--------|------|------|------|------|---------|---------|"
    rows = [header, sep]

    for r in results:
        m = r.get("metrics", {})
        method = r.get("method", "?")
        p = m.get("precision", "—")
        rec = m.get("recall", "—")
        f1 = m.get("f1", "—")
        ccr = m.get("ccr", "—")
        queries = r.get("num_queries", r.get("num_oracle_queries", "—"))
        elapsed = r.get("elapsed_seconds", "—")

        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else str(v)

        rows.append(
            f"| {method} | {fmt(p)} | {fmt(rec)} | {fmt(f1)} "
            f"| {fmt(ccr)} | {queries} | {elapsed} |"
        )

    return "\n".join(rows)


def format_model_table(results: list[dict]) -> str:
    """모델 비교 테이블 (Exp 2)."""
    header = "| Model | Method | P | R | F1 | CCR | Queries |"
    sep = "|-------|--------|------|------|------|------|---------|"
    rows = [header, sep]

    for r in results:
        m = r.get("metrics", {})
        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else str(v)
        rows.append(
            f"| {r.get('model','?')} | {r.get('method','?')} "
            f"| {fmt(m.get('precision','—'))} | {fmt(m.get('recall','—'))} "
            f"| {fmt(m.get('f1','—'))} | {fmt(m.get('ccr','—'))} "
            f"| {r.get('num_queries', r.get('num_oracle_queries','—'))} |"
        )
    return "\n".join(rows)


# ── Unified evaluation ───────────────────────────────────────────────────────

def evaluate_fca_result(result_path: str, gold_path: str) -> dict:
    """FCA 결과 JSON을 gold와 비교하여 metrics 계산."""
    with open(result_path) as f:
        result = json.load(f)
    gold = json.load(open(gold_path))

    acc = knowledge_accuracy_fca(
        result["implications"],
        gold["canonical_basis"],
    )
    ccr = cross_answer_contradiction_rate(
        result["implications"],
        gold["objects"],
    )

    result["metrics"] = {**acc, "ccr": ccr["ccr"]}
    result["ccr_details"] = ccr
    return result


def evaluate_baseline_result(result_path: str, gold_path: str) -> dict:
    """Baseline 결과 JSON에 CCR 추가."""
    with open(result_path) as f:
        result = json.load(f)
    gold = json.load(open(gold_path))

    # Accepted implications = predicted True
    accepted = [
        p for p in result["predictions"] if p["predicted"]
    ]
    ccr = cross_answer_contradiction_rate(accepted, gold["objects"])
    result["metrics"]["ccr"] = ccr["ccr"]
    result["ccr_details"] = ccr
    return result
