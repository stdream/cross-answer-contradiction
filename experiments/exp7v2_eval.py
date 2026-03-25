"""
Exp 7v2 Evaluation: Base vs FCA-DPO vs Random-DPO
==================================================
3 models × 2 domains = 6 runs.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from domain import DOMAINS
from oracle import OllamaOracle, OracleConfig
from fca_engine import full_exploration
from evaluate import knowledge_accuracy_fca, cross_answer_contradiction_rate
from baselines.common import load_gold, save_result

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-5s: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent.parent / "results" / "exp7v2" / "eval"
GOLD_DIR = Path(__file__).resolve().parent.parent / "gold_standards"


def run_eval(model: str, domain_name: str, gold_file: str, label: str) -> dict:
    """Single FCA exploration + evaluation."""
    domain = DOMAINS[domain_name]
    gold_path = GOLD_DIR / gold_file
    gold = load_gold(str(gold_path))

    config = OracleConfig(
        model=model, temperature=0.1,
        self_correction=True, consistency_check=True, structured_query=True,
    )
    oracle = OllamaOracle(domain=domain, config=config)

    logger.info("Running: %s (%s × %s)", label, model, domain_name)
    t0 = time.time()

    result = full_exploration(
        attributes=domain["attributes"],
        oracle=oracle,
        initial_objects=domain.get("initial_examples") or None,
    )
    elapsed = time.time() - t0

    impls = [
        {"premise": sorted(i.premise), "conclusion": sorted(i.conclusion)}
        for i in result.implications
    ]
    acc = knowledge_accuracy_fca(impls, gold["canonical_basis"])
    ccr = cross_answer_contradiction_rate(impls, gold["objects"])

    summary = {
        "label": label,
        "model": model,
        "domain": domain_name,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "elapsed_seconds": round(elapsed, 2),
        "num_queries": oracle.total_queries,
        "num_implications": result.num_implications,
        "num_counterexamples": result.num_counterexamples,
        "num_contradictions": oracle.num_contradictions,
        "num_corrections": oracle.num_corrections,
        "implications": impls,
        "metrics": {**acc, "ccr": ccr["ccr"]},
    }

    path = EVAL_DIR / f"{label}.json"
    save_result(summary, str(path))
    return summary


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    models = [
        ("qwen2.5:1.5b", "base"),
        ("qwen2.5-1.5b-dpo-random:latest", "random_dpo"),
        ("qwen2.5-1.5b-dpo-fca:latest", "fca_dpo"),
    ]
    domains = [
        ("countries", "countries.json"),
        ("animals", "animals.json"),
    ]

    results = []
    for model, cond in models:
        for domain_name, gold_file in domains:
            label = f"{cond}_{domain_name}"
            try:
                r = run_eval(model, domain_name, gold_file, label)
                results.append(r)
                m = r["metrics"]
                print(
                    f"  {label:25s} P={m['precision']:.4f} R={m['recall']:.4f} "
                    f"F1={m['f1']:.4f} CCR={m['ccr']:.4f} "
                    f"Contradictions={r['num_contradictions']} "
                    f"Corrections={r['num_corrections']} Q={r['num_queries']}"
                )
            except Exception as e:
                print(f"  {label:25s} ERROR: {e}")
                results.append({
                    "label": label, "model": model, "domain": domain_name,
                    "metrics": {"precision": 0, "recall": 0, "f1": 0, "ccr": 0},
                    "num_contradictions": 0, "num_corrections": 0,
                    "num_queries": 0, "error": str(e),
                })

    # Generate Table 7
    print("\n" + "=" * 80)
    print("TABLE 7: DPO Evaluation")
    print("=" * 80)

    header = (
        "| Condition | Countries P | R | F1 | CCR | Contrad. "
        "| Animals P | R | F1 | CCR | Contrad. |"
    )
    sep = (
        "|-----------|------------|------|------|------|----------"
        "|----------|------|------|------|----------|"
    )
    print(header)
    print(sep)

    for _, cond in models:
        rc = next((r for r in results if r["label"] == f"{cond}_countries"), None)
        ra = next((r for r in results if r["label"] == f"{cond}_animals"), None)

        def fmt(r, key):
            if r is None:
                return "—"
            if key in r:
                return str(r[key])
            m = r.get("metrics", {})
            return f"{m.get(key, 0):.4f}"

        def fmti(r, key):
            if r is None:
                return "—"
            return str(r.get(key, 0))

        row = (
            f"| {cond:9s} "
            f"| {fmt(rc,'precision')} | {fmt(rc,'recall')} | {fmt(rc,'f1')} "
            f"| {fmt(rc,'ccr')} | {fmti(rc,'num_contradictions')} "
            f"| {fmt(ra,'precision')} | {fmt(ra,'recall')} | {fmt(ra,'f1')} "
            f"| {fmt(ra,'ccr')} | {fmti(ra,'num_contradictions')} |"
        )
        print(row)

    # Save table
    lines = [
        "# Table 7: DPO Evaluation (Base vs FCA-DPO vs Random-DPO)",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        header, sep,
    ]
    for _, cond in models:
        rc = next((r for r in results if r["label"] == f"{cond}_countries"), None)
        ra = next((r for r in results if r["label"] == f"{cond}_animals"), None)

        def fmt(r, key):
            if r is None: return "—"
            m = r.get("metrics", {})
            return f"{m.get(key, 0):.4f}"
        def fmti(r, key):
            if r is None: return "—"
            return str(r.get(key, 0))

        lines.append(
            f"| {cond:9s} "
            f"| {fmt(rc,'precision')} | {fmt(rc,'recall')} | {fmt(rc,'f1')} "
            f"| {fmt(rc,'ccr')} | {fmti(rc,'num_contradictions')} "
            f"| {fmt(ra,'precision')} | {fmt(ra,'recall')} | {fmt(ra,'f1')} "
            f"| {fmt(ra,'ccr')} | {fmti(ra,'num_contradictions')} |"
        )

    with open(EVAL_DIR.parent / "table7.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nSaved: {EVAL_DIR.parent / 'table7.md'}")


if __name__ == "__main__":
    main()
