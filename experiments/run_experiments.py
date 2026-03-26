"""
Experiment Runner -- Automated execution of Exp 1-5
====================================================
Usage:
    python experiments/run_experiments.py --exp 1
    python experiments/run_experiments.py --exp all
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from domain import DOMAINS, SCALING_DOMAINS
from oracle import OllamaOracle, OracleConfig
from fca_engine import full_exploration
from evaluate import (
    knowledge_accuracy_fca,
    cross_answer_contradiction_rate,
    format_comparison_table,
    format_model_table,
)
from baselines import baseline_vanilla, baseline_cot, baseline_selfconsistency
from baselines.common import load_gold, save_result

logger = logging.getLogger(__name__)

GOLD_DIR = Path(__file__).resolve().parent.parent / "gold_standards"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ── FCA run helper ───────────────────────────────────────────────────────────

def run_fca(
    domain_name: str,
    model: str,
    gold_path: Path,
    output_dir: Path,
    self_correction: bool = True,
    consistency_check: bool = True,
    structured_query: bool = True,
    method_label: str = "fca",
) -> dict:
    """Run FCA exploration + compare against gold."""
    all_domains = {**DOMAINS, **SCALING_DOMAINS}
    domain = all_domains[domain_name]
    gold = load_gold(str(gold_path))

    config = OracleConfig(
        model=model,
        self_correction=self_correction,
        consistency_check=consistency_check,
        structured_query=structured_query,
    )
    oracle = OllamaOracle(domain=domain, config=config)

    logger.info("FCA [%s]: %s × %s", method_label, domain_name, model)
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
        "method": method_label,
        "domain": domain_name,
        "model": model,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "elapsed_seconds": round(elapsed, 2),
        "num_queries": oracle.total_queries,
        "num_oracle_queries": oracle.total_queries,
        "num_implications": result.num_implications,
        "num_counterexamples": result.num_counterexamples,
        "num_contradictions": oracle.num_contradictions,
        "num_corrections": oracle.num_corrections,
        "implications": impls,
        "context_objects": {
            n: sorted(a) for n, a in result.context.objects.items()
        },
        "metrics": {**acc, "ccr": ccr["ccr"]},
        "ccr_details": ccr,
        "config": {
            "self_correction": self_correction,
            "consistency_check": consistency_check,
            "structured_query": structured_query,
        },
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{method_label}_{domain_name}_{model.replace(':','_')}_{ts}.json"
    save_result(summary, str(path))
    return summary


def run_baseline(
    method: str,
    domain_name: str,
    model: str,
    gold_path: Path,
    output_dir: Path,
    **kwargs,
) -> dict:
    """Run baseline + add CCR."""
    all_domains = {**DOMAINS, **SCALING_DOMAINS}
    domain = all_domains[domain_name]
    gold = load_gold(str(gold_path))

    runner = {
        "vanilla": baseline_vanilla,
        "cot": baseline_cot,
        "selfconsistency_k5": baseline_selfconsistency,
        "selfconsistency_k10": baseline_selfconsistency,
    }[method]

    run_kwargs = dict(
        domain_name=domain_name,
        domain_desc=domain["description"],
        model=model,
        gold_path=str(gold_path),
        output_dir=str(output_dir),
    )
    if method == "selfconsistency_k5":
        run_kwargs["k"] = 5
    elif method == "selfconsistency_k10":
        run_kwargs["k"] = 10

    logger.info("Baseline [%s]: %s × %s", method, domain_name, model)
    result = runner.run(**run_kwargs)

    # Add CCR
    accepted = [p for p in result["predictions"] if p["predicted"]]
    ccr = cross_answer_contradiction_rate(accepted, gold["objects"])
    result["metrics"]["ccr"] = ccr["ccr"]
    result["ccr_details"] = ccr

    return result


# ── Experiments ──────────────────────────────────────────────────────────────

def exp1_main(model: str = "qwen2.5:7b"):
    """Exp 1: FCA vs baselines on countries domain."""
    print("\n" + "=" * 60)
    print("EXP 1: Main Result — FCA vs Baselines (countries)")
    print("=" * 60)

    domain = "countries"
    gold_path = GOLD_DIR / "countries.json"
    out = RESULTS_DIR / "exp1"

    results = []

    # Baselines
    for method in ["vanilla", "cot", "selfconsistency_k5", "selfconsistency_k10"]:
        r = run_baseline(method, domain, model, gold_path, out)
        results.append(r)
        _print_short(r)

    # FCA
    r = run_fca(domain, model, gold_path, out)
    results.append(r)
    _print_short(r)

    # Summary table
    table = format_comparison_table(results)
    print("\n" + table)
    _save_summary(out, "exp1_summary.md", table, results)
    return results


def exp2_models(models: list[str] | None = None):
    """Exp 2: FCA + vanilla across models on countries."""
    print("\n" + "=" * 60)
    print("EXP 2: Across Models (countries)")
    print("=" * 60)

    if models is None:
        models = ["qwen2.5:7b"]
    domain = "countries"
    gold_path = GOLD_DIR / "countries.json"
    out = RESULTS_DIR / "exp2"

    results = []
    for m in models:
        r_vanilla = run_baseline("vanilla", domain, m, gold_path, out)
        results.append(r_vanilla)
        _print_short(r_vanilla)

        r_fca = run_fca(domain, m, gold_path, out)
        results.append(r_fca)
        _print_short(r_fca)

    table = format_model_table(results)
    print("\n" + table)
    _save_summary(out, "exp2_summary.md", table, results)
    return results


def exp3_domains(model: str = "qwen2.5:7b"):
    """Exp 3: FCA + vanilla across domains."""
    print("\n" + "=" * 60)
    print("EXP 3: Across Domains")
    print("=" * 60)

    out = RESULTS_DIR / "exp3"
    results = []

    for domain_name, gold_file in [
        ("countries", "countries.json"),
        ("animals", "animals.json"),
    ]:
        gold_path = GOLD_DIR / gold_file
        if not gold_path.exists():
            logger.warning("Skipping %s — no gold standard", domain_name)
            continue

        r_vanilla = run_baseline("vanilla", domain_name, model, gold_path, out)
        results.append(r_vanilla)
        _print_short(r_vanilla)

        r_fca = run_fca(domain_name, model, gold_path, out)
        results.append(r_fca)
        _print_short(r_fca)

    table = format_comparison_table(results)
    print("\n" + table)
    _save_summary(out, "exp3_summary.md", table, results)
    return results


def exp4_ablation(model: str = "qwen2.5:7b"):
    """Exp 4: Ablation on countries."""
    print("\n" + "=" * 60)
    print("EXP 4: Ablation Study (countries)")
    print("=" * 60)

    domain = "countries"
    gold_path = GOLD_DIR / "countries.json"
    out = RESULTS_DIR / "exp4"

    configs = [
        ("fca_full", True, True, True),
        ("fca_no_selfcorr", False, True, True),
        ("fca_no_consistency", False, False, True),
        ("fca_no_structure", False, False, False),
    ]

    results = []
    for label, sc, cc, sq in configs:
        r = run_fca(
            domain, model, gold_path, out,
            self_correction=sc,
            consistency_check=cc,
            structured_query=sq,
            method_label=label,
        )
        results.append(r)
        _print_short(r)

    table = format_comparison_table(results)
    print("\n" + table)
    _save_summary(out, "exp4_summary.md", table, results)
    return results


def exp5_scaling(model: str = "qwen2.5:7b"):
    """Exp 5: Scaling with attribute count."""
    print("\n" + "=" * 60)
    print("EXP 5: Scaling Analysis (countries variants)")
    print("=" * 60)

    out = RESULTS_DIR / "exp5"
    gold_path = GOLD_DIR / "countries.json"
    gold = load_gold(str(gold_path))

    results = []
    for domain_name in ["countries_10", "countries_15"]:
        all_domains = {**DOMAINS, **SCALING_DOMAINS}
        domain = all_domains[domain_name]
        n_attrs = len(domain["attributes"])

        # Build a sub-gold for this attribute subset
        sub_attrs = set(domain["attributes"])
        sub_objects = {}
        for name, attrs_list in gold["objects"].items():
            sub_objects[name] = sorted(set(attrs_list) & sub_attrs)

        sub_gold_path = out / f"_gold_{domain_name}.json"
        sub_gold_path.parent.mkdir(parents=True, exist_ok=True)

        # Compute sub-gold canonical basis
        from gold_standards.build import GoldOracle
        from fca_engine import full_exploration as fe
        sub_gold_objects = {n: set(a) for n, a in sub_objects.items()}
        sub_oracle = GoldOracle(sub_gold_objects)
        sub_result = fe(domain["attributes"], sub_oracle)
        sub_basis = [
            {"premise": sorted(i.premise), "conclusion": sorted(i.conclusion)}
            for i in sub_result.implications
        ]

        sub_gold = {
            "domain": domain_name,
            "attributes": domain["attributes"],
            "objects": sub_objects,
            "canonical_basis": sub_basis,
        }
        with open(sub_gold_path, "w") as f:
            json.dump(sub_gold, f, indent=2)

        r = run_fca(domain_name, model, sub_gold_path, out)
        r["num_attributes"] = n_attrs
        results.append(r)
        _print_short(r)

    # Also run the full 15-attr countries
    r = run_fca("countries", model, gold_path, out, method_label="fca_15full")
    r["num_attributes"] = 15
    results.append(r)
    _print_short(r)

    header = "| Attrs | Implications | Queries | Time(s) | P | R | F1 | CCR |"
    sep = "|-------|-------------|---------|---------|------|------|------|------|"
    rows = [header, sep]
    for r in results:
        m = r["metrics"]
        rows.append(
            f"| {r.get('num_attributes','?')} "
            f"| {r['num_implications']} | {r['num_queries']} "
            f"| {r['elapsed_seconds']} "
            f"| {m['precision']:.4f} | {m['recall']:.4f} "
            f"| {m['f1']:.4f} | {m['ccr']:.4f} |"
        )
    table = "\n".join(rows)
    print("\n" + table)
    _save_summary(out, "exp5_summary.md", table, results)
    return results


# ── Helpers ──────────────────────────────────────────────────────────────────

def _print_short(r: dict) -> None:
    m = r.get("metrics", {})
    print(
        f"  {r['method']:25s} P={m.get('precision','—'):.4f} "
        f"R={m.get('recall','—'):.4f} F1={m.get('f1','—'):.4f} "
        f"CCR={m.get('ccr','—'):.4f} Q={r.get('num_queries', r.get('num_oracle_queries','?'))} "
        f"T={r['elapsed_seconds']}s"
    )


def _save_summary(out_dir: Path, filename: str, table: str, results: list) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / filename, "w") as f:
        f.write(f"# {filename.replace('.md','').replace('_',' ').title()}\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(table + "\n")
    # Also save raw results
    with open(out_dir / filename.replace(".md", ".json"), "w") as f:
        json.dump(
            [_strip_predictions(r) for r in results],
            f, indent=2, ensure_ascii=False,
        )


def _strip_predictions(r: dict) -> dict:
    """Remove large predictions array from result (for summary)."""
    out = {k: v for k, v in r.items() if k not in ("predictions", "context_objects")}
    if "predictions" in r:
        out["num_predictions"] = len(r["predictions"])
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Run experiments")
    p.add_argument("--exp", default="1", help="Experiment number (1-5 or 'all')")
    p.add_argument("--model", default="qwen2.5:7b")
    args = p.parse_args()

    exps = {
        "1": lambda: exp1_main(args.model),
        "2": lambda: exp2_models([args.model]),
        "3": lambda: exp3_domains(args.model),
        "4": lambda: exp4_ablation(args.model),
        "5": lambda: exp5_scaling(args.model),
    }

    if args.exp == "all":
        all_results = {}
        for k, fn in exps.items():
            all_results[f"exp{k}"] = fn()
    elif args.exp in exps:
        exps[args.exp]()
    else:
        print(f"Unknown experiment: {args.exp}. Choose 1-5 or 'all'.")


if __name__ == "__main__":
    main()
