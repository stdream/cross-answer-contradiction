"""
Exp 7v2: Large-scale DPO Pair Collection
=========================================
7 FCA exploration runs × multiple models/domains → 100+ DPO pairs.

Usage: python experiments/exp7v2_dpo_pairs.py
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fca_engine import full_exploration
from oracle import OllamaOracle, OracleConfig
from domain import DOMAINS, SCALING_DOMAINS
from baselines.common import load_gold

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "exp7v2"
RAW_DIR = RESULTS_DIR / "raw"
GOLD_DIR = Path(__file__).resolve().parent.parent / "gold_standards"


def extract_dpo_pairs(oracle: OllamaOracle, source_model: str, source_domain: str, domain_attrs: str) -> list[dict]:
    """Oracle query log에서 self-correction 이벤트를 DPO pair로 변환."""
    pairs = []
    logs = oracle.query_log

    for i, ql in enumerate(logs):
        prompt = ql.prompt
        response = ql.response

        if 'We confirmed that things with' not in prompt or 'actually have' not in prompt:
            continue

        low = response.lower().strip()
        corrected_to_yes = low.startswith("yes")

        obj, attr = "", ""
        try:
            if 'You said "' in prompt:
                obj = prompt.split('You said "')[1].split('"')[0]
            if 'actually have "' in prompt:
                attr = prompt.split('actually have "')[1].split('"')[0]
        except (IndexError, ValueError):
            continue

        if not obj or not attr:
            continue

        # Find original classification prompt
        original_prompt = None
        for j in range(i - 1, max(i - 60, -1), -1):
            prev = logs[j]
            if 'We confirmed' in prev.prompt:
                continue
            if obj.lower() in prev.prompt.lower() and attr.lower() in prev.prompt.lower():
                original_prompt = prev.prompt
                break

        if original_prompt is None:
            original_prompt = (
                f'We are classifying {oracle.domain["name"]} '
                f"by their common properties.\n"
                f"Answer YES if the property commonly applies "
                f"to the item, NO if not.\n\n"
                f"Item: {obj}\n"
                f"Property: {attr}\n"
                f"Answer:"
            )

        pairs.append({
            "prompt": original_prompt,
            "chosen": "YES" if corrected_to_yes else "NO",
            "rejected": "NO" if corrected_to_yes else "YES",
            "metadata": {
                "source_model": source_model,
                "source_domain": source_domain,
                "domain_attrs": domain_attrs,
                "object": obj,
                "attribute": attr,
                "correction_type": "closure_violation",
                "correction_success": corrected_to_yes,
            },
        })

    # Also extract contradiction-detected-but-not-corrected cases
    # These show up as consistency violations in the exploration log
    # We can detect them by looking for objects whose attributes differ
    # from what closure would require, but no self-correction query followed
    # (This is implicit — if contradiction was detected but correction failed,
    #  the pair still has value with chosen = closure-required answer)

    return pairs


def run_single_exploration(
    model: str,
    domain_name: str,
    gold_path: str,
    label: str,
) -> tuple[dict, list[dict]]:
    """Single FCA exploration → (summary, pairs)."""
    all_domains = {**DOMAINS, **SCALING_DOMAINS}
    domain = all_domains[domain_name]
    n_attrs = len(domain["attributes"])

    config = OracleConfig(
        model=model, temperature=0.1,
        self_correction=True, consistency_check=True, structured_query=True,
    )
    oracle = OllamaOracle(domain=domain, config=config)

    print(f"  Running: {label} ({model} × {domain_name}, {n_attrs} attrs)")
    t0 = time.time()

    result = full_exploration(
        attributes=domain["attributes"],
        oracle=oracle,
        initial_objects=domain.get("initial_examples") or None,
    )
    elapsed = time.time() - t0

    pairs = extract_dpo_pairs(oracle, model, domain_name, str(n_attrs))

    summary = {
        "label": label,
        "model": model,
        "domain": domain_name,
        "attrs": n_attrs,
        "implications": result.num_implications,
        "counterexamples": result.num_counterexamples,
        "contradictions": oracle.num_contradictions,
        "corrections": oracle.num_corrections,
        "queries": oracle.total_queries,
        "elapsed": round(elapsed, 1),
        "pairs": len(pairs),
    }

    print(f"    → {result.num_implications} impls, "
          f"{oracle.num_contradictions} contradictions, "
          f"{oracle.num_corrections} corrections, "
          f"{len(pairs)} pairs, {elapsed:.0f}s")

    return summary, pairs


def save_jsonl(pairs: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Define runs (small first, 30-attr last)
    runs = [
        ("qwen2.5:1.5b", "countries", "countries.json", "1.5b_countries", "pairs_1.5b_countries.jsonl"),
        ("qwen2.5:1.5b", "animals", "animals.json", "1.5b_animals", "pairs_1.5b_animals.jsonl"),
        ("qwen2.5:7b", "countries", "countries.json", "7b_countries", "pairs_7b_countries.jsonl"),
        ("qwen2.5:7b", "animals", "animals.json", "7b_animals", "pairs_7b_animals.jsonl"),
        ("llama3.1:8b", "countries", "countries.json", "8b_countries", "pairs_8b_countries.jsonl"),
        ("llama3.1:8b", "animals", "animals.json", "8b_animals", "pairs_8b_animals.jsonl"),
        ("qwen2.5:7b", "countries_30", "countries_30.json", "7b_countries30", "pairs_7b_countries30.jsonl"),
    ]

    all_summaries = []
    all_pairs = []
    cumulative = 0

    for model, domain_name, gold_file, label, raw_file in runs:
        gold_path = GOLD_DIR / gold_file
        if not gold_path.exists():
            print(f"  SKIP {label}: gold standard {gold_path} not found")
            continue

        print(f"\n{'='*60}")
        try:
            summary, pairs = run_single_exploration(model, domain_name, str(gold_path), label)
            all_summaries.append(summary)
            all_pairs.extend(pairs)
            cumulative += len(pairs)

            # Save raw pairs immediately (checkpoint)
            save_jsonl(pairs, RAW_DIR / raw_file)
            print(f"    Saved: {RAW_DIR / raw_file} ({len(pairs)} pairs)")
            print(f"    Cumulative total: {cumulative} pairs")

        except Exception as e:
            print(f"  ERROR {label}: {e}")
            all_summaries.append({
                "label": label, "model": model, "domain": domain_name,
                "error": str(e), "pairs": 0,
            })

    # ── Post-processing ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("POST-PROCESSING")
    print(f"{'='*60}")

    # Deduplicate
    seen = set()
    unique_pairs = []
    for p in all_pairs:
        key = (p["prompt"][:200], p["chosen"], p["rejected"])
        if key not in seen and len(p["prompt"]) > 30:
            seen.add(key)
            unique_pairs.append(p)

    print(f"  Raw pairs: {len(all_pairs)}")
    print(f"  Unique pairs (after dedup + quality filter): {len(unique_pairs)}")

    # Save combined files
    save_jsonl(unique_pairs, RESULTS_DIR / "dpo_train_all.jsonl")

    countries_pairs = [p for p in unique_pairs if "countries" in p["metadata"]["source_domain"]]
    animals_pairs = [p for p in unique_pairs if "animals" in p["metadata"]["source_domain"]]
    save_jsonl(countries_pairs, RESULTS_DIR / "dpo_train_countries.jsonl")
    save_jsonl(animals_pairs, RESULTS_DIR / "dpo_train_animals.jsonl")

    # Random control
    rng = random.Random(999)
    random_pairs = []
    for p in unique_pairs:
        rp = {**p, "metadata": {**p["metadata"], "randomized": True}}
        if rng.random() < 0.5:
            rp["chosen"], rp["rejected"] = rp["rejected"], rp["chosen"]
        random_pairs.append(rp)
    save_jsonl(random_pairs, RESULTS_DIR / "dpo_random.jsonl")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    lines = [
        "# Exp 7v2: Large-scale DPO Pair Collection",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Run Results",
        "",
        "| # | Model | Domain | Attrs | Impls | Contradictions | Corrections | Pairs | Time |",
        "|---|-------|--------|-------|-------|----------------|-------------|-------|------|",
    ]
    for i, s in enumerate(all_summaries, 1):
        if "error" in s:
            lines.append(f"| {i} | {s['model']} | {s['domain']} | — | ERROR | — | — | 0 | — |")
        else:
            lines.append(
                f"| {i} | {s['model']} | {s['domain']} | {s['attrs']} "
                f"| {s['implications']} | {s['contradictions']} | {s['corrections']} "
                f"| {s['pairs']} | {s['elapsed']}s |"
            )

    # Stats
    model_counts = Counter(p["metadata"]["source_model"] for p in unique_pairs)
    domain_counts = Counter(p["metadata"]["source_domain"] for p in unique_pairs)
    chosen_counts = Counter(p["chosen"] for p in unique_pairs)
    success_counts = Counter(p["metadata"]["correction_success"] for p in unique_pairs)
    obj_counts = Counter(p["metadata"]["object"] for p in unique_pairs)

    lines.extend([
        "",
        "## Pair Statistics",
        "",
        f"- **Raw pairs**: {len(all_pairs)}",
        f"- **Unique pairs**: {len(unique_pairs)}",
        f"- **Countries pairs**: {len(countries_pairs)}",
        f"- **Animals pairs**: {len(animals_pairs)}",
        "",
        "### By model",
        "",
    ])
    for m, c in model_counts.most_common():
        lines.append(f"- {m}: {c}")

    lines.extend(["", "### By domain", ""])
    for d, c in domain_counts.most_common():
        lines.append(f"- {d}: {c}")

    lines.extend(["", "### chosen distribution", ""])
    for v, c in chosen_counts.most_common():
        lines.append(f"- chosen={v}: {c}")

    lines.extend(["", "### Correction success", ""])
    for v, c in success_counts.most_common():
        lines.append(f"- {v}: {c}")

    lines.extend(["", "### Top objects (most pairs)", ""])
    for obj, c in obj_counts.most_common(10):
        lines.append(f"- {obj}: {c}")

    lines.extend([
        "",
        "## Files",
        "",
        f"- `dpo_train_all.jsonl`: {len(unique_pairs)} unique pairs",
        f"- `dpo_train_countries.jsonl`: {len(countries_pairs)} pairs",
        f"- `dpo_train_animals.jsonl`: {len(animals_pairs)} pairs",
        f"- `dpo_random.jsonl`: {len(random_pairs)} control pairs",
        f"- `raw/`: per-run raw pairs",
    ])

    summary_text = "\n".join(lines)
    print(summary_text)

    with open(RESULTS_DIR / "pair_summary_v2.md", "w") as f:
        f.write(summary_text + "\n")

    # Save run summaries
    with open(RESULTS_DIR / "run_summaries.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\nDone. {len(unique_pairs)} unique DPO pairs collected.")


if __name__ == "__main__":
    main()
