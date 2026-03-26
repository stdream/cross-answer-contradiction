"""
Exp 7v3: DPO Classification Accuracy — direct measurement without FCA exploration.
==================================================================================
3 models × 2 domains, all (object, attribute) pairs vs gold truth.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines.common import call_ollama, load_gold

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-5s: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "exp7v3"
GOLD_DIR = Path(__file__).resolve().parent.parent / "gold_standards"
DPO_PAIRS_PATH = Path(__file__).resolve().parent.parent / "results" / "exp7v2" / "dpo_train_all.jsonl"

PROMPT_TEMPLATE = (
    "We are classifying {domain} by their common properties.\n"
    "Answer YES if the property commonly applies to the item, NO if not.\n\n"
    "Item: {obj}\n"
    "Property: {attr}\n"
    "Answer:"
)


def parse_yes_no(response: str) -> bool | None:
    low = response.lower().strip()
    if low.startswith("yes"):
        return True
    if low.startswith("no"):
        return False
    return None


def load_seen_pairs() -> set[tuple[str, str]]:
    """Extract (object, attribute) pairs from DPO training data."""
    seen = set()
    if not DPO_PAIRS_PATH.exists():
        return seen
    with open(DPO_PAIRS_PATH) as f:
        for line in f:
            p = json.loads(line)
            obj = p.get("metadata", {}).get("object", "").lower()
            attr = p.get("metadata", {}).get("attribute", "").lower()
            if obj and attr:
                seen.add((obj, attr))
    return seen


def run_classification(
    model: str,
    domain_name: str,
    gold: dict,
    seen_pairs: set[tuple[str, str]],
) -> dict:
    """Run classification for all (object, attribute) pairs."""
    attributes = gold["attributes"]
    objects = gold["objects"]
    total = len(objects) * len(attributes)

    results = []
    correct = 0
    seen_correct = 0
    seen_total = 0
    unseen_correct = 0
    unseen_total = 0
    per_attr: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})

    t0 = time.time()
    done = 0

    for obj_name, obj_attrs_list in sorted(objects.items()):
        obj_attrs = set(obj_attrs_list)
        for attr in attributes:
            gold_val = attr in obj_attrs
            prompt = PROMPT_TEMPLATE.format(domain=domain_name, obj=obj_name.replace("_", " "), attr=attr)
            resp = call_ollama(prompt, model, temperature=0.1)
            predicted = parse_yes_no(resp)
            if predicted is None:
                predicted = False

            is_correct = predicted == gold_val
            is_seen = (obj_name.lower(), attr.lower()) in seen_pairs or \
                      (obj_name.replace("_", " ").lower(), attr.lower()) in seen_pairs

            results.append({
                "object": obj_name,
                "attribute": attr,
                "gold": gold_val,
                "predicted": predicted,
                "correct": is_correct,
                "seen": is_seen,
            })

            if is_correct:
                correct += 1
            per_attr[attr]["total"] += 1
            if is_correct:
                per_attr[attr]["correct"] += 1

            if is_seen:
                seen_total += 1
                if is_correct:
                    seen_correct += 1
            else:
                unseen_total += 1
                if is_correct:
                    unseen_correct += 1

            done += 1

        if done % (len(attributes) * 5) == 0:
            logger.info("  %s × %s: %d/%d (%.0f%%)", model, domain_name, done, total, 100 * done / total)

    elapsed = time.time() - t0

    return {
        "model": model,
        "domain": domain_name,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0,
        "seen_total": seen_total,
        "seen_correct": seen_correct,
        "seen_accuracy": round(seen_correct / seen_total, 4) if seen_total else 0,
        "unseen_total": unseen_total,
        "unseen_correct": unseen_correct,
        "unseen_accuracy": round(unseen_correct / unseen_total, 4) if unseen_total else 0,
        "per_attr": {a: {"accuracy": round(v["correct"] / v["total"], 4), **v} for a, v in per_attr.items()},
        "elapsed_seconds": round(elapsed, 2),
        "raw_results": results,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    models = [
        ("qwen2.5:1.5b", "base"),
        ("qwen2.5-1.5b-dpo-random:latest", "random_dpo"),
        ("qwen2.5-1.5b-dpo-fca:latest", "fca_dpo"),
    ]
    domains = [
        ("countries", "countries.json"),
        ("animals", "animals.json"),
    ]

    seen_pairs = load_seen_pairs()
    logger.info("Loaded %d seen (object, attribute) pairs from DPO training data", len(seen_pairs))

    all_results = {}

    for model, cond in models:
        for domain_name, gold_file in domains:
            gold = load_gold(str(GOLD_DIR / gold_file))
            label = f"{cond}_{domain_name}"
            logger.info("Running: %s (%s × %s)", label, model, domain_name)

            try:
                r = run_classification(model, domain_name, gold, seen_pairs)
                all_results[label] = r
                logger.info(
                    "  %s: acc=%.4f (seen=%.4f/%d, unseen=%.4f/%d) T=%.0fs",
                    label, r["accuracy"],
                    r["seen_accuracy"], r["seen_total"],
                    r["unseen_accuracy"], r["unseen_total"],
                    r["elapsed_seconds"],
                )
            except Exception as e:
                logger.error("  %s ERROR: %s", label, e)
                all_results[label] = {"error": str(e), "model": model, "domain": domain_name}

    # Save raw results (without individual raw_results to keep file manageable)
    save_data = {}
    for k, v in all_results.items():
        save_data[k] = {key: val for key, val in v.items() if key != "raw_results"}
    with open(RESULTS_DIR / "classification_results.json", "w") as f:
        json.dump(save_data, f, indent=2)

    # Generate summary
    lines = [
        "# Exp 7v3: DPO Classification Accuracy",
        "",
        f"Generated: {datetime.now().isoformat()}",
        f"DPO seen pairs: {len(seen_pairs)}",
        "",
        "## Overall Accuracy",
        "",
        "| Condition | Countries Acc | Animals Acc | Total Acc |",
        "|-----------|-------------|------------|-----------|",
    ]

    for _, cond in models:
        rc = all_results.get(f"{cond}_countries", {})
        ra = all_results.get(f"{cond}_animals", {})
        c_acc = rc.get("accuracy", 0)
        a_acc = ra.get("accuracy", 0)
        c_n = rc.get("total", 0)
        a_n = ra.get("total", 0)
        c_corr = rc.get("correct", 0)
        a_corr = ra.get("correct", 0)
        total_acc = round((c_corr + a_corr) / (c_n + a_n), 4) if (c_n + a_n) else 0
        lines.append(f"| {cond} | {c_acc:.4f} ({c_corr}/{c_n}) | {a_acc:.4f} ({a_corr}/{a_n}) | {total_acc:.4f} |")

    lines.extend([
        "",
        "## Seen vs Unseen (FCA-DPO training data)",
        "",
        "| Condition | Seen Acc | Seen N | Unseen Acc | Unseen N |",
        "|-----------|---------|--------|-----------|----------|",
    ])

    for _, cond in models:
        # Combine countries + animals
        seen_c, seen_t, unseen_c, unseen_t = 0, 0, 0, 0
        for dom in ["countries", "animals"]:
            r = all_results.get(f"{cond}_{dom}", {})
            seen_c += r.get("seen_correct", 0)
            seen_t += r.get("seen_total", 0)
            unseen_c += r.get("unseen_correct", 0)
            unseen_t += r.get("unseen_total", 0)
        s_acc = round(seen_c / seen_t, 4) if seen_t else 0
        u_acc = round(unseen_c / unseen_t, 4) if unseen_t else 0
        lines.append(f"| {cond} | {s_acc:.4f} | {seen_t} | {u_acc:.4f} | {unseen_t} |")

    # Per-attribute delta for countries
    base_c = all_results.get("base_countries", {})
    fca_c = all_results.get("fca_dpo_countries", {})
    if base_c.get("per_attr") and fca_c.get("per_attr"):
        lines.extend([
            "",
            "## Per-Attribute Accuracy (Countries)",
            "",
            "| Attribute | Base | FCA-DPO | Delta |",
            "|-----------|------|---------|-------|",
        ])
        for attr in sorted(base_c["per_attr"]):
            b = base_c["per_attr"][attr]["accuracy"]
            f = fca_c["per_attr"].get(attr, {}).get("accuracy", 0)
            delta = round(f - b, 4)
            lines.append(f"| {attr} | {b:.4f} | {f:.4f} | {delta:+.4f} |")

    # Consistency check: base wrong → fca correct, and vice versa
    base_results_map = {}
    fca_results_map = {}
    for label, data in all_results.items():
        if "raw_results" not in data:
            continue
        for r in data["raw_results"]:
            key = (r["object"], r["attribute"], data.get("domain", label.split("_", 1)[-1]))
            if "base" in label:
                base_results_map[key] = r
            elif "fca_dpo" in label:
                fca_results_map[key] = r

    if base_results_map and fca_results_map:
        base_wrong_fca_right = 0
        base_right_fca_wrong = 0
        both_checked = 0
        for key in base_results_map:
            if key in fca_results_map:
                both_checked += 1
                b = base_results_map[key]["correct"]
                f = fca_results_map[key]["correct"]
                if not b and f:
                    base_wrong_fca_right += 1
                elif b and not f:
                    base_right_fca_wrong += 1

        lines.extend([
            "",
            "## Consistency Analysis",
            "",
            f"- Pairs compared: {both_checked}",
            f"- Base wrong → FCA-DPO correct: {base_wrong_fca_right}",
            f"- Base correct → FCA-DPO wrong (regression): {base_right_fca_wrong}",
            f"- Net improvement: {base_wrong_fca_right - base_right_fca_wrong}",
        ])

    summary_text = "\n".join(lines)
    print(summary_text)

    with open(RESULTS_DIR / "classification_summary.md", "w") as f:
        f.write(summary_text + "\n")

    logger.info("Saved: %s", RESULTS_DIR / "classification_summary.md")


if __name__ == "__main__":
    main()
