"""
Exp 7: DPO Preference Pair Generation
======================================
FCA exploration 중 consistency checker가 탐지하는 contradiction을
DPO training pair (chosen=corrected, rejected=original)로 변환.

Usage: python experiments/exp7_dpo_pairs.py
"""
from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generate_synthetic_fca_dataset import generate_dataset
from fca_engine import FormalContext, Implication, full_exploration, check_consistency
from oracle import OllamaOracle, OracleConfig
from domain import DOMAINS

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "exp7"
GOLD_DIR = Path(__file__).resolve().parent.parent / "gold_standards"


# ── DPO pair extraction from OllamaOracle logs ──────────────────────────────

def extract_dpo_pairs_from_oracle(
    oracle: OllamaOracle,
    source_label: str,
) -> list[dict]:
    """Oracle의 query log에서 self-correction 이벤트를 DPO pair로 변환."""
    pairs = []
    logs = oracle.query_log

    # Self-correction 패턴: "You said X has [premise]. We confirmed ... Does X actually have [attr]?"
    # 이 질문에 YES로 답한 것 = correction 성공
    # 원래 응답은 NO였음 (contradiction 발생 원인)
    for i, ql in enumerate(logs):
        prompt = ql.prompt
        response = ql.response

        if 'We confirmed that things with' in prompt and 'actually have' in prompt:
            # This is a self-correction query
            low = response.lower().strip()
            corrected_to_yes = low.startswith("yes")

            # Extract object and attribute from prompt
            obj = ""
            attr = ""
            try:
                # "You said "X" has: [...]"
                if 'You said "' in prompt:
                    obj = prompt.split('You said "')[1].split('"')[0]
                # 'Does "X" actually have "attr"?'
                if 'actually have "' in prompt:
                    attr = prompt.split('actually have "')[1].split('"')[0]
            except (IndexError, ValueError):
                continue

            if not obj or not attr:
                continue

            # Find the original classification query for this object+attribute
            # It's the query that said NO (causing the contradiction)
            original_prompt = None
            for j in range(i - 1, max(i - 50, -1), -1):
                prev = logs[j]
                if obj.lower() in prev.prompt.lower() and attr.lower() in prev.prompt.lower():
                    if 'We confirmed' not in prev.prompt:  # not a correction query
                        original_prompt = prev.prompt
                        break

            if original_prompt is None:
                # Use the classification prompt template
                original_prompt = (
                    f'We are classifying {oracle.domain["name"]} '
                    f"by their common properties.\n"
                    f"Answer YES if the property commonly applies "
                    f"to the item, NO if not.\n\n"
                    f"Item: {obj}\n"
                    f"Property: {attr}\n"
                    f"Answer:"
                )

            pair = {
                "prompt": original_prompt,
                "chosen": "YES" if corrected_to_yes else "NO",
                "rejected": "NO" if corrected_to_yes else "YES",
                "metadata": {
                    "source": source_label,
                    "object": obj,
                    "attribute": attr,
                    "correction_type": "closure_violation",
                    "correction_success": corrected_to_yes,
                },
            }
            pairs.append(pair)

    return pairs


# ── Synthetic world with SLM oracle ─────────────────────────────────────────

class SyntheticSLMOracle:
    """Synthetic world의 ground truth를 참조하되 SLM처럼 동작하는 oracle.

    noise_rate > 0이면 각 속성 응답을 flip하여 contradiction 유발.
    """

    def __init__(
        self,
        world_examples: list[set[str]],
        attributes: list[str],
        noise_rate: float = 0.0,
        rng: random.Random | None = None,
    ):
        self.world = world_examples
        self.attributes = attributes
        self.noise_rate = noise_rate
        self.rng = rng or random.Random()
        self.query_log: list[dict] = []
        self.total_queries: int = 0
        self.num_corrections: int = 0
        self.num_contradictions: int = 0
        self._confirmed: list[Implication] = []
        self.dpo_pairs: list[dict] = []

    def _noisy_attrs(self, attrs: set[str]) -> set[str]:
        result = set()
        for a in self.attributes:
            has_it = a in attrs
            if self.rng.random() < self.noise_rate:
                has_it = not has_it
                # Record the flip for DPO pair generation
                self.dpo_pairs.append({
                    "prompt": f"Does this entity have the property \"{a}\"?\nAnswer YES or NO.",
                    "chosen": "YES" if (a in attrs) else "NO",  # ground truth
                    "rejected": "NO" if (a in attrs) else "YES",  # flipped
                    "metadata": {
                        "attribute": a,
                        "correction_type": "noise_flip",
                        "ground_truth_has": a in attrs,
                    },
                })
            if has_it:
                result.add(a)
        return result

    def confirm_implication(
        self,
        premise: frozenset[str],
        conclusion: frozenset[str],
        context: FormalContext,
    ) -> tuple[bool, str | None, set[str] | None]:
        self.total_queries += 1

        for i, ex in enumerate(self.world):
            if premise <= ex and not conclusion <= ex:
                noisy_ex = self._noisy_attrs(ex)
                name = f"obj_{i}"

                # Consistency check against confirmed implications
                if self._confirmed:
                    violations = check_consistency(name, noisy_ex, self._confirmed)
                    if violations:
                        self.num_contradictions += len(violations)
                        # Self-correction: revert to ground truth for violated attrs
                        corrected = set(noisy_ex)
                        for v in violations:
                            for m in v.missing_attrs:
                                if m in ex:  # ground truth has it
                                    corrected.add(m)
                                    self.num_corrections += 1
                                    self.dpo_pairs.append({
                                        "prompt": f"Does entity obj_{i} have \"{m}\"?\nAnswer YES or NO.",
                                        "chosen": "YES",
                                        "rejected": "NO",
                                        "metadata": {
                                            "attribute": m,
                                            "correction_type": "closure_violation_corrected",
                                            "object": f"obj_{i}",
                                        },
                                    })
                        noisy_ex = corrected

                if premise <= noisy_ex and not conclusion <= noisy_ex:
                    if name not in context.objects:
                        return (False, name, noisy_ex)

        self._confirmed.append(Implication(premise, conclusion))
        return (True, None, None)


# ── Main ─────────────────────────────────────────────────────────────────────

def run_synthetic_worlds(seeds: list[int], noise_rate: float = 0.0, label_prefix: str = "synthetic") -> list[dict]:
    """여러 synthetic world에서 exploration 실행 → DPO pairs 수집."""
    all_pairs = []
    for seed in seeds:
        ds = generate_dataset(
            num_attributes=8, num_rules=10,
            num_world_examples=200, num_items=0, seed=seed,
        )
        attrs = ds["attributes"]
        world = [set(ex) for ex in ds["world_examples"]]

        rng = random.Random(seed * 7 + 1)
        oracle = SyntheticSLMOracle(world, attrs, noise_rate=noise_rate, rng=rng)
        result = full_exploration(attrs, oracle, max_iterations=50_000)

        label = f"{label_prefix}_seed{seed}_noise{noise_rate}"
        pairs = oracle.dpo_pairs
        for p in pairs:
            p["metadata"]["source"] = label

        print(f"  {label}: {result.num_implications} impls, "
              f"{oracle.num_contradictions} contradictions, "
              f"{oracle.num_corrections} corrections, "
              f"{len(pairs)} pairs")
        all_pairs.extend(pairs)

    return all_pairs


def run_countries_slm(model: str = "qwen2.5:1.5b") -> list[dict]:
    """Countries domain에서 SLM oracle로 exploration → DPO pairs 수집."""
    domain = DOMAINS["countries"]
    config = OracleConfig(
        model=model, temperature=0.1,
        self_correction=True, consistency_check=True, structured_query=True,
    )
    oracle = OllamaOracle(domain=domain, config=config)

    print(f"  Running FCA: countries × {model}")
    t0 = time.time()
    result = full_exploration(
        attributes=domain["attributes"],
        oracle=oracle,
        initial_objects=domain.get("initial_examples") or None,
    )
    elapsed = time.time() - t0

    pairs = extract_dpo_pairs_from_oracle(oracle, f"countries_{model}")
    print(f"  countries: {result.num_implications} impls, "
          f"{oracle.num_contradictions} contradictions, "
          f"{oracle.num_corrections} corrections, "
          f"{len(pairs)} pairs, {elapsed:.0f}s")

    return pairs


def save_pairs(pairs: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  Saved {len(pairs)} pairs → {path}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Task 1: Synthetic worlds (train + eval) ─────────────────────────
    print("=" * 60)
    print("Task 1: Synthetic worlds (noise=0)")
    print("=" * 60)
    train_pairs_syn = run_synthetic_worlds([42, 43, 44], noise_rate=0.0, label_prefix="syn_train")
    eval_pairs_syn = run_synthetic_worlds([45, 46], noise_rate=0.0, label_prefix="syn_eval")

    # ── Task 2: Countries with SLM ──────────────────────────────────────
    print()
    print("=" * 60)
    print("Task 2: Countries domain (qwen2.5:1.5b)")
    print("=" * 60)
    countries_pairs = run_countries_slm("qwen2.5:1.5b")

    # ── Task 3: Check pair count, supplement if needed ──────────────────
    total_train = len(train_pairs_syn) + len(countries_pairs)
    print()
    print(f"Total train pairs so far: {total_train}")

    supplement_pairs = []
    if total_train < 50:
        print()
        print("=" * 60)
        print("Task 3: Supplementing with noisy synthetic worlds")
        print("=" * 60)
        # 5% noise injection for more contradictions
        supplement_pairs = run_synthetic_worlds(
            list(range(47, 57)), noise_rate=0.05, label_prefix="syn_noisy",
        )
        print(f"  Supplement: {len(supplement_pairs)} additional pairs")

        if total_train + len(supplement_pairs) < 50:
            # 10% noise
            more = run_synthetic_worlds(
                list(range(57, 67)), noise_rate=0.10, label_prefix="syn_noisy10",
            )
            supplement_pairs.extend(more)
            print(f"  More supplement (10% noise): {len(more)} additional pairs")

    all_train = train_pairs_syn + countries_pairs + supplement_pairs
    print(f"\nFinal train pairs: {len(all_train)}")

    # ── Save pairs ──────────────────────────────────────────────────────
    save_pairs(all_train, RESULTS_DIR / "dpo_pairs_synthetic_train.jsonl")
    save_pairs(eval_pairs_syn, RESULTS_DIR / "dpo_pairs_synthetic_eval.jsonl")
    save_pairs(countries_pairs, RESULTS_DIR / "dpo_pairs_countries.jsonl")

    # ── Task 4: Random-DPO control ──────────────────────────────────────
    print()
    print("=" * 60)
    print("Task 4: Random-DPO control pairs")
    print("=" * 60)
    rng = random.Random(999)
    random_pairs = []
    for p in all_train:
        rp = dict(p)
        rp["metadata"] = {**p["metadata"], "randomized": True}
        if rng.random() < 0.5:
            rp["chosen"], rp["rejected"] = rp["rejected"], rp["chosen"]
        random_pairs.append(rp)
    save_pairs(random_pairs, RESULTS_DIR / "dpo_pairs_random.jsonl")

    # ── Task 5: Summary ─────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Task 5: Summary")
    print("=" * 60)

    summary = [
        "# Exp 7: DPO Preference Pair Generation",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Pair Counts",
        "",
        f"| Source | Pairs |",
        f"|--------|-------|",
        f"| Synthetic train (seed 42-44, noise=0) | {len(train_pairs_syn)} |",
        f"| Countries (qwen2.5:1.5b) | {len(countries_pairs)} |",
        f"| Synthetic supplement (noisy) | {len(supplement_pairs)} |",
        f"| **Total train** | **{len(all_train)}** |",
        f"| Synthetic eval (seed 45-46) | {len(eval_pairs_syn)} |",
        f"| Random-DPO control | {len(random_pairs)} |",
        "",
        "## Correction Statistics",
        "",
    ]

    # Count correction types
    type_counts: dict[str, int] = {}
    for p in all_train:
        ct = p["metadata"].get("correction_type", "unknown")
        type_counts[ct] = type_counts.get(ct, 0) + 1

    for ct, cnt in sorted(type_counts.items()):
        summary.append(f"- {ct}: {cnt}")

    summary.extend([
        "",
        "## Files",
        "",
        f"- `dpo_pairs_synthetic_train.jsonl`: {len(all_train)} pairs",
        f"- `dpo_pairs_synthetic_eval.jsonl`: {len(eval_pairs_syn)} pairs",
        f"- `dpo_pairs_countries.jsonl`: {len(countries_pairs)} pairs",
        f"- `dpo_pairs_random.jsonl`: {len(random_pairs)} pairs (control)",
    ])

    summary_text = "\n".join(summary)
    print(summary_text)

    with open(RESULTS_DIR / "pair_summary.md", "w") as f:
        f.write(summary_text + "\n")

    print(f"\nSaved: {RESULTS_DIR / 'pair_summary.md'}")


if __name__ == "__main__":
    main()
