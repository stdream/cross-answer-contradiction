"""
Exp 6: Synthetic Noise Study
=============================
Measures the effect of oracle noise on FCA exploration.
Runs quickly without SLM using synthetic world + noisy oracle.

Usage:
    python experiments/exp6_synthetic.py
"""
from __future__ import annotations

import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.generate_synthetic_fca_dataset import (
    generate_dataset,
    closure,
    Rule,
)
from fca_engine import (
    FormalContext,
    Implication,
    full_exploration,
    closure_under_implications,
)
from evaluate import knowledge_accuracy_fca

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "exp6"


# ── Noisy Oracle ─────────────────────────────────────────────────────────────

class NoisyOracle:
    """Oracle that injects errors at noise_rate into the ground truth world.

    - confirm_implication: finds counterexamples from world examples,
      flipping counterexample attributes with noise_rate probability.
    - noise=0 means perfect oracle.
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
        self.num_contradictions = 0

    def _noisy_attrs(self, attrs: set[str]) -> set[str]:
        """Flip each attribute's yes/no with noise_rate probability."""
        result = set()
        for a in self.attributes:
            has_it = a in attrs
            if self.rng.random() < self.noise_rate:
                has_it = not has_it  # flip
            if has_it:
                result.add(a)
        return result

    def confirm_implication(
        self,
        premise: frozenset[str],
        conclusion: frozenset[str],
        context: FormalContext,
    ) -> tuple[bool, str | None, set[str] | None]:
        # Find actual counterexample from world
        for i, ex in enumerate(self.world):
            if premise <= ex and not conclusion <= ex:
                # Counterexample found -- inject noise
                noisy_ex = self._noisy_attrs(ex)
                name = f"obj_{i}"

                # Check if still a valid counterexample after noise
                if premise <= noisy_ex and not conclusion <= noisy_ex:
                    if name not in context.objects:
                        return (False, name, noisy_ex)

        # No counterexample (or invalidated by noise) -> accept
        # When noise_rate > 0, some implications that do not actually hold may be accepted
        return (True, None, None)


# ── Gold basis computation ───────────────────────────────────────────────────

def compute_gold_basis(
    attributes: list[str],
    world_examples: list[set[str]],
) -> list[dict]:
    """Compute gold canonical basis using noise=0 perfect oracle."""
    oracle = NoisyOracle(world_examples, attributes, noise_rate=0.0)
    result = full_exploration(attributes, oracle)
    return [
        {"premise": sorted(impl.premise), "conclusion": sorted(impl.conclusion)}
        for impl in result.implications
    ]


# ── Single run ───────────────────────────────────────────────────────────────

def run_single(
    attributes: list[str],
    world_examples: list[set[str]],
    gold_basis: list[dict],
    noise_rate: float,
    seed: int,
) -> dict:
    """Run a single noisy exploration -> return metrics."""
    rng = random.Random(seed)
    oracle = NoisyOracle(world_examples, attributes, noise_rate, rng)

    t0 = time.time()
    result = full_exploration(attributes, oracle, max_iterations=50_000)
    elapsed = time.time() - t0

    discovered = [
        {"premise": sorted(impl.premise), "conclusion": sorted(impl.conclusion)}
        for impl in result.implications
    ]

    acc = knowledge_accuracy_fca(discovered, gold_basis)

    return {
        "noise_rate": noise_rate,
        "seed": seed,
        "num_implications": result.num_implications,
        "num_counterexamples": result.num_counterexamples,
        "num_questions": result.num_questions,
        "elapsed_seconds": round(elapsed, 4),
        "precision": acc["precision"],
        "recall": acc["recall"],
        "f1": acc["f1"],
    }


# ── Main experiment ──────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Generate synthetic world
    # Try multiple seeds to select a world with a sufficiently rich gold basis
    print("Generating synthetic world...")
    best_ds, best_basis_len = None, 0
    for try_seed in [42, 43, 44, 45, 99, 123, 200, 314, 500, 777]:
        ds_candidate = generate_dataset(
            num_attributes=8,
            num_rules=10,
            num_world_examples=200,
            num_items=0,
            seed=try_seed,
        )
        world_candidate = [set(ex) for ex in ds_candidate["world_examples"]]
        basis_candidate = compute_gold_basis(ds_candidate["attributes"], world_candidate)
        if len(basis_candidate) > best_basis_len:
            best_ds = ds_candidate
            best_basis_len = len(basis_candidate)
            best_seed = try_seed
        if best_basis_len >= 8:
            break
    ds = best_ds
    print(f"  Selected seed={best_seed} with {best_basis_len} gold implications")
    attributes = ds["attributes"]
    world_examples = [set(ex) for ex in ds["world_examples"]]
    hidden_rules = ds["hidden_rules"]

    print(f"  Attributes: {len(attributes)}")
    print(f"  Hidden rules: {len(hidden_rules)}")
    print(f"  World examples: {len(world_examples)}")

    # 2. Compute gold basis (noise=0)
    print("Computing gold basis (noise=0)...")
    gold_basis = compute_gold_basis(attributes, world_examples)
    print(f"  Gold basis: {len(gold_basis)} implications")
    for b in gold_basis:
        p = ", ".join(b["premise"]) or "∅"
        c = ", ".join(sorted(set(b["conclusion"]) - set(b["premise"])))
        print(f"    {{{p}}} → {{{c}}}")

    # 3. Run experiments
    noise_rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
    n_repeats = 3
    all_results: list[dict] = []

    print(f"\nRunning: {len(noise_rates)} noise rates × {n_repeats} repeats")
    print("-" * 70)

    for nr in noise_rates:
        run_results = []
        for rep in range(n_repeats):
            seed = 1000 + rep * 100 + int(nr * 1000)
            r = run_single(attributes, world_examples, gold_basis, nr, seed)
            run_results.append(r)
            all_results.append(r)

        # Per-noise summary
        ps = [r["precision"] for r in run_results]
        rs = [r["recall"] for r in run_results]
        fs = [r["f1"] for r in run_results]
        qs = [r["num_questions"] for r in run_results]
        ni = [r["num_implications"] for r in run_results]

        p_mean, p_std = statistics.mean(ps), (statistics.stdev(ps) if len(ps) > 1 else 0)
        r_mean, r_std = statistics.mean(rs), (statistics.stdev(rs) if len(rs) > 1 else 0)
        f_mean, f_std = statistics.mean(fs), (statistics.stdev(fs) if len(fs) > 1 else 0)

        print(
            f"  noise={nr:.2f}  "
            f"P={p_mean:.3f}±{p_std:.3f}  "
            f"R={r_mean:.3f}±{r_std:.3f}  "
            f"F1={f_mean:.3f}±{f_std:.3f}  "
            f"impls={statistics.mean(ni):.0f}  "
            f"questions={statistics.mean(qs):.0f}"
        )

    # 4. Save raw results
    raw_path = RESULTS_DIR / "exp6_raw.json"
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # 5. Generate summary table
    summary_lines = [
        "# Exp 6: Synthetic Noise Study",
        "",
        f"Generated: {datetime.now().isoformat()}",
        f"World: {len(attributes)} attrs, {len(hidden_rules)} rules, "
        f"{len(world_examples)} examples, {len(gold_basis)} gold implications",
        "",
        "| Noise Rate | P (mean±std) | R (mean±std) | F1 (mean±std) | Impls | Questions |",
        "|------------|-------------|-------------|--------------|-------|-----------|",
    ]

    for nr in noise_rates:
        runs = [r for r in all_results if r["noise_rate"] == nr]
        ps = [r["precision"] for r in runs]
        rs = [r["recall"] for r in runs]
        fs = [r["f1"] for r in runs]
        ni = [r["num_implications"] for r in runs]
        qs = [r["num_questions"] for r in runs]

        def fmt(vals):
            m = statistics.mean(vals)
            s = statistics.stdev(vals) if len(vals) > 1 else 0
            return f"{m:.3f}±{s:.3f}"

        summary_lines.append(
            f"| {nr:.2f} | {fmt(ps)} | {fmt(rs)} | {fmt(fs)} "
            f"| {statistics.mean(ni):.0f} | {statistics.mean(qs):.0f} |"
        )

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    summary_path = RESULTS_DIR / "exp6_summary.md"
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")

    print(f"\nSaved: {raw_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
