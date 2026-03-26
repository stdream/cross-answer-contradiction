
"""
Synthetic dataset generator for FCA-guided hallucination research.

What it creates
---------------
1) A hidden rule world of implications over binary attributes.
2) Consistent examples sampled from that world.
3) Query items of the form: "Given A, does B always follow?"
4) Labels:
   - entailment: whether A -> B is true in the hidden world
   - contradiction_example: an example satisfying A but not B, if one exists

This is useful for:
- testing contradiction detection
- measuring hallucination on implication-style questions
- building correction pairs for SFT / DPO

Example
-------
python generate_synthetic_fca_dataset.py \
    --num-attributes 8 \
    --num-rules 10 \
    --num-train 1000 \
    --num-test 200 \
    --seed 42 \
    --out synthetic_fca_dataset.jsonl
"""

from __future__ import annotations
import argparse
import itertools
import json
import random
from dataclasses import dataclass
from typing import Iterable, List, Set, Tuple, Dict, Optional


@dataclass(frozen=True)
class Rule:
    premise: frozenset[str]
    conclusion: frozenset[str]

    def to_dict(self) -> dict:
        return {
            "premise": sorted(self.premise),
            "conclusion": sorted(self.conclusion),
        }


def powerset(xs: List[str]) -> Iterable[Tuple[str, ...]]:
    for r in range(len(xs) + 1):
        yield from itertools.combinations(xs, r)


def closure(seed: Set[str], rules: List[Rule]) -> Set[str]:
    """Compute closure of seed under implication rules."""
    current = set(seed)
    changed = True
    while changed:
        changed = False
        for rule in rules:
            if rule.premise.issubset(current) and not rule.conclusion.issubset(current):
                current |= set(rule.conclusion)
                changed = True
    return current


def sample_hidden_rules(
    attrs: List[str],
    num_rules: int,
    rng: random.Random,
    max_premise_size: int = 3,
    max_conclusion_size: int = 2,
) -> List[Rule]:
    """
    Sample non-trivial hidden rules.
    Rules are of the form premise -> conclusion where premise and conclusion are disjoint.
    """
    rules: List[Rule] = []
    seen = set()

    attempts = 0
    while len(rules) < num_rules and attempts < num_rules * 100:
        attempts += 1
        premise_size = rng.randint(0, min(max_premise_size, len(attrs) - 1))
        premise = frozenset(rng.sample(attrs, k=premise_size))
        remaining = [a for a in attrs if a not in premise]
        if not remaining:
            continue
        conc_size = rng.randint(1, min(max_conclusion_size, len(remaining)))
        conclusion = frozenset(rng.sample(remaining, k=conc_size))

        # Skip trivial/duplicate rules
        key = (tuple(sorted(premise)), tuple(sorted(conclusion)))
        if key in seen:
            continue
        seen.add(key)
        rules.append(Rule(premise, conclusion))

    # Optional dedup by logical redundancy could be added later.
    return rules


def sample_consistent_example(attrs: List[str], rules: List[Rule], rng: random.Random) -> Set[str]:
    """
    Sample a random seed, then close it under the hidden rules.
    This guarantees consistency with the hidden world.
    """
    seed_size = rng.randint(0, len(attrs))
    seed = set(rng.sample(attrs, k=seed_size))
    return closure(seed, rules)


def build_world_examples(
    attrs: List[str],
    rules: List[Rule],
    n: int,
    rng: random.Random,
) -> List[Set[str]]:
    examples = []
    seen = set()
    attempts = 0
    while len(examples) < n and attempts < n * 20:
        attempts += 1
        ex = frozenset(sample_consistent_example(attrs, rules, rng))
        if ex not in seen:
            seen.add(ex)
            examples.append(set(ex))
    return examples


def implication_holds(premise: Set[str], conclusion: Set[str], examples: List[Set[str]]) -> bool:
    """Check if A -> B holds in the finite world represented by examples."""
    for ex in examples:
        if premise.issubset(ex) and not conclusion.issubset(ex):
            return False
    return True


def find_counterexample(premise: Set[str], conclusion: Set[str], examples: List[Set[str]]) -> Optional[List[str]]:
    """Return one counterexample ex with A subset ex and B not subset ex."""
    for ex in examples:
        if premise.issubset(ex) and not conclusion.issubset(ex):
            return sorted(ex)
    return None


def sample_query(
    attrs: List[str],
    rng: random.Random,
    max_premise_size: int = 3,
    max_conclusion_size: int = 2,
) -> Tuple[Set[str], Set[str]]:
    p_size = rng.randint(0, min(max_premise_size, len(attrs)))
    premise = set(rng.sample(attrs, k=p_size))
    remaining = [a for a in attrs if a not in premise]
    c_size = rng.randint(1, min(max_conclusion_size, len(remaining)))
    conclusion = set(rng.sample(remaining, k=c_size))
    return premise, conclusion


def make_natural_language_question(premise: Set[str], conclusion: Set[str]) -> str:
    left = ", ".join(sorted(premise)) if premise else "nothing in particular"
    right = ", ".join(sorted(conclusion))
    return f"If an instance has {left}, must it also have {right}?"


def generate_dataset(
    num_attributes: int,
    num_rules: int,
    num_world_examples: int,
    num_items: int,
    seed: int,
) -> Dict[str, object]:
    rng = random.Random(seed)
    attrs = [f"a{i}" for i in range(1, num_attributes + 1)]

    hidden_rules = sample_hidden_rules(attrs, num_rules, rng)
    world_examples = build_world_examples(attrs, hidden_rules, num_world_examples, rng)

    items = []
    seen_queries = set()

    attempts = 0
    while len(items) < num_items and attempts < num_items * 50:
        attempts += 1
        premise, conclusion = sample_query(attrs, rng)
        qkey = (tuple(sorted(premise)), tuple(sorted(conclusion)))
        if qkey in seen_queries:
            continue
        seen_queries.add(qkey)

        label = implication_holds(premise, conclusion, world_examples)
        cex = None if label else find_counterexample(premise, conclusion, world_examples)

        item = {
            "question": make_natural_language_question(premise, conclusion),
            "premise": sorted(premise),
            "conclusion": sorted(conclusion),
            "label_entails": label,
            "counterexample": cex,
        }
        items.append(item)

    return {
        "attributes": attrs,
        "hidden_rules": [r.to_dict() for r in hidden_rules],
        "world_examples": [sorted(ex) for ex in world_examples],
        "items": items,
    }


def write_jsonl(dataset: Dict[str, object], out_path: str) -> None:
    meta = {
        "attributes": dataset["attributes"],
        "hidden_rules": dataset["hidden_rules"],
        "world_examples": dataset["world_examples"],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "metadata", **meta}, ensure_ascii=False) + "\n")
        for item in dataset["items"]:
            row = {"type": "item", **item}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-attributes", type=int, default=8)
    parser.add_argument("--num-rules", type=int, default=10)
    parser.add_argument("--num-world-examples", type=int, default=200)
    parser.add_argument("--num-train", type=int, default=1000)
    parser.add_argument("--num-test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-train", type=str, default="synthetic_fca_train.jsonl")
    parser.add_argument("--out-test", type=str, default="synthetic_fca_test.jsonl")
    args = parser.parse_args()

    train_ds = generate_dataset(
        num_attributes=args.num_attributes,
        num_rules=args.num_rules,
        num_world_examples=args.num_world_examples,
        num_items=args.num_train,
        seed=args.seed,
    )
    test_ds = generate_dataset(
        num_attributes=args.num_attributes,
        num_rules=args.num_rules,
        num_world_examples=args.num_world_examples,
        num_items=args.num_test,
        seed=args.seed + 1,
    )

    write_jsonl(train_ds, args.out_train)
    write_jsonl(test_ds, args.out_test)

    print(f"Wrote train: {args.out_train}")
    print(f"Wrote test:  {args.out_test}")
    print(f"Train hidden rules: {len(train_ds['hidden_rules'])}")
    print(f"Train items: {len(train_ds['items'])}")
    print(f"Test items:  {len(test_ds['items'])}")


if __name__ == "__main__":
    main()
