"""
Experiment Runner — CLI + JSONL logging
========================================
Usage:
    python run.py --domain fruits --model qwen2.5:7b
    python run.py --domain countries --model qwen2.5:7b
    python run.py --config experiments/exp1_main.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from domain import DOMAINS, SCALING_DOMAINS
from fca_engine import full_exploration
from oracle import OllamaOracle, OracleConfig

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_single(
    domain_name: str,
    model: str = "qwen2.5:7b",
    output_dir: str = "results",
    self_correction: bool = True,
    temperature: float = 0.1,
) -> dict:
    """Run a single FCA exploration."""
    all_domains = {**DOMAINS, **SCALING_DOMAINS}
    if domain_name not in all_domains:
        print(
            f"Unknown domain '{domain_name}'. "
            f"Available: {sorted(all_domains)}",
            file=sys.stderr,
        )
        sys.exit(1)
    domain = all_domains[domain_name]

    config = OracleConfig(
        model=model,
        temperature=temperature,
        self_correction=self_correction,
    )
    oracle = OllamaOracle(domain=domain, config=config)

    logger.info(
        "=== %s × %s (self_correction=%s) ===",
        domain_name, model, self_correction,
    )
    t0 = time.time()

    initial = domain.get("initial_examples") or None
    result = full_exploration(
        attributes=domain["attributes"],
        oracle=oracle,
        initial_objects=initial,
    )

    elapsed = time.time() - t0

    # Console summary
    print(f"\n{'='*50}")
    print(f"Domain: {domain_name} | Model: {model}")
    print(f"Implications: {result.num_implications}")
    for impl in result.implications:
        print(f"  {impl}")
    print(f"Context objects: {len(result.context.objects)}")
    print(f"Counterexamples: {result.num_counterexamples}")
    print(f"Oracle queries: {oracle.total_queries}")
    print(f"Contradictions detected: {oracle.num_contradictions}")
    print(f"Self-corrections: {oracle.num_corrections}")
    print(f"Time: {elapsed:.1f}s")
    print(f"{'='*50}\n")

    # Save results
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{domain_name}_{model.replace(':', '_')}_{ts}"

    summary = {
        "run_id": run_id,
        "domain": domain_name,
        "model": model,
        "method": "fca",
        "timestamp": ts,
        "elapsed_seconds": round(elapsed, 2),
        "num_implications": result.num_implications,
        "num_counterexamples": result.num_counterexamples,
        "num_oracle_queries": oracle.total_queries,
        "num_contradictions": oracle.num_contradictions,
        "num_corrections": oracle.num_corrections,
        "implications": [
            {"premise": sorted(i.premise), "conclusion": sorted(i.conclusion)}
            for i in result.implications
        ],
        "context_objects": {
            name: sorted(attrs)
            for name, attrs in result.context.objects.items()
        },
        "config": {
            "temperature": temperature,
            "self_correction": self_correction,
        },
    }

    with open(out / f"{run_id}.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # JSONL log: exploration events + oracle queries
    with open(out / f"{run_id}.jsonl", "w") as f:
        for entry in result.exploration_log:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        for ql in oracle.query_log:
            f.write(json.dumps({
                "type": "query",
                "model": ql.model,
                "prompt": ql.prompt,
                "response": ql.response,
            }, ensure_ascii=False) + "\n")

    logger.info("Saved: %s", out / f"{run_id}.json")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(
        description="AutoOntology — FCA Attribute Exploration Runner",
    )
    p.add_argument("--domain", help="Domain name (fruits, countries, animals, se_concepts)")
    p.add_argument("--model", default="qwen2.5:7b", help="Ollama model (default: qwen2.5:7b)")
    p.add_argument("--output", default="results", help="Output directory (default: results)")
    p.add_argument("--config", help="Batch experiment config JSON")
    p.add_argument("--no-self-correction", action="store_true")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("-v", "--verbose", action="store_true")

    args = p.parse_args()
    setup_logging(args.verbose)

    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        runs = cfg.get("runs", [cfg])
        for rc in runs:
            run_single(
                domain_name=rc["domain"],
                model=rc.get("model", args.model),
                output_dir=args.output,
                self_correction=not args.no_self_correction,
                temperature=args.temperature,
            )
    elif args.domain:
        run_single(
            domain_name=args.domain,
            model=args.model,
            output_dir=args.output,
            self_correction=not args.no_self_correction,
            temperature=args.temperature,
        )
    else:
        p.print_help()


if __name__ == "__main__":
    main()
