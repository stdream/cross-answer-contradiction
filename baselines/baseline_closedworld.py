"""
Baseline: Closed-World -- rule judgment with explicit list of 50 countries
=========================================================================
Addresses reviewer concern: fairness of open-world baseline vs closed-world FCA.
Requests rule judgment within the same closed-world (50 countries).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime

from baselines.common import (
    call_ollama, parse_yes_no, load_gold, generate_test_set,
    save_result, compute_baseline_metrics,
)

logger = logging.getLogger(__name__)


def _format_closed_world_prompt(
    premise: list[str],
    conclusion: list[str],
    country_list: str,
    cot: bool = False,
) -> str:
    premise_str = ", ".join(premise) or "no specific properties"
    added = sorted(set(conclusion) - set(premise))
    added_str = ", ".join(added)

    if cot:
        return (
            f"Consider the following 50 countries: {country_list}.\n"
            f"Among ONLY these 50 countries, we claim: "
            f"if a country has [{premise_str}], then it also has [{added_str}].\n"
            f"Think step by step about which of the 50 countries have "
            f"[{premise_str}], then check if they all also have [{added_str}].\n"
            f"Answer YES or NO at the end."
        )
    else:
        return (
            f"Consider the following 50 countries: {country_list}.\n"
            f"Among ONLY these 50 countries, we claim: "
            f"if a country has [{premise_str}], then it also has [{added_str}].\n"
            f"Does this rule hold for all 50 countries listed above?\n"
            f"Answer only YES or NO."
        )


def run(
    domain_name: str,
    model: str,
    gold_path: str,
    output_dir: str = "results",
    seed: int = 42,
    temperature: float = 0.1,
    cot: bool = False,
) -> dict:
    gold = load_gold(gold_path)
    test_set = generate_test_set(gold, seed=seed)

    # Build country list string from gold objects
    countries = sorted(gold["objects"].keys())
    country_list = ", ".join(c.replace("_", " ").title() for c in countries)

    method = "closedworld_cot" if cot else "closedworld_vanilla"

    predictions: list[dict] = []
    num_queries = 0
    t0 = time.time()

    for item in test_set:
        prompt = _format_closed_world_prompt(
            item["premise"], item["conclusion"], country_list, cot=cot,
        )
        resp = call_ollama(prompt, model, temperature)

        # For CoT, extract last YES/NO
        predicted = None
        if cot:
            for line in reversed(resp.split("\n")):
                predicted = parse_yes_no(line.strip())
                if predicted is not None:
                    break
        if predicted is None:
            predicted = parse_yes_no(resp)
        if predicted is None:
            predicted = False

        num_queries += 1
        predictions.append({**item, "predicted": predicted})

    elapsed = time.time() - t0
    metrics = compute_baseline_metrics(predictions)

    result = {
        "method": method,
        "domain": domain_name,
        "model": model,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "elapsed_seconds": round(elapsed, 2),
        "num_queries": num_queries,
        "num_test_items": len(test_set),
        "predictions": predictions,
        "metrics": metrics,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{output_dir}/{method}_{domain_name}_{model.replace(':', '_')}_{ts}.json"
    save_result(result, path)
    return result
