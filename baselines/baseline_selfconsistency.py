"""
Baseline: Self-Consistency -- k-shot majority voting
====================================================
Repeat the same question k times, decide by majority vote.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime

from baselines.common import (
    call_ollama, parse_yes_no, load_gold, generate_test_set,
    format_implication_question, save_result, compute_baseline_metrics,
)

logger = logging.getLogger(__name__)


def run(
    domain_name: str,
    domain_desc: str,
    model: str,
    gold_path: str,
    output_dir: str = "results",
    seed: int = 42,
    temperature: float = 0.7,  # SC uses higher temperature for diversity
    k: int = 5,
) -> dict:
    gold = load_gold(gold_path)
    test_set = generate_test_set(gold, seed=seed)

    predictions: list[dict] = []
    num_queries = 0
    t0 = time.time()

    for item in test_set:
        prompt = format_implication_question(
            item["premise"], item["conclusion"], domain_desc,
        )
        votes: list[bool] = []
        for _ in range(k):
            resp = call_ollama(prompt, model, temperature)
            v = parse_yes_no(resp)
            if v is not None:
                votes.append(v)
            num_queries += 1

        # Majority vote (tie → False)
        if votes:
            predicted = sum(votes) > len(votes) / 2
        else:
            predicted = False

        predictions.append({
            **item,
            "predicted": predicted,
            "votes": votes,
            "vote_yes": sum(votes) if votes else 0,
            "vote_total": len(votes),
        })

    elapsed = time.time() - t0
    metrics = compute_baseline_metrics(predictions)

    method = f"selfconsistency_k{k}"
    result = {
        "method": method,
        "domain": domain_name,
        "model": model,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "elapsed_seconds": round(elapsed, 2),
        "num_queries": num_queries,
        "num_test_items": len(test_set),
        "k": k,
        "temperature": temperature,
        "predictions": predictions,
        "metrics": metrics,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{output_dir}/{method}_{domain_name}_{model.replace(':', '_')}_{ts}.json"
    save_result(result, path)
    return result
