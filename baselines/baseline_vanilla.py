"""
Baseline: Vanilla — 직접 함의 질문 (구조 없음)
===============================================
Gold canonical basis + invalid implications을 SLM에 직접 질문.
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
    temperature: float = 0.1,
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
        resp = call_ollama(prompt, model, temperature)
        predicted = parse_yes_no(resp)
        if predicted is None:
            predicted = False
        num_queries += 1
        predictions.append({**item, "predicted": predicted})

    elapsed = time.time() - t0
    metrics = compute_baseline_metrics(predictions)

    result = {
        "method": "vanilla",
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
    path = f"{output_dir}/vanilla_{domain_name}_{model.replace(':', '_')}_{ts}.json"
    save_result(result, path)
    return result
