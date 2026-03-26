"""
Baseline: Structured Survey -- exhaustive survey then compute basis via FCA engine
==================================================================================
Without the FCA exploration algorithm, query the SLM for all object x attribute pairs,
build the cross-table, then compute the canonical basis using fca_engine.

"Uses the FCA engine, but with exhaustive survey instead of exploration" baseline.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

from baselines.common import (
    call_ollama, load_gold, save_result,
)
from fca_engine import FormalContext, Implication, full_exploration

logger = logging.getLogger(__name__)

# ── Gold oracle for computing basis from a built context ─────────────────────

class ContextOracle:
    """Oracle for computing canonical basis from a fully constructed context."""

    def __init__(self, context: FormalContext):
        self.context = context

    def confirm_implication(
        self,
        premise: frozenset[str],
        conclusion: frozenset[str],
        context: FormalContext,
    ) -> tuple[bool, str | None, set[str] | None]:
        # Find counterexample among all objects in the current context
        for name in sorted(self.context.objects):
            if name not in context.objects:
                attrs = self.context.objects[name]
                if premise <= attrs and not conclusion <= attrs:
                    return (False, name, attrs)
        return (True, None, None)


def run(
    domain_name: str,
    domain_desc: str,
    model: str,
    gold_path: str,
    output_dir: str = "results",
    temperature: float = 0.1,
) -> dict:
    """Run exhaustive survey baseline."""
    gold = load_gold(gold_path)
    attributes = gold["attributes"]
    gold_objects = list(gold["objects"].keys())

    # Phase 1: query SLM for all object x attribute pairs (exhaustive survey)
    num_queries = 0
    t0 = time.time()
    slm_context: dict[str, set[str]] = {}

    prompt_template = (
        "We are classifying {domain} by their common properties.\n"
        "Answer YES if the property commonly applies to the item, NO if not.\n\n"
        "Item: {obj}\n"
        "Property: {attr}\n"
        "Answer:"
    )

    for obj in gold_objects:
        obj_attrs: set[str] = set()
        for attr in attributes:
            prompt = prompt_template.format(
                domain=domain_name, obj=obj, attr=attr,
            )
            resp = call_ollama(prompt, model, temperature)
            num_queries += 1
            low = resp.lower().strip()
            if low.startswith("yes"):
                obj_attrs.add(attr)
        slm_context[obj] = obj_attrs
        logger.info("Surveyed %s: %d attrs", obj, len(obj_attrs))

    survey_time = time.time() - t0
    logger.info(
        "Survey done: %d objects × %d attrs = %d queries in %.1fs",
        len(gold_objects), len(attributes), num_queries, survey_time,
    )

    # Phase 2: compute FCA canonical basis from SLM cross-table
    t1 = time.time()
    fc = FormalContext(attributes, slm_context)
    oracle = ContextOracle(fc)
    result = full_exploration(attributes, oracle, initial_objects=slm_context)
    basis_time = time.time() - t1

    elapsed = time.time() - t0

    discovered = [
        {"premise": sorted(impl.premise), "conclusion": sorted(impl.conclusion)}
        for impl in result.implications
    ]

    # Evaluate against gold
    from evaluate import knowledge_accuracy_fca, cross_answer_contradiction_rate
    acc = knowledge_accuracy_fca(discovered, gold["canonical_basis"])
    ccr = cross_answer_contradiction_rate(discovered, gold["objects"])

    summary = {
        "method": "structured_survey",
        "domain": domain_name,
        "model": model,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "elapsed_seconds": round(elapsed, 2),
        "survey_seconds": round(survey_time, 2),
        "basis_seconds": round(basis_time, 2),
        "num_queries": num_queries,
        "num_objects_surveyed": len(gold_objects),
        "num_attributes": len(attributes),
        "num_implications": result.num_implications,
        "implications": discovered,
        "slm_context": {
            name: sorted(attrs) for name, attrs in slm_context.items()
        },
        "metrics": {**acc, "ccr": ccr["ccr"]},
        "ccr_details": ccr,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{output_dir}/structured_{domain_name}_{model.replace(':', '_')}_{ts}.json"
    save_result(summary, path)
    return summary
