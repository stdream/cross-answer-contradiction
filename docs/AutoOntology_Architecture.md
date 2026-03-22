# AutoOntology — Project Architecture

> Autoresearch pattern for knowledge discovery with formal completeness guarantees
>
> Design doc v1.0 | March 22, 2026

---

## Philosophy: Three Files That Matter

Following Karpathy's autoresearch pattern:

```
autoresearch (Karpathy)          AutoOntology (Ours)
─────────────────────────────    ─────────────────────────────
prepare.py  (fixed, data+eval)   fca_engine.py  (fixed, algorithm+eval)
train.py    (agent edits)        domain.py      (defines exploration scope)
program.md  (human writes)       program.md     (human writes research direction)
```

**Key difference**: In autoresearch, the *agent* modifies train.py (code optimization).
In AutoOntology, the *agent* (SLM) answers questions from fca_engine.py (knowledge discovery).
The human edits domain.py to define *what* to explore, and program.md to guide *how*.

---

## Directory Structure

```
autoontology/
├── program.md              # Human-written research direction (autoresearch style)
├── fca_engine.py           # Core FCA algorithm (DO NOT MODIFY - like prepare.py)
├── domain.py               # Domain definitions (human configures exploration scope)
├── oracle.py               # SLM oracle interface (Ollama connection)
├── run.py                  # Main entry point - kicks off exploration
├── evaluate.py             # Metrics & evaluation (completeness, contradiction rate)
├── results/                # Auto-generated exploration logs
│   ├── exploration_log.jsonl
│   ├── contradictions.jsonl
│   └── canonical_basis.json
├── baselines/              # Baseline comparison scripts
│   ├── baseline_cot.py
│   ├── baseline_selfconsistency.py
│   └── baseline_vanilla.py
├── dpo/                    # Stage 2: DPO data generation
│   ├── generate_pairs.py   # Convert contradiction logs to DPO format
│   └── pairs/              # Generated preference pairs
├── tests/
│   └── test_fca.py         # Unit tests for FCA engine correctness
└── pyproject.toml
```

---

## Module Design

### 1. `fca_engine.py` — The Fixed Core (≈300 lines)

This is the equivalent of Karpathy's prepare.py. **Never modified by the agent.**
Contains the mathematical machinery that guarantees completeness.

```python
"""
FCA Attribute Exploration Engine
================================
Implements Algorithm 19 from Ganter & Obiedkov (2016) Ch.4.
This file is FIXED. Do not modify. Correctness depends on it.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable

# --- Core Data Structures ---

@dataclass
class FormalContext:
    """(E, M, J) - examples with their attributes"""
    objects: list[str]                          # object names
    attributes: list[str]                       # attribute names  
    incidence: dict[str, set[str]]              # obj -> set of attrs
    
    def intent(self, obj: str) -> frozenset[str]:
        """Single object's attribute set (g^I)"""
        return frozenset(self.incidence.get(obj, set()))
    
    def common_attributes(self, objs: set[str]) -> frozenset[str]:
        """A' = attributes shared by all objects in A"""
        if not objs:
            return frozenset(self.attributes)
        return frozenset.intersection(*(
            frozenset(self.incidence[o]) for o in objs
        ))
    
    def common_objects(self, attrs: set[str]) -> frozenset[str]:
        """B' = objects that have all attributes in B"""
        if not attrs:
            return frozenset(self.objects)
        return frozenset(
            o for o in self.objects 
            if attrs <= self.incidence.get(o, set())
        )
    
    def double_prime(self, attrs: set[str]) -> frozenset[str]:
        """A'' = closure of attribute set A in the example context"""
        return self.common_attributes(self.common_objects(attrs))
    
    def add_object(self, name: str, attrs: set[str]):
        """Add a new counterexample to the context"""
        if name not in self.objects:
            self.objects.append(name)
        self.incidence[name] = set(attrs)


@dataclass 
class Implication:
    """A -> B where A, B are sets of attributes"""
    premise: frozenset[str]
    conclusion: frozenset[str]
    
    def holds_for(self, attrs: set[str]) -> bool:
        """Does this implication hold for a given attribute set?"""
        return not (self.premise <= attrs) or (self.conclusion <= attrs)
    
    def __repr__(self):
        p = ",".join(sorted(self.premise)) if self.premise else "∅"
        c = ",".join(sorted(self.conclusion - self.premise))
        return f"{{{p}}} → {{{c}}}"


@dataclass
class ExplorationState:
    """Complete state of an ongoing exploration"""
    context: FormalContext
    confirmed: list[Implication] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)  # {implication, counterexample, step}
    contradictions: list[dict] = field(default_factory=list)
    step: int = 0
    complete: bool = False


# --- Closure Operations ---

def closure_under_implications(attrs: set[str], implications: list[Implication]) -> frozenset[str]:
    """
    L(A) = closure of attribute set A under implication set L.
    Iteratively applies all implications until fixpoint.
    """
    result = set(attrs)
    changed = True
    while changed:
        changed = False
        for impl in implications:
            if impl.premise <= result and not (impl.conclusion <= result):
                result |= impl.conclusion
                changed = True
    return frozenset(result)


# --- Lectic Order (Next Closure) ---

def lectic_less(a: frozenset, b: frozenset, order: list[str]) -> bool:
    """Is a lectically smaller than b? (Definition 3, Ganter & Obiedkov)"""
    for m in order:
        if (m in b) and (m not in a):
            return True
        if (m in a) and (m not in b):
            return False
    return False


def next_closure(current: frozenset, M: list[str], implications: list[Implication]) -> Optional[frozenset]:
    """
    Compute the lectically next closed set after 'current'.
    Returns None if current is already M (exploration complete).
    Algorithm from Section 2.2.2 of Ganter & Obiedkov (2016).
    """
    for i in range(len(M) - 1, -1, -1):
        m = M[i]
        if m in current:
            continue
        # Try adding m
        candidate = frozenset({x for x in current if M.index(x) < i} | {m})
        closed = closure_under_implications(candidate, implications)
        # Check: is the closure lectically valid?
        valid = True
        for j in range(i):
            if M[j] in closed and M[j] not in candidate:
                valid = False
                break
        if valid:
            return closed
    return None  # current was M, exploration is done


# --- Main Exploration Loop ---

@dataclass
class ExplorationResult:
    implication: Implication
    premise_readable: str
    conclusion_readable: str
    question: str

def generate_question(state: ExplorationState) -> Optional[ExplorationResult]:
    """
    Find the next question to ask the oracle.
    Returns None if exploration is complete.
    
    Core logic (Algorithm 19):
    Find the lectically smallest A where L(A) = A but A ≠ A^JJ.
    If found, ask: "Does A → A^JJ hold in the domain?"
    """
    M = state.context.attributes
    
    if state.step == 0:
        # Start with empty set
        A = closure_under_implications(frozenset(), state.confirmed)
    else:
        # Find next closure after previous premise
        A = None  # Will be set by the caller via advance()
        return None  # Placeholder - actual logic in explore_step
    
    return _check_and_formulate(A, state)


def _check_and_formulate(A: frozenset, state: ExplorationState) -> Optional[ExplorationResult]:
    """Check if A generates a question, formulate it if so."""
    A_double_prime = state.context.double_prime(A)
    
    if A == A_double_prime:
        return None  # A is already closed in examples, no question needed
    
    impl = Implication(premise=A, conclusion=A_double_prime)
    
    p_str = ", ".join(sorted(A)) if A else "∅"
    new_attrs = A_double_prime - A
    c_str = ", ".join(sorted(new_attrs))
    
    question = f"If an entity has attributes {{{p_str}}}, must it also have {{{c_str}}}?"
    
    return ExplorationResult(
        implication=impl,
        premise_readable=p_str,
        conclusion_readable=c_str,
        question=question
    )


def explore_step(
    state: ExplorationState,
    oracle_fn: Callable  # (question: str, implication: Implication) -> OracleResponse
) -> bool:
    """
    Execute one step of attribute exploration.
    Returns True if exploration should continue, False if complete.
    
    This is the inner+outer loop of Algorithm 19, one iteration.
    """
    M = state.context.attributes
    
    # Compute current A (lectically smallest L-closed set where A ≠ A^JJ)
    if state.step == 0:
        A = closure_under_implications(frozenset(), state.confirmed)
    else:
        # We need to track the previous A and compute next
        # This is handled by the exploration loop in run.py
        pass
    
    # The full loop is implemented in run.py's main loop
    # This function handles one oracle interaction
    return True


def full_exploration(
    state: ExplorationState,
    oracle_fn: Callable,
    max_steps: int = 1000,
    on_step: Callable = None  # callback for logging
) -> ExplorationState:
    """
    Run complete attribute exploration.
    
    Algorithm 19 (Ganter & Obiedkov 2016):
    L := ∅, A := ∅
    while A ≠ M:
        while A ≠ A^JJ:
            if A^II = A^JJ: 
                L := L ∪ {A → A^JJ}; break
            else:
                extend E by counterexample
        A := NextClosure(A, M, L)
    return L
    """
    M = state.context.attributes
    A = closure_under_implications(frozenset(), state.confirmed)
    
    while A is not None and state.step < max_steps:
        A_jj = state.context.double_prime(A)
        
        if A == A_jj:
            # A is closed in examples, move to next
            A = next_closure(A, M, state.confirmed)
            continue
        
        # A ≠ A^JJ → ask the oracle
        result = _check_and_formulate(A, state)
        if result is None:
            A = next_closure(A, M, state.confirmed)
            continue
        
        state.step += 1
        response = oracle_fn(result.question, result.implication)
        
        if on_step:
            on_step(state.step, result, response)
        
        if response.confirmed:
            # Oracle confirms: add implication
            state.confirmed.append(result.implication)
            # Recompute A under new implications
            A = closure_under_implications(A, state.confirmed)
            # If now A == A^JJ, move to next closure
            if A == state.context.double_prime(A):
                A = next_closure(A, M, state.confirmed)
        else:
            # Oracle rejects: add counterexample
            if response.counterexample:
                ce = response.counterexample
                # Consistency check: does counterexample violate any confirmed implication?
                contradiction = check_consistency(ce, state.confirmed)
                if contradiction:
                    state.contradictions.append({
                        "step": state.step,
                        "counterexample": ce,
                        "violated_implication": str(contradiction),
                        "question": result.question
                    })
                    # Re-query oracle with contradiction evidence
                    # (handled by oracle.py's self-correction loop)
                else:
                    state.context.add_object(ce["name"], ce["attributes"])
                    state.rejected.append({
                        "step": state.step,
                        "implication": str(result.implication),
                        "counterexample": ce["name"]
                    })
            # Recompute A^JJ with new context
            # Stay on same A (inner while loop)
    
    if A is None:
        state.complete = True
    
    return state


def check_consistency(
    counterexample: dict,  # {"name": str, "attributes": set[str]}
    confirmed: list[Implication]
) -> Optional[Implication]:
    """
    Check if a counterexample violates any confirmed implication.
    Returns the violated implication, or None if consistent.
    
    THIS IS THE HALLUCINATION DETECTOR.
    """
    attrs = set(counterexample["attributes"])
    for impl in confirmed:
        if not impl.holds_for(attrs):
            return impl
    return None


# --- Evaluation ---

def completeness_score(state: ExplorationState) -> dict:
    """
    Measure how close we are to completeness.
    For every possible L-closed set A, check if L(A) == A^JJ.
    """
    M = state.context.attributes
    total = 0
    matched = 0
    
    A = closure_under_implications(frozenset(), state.confirmed)
    while A is not None:
        total += 1
        A_jj = state.context.double_prime(A)
        if A == A_jj or closure_under_implications(A, state.confirmed) == A_jj:
            matched += 1
        A = next_closure(A, M, state.confirmed)
    
    return {
        "total_closed_sets": total,
        "matched": matched,
        "completeness": matched / total if total > 0 else 1.0,
        "confirmed_implications": len(state.confirmed),
        "counterexamples": len(state.context.objects),
        "contradictions_detected": len(state.contradictions)
    }
```

### 2. `oracle.py` — SLM Interface (≈150 lines)

Connects to Ollama. Handles prompt formatting, response parsing, and self-correction loop.

```python
"""
SLM Oracle Interface
====================
Connects to Ollama for SLM inference.
Handles: prompt formatting, response parsing, self-correction on contradiction.
"""

import json
import requests
from dataclasses import dataclass
from typing import Optional
from fca_engine import Implication, ExplorationState

OLLAMA_URL = "http://localhost:11434/api/generate"

@dataclass
class OracleResponse:
    confirmed: bool
    counterexample: Optional[dict] = None  # {"name": str, "attributes": set[str]}
    raw_response: str = ""
    retries: int = 0


class OllamaOracle:
    def __init__(self, model: str = "qwen2.5:7b", domain_context: str = ""):
        self.model = model
        self.domain_context = domain_context
        self.max_retries = 3
        self.history = []  # Track all Q&A for analysis
    
    def query(self, question: str, implication: Implication, 
              state: ExplorationState = None) -> OracleResponse:
        """
        Ask the SLM oracle a yes/no question about an implication.
        Includes self-correction loop if contradiction detected.
        """
        prompt = self._format_prompt(question, implication, state)
        
        for attempt in range(self.max_retries):
            raw = self._call_ollama(prompt)
            response = self._parse_response(raw, implication, state)
            
            if response is None:
                # Parse failure, retry with clarification
                prompt = self._format_retry_prompt(question, raw)
                continue
            
            # Check for contradiction with confirmed knowledge
            if not response.confirmed and response.counterexample and state:
                from fca_engine import check_consistency
                violation = check_consistency(response.counterexample, state.confirmed)
                
                if violation:
                    # HALLUCINATION DETECTED
                    response.retries = attempt + 1
                    
                    if attempt < self.max_retries - 1:
                        # Self-correction: re-query with contradiction evidence
                        prompt = self._format_correction_prompt(
                            question, response.counterexample, violation
                        )
                        continue
                    else:
                        # Max retries reached, log contradiction
                        return response
            
            response.retries = attempt
            self.history.append({
                "question": question,
                "implication": str(implication),
                "response": response,
                "attempts": attempt + 1
            })
            return response
        
        # Fallback: couldn't get valid response
        return OracleResponse(confirmed=True, raw_response="[FALLBACK: no valid response]")
    
    def _format_prompt(self, question: str, impl: Implication, 
                       state: ExplorationState = None) -> str:
        """Format the exploration question as an SLM prompt."""
        
        # Build context from confirmed knowledge
        knowledge = ""
        if state and state.confirmed:
            rules = "\n".join(f"  - {impl}" for impl in state.confirmed[-10:])
            knowledge = f"\nAlready confirmed rules:\n{rules}\n"
        
        if state and state.context.objects:
            examples = "\n".join(
                f"  - {obj}: {{{', '.join(sorted(state.context.incidence[obj]))}}}"
                for obj in state.context.objects[-10:]
            )
            knowledge += f"\nKnown examples:\n{examples}\n"
        
        return f"""You are a domain expert answering questions about {self.domain_context}.

{knowledge}
Question: {question}

You must answer in this exact JSON format:
{{"answer": "yes"}} if the rule always holds, OR
{{"answer": "no", "counterexample": {{"name": "<entity name>", "attributes": ["attr1", "attr2", ...]}}}}

Respond with ONLY the JSON, nothing else."""
    
    def _format_correction_prompt(self, question: str, 
                                   counterexample: dict, 
                                   violated: Implication) -> str:
        """Re-query with contradiction evidence (self-correction)."""
        
        return f"""CORRECTION NEEDED: Your previous answer contained a contradiction.

You provided counterexample "{counterexample['name']}" with attributes {counterexample['attributes']}.
But this VIOLATES the already confirmed rule: {violated}

Please reconsider. Either:
1. The rule {violated} is wrong (but you confirmed it earlier), OR  
2. Your counterexample is wrong (it should have different attributes), OR
3. The original question should be answered "yes" (the rule does hold)

Original question: {question}

Respond with ONLY valid JSON:
{{"answer": "yes"}} OR
{{"answer": "no", "counterexample": {{"name": "<name>", "attributes": ["attr1", ...]}}}}"""
    
    def _format_retry_prompt(self, question: str, bad_response: str) -> str:
        return f"""Your previous response could not be parsed as JSON.
Previous response: {bad_response[:200]}

Please answer this question with ONLY valid JSON:
{question}

Format: {{"answer": "yes"}} or {{"answer": "no", "counterexample": {{"name": "...", "attributes": [...]}}}}"""
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API."""
        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 256}
            }, timeout=60)
            return resp.json().get("response", "")
        except Exception as e:
            return f"[ERROR: {e}]"
    
    def _parse_response(self, raw: str, impl: Implication, 
                        state: ExplorationState) -> Optional[OracleResponse]:
        """Parse SLM JSON response into OracleResponse."""
        try:
            # Extract JSON from response (handle markdown code blocks)
            text = raw.strip()
            if "```" in text:
                text = text.split("```")[1].replace("json", "").strip()
            
            data = json.loads(text)
            
            if data.get("answer", "").lower() == "yes":
                return OracleResponse(confirmed=True, raw_response=raw)
            
            elif data.get("answer", "").lower() == "no":
                ce = data.get("counterexample", {})
                if ce and "name" in ce and "attributes" in ce:
                    # Validate: counterexample must have the premise attributes
                    ce_attrs = set(ce["attributes"])
                    if impl.premise <= ce_attrs and not (impl.conclusion <= ce_attrs):
                        return OracleResponse(
                            confirmed=False,
                            counterexample={"name": ce["name"], "attributes": ce_attrs},
                            raw_response=raw
                        )
                    else:
                        return None  # Invalid counterexample, retry
                return None
            
            return None
        except (json.JSONDecodeError, KeyError):
            return None
```

### 3. `domain.py` — Exploration Scope (Human Configures)

```python
"""
Domain Configuration
====================
Human configures WHAT to explore. This is the equivalent of
choosing what dataset/model to optimize in autoresearch.
"""

# --- Domain: Systems Engineering Ontology (BFO-based) ---
SE_DOMAIN = {
    "name": "Systems Engineering Concepts",
    "context": "systems engineering ontology based on BFO/ISO 21838-2",
    "attributes": [
        "is_continuant",        # exists at a time (vs process)
        "is_occurrent",         # unfolds in time (process)
        "is_independent",       # exists on its own
        "is_dependent",         # depends on another entity
        "has_function",         # has an intended function
        "has_requirement",      # associated with requirements
        "is_physical",          # has physical manifestation
        "is_informational",     # information entity
        "is_verifiable",        # can be verified/tested
        "is_decomposable",      # can be broken into sub-parts
    ],
    "initial_examples": {
        # Seed the exploration with a few known examples
        "satellite": {"is_continuant", "is_independent", "is_physical", 
                      "has_function", "has_requirement", "is_decomposable"},
        "orbit_maneuver": {"is_occurrent", "has_requirement", "is_verifiable"},
    }
}

# --- Domain: Toy Fruit Example (for validation) ---
FRUIT_DOMAIN = {
    "name": "Fruits",
    "context": "properties of common fruits",
    "attributes": ["red", "sweet", "has_seed"],
    "initial_examples": {}
}

# --- Domain: Biomedical (small fragment) ---
BIO_DOMAIN = {
    "name": "Biomedical Entities",
    "context": "types of biomedical entities in a clinical ontology",
    "attributes": [
        "is_disease", "is_symptom", "is_treatment", 
        "is_chronic", "is_infectious", "requires_medication",
        "is_preventable", "has_biomarker"
    ],
    "initial_examples": {
        "diabetes_type2": {"is_disease", "is_chronic", "requires_medication", 
                           "is_preventable", "has_biomarker"},
    }
}

# Select active domain
ACTIVE_DOMAIN = SE_DOMAIN
```

### 4. `run.py` — Main Entry Point (≈100 lines)

```python
"""
AutoOntology Runner
===================
Main entry point. Like kicking off an autoresearch experiment.
Usage: python run.py [--domain fruits|se|bio] [--model qwen2.5:7b]
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

from fca_engine import (
    FormalContext, ExplorationState, full_exploration, 
    completeness_score
)
from oracle import OllamaOracle
from domain import FRUIT_DOMAIN, SE_DOMAIN, BIO_DOMAIN

DOMAINS = {"fruits": FRUIT_DOMAIN, "se": SE_DOMAIN, "bio": BIO_DOMAIN}

def main():
    parser = argparse.ArgumentParser(description="AutoOntology Explorer")
    parser.add_argument("--domain", default="fruits", choices=DOMAINS.keys())
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    
    domain = DOMAINS[args.domain]
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    print(f"{'='*60}")
    print(f"AutoOntology Explorer")
    print(f"Domain: {domain['name']}")
    print(f"Attributes: {len(domain['attributes'])}")
    print(f"Model: {args.model}")
    print(f"{'='*60}\n")
    
    # Initialize context with seed examples
    context = FormalContext(
        objects=list(domain["initial_examples"].keys()),
        attributes=domain["attributes"],
        incidence={k: set(v) for k, v in domain["initial_examples"].items()}
    )
    
    state = ExplorationState(context=context)
    oracle = OllamaOracle(model=args.model, domain_context=domain["context"])
    
    # Exploration log
    log_path = output_dir / "exploration_log.jsonl"
    log_file = open(log_path, "w")
    
    start_time = time.time()
    
    def on_step(step, result, response):
        elapsed = time.time() - start_time
        status = "CONFIRMED" if response.confirmed else "REJECTED"
        contradiction = " [CONTRADICTION!]" if response.retries > 0 else ""
        
        print(f"  Step {step:3d} | {status:9s} | {result.implication}{contradiction}")
        
        log_entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "elapsed_sec": round(elapsed, 1),
            "question": result.question,
            "implication": str(result.implication),
            "confirmed": response.confirmed,
            "counterexample": response.counterexample,
            "retries": response.retries,
            "raw_response": response.raw_response[:500]
        }
        log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        log_file.flush()
    
    # Run exploration
    print("Starting exploration...\n")
    state = full_exploration(state, oracle.query, args.max_steps, on_step)
    
    elapsed = time.time() - start_time
    log_file.close()
    
    # Evaluation
    scores = completeness_score(state)
    
    # Save results
    results = {
        "domain": domain["name"],
        "model": args.model,
        "elapsed_sec": round(elapsed, 1),
        "complete": state.complete,
        "canonical_basis": [str(impl) for impl in state.confirmed],
        "num_examples": len(state.context.objects),
        "examples": {o: sorted(state.context.incidence[o]) 
                     for o in state.context.objects},
        "scores": scores,
        "contradictions": state.contradictions,
        "oracle_history_size": len(oracle.history)
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    if state.contradictions:
        with open(output_dir / "contradictions.jsonl", "w") as f:
            for c in state.contradictions:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"EXPLORATION {'COMPLETE' if state.complete else 'STOPPED (max steps)'}")
    print(f"{'='*60}")
    print(f"  Steps:          {state.step}")
    print(f"  Time:           {elapsed:.1f}s")
    print(f"  Rules found:    {len(state.confirmed)}")
    print(f"  Examples:       {len(state.context.objects)}")
    print(f"  Contradictions: {len(state.contradictions)}")
    print(f"  Completeness:   {scores['completeness']:.1%}")
    print(f"\nCanonical basis:")
    for impl in state.confirmed:
        print(f"    {impl}")
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
```

### 5. `evaluate.py` — Metrics (≈80 lines)

```python
"""
Evaluation & Comparison
=======================
Compare FCA-guided exploration vs baselines.
"""

import json
from pathlib import Path
from fca_engine import Implication

def load_gold_standard(path: str) -> list[Implication]:
    """Load manually curated gold standard implications."""
    with open(path) as f:
        data = json.load(f)
    return [
        Implication(frozenset(item["premise"]), frozenset(item["conclusion"]))
        for item in data
    ]

def axiom_recovery_metrics(discovered: list[Implication], 
                           gold: list[Implication]) -> dict:
    """Precision, Recall, F1 of axiom recovery vs gold standard."""
    disc_set = {(impl.premise, impl.conclusion) for impl in discovered}
    gold_set = {(impl.premise, impl.conclusion) for impl in gold}
    
    tp = len(disc_set & gold_set)
    precision = tp / len(disc_set) if disc_set else 0
    recall = tp / len(gold_set) if gold_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1": f1, 
            "true_positives": tp, "discovered": len(disc_set), "gold": len(gold_set)}

def contradiction_rate(log_path: str) -> dict:
    """Analyze contradiction frequency from exploration log."""
    steps = []
    with open(log_path) as f:
        for line in f:
            steps.append(json.loads(line))
    
    total = len(steps)
    contradictions = sum(1 for s in steps if s.get("retries", 0) > 0)
    confirmed = sum(1 for s in steps if s.get("confirmed"))
    rejected = sum(1 for s in steps if not s.get("confirmed"))
    
    return {
        "total_steps": total,
        "contradictions": contradictions,
        "contradiction_rate": contradictions / total if total > 0 else 0,
        "confirmed_rules": confirmed,
        "rejected_rules": rejected,
    }

def compare_with_baselines(fca_results: str, baseline_dir: str) -> dict:
    """Compare FCA exploration results with baseline methods."""
    with open(fca_results) as f:
        fca = json.load(f)
    
    comparison = {"fca": fca["scores"]}
    
    for baseline_file in Path(baseline_dir).glob("*.json"):
        with open(baseline_file) as f:
            bl = json.load(f)
        comparison[baseline_file.stem] = bl.get("scores", {})
    
    return comparison
```

### 6. `program.md` — Research Direction (Human Writes)

```markdown
# AutoOntology Research Program

## Goal
Discover the complete implication theory of the active domain using 
an SLM as domain expert oracle, with formal completeness guarantees.

## Current experiment
- Domain: Systems Engineering (BFO-based, 10 attributes)
- Model: qwen2.5:7b via Ollama
- Compare with: vanilla prompting, CoT, self-consistency (5-shot)

## What to measure
1. How many contradictions does the SLM produce during exploration?
2. Does the self-correction loop resolve them?
3. How does the final canonical basis compare to gold standard?
4. How many questions were needed to reach completeness?

## What to try next
- Increase domain to 15, then 20 attributes
- Switch to EXAONE-3.5-7.8B, compare contradiction rates
- Add background knowledge to reduce question count
- Try multi-oracle (two SLMs, majority vote on disagreements)
```

---

## Data Flow

```
Human writes                Agent (SLM) answers          FCA engine evaluates
─────────────              ─────────────────────         ──────────────────────
program.md ──────┐
                 │
domain.py ───────┼──> run.py ──> fca_engine.py ──question──> oracle.py ──> Ollama
                 │         ↑                                      │
                 │         │         ┌──────────────────┐         │
                 │         └─────────│ consistency check │<────────┘
                 │                   │ (hallucination    │
                 │                   │  detector)        │
                 │                   └────────┬─────────┘
                 │                            │
                 │                   ┌────────▼─────────┐
                 └──────────────────>│  results/         │
                                     │  - log.jsonl      │
                                     │  - contradictions │
                                     │  - basis.json     │
                                     └──────────────────┘
```

---

## Quick Start (Mac Mini M4)

```bash
# 1. Ensure Ollama is running with a model
ollama pull qwen2.5:7b

# 2. Install dependencies
pip install requests

# 3. Run toy example first (validation)
python run.py --domain fruits --model qwen2.5:7b

# 4. Run SE ontology exploration
python run.py --domain se --model qwen2.5:7b

# 5. Compare with baselines
python baselines/baseline_vanilla.py --domain se
python baselines/baseline_cot.py --domain se
python evaluate.py --compare results/ baselines/
```

---

## Implementation Priority

| Priority | File | Effort | Status |
|----------|------|--------|--------|
| 1 | fca_engine.py (core algorithm) | 2-3 days | Start here |
| 2 | oracle.py (Ollama connection) | 1 day | |
| 3 | domain.py (fruit toy example) | 0.5 day | |
| 4 | run.py (main loop) | 1 day | |
| 5 | test_fca.py (verify correctness) | 1 day | Critical |
| 6 | evaluate.py + baselines | 2 days | |
| 7 | domain.py (SE + bio domains) | 1-2 days | |
