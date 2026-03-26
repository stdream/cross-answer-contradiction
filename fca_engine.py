"""
FCA Attribute Exploration Engine — Algorithm 19 (Ganter & Obiedkov 2016 Ch.4)
=============================================================================
Mathematical algorithm. Do not modify after implementation + tests pass.
Foundation for all experimental comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import logging

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Implication:
    """FCA implication A -> B (premise ⊆ conclusion)."""
    premise: frozenset[str]
    conclusion: frozenset[str]

    def __post_init__(self):
        if not self.premise <= self.conclusion:
            raise ValueError(
                f"Premise must be ⊆ conclusion: {self.premise} ⊄ {self.conclusion}"
            )

    @property
    def added(self) -> frozenset[str]:
        """conclusion \\ premise."""
        return self.conclusion - self.premise

    def holds_for(self, attrs: set[str] | frozenset[str]) -> bool:
        """Check whether this implication holds for the given attribute set."""
        return not (self.premise <= attrs) or (self.conclusion <= attrs)

    def __repr__(self) -> str:
        p = ", ".join(sorted(self.premise)) or "∅"
        a = ", ".join(sorted(self.added))
        return f"{{{p}}} → {{{a}}}"


@dataclass
class ConsistencyViolation:
    """Contradiction detection: object violates a confirmed implication."""
    object_name: str
    object_attrs: frozenset[str]
    violated_implication: Implication
    missing_attrs: frozenset[str]


@dataclass
class ExplorationResult:
    """Attribute exploration result."""
    implications: list[Implication]
    context: FormalContext
    exploration_log: list[dict]
    num_questions: int
    num_counterexamples: int

    @property
    def num_implications(self) -> int:
        return len(self.implications)


# ── Formal Context ───────────────────────────────────────────────────────────

class FormalContext:
    """Formal context (G, M, I) -- object x attribute cross-table."""

    def __init__(
        self,
        attributes: list[str],
        objects: dict[str, set[str]] | None = None,
    ):
        self.attributes: list[str] = list(attributes)
        self.objects: dict[str, set[str]] = {}
        if objects:
            for name, attrs in objects.items():
                self.objects[name] = set(attrs)

    def add_object(self, name: str, attrs: set[str]) -> None:
        self.objects[name] = set(attrs)

    def extent(self, attrs: set[str] | frozenset[str]) -> frozenset[str]:
        """A' -- set of objects that have all attrs."""
        if not attrs:
            return frozenset(self.objects.keys())
        return frozenset(
            obj
            for obj, obj_attrs in self.objects.items()
            if set(attrs) <= obj_attrs
        )

    def intent(self, objs: set[str] | frozenset[str]) -> frozenset[str]:
        """B' -- set of attributes shared by the objects."""
        if not objs:
            return frozenset(self.attributes)
        result: set[str] | None = None
        for obj in objs:
            if obj in self.objects:
                if result is None:
                    result = set(self.objects[obj])
                else:
                    result &= self.objects[obj]
        return frozenset(result) if result is not None else frozenset(self.attributes)

    def double_prime(self, attrs: set[str] | frozenset[str]) -> frozenset[str]:
        """A'' = (A')' -- attribute closure within the context."""
        return self.intent(self.extent(attrs))

    def __repr__(self) -> str:
        return f"FormalContext({len(self.objects)} objs × {len(self.attributes)} attrs)"


# ── Core Algorithms ──────────────────────────────────────────────────────────

def closure_under_implications(
    attrs: set[str] | frozenset[str],
    implications: list[Implication],
) -> frozenset[str]:
    """L(A) -- closure of A under implication set L. Iterate until fixpoint."""
    closed = set(attrs)
    changed = True
    while changed:
        changed = False
        for impl in implications:
            if impl.premise <= closed and not impl.conclusion <= closed:
                closed |= impl.conclusion
                changed = True
    return frozenset(closed)


def next_closure(
    current: frozenset[str],
    attributes: list[str],
    implications: list[Implication],
) -> frozenset[str] | None:
    """Next L-closed set after current in lecticographic order.

    Returns None when current is the last closed set.
    """
    n = len(attributes)
    attr_idx = {a: i for i, a in enumerate(attributes)}

    for i in range(n - 1, -1, -1):
        m_i = attributes[i]
        if m_i in current:
            continue

        # oplus(current, m_i) = (current ∩ {a : index(a) < i}) ∪ {m_i}
        prefix = frozenset(a for a in current if attr_idx[a] < i)
        closed = closure_under_implications(prefix | {m_i}, implications)

        # Check that no new elements below index i were added
        if all(
            attributes[j] not in closed or attributes[j] in current
            for j in range(i)
        ):
            return closed

    return None


# ── Oracle Protocol ──────────────────────────────────────────────────────────

@runtime_checkable
class Oracle(Protocol):
    """Exploration oracle interface."""

    def confirm_implication(
        self,
        premise: frozenset[str],
        conclusion: frozenset[str],
        context: FormalContext,
    ) -> tuple[bool, str | None, set[str] | None]:
        """Confirm implication. Returns (accepted, counterexample_name, counterexample_attrs)."""
        ...


# ── Attribute Exploration — Algorithm 19 ─────────────────────────────────────

def full_exploration(
    attributes: list[str],
    oracle: Oracle,
    initial_objects: dict[str, set[str]] | None = None,
    max_iterations: int = 10_000,
) -> ExplorationResult:
    """Run Algorithm 19 (Ganter & Obiedkov 2016 Ch.4).

    Args:
        attributes: attribute list M (fixed order)
        oracle: implication confirmation oracle
        initial_objects: initial objects (optional)
        max_iterations: safety limit
    """
    context = FormalContext(attributes, initial_objects)
    implications: list[Implication] = []
    log: list[dict] = []
    num_questions = 0
    num_counterexamples = 0

    A = closure_under_implications(frozenset(), implications)
    iteration = 0

    while A is not None and iteration < max_iterations:
        iteration += 1

        # Check whether A is an intent, otherwise propose an implication
        while True:
            A_pp = context.double_prime(A)

            if A == A_pp:
                log.append({"type": "intent", "set": sorted(A)})
                break

            # Implication candidate: A -> A''
            num_questions += 1
            accepted, ce_name, ce_attrs = oracle.confirm_implication(
                A, A_pp, context
            )

            if accepted:
                impl = Implication(premise=A, conclusion=A_pp)
                implications.append(impl)
                log.append({
                    "type": "implication",
                    "premise": sorted(A),
                    "conclusion": sorted(A_pp),
                })
                logger.info("Accepted: %s", impl)
                break

            assert ce_name is not None and ce_attrs is not None, \
                "Oracle rejected but provided no counterexample"
            context.add_object(ce_name, ce_attrs)
            num_counterexamples += 1
            log.append({
                "type": "counterexample",
                "object": ce_name,
                "attrs": sorted(ce_attrs),
            })
            logger.info("Counterexample: %s %s", ce_name, sorted(ce_attrs))

        A = next_closure(A, attributes, implications)

    return ExplorationResult(
        implications=implications,
        context=context,
        exploration_log=log,
        num_questions=num_questions,
        num_counterexamples=num_counterexamples,
    )


# ── Consistency Checking (Hallucination Detector) ────────────────────────────

def check_consistency(
    object_name: str,
    object_attrs: set[str] | frozenset[str],
    implications: list[Implication],
) -> list[ConsistencyViolation]:
    """Check whether object attributes are consistent with confirmed implications.

    Violation = hallucination: SLM said premise holds but conclusion does not.
    """
    attrs_fs = frozenset(object_attrs)
    return [
        ConsistencyViolation(
            object_name=object_name,
            object_attrs=attrs_fs,
            violated_implication=impl,
            missing_attrs=impl.conclusion - attrs_fs,
        )
        for impl in implications
        if impl.premise <= attrs_fs and not impl.conclusion <= attrs_fs
    ]
