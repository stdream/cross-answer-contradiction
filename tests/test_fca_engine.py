"""Tests for fca_engine.py -- Algorithm 19 verification."""
import pytest
from fca_engine import (
    Implication,
    FormalContext,
    ExplorationResult,
    ConsistencyViolation,
    closure_under_implications,
    next_closure,
    full_exploration,
    check_consistency,
)

# ── Fruit domain gold standard ───────────────────────────────────────────────

FRUIT_ATTRS = ["red", "sweet", "has_seed"]

FRUIT_GOLD = {
    "apple":      {"red", "sweet", "has_seed"},
    "tomato":     {"red", "has_seed"},
    "banana":     {"sweet"},
    "watermelon": {"sweet", "has_seed"},
    "lemon":      {"has_seed"},
    "strawberry": {"red", "sweet", "has_seed"},
}


class MockOracle:
    """Gold standard based mock oracle -- provides deterministic counterexamples (sorted by name)."""

    def __init__(self, gold: dict[str, set[str]]):
        self.gold = gold
        self.queries: list[tuple[frozenset[str], frozenset[str]]] = []

    def confirm_implication(
        self,
        premise: frozenset[str],
        conclusion: frozenset[str],
        context: FormalContext,
    ) -> tuple[bool, str | None, set[str] | None]:
        self.queries.append((premise, conclusion))
        for name in sorted(self.gold):
            if name not in context.objects:
                attrs = self.gold[name]
                if premise <= attrs and not conclusion <= attrs:
                    return (False, name, attrs)
        return (True, None, None)


# ── Implication ──────────────────────────────────────────────────────────────

class TestImplication:
    def test_creation(self):
        impl = Implication(frozenset({"red"}), frozenset({"red", "has_seed"}))
        assert impl.added == frozenset({"has_seed"})

    def test_holds_for_satisfied(self):
        impl = Implication(frozenset({"red"}), frozenset({"red", "has_seed"}))
        assert impl.holds_for({"red", "sweet", "has_seed"})

    def test_holds_for_vacuously(self):
        impl = Implication(frozenset({"red"}), frozenset({"red", "has_seed"}))
        assert impl.holds_for({"sweet"})  # premise not satisfied -> true

    def test_holds_for_violated(self):
        impl = Implication(frozenset({"red"}), frozenset({"red", "has_seed"}))
        assert not impl.holds_for({"red", "sweet"})  # has_seed missing

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            Implication(frozenset({"a", "b"}), frozenset({"a"}))

    def test_repr(self):
        impl = Implication(frozenset({"red"}), frozenset({"red", "has_seed"}))
        assert "red" in repr(impl)
        assert "has_seed" in repr(impl)

    def test_empty_premise(self):
        impl = Implication(frozenset(), frozenset({"a"}))
        assert impl.premise == frozenset()
        assert impl.added == frozenset({"a"})
        # ∅ ⊆ anything → must check conclusion
        assert impl.holds_for({"a", "b"})
        assert not impl.holds_for({"b"})


# ── FormalContext ─────────────────────────────────────────────────────────────

class TestFormalContext:
    def test_extent_empty_attrs(self):
        ctx = FormalContext(FRUIT_ATTRS, FRUIT_GOLD)
        assert ctx.extent(set()) == frozenset(FRUIT_GOLD.keys())

    def test_extent_red(self):
        ctx = FormalContext(FRUIT_ATTRS, FRUIT_GOLD)
        assert ctx.extent({"red"}) == frozenset({"apple", "tomato", "strawberry"})

    def test_extent_no_match(self):
        ctx = FormalContext(["a", "b"], {"x": {"a"}})
        assert ctx.extent({"a", "b"}) == frozenset()

    def test_intent(self):
        ctx = FormalContext(FRUIT_ATTRS, FRUIT_GOLD)
        assert ctx.intent({"apple", "tomato", "strawberry"}) == frozenset({"red", "has_seed"})

    def test_intent_empty_objs(self):
        ctx = FormalContext(FRUIT_ATTRS, FRUIT_GOLD)
        assert ctx.intent(set()) == frozenset(FRUIT_ATTRS)

    def test_double_prime_red(self):
        ctx = FormalContext(FRUIT_ATTRS, FRUIT_GOLD)
        assert ctx.double_prime({"red"}) == frozenset({"red", "has_seed"})

    def test_double_prime_sweet(self):
        ctx = FormalContext(FRUIT_ATTRS, FRUIT_GOLD)
        assert ctx.double_prime({"sweet"}) == frozenset({"sweet"})

    def test_double_prime_empty_full_context(self):
        ctx = FormalContext(FRUIT_ATTRS, FRUIT_GOLD)
        # No common attributes among the 6 fruits
        assert ctx.double_prime(set()) == frozenset()

    def test_double_prime_empty_empty_context(self):
        ctx = FormalContext(FRUIT_ATTRS)
        # No objects -> vacuously M
        assert ctx.double_prime(set()) == frozenset(FRUIT_ATTRS)

    def test_add_object(self):
        ctx = FormalContext(FRUIT_ATTRS)
        ctx.add_object("apple", {"red", "sweet", "has_seed"})
        assert "apple" in ctx.objects
        assert ctx.extent({"red"}) == frozenset({"apple"})


# ── closure_under_implications ───────────────────────────────────────────────

class TestClosureUnderImplications:
    def test_empty_implications(self):
        assert closure_under_implications({"red"}, []) == frozenset({"red"})

    def test_single_implication(self):
        impl = Implication(frozenset({"red"}), frozenset({"red", "has_seed"}))
        assert closure_under_implications({"red"}, [impl]) == frozenset({"red", "has_seed"})

    def test_chained(self):
        i1 = Implication(frozenset({"a"}), frozenset({"a", "b"}))
        i2 = Implication(frozenset({"b"}), frozenset({"b", "c"}))
        assert closure_under_implications({"a"}, [i1, i2]) == frozenset({"a", "b", "c"})

    def test_no_match(self):
        impl = Implication(frozenset({"a"}), frozenset({"a", "b"}))
        assert closure_under_implications({"c"}, [impl]) == frozenset({"c"})

    def test_empty_set(self):
        impl = Implication(frozenset({"a"}), frozenset({"a", "b"}))
        assert closure_under_implications(set(), [impl]) == frozenset()

    def test_already_closed(self):
        impl = Implication(frozenset({"a"}), frozenset({"a", "b"}))
        assert closure_under_implications({"a", "b"}, [impl]) == frozenset({"a", "b"})


# ── next_closure ─────────────────────────────────────────────────────────────

class TestNextClosure:
    def test_full_enumeration_no_implications(self):
        """M = {a, b, c}, L = {} -> enumerate all 2^3 subsets in lecticographic order."""
        attrs = ["a", "b", "c"]
        seq = [frozenset()]
        current = frozenset()
        while True:
            nxt = next_closure(current, attrs, [])
            if nxt is None:
                break
            seq.append(nxt)
            current = nxt

        expected = [
            frozenset(),
            frozenset({"c"}),
            frozenset({"b"}),
            frozenset({"b", "c"}),
            frozenset({"a"}),
            frozenset({"a", "c"}),
            frozenset({"a", "b"}),
            frozenset({"a", "b", "c"}),
        ]
        assert seq == expected

    def test_with_implications(self):
        """L = {{a} -> {a,c}} -> {a} and {a,b} are not L-closed."""
        attrs = ["a", "b", "c"]
        impl = Implication(frozenset({"a"}), frozenset({"a", "c"}))

        seq = [closure_under_implications(frozenset(), [impl])]
        current = seq[0]
        while True:
            nxt = next_closure(current, attrs, [impl])
            if nxt is None:
                break
            seq.append(nxt)
            current = nxt

        expected = [
            frozenset(),
            frozenset({"c"}),
            frozenset({"b"}),
            frozenset({"b", "c"}),
            frozenset({"a", "c"}),
            frozenset({"a", "b", "c"}),
        ]
        assert seq == expected

    def test_from_M_returns_none(self):
        attrs = ["a", "b"]
        assert next_closure(frozenset({"a", "b"}), attrs, []) is None

    def test_single_attribute(self):
        attrs = ["x"]
        assert next_closure(frozenset(), attrs, []) == frozenset({"x"})
        assert next_closure(frozenset({"x"}), attrs, []) is None


# ── full_exploration (fruit E2E) ─────────────────────────────────────────────

class TestFullExploration:
    def test_fruit_canonical_basis(self):
        """Fruit domain canonical basis: only {red} -> {has_seed}."""
        oracle = MockOracle(FRUIT_GOLD)
        result = full_exploration(FRUIT_ATTRS, oracle)

        assert result.num_implications == 1
        impl = result.implications[0]
        assert impl.premise == frozenset({"red"})
        assert impl.conclusion == frozenset({"red", "has_seed"})

    def test_fruit_soundness(self):
        """All gold objects satisfy the explored implications."""
        oracle = MockOracle(FRUIT_GOLD)
        result = full_exploration(FRUIT_ATTRS, oracle)

        for name, attrs in FRUIT_GOLD.items():
            for impl in result.implications:
                assert impl.holds_for(attrs), f"{name} violates {impl}"

    def test_fruit_exploration_stats(self):
        """Exploration stats: 4 counterexamples, 5 oracle questions."""
        oracle = MockOracle(FRUIT_GOLD)
        result = full_exploration(FRUIT_ATTRS, oracle)

        assert result.num_counterexamples == 4
        assert result.num_questions == 5

    def test_fruit_context_objects(self):
        """Objects collected as counterexamples: banana, lemon, tomato, watermelon."""
        oracle = MockOracle(FRUIT_GOLD)
        result = full_exploration(FRUIT_ATTRS, oracle)

        assert len(result.context.objects) == 4
        assert set(result.context.objects.keys()) == {
            "banana", "lemon", "tomato", "watermelon",
        }

    def test_fruit_log_structure(self):
        oracle = MockOracle(FRUIT_GOLD)
        result = full_exploration(FRUIT_ATTRS, oracle)

        types = {e["type"] for e in result.exploration_log}
        assert "intent" in types
        assert "counterexample" in types
        assert "implication" in types

    def test_with_initial_objects(self):
        """Same canonical basis even with initial objects."""
        oracle = MockOracle(FRUIT_GOLD)
        initial = {"apple": {"red", "sweet", "has_seed"}}
        result = full_exploration(FRUIT_ATTRS, oracle, initial_objects=initial)

        assert result.num_implications == 1
        assert result.implications[0].premise == frozenset({"red"})
        assert result.implications[0].conclusion == frozenset({"red", "has_seed"})

    def test_trivial_domain(self):
        """All objects have identical attributes -> many implications."""
        gold = {"x": {"a", "b"}, "y": {"a", "b"}}
        oracle = MockOracle(gold)
        result = full_exploration(["a", "b"], oracle)

        # All objects have {a,b} -> empty -> {a,b} implication
        assert any(
            impl.premise == frozenset() for impl in result.implications
        )


# ── check_consistency ────────────────────────────────────────────────────────

class TestCheckConsistency:
    def test_no_violation(self):
        impl = Implication(frozenset({"red"}), frozenset({"red", "has_seed"}))
        violations = check_consistency("apple", {"red", "sweet", "has_seed"}, [impl])
        assert violations == []

    def test_violation_detected(self):
        impl = Implication(frozenset({"red"}), frozenset({"red", "has_seed"}))
        violations = check_consistency("bad", {"red", "sweet"}, [impl])

        assert len(violations) == 1
        assert violations[0].missing_attrs == frozenset({"has_seed"})
        assert violations[0].object_name == "bad"

    def test_vacuously_true(self):
        impl = Implication(frozenset({"red"}), frozenset({"red", "has_seed"}))
        violations = check_consistency("banana", {"sweet"}, [impl])
        assert violations == []

    def test_multiple_violations(self):
        i1 = Implication(frozenset({"a"}), frozenset({"a", "b"}))
        i2 = Implication(frozenset({"a"}), frozenset({"a", "c"}))
        violations = check_consistency("x", {"a"}, [i1, i2])
        assert len(violations) == 2

    def test_empty_implications(self):
        violations = check_consistency("any", {"red", "sweet"}, [])
        assert violations == []


# ── End-to-end ───────────────────────────────────────────────────────────────

class TestEndToEnd:
    def test_exploration_then_consistency(self):
        """After exploration, all gold objects are consistent with implications."""
        oracle = MockOracle(FRUIT_GOLD)
        result = full_exploration(FRUIT_ATTRS, oracle)

        for name, attrs in FRUIT_GOLD.items():
            violations = check_consistency(name, attrs, result.implications)
            assert violations == [], f"{name}: {violations}"

    def test_hallucination_detection(self):
        """Detect hallucination (contradictory answer) after exploration."""
        oracle = MockOracle(FRUIT_GOLD)
        result = full_exploration(FRUIT_ATTRS, oracle)

        # "cherry is red but not has_seed" = hallucination
        violations = check_consistency(
            "cherry_hallucinated", {"red", "sweet"}, result.implications
        )
        assert len(violations) == 1
        assert "has_seed" in violations[0].missing_attrs

    def test_consistent_new_object(self):
        """Consistent new object after exploration has no violations."""
        oracle = MockOracle(FRUIT_GOLD)
        result = full_exploration(FRUIT_ATTRS, oracle)

        # "cherry: red + has_seed" = consistent with implications
        violations = check_consistency(
            "cherry", {"red", "has_seed"}, result.implications
        )
        assert violations == []
