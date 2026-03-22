# AutoOntology Research Program v2

## Mission
FCA 속성탐색을 SLM의 cross-answer contradiction을 감지/교정하는 symbolic guardrail로 활용하여,
기존 방법(CoT, self-consistency)이 제공하지 못하는 수학적 완전성 보장을 달성한다.

## Core Claim
> Self-consistency catches within-question noise.
> FCA catches between-question contradictions.
> Only FCA guarantees: "when I stop, ALL contradictions are resolved."

## Experiment Roadmap
1. **Main (Exp 1)**: FCA vs 4 baselines on Countries domain → contradiction rate & accuracy
2. **Models (Exp 2)**: 1.5B → 7B → 7.8B → 14B → FCA helps weak models more
3. **Domains (Exp 3)**: factual → commonsense → specialized → harder = more hallucination = more FCA value
4. **Ablation (Exp 4)**: Full → −correction → −check → −structured = each component contributes
5. **Scaling (Exp 5)**: 10 → 30 attrs → honest about limits
6. **DPO (Exp 6)**: contradiction traces as training signal (cloud, optional)

## What to Measure
- Cross-answer contradiction rate (THE key metric)
- Knowledge accuracy P/R/F1 vs gold standard
- Self-correction success rate (what % of contradictions get resolved?)
- Query efficiency (don't want 10x more questions than baselines)

## Key Insight
Autoresearch metric (val_bpb): greedy, no completeness guarantee.
AutoOntology metric (L(A) = A^JJ): provably complete.
"Any metric you care about can be autoresearched" — ours comes with a proof.
