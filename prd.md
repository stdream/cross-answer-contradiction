# AutoOntology — Product Requirements Document v2

> Updated: 2026-03-22 | Experiment-driven redesign

## Paper Pitch (One Sentence)
> "We reduce SLM cross-answer contradiction rates by X% using formal concept analysis as a structured guardrail, with provable completeness guarantees that no existing method provides."

## Problem
1. SLM은 질문마다 모순된 답변을 한다 (cross-answer inconsistency).
2. 기존 CoT, self-consistency는 단일 질문 내 일관성만 개선, 질문 간 모순은 못 잡음.
3. 어떤 기존 방법도 "모든 모순을 해소했다"는 수학적 보장을 제공하지 않음.

## Solution
FCA 속성탐색이 SLM에게 구조화된 yes/no 질문을 던지고, 매 응답을 기존 지식과 교차 검증하여 모순 감지/교정. 탐색 종료 시 완전성 수학적 보장.

---

## Experiment Design (논문 핵심 — 6개 실험)

### Exp 1: Main Result — FCA vs baselines
Methods: Vanilla, CoT, Self-consistency(k=5,10), **FCA-guided(ours)**
Metrics: cross-answer contradiction rate(↓), knowledge accuracy P/R/F1(↑), query count

### Exp 2: Across Models — 약한 모델일수록 효과 큰가?
Models: Qwen2.5-1.5B, Qwen2.5-7B, EXAONE-3.5-7.8B, Qwen2.5-14B
Expected: 작은 모델일수록 FCA 개선폭 큼

### Exp 3: Across Domains — factual, commonsense, specialized
D1: Countries(15 attrs, Wikidata gold), D2: Animals(15 attrs, curated), D3: SE(12 attrs, expert)
Expected: D3(specialized)에서 hallucination 가장 높고 FCA 개선폭도 최대

### Exp 4: Ablation — 컴포넌트별 기여
Full → −self-correction → −consistency check → −structured Q (=Vanilla)
Expected: consistency check의 기여가 가장 큼

### Exp 5: Scaling — 속성 10→15→20→30
Expected: 20개까지 실용적, 한계 정직히 보여주고 future work 제시

### Exp 6 (Optional): DPO fine-tuning with contradiction traces
Base vs Random-DPO vs FCA-DPO. Cloud A100 필요.

---

## Domains (역할: hallucination 측정 벤치마크)

### D0: Fruit (internal test only, 논문에 안 나감)
attrs: {red, sweet, has_seed} | 역할: FCA 엔진 unit test

### D1: Countries (main factual benchmark)
attrs(15): is_in_europe, is_in_asia, has_coastline, is_island_nation, is_UN_member, is_NATO_member, is_EU_member, is_G7, has_nuclear_weapons, population_over_50M, population_over_100M, gdp_top_20, is_democracy, is_monarchy, has_official_english
objects: ~50 countries | gold: Wikidata SPARQL

### D2: Animals (commonsense benchmark)
attrs(15): is_mammal, is_bird, is_reptile, is_fish, can_fly, lives_in_water, is_domestic, is_carnivore, is_herbivore, has_fur, lays_eggs, is_nocturnal, is_endangered, lives_in_groups, larger_than_human
objects: ~40 animals | edge cases: platypus, penguin, bat

### D3: SE Concepts (domain-specific benchmark)
attrs(12): is_continuant, is_occurrent, is_independent, is_dependent, has_function, has_requirement, is_physical, is_informational, is_verifiable, is_decomposable, has_interface, is_reusable
objects: system, component, function, requirement, ... | gold: BFO expert

---

## Architecture (3-file + experiment modules)

| File | Role | Modified by |
|---|---|---|
| `fca_engine.py` | Core algorithm (FIXED) | Nobody |
| `domain.py` | Domain definitions | Human |
| `program.md` | Research direction | Human |
| `oracle.py` | SLM interface + self-correction | Human (initial) |
| `run.py` | CLI + batch experiment runner | Human (initial) |
| `evaluate.py` | All metrics & analysis | Human (initial) |
| `baselines/` | 4 baseline implementations | Human (initial) |
| `gold_standards/` | Ground truth JSONs | Human (curated) |
| `experiments/` | Experiment config JSONs | Human (per exp) |

## Infrastructure
- Mac Mini M4 64GB: Exp 1-5 전부 ($0)
- Ollama: Qwen-1.5B/7B/14B, EXAONE-7.8B ($0)
- Cloud A100: Exp 6 only (~$100-200)

## Success Criteria
1. Exp 1: FCA가 모든 baseline 대비 contradiction rate 최저 & accuracy 최고
2. Exp 2: 최소 3개 모델에서 일관된 개선
3. Exp 3: 3개 도메인 모두 baseline 대비 개선
4. Exp 4: 각 ablation에서 성능 하락 관찰
5. Exp 5: 20 attributes까지 practical time

## File Structure
```
autoontology/
├── prd.md, CLAUDE.md, program.md, progress.md, implementation-plan.md
├── fca_engine.py, oracle.py, domain.py, run.py, evaluate.py
├── baselines/ (vanilla, cot, selfconsistency)
├── gold_standards/ (countries.json, animals.json, se_concepts.json, build_countries.py)
├── experiments/ (exp1_main.json ~ exp5_scaling.json)
├── results/, tests/, dpo/, docs/
└── pyproject.toml
```
