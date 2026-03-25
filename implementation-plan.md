# Implementation Plan v2

> Updated: 2026-03-22 | Experiment-driven
> Current: Phase 4 (experiments ready to run)

---

## Phase 1: FCA Engine + Tests ✅
**Goal**: 수학적으로 정확한 FCA 엔진
**시간**: 2-3일 | **의존성**: 없음

### Tasks
1. `fca_engine.py` 핵심 구조 ⬜
   - FormalContext, Implication, ExplorationState
   - closure_under_implications(), next_closure()
   - full_exploration() (Algorithm 19)
   - check_consistency() (hallucination detector)
   - completeness_score()

2. `tests/test_fca_engine.py` ⬜
   - test_closure, test_next_closure, test_full_exploration_fruit
   - test_check_consistency (모순 감지 검증)

### 완료 기준
- [ ] pytest 전체 통과
- [ ] 과일 mock에서 알려진 정준 기저 일치
- [ ] check_consistency가 모순 반례 정확히 감지

### 과일 Gold Standard
```
M = {red, sweet, has_seed}
Domain: apple{r,s,d}, tomato{r,d}, banana{s}, watermelon{s,d}, lemon{d}, strawberry{r,s,d}
Expected basis: {red} → {red, has_seed}
(빨간 과일은 반드시 씨가 있다)
```

---

## Phase 2: Oracle + Domain + Runner ✅
**Goal**: Ollama 연결, 실제 SLM 탐색 실행
**시간**: 2일 | **의존성**: Phase 1

### Tasks
1. `oracle.py` ⬜ — Ollama 연결, JSON prompt, self-correction loop
2. `domain.py` ⬜ — D0(fruit), D1(countries), D2(animals), D3(SE)
3. `run.py` ⬜ — CLI, JSONL logging, results.json
4. Integration test ⬜ — `python run.py --domain fruits`

### 완료 기준
- [ ] fruit domain Ollama 탐색 완료
- [ ] exploration_log.jsonl 정상 기록

---

## Phase 3: Gold Standards + Baselines ✅
**Goal**: 비교 실험 인프라 구축
**시간**: 3일 | **의존성**: Phase 2

### Tasks
1. `gold_standards/` ⬜
   - build_countries.py: Wikidata SPARQL → countries.json (50 countries × 15 attrs)
   - animals.json: 수동 curate (40 animals × 15 attrs) 
   - se_concepts.json: BFO 기반 expert curate (20 concepts × 12 attrs)
   - 각 도메인의 gold standard canonical basis 계산 (fca_engine으로)

2. `baselines/` ⬜
   - baseline_vanilla.py: 모든 가능한 함의를 구조 없이 직접 질문
   - baseline_cot.py: CoT prompting 추가
   - baseline_selfconsistency.py: k-shot voting (k=5, k=10)
   - 공통 interface: 동일 domain, 동일 output format

3. Baseline metric compatibility ⬜
   - 모든 method의 output을 동일 형태로 변환
   - evaluate.py가 통합 비교 가능

### 완료 기준
- [ ] 3개 도메인 gold standard JSON 생성
- [ ] 3개 baseline 실행 + 결과 저장
- [ ] FCA vs baseline 비교 가능

---

## Phase 4: Evaluation + Full Experiments ⬜
**Goal**: Exp 1-5 실행, 논문용 결과 생성
**시간**: 3-4일 | **의존성**: Phase 3

### Tasks
1. `evaluate.py` 완성 ⬜
   - cross_answer_contradiction_rate()
   - knowledge_accuracy() (P/R/F1 vs gold)
   - ablation_analysis()
   - scaling_analysis()
   - model_comparison()
   - 결과 테이블 markdown/LaTeX 자동 생성

2. Experiment configs ⬜
   - exp1_main.json: 5 methods × 1 model × D1 domain
   - exp2_models.json: FCA × 4 models × D1 domain
   - exp3_domains.json: FCA × 1 model × 3 domains
   - exp4_ablation.json: 4 variants × 1 model × D1 domain
   - exp5_scaling.json: FCA × 1 model × D1(10,15,20,30 attrs)

3. 전체 실험 실행 ⬜
   - Exp 1-5 순차 실행
   - 결과 분석 + 시각화

### 완료 기준
- [ ] Exp 1-5 결과 테이블 생성
- [ ] 주요 finding 3개 이상 확인
- [ ] 논문 실험 섹션에 바로 넣을 수 있는 수준

---

## Phase 5: arXiv + Paper ⬜
**Goal**: arXiv preprint 업로드
**시간**: 1주 | **의존성**: Phase 4

### Tasks
1. arXiv preprint (Exp 1-5 결과)
2. (Optional) Exp 6: DPO on cloud A100

---

## Progress Tracking

| Phase | Status | Started | Completed |
|-------|--------|---------|-----------|
| Phase 1: FCA Engine | ✅ | 2026-03-22 | 2026-03-22 |
| Phase 2: Oracle+Runner | ✅ | 2026-03-22 | 2026-03-22 |
| Phase 3: Gold+Baselines | ✅ | 2026-03-22 | 2026-03-22 |
| Phase 4: Experiments | ⬜ | | |
| Phase 5: Paper | ⬜ | | |
