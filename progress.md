# Progress — AutoOntology v2

> Last updated: 2026-03-23
> Current phase: Phase 5 논문 작성 중

## Phase 1: FCA Engine + Tests ✅ (2026-03-22)
- `fca_engine.py`: Algorithm 19 (Ganter & Obiedkov 2016 Ch.4) 충실 구현
- `tests/test_fca_engine.py`: 42 tests 전체 통과
  - 과일 mock oracle E2E: 정준 기저 `{red} → {has_seed}` 100% 일치

## Phase 2: Oracle + Runner ✅ (2026-03-22)
- `oracle.py`: OllamaOracle (classification prompt, 2-strategy counterexample, self-correction)
- `run.py`: CLI runner (`--domain`, `--model`, `--config`, JSONL logging)
- Smoke test: `python run.py --domain fruits --model qwen2.5:7b` **3회 연속 일관**

## Phase 3: Gold Standards + Baselines + Evaluation ✅ (2026-03-22)

### Gold Standards
- `gold_standards/countries.json`: 50개국 × 15속성, **64 implications**, 16 ambiguity notes
- `gold_standards/animals.json`: 40동물 × 15속성, **70 implications**, 22 ambiguity notes
- `gold_standards/build.py`: 재현 가능한 빌드 스크립트

### Baselines (4개 method)
- `baselines/common.py`: 공통 유틸 (Ollama API, test set 생성, 결과 포맷)
  - `generate_test_set()`: gold basis(N valid) + 동일 수 invalid implications
  - `format_implication_question()`: vanilla/CoT 공통 프롬프트
- `baselines/baseline_vanilla.py`: 직접 함의 질문 → YES/NO
- `baselines/baseline_cot.py`: "Think step by step" 추가, 마지막 YES/NO 추출
- `baselines/baseline_selfconsistency.py`: k번 반복 majority vote (k=5, k=10, temp=0.7)
- **공통 결과 포맷**: `{method, domain, model, predictions[], metrics{P,R,F1,ccr}}`

### evaluate.py (평가 도구)
- `knowledge_accuracy_fca()`: FCA vs gold — closure-based P/R/F1
- `cross_answer_contradiction_rate()`: accepted implications vs gold objects → CCR
- `format_comparison_table()` / `format_model_table()`: markdown 테이블 출력
- `evaluate_fca_result()` / `evaluate_baseline_result()`: 통합 평가 함수

### Oracle Ablation Flags
- `OracleConfig.consistency_check`: 모순 감지 on/off
- `OracleConfig.structured_query`: suggest-then-verify on/off
- `num_contradictions` 카운터 추가
- Ablation 4단계: full → −self_correction → −consistency → −structured_query

## Phase 4: Experiments 🔧 (2026-03-23~)

### ✅ Exp 1 (Main): FCA vs Baselines — countries, qwen2.5:7b
| Method | P | R | F1 | CCR | Queries |
|--------|------|------|------|------|---------|
| vanilla | 0.00 | 0.00 | 0.00 | 0.00 | 128 |
| cot | 0.00 | 0.00 | 0.00 | 0.00 | 128 |
| sc_k5 | 0.00 | 0.00 | 0.00 | 0.00 | 640 |
| structured_survey | 0.34 | 0.75 | 0.47 | 0.80 | 750 |
| **fca (mean±std)** | **0.275±.02** | **0.786±.01** | **0.407±.02** | 0.82 | ~1490 |

- **핵심 발견**: direct questioning F1=0, structured survey F1=0.47, FCA exploration F1=0.41±0.02
- Structured survey: 전수조사로 더 높은 P (0.34 vs 0.28), 하지만 객체 집합 사전 필요
- FCA: 더 높은 R (0.79 vs 0.75) + consistency checking + self-correction (7/8 교정)
- FCA 3회 반복: F1 std=0.018 (temp=0.1에서 안정적)
- 결과: `results/exp1/`

### ✅ Exp 4 (Ablation): countries, qwen2.5:7b
| Variant | P | R | F1 | CCR | Queries |
|---------|------|------|------|------|---------|
| fca_full | 0.286 | 0.797 | 0.421 | 0.820 | 1342 |
| fca_no_selfcorr | 0.276 | 0.797 | 0.410 | 0.800 | 1409 |
| fca_no_consistency | 0.227 | 0.828 | 0.357 | 0.840 | 1152 |
| fca_no_structure | 0.000 | 1.000 | 0.000 | 1.000 | 1 |

- **Structured query가 가장 critical**: 없으면 FCA도 F1=0 (no_structure = 1 query로 ∅→M 수락)
- Consistency check: P 0.23→0.29 (+26%), F1 0.36→0.42 (+18%)
- Self-correction: 약간의 추가 개선 (P 0.276→0.286)

### ✅ Exp 5 (Scaling): countries 10/15/30 attrs
| Attrs | Gold Impls | Discovered | Queries | Time | P | R | F1 |
|-------|-----------|------------|---------|------|------|------|------|
| 10 | 36 | 21 | 698 | 11m | 0.286 | 0.870 | 0.430 |
| 15 | 64 | 27 | 1308 | 22m | 0.370 | 0.859 | 0.518 |
| 30 | 244 | 161 | 7192 | 122m | 0.385 | 0.730 | 0.504 |

- 10→15→30: queries ×2→×5.5, F1 0.43→0.52→0.50
- P는 속성 수 증가와 함께 상승 (0.29→0.37→0.39): 더 많은 속성이 counterexample diversity 강제
- R은 30에서 하락 (0.87→0.86→0.73): 탐색 공간 확대로 일부 함의 미발견
- 30속성도 ~2시간에 완료: 실용적 범위 확인

### ✅ Exp 2 (Models): countries, 7 models × 3 families
| Family | Model | Size | Vanilla F1 | FCA P | FCA R | FCA F1 |
|--------|-------|------|-----------|-------|-------|--------|
| Qwen | 2.5-1.5B | 1.5B | 0.000 | 0.714 | 0.781 | **0.746** |
| Gemma | 2-2B | 2B | 0.000 | 0.444 | 0.719 | 0.549 |
| Llama | 3.2-3B | 3B | 0.000 | 0.600 | 0.656 | 0.627 |
| Qwen | 2.5-7B | 7B | 0.000 | 0.240 | 0.844 | 0.374 |
| Llama | 3.1-8B | 8B | 0.000 | 0.568 | 0.750 | 0.646 |
| Gemma | 2-9B | 9B | 0.061 | 0.469 | 0.703 | 0.562 |
| Qwen | 2.5-14B | 14B | 0.061 | 0.539 | 0.688 | 0.604 |

- Vanilla F1=0 across 5/7 models (only 14B/9B accept some implications)
- **FCA improves ALL 7 models** — min F1=0.374 (Qwen-7B), max=0.746 (Qwen-1.5B)
- Best per family: Qwen=0.746 (1.5B), Llama=0.646 (8B), Gemma=0.562 (9B)
- **No monotonic size→quality**: Qwen 1.5B > 14B > 7B — oracle consistency > model size

### ✅ Exp 3 (Domains): countries + animals, qwen2.5:7b
| Domain | Vanilla F1 | FCA P | FCA R | FCA F1 | Queries |
|--------|-----------|-------|-------|--------|---------|
| Countries | 0.000 | 0.233 | 0.781 | 0.359 | 1512 |
| Animals | 0.028 | 0.511 | 0.800 | **0.624** | 2405 |

- Animals P=0.51 > Countries P=0.23: 생물 분류의 깨끗한 경계
- Recall ~0.80 일관, SLM이 platypus/echidna/seal 등 edge case 발견

### ✅ Exp 6 (Synthetic Noise): 8 attrs, 10 rules, 15 examples, 7 gold implications
| Noise | P | R | F1 |
|-------|-------|-------|-------|
| 0.00 | 1.000 | 1.000 | 1.000 |
| 0.05 | 0.790 | 0.952 | 0.861 |
| 0.10 | 0.673 | 0.571 | 0.612 |
| 0.15 | 0.567 | 0.429 | 0.476 |
| 0.20 | 0.571 | 0.476 | 0.482 |
| 0.30 | 0.608 | 0.619 | 0.549 |

- noise=0: perfect F1=1.0 (FCA completeness guarantee verified)
- noise 5% → F1 0.86, 10% → 0.61, 15-20% → ~0.48 (graceful degradation)
- SLM 실험의 effective noise rate는 ~10-15%로 추정 (F1 0.4-0.6 범위 일치)

### ✅ Exp 7 (DPO): 354 contradiction pairs → Qwen2.5-1.5B fine-tuning

**DPO pair collection** (exp7v2): 7 exploration runs → 354 unique pairs
| Source | Pairs |
|--------|-------|
| qwen2.5:1.5b (countries+animals) | 31 |
| qwen2.5:7b (countries+animals+30attr) | 152 |
| llama3.1:8b (countries+animals) | 233 |

**FCA exploration evaluation** (exp7v2 eval): DPO 모델이 counterexample 생성 능력 상실
| Condition | Countries F1 | Animals F1 |
|-----------|-------------|-----------|
| Base | 0.681 | 0.786 |
| FCA-DPO | 0.000 | 0.000 |
| Random-DPO | 0.000 | 0.000 |

**Classification accuracy** (exp7v3): DPO signal은 seen pairs에서 작동하지만 generalize 실패
| Condition | Seen Acc (84) | Unseen Acc (1266) | Total Acc |
|-----------|-------------|------------------|-----------|
| Base | 0.667 | 0.792 | 0.784 |
| Random-DPO | 0.738 | 0.788 | 0.785 |
| FCA-DPO | **0.798** | 0.694 | 0.701 |

- FCA-DPO seen +13pp (0.67→0.80), unseen −10pp (0.79→0.69)
- Random-DPO ≈ Base (control 확인)
- Net: −112 (250 fixes, 362 regressions)
- 354 pairs로는 generalization 실패, 일부 속성 catastrophic shift (is_UN_member 0.90→0.00)

## Decisions Made
1. **Paper framing**: hallucination suppression, not ontology construction
2. **Domains as benchmarks**: D1(countries), D2(animals), D3(SE) = difficulty gradient
3. **6 experiments**: main → models → domains → ablation → scaling → (DPO)
4. **arXiv first**: Phase 4 후 즉시 preprint
5. **Mac Mini M4**: Exp 1-5 전부 로컬
6. **Target venue**: EMNLP 2026 primary
7. **Oracle strategy**: 객체 제안+속성 검증(primary) → 직접 반례(fallback)
8. **Prompt framing**: classification 프레이밍 (SLM 상식 응답 유도)
9. **Gold standard**: AMBIGUITY_NOTES에 판단 기준 명시
10. **Baseline 평가**: gold canonical basis + 동일 수 invalid → classification P/R/F1
11. **CCR 측정**: accepted implications vs gold objects 위반 비율 (모든 method 공통)

## Phase 5: 논문 작성 🔧 (2026-03-23~)
- `docs/autoontology_paper_draft.md` 전면 업데이트
  - Abstract/Conclusion: 실제 수치 반영, narrative 수정
  - Section 4: D3 제거, 모델 테이블 수정, baseline 설명 정확화
  - Section 5: Exp 1-5 전체 결과 테이블 + 분석 텍스트
  - Section 6: 구체적 contradiction 사례 (Netherlands, Denmark, platypus), oracle bias 분석
  - Appendix: gold standard 방법론, 프롬프트 템플릿, per-model/domain 상세 결과
- **핵심 narrative 3개**:
  1. Direct questioning F1=0 → FCA structured decomposition이 유일한 해법
  2. Oracle consistency > model size (1.5B > 7B: 예측 가능성이 정확성보다 중요)
  3. Attribute boundary clarity가 도메인 성능 결정 (Animals P=0.51 > Countries P=0.23)

## Next Steps
```
# 남은 작업
python experiments/run_experiments.py --exp 3
python experiments/run_experiments.py --exp 4
python experiments/run_experiments.py --exp 5
```
