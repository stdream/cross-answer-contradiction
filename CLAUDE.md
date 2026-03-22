# CLAUDE.md — AutoOntology Project Guidelines v2

## Project: SLM Hallucination Guardrail via FCA Attribute Exploration

이 프로젝트의 핵심 메시지는 **ontology가 아니라 hallucination suppression**이다.
도메인(countries, animals, SE)은 hallucination 측정 벤치마크일 뿐, 응용 목적이 아니다.

## Workflow: Document & Clear 패턴

1. `prd.md` 읽기 → 요구사항 파악
2. `implementation-plan.md` 확인 → 현재 Phase 파악
3. Phase별로 작업 수행
4. 세션 종료 시 `progress.md` 업데이트
5. compact 대신 `progress.md`로 세션 간 상태 공유

## Tech Stack

- Python 3.11+ (Mac Mini M4, macOS)
- Ollama (`http://localhost:11434`)
- Models: `qwen2.5:1.5b`, `qwen2.5:7b`, `exaone3.5:7.8b`, `qwen2.5:14b`
- Dependencies: `requests` (Ollama API), `pytest` (testing)
- No torch/transformers (inference는 Ollama가 처리)

## Core Principles

### 1. fca_engine.py는 절대 수정하지 않는다
수학적 알고리즘. 한번 구현+테스트 통과 후 불변. 모든 실험 비교의 기반.

### 2. Algorithm 19 충실 구현
Ganter & Obiedkov (2016) Ch.4 기본 버전:
```
L := ∅, A := ∅
while A ≠ M:
    while A ≠ A^JJ:
        if A^II = A^JJ: L := L ∪ {A → A^JJ}; break
        else: extend E by counterexample
    A := NextClosure(A, M, L)
return L
```

### 3. check_consistency = 논문의 핵심 contribution
반례가 확인된 함의를 위반하면 = hallucination detected.
위반 증거를 prompt에 포함해서 재질의 = self-correction.

### 4. 모든 실험은 재현 가능
- Ollama temperature=0.1 (거의 deterministic)
- random seed 고정
- 모든 raw response 로그 저장
- Gold standard는 스크립트로 재생성 가능

### 5. 실험 결과가 논문의 전부
Exp 1(main) → Exp 2(models) → Exp 3(domains) → Exp 4(ablation) → Exp 5(scaling)
이 순서대로 구현하고, 각 실험이 독립적으로 실행 가능해야 한다.

## Implementation Phases

### Phase 1: FCA Engine + Tests
- fca_engine.py 구현 (Algorithm 19)
- tests/test_fca_engine.py (mock oracle로 과일 E2E)
- 완료 기준: 과일 도메인 정준 기저 100% 일치

### Phase 2: Oracle + Runner
- oracle.py (Ollama 연결, prompt formatting, self-correction loop)
- domain.py (D0-D3 정의)
- run.py (CLI, JSONL logging)
- 완료 기준: `python run.py --domain fruits` 정상 실행

### Phase 3: Gold Standards + Baselines
- gold_standards/ (countries.json, animals.json, se_concepts.json)
- baselines/ (vanilla, cot, selfconsistency)
- 완료 기준: 모든 baseline이 동일 metric으로 평가 가능

### Phase 4: Evaluation + Experiments
- evaluate.py (모든 metric)
- experiments/ (exp1~exp5 config + runner)
- 완료 기준: Exp 1-5 결과 생성, 비교 테이블 출력

### Phase 5: arXiv + (Optional) DPO
- 논문 작성, arXiv 업로드
- Exp 6 (cloud A100, DPO)

## Coding Style

- Type hints, dataclass, f-string
- Docstring 간결 (한 줄 + 필요시 Args/Returns)
- 한국어 주석 OK
- 실험 config는 JSON, 결과는 JSONL + summary JSON

## Testing

```bash
pytest tests/ -v                           # 전체
pytest tests/test_fca_engine.py -v         # FCA only
pytest tests/ -v -m "not requires_ollama"  # Ollama 없이
```

## Running

```bash
# Single experiment
python run.py --domain countries --model qwen2.5:7b

# Baseline
python run.py --domain countries --model qwen2.5:7b --method vanilla

# Batch (experiment config)
python run.py --config experiments/exp1_main.json

# Evaluate
python evaluate.py --results results/ --gold gold_standards/
```

## Key References

- Ganter & Obiedkov (2016) Ch.4 Algorithm 19
- Ganter & Obiedkov (2016) Ch.2.2 Next Closure
- Borchmann (2012) arXiv:1202.4824 General Attribute Exploration
- Karpathy (2026) autoresearch pattern
