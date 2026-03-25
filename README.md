# Beyond Single-Answer Hallucination

**Detecting and Learning from Cross-Answer Contradictions in Language Models via Formal Concept Analysis**

[![arXiv](https://img.shields.io/badge/arXiv-2026.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2026.xxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Language models produce plausible but logically inconsistent responses across different questions about the same domain. We introduce a framework that uses **attribute exploration** from Formal Concept Analysis (FCA) as a symbolic guardrail—decomposing abstract rule judgment into concrete object-level binary questions and cross-checking every response against previously confirmed knowledge.

## Key Findings

1. **Direct implication questioning fails completely** — all baselines achieve F1=0 because SLMs cannot judge universal rules, even in a closed-world setting with explicitly enumerated entities.
2. **FCA achieves F1=0.37–0.75 across 7 models and 3 families** (Qwen, Llama, Gemma). Oracle consistency matters more than model size: Qwen2.5-1.5B outperforms 7B and 14B.
3. **DPO with contradiction traces** shows valid signal on seen pairs (+13pp accuracy) but fails to generalize with 354 training examples, pointing to data scale as the bottleneck.

## Quick Start

```bash
# Requirements: Python 3.10+, Ollama running locally
pip install requests pytest

# Build gold standards
python gold_standards/build.py

# Run FCA exploration
ollama pull qwen2.5:7b
python run.py --domain countries --model qwen2.5:7b

# Evaluate against gold standard
python -c "
from evaluate import evaluate_fca_result
r = evaluate_fca_result('results/YOUR_RESULT.json', 'gold_standards/countries.json')
print(f'P={r[\"metrics\"][\"precision\"]:.3f} R={r[\"metrics\"][\"recall\"]:.3f} F1={r[\"metrics\"][\"f1\"]:.3f}')
"

# Run tests
pytest tests/test_fca_engine.py -v
```

## Project Structure

```
├── fca_engine.py          # Core FCA algorithm (Algorithm 19, Ganter & Obiedkov 2016)
├── oracle.py              # SLM oracle with structured querying & self-correction
├── run.py                 # CLI runner with JSONL logging
├── evaluate.py            # Metrics: closure-based P/R/F1, CCR
├── domain.py              # Domain definitions (countries, animals, scaling variants)
├── gold_standards/
│   ├── build.py           # Reproducible gold standard generation
│   ├── countries.json     # 50 countries × 15 attributes, 64 implications
│   ├── countries_30.json  # 50 countries × 30 attributes, 244 implications
│   └── animals.json       # 40 animals × 15 attributes, 70 implications
├── baselines/
│   ├── baseline_vanilla.py          # Direct implication questioning
│   ├── baseline_cot.py              # Chain-of-thought variant
│   ├── baseline_selfconsistency.py  # k-shot majority voting
│   ├── baseline_closedworld.py      # Closed-world variant
│   ├── baseline_structured.py       # Exhaustive survey + FCA engine
│   └── common.py                    # Shared utilities
├── experiments/
│   ├── run_experiments.py   # Exp 1–5 runner
│   ├── exp6_synthetic.py    # Synthetic noise study
│   ├── exp7v2_dpo_pairs.py  # DPO pair collection
│   └── exp7v3_classification.py  # DPO classification evaluation
└── tests/
    └── test_fca_engine.py   # 42 unit + integration tests
```

## Experiments

| Exp | Question | Result |
|-----|----------|--------|
| 1 | FCA vs baselines? | All baselines F1=0; FCA F1=0.41±0.02 (7B) |
| 2 | Across models & families? | Best: Qwen-1.5B F1=0.75; all 7 models improved |
| 3 | Across domains? | Countries F1=0.36, Animals F1=0.62 |
| 4 | Which component matters? | Structured query critical (+0.42), consistency +0.06 |
| 5 | Does it scale? | 10→15→30 attrs: F1=0.43→0.52→0.50, 30 attrs in 2hrs |
| 6 | Noise sensitivity? | Perfect at 0%, graceful degradation to 20% |
| 7 | DPO fine-tuning? | Seen accuracy +13pp, unseen −10pp (354 pairs insufficient) |

## How It Works

```
┌─────────────┐    question     ┌─────────────┐
│  FCA Engine  │───────────────▶│  SLM Oracle  │
│ (Algorithm   │◀───────────────│  (Ollama)    │
│   19)        │  answer        └──────┬───────┘
│              │                       │
│  L: rules    │    ┌──────────────────▼──────────┐
│  E: examples │◀───│  Consistency Checker         │
│              │    │  - cross-answer validation   │
└──────────────┘    │  - evidence-based correction │
                    └─────────────────────────────┘
```

1. FCA proposes a candidate implication (e.g., "Do all NATO members have coastline?")
2. Oracle suggests diverse objects, verifies each attribute with binary yes/no questions
3. Consistency checker cross-validates against previously confirmed knowledge
4. On contradiction: evidence-based self-correction re-query
5. Process terminates with provable completeness guarantee

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai/) with any supported model (tested: Qwen2.5, Llama3, Gemma2)
- ~64GB RAM for 14B models; 8GB sufficient for 1.5B–3B

## Citation

```bibtex
@article{autoontology2026,
  title={Beyond Single-Answer Hallucination: Detecting and Learning from
         Cross-Answer Contradictions in Language Models via Formal Concept Analysis},
  author={Anonymous},
  journal={arXiv preprint arXiv:2026.xxxxx},
  year={2026}
}
```

## License

MIT
