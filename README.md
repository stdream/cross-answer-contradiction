# AutoOntology

> SLM hallucination guardrail with provable completeness guarantees
> via FCA attribute exploration.

**Core claim**: Self-consistency catches within-question noise. FCA catches between-question contradictions. Only FCA guarantees completeness.

## Quick Start

```bash
ollama pull qwen2.5:7b
pip install -e ".[dev]"
python run.py --domain countries --model qwen2.5:7b
pytest tests/ -v
```

## How It Works

```
FCA asks structured question → SLM answers → FCA checks consistency
→ if contradiction: re-query with evidence → repeat until provably complete
```

## Experiments

| Exp | Question | Key metric |
|-----|----------|-----------|
| 1 | FCA vs baselines? | Contradiction rate, accuracy |
| 2 | Helps weak models more? | Improvement by model size |
| 3 | Works across domains? | Factual → commonsense → specialized |
| 4 | Which component matters? | Ablation of each piece |
| 5 | Does it scale? | Attributes 10 → 30 |

## References

- Ganter & Obiedkov (2016) *Conceptual Exploration*, Springer
- Karpathy (2026) *autoresearch*, GitHub
