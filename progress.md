# Progress — AutoOntology v2

> Last updated: 2026-03-22
> Current phase: Phase 1 (not started)

## Status: INITIALIZED (experiment-driven redesign complete)

## Completed
- [x] Research ideation (FCA + LLM hallucination guardrail)
- [x] Literature review (gap confirmed: no LLM-as-oracle in attribute exploration)
- [x] autoresearch pattern connection identified
- [x] Experiment design: 6 experiments (main, models, domains, ablation, scaling, DPO)
- [x] Domain redesign: hallucination benchmarks not application demos
- [x] prd.md v2, CLAUDE.md v2, implementation-plan.md v2
- [x] Full project structure created

## Decisions Made
1. **Paper framing**: hallucination suppression, not ontology construction
2. **Domains as benchmarks**: D1(countries), D2(animals), D3(SE) = difficulty gradient
3. **6 experiments**: main → models → domains → ablation → scaling → (DPO)
4. **arXiv first**: Phase 4 후 즉시 preprint, then conference full version
5. **Mac Mini M4**: Exp 1-5 전부 로컬, Exp 6만 cloud
6. **Target venue**: EMNLP 2026 primary, NeurIPS D&B or ACL workshop backup

## Next Steps
1. Phase 1 시작: fca_engine.py 구현
2. 과일 mock oracle로 Algorithm 19 검증
