# AutoOntology Research Memo

> **Autonomous Knowledge Discovery with Formal Completeness Guarantees via Attribute Exploration and Small Language Models**
>
> KELAB, Hanyang University MOT | March 22, 2026

---

## 1. Research Origin

From Ganter & Obiedkov (2016) *Conceptual Exploration* Chapter 4 → LLM-era relevance question → FCA as symbolic hallucination guardrail → Karpathy autoresearch pattern connection.

**Core insight**: Attribute exploration is an autonomous agent loop over *knowledge space* (vs autoresearch's *code space*), with provable completeness that autoresearch lacks.

## 2. Core Idea: FCA as Symbolic Guardrail

### Completeness Proof (Intuition)
- **From above**: Confirmed rules L narrow possibilities (eliminate impossible combinations)
- **From below**: Counterexamples E prove possibilities (anchor feasible combinations)
- **Convergence**: Algorithm terminates when L(A) = A^JJ for all A → no gap remains

### Three Hallucination Suppression Mechanisms
1. **Contradiction detection**: Cross-answer inconsistency caught by FCA engine
2. **Structured binary questioning**: Yes/no constrains output space
3. **Monotonic accumulation**: Knowledge base only grows, never contradicts

### Advantage vs Existing Methods
- RAG, CoT, self-consistency = heuristic, no completeness guarantee
- FCA exploration = provably complete, minimal questioning, cross-answer consistency

## 3. Literature Gap

**Existing work**: FCA+LLM for RCA interpretation (LIRMM 2025), LLM as ontology oracle (Ciatto 2025), DL+FCA ontology learning (Baader-Distel 2009, Kriegel 2020), LLMs4OL challenge (ISWC 2024/2025)

**The gap**: No published work uses LLM as domain expert oracle *within* attribute exploration algorithm while leveraging FCA completeness as hallucination detection mechanism.

## 4. Paper Plan

### Title
**"AutoOntology: Autonomous Knowledge Discovery with Formal Completeness Guarantees via Attribute Exploration and Small Language Models"**

### Positioning
LLM agents paper (not FCA paper). Narrative: autoresearch for knowledge, with formal guarantees.

### Two-Stage Contribution
| Stage | Contribution | Infra |
|-------|-------------|-------|
| Stage 1 | FCA exploration + SLM oracle + self-correction loop | Mac Mini M4 ($0) |
| Stage 2 | Contradiction traces → DPO preference pairs → SLM fine-tuning | Cloud A100 ($100-200) |

### Experimental Design
- **Models**: Qwen2.5-7B, EXAONE-3.5-7.8B (Ollama on Mac Mini M4)
- **Domains**: (a) Toy, (b) SE ontology (BFO), (c) Biomedical subset
- **Baselines**: Vanilla SLM, CoT, Self-consistency (5-shot)
- **Metrics**: Contradiction rate, questions-to-convergence, canonical basis accuracy, axiom F1

### Target Venues
| Venue | Deadline | Priority |
|-------|----------|----------|
| arXiv preprint | Week 4 end | IMMEDIATE (priority claim) |
| EMNLP 2026 | ~Jun 2026 | Primary |
| NAACL 2027 | ~Oct 2026 | Secondary |
| ACL workshop TrustNLP | Various | Backup |

## 5. Execution Timeline (8 Weeks)

| Week | Task | Infra | Deliverable |
|------|------|-------|-------------|
| 1-2 | Prototype FCA engine + Ollama | Mac Mini ($0) | Working prototype |
| 3-4 | Inference experiments + baselines | Mac Mini ($0) | Stage 1 results |
| 4 end | arXiv upload | - | Priority claim |
| 5-6 | DPO training + ablation | Cloud A100 ($100-200) | Stage 2 results |
| 7-8 | Write + submit | - | Conference paper |

## 6. Future Research Questions

- **RQ3**: Scalability - hierarchical decomposition for 100+ attributes (IJCAI theory)
- **RQ4**: SE ontology application with BFO/TypeDB (Applied Ontology / IPM)
- **Multi-oracle**: Multiple SLMs as competing oracles, FCA arbitrating
- **Benchmark release**: AutoOntology as open framework

## 7. Infrastructure

| Resource | Role | Cost |
|----------|------|------|
| Mac Mini M4 64GB | All inference, prototyping, eval | $0 (owned) |
| Cloud A100 (1-2 days) | DPO fine-tuning | ~$100-200 |
| Ollama + Python FCA | SLM oracle + exploration engine | $0 |
| **Total** | | **$100-200** |

## 8. Key References

- Ganter & Obiedkov (2016) Conceptual Exploration, Springer [Ch.4]
- Baader, Ganter, Sertkaya, Sattler (2007) Completing DL KBs using FCA. IJCAI
- Borchmann (2012) General Form of Attribute Exploration. arXiv:1202.4824
- Kriegel (2020) Constructing DL Ontologies using FCA. KI Journal
- Ciatto et al. (2025) LLMs as oracles for ontologies. Knowledge-Based Systems
- Huchard et al. (2025) Empowering RCA using LLM. LIRMM
- LLMs4OL Challenge (ISWC 2024/2025)
- Karpathy (2026) autoresearch. GitHub (48K stars)

---

*"Any metric you care about that is reasonably efficient to evaluate can be autoresearched by an agent swarm."* — Karpathy, March 2026

Our metric is L(A) = A^JJ convergence. It's not just efficient to evaluate — it's **provably complete**.
