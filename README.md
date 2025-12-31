# Radiant RAG Pipeline

## Agentic Retrieval-Augmented Generation System

---

## Project Summary

Radiant is a modular, agentic RAG (Retrieval-Augmented Generation) pipeline that combines intelligent query processing, hierarchical document retrieval, and iterative answer refinement. The system uses a multi-agent architecture where specialized agents collaborate to transform user queries into high-quality, citation-backed responses.

### Key Features

- **Multi-Agent Architecture**: 18+ specialized agents handle distinct pipeline stages
- **Iterative Refinement**: Critic → Policy → Rewrite loop improves answer quality
- **Hierarchical Retrieval**: Dual-index support for leaf and parent chunks
- **Query Enhancement**: PRF (Pseudo-Relevance Feedback) and Query Expansion
- **Cross-Encoder Reranking**: Improves retrieval precision
- **Multi-Backend LLM Support**: HuggingFace, vLLM, Ollama, OpenAI-compatible APIs
- **Comprehensive Telemetry**: Event-based monitoring and performance metrics
- **HTML Report Generation**: Visual debugging and evaluation reports

### Pipeline Flow

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Query Processing Phase                       │
│  Router → Decomposition → Planner → Guardrail                   │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Retrieval Enhancement Phase                  │
│  PRF → Query Expansion                                          │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Document Retrieval Phase                     │
│  Hybrid Retriever → Cross-Encoder Reranker                      │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Generation Loop (Iterative)                  │
│  Generator → Critic → Policy ─┬─→ FINALIZE → PostProcessor      │
│       ▲                       │                                 │
│       └── Rewriter ←──────────┴─→ REWRITE                       │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
Final Answer + Citations
```

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/dshipley71/radiant.git
cd radiant

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from core.orchestrator import register_default_agents, agentic_once_with_metadata

# Initialize the pipeline
register_default_agents(config_path="config/config.fast.yaml")

# Run a query
result = agentic_once_with_metadata("What is hierarchical RAG?")

print(result["answer"].text)
print(f"Citations: {len(result['citations'])}")
```

### Generate HTML Report

```bash
# Full Agentic RAG report
python agentic_rag_report.py --config config/config.fast.yaml

# Retrieval-only (BM25) debugging
python agentic_rag_report.py --mode retrieval --config config/config.fast.yaml
```

---

## Architecture Overview

### Component Hierarchy

```
radiant/
├── core/                    # Core infrastructure
│   ├── orchestrator.py      # Pipeline coordinator
│   ├── interfaces.py        # Agent base classes (18 interfaces)
│   ├── schemas.py           # Pydantic data models (70 schemas)
│   └── llm_router.py        # LLM abstraction layer
│
├── agents/                  # Agent implementations
│   ├── router.py            # Query classification
│   ├── planner.py           # Execution planning
│   ├── decomposition.py     # Query decomposition
│   ├── guardrail.py         # Safety validation
│   ├── prf.py               # Pseudo-relevance feedback
│   ├── qe.py                # Query expansion
│   ├── retriever.py         # Document retrieval
│   ├── rerank.py            # Cross-encoder reranking
│   ├── generator.py         # Answer generation
│   ├── critic.py            # Answer evaluation
│   ├── policy.py            # Decision making
│   ├── rewrite.py           # Query rewriting
│   ├── postprocess.py       # Answer formatting
│   └── telemetry.py         # Event logging
│
├── agentic_rag_report.py    # Report generator entry point
├── agentic_report_html.py   # HTML rendering utilities
│
└── config/
    └── config.fast.yaml     # Main configuration
```

---

## Documentation Reference

The following documentation provides detailed technical references for each component, organized from high-level orchestration down to individual agents.

### Report Generation

| Document | Description |
|----------|-------------|
| [AgenticRagReport_Documentation.md](docs/AgenticRagReport_Documentation.md) | HTML report generator and smoke test tool. Covers RAG and retrieval-only modes, CLI/programmatic usage, metrics computation. |
| [AgenticReportHtml_Documentation.md](docs/AgenticReportHtml_Documentation.md) | HTML rendering utilities. Documents all rendering functions, CSS styles, score normalization, backend classification. |

### Core Infrastructure

| Document | Description |
|----------|-------------|
| [Orchestrator_Documentation.md](docs/Orchestrator_Documentation.md) | Central pipeline coordinator. Covers agent registry, configuration loading, 5 pipeline phases, iteration logic, telemetry events. |
| [CoreInterfaces_Documentation.md](docs/CoreInterfaces_Documentation.md) | Abstract base classes defining contracts for all 18 agent types. Documents interface segregation and schema contracts. |
| [CoreSchemas_Documentation.md](docs/CoreSchemas_Documentation.md) | All 70 Pydantic data models. Covers 7 enums, input/output schemas for all agents, schema relationships. |
| [LLMRouter_Documentation.md](docs/LLMRouter_Documentation.md) | Unified LLM abstraction layer. Supports HuggingFace, vLLM, Ollama, OpenAI-compatible APIs with lazy loading. |

### Query Processing Agents

| Document | Description |
|----------|-------------|
| [BasicRouterAgent_Documentation.md](docs/BasicRouterAgent_Documentation.md) | Query classification agent. Determines query type, enables/disables pipeline features (QE, PRF, rerank). |
| [BasicDecompositionAgent_Documentation.md](docs/BasicDecompositionAgent_Documentation.md) | Query decomposition agent. Splits complex queries into sub-questions, identifies comparison pairs. |
| [BasicPlannerAgent_Documentation.md](docs/BasicPlannerAgent_Documentation.md) | Execution planning agent. Creates Plan objects with retrieval mode, iteration limits, feature flags. |
| [BasicGuardrailAgent_Documentation.md](docs/BasicGuardrailAgent_Documentation.md) | Safety validation agent. Enforces resource limits, validates plan parameters. |

### Retrieval Enhancement Agents

| Document | Description |
|----------|-------------|
| [BasicPRFAgent_Documentation.md](docs/BasicPRFAgent_Documentation.md) | Pseudo-relevance feedback agent. Extracts expansion terms from initial BM25 results using TF-IDF. |
| [LLMQEAgent_Documentation.md](docs/LLMQEAgent_Documentation.md) | LLM-based query expansion agent. Generates semantic query variants for improved recall. |

### Document Retrieval Agents

| Document | Description |
|----------|-------------|
| [HybridRetrievalAgent_Documentation.md](docs/HybridRetrievalAgent_Documentation.md) | Hybrid retrieval agent. Combines dense (ChromaDB) and sparse (BM25) retrieval with RRF fusion. |
| [BasicRerankAgent_Documentation.md](docs/BasicRerankAgent_Documentation.md) | Cross-encoder reranking agent. Re-scores candidates using transformer models for precision. |

### Generation Agents

| Document | Description |
|----------|-------------|
| [LLMGeneratorAgent_Documentation.md](docs/LLMGeneratorAgent_Documentation.md) | Answer generation agent. Creates RAG prompts, generates answers with citations. |
| [BasicCriticAgent_Documentation.md](docs/BasicCriticAgent_Documentation.md) | Answer evaluation agent. Assesses hallucination risk, coverage score, identifies missing topics. |
| [BasicPolicyAgent_Documentation.md](docs/BasicPolicyAgent_Documentation.md) | Decision-making agent. Returns FINALIZE/REWRITE/CONTINUE based on critic feedback and metrics. |
| [LLMQueryRewriteAgent_Documentation.md](docs/LLMQueryRewriteAgent_Documentation.md) | Query rewriting agent. Refines queries based on critic feedback to improve retrieval. |
| [BasicPostProcessorAgent_Documentation.md](docs/BasicPostProcessorAgent_Documentation.md) | Answer formatting agent. Applies markdown/plain formatting, adds critic notes. |

### Infrastructure Agents

| Document | Description |
|----------|-------------|
| [BasicTelemetryAgent_Documentation.md](docs/BasicTelemetryAgent_Documentation.md) | Telemetry logging agent. Passive event acknowledgment (orchestrator stores events). |

---

## Configuration

### Main Configuration File

The pipeline is configured via `config/config.fast.yaml`.

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `AGENTIC_RAG_CONFIG` | Config file path | `config/config.fast.yaml` |
| `AGENTIC_MAX_ITERS` | Max iteration override | From config |
| `AGENTIC_MIN_ITERS` | Min iteration override | From config |
| `OPENAI_API_KEY` | API key for OpenAI-compatible backends | - |

---

## Agent Reference

### Agent Types by Category

#### Query Processing
| Agent | Interface | Purpose |
|-------|-----------|---------|
| BasicRouterAgent | RouterAgent | Classify query type and features |
| BasicDecompositionAgent | DecompositionAgent | Split into sub-queries |
| BasicPlannerAgent | PlannerAgent | Create execution plan |
| BasicGuardrailAgent | GuardrailAgent | Validate and constrain plan |

#### Retrieval Enhancement
| Agent | Interface | Purpose |
|-------|-----------|---------|
| BasicPRFAgent | PRFAgent | Extract PRF expansion terms |
| LLMQEAgent | QEAgent | Generate query variants |

#### Document Retrieval
| Agent | Interface | Purpose |
|-------|-----------|---------|
| HaystackChromaRetrieverAgent | RetrieverAgent | Hybrid dense+sparse retrieval |
| BasicRerankAgent | RerankAgent | Cross-encoder reranking |

#### Generation
| Agent | Interface | Purpose |
|-------|-----------|---------|
| LLMGeneratorAgent | GeneratorAgent | Generate answers |
| BasicCriticAgent | CriticAgent | Evaluate answers |
| BasicPolicyAgent | PolicyAgent | Decide next action |
| LLMQueryRewriteAgent | QueryRewriteAgent | Rewrite queries |
| BasicPostProcessorAgent | PostProcessorAgent | Format output |

#### Infrastructure
| Agent | Interface | Purpose |
|-------|-----------|---------|
| BasicTelemetryAgent | TelemetryAgent | Log events |

---

## Telemetry Events

The orchestrator emits 17+ telemetry event types:

| Event Type | Agent | Description |
|------------|-------|-------------|
| `router.decided` | Router | Query classification complete |
| `decomposition.done` | Decomposition | Query split complete |
| `planner.plan` | Planner | Plan created |
| `guardrail.check` | Guardrail | Plan validated |
| `prf.compute` | PRF | Expansion terms extracted |
| `qe.expand` | QE | Query variants generated |
| `retriever.results` | Retriever | Documents retrieved |
| `rerank.rerank` | Reranker | Candidates reranked |
| `generator.output` | Generator | Answer generated |
| `critic.evaluate` | Critic | Answer evaluated |
| `policy.decision` | Policy | Decision made |
| `rewrite.rewrite` | Rewriter | Query rewritten |
| `postprocess.format` | PostProcessor | Output formatted |

---

## Data Schemas

### Key Schema Categories

| Category | Count | Examples |
|----------|-------|----------|
| Enumerations | 7 | BackendEnum, DecisionEnum, PhaseEnum |
| Common | 3 | RequestContext, RuntimeContext, TelemetryTiming |
| Query Processing | 15 | RouterInput/Output, Plan, Decomposition |
| Retrieval | 10 | RetrieverInput/Output, RetrievalResult, Snippet |
| Generation | 17 | GeneratorInput/Output, Answer, Citation |
| Infrastructure | 10 | TelemetryEvent, SafetyInput/Output |

### Core Data Flow

```
RouterInput → RouterOutput (RouterProfile)
    ↓
DecompositionInput → DecompositionOutput (Decomposition)
    ↓
PlannerInput → PlannerOutput (Plan)
    ↓
RetrieverInput → RetrieverOutput (List[RetrievalResult])
    ↓
GeneratorInput → GeneratorOutput (Answer, Citations)
    ↓
CriticInput → CriticOutput (CriticFeedback)
    ↓
PolicyInput → PolicyOutput (DecisionEnum)
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=core --cov=agents --cov-report=html

# Run specific agent tests
pytest tests/test_router.py -v
```

### Smoke Test

```bash
# Quick validation of full pipeline
python agentic_rag_report.py --config config/config.fast.yaml \
    --query "What is RAG?"
```

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Code Style

- Type hints required for all functions
- Pydantic models for all data structures
- Docstrings for public methods
- Follow existing agent patterns

---

## License

See [LICENSE](LICENSE) for details.

---

## References

- **Repository**: https://github.com/dshipley71/radiant
- **Documentation**: See `docs/` directory
- **Issues**: https://github.com/dshipley71/radiant/issues

---

*Radiant RAG Pipeline - Intelligent, Iterative, Citation-Backed Answers*
