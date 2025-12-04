from __future__ import annotations

from abc import ABC, abstractmethod

from agents_schemas import (
    CriticInput,
    CriticOutput,
    DecompositionInput,
    DecompositionOutput,
    GeneratorInput,
    GeneratorOutput,
    GuardrailInput,
    GuardrailOutput,
    IndexInput,
    IndexOutput,
    PlannerInput,
    PlannerOutput,
    PolicyInput,
    PolicyOutput,
    PRFInput,
    PRFOutput,
    QEInput,
    QEOutput,
    QueryRewriteInput,
    QueryRewriteOutput,
    RetrieverInput,
    RetrieverOutput,
    RerankInput,
    RerankOutput,
    RouterInput,
    RouterOutput,
    SafetyInput,
    SafetyOutput,
    TelemetryEvent,
    TelemetryOutput,
    ToolExecutionInput,
    ToolExecutionOutput,
    TranslationInput,
    TranslationOutput,
    PostprocessInput,
    PostprocessOutput,
)


class BaseAgent(ABC):
    """Base interface for all agents."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this agent implementation."""
        ...

    @property
    def role(self) -> str:
        """
        Logical role used for registry lookups.

        By default, returns the class attribute `role` if present, otherwise the class name.
        Subclasses may override this property if they need dynamic behavior.
        """
        return getattr(self.__class__, "role", self.__class__.__name__)

    @abstractmethod
    def describe(self) -> str:
        """Short description of what this agent does."""
        ...


class RouterAgent(BaseAgent):
    role = "router"

    def describe(self) -> str:
        return "RouterAgent: classify query and set high-level toggles."

    @abstractmethod
    def route(self, inp: RouterInput) -> RouterOutput:
        ...


class DecompositionAgent(BaseAgent):
    role = "decomposition"

    def describe(self) -> str:
        return "DecompositionAgent: detect multi-part queries."

    @abstractmethod
    def decompose(self, inp: DecompositionInput) -> DecompositionOutput:
        ...


class PlannerAgent(BaseAgent):
    role = "planner"

    def describe(self) -> str:
        return "PlannerAgent: build an execution plan from router/decomposition."

    @abstractmethod
    def plan(self, inp: PlannerInput) -> PlannerOutput:
        ...


class GuardrailAgent(BaseAgent):
    role = "guardrail"

    def describe(self) -> str:
        return "GuardrailAgent: validate and adjust the plan."

    @abstractmethod
    def validate_plan(self, inp: GuardrailInput) -> GuardrailOutput:
        ...


class TranslationAgent(BaseAgent):
    role = "translation"

    def describe(self) -> str:
        return "TranslationAgent: detect language and optionally translate text."

    @abstractmethod
    def normalize(self, inp: TranslationInput) -> TranslationOutput:
        ...


class QEAgent(BaseAgent):
    role = "qe"

    def describe(self) -> str:
        return "QEAgent: perform LLM-based query expansion."

    @abstractmethod
    def expand(self, inp: QEInput) -> QEOutput:
        ...


class PRFAgent(BaseAgent):
    role = "prf"

    def describe(self) -> str:
        return "PRFAgent: compute pseudo-relevance feedback."

    @abstractmethod
    def compute(self, inp: PRFInput) -> PRFOutput:
        ...


class RetrieverAgent(BaseAgent):
    role = "retriever"

    def describe(self) -> str:
        return "RetrieverAgent: perform hierarchical hybrid retrieval."

    @abstractmethod
    def retrieve(self, inp: RetrieverInput) -> RetrieverOutput:
        ...


class RerankAgent(BaseAgent):
    role = "rerank"

    def describe(self) -> str:
        return "RerankAgent: rerank retrieval results with a cross-encoder."

    @abstractmethod
    def rerank(self, inp: RerankInput) -> RerankOutput:
        ...


class GeneratorAgent(BaseAgent):
    role = "generator"

    def describe(self) -> str:
        return "GeneratorAgent: generate a RAG answer from context snippets."

    @abstractmethod
    def generate(self, inp: GeneratorInput) -> GeneratorOutput:
        ...


class CriticAgent(BaseAgent):
    role = "critic"

    def describe(self) -> str:
        return "CriticAgent: evaluate coverage and hallucination risk."

    @abstractmethod
    def evaluate(self, inp: CriticInput) -> CriticOutput:
        ...


class PolicyAgent(BaseAgent):
    role = "policy"

    def describe(self) -> str:
        return "PolicyAgent: decide whether to continue, rewrite, or finalize."

    @abstractmethod
    def decide(self, inp: PolicyInput) -> PolicyOutput:
        ...


class QueryRewriteAgent(BaseAgent):
    role = "rewrite"

    def describe(self) -> str:
        return "QueryRewriteAgent: refine the query based on critic feedback."

    @abstractmethod
    def rewrite(self, inp: QueryRewriteInput) -> QueryRewriteOutput:
        ...


class PostProcessorAgent(BaseAgent):
    role = "postprocess"

    def describe(self) -> str:
        return "PostProcessorAgent: format the final answer."

    @abstractmethod
    def format(self, inp: PostprocessInput) -> PostprocessOutput:
        ...


class TelemetryAgent(BaseAgent):
    role = "telemetry"

    def describe(self) -> str:
        return "TelemetryAgent: log events and metrics."

    @abstractmethod
    def log_event(self, event: TelemetryEvent) -> TelemetryOutput:
        ...


class ToolExecutionAgent(BaseAgent):
    role = "tools"

    def describe(self) -> str:
        return "ToolExecutionAgent: execute external tools with policy checks."

    @abstractmethod
    def execute(self, inp: ToolExecutionInput) -> ToolExecutionOutput:
        ...


class SafetyAgent(BaseAgent):
    role = "safety"

    def describe(self) -> str:
        return "SafetyAgent: perform safety checks, redaction, or blocking."

    @abstractmethod
    def check(self, inp: SafetyInput) -> SafetyOutput:
        ...


class IndexManagementAgent(BaseAgent):
    role = "index"

    def describe(self) -> str:
        return "IndexManagementAgent: manage and report index state."

    @abstractmethod
    def manage(self, inp: IndexInput) -> IndexOutput:
        ...
