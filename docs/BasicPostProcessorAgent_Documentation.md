# BasicPostProcessorAgent Documentation

## Technical Reference for the Radiant RAG Pipeline Response Formatting

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Context](#architecture-context)
3. [Class Structure](#class-structure)
4. [Core Functionality](#core-functionality)
5. [Output Formatting](#output-formatting)
6. [Data Flow](#data-flow)
7. [Testing Strategies](#testing-strategies)
8. [Recommendations and Improvements](#recommendations-and-improvements)
9. [Usage Examples](#usage-examples)

---

## Overview

### Purpose

The `BasicPostProcessorAgent` is the response formatting component within the Radiant RAG pipeline. It takes the generated answer and enriches it with optional metadata sections including critic notes and language information before final delivery to the user.

### Key Responsibilities

- Format the base answer text
- Optionally append critic feedback notes
- Optionally append language observation notes
- Extract and deduplicate languages from context snippets
- Build structured metadata for downstream use

### Design Philosophy

The agent follows a **conditional enrichment** pattern where additional information is appended based on user preferences. This allows the same pipeline to produce both minimal (answer-only) and verbose (answer + diagnostics) outputs without code changes.

---

## Architecture Context

### Position in the Radiant Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    PostprocessInput                             │
│  answer | context_snippets | critic_feedback | preferences      │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  BasicPostProcessorAgent                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  1. Base Answer                                         │   │
│  │     └─ answer.text                                      │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  2. Critic Notes (if enabled)                           │   │
│  │     └─ ---                                              │   │
│  │     └─ Critic notes:                                    │   │
│  │     └─ - note 1                                         │   │
│  │     └─ - note 2                                         │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  3. Language Notes (if enabled)                         │   │
│  │     └─ ---                                              │   │
│  │     └─ Languages observed: en, es, fr                   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PostprocessOutput                            │
│  final_text | metadata                                          │
└─────────────────────────────────────────────────────────────────┘
```

### Related Components

| Component | Relationship |
|-----------|--------------|
| `PostProcessorAgent` | Abstract base class (from `core.interfaces`) |
| `PostprocessInput` | Input schema with answer, snippets, feedback, preferences |
| `PostprocessOutput` | Output schema with final text and metadata |
| `PostprocessMetadata` | Structured metadata for downstream use |
| `Answer` | Generated answer with text property |
| `CriticOutput` | Feedback from BasicCriticAgent |
| `ContextSnippet` | Retrieved snippet with language info |

---

## Class Structure

### Inheritance

```python
class BasicPostProcessorAgent(PostProcessorAgent):
    """Formats final answer with critic + language notes."""
```

### Class Attributes

| Attribute | Type | Value | Description |
|-----------|------|-------|-------------|
| `role` | `str` | `"postprocess"` | Agent role identifier |

### Properties

| Property | Return Type | Description |
|----------|-------------|-------------|
| `name` | `str` | Returns `"BasicPostProcessorAgent"` |

### Methods

| Method | Visibility | Purpose |
|--------|------------|---------|
| `describe()` | Public | Returns agent description |
| `format(inp)` | Public | Main formatting method |

---

## Core Functionality

### The `format()` Method

Primary method that formats the final response.

**Signature:**
```python
def format(self, inp: PostprocessInput) -> PostprocessOutput
```

**Parameters:**
- `inp` (`PostprocessInput`): Contains answer, context snippets, critic feedback, and preferences

**Returns:**
- `PostprocessOutput`: Contains formatted text and metadata

**Processing Steps:**

1. **Extract Base Answer**
   - Get `answer.text` or empty string if None

2. **Extract Languages**
   - Collect unique languages from context snippets
   - Sort alphabetically
   - Filter out None/empty values

3. **Build Output Lines**
   - Start with base answer
   - Conditionally add critic notes section
   - Conditionally add language notes section

4. **Build Metadata**
   - Combine critic notes into summary string
   - Include language list

5. **Return Output**
   - Join lines with newlines
   - Package with metadata

---

## Output Formatting

### Output Structure

```
┌─────────────────────────────────────────────────────────────────┐
│ BASE ANSWER                                                     │
│ The answer to your question is...                               │
├─────────────────────────────────────────────────────────────────┤
│ CRITIC NOTES (optional)                                         │
│                                                                 │
│ ---                                                             │
│ Critic notes:                                                   │
│ - Low coverage of available context (few relevant snippets).    │
│ - Answer is very short; consider elaborating if user needs...   │
├─────────────────────────────────────────────────────────────────┤
│ LANGUAGE NOTES (optional)                                       │
│                                                                 │
│ ---                                                             │
│ Languages observed: de, en, fr                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Formatting Rules

| Element | Format |
|---------|--------|
| Section separator | `\n---\n` |
| Critic header | `Critic notes:` |
| Critic items | `- {note}` |
| Language header | `Languages observed: ` |
| Language list | Comma-separated, sorted alphabetically |

### Output Variants

| Preferences | Output Contains |
|-------------|-----------------|
| Both disabled | Answer only |
| Critic enabled | Answer + critic notes |
| Language enabled | Answer + languages |
| Both enabled | Answer + critic notes + languages |

### Language Extraction Logic

```python
# Extract unique, non-empty languages, sorted
langs = sorted({
    cs.lang 
    for cs in inp.context_snippets 
    if cs.lang
}) if inp.context_snippets else []
```

**Behavior:**
- Uses set comprehension for deduplication
- Filters out `None` and empty strings via `if cs.lang`
- Sorts alphabetically
- Handles `None` context_snippets gracefully

---

## Data Flow

### Input Schema: `PostprocessInput`

```python
@dataclass
class PostprocessInput:
    answer: Answer                           # Generated answer
    context_snippets: List[ContextSnippet]   # Retrieved snippets
    critic_feedback: CriticOutput            # Quality feedback
    preferences: PostprocessPreferences      # Formatting preferences
```

### Output Schema: `PostprocessOutput`

```python
@dataclass
class PostprocessOutput:
    final_text: str                  # Formatted response text
    metadata: PostprocessMetadata    # Structured metadata
```

### Metadata Schema: `PostprocessMetadata`

```python
@dataclass
class PostprocessMetadata:
    critic_summary: str       # Semicolon-joined critic notes
    languages: List[str]      # Sorted list of observed languages
```

### Preferences Schema

```python
@dataclass
class PostprocessPreferences:
    include_critic_note: bool     # Whether to append critic notes
    include_language_notes: bool  # Whether to append language info
```

---

## Testing Strategies

### Unit Tests

#### 1. Basic Formatting Tests

```python
import pytest
from unittest.mock import Mock
from postprocess_basic_agent import BasicPostProcessorAgent
from core.schemas import PostprocessInput

@pytest.fixture
def agent():
    return BasicPostProcessorAgent()

@pytest.fixture
def make_input():
    def _make(
        answer_text: str = "This is the answer.",
        snippets: list = None,
        critic_notes: list = None,
        include_critic: bool = False,
        include_lang: bool = False
    ):
        answer = Mock()
        answer.text = answer_text
        
        critic_feedback = Mock()
        critic_feedback.notes = critic_notes or []
        
        preferences = Mock()
        preferences.include_critic_note = include_critic
        preferences.include_language_notes = include_lang
        
        return PostprocessInput(
            answer=answer,
            context_snippets=snippets or [],
            critic_feedback=critic_feedback,
            preferences=preferences
        )
    return _make

class TestBasicFormatting:
    
    def test_answer_only(self, agent, make_input):
        inp = make_input(answer_text="Simple answer.")
        output = agent.format(inp)
        
        assert output.final_text == "Simple answer."
        assert output.metadata.critic_summary == ""
        assert output.metadata.languages == []
    
    def test_empty_answer(self, agent, make_input):
        inp = make_input(answer_text="")
        output = agent.format(inp)
        
        assert output.final_text == ""
    
    def test_none_answer_text(self, agent):
        answer = Mock()
        answer.text = None
        
        inp = PostprocessInput(
            answer=answer,
            context_snippets=[],
            critic_feedback=Mock(notes=[]),
            preferences=Mock(include_critic_note=False, include_language_notes=False)
        )
        output = agent.format(inp)
        
        assert output.final_text == ""
    
    def test_multiline_answer_preserved(self, agent, make_input):
        inp = make_input(answer_text="Line 1\nLine 2\nLine 3")
        output = agent.format(inp)
        
        assert "Line 1\nLine 2\nLine 3" in output.final_text
```

#### 2. Critic Notes Tests

```python
class TestCriticNotes:
    
    def test_critic_notes_when_enabled(self, agent, make_input):
        inp = make_input(
            answer_text="Answer.",
            critic_notes=["Note 1", "Note 2"],
            include_critic=True
        )
        output = agent.format(inp)
        
        assert "---" in output.final_text
        assert "Critic notes:" in output.final_text
        assert "- Note 1" in output.final_text
        assert "- Note 2" in output.final_text
    
    def test_critic_notes_disabled(self, agent, make_input):
        inp = make_input(
            answer_text="Answer.",
            critic_notes=["Note 1"],
            include_critic=False
        )
        output = agent.format(inp)
        
        assert "Critic notes:" not in output.final_text
        assert "Note 1" not in output.final_text
    
    def test_critic_notes_enabled_but_empty(self, agent, make_input):
        inp = make_input(
            answer_text="Answer.",
            critic_notes=[],
            include_critic=True
        )
        output = agent.format(inp)
        
        # Should not add section if no notes
        assert "Critic notes:" not in output.final_text
    
    def test_critic_summary_in_metadata(self, agent, make_input):
        inp = make_input(
            answer_text="Answer.",
            critic_notes=["Note 1", "Note 2"],
            include_critic=True
        )
        output = agent.format(inp)
        
        assert output.metadata.critic_summary == "Note 1; Note 2"
    
    def test_critic_summary_empty_when_no_notes(self, agent, make_input):
        inp = make_input(answer_text="Answer.", critic_notes=[])
        output = agent.format(inp)
        
        assert output.metadata.critic_summary == ""
```

#### 3. Language Notes Tests

```python
class TestLanguageNotes:
    
    def test_language_notes_when_enabled(self, agent, make_input):
        snippets = [
            Mock(lang="en"),
            Mock(lang="fr"),
            Mock(lang="de"),
        ]
        inp = make_input(
            answer_text="Answer.",
            snippets=snippets,
            include_lang=True
        )
        output = agent.format(inp)
        
        assert "---" in output.final_text
        assert "Languages observed:" in output.final_text
        assert "de, en, fr" in output.final_text  # Sorted
    
    def test_language_notes_disabled(self, agent, make_input):
        snippets = [Mock(lang="en")]
        inp = make_input(
            answer_text="Answer.",
            snippets=snippets,
            include_lang=False
        )
        output = agent.format(inp)
        
        assert "Languages observed:" not in output.final_text
    
    def test_languages_deduplicated(self, agent, make_input):
        snippets = [
            Mock(lang="en"),
            Mock(lang="en"),
            Mock(lang="fr"),
            Mock(lang="en"),
        ]
        inp = make_input(
            answer_text="Answer.",
            snippets=snippets,
            include_lang=True
        )
        output = agent.format(inp)
        
        # Should only show each language once
        assert output.final_text.count("en") == 1
        assert "en, fr" in output.final_text
    
    def test_languages_sorted_alphabetically(self, agent, make_input):
        snippets = [
            Mock(lang="zh"),
            Mock(lang="ar"),
            Mock(lang="en"),
        ]
        inp = make_input(
            answer_text="Answer.",
            snippets=snippets,
            include_lang=True
        )
        output = agent.format(inp)
        
        assert "ar, en, zh" in output.final_text
    
    def test_none_languages_filtered(self, agent, make_input):
        snippets = [
            Mock(lang="en"),
            Mock(lang=None),
            Mock(lang="fr"),
            Mock(lang=""),
        ]
        inp = make_input(
            answer_text="Answer.",
            snippets=snippets,
            include_lang=True
        )
        output = agent.format(inp)
        
        assert "None" not in output.final_text
        assert "en, fr" in output.final_text
    
    def test_languages_in_metadata(self, agent, make_input):
        snippets = [
            Mock(lang="en"),
            Mock(lang="fr"),
        ]
        inp = make_input(
            answer_text="Answer.",
            snippets=snippets,
            include_lang=True
        )
        output = agent.format(inp)
        
        assert output.metadata.languages == ["en", "fr"]
    
    def test_empty_snippets_no_language_section(self, agent, make_input):
        inp = make_input(
            answer_text="Answer.",
            snippets=[],
            include_lang=True
        )
        output = agent.format(inp)
        
        assert "Languages observed:" not in output.final_text
    
    def test_none_snippets_handled(self, agent):
        answer = Mock()
        answer.text = "Answer."
        
        inp = PostprocessInput(
            answer=answer,
            context_snippets=None,
            critic_feedback=Mock(notes=[]),
            preferences=Mock(include_critic_note=False, include_language_notes=True)
        )
        output = agent.format(inp)
        
        assert "Languages observed:" not in output.final_text
```

#### 4. Combined Output Tests

```python
class TestCombinedOutput:
    
    def test_both_sections_enabled(self, agent, make_input):
        snippets = [Mock(lang="en"), Mock(lang="de")]
        inp = make_input(
            answer_text="Main answer.",
            snippets=snippets,
            critic_notes=["Quality note"],
            include_critic=True,
            include_lang=True
        )
        output = agent.format(inp)
        
        # Check order: answer -> critic -> languages
        assert output.final_text.index("Main answer.") < output.final_text.index("Critic notes:")
        assert output.final_text.index("Critic notes:") < output.final_text.index("Languages observed:")
    
    def test_section_separators(self, agent, make_input):
        snippets = [Mock(lang="en")]
        inp = make_input(
            answer_text="Answer.",
            snippets=snippets,
            critic_notes=["Note"],
            include_critic=True,
            include_lang=True
        )
        output = agent.format(inp)
        
        # Should have two separators
        assert output.final_text.count("---") == 2
```

#### 5. Edge Case Tests

```python
class TestEdgeCases:
    
    def test_special_characters_in_notes(self, agent, make_input):
        inp = make_input(
            answer_text="Answer.",
            critic_notes=["Note with <html> & 'quotes'"],
            include_critic=True
        )
        output = agent.format(inp)
        
        assert "<html>" in output.final_text
        assert "&" in output.final_text
    
    def test_unicode_languages(self, agent, make_input):
        snippets = [Mock(lang="日本語"), Mock(lang="العربية")]
        inp = make_input(
            answer_text="Answer.",
            snippets=snippets,
            include_lang=True
        )
        output = agent.format(inp)
        
        assert "日本語" in output.final_text or "العربية" in output.final_text
    
    def test_very_long_answer(self, agent, make_input):
        long_answer = "A" * 10000
        inp = make_input(answer_text=long_answer)
        output = agent.format(inp)
        
        assert len(output.final_text) >= 10000
    
    def test_many_critic_notes(self, agent, make_input):
        notes = [f"Note {i}" for i in range(100)]
        inp = make_input(
            answer_text="Answer.",
            critic_notes=notes,
            include_critic=True
        )
        output = agent.format(inp)
        
        assert "Note 0" in output.final_text
        assert "Note 99" in output.final_text
```

#### 6. Agent Interface Tests

```python
class TestAgentInterface:
    
    def test_name_property(self, agent):
        assert agent.name == "BasicPostProcessorAgent"
    
    def test_describe_method(self, agent):
        description = agent.describe()
        assert isinstance(description, str)
        assert len(description) > 0
    
    def test_role_attribute(self, agent):
        assert agent.role == "postprocess"
```

### Test Commands

```bash
# Run all postprocessor tests
pytest test_postprocess_basic_agent.py -v

# Run with coverage
pytest test_postprocess_basic_agent.py --cov=postprocess_basic_agent --cov-report=html

# Run specific test class
pytest test_postprocess_basic_agent.py::TestLanguageNotes -v
```

---

## Recommendations and Improvements

### Critical Issues

#### 1. No HTML/Markdown Escaping

**Problem:** Special characters in notes could break downstream rendering.

**Recommendation:** Add escaping option:

```python
import html

def format(self, inp: PostprocessInput) -> PostprocessOutput:
    # ...
    if inp.preferences.escape_html:
        for n in inp.critic_feedback.notes:
            lines.append(f"- {html.escape(n)}")
    else:
        for n in inp.critic_feedback.notes:
            lines.append(f"- {n}")
```

#### 2. Hardcoded Section Format

**Problem:** Output format is hardcoded (Markdown-style), may not suit all UIs.

**Recommendation:** Add format templates:

```python
@dataclass
class FormatTemplates:
    section_separator: str = "\n---\n"
    critic_header: str = "Critic notes:"
    critic_item: str = "- {note}"
    language_header: str = "Languages observed: "

class BasicPostProcessorAgent:
    def __init__(self, templates: FormatTemplates = None):
        self.templates = templates or FormatTemplates()
```

---

### High Priority Improvements

#### 3. Additional Metadata Sections

**Problem:** Only critic notes and languages supported.

**Recommendation:** Add more optional sections:

```python
@dataclass
class PostprocessPreferences:
    include_critic_note: bool = False
    include_language_notes: bool = False
    include_sources: bool = False        # NEW: List source documents
    include_confidence: bool = False     # NEW: Show confidence score
    include_timing: bool = False         # NEW: Show processing time

def format(self, inp: PostprocessInput) -> PostprocessOutput:
    # ... existing logic ...
    
    if inp.preferences.include_sources:
        sources = self._extract_sources(inp.context_snippets)
        lines.append("\n---\nSources:")
        for src in sources:
            lines.append(f"- {src}")
    
    if inp.preferences.include_confidence:
        confidence = 1.0 - inp.critic_feedback.hallucination_risk
        lines.append(f"\n---\nConfidence: {confidence:.0%}")
```

#### 4. Logging and Observability

**Problem:** No visibility into formatting operations.

**Recommendation:** Add structured logging:

```python
import logging
logger = logging.getLogger(__name__)

def format(self, inp: PostprocessInput) -> PostprocessOutput:
    # ... formatting logic ...
    
    logger.info(
        "postprocess_complete",
        extra={
            "answer_length": len(base),
            "final_length": len(final_text),
            "num_critic_notes": len(inp.critic_feedback.notes or []),
            "num_languages": len(langs),
            "sections_added": {
                "critic": inp.preferences.include_critic_note and bool(inp.critic_feedback.notes),
                "languages": inp.preferences.include_language_notes and bool(langs),
            }
        }
    )
    
    return output
```

#### 5. Output Format Options

**Problem:** Only plain text/Markdown output supported.

**Recommendation:** Support multiple output formats:

```python
class OutputFormat(Enum):
    PLAIN = "plain"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"

def format(self, inp: PostprocessInput, output_format: OutputFormat = OutputFormat.MARKDOWN) -> PostprocessOutput:
    if output_format == OutputFormat.HTML:
        return self._format_html(inp)
    elif output_format == OutputFormat.JSON:
        return self._format_json(inp)
    else:
        return self._format_markdown(inp)

def _format_html(self, inp: PostprocessInput) -> PostprocessOutput:
    html_parts = [f"<div class='answer'>{html.escape(base)}</div>"]
    
    if inp.preferences.include_critic_note and inp.critic_feedback.notes:
        html_parts.append("<hr><div class='critic-notes'>")
        html_parts.append("<h4>Critic notes:</h4><ul>")
        for n in inp.critic_feedback.notes:
            html_parts.append(f"<li>{html.escape(n)}</li>")
        html_parts.append("</ul></div>")
    
    # ...
```

---

### Medium Priority Improvements

#### 6. Configurable Preferences Defaults

**Problem:** No way to set default preferences.

**Recommendation:** Add configuration:

```python
@dataclass
class PostprocessConfig:
    default_include_critic: bool = False
    default_include_languages: bool = False
    max_critic_notes: int = 10
    max_languages: int = 10

class BasicPostProcessorAgent:
    def __init__(self, config: PostprocessConfig = None):
        self.config = config or PostprocessConfig()
```

#### 7. Language Code Normalization

**Problem:** Language codes may be inconsistent (en, EN, eng, english).

**Recommendation:** Normalize language codes:

```python
import pycountry  # or langcodes library

def _normalize_language(self, lang: str) -> str:
    """Normalize language code to ISO 639-1."""
    if not lang:
        return None
    
    lang = lang.lower().strip()
    
    # Map common variations
    lang_map = {
        "english": "en",
        "eng": "en",
        "french": "fr",
        "fra": "fr",
        # ...
    }
    
    return lang_map.get(lang, lang)
```

#### 8. Truncation for Long Sections

**Problem:** No limit on section length.

**Recommendation:** Add truncation with indication:

```python
def format(self, inp: PostprocessInput) -> PostprocessOutput:
    # ...
    
    if inp.preferences.include_critic_note and inp.critic_feedback.notes:
        notes = inp.critic_feedback.notes[:self.config.max_critic_notes]
        lines.append("\n---\nCritic notes:")
        for n in notes:
            lines.append(f"- {n}")
        
        if len(inp.critic_feedback.notes) > self.config.max_critic_notes:
            remaining = len(inp.critic_feedback.notes) - self.config.max_critic_notes
            lines.append(f"- ... and {remaining} more notes")
```

---

### Low Priority / Future Enhancements

#### 9. Custom Section Support

**Recommendation:** Allow arbitrary custom sections:

```python
@dataclass
class CustomSection:
    header: str
    items: List[str]
    enabled: bool = True

def format(self, inp: PostprocessInput, custom_sections: List[CustomSection] = None):
    # ... standard formatting ...
    
    if custom_sections:
        for section in custom_sections:
            if section.enabled and section.items:
                lines.append(f"\n---\n{section.header}")
                for item in section.items:
                    lines.append(f"- {item}")
```

#### 10. Template Engine Integration

**Recommendation:** Use Jinja2 for flexible templates:

```python
from jinja2 import Template

DEFAULT_TEMPLATE = """
{{ answer }}
{% if critic_notes %}

---
Critic notes:
{% for note in critic_notes %}
- {{ note }}
{% endfor %}
{% endif %}
{% if languages %}

---
Languages observed: {{ languages | join(", ") }}
{% endif %}
"""

def format(self, inp: PostprocessInput) -> PostprocessOutput:
    template = Template(self.config.template or DEFAULT_TEMPLATE)
    final_text = template.render(
        answer=inp.answer.text or "",
        critic_notes=inp.critic_feedback.notes if inp.preferences.include_critic_note else [],
        languages=langs if inp.preferences.include_language_notes else [],
    )
```

#### 11. Internationalization (i18n)

**Recommendation:** Support localized headers:

```python
TRANSLATIONS = {
    "en": {
        "critic_header": "Critic notes:",
        "language_header": "Languages observed:",
    },
    "es": {
        "critic_header": "Notas del crítico:",
        "language_header": "Idiomas observados:",
    },
    "fr": {
        "critic_header": "Notes du critique:",
        "language_header": "Langues observées:",
    },
}

def format(self, inp: PostprocessInput, locale: str = "en") -> PostprocessOutput:
    strings = TRANSLATIONS.get(locale, TRANSLATIONS["en"])
    # Use strings["critic_header"] instead of hardcoded text
```

---

## Usage Examples

### Basic Usage

```python
from postprocess_basic_agent import BasicPostProcessorAgent
from core.schemas import PostprocessInput, Answer, CriticOutput, ContextSnippet

# Initialize agent
agent = BasicPostProcessorAgent()

# Create input
answer = Answer(text="RAG combines retrieval with generation for better accuracy.")

snippets = [
    ContextSnippet(chunk_id="1", text="...", lang="en"),
    ContextSnippet(chunk_id="2", text="...", lang="fr"),
]

critic_feedback = CriticOutput(
    coverage_score=0.5,
    hallucination_risk=0.5,
    notes=["Low coverage of available context."],
    missing_topics=[],
    ambiguities=[],
    unsupported_claims=[]
)

preferences = PostprocessPreferences(
    include_critic_note=True,
    include_language_notes=True
)

inp = PostprocessInput(
    answer=answer,
    context_snippets=snippets,
    critic_feedback=critic_feedback,
    preferences=preferences
)

# Format
output = agent.format(inp)

print(output.final_text)
# RAG combines retrieval with generation for better accuracy.
#
# ---
# Critic notes:
# - Low coverage of available context.
#
# ---
# Languages observed: en, fr
```

### Pipeline Integration

```python
class RAGPipeline:
    def __init__(self):
        self.retriever = HybridRetrievalAgent()
        self.generator = LLMGeneratorAgent()
        self.critic = BasicCriticAgent()
        self.postprocessor = BasicPostProcessorAgent()
    
    def process(
        self, 
        query: str, 
        plan: Plan,
        include_diagnostics: bool = False
    ) -> str:
        # Retrieve, generate, evaluate
        retriever_output = self.retriever.retrieve(...)
        generator_output = self.generator.run(...)
        critic_output = self.critic.evaluate(...)
        
        # Format final output
        preferences = PostprocessPreferences(
            include_critic_note=include_diagnostics,
            include_language_notes=include_diagnostics
        )
        
        postprocess_output = self.postprocessor.format(PostprocessInput(
            answer=generator_output["answer"],
            context_snippets=retriever_output.snippets,
            critic_feedback=critic_output,
            preferences=preferences
        ))
        
        return postprocess_output.final_text
```

### Production vs Debug Mode

```python
def get_response(query: str, debug_mode: bool = False) -> Dict[str, Any]:
    # ... pipeline processing ...
    
    # Configure preferences based on mode
    preferences = PostprocessPreferences(
        include_critic_note=debug_mode,
        include_language_notes=debug_mode
    )
    
    output = postprocessor.format(PostprocessInput(
        answer=answer,
        context_snippets=snippets,
        critic_feedback=critic_feedback,
        preferences=preferences
    ))
    
    return {
        "response": output.final_text,
        "metadata": output.metadata if debug_mode else None
    }
```

### Accessing Metadata

```python
output = agent.format(inp)

# Access structured metadata
print(f"Critic Summary: {output.metadata.critic_summary}")
# "Low coverage of available context.; Answer is very short."

print(f"Languages: {output.metadata.languages}")
# ["de", "en", "fr"]

# Use metadata for analytics
log_response_metadata(
    critic_summary=output.metadata.critic_summary,
    language_count=len(output.metadata.languages),
    has_quality_issues=bool(output.metadata.critic_summary)
)
```

---

## Appendix

### Glossary

| Term | Definition |
|------|------------|
| **Postprocessing** | Final formatting before user delivery |
| **Critic Notes** | Quality feedback from the critic agent |
| **Language Notes** | Observed languages in source documents |
| **Metadata** | Structured data about the response |

### Output Section Reference

| Section | Condition | Format |
|---------|-----------|--------|
| Answer | Always | Plain text |
| Critic Notes | `include_critic_note AND notes exist` | Markdown list |
| Languages | `include_language_notes AND langs exist` | Comma-separated |

### Preferences Reference

| Preference | Type | Default | Effect |
|------------|------|---------|--------|
| `include_critic_note` | bool | False | Add critic notes section |
| `include_language_notes` | bool | False | Add languages section |

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Initial | Basic formatting with critic and language notes |

### References

- Radiant Repository: https://github.com/dshipley71/radiant
- Related Files: `critic_basic_agent.py`, `core/schemas.py`, `orchestrator.py`

---

*Document generated for the Radiant RAG Pipeline project. For questions or contributions, please refer to the project repository.*
