# BasicPostProcessorAgent Documentation

Technical reference for the Radiant RAG pipeline post-processor agent.

---

## Overview

The `BasicPostProcessorAgent` formats the final answer by appending critic notes and language observations based on user preferences.

**Module Location:** `agents/postprocess.py`

**Interface:** `PostProcessorAgent` (from `core.interfaces`)

---

## Class Definition

```python
class BasicPostProcessorAgent(PostProcessorAgent):
    """Formats final answer with critic + language notes."""
    
    role = "postprocess"
    
    @property
    def name(self) -> str:
        return "BasicPostProcessorAgent"
    
    def describe(self) -> str:
        return "Formats final answer with critic + language notes."
    
    def format(self, inp: PostprocessInput) -> PostprocessOutput:
        ...
```

---

## Functionality

### Main Method: `format()`

**Input:** `PostprocessInput`
- `ctx`: Request context
- `query`: User's query
- `answer`: Generated answer
- `critic_feedback`: Critic evaluation results
- `context_snippets`: Retrieved context snippets
- `preferences`: Formatting preferences

**Output:** `PostprocessOutput`
- `final_text`: Formatted answer string
- `metadata`: Processing metadata

---

## Implementation

```python
def format(self, inp: PostprocessInput) -> PostprocessOutput:
    base = inp.answer.text or ""
    lines: List[str] = [base]

    langs = sorted({cs.lang for cs in inp.context_snippets if cs.lang}) if inp.context_snippets else []

    if inp.preferences.include_critic_note and inp.critic_feedback.notes:
        lines.append("\n---\nCritic notes:")
        for n in inp.critic_feedback.notes:
            lines.append(f"- {n}")

    if inp.preferences.include_language_notes and langs:
        lines.append("\n---\nLanguages observed: " + ", ".join(langs))

    final_text = "\n".join(lines)
    meta = PostprocessMetadata(
        critic_summary="; ".join(inp.critic_feedback.notes) if inp.critic_feedback.notes else "",
        languages=langs,
    )
    return PostprocessOutput(final_text=final_text, metadata=meta)
```

---

## Output Format

```
<answer text>

---
Critic notes:
- <note 1>
- <note 2>

---
Languages observed: en, fr, de
```

---

## Preferences

| Preference | Type | Default | Effect |
|------------|------|---------|--------|
| `format` | str | `"markdown"` | Output format (currently unused) |
| `include_critic_note` | bool | `True` | Append critic notes |
| `include_language_notes` | bool | `True` | Append observed languages |

---

## Related Documentation

- [CoreInterfaces_Documentation.md](CoreInterfaces_Documentation.md) - `PostProcessorAgent` interface
- [CoreSchemas_Documentation.md](CoreSchemas_Documentation.md) - `PostprocessInput`, `PostprocessOutput` schemas
- [BasicCriticAgent_Documentation.md](BasicCriticAgent_Documentation.md) - Critic feedback source
