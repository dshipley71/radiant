# RadiantManagerAgent Documentation

Technical reference for the AutoGen-compatible Radiant manager agent.

---

## Overview

`RadiantManagerAgent` is an AutoGen `AssistantAgent` subclass that wraps the Radiant Agentic RAG pipeline. It provides a conversational interface where the LLM automatically delegates question answering to Radiant via the `radiant_tool` function.

**Module Location:** `radiant_manager_agent.py` (project root)

---

## Class Definition

```python
class RadiantManagerAgent(AssistantAgent):
    """
    Manager agent that uses a single Radiant tool to answer user queries.
    """
    
    def __init__(
        self,
        model_client: OpenAIChatCompletionClient,
        *,
        name: str = "radiant_manager",
        model_client_stream: bool = True,
    ) -> None:
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_client` | `OpenAIChatCompletionClient` | required | AutoGen model client for LLM calls |
| `name` | `str` | `"radiant_manager"` | Agent identifier name |
| `model_client_stream` | `bool` | `True` | Enable streaming responses |

---

## System Message

The agent is configured with the following system prompt:

```
You are RadiantManagerAgent, a specialized agent that delegates question
answering to the Radiant Agentic RAG pipeline via the `radiant_tool` function.

- Always, for each user query, call the `radiant_tool(query, config_path?, history?)` tool.
- The tool returns a JSON object with:
    {
        "answer_text": <final answer string or null>,
        "meta": {...}  // detailed metadata: router, plan, retrieval, citations, etc.
    }

Behavior guidelines:
- If `answer_text` is non-null and non-empty, use it as the core of your reply.
- You MAY lightly rephrase or clean up the wording, but do not change the meaning.
- If `answer_text` is null or empty, inspect `meta` to see if:
    - there is an error or guardrail rejection you should explain to the user, or
    - there are context snippets you can summarize.
- Do NOT dump the entire `meta` object to the user. It is for internal debugging.
- You MAY include a small, human-friendly debug section (like "Sources" or a short 
  summary of what was retrieved), but keep it concise and readable.
- If Radiant indicates the query is unsafe or cannot be answered, explain that clearly 
  and politely.
- For follow-up questions, you can pass conversation history to help resolve pronouns 
  and references.

Be concise and helpful in your final responses.
```

---

## Usage Examples

### Basic Usage

```python
import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from radiant_manager_agent import RadiantManagerAgent

async def main():
    # Create model client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key="sk-your-key-here"
    )
    
    try:
        # Create agent
        agent = RadiantManagerAgent(model_client=model_client)
        
        # Run query
        result = await agent.run(task="What is hierarchical RAG?")
        
        # Get response
        if result.messages:
            print(result.messages[-1].content)
    finally:
        await model_client.close()

asyncio.run(main())
```

### Custom Name

```python
agent = RadiantManagerAgent(
    model_client=model_client,
    name="knowledge_assistant"
)
```

### Disable Streaming

```python
agent = RadiantManagerAgent(
    model_client=model_client,
    model_client_stream=False
)
```

### Different LLM Models

```python
# Using GPT-4
model_client = OpenAIChatCompletionClient(
    model="gpt-4",
    api_key="sk-..."
)

# Using Azure OpenAI
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

model_client = AzureOpenAIChatCompletionClient(
    azure_deployment="my-deployment",
    azure_endpoint="https://my-resource.openai.azure.com/",
    api_key="..."
)

agent = RadiantManagerAgent(model_client=model_client)
```

---

## Conversation Flow

### Single Query

```
User: "What is MCP?"
        │
        ▼
RadiantManagerAgent
        │
        ▼
LLM decides to call radiant_tool(query="What is MCP?")
        │
        ▼
radiant_tool executes Radiant pipeline
        │
        ▼
Returns: {"answer_text": "MCP is...", "meta": {...}}
        │
        ▼
LLM formats response using answer_text
        │
        ▼
User receives: "MCP (Model Context Protocol) is..."
```

### Multi-Turn Conversation

```
User: "What is RAG?"
        │
        ▼
Agent calls: radiant_tool(query="What is RAG?")
Agent responds: "RAG stands for..."
        │
        ▼
User: "How does it handle long documents?"
        │
        ▼
Agent calls: radiant_tool(
    query="How does it handle long documents?",
    history=[
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "RAG stands for..."}
    ]
)
Agent responds: "RAG handles long documents by..."
```

---

## Response Handling

### Successful Response

When `answer_text` is available:

```python
# Tool returns
{"answer_text": "Hierarchical RAG is a technique that...", "meta": {...}}

# Agent responds
"Hierarchical RAG is a technique that organizes documents into multiple 
levels of granularity..."
```

### No Answer Available

When `answer_text` is `None`:

```python
# Tool returns
{"answer_text": None, "meta": {"guardrail": {"blocked": True, "reason": "..."}}}

# Agent responds
"I wasn't able to find a direct answer to your question. The query may have 
been outside the scope of the available knowledge base."
```

### Error Handling

When Radiant reports an error:

```python
# Tool returns with error in meta
{"answer_text": None, "meta": {"error": "Connection timeout"}}

# Agent responds
"I encountered an issue while processing your question. Please try again 
or rephrase your query."
```

---

## Integration with AutoGen Teams

### With UserProxyAgent

```python
from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat

user = UserProxyAgent(name="user")
radiant = RadiantManagerAgent(model_client=model_client)

team = RoundRobinGroupChat(
    participants=[user, radiant],
    max_turns=10
)

result = await team.run(task="Research hierarchical RAG techniques")
```

### With Multiple Agents

```python
from autogen_agentchat.agents import AssistantAgent

# Radiant for knowledge retrieval
knowledge_agent = RadiantManagerAgent(
    model_client=model_client,
    name="knowledge_retriever"
)

# Separate agent for analysis
analyst = AssistantAgent(
    name="analyst",
    model_client=model_client,
    system_message="You analyze information and provide insights."
)

# Combine in team
team = RoundRobinGroupChat(
    participants=[knowledge_agent, analyst]
)
```

---

## Customization

### Extending the Agent

```python
class CustomRadiantAgent(RadiantManagerAgent):
    def __init__(self, model_client, custom_tools=None, **kwargs):
        super().__init__(model_client, **kwargs)
        
        # Add additional tools
        if custom_tools:
            self._tools.extend(custom_tools)
    
    async def on_messages(self, messages, cancellation_token):
        # Custom pre-processing
        processed = self._preprocess(messages)
        
        # Call parent
        return await super().on_messages(processed, cancellation_token)
```

### Custom System Message

```python
class VerboseRadiantAgent(RadiantManagerAgent):
    def __init__(self, model_client, **kwargs):
        super().__init__(model_client, **kwargs)
        
        # Override system message
        self._system_messages = [
            SystemMessage(content="""
            You are a detailed research assistant. When using radiant_tool:
            - Always include source citations
            - Provide confidence levels
            - Suggest follow-up questions
            """)
        ]
```

---

## Debugging

### Inspect Tool Calls

```python
result = await agent.run(task="What is RAG?")

# Check all messages including tool calls
for msg in result.messages:
    print(f"{msg.__class__.__name__}: {msg}")
```

### Access Raw Tool Output

```python
# After running agent, check tool results in messages
for msg in result.messages:
    if hasattr(msg, 'content') and isinstance(msg.content, list):
        for item in msg.content:
            if hasattr(item, 'result'):
                print(f"Tool result: {item.result}")
```

---

## Best Practices

1. **Always close model client:**
   ```python
   try:
       result = await agent.run(task="...")
   finally:
       await model_client.close()
   ```

2. **Use appropriate models:** GPT-4o-mini is cost-effective for most queries; use GPT-4 for complex reasoning.

3. **Handle timeouts:** Radiant pipeline can take time; consider timeout settings.

4. **Monitor token usage:** The `meta` object can be large; the agent is instructed not to dump it to users.

---

## Related Documentation

- [AutoGenIntegration_Documentation.md](AutoGenIntegration_Documentation.md) - Integration overview
- [RadiantAutogenWrappers_Documentation.md](RadiantAutogenWrappers_Documentation.md) - Wrapper module details
- [Orchestrator_Documentation.md](Orchestrator_Documentation.md) - Radiant pipeline internals
