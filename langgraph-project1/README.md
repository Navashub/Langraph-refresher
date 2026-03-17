# LangGraph Project 1 — Q&A Agent with Ollama

A minimal, well-commented introduction to [LangGraph](https://github.com/langchain-ai/langgraph) that demonstrates how to build a single-node Q&A agent powered by a **local LLM via [Ollama](https://ollama.com)**. This project is designed as a learning reference for the core LangGraph concepts: State, Nodes, Edges, and Graph compilation.

---

## What This Project Does

When you run `qa_agent.py`, you get an interactive command-line Q&A session:

```
Q&A Agent ready. Type 'quit' to exit.

You: What is LangGraph?

Agent: LangGraph is a library for building stateful, multi-actor applications
with LLMs using a graph-based execution model. It extends LangChain with
explicit state management and cyclic graph support.
```

Under the hood, every question you type is passed through a compiled LangGraph graph that routes it to a local Ollama model and returns the answer.

---

## Project Structure

```
langgraph-project1/
├── qa_agent.py        # Core logic — the LangGraph Q&A agent
├── main.py            # Placeholder entry point
├── pyproject.toml     # Project metadata and dependencies
├── .python-version    # Pins Python version to 3.12
└── uv.lock            # Locked dependency tree (managed by uv)
```

---

## Core Concepts Demonstrated

This project walks through the four fundamental building blocks of any LangGraph application.

### 1. State — the shared data container

A `TypedDict` that defines every piece of data flowing through the graph. Each node reads from it and can write updated values back.

```python
class State(TypedDict):
    question: str   # set by the caller before the graph runs
    answer: str     # written by the call_ollama node
```

Think of `State` as the "memory" that gets passed between nodes. You declare upfront exactly what keys exist and what type each holds.

### 2. Node — a single unit of work

A node is a plain Python function that accepts the current `State` and returns a dictionary of keys to update.

```python
def call_ollama(state: State) -> dict:
    messages = [
        SystemMessage(content="You are a concise assistant. Answer in 2-3 sentences max."),
        HumanMessage(content=state["question"])
    ]
    response = llm.invoke(messages)
    return {"answer": response.content}
```

Key points:
- The node receives the **full** current state.
- It only needs to return the keys it **changed** — LangGraph merges the returned dict back into the state automatically.
- The `SystemMessage` constrains the model to give short, concise answers.

### 3. Graph — wiring nodes together with edges

A `StateGraph` is the container. You add nodes and then connect them with directed edges.

```python
graph_builder = StateGraph(State)
graph_builder.add_node("call_ollama", call_ollama)

graph_builder.add_edge(START, "call_ollama")   # entry point
graph_builder.add_edge("call_ollama", END)     # exit point
```

The graph for this project is a simple linear pipeline:

```
 START  →  call_ollama  →  END
```

`START` and `END` are special sentinel nodes provided by LangGraph that mark where execution begins and finishes.

### 4. Compilation — turning the builder into a runnable

```python
graph = graph_builder.compile()
```

`compile()` validates the graph structure and returns an executable object. Once compiled, the graph is invoked with an initial state dict:

```python
result = graph.invoke({"question": "What is Python?"})
print(result["answer"])
```

---

## How the Interactive Loop Works

The `__main__` block in `qa_agent.py` runs a simple REPL:

1. Prompt the user for a question.
2. Pass `{"question": <input>}` to `graph.invoke()`.
3. LangGraph runs the graph: `START → call_ollama → END`.
4. The `call_ollama` node sends the question to the local Ollama model with a system prompt.
5. The model's response is stored in `state["answer"]`.
6. The final state dict is returned by `invoke()`, and the answer is printed.
7. Loop back to step 1 until the user types `quit`, `exit`, or `q`.

---

## Tech Stack

| Component | Purpose |
|---|---|
| **LangGraph** `>=1.1.2` | Graph execution engine, state management |
| **LangChain Ollama** `>=1.0.1` | LangChain integration for local Ollama models |
| **Ollama** (external) | Runs the LLM locally on your machine |
| **llama3.1:8b** | Default model — swappable for any pulled Ollama model |
| **Python 3.12** | Runtime |
| **uv** | Fast Python package and project manager |

---

## Prerequisites

1. **Python 3.12** installed.
2. **uv** installed — [install guide](https://docs.astral.sh/uv/getting-started/installation/).
3. **Ollama** installed and running — [install guide](https://ollama.com/download).
4. The `llama3.1:8b` model pulled locally:

```bash
ollama pull llama3.1:8b
```

You can use any other model you have pulled by changing the `model=` value in `qa_agent.py`:

```python
llm = ChatOllama(model="llama3.1:8b")  # change to e.g. "mistral", "phi3", etc.
```

---

## Setup & Running

```bash
# 1. Clone or navigate to the project directory
cd langgraph-project1

# 2. Install dependencies (uv will create a .venv automatically)
uv sync

# 3. Run the Q&A agent
uv run python qa_agent.py
```

To exit the interactive session, type `quit`, `exit`, or `q`.

---

## Extending This Project

Some natural next steps to explore more LangGraph features:

- **Add more nodes** — e.g., a validation node before the LLM call, or a formatting node after.
- **Add conditional edges** — route to different nodes based on the content of the question (`add_conditional_edges`).
- **Add memory / conversation history** — replace the `question`/`answer` state with a `messages: list` key to accumulate a full chat history.
- **Add tools** — give the agent access to web search, calculators, or code execution using LangChain tools.
- **Persistence** — use LangGraph's built-in checkpointers to save graph state between runs.