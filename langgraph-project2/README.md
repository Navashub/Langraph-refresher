# LangGraph Multi-Step Reasoning Agent

## Overview

This project implements a **Multi-Step Reasoning Agent** using [LangGraph](https://python.langchain.com/docs/langgraph) and [LangChain](https://python.langchain.com/), powered by a local Ollama model (`llama3.1:8b`). The AI reads a user prompt and goes through a multi-stage cognitive process — **Planning**, **Reasoning**, and **Summarizing** — mimicking how a human would break down and solve a complex problem.

Everything operates locally using the Ollama service, making it completely private and relatively fast, depending on your machine's hardware.

---

## Prerequisites

To run this code properly, you'll need the following installed:

1. **Python 3.8+**
2. **Packages**: `langchain-core`, `langchain-ollama`, `langgraph`, `typing`
   ```bash
   pip install langchain-core langchain-ollama langgraph
   ```
3. **Ollama**: You must have [Ollama](https://ollama.com/) installed and running on your local machine.
4. **Llama 3.1 Model**: You need to pull the specific model the script uses. Open your terminal and run:
   ```bash
   ollama run llama3.1:8b
   ```

---

## Application Architecture

The core of the application is a directed graph built using `StateGraph`.

### The State (Memory)
The `State` object acts as the shared memory for the application during a single execution. It defines four variables:
- `question`: The initial input provided by the user.
- `plan`: A structured outline of how to solve the question.
- `reasoning`: Detailed, step-by-step thought process following the plan.
- `answer`: The final, distilled answer presented back to the user.

### Graph Flow
The application processes inputs sequentially through three main **Nodes** (steps):
```text
[ START ] 
   ↓
[ planner ]  → analyzes the question and creates a plan.
   ↓
[ reasoner ] → follows the plan and thinks out loud.
   ↓
[ summarizer ] → summarizes the complex reasoning into a final answer.
   ↓
[ END ]
```

---

## Step-by-Step Code Explanation

### 1. Imports and Model Initialization
```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOllama(model="llama3.1:8b")
```
The necessary LangChain and LangGraph components are imported. We then initialize our LLM wrapper pointing to our local `llama3.1:8b` via Ollama.

### 2. Defining the State
```python
class State(TypedDict):
    question: str
    plan: str
    reasoning: str
    answer: str
```
A `TypedDict` defines the schema of our graph's State. Think of it as a dictionary that gets passed from node to node. Each node reads from it and returns an update for specific keys.

### 3. Creating the Nodes
Nodes in LangGraph are simply Python functions that take the `state` as an argument and return a dictionary containing the state updates.

#### A. Planner Node
```python
def planner(state: State) -> dict:
```
- **Goal:** Takes the `question` from the state. 
- **Action:** Uses a `SystemMessage` to instruct the LLM to be a "planning assistant" that generates a 3-4 point numbered list.
- **Output:** Returns `{"plan": ...}` which updates the `plan` key in our State.

#### B. Reasoner Node
```python
def reasoner(state: State) -> dict:
```
- **Goal:** Takes both the `question` and the freshly generated `plan`.
- **Action:** Instructs the LLM to carefully work through each point of the plan step-by-step.
- **Output:** Returns `{"reasoning": ...}` updating the `reasoning` key.

#### C. Summarizer Node
```python
def summarizer(state: State) -> dict:
```
- **Goal:** Takes the `question`, `plan`, and detailed `reasoning`.
- **Action:** Instructs the LLM to write a concise final answer in 2-4 sentences without repeating everything it just thought about.
- **Output:** Returns `{"answer": ...}` updating the final component of the state.

### 4. Building the Graph
```python
graph_builder = StateGraph(State)

graph_builder.add_node("planner", planner)
graph_builder.add_node("reasoner", reasoner)
graph_builder.add_node("summarizer", summarizer)

graph_builder.add_edge(START, "planner")
graph_builder.add_edge("planner", "reasoner")
graph_builder.add_edge("reasoner", "summarizer")
graph_builder.add_edge("summarizer", END)

graph = graph_builder.compile()
```
Here, we instantiate `StateGraph` using our `State` class. We add all three of our functional nodes, and then carefully wire them together using `add_edge()`. Finally, `.compile()` binds everything together into a usable/invokable application.

### 5. Running the Interactive CLI
```python
if __name__ == "__main__":
    ...
```
A continuous `while True` loop is set up to allow the user to keep asking questions. When the user types a question, it is passed into the initial state via `graph.invoke({"question": question})`. 

The graph runs through Planner → Reasoner → Summarizer, prints progress to the console during transit, and finally outputs the distilled `"answer"`.

---

## How to Run

1. Open a terminal in the project directory.
2. Ensure Ollama is running in the background.
3. Run the script:
   ```bash
   python reasoning_agent.py
   ```
4. Ask your question at the `You:` prompt and observe as the AI plans, reasons, and eventually answers! Type `quit`, `exit`, or `q` to terminate the application.
