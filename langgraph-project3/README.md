# LangGraph Router Agent

This project demonstrates a simple conversational agent built using **LangGraph** and **LangChain**, powered by a local LLM via **Ollama**. It showcases how to implement conditional routing within a graph by classifying user questions and directing them to specialized processing nodes.

## Overview

The agent takes a user's question and intelligently decides how to answer it based on the nature of the query. It uses a primary classifier to categorize the input and then routes it down one of three specific paths:
- **Factual**: For grounded, precise questions (e.g., science, history, definitions).
- **Creative**: For open-ended, imaginative requests (e.g., stories, brainstorming).
- **Fallback (Unknown)**: For ambiguous or unclear requests.

## Key Concepts Demonstrated

- **State Management (`TypedDict`)**: Defining a structured state that is passed between nodes.
- **Classification Node**: Using an LLM prompting strategy specifically designed to output a single categorical word.
- **Conditional Routing (`add_conditional_edges`)**: Dynamically deciding the next node in the graph based on the output of a previous node.
- **Specialized Execution Nodes**: Having different system prompts depending on the logical path taken.
- **Local LLM Integration**: Utilizing `llama3.1:8b` through `ChatOllama` for local, private inference.

## Project Structure

- `router_agent.py`: The core script that defines the state, all nodes (classifier, factual, creative, fallback), the routing logic, graph construction, and an interactive command-line loop.

## How It Works

1. **State Definition**: The `State` dictionary tracks the `question` (input), `route` (classification result), and `answer` (final output).
2. **`classifier` Node**: The initial node that receives the question. It prompts the LLM to categorize the question into exactly one of three categories: "factual", "creative", or "unknown". It cleans the output and sets the `route` state.
3. **Routing logic (`route_question`)**: A simple function that reads the `route` from the state. LangGraph uses the return value of this function in its conditional edges mapping to determine the next node.
4. **Execution Nodes**:
   - **`factual_node`**: Receives a strict system prompt to be concise and accurate.
   - **`creative_node`**: Receives a prompt to be imaginative and engaging.
   - **`fallback_node`**: Provides a graceful degradation by asking the user to clarify their request without calling the LLM again.
5. **Graph Compilation**: The graph is constructed with a `START` edge to the classifier, conditional edges from the classifier to the specific nodes, and edges from all specific nodes to the `END`.

## Prerequisites

To run this project, you need:

- Python 3.8+
- The following Python packages:
  - `langgraph`
  - `langchain-ollama`
  - `langchain-core`
- **Ollama** installed and running on your local machine.
- The `llama3.1:8b` model pulled in Ollama (`ollama run llama3.1:8b`).

## Usage

Run the agent via the terminal:

```bash
python router_agent.py
```

Ask the agent questions and observe the console output to see which path the classifier determines is most appropriate for your prompt! To exit, simply type `quit`, `exit`, or `q`.
