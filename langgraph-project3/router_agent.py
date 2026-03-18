from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOllama(model="llama3.1:8b")


# ── 1. State ─────────────────────────────────────────────────────────────────
# "route" is the key that the classifier writes and the routing function reads.
# Literal[] is just a type hint — it documents the allowed values.

class State(TypedDict):
    question: str
    route: Literal["factual", "creative", "unknown"]  # written by classifier
    answer: str                                         # written by whichever node runs


# ── 2. Nodes ─────────────────────────────────────────────────────────────────

def classifier(state: State) -> dict:
    """
    Reads the question and decides what kind of task it is.
    Writes a single word to state["route"].
    """
    print("\n[Classifier] Deciding route...")

    messages = [
        SystemMessage(content="""You are a query classifier. 
Classify the user's question into exactly one category:
- "factual"  → questions with a clear, researched answer (science, history, definitions, how-things-work)
- "creative" → open-ended requests (write a story, generate ideas, brainstorm, imagine)
- "unknown"  → unclear, ambiguous, or completely off-topic requests

Respond with ONLY one word: factual, creative, or unknown. Nothing else."""),
        HumanMessage(content=state["question"])
    ]

    response = llm.invoke(messages)
    # Clean up — model sometimes adds punctuation or spaces
    route = response.content.strip().lower().strip(".,!?\"'")

    # Safety net: if the model returns something unexpected, default to unknown
    if route not in ("factual", "creative", "unknown"):
        route = "unknown"

    print(f"[Classifier] Route decided: '{route}'")
    return {"route": route}


def factual_node(state: State) -> dict:
    """Handles factual questions — precise, grounded answers."""
    print("[Node] Taking factual path...")

    messages = [
        SystemMessage(content=(
            "You are a knowledgeable assistant. Answer factual questions accurately and concisely. "
            "Stick to what is known. If you are uncertain, say so."
        )),
        HumanMessage(content=state["question"])
    ]

    response = llm.invoke(messages)
    return {"answer": response.content}


def creative_node(state: State) -> dict:
    """Handles creative tasks — imaginative, expressive responses."""
    print("[Node] Taking creative path...")

    messages = [
        SystemMessage(content=(
            "You are a creative assistant. Be imaginative, original, and engaging. "
            "For creative writing, use vivid language. For brainstorming, generate diverse ideas."
        )),
        HumanMessage(content=state["question"])
    ]

    response = llm.invoke(messages)
    return {"answer": response.content}


def fallback_node(state: State) -> dict:
    """Handles unclear or unknown requests gracefully."""
    print("[Node] Taking fallback path...")

    return {
        "answer": (
            f"I'm not sure how to categorize your request: '{state['question']}'\n\n"
            "Could you clarify? For example:\n"
            "- If you want a factual answer, try: 'Explain how X works'\n"
            "- If you want something creative, try: 'Write a short story about X'\n"
            "- Or just rephrase and I'll do my best!"
        )
    }


# ── 3. The Routing Function ───────────────────────────────────────────────────
# This is NOT a node. It's a plain function that LangGraph calls to decide
# which node to go to next. It reads state and returns a string key.

def route_question(state: State) -> str:
    """
    Called after classifier runs.
    Returns a string that maps to a node name via the conditional edges dict.
    """
    return state["route"]   # "factual", "creative", or "unknown"


# ── 4. Build the Graph ───────────────────────────────────────────────────────

graph_builder = StateGraph(State)

# Register all nodes
graph_builder.add_node("classifier", classifier)
graph_builder.add_node("factual_node", factual_node)
graph_builder.add_node("creative_node", creative_node)
graph_builder.add_node("fallback_node", fallback_node)

# Normal edge: START → classifier (always)
graph_builder.add_edge(START, "classifier")

# Conditional edge: after classifier, call route_question to decide where to go
graph_builder.add_conditional_edges(
    "classifier",       # source node
    route_question,     # routing function — receives state, returns a string
    {                   # map returned string → node name
        "factual":  "factual_node",
        "creative": "creative_node",
        "unknown":  "fallback_node",
    }
)

# All three destination nodes go to END
graph_builder.add_edge("factual_node", END)
graph_builder.add_edge("creative_node", END)
graph_builder.add_edge("fallback_node", END)

graph = graph_builder.compile()


# ── 5. Run it ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Router agent ready. Ask me anything. Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        result = graph.invoke({"question": question})

        print(f"  [routed to: {result['route']}]")
        print(f"\nAgent: {result['answer']}\n")