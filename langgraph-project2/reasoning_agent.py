from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOllama(model="llama3.1:8b")


# ── 1. State ─────────────────────────────────────────────────────────────────
# Notice it now has 4 keys. Each node is responsible for filling one of them.

class State(TypedDict):
    question: str    # set by the user at invoke time
    plan: str        # filled by planner node
    reasoning: str   # filled by reasoner node
    answer: str      # filled by summarizer node


# ── 2. Nodes ─────────────────────────────────────────────────────────────────

def planner(state: State) -> dict:
    """
    Takes the raw question and produces a plan:
    a short numbered list of things to think about.
    """
    print("\n[1/3] Planning...")

    messages = [
        SystemMessage(content=(
            "You are a planning assistant. Given a question, output a short numbered list "
            "of 3-4 key points or sub-questions that need to be considered to answer it well. "
            "Be concise. Only output the list, nothing else."
        )),
        HumanMessage(content=state["question"])
    ]

    response = llm.invoke(messages)
    print(f"    Plan:\n{response.content}")
    return {"plan": response.content}


def reasoner(state: State) -> dict:
    """
    Takes the question AND the plan, then thinks through each point carefully.
    Notice it reads state["plan"] — that's only possible because planner ran first.
    """
    print("\n[2/3] Reasoning...")

    messages = [
        SystemMessage(content=(
            "You are a careful reasoning assistant. You will be given a question and a plan. "
            "Work through each point in the plan step by step. "
            "Think out loud. Show your reasoning clearly."
        )),
        HumanMessage(content=(
            f"Question: {state['question']}\n\n"
            f"Plan to follow:\n{state['plan']}"
        ))
    ]

    response = llm.invoke(messages)
    print(f"    Reasoning (first 120 chars): {response.content[:120]}...")
    return {"reasoning": response.content}


def summarizer(state: State) -> dict:
    """
    Takes the question, plan AND reasoning, then writes a clean final answer.
    It can see everything — it just distills it.
    """
    print("\n[3/3] Summarizing...")

    messages = [
        SystemMessage(content=(
            "You are a summarizer. You will be given a question, a plan, and detailed reasoning. "
            "Write a clear, concise final answer (2-4 sentences). "
            "Do not repeat the plan or reasoning — just give the answer."
        )),
        HumanMessage(content=(
            f"Question: {state['question']}\n\n"
            f"Plan:\n{state['plan']}\n\n"
            f"Reasoning:\n{state['reasoning']}"
        ))
    ]

    response = llm.invoke(messages)
    return {"answer": response.content}


# ── 3. Build the Graph ───────────────────────────────────────────────────────

graph_builder = StateGraph(State)

# Register all three nodes
graph_builder.add_node("planner", planner)
graph_builder.add_node("reasoner", reasoner)
graph_builder.add_node("summarizer", summarizer)

# Wire them in sequence
graph_builder.add_edge(START, "planner")
graph_builder.add_edge("planner", "reasoner")
graph_builder.add_edge("reasoner", "summarizer")
graph_builder.add_edge("summarizer", END)

graph = graph_builder.compile()


# ── 4. Run it ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Multi-step reasoning agent. Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        result = graph.invoke({"question": question})
        print(f"\nAgent: {result['answer']}\n")