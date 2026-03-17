from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage


# ── 1. Define the State ──────────────────────────────────────────────────────
# This is the typed dictionary that flows through your graph.
# Every node reads from it and can write new keys back to it.

class State(TypedDict):
    question: str   # the user's input
    answer: str     # the model's response (added by our node)


# ── 2. Define the Node ───────────────────────────────────────────────────────
# A node is just a function: State in → State out.
# It must return a dict with the keys it wants to update.

llm = ChatOllama(model="llama3.1:8b")  # change to any model you have pulled

# def call_ollama(state: State) -> dict:
#     """Send the question to Ollama and store the answer in state."""
#     print(f"\n[Node] Received question: {state['question']}")
    
#     response = llm.invoke(state["question"])
#     answer = response.content
    
#     print(f"[Node] Got answer: {answer[:80]}...")  # print first 80 chars
#     return {"answer": answer}

def call_ollama(state: State) -> dict:
    messages = [
        SystemMessage(content="You are a concise assistant. Answer in 2-3 sentences max."),
        HumanMessage(content=state["question"])
    ]
    response = llm.invoke(messages)
    return {"answer": response.content}


# ── 3. Build the Graph ───────────────────────────────────────────────────────
# Create a StateGraph, add your node, then wire the edges.

graph_builder = StateGraph(State)

# Add the node — give it a name and point to the function
graph_builder.add_node("call_ollama", call_ollama)

# Wire the edges: START → call_ollama → END
graph_builder.add_edge(START, "call_ollama")
graph_builder.add_edge("call_ollama", END)

# Compile turns the builder into a runnable graph
graph = graph_builder.compile()


# ── 4. Run it ────────────────────────────────────────────────────────────────

# Add this at the bottom of qa_agent.py, replacing the __main__ block

if __name__ == "__main__":
    print("Q&A Agent ready. Type 'quit' to exit.\n")
    
    while True:
        question = input("You: ").strip()
        
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        
        result = graph.invoke({"question": question})
        print(f"\nAgent: {result['answer']}\n")