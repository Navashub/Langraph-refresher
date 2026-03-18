from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


llm = ChatOllama(model="llama3.1:8b")

# The system prompt gives your assistant a persistent personality
SYSTEM_PROMPT = SystemMessage(content="""You are a helpful, friendly personal assistant. 
You remember everything said in this conversation.
Be concise but warm. If the user refers to something mentioned earlier, use that context naturally.""")


# ── 1. The Node ───────────────────────────────────────────────────────────────
# With MessagesState we don't need to define State ourselves.
# state["messages"] is the full conversation history as a list.

def chat(state: MessagesState) -> dict:
    """
    Prepend the system prompt, send the full history to Ollama,
    append the response. That's the whole memory mechanism.
    """
    # Build the full message list: system prompt + all conversation history
    messages = [SYSTEM_PROMPT] + state["messages"]

    response = llm.invoke(messages)

    # Returning {"messages": [...]} APPENDS to state["messages"] — not replaces
    return {"messages": [response]}


# ── 2. Build the Graph ────────────────────────────────────────────────────────

graph_builder = StateGraph(MessagesState)
graph_builder.add_node("chat", chat)
graph_builder.add_edge(START, "chat")
graph_builder.add_edge("chat", END)

# This single line is what enables memory across turns
checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)


# ── 3. Helper to chat cleanly ─────────────────────────────────────────────────

def ask(question: str, thread_id: str = "default") -> str:
    """
    Send a message to the graph on a specific thread.
    Same thread_id = same conversation memory.
    """
    config = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config
    )

    # The last message in the list is always the AI's latest response
    return result["messages"][-1].content


# ── 4. Inspect conversation history ──────────────────────────────────────────

def get_history(thread_id: str = "default") -> list:
    """
    Pull the full saved state for a thread — useful for debugging.
    """
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    return state.values.get("messages", [])


# ── 5. Run it ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Memory chatbot. Commands:")
    print("  /new         — start a new conversation thread")
    print("  /switch <id> — switch to a different thread")
    print("  /history     — show this thread's message history")
    print("  /threads     — list active threads")
    print("  quit         — exit\n")

    current_thread = "thread-1"
    active_threads = {"thread-1"}
    thread_counter = 1

    print(f"Started on thread: {current_thread}\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        # ── Commands ──────────────────────────────────────────────────────────
        if user_input == "quit":
            break

        elif user_input == "/new":
            thread_counter += 1
            current_thread = f"thread-{thread_counter}"
            active_threads.add(current_thread)
            print(f"  [started new thread: {current_thread}]\n")
            continue

        elif user_input.startswith("/switch "):
            target = user_input.split(" ", 1)[1].strip()
            current_thread = target
            active_threads.add(target)
            print(f"  [switched to thread: {current_thread}]\n")
            continue

        elif user_input == "/history":
            print(f"\n  [History for {current_thread}]")
            for msg in get_history(current_thread):
                role = "You" if isinstance(msg, HumanMessage) else "Bot"
                print(f"  {role}: {msg.content[:100]}")
            print()
            continue

        elif user_input == "/threads":
            print(f"  Active threads: {', '.join(sorted(active_threads))}")
            print(f"  Current: {current_thread}\n")
            continue

        # ── Normal chat ───────────────────────────────────────────────────────
        response = ask(user_input, thread_id=current_thread)
        print(f"  [{current_thread}] Bot: {response}\n")