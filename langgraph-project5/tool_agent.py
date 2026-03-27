from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from datetime import datetime
import math
import os


# ── 1. Define the tools ───────────────────────────────────────────────────────
# Each tool is a plain Python function with the @tool decorator.
# The docstring IS the tool's description — the LLM reads it to decide when to use it.

@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    Use this for any arithmetic, percentages, or calculations.
    Examples: '2 + 2', '100 * 0.15', 'math.sqrt(144)', '2 ** 10'
    """
    try:
        # Give it access to math functions
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Could not calculate '{expression}': {e}"


@tool
def search_web(query: str) -> str:
    """
    Search the web for current information about a topic.
    Use this when you need facts, news, or information you might not know.
    Input: a clear search query string.
    """
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No results found."
        # Format results cleanly for the LLM
        formatted = []
        for r in results:
            formatted.append(f"Title: {r['title']}\nSnippet: {r['body']}\n")
        return "\n".join(formatted)
    except Exception as e:
        return f"Search failed: {e}"


@tool
def get_datetime() -> str:
    """
    Get the current date and time.
    Use this when the user asks about today's date, current time, or day of week.
    No input needed.
    """
    now = datetime.now()
    return now.strftime("Today is %A, %B %d %Y. Current time: %H:%M")


@tool
def read_file(filepath: str) -> str:
    """
    Read and return the contents of a local text file.
    Use this when the user asks you to read, summarize, or analyze a file.
    Input: the full or relative path to the file.
    """
    try:
        # Safety: only allow reading files in current directory
        safe_path = os.path.basename(filepath)
        with open(safe_path, "r") as f:
            content = f.read()
        return f"Contents of {safe_path}:\n\n{content}"
    except FileNotFoundError:
        return f"File '{filepath}' not found in the current directory."
    except Exception as e:
        return f"Could not read file: {e}"


@tool
def save_note(filename: str, content: str) -> str:
    """
    Save a note or text to a file.
    Use this when the user asks to save, write, or store something to a file.
    Inputs: filename (string) and content (string to save).
    """
    try:
        safe_name = os.path.basename(filename)
        with open(safe_name, "w") as f:
            f.write(content)
        return f"Successfully saved to '{safe_name}'"
    except Exception as e:
        return f"Could not save file: {e}"


# ── 2. Set up LLM with tools ──────────────────────────────────────────────────

tools = [calculate, search_web, get_datetime, read_file, save_note]

llm = ChatOllama(model="llama3.1:8b")
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = SystemMessage(content="""You are a capable personal assistant with access to tools.

When you need to:
- Do math or calculations → use the calculate tool
- Look up current info or facts → use the search_web tool  
- Find out today's date or time → use the get_datetime tool
- Read a local file → use the read_file tool
- Save something to a file → use the save_note tool

Think carefully about whether you need a tool. If you can answer confidently from knowledge, do so directly.
When you use a tool, wait for its result before answering.
Always give a clear, helpful final answer.""")


# ── 3. Nodes ──────────────────────────────────────────────────────────────────

def agent(state: MessagesState) -> dict:
    """
    The reasoning node. Sends full history to the LLM.
    The LLM either returns a tool_call or a final answer.
    """
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ToolNode handles tool execution automatically — no manual node needed


# ── 4. Build the graph ────────────────────────────────────────────────────────

graph_builder = StateGraph(MessagesState)

graph_builder.add_node("agent", agent)
graph_builder.add_node("tools", ToolNode(tools))  # built-in tool executor

graph_builder.add_edge(START, "agent")

# tools_condition is a built-in routing function:
# → returns "tools" if last message has a tool call
# → returns END if last message is a plain response
graph_builder.add_conditional_edges("agent", tools_condition)

# After tools run, always go back to agent to reason with the result
graph_builder.add_edge("tools", "agent")

checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)


# ── 5. Run it ─────────────────────────────────────────────────────────────────

def ask(question: str, thread_id: str = "main", verbose: bool = True) -> str:
    config = {"configurable": {"thread_id": thread_id}}

    # Stream so we can show tool usage as it happens
    response_text = ""
    for event in graph.stream(
        {"messages": [HumanMessage(content=question)]},
        config=config,
        stream_mode="values"
    ):
        last_msg = event["messages"][-1]

        # Show tool calls as they happen
        if verbose and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            for tc in last_msg.tool_calls:
                print(f"  [tool call] {tc['name']}({tc['args']})")

        # Capture the final text response
        if hasattr(last_msg, "content") and last_msg.content:
            response_text = last_msg.content

    return response_text


if __name__ == "__main__":
    print("Tool agent ready. I can calculate, search, read files, and save notes.")
    print("Type 'quit' to exit.\n")

    thread_id = "main"

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break

        print("Bot: ", end="", flush=True)
        answer = ask(user_input, thread_id=thread_id)
        print(answer)
        print()
