# LangGraph Chatbot with Memory

This project demonstrates how to build a conversational assistant using **LangGraph**, **LangChain**, and **Ollama**. It features a command-line interface that allows you to interact with a local Large Language Model (`llama3.1:8b`) while maintaining conversation history across multiple threads.

## Project Structure

There are two main implementations provided in this repository, showcasing different approaches to memory management:

1. **`chatbot_memory.py`**: 
   A conversational agent that uses `MemorySaver` to store conversation history strictly in-memory. It maintains context as long as the script is running and supports multiple conversation threads, allowing you to switch between different discussions seamlessly.

2. **`chatbot_memory_trim.py`**: 
   An advanced version that introduces two key enhancements:
   - **Context Management**: Prompts can get too large over long conversations. This version uses `trim_messages` to retain only the last 10 messages within the state. This prevents the prompt context window from overflowing.
   - **Persistent Storage**: Replaces the in-memory saver with `SqliteSaver`, storing the conversation history in a local SQLite database (`memory.db`). This means your chat context, history, and threads will persist even if you restart the script.

## Prerequisites

Before running the scripts, ensure you have the following installed:

- Python 3.8+
- [Ollama](https://ollama.com/) installed and running locally.
- Required Python packages. You can install them via pip:
  ```bash
  pip install langgraph langchain-ollama langchain-core
  ```

Additionally, you need to pull the specific open-source model used in the scripts:
```bash
ollama pull llama3.1:8b
```

## How to Run

1. Make sure your local Ollama application is running in the background.
2. Open your terminal in the project directory.
3. Run either of the chatbot scripts using Python:

   To run the basic in-memory version:
   ```bash
   python chatbot_memory.py
   ```

   To run the version with message trimming and SQLite persistence:
   ```bash
   python chatbot_memory_trim.py
   ```

## Available Commands

Once you start the script, you will enter a command-line interface. You can type natural language messages to chat directly, or use the following commands to manage conversation threads:

- `/new` — Start a completely new conversation thread.
- `/switch <id>` — Switch to a different, existing thread (e.g., `/switch thread-2`).
- `/history` — Display the full message history for the current active thread.
- `/threads` — List all currently active conversation threads.
- `quit` — Exit the application.
