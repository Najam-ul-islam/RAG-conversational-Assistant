# app_conversational_rag.py
from typing import TypedDict, List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv, find_dotenv
from colorama import Style, init, Fore
import os

# Initialize colorama for colored terminal output
init(autoreset=True)
# -----------------------------
# 0) Config
# -----------------------------
# Load environment variables from .env file
_:bool = load_dotenv(find_dotenv())
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

# -----------------------------
# 1) LLM + Embeddings + VectorStore
# -----------------------------
llm = OllamaLLM(model=OLLAMA_LLM_MODEL)
embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# (Optional) If you need to bootstrap the index the first time:

# loader = PyPDFLoader(r"D:\Machine learning Training\RAG\rag_langchain\documents\CV Muhammad Najam ul islam.pdf")
# pages = loader.load()
# vectorstore.add_documents(pages)
# Save the vectorstore to disk

# -----------------------------
# 2) State schema for LangGraph
# -----------------------------
class ChatState(TypedDict):
    # rolling memory of messages across turns
    messages: List[Dict[str, str]]  # each: {"role": "user"|"assistant", "content": str}
    # per-turn working fields
    standalone_question: str
    context: List[str]
    answer: str

# -----------------------------
# 3) Helpers
# -----------------------------
def _format_history(messages: List[Dict[str, str]], last_n: int = 6) -> str:
    """Format last N turns for prompts."""
    hist = messages[-last_n:]
    out = []
    for m in hist:
        role = "User" if m["role"] == "user" else "Assistant"
        out.append(f"{role}: {m['content']}")
    return "\n".join(out)

# -----------------------------
# 4) LangGraph Nodes
# -----------------------------
def rewrite_question(state: ChatState) -> ChatState:
    """Turn the latest user input into a standalone query using chat history."""
    history = _format_history(state["messages"], last_n=8)
    latest_user_msg = next((m["content"] for m in reversed(state["messages"]) if m["role"] == "user"), "")
    prompt = f"""You rewrite the user's latest question into a standalone query so it can be searched without prior context.

Chat History:
{history}

Latest user question:
{latest_user_msg}

Rewrite the latest user question as a single, clear, self-contained query. Do not answer it."""
    rewritten = llm.invoke(prompt).strip()
    state["standalone_question"] = rewritten
    return state

def retrieve(state: ChatState) -> ChatState:
    """Retrieve top-k documents for the rewritten question."""
    docs = retriever.invoke(state["standalone_question"])
    state["context"] = [d.page_content for d in docs]
    return state

def generate(state: ChatState) -> ChatState:
    """Generate the final answer using context + history."""
    history = _format_history(state["messages"], last_n=8)
    context_blob = "\n\n---\n".join(state["context"]) if state["context"] else "NO RELEVANT CONTEXT"
    prompt = f"""You are a helpful assistant. Use CONTEXT to answer the user's latest question.
If the answer isn't in the context, say you don't know. Be concise and cite facts from context.

CHAT HISTORY (recent):
{history}

CONTEXT:
{context_blob}

QUESTION (standalone):
{state['standalone_question']}

FINAL ANSWER:"""
    response = llm.invoke(prompt).strip()
    state["answer"] = response
    # append assistant reply to memory
    state["messages"].append({"role": "assistant", "content": response})
    return state

# -----------------------------
# 5) Build the LangGraph
# -----------------------------
graph = StateGraph(ChatState)
graph.add_node("rewrite_question", rewrite_question)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

graph.set_entry_point("rewrite_question")
graph.add_edge("rewrite_question", "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

app = graph.compile()

# from colorama import Fore, Style

from typing import Optional

def print_table(title: str, instructions: list, examples: Optional[list] = None):
    # Determine max width of the table
    content_lines = [title] + instructions + (examples if examples else [])
    max_len = max(len(line) for line in content_lines) + 10  # padding
    
    # Top border
    print("\n" + Fore.YELLOW + "╔" + "═" * max_len + "╗")
    
    # Title row
    print("║" + title.center(max_len) + "║")
    
    # Separator
    print("╠" + "═" * max_len + "╣")
    
    # Instructions
    print("║ " + Fore.RED + "Instructions!!!".ljust(max_len-1) + Style.RESET_ALL + "║")
    print("║" + " " * max_len + "║")  # empty line
    for instr in instructions:
        print("║ " + Fore.BLUE + instr.ljust(max_len-1) + Style.RESET_ALL + "║")
    
    # Separator for examples
    if examples:
        print("║" + " " * max_len + "║")  # empty line
        print("╠" + "═" * max_len + "╣")
        for ex in examples:
            print("║ " + Fore.CYAN + ex.ljust(max_len-1) + Style.RESET_ALL + "║")
    
    # Bottom border
    print("╚" + "═" * max_len + "╝")



# -----------------------------
# 6) Multi-turn driver
# -----------------------------
def run_turn(app, state: ChatState, user_input: str) -> ChatState:
    # push user message into memory
    state["messages"].append({"role": "user", "content": user_input})
    # invoke the graph; it will update standalone_question/context/answer and append assistant msg
    result = app.invoke(state)
    return result

if __name__ == "__main__":
    # initial memory (empty chat)
    state: ChatState = {
        "messages": [],
        "standalone_question": "",
        "context": [],
        "answer": ""
    }
    # Welcome message
    title = "RAG Conversational Assistant"
    instructions = [
        "• You can ask about documents, context, or any other information.",
        "• You can upload documents to the vector store to enhance context.",
        "• You can also ask follow-up questions based on previous answers.",
        "• To upload documents, type UPLOAD <file_path>",
        "• To exit, type exit or quit"
    ]
    examples = ["Example: do you have any apartment with unit size of 370?"]

    print_table(title, instructions, examples)
    # Interactive loop for multi-turn conversation

    while True:
        user_input = input(f"\n{Fore.GREEN + 'User: '}")
        # user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        state = run_turn(app, state, user_input)
        print(f"\n{Fore.BLUE + "Assistant:"}", state["answer"])