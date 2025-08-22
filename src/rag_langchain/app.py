from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END
import os


# Step 1: Define Gemini model + embeddings

llm = OllamaLLM(model="llama3.2")
print(f"LLM initialized with Ollama model.{llm.model}")
# print(llm.invoke("Hello, how are you?"))
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GEMINI_API_KEY"))
embeddings = OllamaEmbeddings(base_url="http://localhost:11434",model="nomic-embed-text:latest",
)
# Step 2: Load vector DB (Chroma in this case)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Step 3: Define state schema
from typing import TypedDict, List

class RAGState(TypedDict):
    question: str
    context: List[str]
    answer: str

# Step 4: Define LangGraph nodes
def retrieve(state: RAGState):
    docs = vectorstore.similarity_search(state["question"], k=3)
    state["context"] = [d.page_content for d in docs]
    return state

def generate(state: RAGState):
    prompt = f"Answer based on context:\n\n{state['context']}\n\nQuestion: {state['question']}"
    resp = llm.invoke(prompt)
    state["answer"] = resp
    return state

# Step 5: Build graph
graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

app = graph.compile()

# Step 6: Run RAG with Gemini
result = app.invoke({
    "question": "do have any appartment with unit size of 370?",
    "context": [],
    "answer": ""
})
print(result["answer"])
