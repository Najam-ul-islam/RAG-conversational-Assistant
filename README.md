# RAG Conversational Assistant

This repository implements a **Retrieval-Augmented Generation (RAG) conversational assistant** using [LangChain](https://www.langchain.com/), [Ollama](https://ollama.ai/), and [ChromaDB](https://docs.trychroma.com/). It supports **multi-turn conversations with memory**, document retrieval, and context-aware responses.

---

## 📂 Project Structure

- **`rag_conversation.py`** – Interactive conversational RAG assistant powered by **LangGraph**.  
  - Uses Ollama LLM for query rewriting and answer generation.  
  - Uses ChromaDB for document retrieval.  
  - Maintains **multi-turn memory** across the conversation.  
  - Displays a styled **welcome/instructions table** in the terminal.  

- **`main.py`** – Script to load documents (PDFs), split them into chunks, embed them, and store them in ChromaDB.  
  - Supports embeddings via **Ollama** and **Google Generative AI Embeddings** (Gemini).  
  - Demonstrates similarity search queries against the vector store.  

---

## 🚀 Features

- ✅ **Multi-turn conversation with memory**  
- ✅ **Retrieval-Augmented Generation (RAG)** using ChromaDB  
- ✅ **Standalone query rewriting** (handles follow-ups gracefully)  
- ✅ **PDF ingestion and chunking** with `RecursiveCharacterTextSplitter`  
- ✅ **Embeddings via Ollama (`nomic-embed-text`)**  
- ✅ **Terminal-based conversational loop with instructions**  

---

## 🔧 Installation

### 1. Clone the repository
```bash
git clone https://github.com/Najam-ul-islam/RAG-conversational-Assistant.git
cd RAG-conversational-Assistant
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ⚙️ Environment Setup

Create a `.env` file in the root directory:

```env
OLLAMA_LLM_MODEL=llama3.2
OLLAMA_EMBED_MODEL=nomic-embed-text
CHROMA_DIR=./chroma_db
GEMINI_API_KEY=your_google_gemini_api_key   # (optional, if using Gemini embeddings)
```

Make sure [Ollama](https://ollama.ai/) is running locally.

---

## 📘 Usage

### 1. Load and index documents
Update the file path in `main.py` to point to your PDF:

```python
file_path = r"D:\Machine learning Training\RAG\rag_langchain\documents\Riviera.pdf"
```

Run:
```bash
python main.py
```
This will load the PDF, chunk it, embed it, and save vectors in ChromaDB.

---

### 2. Start the Conversational RAG Assistant
```bash
python rag_conversation.py
```

You’ll see an instruction table:

```
╔══════════════════════════════════════════════════════╗
║              RAG Conversational Assistant            ║
╠══════════════════════════════════════════════════════╣
║ Instructions!!!                                      ║
║ • You can ask about documents, context, or ...       ║
║ • To upload documents, type UPLOAD <file_path>       ║
║ • To exit, type exit or quit                         ║
╚══════════════════════════════════════════════════════╝
```

Example:
```
User: Who is Muhammad Najam’s father?
Assistant: His father is mentioned as ... (from context)
```

---

## 🛠️ Tech Stack

- **LangChain** – framework for LLM-powered apps  
- **LangGraph** – stateful workflows for multi-turn conversations  
- **Ollama** – local LLM and embeddings  
- **ChromaDB** – persistent vector database  
- **Colorama** – styled CLI output  

---

## 📌 Next Steps

- [ ] Add support for multiple document types (txt, docx, etc.)  
- [ ] Expose as a REST API (FastAPI) or Web UI (Streamlit/Next.js)  
- [ ] Add citation formatting for retrieved docs  
- [ ] Dockerize for easy deployment  
