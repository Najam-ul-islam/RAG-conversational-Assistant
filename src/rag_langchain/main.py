from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
import os
from dotenv import load_dotenv, find_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
import argparse

_: bool = load_dotenv(find_dotenv())


# Load the envirnment variables
gemini_api_key = os.getenv("GEMINI_API_KEY")
# model = os.getenv("MODEL")
# embeding_model = os.getenv("EMBEDING_MODEL")


#  File path
# file_path=r"D:\Machine learning Training\RAG\rag_langchain\documents\Riviera 68 & 69 Availability.pdf"
file_path =r"D:\Machine learning Training\RAG\rag_langchain\documents\Riviera.pdf"
# STEP 1: Load the PDF document
# ===============================
loader = PyPDFLoader(file_path)
pages = []
for page in loader.load():
    pages.append(page)
print(f"Number of pages loaded: {len(pages)}")
print(f"Total pages content: {sum(len(page.page_content) for page in pages)} characters")

# STEP 2: Split the document into chunks
# =========================================
document = "\n".join([page.page_content for page in pages])

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=10,
    length_function=len
)
# Split the document into chunks
document = "\n".join([page.page_content for page in pages])
texts = text_splitter.create_documents([document])
# print(f"Number of chunks with metadata: {len(texts)}")
# print(texts[0])

# Create embeddings for the chunks
# embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",
#                                            google_api_key=gemini_api_key,
#                                            task_type="RETRIEVAL_QUERY"
#                                            )
# embeddings.embed_documents([text.page_content for text in texts])
# vectors = embeddings.embed_query("Hello World")
# print(f"Vectors: \n{vectors[10:20]}")  # Display a slice of the vectors
# Custome model for embeddings
ollama_embeddings = OllamaEmbeddings(
    base_url="http://localhost:11434",
    model="nomic-embed-text:latest",
)
# ollama_embeddings.embed_query("Hello World")
# print(f"Vectors: \n{ollama_embeddings.embed_query('Hello World of langchain')}")
# vectors = ollama_embeddings.embed_documents([text.page_content for text in texts])
# print(f"Vectors: \n{vectors}")
# print(f"Number of vectors: {len(vectors)}")
# TODO: Create a vector store from the chunks
# vectorstore = InMemoryVectorStore.from_documents(
#     documents=texts,
#     embedding=embeddings
# )
# vectorstore = FAISS.from_documents(texts, embeddings)

vectorstore = Chroma.from_documents(texts, ollama_embeddings, persist_directory="./chroma_db")
# Use the vectorstore as a retriever
# retriever = vectorstore.as_retriever()
# Retrieve the most similar text
# retrieved_documents = retriever.invoke("what is his father name?")
# show the retrieved document's content
# print(f"Retrived text: \n{retrieved_documents[0].page_content}")
# TODO: Create a retriever from the vector store
# TODO: Create a language model

# STEP 5: Do a similarity search
query = "What is Muhammad Najam's father name?"
results = vectorstore.similarity_search(query, k=2)


# results = vectorstore.similarity_search_with_score(
#     query, k=2
# )
# # print(f"* [SIM={results}]")
# for res, score in results:
#     print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")

print("\nTop results:")
for i, doc in enumerate(results, start=1):
    print(f"\nResult {i}:\n{doc.page_content[:500]}...")