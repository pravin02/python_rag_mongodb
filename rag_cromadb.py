import ollama
import time
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

import chromadb
from chromadb.utils import embedding_functions

DB_NAME = "cromadb"
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "nomic-embed-text-v2-moe"
GENERATION_MODEL = "gemma4"
MAX_TOKENS = 512

client = chromadb.PersistentClient(path="./documents")

collection = client.get_or_create_collection(
    COLLECTION_NAME,
    configuration={
        "hnsw": {
            "space": "cosine",
            "ef_construction": 200,
            "max_neighbors": 32,
        }
    },
)


# Example: Using Ollama as an embedding function
# You need to define a wrapper that matches Chroma's expected interface
def ollama_embedding_function(texts):
    """
    Custom embedding function for ChromaDB.
    `texts` is a list of strings.
    This function should return a list of embedding vectors (lists of floats).
    """
    import requests

    embeddings = []
    for text in texts:
        # Example request to Ollama API (adjust model name and endpoint as needed)
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
        )
        data = response.json()
        embeddings.append(data["embedding"])
    return embeddings


def index_pdf_documents(file_path):
    """
    Processes raw documents: chunnks them, generates embeddings
    and stores in Postgres."""
    print("\n " + "=" * 50)
    print("STARTING DOCUMENT INDEXING (EMBEDDING PHASE)")
    print("=" * 50)
    pdf_loader = PyPDFLoader(file_path)
    docs = pdf_loader.load()

    splitter = CharacterTextSplitter(separator=" ", chunk_size=500, chunk_overlap=80)

    chunks = splitter.split_documents(docs)

    for i, doc in enumerate(chunks):
        print(f"\nProcessing Document {i+1}/{len(chunks)}...")
        print(f" Chunk {i+1}/{len(chunks)}: Embedding...")
        collection.add(documents=[doc.page_content], ids=[f"policy_{i}"])


# ------ 3. Retrieval and generation (The Query Phase) -------
def retrieve_context(query: str, top_k: int = 5) -> List[str]:
    """
    Generates an embedding for the query and uses CromaDB to find the
    monst semantically simillar chunks (simulated vector search).
    """
    results = collection.query(query_texts=[query], n_results=top_k)
    return results.get("documents")[0];


def generate_answer(query: str, context: List[str]) -> str:
    """Used the retrived context and the original query to generate final answer"""

    print("\n" + "=" * 50)
    print("STEP 2 : Generation (Synthesizing the answer)")
    print("=" * 50)

    # 1. construct the prompt
    context_string = "\n---\n".join(context)
    prompt = f"""
    You are an expert Financial Figure. Your task is to answer the user's question
    based ONLY on the context provided below. If the context does not contain the answer, 
    you must state clearly that the information is not available.

    CONTEXT:
    ---
    {context_string}
    ---

    QUESTION: {query}

    ANSWER:
    """

    # 2. call ollama LLM
    try:
        response = ollama.generate(
            model=GENERATION_MODEL,
            prompt=prompt,
            options={"temperature": 0.1, "num_predict": MAX_TOKENS},
        )
        return response["response"].strip()
    except Exception as e:
        return f"!!! Error generating answer via ollama: {e}"


# Main Execution
if __name__ == "__main__":
    # ====================================================
    # PHASE 1: INDEXING (Run this once, or when data changes)
    # ====================================================
    # index_documents(DOCUMEN_DATASET)

    # pdf_file_path = "./dataset/mediarelease-en.pdf"
    # index_pdf_documents(pdf_file_path)

    # ====================================================
    # PHASE 2: QUERYING (The actual RAG process)
    # ====================================================
    while True:
        user_query = input("\nAsk your question: ")

        # questions = generate_multiple_questions_from_prompt(user_query)
        # print("\nQuestions generated from entered queries")
        # print(questions)
        # print("\n")
        # 1. retrieve context
        retrieved_context = retrieve_context(user_query)
        if retrieved_context:
            print("\n [DEBUG] Retrived context snippets:")
            print(f"-----Context {retrieved_context}")

            # 2. generate ansert
            final_answer = generate_answer(user_query, retrieved_context)
            print("\n=================================================")
            print("          ✨ FINAL SYSTEM ANSWER ✨")
            print("=================================================")
            print(final_answer)
            print("=================================================")
        else:
            print(
                "\n[CRITICAL ERROR] Could not retrieve context. Check your database connection."
            )
