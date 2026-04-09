import ollama
from pymongo import MongoClient
import time
from typing import List, Dict

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "rag_database"
COLLECTION_NAME = "text_embeddings"
EMBEDDING_MODEL = "nomic-embed-text-v2-moe"
GENERATION_MODEL = "gemma4"
MAX_TOKENS = 512


# ----- 1. Data utitlities ------
def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """Simple chunking mechanism for large documents."""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i : i + chunk_size])
    return chunks


# ------- 2. embedding and indexing (The ingestion phase) --------
def get_embedding(text: str) -> list[float]:
    """Calls Ollama to generate embeddings for a piece of text."""
    print(f" -> Generating embedding for text chunk....")
    try:
        response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
        return response["embedding"]
    except Exception as e:
        print(f"!!! Error calling ollama for embeddings : {e}")
        return []


def index_documents(documents: List[str]):
    """
    Processes raw documents: chunnks them, generates embeddings
    and stores in MongoDB."""
    print("\n " + "=" * 50)
    print("STARTING DOCUMENT INDEXING (EMBEDDING PHASE)")
    print("=" * 50)
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client[DB_NAME]
    collection = db[COLLECTION_NAME]
    collection.delete_many({})
    print(f" Cleaned existing collection: {COLLECTION_NAME}")
    chunked_data = []
    for i, doc_text in enumerate(documents):
        print(f"\nProcessing Document {i+1}/{len(documents)}...")
        chunks = chunk_text(doc_text)

        for j, chunk in enumerate(chunks):
            print(f" Chunk {j+1}/{len(chunks)}: Embedding...")
            embedding = get_embedding(chunk)
            if embedding:
                chunked_data.append(
                    {
                        "text": chunk,
                        "source_doc": f"doc_{i+1}",
                        "embedding": embedding,
                        "score": 1.0,  # placeholder for relevance
                    }
                )
            time.sleep(0.1)
            # slow down slightly to avoid rate limiting
    if chunked_data:
        result = collection.insert_many(chunked_data)
        print(f"\n Indexing Complete!")
        print(f"Stored {len(result.inserted_ids)} documents in MongoDB")
    else:
        print("Failed to index any documents.")

    mongo_client.close()


# ------ 3. Retrieval and generation (The Query Phase) -------
def retrieve_context(query: str, top_k: int = 3) -> List[str]:
    """
    Generates an embedding for the query and uses MongoDB to find the
    monst semantically simillar chunks (simulated vector search).
    """
    print("\n" + "=" * 50)
    print("STEP 1: Retrieval (finding context)")
    print("=" * 50)

    # 1. Embed the query
    query_embedding = get_embedding(query)
    if not query_embedding:
        return []

    # 2. Connect to MongoDB
    mongo_client = MongoClient(MONGO_URI)
    collection = mongo_client[DB_NAME][COLLECTION_NAME]

    # NOTE: For true vector search, you should use a dedicated vector index
    # (like HNSW or specialized Atlas feature). Here, we simulate similarity
    # by fetching top documents based on a simple proximity measure
    # (or, in a real scenario, using $vectorSearch).
    print(f"Searching MongoDB for top {top_k} simillar chunks....")
    # *** Placeholder for Vector Search ***
    # Since standard pymongo requires specific vector indexing
    # (e.g., Atlas/Wrangler), we will simulate fetching top results
    # by selecting random chunks for demonstration clarity.
    # In a production system, this query would use $vectorSearch.

    # Realistic (but simplified) MongoDB retrieval simulation:
    cursor = collection.find().sort("score", -1).limit(top_k)

    context_list = []
    for doc in cursor:
        context_list.append(doc["text"])
    mongo_client.close()
    return context_list


def generate_answer(query: str, context: List[str]) -> str:
    """Used the retrived context and the original query to generate final answer"""

    print("\n" + "=" * 50)
    print("STEP 2 : Generation (Synthesizing the answer)")
    print("=" * 50)

    # 1. construct the prompt
    context_string = "\n---\n".join(context)
    prompt = f"""
    You are an expert question-answering system. Your task is to answer the user's question
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
    DOCUMEN_DATASET = [
        """
        Python is a high-level, interpreted programming language known for its readability. 
        It was created by Guido van Rossum and first released in 1991. Python's structure 
        makes it easy to learn, which is why it is heavily used in data science, AI, and web development. 
        Major frameworks include Django and Flask.
        """,
        """
        Retrieval-Augmented Generation (RAG) is an architectural pattern that aims to 
        improve the factual accuracy of LLMs. Instead of relying only on the model's training data, 
        RAG retrieves external, relevant documents (the "context") and passes that context 
        to the LLM, allowing it to ground its answer in specific, verifiable information.
        The process involves embedding documents, storing them in a vector database, 
        querying for similar vectors, and finally generating the response.
        """,
    ]

    # ====================================================
    # PHASE 1: INDEXING (Run this once, or when data changes)
    # ====================================================
    index_documents(DOCUMEN_DATASET)

    # ====================================================
    # PHASE 2: QUERYING (The actual RAG process)
    # ====================================================
    user_query = "What is the purpose of using RAG in AI, and what are the main frameworks mentioned?"

    # 1. retrieve context
    retrieved_context = retrieve_context(user_query, 2)

    if retrieve_context:
        print("\n [DEBUG] Retrived context snippets:")
        for i, context in enumerate(retrieved_context):
            print(f"-----Context {i+1} ------\n {context[:100]}...")

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
