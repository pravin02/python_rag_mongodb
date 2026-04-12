import ollama
import time
from typing import List
import psycopg2
from pgvector.psycopg2 import register_vector


DB_NAME = "postgres"
DB_URI = f"postgresql://postgres:postgres@localhost:5432/{DB_NAME}"
TABLE_NAME = "documents"

EMBEDDING_MODEL = "nomic-embed-text-v2-moe"
GENERATION_MODEL = "gemma4"
MAX_TOKENS = 512

conn = psycopg2.connect(DB_URI)

register_vector(conn)


def create_table():
    with conn.cursor() as cur:
        cur.execute(
            """
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding VECTOR(768)
                    )
                    """
        )
        conn.commit()
    print("postgres.documents table created successfully.")


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


def delete_all_records():
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE TABLE {TABLE_NAME}")
        conn.commit()
    print(f"{DB_NAME}.{TABLE_NAME} table truncated successfully.")


def insert_record(content: str, embedding):
    with conn.cursor() as cur:
        # try:
        cur.execute(
            """
                        INSERT INTO documents(content, embedding) 
                        VALUES (%s, %s)
                        """,
            (content, embedding),
        )
        conn.commit()
        print(f"Record inserted successfully in {DB_NAME}.{TABLE_NAME} table")
    # except Exception as e:
    #     print(f"Failed to insert record due to:>>  {e}")
    # conn.commit()


def index_documents(documents: List[str]):
    """
    Processes raw documents: chunnks them, generates embeddings
    and stores in MongoDB."""
    print("\n " + "=" * 50)
    print("STARTING DOCUMENT INDEXING (EMBEDDING PHASE)")
    print("=" * 50)

    delete_all_records()

    chunked_data = []
    for i, doc_text in enumerate(documents):
        print(f"\nProcessing Document {i+1}/{len(documents)}...")
        chunks = chunk_text(doc_text)

        for j, chunk in enumerate(chunks):
            print(f" Chunk {j+1}/{len(chunks)}: Embedding...")
            embedding = get_embedding(chunk)
            if embedding:
                insert_record(chunk, embedding)


def index_pdf_documents(file_path):
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import CharacterTextSplitter

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
        embedding = get_embedding(doc.page_content)
        if embedding:
            inser_record(doc.page_content, embedding)


# ------ 3. Retrieval and generation (The Query Phase) -------
def retrieve_context(query: str, top_k: int = 5) -> List[str]:
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

    print(f"Searching PGVECTOR for top {top_k} simillar chunks....")
    import numpy as np

    query_embedding = get_embedding(query)
    query_embedding = np.array(query_embedding)

    semantic_results = get_semantic_search_results(query_embedding, top_k)
    full_text_search_result = get_full_text_search_results(query, top_k)

    return semantic_results + full_text_search_result


def retrieve_context_from_queries(queries: List[str], top_k: int = 5) -> List[str]:
    print("\n" + "=" * 50)
    print("STEP 1: Retrieval context from multiple queries")
    print("=" * 50)

    results = []
    for query in queries:
        # 1. Embed the query
        query_embedding = get_embedding(query)
        if not query_embedding:
            return []

        print(f"Searching PGVECTOR for top {top_k} simillar chunks....")
        import numpy as np

        query_embedding = get_embedding(query)
        query_embedding = np.array(query_embedding)

        semantic_results = get_semantic_search_results(query_embedding, top_k)
        full_text_search_result = get_full_text_search_results(query, top_k)
        results.append(semantic_results + full_text_search_result)
    return results


def get_semantic_search_results(query_embedding, top_k):
    """------------ Retriveing By Semantic Seach Start --------------"""
    with conn.cursor() as cur:
        try:
            cur.execute(
                """
                SELECT id, content, 1 - (embedding <-> %s) AS similarity
                FROM documents
                ORDER BY embedding <-> %s desc
                LIMIT %s
            """,
                (query_embedding, query_embedding, top_k),
            )
            print(f"Context retrieved successfully...")

            context_list = []
            for row in cur.fetchall():
                context_list.append(row)
                print(f"\n ID : {row[0]}\nContent: {row[1]}\nScore: {row[2]}")

            return context_list
        except Exception as e:
            print(f"get_semantic_search_results: Failed to fetch record due to:>>  {e}")
    """ ------------ Retriveing By Semantic Seach End --------------"""
    return []


def get_full_text_search_results(query, top_k):
    """------------ Retriveing By Full Text Seach Start --------------"""
    with conn.cursor() as cur:
        try:
            cur.execute(
                """SELECT id, content FROM documents, plainto_tsquery('english', %s) query WHERE to_tsvector('english', content) @@ query 
                    ORDER BY ts_rank_cd(to_tsvector('english', content), query) DESC LIMIT %s
                """,
                (query, top_k),
            )
            print(f"Full text retrieved successfully...")

            context_list = []
            for row in cur.fetchall():
                context_list.append(row)
                print(f"\n ID : {row[0]}\nContent: {row[1]}")

            return context_list
        except Exception as e:
            print(
                f"get_full_text_search_results: Failed to fetch record due to:>>  {e}"
            )
    """ ------------ Retriveing By Full Text Seach End --------------"""
    return []


def generate_multiple_questions_from_prompt(query: str) -> List[str]:
    """------generate_multiple_questions_from_prompt -----------"""

    print("\n" + "=" * 50)
    print("STEP : generate at most 3 question from users entered query")
    print("=" * 50)
    prompt = f"""
    You are the logical reasoner and responsble to understand semantics to 
    generate at most 3 questions from users entered query which would be feed 
    to pgvector database to retrieve context.
    
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
        print(f"generate_multiple_questions_from_prompt: questions {response["response"]}")
        return response["response"].strip()
    except Exception as e:
        f"!!! Error generating questions via ollama: {e}"
    return []


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

    # create_table()

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
        context_content = []
        if retrieved_context:
            print("\n [DEBUG] Retrived context snippets:")
            for i, context in enumerate(retrieved_context):
                context_content.append(context[1])
                # print(f"-----Context {i+1} ------\n {context[1][:100]}...")

            # 2. generate ansert
            final_answer = generate_answer(user_query, context_content)
            print("\n=================================================")
            print("          ✨ FINAL SYSTEM ANSWER ✨")
            print("=================================================")
            print(final_answer)
            print("=================================================")
        else:
            print(
                "\n[CRITICAL ERROR] Could not retrieve context. Check your database connection."
            )
