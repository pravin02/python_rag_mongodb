
import psycopg2
conn = psycopg2.connect("postgresql://postgres:postgres@localhost:5432/postgres")

from pgvector.psycopg2 import register_vector
register_vector(conn)

def create_table():
    with conn.cursor() as cur:
        cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding VECTOR(768)
                    )
                    """)
        conn.commit()
    print("postgres.documents table created successfully.")

def fetchall_and_print():
    with conn.cursor() as cur:
        cur.execute("""
            SELECT * FROM documents
            """
        )
        results = cur.fetchall();
        if not results:
            print("No records")
            return
        for id, content, embedding in results:
            print(f"ID: {id}\n Content:{content}\nEmbedding:{embedding}")

create_table()


import ollama

def get_embedding(text: str) -> list[float]:
    try:
        response = ollama.embeddings(
            model="nomic-embed-text-v2-moe",
            prompt=text
        )
        return response["embedding"]
    except Exception as e:
        print(f"Error while fetching embedding from ollama. Error : {e}")
    return []

# documents = [
#     "PostgreSQL is a powerful open-source relational database.",
#     "pgvector adds vector similarity search to PostgreSQL.",
#     "Python is a high-level programming language.",
# ]

# with conn.cursor() as cur:
#     for doc in documents:
#         embedding = get_embedding(doc)
#         cur.execute(
#             "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
#             (doc, embedding)
#         )
#     conn.commit()
 
fetchall_and_print()


def retrieve_context(query, top_k:int= 3) :
    import numpy as np    
    query_embedding = get_embedding(query)
    query_embedding = np.array(query_embedding)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT content, 1 - (embedding <=> %s) AS similarity
            FROM documents
            ORDER BY embedding <=> %s
            LIMIT 5
        """, (query_embedding, query_embedding))

        results = cur.fetchall()
        for content, similarity in results:
            print(f"[{similarity:.3f}] {content}")

query = "How does vector search work in PostgreSQL?"
retrieve_context(query)