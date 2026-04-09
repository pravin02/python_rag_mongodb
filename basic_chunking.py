from typing import List, Dict


# ----- 1. Data utitlities ------
def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """Simple chunking mechanism for large documents."""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i : i + chunk_size])
    return chunks


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

for doc in DOCUMEN_DATASET:
    chunks = chunk_text(doc, 50)
    print("Total chunks are ", len(chunks))
    for i in range(0, len(chunks)):
        print(f"\n {i+1}  --- {len(chunks[i])}  -- {chunks[i]}")
