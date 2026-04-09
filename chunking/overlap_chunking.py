# from typing import List, Dict


# # ----- 1. Data utitlities ------
# def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
#     """Simple chunking mechanism for large documents."""
#     chunks = []
#     for i in range(0, len(text), chunk_size):
#         chunks.append(text[i : i + chunk_size])
#     return chunks


# file_path = "./dataset/dataset.txt"

# try:
#     with open(file_path, "r", encoding="utf-8") as file:
#         line_number = 0
#         for line in file:
#             line_number += 1
#             chunks = chunk_text(line.strip(), 50)
#             print("Total chunks are ", len(chunks))
#             for i in range(0, len(chunks)):
#                 print(f"\n {i+1}  --- {len(chunks[i])}  -- {chunks[i]}")
# except FileNotFoundError as e:
#     print(f"Error while reading file '{file_path}'. Error: {e}")


# ------- 2. using langchain -------------
# doc = """
#         Python is a high-level, interpreted programming language known for its readability. 
#         It was created by Guido van Rossum and first released in 1991. Python's structure 
#         makes it easy to learn, which is why it is heavily used in data science, AI, and web development. 
#         Major frameworks include Django and Flask.
        
#         Retrieval-Augmented Generation (RAG) is an architectural pattern that aims to 
#         improve the factual accuracy of LLMs. Instead of relying only on the model's training data, 
#         RAG retrieves external, relevant documents (the "context") and passes that context 
#         to the LLM, allowing it to ground its answer in specific, verifiable information.
#         The process involves embedding documents, storing them in a vector database, 
#         querying for similar vectors, and finally generating the response.
#         """
# from langchain_text_splitters import CharacterTextSplitter

# splitter = CharacterTextSplitter(separator=" ", chunk_size=70, chunk_overlap=15)
# chunks = splitter.split_text(doc)
# for i in range(0, len(chunks)):
#     print(f"{i+1} ---- {len(chunks[i])} ---- {chunks[i]}")


# ---------- Reading PDF File ----------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

pdf_file_path = "./dataset/mediarelease-en.pdf"
pdf_loader = PyPDFLoader(pdf_file_path)
docs = pdf_loader.load()

splitter = CharacterTextSplitter(separator=" ", 
                                 chunk_size=500, chunk_overlap=80)

chunks =splitter.split_documents(docs)


for i in range(0, len(chunks)):
    if i == 3:
        break
    # print("###DOC####",docs[i])

    print("\n\n###CHUNK####", chunks[i].page_content)