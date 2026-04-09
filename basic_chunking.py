from typing import List


# ----- 1. custom code ------
# def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
#     """Simple chunking mechanism for large documents."""
#     chunks = []
#     for i in range(0, len(text), chunk_size):
#         chunks.append(text[i : i + chunk_size])
#     return chunks

# file_path = "dataset/dataset.txt"

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
# from langchain_text_splitters import CharacterTextSplitter

# splitter = CharacterTextSplitter(
#     separator=" ",chunk_size=70,
#                                   chunk_overlap=0)
# chunks = splitter.split_text(file_path)
# for i in range(0, len(chunks)):
#     print(f"{i+1} ---- {len(chunks[i])} ---- {chunks[i]}")


# ---------- Reading PDF File ----------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

pdf_file_path = "./dataset/mediarelease-en.pdf"
pdf_loader = PyPDFLoader(pdf_file_path)
docs = pdf_loader.load()

splitter = CharacterTextSplitter(separator=" ", chunk_size=500, chunk_overlap=00)

chunks = splitter.split_documents(docs)


for i in range(0, len(chunks)):
    if i == 3:
        break
    # print("###DOC####",docs[i])

    print("\n\n###CHUNK####", chunks[i].page_content)

type(docs)
type(chunks)
