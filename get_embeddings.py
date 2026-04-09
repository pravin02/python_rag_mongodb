import ollama


prompt = input("Please enter text to embbed : ")

response = ollama.embeddings(model="nomic-embed-text-v2-moe", prompt=prompt)

print("\n\n ", response["embedding"])