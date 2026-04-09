from pymongo import MongoClient


MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "rag_database"
COLLECTION_NAME = "text_embeddings"

mongo_client = MongoClient(MONGO_URI)
collection = mongo_client[DB_NAME][COLLECTION_NAME]

cursor = collection.find()

for record in cursor:
    print(f"\n\n {len(record["text"])} --- {record["text"]}")
