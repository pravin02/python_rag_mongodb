# python_rag_mongodb

Program: chat_with_model.py

Description: This program demostrates how python can communicate with ollama server to get the response from selected model

How to run: python chat_with_model.py


Program: pgvector_test.py

Description: This program demonstrates how python can conmmiunicate with postgress with vector capabilities to store embeddings and multiple search capabilities

Install podman and pull docker image 
docker.io/pgvector/pgvector:0.8.2-pg18-trixie
variables POSTGRES_PASSWORD=postgres

install pgvector and psycopg
pip install pgvector
pip install psycopg2-binary

How to Run: python pgvector_test.py