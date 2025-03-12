import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import uuid

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def create_temp_db():
    persist_dir = os.path.join(os.getcwd(), "chroma_db_insurance")
    client = chromadb.PersistentClient(path=persist_dir)
    
    # Create a unique collection for this session
    collection_name = f"insurance_{uuid.uuid4().hex[:8]}"
    collection = client.create_collection(name=collection_name)
    
    return collection

def add_to_db(collection, chunks):
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            ids=[f"doc_{i}"],
            embeddings=[embedding_model.encode(chunk)]
        )

def search_similar(collection, query, top_k=5):
    query_embedding = embedding_model.encode(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    docs = results["documents"][0]
    
    # Filter context chunks to those most likely to contain the name in question
    name_parts = query.lower().split()
    filtered = [doc for doc in docs if any(name in doc.lower() for name in name_parts)]

    return filtered if filtered else docs  # Fallback if no match

