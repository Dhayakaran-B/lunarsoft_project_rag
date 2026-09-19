import re

import chromadb
from ollama import Client
from rank_bm25 import BM25Okapi

from config import (
    EMBED_MODEL,
    CHROMA_PATH,
    COLLECTION_NAME,
    TOP_K,
)


client = Client()

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)


# --------------------------------------------------
# Tokenization
# --------------------------------------------------

def tokenize(text):

    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )


# --------------------------------------------------
# Load all documents for BM25
# --------------------------------------------------

def load_documents():

    data = collection.get(
        include=[
            "documents",
            "metadatas"
        ]
    )

    documents = data["documents"]
    metadatas = data["metadatas"]

    return documents, metadatas


# --------------------------------------------------
# Vector Search
# --------------------------------------------------

def vector_search(question, n_results=10):

    response = client.embed(
        model=EMBED_MODEL,
        input=question
    )

    query_embedding = response["embeddings"][0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    output = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):

        output.append({
            "text": document,
            "page": metadata["page"],
            "distance": distance
        })

    return output


# --------------------------------------------------
# BM25 Keyword Search
# --------------------------------------------------

def keyword_search(question, n_results=10):

    documents, metadatas = load_documents()

    tokenized_documents = [
        tokenize(doc)
        for doc in documents
    ]

    bm25 = BM25Okapi(tokenized_documents)

    query_tokens = tokenize(question)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for index in ranked_indices[:n_results]:

        if scores[index] <= 0:
            continue

        results.append({
            "text": documents[index],
            "page": metadatas[index]["page"],
            "bm25_score": float(scores[index])
        })

    return results


# --------------------------------------------------
# Hybrid Retrieval
# --------------------------------------------------

def hybrid_search(question):

    vector_results = vector_search(
        question,
        n_results=10
    )

    keyword_results = keyword_search(
        question,
        n_results=10
    )

    combined = {}

    # Vector results
    for rank, result in enumerate(vector_results):

        key = result["text"]

        if key not in combined:

            combined[key] = {
                "text": result["text"],
                "page": result["page"],
                "vector_distance": result["distance"],
                "score": 0
            }

        # Reciprocal Rank Fusion
        combined[key]["score"] += 1 / (60 + rank + 1)

    # Keyword results
    for rank, result in enumerate(keyword_results):

        key = result["text"]

        if key not in combined:

            combined[key] = {
                "text": result["text"],
                "page": result["page"],
                "vector_distance": None,
                "score": 0
            }

        combined[key]["score"] += 1 / (60 + rank + 1)

    # Sort by combined score
    results = sorted(
        combined.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:TOP_K]


# --------------------------------------------------
# Public retrieval function
# --------------------------------------------------

def retrieve(question):

    return hybrid_search(question)