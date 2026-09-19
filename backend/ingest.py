import os
import hashlib

import chromadb
from pypdf import PdfReader
from ollama import Client

from config import (
    PDF_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBED_MODEL,
    CHROMA_PATH,
    COLLECTION_NAME,
)


# --------------------------------------------------
# Ollama
# --------------------------------------------------

client = Client()


# --------------------------------------------------
# ChromaDB
# --------------------------------------------------

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


# --------------------------------------------------
# PDF extraction
# --------------------------------------------------

def extract_pdf():

    reader = PdfReader(PDF_PATH)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if not text:
            continue

        text = text.strip()

        if text:
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


# --------------------------------------------------
# Chunking
# --------------------------------------------------

def create_chunks(pages):

    chunks = []

    for page_data in pages:

        text = page_data["text"]
        page_number = page_data["page"]

        # Split by paragraphs
        paragraphs = [
            p.strip()
            for p in text.split("\n")
            if p.strip()
        ]

        current_chunk = ""

        for paragraph in paragraphs:

            # If adding this paragraph stays within the target size
            if len(current_chunk) + len(paragraph) <= CHUNK_SIZE:

                if current_chunk:
                    current_chunk += "\n\n"

                current_chunk += paragraph

            else:

                # Save current chunk
                if current_chunk:

                    chunks.append({
                        "text": current_chunk,
                        "page": page_number
                    })

                # Start a new chunk
                current_chunk = paragraph

        # Save remaining text
        if current_chunk:

            chunks.append({
                "text": current_chunk,
                "page": page_number
            })

    return chunks


# --------------------------------------------------
# Generate embeddings
# --------------------------------------------------

def generate_embedding(text):

    response = client.embed(
        model=EMBED_MODEL,
        input=text
    )

    return response["embeddings"][0]


# --------------------------------------------------
# Generate stable ID
# --------------------------------------------------

def create_id(text, page):

    raw = f"{page}:{text}"

    return hashlib.md5(
        raw.encode("utf-8")
    ).hexdigest()


# --------------------------------------------------
# Store chunks
# --------------------------------------------------

def store_chunks(chunks):

    documents = []
    embeddings = []
    metadatas = []
    ids = []

    for chunk in chunks:

        text = chunk["text"]
        page = chunk["page"]

        print(f"Embedding page {page}...")

        embedding = generate_embedding(text)

        documents.append(text)
        embeddings.append(embedding)

        metadatas.append({
            "page": page
        })

        ids.append(
            create_id(text, page)
        )

    if documents:

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    if not os.path.exists(PDF_PATH):

        print(f"PDF not found: {PDF_PATH}")
        return

    print("Reading PDF...")

    pages = extract_pdf()

    print(f"Pages extracted: {len(pages)}")

    print("Creating chunks...")

    chunks = create_chunks(pages)

    print(f"Chunks created: {len(chunks)}")

    print("Generating embeddings and storing in ChromaDB...")

    store_chunks(chunks)

    print("\nIngestion complete.")
    print(f"Total chunks stored: {len(chunks)}")


if __name__ == "__main__":
    main()