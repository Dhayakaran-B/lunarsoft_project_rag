# PDF RAG Test System

A lightweight Retrieval-Augmented Generation (RAG) pipeline for question answering over a provided PDF document.

The purpose of this project is to test how effectively a language model can retrieve information from a document and generate answers that are grounded only in the retrieved document content.

The system is specifically designed to avoid unsupported answers. When the required information cannot be found in the provided document, the system instructs the model to respond:

"I don't know based on the provided document."

---

## Project Overview

This project implements a complete document-based RAG pipeline:

PDF DOCUMENT
    |
    v
PDF Text Extraction
    |
    v
Chunking
    |
    v
EmbeddingGemma
    |
    v
ChromaDB
    |
    v
Hybrid Retrieval
(Vector Search + BM25)
    |
    v
Relevant Context
    |
    v
Ollama LLM
    |
    v
Grounded Answer

---

# Features

- PDF document ingestion
- Page-aware text extraction
- Paragraph-based document chunking
- Semantic embeddings using EmbeddingGemma
- ChromaDB vector storage
- Semantic vector retrieval
- BM25 keyword retrieval
- Hybrid retrieval using vector search + BM25
- Page number tracking
- Context-restricted LLM generation
- Temperature set to 0 for deterministic generation
- "I don't know" fallback for unsupported questions
- Terminal-based question answering
- No external frontend or database server required

---

# Technologies Used

Technology       Purpose
Python           Main programming language
PyPDF            PDF text extraction
Ollama           Local model interface
EmbeddingGemma   Text embeddings
ChromaDB         Vector database
BM25             Keyword-based retrieval
Rank-BM25        BM25 implementation
Python-dotenv    Environment configuration

---

# Project Structure

rag-pdf-test/
|
├── backend/
│   |
│   ├── data/
│   │   ├── document.pdf
│   │   └── chroma_db/
│   |
│   ├── config.py
│   ├── ingest.py
│   ├── retrieve.py
│   ├── generate.py
│   ├── guardrails.py
│   ├── main.py
│   ├── test_retrieval.py
│   └── requirements.txt
|
├── .gitignore
└── README.md

---

# Description of the Files

## config.py

Contains the configuration used throughout the RAG pipeline.

It defines:

- Chat model
- Embedding model
- PDF path
- Chunk size
- Retrieval parameters
- ChromaDB location
- Collection name

Example:

CHAT_MODEL = "gemma4:31b-cloud"
EMBED_MODEL = "embeddinggemma"

---

## ingest.py

Responsible for converting the PDF into searchable information.

The process is:

PDF
 -> Extract text
 -> Extract page numbers
 -> Split text into chunks
 -> Generate embeddings
 -> Store in ChromaDB

The PDF is processed page-by-page so that retrieved information can later be associated with its original page.

---

## retrieve.py

Implements the retrieval stage.

Two retrieval methods are used.

### 1. Vector Search

The question is converted into an embedding using EmbeddingGemma.

ChromaDB then finds document chunks that are semantically similar to the question.

### 2. BM25 Search

BM25 performs keyword-based retrieval.

This is useful for questions involving:

- Names
- Dates
- Places
- Specific terminology
- Exact historical events

For example:

Who was Warren Hastings?

Keyword retrieval can directly identify chunks containing:

Warren
Hastings

while vector retrieval can identify semantically related passages.

### Hybrid Retrieval

The results from both methods are combined using Reciprocal Rank Fusion (RRF).

Vector Search
+
BM25 Search
    |
    v
Hybrid ranking
    |
    v
Top relevant chunks

---

## generate.py

Responsible for sending the retrieved context to the Ollama language model.

The model is explicitly instructed to:

- Use only the retrieved document context
- Avoid general knowledge
- Avoid guessing
- Avoid inventing information
- Include page references
- Respond with "I don't know based on the provided document." when sufficient evidence is unavailable

The generation temperature is set to:

temperature = 0

This reduces unnecessary variation in responses.

---

## guardrails.py

Contains the basic grounding checks used by the application.

The goal is to prevent the system from producing an answer when there is insufficient evidence in the retrieved document.

---

## main.py

The main terminal application.

Running:

python main.py

starts an interactive question-answering session.

Example:

============================================================
PDF RAG TEST SYSTEM
============================================================

Ask a question (or type 'exit'):

The user can then ask questions about the document.

---

## test_retrieval.py

Used to test the retrieval system independently from the language model.

This is useful for determining whether the RAG pipeline is finding the correct parts of the document.

It displays:

- Retrieved page
- Retrieval score
- Retrieved text

This makes it possible to debug retrieval quality before evaluating the LLM.

---

# Document Used

The test document used during development was a detailed historical PDF covering British India / British rule in India.

The document was selected because it contains a large number of:

- Historical events
- People
- Dates
- Locations
- Political developments
- Military conflicts
- Administrative changes

This makes it useful for testing semantic retrieval as well as exact entity retrieval.

The document used during testing contained:

44 pages

The original implementation produced 178 chunks.

After improving the chunking strategy to preserve paragraph boundaries, the document was re-ingested into a smaller set of more semantically meaningful chunks.

The exact number of chunks may vary depending on the document formatting.

---

# Why Hybrid Retrieval?

A purely vector-based RAG system does not always perform well when questions contain specific entities.

For example:

Who was Warren Hastings?

A semantic search system may retrieve conceptually related historical content without necessarily prioritizing the exact passage mentioning Warren Hastings.

BM25 helps by explicitly considering keyword matches.

Therefore:

Vector Search

handles semantic similarity, while:

BM25

handles lexical similarity.

Combining them provides a more robust retrieval mechanism.

---

# Hallucination Control

The primary objective of this project is grounded question answering.

The system does not attempt to make the language model know everything.

Instead, the model is given a restricted context:

QUESTION
   +
RETRIEVED DOCUMENT CHUNKS

The system prompt instructs the model that the retrieved document is its only source of information.

For example, if the PDF contains information about Warren Hastings:

Question:
Who was Warren Hastings?

-> Retrieve relevant chunks
-> Send chunks to LLM
-> Generate answer

But if the user asks:

What is the capital of Japan?

and the document contains no information about Japan, the desired response is:

"I don't know based on the provided document."

This prevents the model from answering from its pretrained general knowledge.

---

# Important Limitation

RAG does not mathematically guarantee zero hallucinations.

The system reduces hallucinations through multiple layers:

User Question
     |
     v
Query Embedding
     |
     v
Hybrid Retrieval
     |
     v
Relevant Context
     |
     v
Context Filtering
     |
     v
Ollama LLM
     |
     v
Grounded Generation

The quality of the final answer therefore depends on:

1. PDF text extraction
2. Chunking quality
3. Embedding quality
4. Retrieval quality
5. Retrieved context
6. LLM behavior
7. Grounding instructions

The project is intended as a test and experimentation pipeline rather than a guarantee of zero hallucinations.

---

# Requirements

## Software

- Python 3.10+
- Ollama
- Git
- Windows/Linux/macOS

## Models

The project uses:

### Embedding model

embeddinggemma

### Language model

gemma4:31b-cloud

The exact chat model can be changed in config.py.

---

# Installation

## 1. Clone the repository

git clone YOUR_REPOSITORY_URL

Then:

cd rag-pdf-test

---

## 2. Create a virtual environment

Navigate to the backend:

cd backend

Create the virtual environment:

python -m venv .venv

### Windows

Activate it using PowerShell:

.\.venv\Scripts\Activate.ps1

You should see:

(.venv)

at the beginning of your terminal.

### Linux/macOS

source .venv/bin/activate

---

# 3. Install Python dependencies

Run:

pip install -r requirements.txt

The main dependencies include:

fastapi
uvicorn
pypdf
chromadb
ollama
python-dotenv
numpy
rank-bm25

FastAPI is included in the environment because it was explored during development, although the final submission version uses the terminal-based application.

---

# 4. Install the Ollama embedding model

Make sure Ollama is installed and running.

Then:

ollama pull embeddinggemma

Verify the model:

ollama list

You should see:

embeddinggemma

---

# 5. Configure the document

Place your PDF inside:

backend/data/document.pdf

The application expects the default path:

data/document.pdf

This can be changed in:

config.py

---

# 6. Ingest the document

From the backend directory:

python ingest.py

The program will:

1. Read the PDF
2. Extract the text
3. Preserve page information
4. Create document chunks
5. Generate embeddings
6. Store the embeddings in ChromaDB

Example:

Reading PDF...
Pages extracted: 44

Creating chunks...
Chunks created: XX

Generating embeddings and storing in ChromaDB...

Embedding page 1...
Embedding page 2...
...

Ingestion complete.
Total chunks stored: XX

---

# 7. Test retrieval

Before asking the language model questions, retrieval can be tested independently:

python test_retrieval.py

Enter a question about the document.

The program displays the retrieved pages and text.

This allows retrieval quality to be examined separately from LLM generation.

---

# 8. Run the RAG application

Run:

python main.py

You should see:

============================================================
PDF RAG TEST SYSTEM
============================================================

Ask a question (or type 'exit'):

Ask a question about the document.

To stop the program:

exit

---

# Example Interaction

Ask a question:
Who was Warren Hastings?

The system:

1. Embeds the question
2. Searches ChromaDB
3. Performs BM25 retrieval
4. Combines the retrieval results
5. Selects relevant document chunks
6. Sends those chunks to the Ollama model
7. Generates a grounded response

The response also provides the pages used as sources.

---

# Testing Hallucination Resistance

A useful evaluation strategy is to test three categories of questions.

## Category 1 - Answer explicitly exists

Example:

Who was Warren Hastings?

Expected:

A document-supported answer.

---

## Category 2 - Answer does not exist

Example:

What is the capital of Japan?

Expected:

I don't know based on the provided document.

---

## Category 3 - Related but unsupported

Example:

How much money did Britain earn from India in a particular year?

if the document does not provide that figure.

Expected:

I don't know based on the provided document.

This third category is particularly important because the model may have general knowledge about the subject but should still restrict itself to the provided document.

---

# Re-ingesting a New Document

If you want to test another PDF:

1. Replace:

backend/data/document.pdf

2. Remove the existing ChromaDB data:

backend/data/chroma_db/

3. Run:

python ingest.py

4. Start the application:

python main.py

This creates a new vector index for the new document.

---

# Configuration

The main settings can be modified in:

config.py

For example:

CHAT_MODEL = "gemma4:31b-cloud"

EMBED_MODEL = "embeddinggemma"

CHUNK_SIZE = 1200

TOP_K = 5

COLLECTION_NAME = "pdf_knowledge"

These values can be changed to experiment with retrieval performance.

---

# Development Process

The project was developed incrementally.

### Stage 1 - PDF Processing

PDF -> Text -> Chunks

### Stage 2 - Semantic Retrieval

Chunks -> Embeddings -> ChromaDB

### Stage 3 - Retrieval Testing

Question -> Embedding -> Relevant Chunks

### Stage 4 - Hybrid Retrieval

Vector Search
      +
BM25
      |
      v
Combined Retrieval

### Stage 5 - LLM Generation

Retrieved Context
       +
Question
       |
       v
Ollama
       |
       v
Answer

### Stage 6 - Grounding

The generation prompt was constrained so that the model should not use information outside the retrieved document.

---

# Security and Privacy

The project does not require an API key for the local Ollama setup.

No credentials should be committed to the repository.

Do not upload:

.env

if it contains credentials.

The Python virtual environment should also not be committed:

.venv/

The ChromaDB index is generated locally and does not need to be committed.

Note that the chat model used in this project is a cloud model. Although the application connects to it through the local Ollama interface, retrieved document text sent to a cloud model may be processed by Ollama's cloud infrastructure.

---

# Future Improvements

Possible improvements include:

- Reranking retrieved chunks
- Better semantic chunking
- Cross-encoder reranking
- Query expansion
- Parent-child document retrieval
- Multi-document support
- Metadata filtering
- Retrieval evaluation metrics
- Automated hallucination evaluation
- Web interface
- Streaming responses
- Conversation history
- Document upload through a web UI
- Persistent cloud vector database
- Citation highlighting

---

# Conclusion

This project demonstrates a complete RAG pipeline for document-grounded question answering.

The main design principle is:

"Retrieve evidence first, then generate an answer from that evidence."

Rather than allowing the language model to answer entirely from its pretrained knowledge, the system provides it with relevant document chunks and instructs it to refuse unsupported questions.

The project can therefore be used as a foundation for experimenting with chunking strategies, embedding models, retrieval algorithms, and hallucination-resistant generation.
