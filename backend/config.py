import os
from dotenv import load_dotenv

load_dotenv()

# Ollama models
CHAT_MODEL = os.getenv(
    "OLLAMA_CHAT_MODEL",
    "gemma4:31b-cloud"
)

EMBED_MODEL = os.getenv(
    "OLLAMA_EMBED_MODEL",
    "embeddinggemma"
)

# PDF / RAG settings
PDF_PATH = "data/document.pdf"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

TOP_K = 5

# Chroma cosine distance:
# lower = more similar
SIMILARITY_THRESHOLD = 0.55

COLLECTION_NAME = "pdf_knowledge"
CHROMA_PATH = "data/chroma_db"