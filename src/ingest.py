"""
RAG Ingestion Pipeline for RACE
Loads clinical text, splits into chunks, creates embeddings, and stores in ChromaDB
"""

import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration
DATA_FILE = "data/clinical_guidelines.txt"
CHROMA_DB_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000  # Increased to capture complete paragraphs
CHUNK_OVERLAP = 100  # Proportional overlap
COLLECTION_NAME = "clinical_guidelines"


def load_text_file(file_path: str) -> str:
    """Load text content from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def split_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list:
    """Split text into chunks using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks


def create_embeddings_model(model_name: str = EMBEDDING_MODEL):
    """Initialize the sentence transformer embedding model."""
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},  # Change to 'cuda' if GPU available
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings


def ingest_to_chromadb(chunks: list, embeddings, persist_directory: str = CHROMA_DB_DIR):
    """Create ChromaDB vector store and persist it."""
    # Create persistent ChromaDB client
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=COLLECTION_NAME
    )

    return vectorstore


def main():
    """Main ingestion pipeline."""
    print("=" * 60)
    print("RACE: RAG Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Load text
    print(f"\n[1/4] Loading text from: {DATA_FILE}")
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

    text = load_text_file(DATA_FILE)
    print(f"[OK] Loaded {len(text)} characters")

    # Step 2: Split into chunks
    print(f"\n[2/4] Splitting text into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    chunks = split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"[OK] Created {len(chunks)} chunks")

    # Display first chunk as example
    print(f"\nExample chunk:\n{'-'*60}\n{chunks[0][:200]}...\n{'-'*60}")

    # Step 3: Create embeddings model
    print(f"\n[3/4] Initializing embedding model: {EMBEDDING_MODEL}")
    embeddings = create_embeddings_model(EMBEDDING_MODEL)
    print(f"[OK] Embedding model loaded")

    # Step 4: Store in ChromaDB
    print(f"\n[4/4] Storing embeddings in ChromaDB: {CHROMA_DB_DIR}")
    vectorstore = ingest_to_chromadb(chunks, embeddings, CHROMA_DB_DIR)
    print(f"[OK] Successfully stored {len(chunks)} chunks in ChromaDB")

    # Verification
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"Total chunks stored: {len(chunks)}")
    print(f"Database location: {os.path.abspath(CHROMA_DB_DIR)}")
    print(f"Collection name: {COLLECTION_NAME}")

    # Test retrieval
    print("\n[TEST] Running sample query...")
    query = "What is the recommended starting dose of metformin?"
    results = vectorstore.similarity_search(query, k=2)
    print(f"Query: '{query}'")
    print(f"Top result preview:\n{results[0].page_content[:200]}...")

    print("\n[OK] RAG pipeline is ready for use!")


if __name__ == "__main__":
    main()
