from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)

from app.ingestion import load_papers, split_documents


def create_vectorstore():
    """Create and persist the Chroma vector database."""

    print("Loading papers...")

    documents = load_papers()

    if not documents:
        print("No documents found.")
        return None

    print("Splitting documents...")

    chunks = split_documents(documents)

    print(f"Embedding {len(chunks)} chunks...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    print("\nVector database created successfully.")
    print(f"Location: {CHROMA_DIR}")

    return vectorstore


if __name__ == "__main__":
    create_vectorstore()