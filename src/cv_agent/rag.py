"""RAG pipeline over CV JSON files: load, chunk, embed, index, retrieve."""
import glob
import json
import os
from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from cv_agent.config import (
    CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR_ENV, EMBEDDING_MODEL_NAME, KAGGLE_INPUT_DIR, TOP_K,
)


@dataclass
class CVIndex:
    vectorstore: FAISS
    embeddings: HuggingFaceEmbeddings
    n_documents: int
    n_chunks: int


def find_cv_json_files(exclude_dir: str | None = None) -> list[str]:
    root = os.environ.get(DATA_DIR_ENV) or KAGGLE_INPUT_DIR
    json_files = glob.glob(f"{root}/**/*.json", recursive=True)
    # Don't accidentally pick up the model's own config/tokenizer json files
    if exclude_dir:
        json_files = [f for f in json_files if os.path.dirname(f) != exclude_dir]

    print("Number of CV JSON files found:", len(json_files))
    return json_files


def load_cv_documents(json_files: list[str]) -> list[Document]:
    documents = []

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            text = data.get("text", "")
            if not text:
                continue

            documents.append(
                Document(page_content=text, metadata={"source": json_path})
            )

        except Exception as e:
            print("Error reading:", json_path, "->", e)

    print("Total CV documents loaded:", len(documents))
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)

    print("Total CV documents:", len(documents))
    print("Total chunks:", len(chunks))
    return chunks


def build_embeddings() -> HuggingFaceEmbeddings:
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
    )
    print("Embedding model loaded successfully!")
    return embeddings


def build_cv_index(model_path: str | None = None) -> CVIndex:
    """Load the CVs, chunk them, embed them and build the FAISS index."""
    json_files = find_cv_json_files(exclude_dir=model_path)
    documents = load_cv_documents(json_files)
    chunks = split_documents(documents)
    embeddings = build_embeddings()

    print("Creating FAISS vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("FAISS vector store created successfully!")

    return CVIndex(vectorstore, embeddings, len(documents), len(chunks))


def search_cvs(vectorstore: FAISS, query: str, k: int = TOP_K) -> str:
    """Top-k chunks formatted as 'Source: <file> / Content: <text>' blocks."""
    retrieved_docs = vectorstore.similarity_search(query, k=k)

    results = []
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "Unknown")
        results.append(f"Source: {source}\nContent:\n{doc.page_content}")

    return "\n\n---\n\n".join(results)
