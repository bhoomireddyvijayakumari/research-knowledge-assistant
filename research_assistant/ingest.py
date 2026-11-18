from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from .config import DOCUMENTS_DIR, CHROMA_DIR, EMBEDDING_MODEL, ensure_directories
from .db import upsert_document, initialize_db

def get_vectorstore():
    ensure_directories()
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name="research_documents",
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )

def ingest_pdf(pdf_path: str | Path) -> dict:
    path = Path(pdf_path)
    if not path.exists() or path.suffix.lower() != ".pdf":
        raise ValueError("Provide an existing PDF file.")
    ensure_directories()
    loader = PyPDFLoader(str(path))
    pages = loader.load()
    if not pages:
        raise ValueError(f"No readable text found in {path.name}. Scanned PDFs may need OCR.")
    for page in pages:
        page.metadata["source"] = path.name
        page.metadata["page"] = int(page.metadata.get("page", 0)) + 1
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise ValueError("The PDF did not produce any text chunks.")
    # Re-indexing a filename replaces its prior chunks to avoid duplicates.
    store = get_vectorstore()
    existing = store.get(where={"source": path.name})
    if existing and existing.get("ids"):
        store.delete(ids=existing["ids"])
    store.add_documents(chunks)
    initialize_db()
    upsert_document(path.name, str(path.resolve()), len(pages), len(chunks))
    return {"filename": path.name, "pages": len(pages), "chunks": len(chunks)}

def ingest_directory(directory: str | Path = DOCUMENTS_DIR) -> list[dict]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    for path in sorted(directory.glob("*.pdf")):
        results.append(ingest_pdf(path))
    return results
