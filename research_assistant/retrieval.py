
import hashlib
import re
from functools import lru_cache
from flashrank import Ranker, RerankRequest
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from .ingest import get_vectorstore
from .config import TOP_K
from .db import list_documents

@lru_cache(maxsize=1)
def get_reranker():
    """
    Load the reranking model once and reuse it.

    TinyBERT is a lightweight model suitable
    for CPU-based local development.
    """
    return Ranker(
        model_name="ms-marco-TinyBERT-L-2-v2",
        max_length=256
    )

# Reciprocal Rank Fusion constant.
# A larger value reduces the influence of very high individual ranks.
RRF_K = 60


def tokenize(text: str) -> list[str]:
    """
    Convert text into lowercase tokens.

    Keeps technical terms such as:
    MGPSO, BiGRU, XGBoost, and text-embedding-3-small.
    """
    return re.findall(r"[\w.-]+", text.lower())


def document_key(doc: Document) -> str:
    """
    Generate a stable identifier for a document chunk.
    Used to combine semantic and keyword search results.
    """
    source = str(doc.metadata.get("source", "unknown"))
    page = str(doc.metadata.get("page", "unknown"))

    raw = f"{source}|{page}|{doc.page_content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    k: int = RRF_K
) -> list[tuple[Document, float, set[str]]]:
    """
    Combine multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank))

    Higher scores indicate that a chunk ranked well across
    multiple retrieval methods.
    """
    fused = {}

    for method_index, documents in enumerate(ranked_lists):
        method_name = (
            "semantic" if method_index == 0 else "keyword"
        )

        for rank, doc in enumerate(documents, start=1):
            key = document_key(doc)

            if key not in fused:
                fused[key] = {
                    "document": doc,
                    "score": 0.0,
                    "methods": set()
                }

            fused[key]["score"] += 1.0 / (k + rank)
            fused[key]["methods"].add(method_name)

    ordered = sorted(
        fused.values(),
        key=lambda item: item["score"],
        reverse=True
    )

    return [
        (item["document"], item["score"], item["methods"])
        for item in ordered
    ]


def search_documents(
    query: str,
    k: int = TOP_K,
    source: str | None = None
) -> list[dict]:
    """
    Search indexed PDFs using hybrid retrieval.

    Combines:
      1. Semantic search using ChromaDB.
      2. Keyword search using BM25.
      3. Reciprocal Rank Fusion.

    Optional source filtering restricts results to one PDF.
    """

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    store = get_vectorstore()

    # --------------------------------------------------
    # 1. Load indexed document chunks for BM25
    # --------------------------------------------------
    filters = {"source": source} if source else None

    stored = store.get(
        where=filters,
        include=["documents", "metadatas"]
    )

    contents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []

    corpus = [
        Document(
            page_content=content,
            metadata=metadata or {}
        )
        for content, metadata in zip(contents, metadatas)
        if content and content.strip()
    ]

    if not corpus:
        return []

    # --------------------------------------------------
    # 2. Semantic retrieval
    # --------------------------------------------------
    candidate_k = min(len(corpus), max(k * 4, 10))

    semantic_results = store.similarity_search(
        query,
        k=candidate_k,
        filter=filters
    )

    # --------------------------------------------------
    # 3. BM25 keyword retrieval
    # --------------------------------------------------
    tokenized_corpus = [
        tokenize(doc.page_content)
        for doc in corpus
    ]

    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = tokenize(query)

    keyword_scores = bm25.get_scores(query_tokens)

    keyword_indexes = sorted(
        range(len(keyword_scores)),
        key=lambda index: keyword_scores[index],
        reverse=True
    )[:candidate_k]

    keyword_results = [
        corpus[index]
        for index in keyword_indexes
        if keyword_scores[index] > 0
    ]

    # --------------------------------------------------
    # 4. Combine results using Reciprocal Rank Fusion
    # --------------------------------------------------
    fused_results = reciprocal_rank_fusion(
        [semantic_results, keyword_results]
    )

    # --------------------------------------------------
    # 5. Prepare candidate passages for reranking
    # --------------------------------------------------

    candidate_count = min(
        len(fused_results),
        max(k * 4, 10)
    )

    candidates = fused_results[:candidate_count]

    passages = []

    for index, (doc, fusion_score, methods) in enumerate(
        candidates
    ):
        passages.append({
            "id": index,
            "text": doc.page_content,
            "meta": {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "unknown"),
                "fusion_score": fusion_score,
                "retrieval_methods": sorted(methods)
            }
        })

    # --------------------------------------------------
    # 6. Rerank candidate passages
    # --------------------------------------------------

    reranker = get_reranker()

    rerank_request = RerankRequest(
        query=query,
        passages=passages
    )

    reranked_results = reranker.rerank(rerank_request)

    # --------------------------------------------------
    # 7. Return the top-k reranked results
    # --------------------------------------------------

    results = []

    for rank, item in enumerate(reranked_results[:k], start=1):

        metadata = item.get("meta", {})

        results.append({
            "rank": rank,
            "source": metadata.get("source", "unknown"),
            "page": metadata.get("page", "unknown"),
            "content": item.get("text", "")[:2500],
            "retrieval_score": round(
                float(metadata.get("fusion_score", 0.0)),
                6
            ),
            "rerank_score": round(
                float(item.get("score", 0.0)),
                6
            ),
            "retrieval_methods": metadata.get(
                "retrieval_methods", []
            )
        })

    return results


def get_document_list() -> list[dict]:
    """Return the list of indexed PDFs."""
    return list_documents()
