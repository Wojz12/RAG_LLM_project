"""Re-ranking module using Cross-Encoder models."""

from typing import List
from sentence_transformers import CrossEncoder


class Reranker:
    """Cross-encoder based re-ranker for improving retrieval quality."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize the Reranker with a CrossEncoder model.

        Args:
            model_name: HuggingFace model name for the cross-encoder.
                Defaults to "cross-encoder/ms-marco-MiniLM-L-6-v2".
        """
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[str]:
        """Re-rank documents based on relevance to the query.

        Args:
            query: The search query.
            documents: List of document texts to re-rank.
            top_k: Number of top documents to return.

        Returns:
            List of top_k documents sorted by relevance score (descending).
        """
        if not documents:
            return []

        pairs = [[query, d] for d in documents]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in ranked[:top_k]]

