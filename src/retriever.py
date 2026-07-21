"""
retriever.py
------------
Implements the retrieval half of RAG (Retrieval-Augmented Generation).

Uses TF-IDF + cosine similarity to find the most relevant chunks of a
student's lecture notes for a given question. TF-IDF is chosen deliberately
over a heavyweight embedding model so this project stays fast, dependency-light,
and easy to explain end-to-end in an interview -- while still demonstrating
core NLP/information-retrieval concepts (vectorization, similarity search).
"""

from dataclasses import dataclass
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.ingest import Chunk


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),   # unigrams + bigrams capture short technical phrases
            max_df=0.9,
        )
        corpus = [c.text for c in chunks]
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 4) -> List[RetrievedChunk]:
        """Return the top_k most relevant chunks for a query, ranked by cosine similarity."""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()

        ranked_indices = scores.argsort()[::-1][:top_k]
        results = [
            RetrievedChunk(chunk=self.chunks[i], score=float(scores[i]))
            for i in ranked_indices
            if scores[i] > 0
        ]
        return results
