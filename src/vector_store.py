import numpy as np
from typing import List, Dict


class VectorStore:
    def __init__(self):
        self.embeddings: List[List[float]] = []
        self.texts: List[str] = []
        self.metadatas: List[Dict] = []

    def add(self, embedding: List[List[float]], text: str, metadata: Dict = None):
        self.embeddings.extend(embedding)
        self.texts.append(text)
        self.metadatas.append(metadata or {})

    def search(self, query_embedding: List[List[float]], top_k: int = 5) -> List[Dict]:
        if not self.embeddings:
            return []

        query_vector = np.array(query_embedding[0])
        embeddings_matrix = np.array(self.embeddings)

        # Compute cosine similarities with zero division protection
        dot_products = np.dot(embeddings_matrix, query_vector)
        norms = np.linalg.norm(embeddings_matrix, axis=1) * np.linalg.norm(query_vector)

        # Avoid division by zero
        norms = np.where(norms == 0, 1e-10, norms)
        similarities = dot_products / norms

        # Replace NaN with 0
        similarities = np.nan_to_num(similarities, nan=0.0)

        # Get top_k indices
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]

        # Return both text and metadata
        return [
            {
                'text': self.texts[i],
                'metadata': self.metadatas[i],
                'score': float(similarities[i])
            }
            for i in top_k_indices
        ]

    def get_all_sources(self) -> List[str]:
        """Get list of unique source documents"""
        sources = set()
        for meta in self.metadatas:
            if isinstance(meta, dict):
                # Try common metadata keys
                src = meta.get('source') or meta.get('file_path') or meta.get('filename')
                if src:
                    sources.add(src)
        return list(sources)