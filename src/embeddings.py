from typing import List
import os


class EmbeddingManager:
    def __init__(self):
        print("Initializing embeddings...")

        try:
            from sentence_transformers import SentenceTransformer

            # Try local cache first to avoid SSL download issues
            local_model_path = r"C:\Users\hbhukya\Desktop\rag-pipeline\models\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf"

            # Check if model.safetensors exists (critical file)
            if os.path.exists(os.path.join(local_model_path, "model.safetensors")):
                print(f"Loading model from local cache: {local_model_path}")
                self.model = SentenceTransformer(local_model_path, device='cpu')
                print("Using sentence-transformers for semantic search")
                self._is_tfidf = False
            else:
                # Local cache incomplete, raise exception to trigger fallback
                raise Exception("Local model cache incomplete, using fallback")

        except Exception as e:
            print(f"Failed to load sentence-transformers: {e}")
            print("Using fallback TF-IDF embeddings (no download required)...")
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.model = TfidfVectorizer(max_features=384, ngram_range=(1, 2))
            self._is_tfidf = True
            self._fitted = False
            self._all_texts = []
            return
        self._is_tfidf = False

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self._is_tfidf:
            import numpy as np
            self._all_texts.extend(texts)
            if not self._fitted:
                self.model.fit(self._all_texts)
                self._fitted = True
            vectors = self.model.transform(texts).toarray()
            if vectors.shape[1] < 384:
                vectors = np.pad(vectors, ((0, 0), (0, 384 - vectors.shape[1])))
            elif vectors.shape[1] > 384:
                vectors = vectors[:, :384]
            return vectors.tolist()
        return self.model.encode(texts, convert_to_numpy=True).tolist()

