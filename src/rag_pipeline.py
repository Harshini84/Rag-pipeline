from typing import Dict
from .document_loader import load_documents_from_directory
from langchain.schema import Document
from .embeddings import EmbeddingManager
from .vector_store import VectorStore
from .llm import generate_answer
from .config import DOCS_PATH, TOP_K


class RAGPipeline:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore()
        self.loaded_files = []  # Track loaded file names

    def ingest_documents(self, directory: str = DOCS_PATH):
        result = load_documents_from_directory(directory)
        chunks = result['chunks']
        self.loaded_files = result['sources']

        if not chunks:
            raise ValueError("No documents found in the specified directory.")

        print(f"Starting ingestion of {len(chunks)} chunks...")

        for chunk in chunks:
            text = chunk.page_content
            metadata = chunk.metadata if hasattr(chunk, 'metadata') else {}
            vector = self.embedding_manager.embed_texts([text])
            self.vector_store.add(vector, text, metadata)

        print(f"Ingestion complete: {len(chunks)} chunks stored")
        return {'sources': self.loaded_files}

    def query(self, question: str, show_chunks: bool = False) -> Dict:
        if not self.vector_store.embeddings:
            raise ValueError("The vector store is empty. Please ingest documents first.")

        from .config import MIN_SCORE_THRESHOLD

        query_vector = self.embedding_manager.embed_texts([question])
        retrived_chunks = self.vector_store.search(query_vector, top_k=TOP_K)

        # Filter out low-relevance chunks to reduce noise
        retrived_chunks = [c for c in retrived_chunks if c['score'] >= MIN_SCORE_THRESHOLD]

        # Extract text strings from dict results
        texts = [chunk['text'] for chunk in retrived_chunks]
        sources = [chunk.get('metadata', {}).get('source', 'Unknown') for chunk in retrived_chunks]

        if not texts:
            return {
                "answer": "No relevant information found in the uploaded documents.",
                "source_used": 0,
                "documents": self.loaded_files,
                "chunks": [],
            }

        context = "\n\n".join(texts)

        prompt = f"""You are a precise information extraction assistant analyzing document content.

Your task: Extract ONLY factual information from the context below to answer the question.

STRICT RULES:
1. Answer MUST come directly from the context - never add external knowledge
2. If information is in the context, provide it clearly and completely
3. Use exact numbers, names, dates, and facts from the context
4. If context has partial information, provide what exists and note what's missing
5. ONLY respond "I cannot find this information in the provided documents" if truly absent
6. Do not make assumptions or inferences beyond what's stated
7. If question asks multiple things, address each point separately

CONTEXT FROM DOCUMENTS:
{context}

QUESTION: {question}

ANSWER (based strictly on context above):"""
        answer = generate_answer(prompt)

        # Prepare chunk info if requested
        chunk_info = []
        if show_chunks:
            for i, (text, source) in enumerate(zip(texts, sources)):
                source_name = source.split('\\')[-1] if '\\' in source else source.split('/')[-1]
                chunk_info.append({
                    'index': i + 1,
                    'source': source_name,
                    'text': text[:200] + '...' if len(text) > 200 else text
                })

        return {
            "answer": answer,
            "source_used": len(texts),
            "documents": self.loaded_files,
            "chunks": chunk_info if show_chunks else [],
        }

    def chat(self):
        print("\nRAG Chat")
        print("Type 'exit' or 'quit' to stop\n")
        while True:
            question = input("User: ").strip()
            if question.lower() in ['exit', 'quit']:
                print("Exiting chat.")
                break
            if not question:
                continue
            result = self.query(question)
            print(f"\nAssistant: {result['answer']}\n(Sources used: {result['source_used']})\n")