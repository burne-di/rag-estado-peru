"""
Vector Store - Embeddings y ChromaDB
"""
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from .chunker import Chunk
from .config import get_settings


class EmbeddingModel:
    """Wrapper para sentence-transformers"""

    def __init__(self, model_name: str | None = None):
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            print(f"Cargando modelo de embeddings: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Genera embeddings para una lista de textos"""
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Genera embedding para una query"""
        return self.model.encode(query).tolist()


class VectorStore:
    """ChromaDB vector store con persistencia"""

    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_dir: str | None = None
    ):
        settings = get_settings()
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name

        # Crear directorio si no existe
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        # Inicializar ChromaDB con persistencia
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        # Obtener o crear colección con similitud coseno
        # Coseno es más apropiado para embeddings de texto
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "RAG Estado Peru documents",
                "hnsw:space": "cosine"  # Usar similitud coseno
            }
        )

        # Modelo de embeddings
        self.embedding_model = EmbeddingModel()

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """
        Añade chunks al vector store.
        Retorna el número de chunks añadidos.
        """
        if not chunks:
            return 0

        # Preparar datos para ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Generar embeddings
        print(f"Generando embeddings para {len(chunks)} chunks...")
        embeddings = self.embedding_model.embed(documents)

        # Añadir a la colección
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        print(f"✓ Añadidos {len(chunks)} chunks al vector store")
        return len(chunks)

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Busca chunks similares a la query.
        Retorna lista de resultados con content, metadata y score.
        """
        settings = get_settings()
        top_k = top_k or settings.top_k_results

        # Generar embedding de la query
        query_embedding = self.embedding_model.embed_query(query)

        # Buscar en ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        # Formatear resultados
        formatted_results = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            # Convertir distancia coseno a score de similitud (0-1)
            # ChromaDB con coseno retorna: distance = 1 - cosine_similarity
            # Por lo tanto: score = 1 - distance
            # Esto da valores entre 0 (opuesto) y 1 (idéntico)
            score = max(0.0, min(1.0, 1 - distance))

            formatted_results.append({
                "chunk_id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": distance,
                "score": score
            })

        return formatted_results

    def count(self) -> int:
        """Retorna el número de chunks en el store"""
        return self.collection.count()

    def clear(self):
        """Elimina todos los documentos de la colección"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={
                "description": "RAG Estado Peru documents",
                "hnsw:space": "cosine"
            }
        )
        print("✓ Vector store limpiado")
