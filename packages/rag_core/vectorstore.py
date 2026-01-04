"""
Vector Store - Embeddings y ChromaDB
"""

import re
import unicodedata
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
        self, collection_name: str = "rag_documents", persist_dir: str | None = None
    ):
        settings = get_settings()
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name

        # Crear directorio si no existe
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        # Inicializar ChromaDB con persistencia
        self.client = chromadb.PersistentClient(
            path=self.persist_dir, settings=ChromaSettings(anonymized_telemetry=False)
        )

        # Obtener o crear colección con similitud coseno
        # Coseno es más apropiado para embeddings de texto
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "RAG Estado Peru documents",
                "hnsw:space": "cosine",  # Usar similitud coseno
            },
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
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
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

        print(
            f"   [VectorStore.search] hybrid_search={settings.hybrid_search}, query='{query[:50]}...'"
        )

        vector_results = self._vector_search(query, top_k)
        if not settings.hybrid_search:
            print("   [VectorStore.search] Usando SOLO vector search")
            return vector_results

        print(
            f"   [VectorStore.search] Usando HYBRID search (vector_weight={settings.vector_weight}, keyword_weight={settings.keyword_weight})"
        )
        keyword_results = self._keyword_search(query, top_k)
        print(
            f"   [VectorStore.search] keyword_results: {len(keyword_results)} matches"
        )

        merged = self._merge_results(
            vector_results,
            keyword_results,
            top_k,
            settings.vector_weight,
            settings.keyword_weight,
        )
        print(
            f"   [VectorStore.search] merged_results: top scores = {[r.get('score', 0) for r in merged[:3]]}"
        )
        return merged

    def _vector_search(self, query: str, top_k: int) -> list[dict]:
        """Busca por similitud vectorial en ChromaDB."""
        query_embedding = self.embedding_model.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        formatted_results = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            score = max(0.0, min(1.0, 1 - distance))
            formatted_results.append(
                {
                    "chunk_id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": distance,
                    "score": score,
                    "score_vector": score,
                    "score_keyword": 0.0,
                }
            )
        return formatted_results

    def _keyword_search(self, query: str, top_k: int) -> list[dict]:
        """Busca por coincidencias de palabras clave."""
        normalized_query = self._normalize_text(query)
        tokens = self._tokenize(normalized_query)
        phrases = self._extract_phrases(tokens)
        print(f"   [_keyword_search] normalized_query='{normalized_query}'")
        print(f"   [_keyword_search] tokens={tokens}")
        print(f"   [_keyword_search] phrases={phrases}")
        if not tokens:
            print("   [_keyword_search] No tokens found, returning empty")
            return []

        scored = []
        total = self.collection.count()
        batch_size = 500
        offset = 0

        while offset < total:
            data = self.collection.get(
                include=["documents", "metadatas"], limit=batch_size, offset=offset
            )
            ids = data.get("ids", [])
            documents = data.get("documents", [])
            metadatas = data.get("metadatas", [])

            for idx, doc in enumerate(documents):
                doc_text = self._normalize_text(doc or "")
                if not doc_text:
                    continue

                score = 0.0
                token_hits = 0
                for token in tokens:
                    if token in doc_text:
                        hits = doc_text.count(token)
                        score += hits
                        token_hits += 1

                phrase_matches = 0
                for phrase in phrases:
                    if phrase in doc_text:
                        score += len(tokens) * 1.5
                        phrase_matches += 1

                exact_match = normalized_query in doc_text
                if exact_match:
                    score += len(tokens) * 2.0

                if score > 0:
                    scored.append(
                        {
                            "chunk_id": ids[idx],
                            "content": doc,
                            "metadata": metadatas[idx],
                            "distance": None,
                            "score": score,
                            "score_vector": 0.0,
                            "score_keyword": score,
                            "exact_match": exact_match,
                            "token_hits": token_hits,
                            "phrase_matches": phrase_matches,
                        }
                    )

            offset += batch_size

        print(f"   [_keyword_search] Total scored matches: {len(scored)}")
        if scored:
            print(
                f"   [_keyword_search] Top match score: {scored[0]['score'] if scored else 0}"
            )
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _merge_results(
        self,
        vector_results: list[dict],
        keyword_results: list[dict],
        top_k: int,
        vector_weight: float,
        keyword_weight: float,
    ) -> list[dict]:
        """Combina resultados de vector y keyword."""
        merged = {}

        # Primero, agregar todos los resultados de vector
        for res in vector_results:
            res["score"] = vector_weight * res["score_vector"]
            merged[res["chunk_id"]] = res

        # Normalizar keyword scores
        max_keyword = max((r["score_keyword"] for r in keyword_results), default=0.0)

        # Luego, agregar/combinar keyword results
        for res in keyword_results:
            keyword_norm = res["score_keyword"] / max_keyword if max_keyword else 0.0
            chunk_id = res["chunk_id"]

            if chunk_id in merged:
                # Chunk existe en ambos - combinar scores
                merged[chunk_id]["score_keyword"] = keyword_norm
                merged[chunk_id]["exact_match"] = res.get("exact_match")
                merged[chunk_id]["token_hits"] = res.get("token_hits")
                merged[chunk_id]["phrase_matches"] = res.get("phrase_matches")
                merged[chunk_id]["score"] = (
                    vector_weight * merged[chunk_id]["score_vector"]
                ) + (keyword_weight * keyword_norm)
            else:
                # Chunk solo en keyword - darle un boost significativo
                # Si tiene exact_match, es MUY relevante aunque no esté en vector top-k
                res["score_keyword"] = keyword_norm
                if res.get("exact_match"):
                    # Exact match: score alto garantizado
                    res["score"] = 0.85 + (keyword_weight * keyword_norm)
                elif res.get("phrase_matches", 0) > 0:
                    # Phrase match: score moderado-alto
                    res["score"] = 0.70 + (keyword_weight * keyword_norm)
                else:
                    # Solo token hits: score basado en keyword
                    res["score"] = 0.50 + (keyword_weight * keyword_norm)
                merged[chunk_id] = res

        combined = list(merged.values())
        combined.sort(key=lambda x: x["score"], reverse=True)

        print(
            f"   [_merge_results] Top 3 after merge: {[(c.get('metadata', {}).get('page'), c.get('score'), c.get('exact_match')) for c in combined[:3]]}"
        )

        return combined[:top_k]

    def _normalize_text(self, text: str) -> str:
        """Normaliza texto para comparaciones simples."""
        normalized = unicodedata.normalize("NFD", text)
        normalized = "".join(
            char for char in normalized if unicodedata.category(char) != "Mn"
        )
        normalized = normalized.lower()
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
        return re.sub(r"\s+", " ", normalized).strip()

    def _tokenize(self, text: str) -> list[str]:
        """Tokeniza texto en palabras relevantes."""
        tokens = re.findall(r"\b\w+\b", text)
        stopwords = {
            "de",
            "del",
            "la",
            "el",
            "los",
            "las",
            "un",
            "una",
            "unos",
            "unas",
            "y",
            "o",
            "u",
            "que",
            "como",
            "para",
            "por",
            "sobre",
            "en",
            "al",
            "se",
            "es",
        }
        roman = {
            "i",
            "ii",
            "iii",
            "iv",
            "v",
            "vi",
            "vii",
            "viii",
            "ix",
            "x",
            "xi",
            "xii",
            "xiii",
            "xiv",
            "xv",
            "xvi",
            "xvii",
            "xviii",
            "xix",
            "xx",
        }
        filtered = []
        for token in tokens:
            if token in stopwords:
                continue
            if len(token) >= 3 or token in roman:
                filtered.append(token)
        return filtered

    def _extract_phrases(self, tokens: list[str]) -> list[str]:
        """Construye bigramas y trigramas para boosting."""
        phrases = []
        for i in range(len(tokens) - 1):
            phrases.append(f"{tokens[i]} {tokens[i + 1]}")
        for i in range(len(tokens) - 2):
            phrases.append(f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]}")
        # Deduplicar manteniendo orden
        seen = set()
        deduped = []
        for phrase in phrases:
            if phrase in seen:
                continue
            seen.add(phrase)
            deduped.append(phrase)
        return deduped

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
                "hnsw:space": "cosine",
            },
        )
        print("✓ Vector store limpiado")
