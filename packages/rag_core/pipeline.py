"""
RAG Pipeline - Orquesta todo el flujo de ingesta y consulta con guardrails
"""
import re
import time
import unicodedata
from pathlib import Path

from .cache import get_cache
from .chunker import chunk_documents
from .config import get_settings
from .generator import GeminiGenerator
from .guardrails import GroundingChecker, PIIScrubber, RefusalPolicy
from .loaders import PDFLoader, load_documents_from_directory
from .router import get_router
from .vectorstore import VectorStore


def normalize_query(text: str) -> str:
    """
    Normaliza la query para mejorar la búsqueda.
    - Remueve acentos
    - Normaliza signos de interrogación
    - Limpia espacios extra
    """
    # Normalizar Unicode (NFD separa caracteres base de diacríticos)
    text = unicodedata.normalize('NFD', text)
    # Remover diacríticos (acentos)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    # Reemplazar signos de interrogación invertidos
    text = text.replace('¿', '').replace('¡', '')
    # Limpiar espacios múltiples
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class RAGPipeline:
    """Pipeline completo de RAG con guardrails"""

    def __init__(
        self,
        enable_guardrails: bool = True,
        enable_cache: bool = True,
        enable_routing: bool = True
    ):
        self.settings = get_settings()
        self.vector_store = VectorStore()
        self.generator = GeminiGenerator()

        # Cache para respuestas
        self.enable_cache = enable_cache
        self.cache = get_cache() if enable_cache else None

        # Model routing
        self.enable_routing = enable_routing
        self.router = get_router() if enable_routing else None

        # Guardrails
        self.enable_guardrails = enable_guardrails
        if enable_guardrails:
            self.grounding_checker = GroundingChecker()
            self.refusal_policy = RefusalPolicy()
            self.pii_scrubber = PIIScrubber()

    def ingest_directory(self, directory: str | Path) -> dict:
        """
        Ingesta todos los PDFs de un directorio.

        Returns:
            dict con estadísticas de la ingesta
        """
        print(f"=== Iniciando ingesta desde: {directory} ===")

        # 1. Cargar documentos
        print("\n1. Cargando documentos...")
        documents = load_documents_from_directory(directory)
        print(f"   Total páginas cargadas: {len(documents)}")

        if not documents:
            return {"status": "error", "message": "No se encontraron documentos"}

        # 2. Dividir en chunks
        print("\n2. Dividiendo en chunks...")
        chunks = chunk_documents(
            documents,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap
        )
        print(f"   Total chunks generados: {len(chunks)}")

        # 3. Añadir al vector store
        print("\n3. Generando embeddings y almacenando...")
        added = self.vector_store.add_chunks(chunks)

        print("\n=== Ingesta completada ===")
        print(f"   Documentos procesados: {len(set(d.metadata['source'] for d in documents))}")
        print(f"   Páginas procesadas: {len(documents)}")
        print(f"   Chunks indexados: {added}")
        print(f"   Total en vector store: {self.vector_store.count()}")

        return {
            "status": "success",
            "documents": len(set(d.metadata['source'] for d in documents)),
            "pages": len(documents),
            "chunks": added,
            "total_indexed": self.vector_store.count()
        }

    def ingest_file(self, file_path: str | Path) -> dict:
        """Ingesta un solo archivo PDF"""
        loader = PDFLoader(file_path)
        documents = loader.load()

        chunks = chunk_documents(
            documents,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap
        )

        added = self.vector_store.add_chunks(chunks)

        return {
            "status": "success",
            "file": str(file_path),
            "pages": len(documents),
            "chunks": added
        }

    def query(self, question: str, top_k: int | None = None, skip_cache: bool = False) -> dict:
        """
        Responde una pregunta usando RAG con guardrails.

        Args:
            question: Pregunta del usuario
            top_k: Número de chunks a recuperar
            skip_cache: Si es True, ignora el caché y fuerza nueva generación

        Returns:
            dict con answer, citations, confidence, refusal, etc.
        """
        start_time = time.time()
        top_k = top_k or self.settings.top_k_results

        # 0. Normalizar query para búsqueda más robusta
        normalized_question = normalize_query(question)
        print(f"   Query normalizada: {normalized_question}")

        # 0.1 Verificar caché
        if self.enable_cache and not skip_cache:
            cached_response = self.cache.get(normalized_question)
            if cached_response:
                cached_response["from_cache"] = True
                cached_response["latency_ms"] = int((time.time() - start_time) * 1000)
                return cached_response

        # 1. Scrub PII de la query (para logs)
        if self.enable_guardrails:
            clean_query, pii_found = self.pii_scrubber.scrub(question)
            if pii_found:
                print(f"⚠ PII detectado en query: {len(pii_found)} elementos")

        # 2. Buscar chunks relevantes (usando query normalizada)
        relevant_chunks = self.vector_store.search(normalized_question, top_k=top_k)

        # Debug: mostrar scores de chunks
        if relevant_chunks:
            scores = [c.get("score", 0) for c in relevant_chunks]
            print(f"   Scores de chunks: {[f'{s:.3f}' for s in scores]}")
            print(f"   Score promedio: {sum(scores)/len(scores):.3f}")

        # 3. Evaluar política de rechazo (pre-generación)
        if self.enable_guardrails:
            refusal_result = self.refusal_policy.evaluate(
                chunks=relevant_chunks,
                query=question
            )

            if refusal_result.should_refuse:
                return {
                    **self.refusal_policy.format_refusal_response(refusal_result),
                    "sources_used": len(relevant_chunks),
                    "latency_ms": int((time.time() - start_time) * 1000),
                    "guardrails": {
                        "pre_refusal": True,
                        "reason": refusal_result.reason.value
                    }
                }

        # 4. Model routing (seleccionar modelo óptimo)
        selected_model = None
        routing_info = None
        if self.enable_routing:
            routing_decision = self.router.route(question, relevant_chunks)
            selected_model = routing_decision.model
            routing_info = {
                "selected_model": routing_decision.model,
                "tier": routing_decision.tier.name,
                "complexity_score": routing_decision.complexity_score,
                "reason": routing_decision.reason
            }
            print(f"   🔀 Routing: {routing_decision.model} (score={routing_decision.complexity_score:.2f})")

        # 5. Generar respuesta
        response = self.generator.generate(
            question,
            relevant_chunks,
            model_override=selected_model
        )

        # Agregar info de routing
        if routing_info:
            response["routing"] = routing_info

        # 6. Verificar grounding (post-generación)
        if self.enable_guardrails and not response.get("refusal"):
            grounding_result = self.grounding_checker.check(
                answer=response["answer"],
                context_chunks=relevant_chunks
            )

            # Evaluar nuevamente con grounding score
            post_refusal = self.refusal_policy.evaluate(
                chunks=relevant_chunks,
                grounding_score=grounding_result.score,
                query=question
            )

            if post_refusal.should_refuse:
                return {
                    **self.refusal_policy.format_refusal_response(post_refusal),
                    "sources_used": len(relevant_chunks),
                    "latency_ms": int((time.time() - start_time) * 1000),
                    "guardrails": {
                        "grounding_score": grounding_result.score,
                        "grounding_details": grounding_result.details,
                        "ungrounded_claims": grounding_result.ungrounded_claims[:3],
                        "post_refusal": True
                    }
                }

            # Agregar info de guardrails a la respuesta
            response["guardrails"] = {
                "grounding_score": grounding_result.score,
                "grounding_details": grounding_result.details,
                "is_grounded": grounding_result.is_grounded
            }

        # 7. Scrub PII de la respuesta (para logs)
        if self.enable_guardrails:
            response_for_log = self.pii_scrubber.scrub_for_logs(response)
            # El response original va al usuario, el scrubbed a logs
            response["_log_safe"] = response_for_log

        # Actualizar latencia total
        response["latency_ms"] = int((time.time() - start_time) * 1000)

        # Guardar en caché (solo respuestas exitosas)
        if self.enable_cache and not response.get("refusal"):
            self.cache.set(normalized_question, response)

        response["from_cache"] = False
        return response

    def get_stats(self) -> dict:
        """Retorna estadísticas del pipeline"""
        stats = {
            "total_chunks": self.vector_store.count(),
            "embedding_model": self.settings.embedding_model,
            "llm_model": self.settings.gemini_model,
            "chunk_size": self.settings.chunk_size,
            "top_k": self.settings.top_k_results,
            "guardrails_enabled": self.enable_guardrails,
            "cache_enabled": self.enable_cache,
            "routing_enabled": self.enable_routing
        }

        if self.enable_cache:
            stats["cache_stats"] = self.cache.get_stats()

        if self.enable_routing:
            stats["available_models"] = ["gemini-2.0-flash-lite", "gemini-2.5-flash"]

        return stats

    def clear(self):
        """Limpia el vector store"""
        self.vector_store.clear()
