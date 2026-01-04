"""
Generator - Generación de respuestas con múltiples providers (Gemini, Groq)
Soporta múltiples modelos, streaming y fallback automático.
"""

import json
import re
import time
from typing import Generator as GenType
from typing import Optional

from .config import get_settings
from .providers import get_available_providers, get_provider

# Prompt que fuerza output JSON estructurado
SYSTEM_PROMPT = """Eres un asistente experto en normativa pública peruana.
Tu tarea es responder preguntas usando la información de los documentos proporcionados.

INSTRUCCIONES:
1. Responde basándote en los documentos proporcionados
2. Incluye citas relevantes de los documentos cuando sea posible
3. Si la información está parcialmente disponible, responde con lo que encuentres
4. Si la pregunta es general (ej. "temas principales"), responde con un resumen sustentado
5. Solo usa "refusal": true si NO hay absolutamente ninguna información relevante
6. El campo "answer" NO puede estar vacío y debe contener al menos 2-3 oraciones
7. Sé útil y proporciona la mejor respuesta posible
8. Responde en español

Responde con un JSON válido con esta estructura:
{
  "answer": "tu respuesta aquí",
  "citations": [
    {
      "quote": "cita del documento",
      "source": "nombre del documento",
      "page": número de página
    }
  ],
  "confidence": 0.0 a 1.0,
  "refusal": false,
  "notes": "opcional: aclaraciones"
}

NOTAS:
- "confidence" refleja tu certeza (0.0 = baja, 1.0 = alta)
- Prioriza dar una respuesta útil sobre rechazar
- Las citas pueden ser paráfrasis si no encuentras texto exacto
- NO uses bloques de código markdown (```), responde SOLO con el JSON puro
- NO agregues texto fuera del JSON"""


class MultiProviderGenerator:
    """
    Generador de respuestas con soporte multi-provider.

    Soporta:
    - Gemini (Google)
    - Groq (LPU, ultra-rápido)

    Con fallback automático si un provider falla.
    """

    def __init__(self, provider_name: Optional[str] = None):
        """
        Args:
            provider_name: "gemini", "groq", o None para auto-detect
        """
        settings = get_settings()

        # Determinar provider
        if provider_name:
            self._provider_name = provider_name
        elif settings.llm_provider != "auto":
            self._provider_name = settings.llm_provider
        else:
            self._provider_name = None  # Auto-detect

        self._provider = None
        self._fallback_providers = ["groq", "gemini"]

    @property
    def provider(self):
        """Obtiene el provider actual (lazy loading)"""
        if self._provider is None:
            self._provider = get_provider(self._provider_name)
        return self._provider

    @property
    def provider_name(self) -> str:
        """Nombre del provider actual"""
        return self.provider.provider_name

    @property
    def model_name(self) -> str:
        """Modelo actual del provider"""
        return self.provider.default_model

    def _build_prompt(self, query: str, context_chunks: list[dict]) -> str:
        """Construye el prompt con contexto"""
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk["metadata"].get("source", "Desconocido")
            page = chunk["metadata"].get("page", "?")
            context_parts.append(
                f"[Documento {i}: {source}, Página {page}]\n{chunk['content']}"
            )

        context_text = "\n\n---\n\n".join(context_parts)

        return f"""{SYSTEM_PROMPT}

DOCUMENTOS DE REFERENCIA:
{context_text}

PREGUNTA DEL USUARIO:
{query}

Responde SOLO con el JSON estructurado:"""

    def generate(
        self,
        query: str,
        context_chunks: list[dict],
        max_tokens: int = 1024,
        model_override: Optional[str] = None,
        provider_override: Optional[str] = None,
    ) -> dict:
        """
        Genera una respuesta estructurada basada en la query y el contexto.

        Args:
            query: Pregunta del usuario
            context_chunks: Lista de chunks recuperados
            max_tokens: Máximo de tokens en la respuesta
            model_override: Modelo específico a usar
            provider_override: Provider específico ("gemini" o "groq")

        Returns:
            dict con answer, citations, confidence, etc.
        """
        start_time = time.time()

        # Determinar provider
        if provider_override:
            provider = get_provider(provider_override)
        else:
            provider = self.provider

        # Determinar modelo
        used_model = model_override or provider.default_model

        # Construir prompt
        prompt = self._build_prompt(query, context_chunks)

        # Generar respuesta con fallback
        try:
            response = provider.generate(
                prompt=prompt,
                model=used_model,
                max_tokens=max_tokens,
                temperature=0.2,
            )
            raw_response = response.text
            used_provider = response.provider
            used_model = response.model

        except Exception as e:
            # Intentar fallback a otro provider
            fallback_result = self._try_fallback(
                prompt, max_tokens, exclude=provider.provider_name
            )
            if fallback_result:
                raw_response = fallback_result["text"]
                used_provider = fallback_result["provider"]
                used_model = fallback_result["model"]
            else:
                return self._error_response(
                    str(e), time.time() - start_time, used_model, provider.provider_name
                )

        # Parsear JSON de la respuesta
        parsed = self._parse_json_response(raw_response)

        # Calcular latencia
        latency_ms = int((time.time() - start_time) * 1000)

        # Enriquecer con metadata de chunks
        enriched_citations = self._enrich_citations(
            parsed.get("citations", []), context_chunks
        )

        if not (parsed.get("answer") or "").strip():
            parsed["answer"] = self._fallback_answer_from_chunks(context_chunks)

        # Calcular confidence basado en scores de retrieval si no viene
        if parsed.get("confidence") is None:
            parsed["confidence"] = self._calculate_confidence(context_chunks)

        return {
            "answer": parsed.get("answer", "Error al procesar respuesta"),
            "citations": enriched_citations,
            "confidence": parsed.get("confidence", 0.0),
            "refusal": parsed.get("refusal", False),
            "notes": parsed.get("notes"),
            "sources_used": len(context_chunks),
            "model": used_model,
            "provider": used_provider,
            "latency_ms": latency_ms,
            "raw_llm_response": raw_response if parsed.get("_parse_error") else None,
        }

    def generate_stream(
        self,
        query: str,
        context_chunks: list[dict],
        max_tokens: int = 1024,
        model_override: Optional[str] = None,
        provider_override: Optional[str] = None,
    ) -> GenType[str, None, dict]:
        """
        Genera respuesta en modo streaming.

        Yields:
            Chunks de texto mientras se genera

        Returns:
            dict final con la respuesta completa
        """
        start_time = time.time()

        # Determinar provider
        if provider_override:
            provider = get_provider(provider_override)
        else:
            provider = self.provider

        used_model = model_override or provider.default_model
        used_provider = provider.provider_name

        prompt = self._build_prompt(query, context_chunks)

        try:
            full_response = ""
            for chunk in provider.generate_stream(
                prompt=prompt,
                model=used_model,
                max_tokens=max_tokens,
                temperature=0.2,
            ):
                full_response += chunk
                yield chunk

            # Parsear respuesta completa al final
            parsed = self._parse_json_response(full_response)
            latency_ms = int((time.time() - start_time) * 1000)

            enriched_citations = self._enrich_citations(
                parsed.get("citations", []), context_chunks
            )

            if parsed.get("confidence") is None:
                parsed["confidence"] = self._calculate_confidence(context_chunks)

            return {
                "answer": parsed.get("answer", "Error al procesar respuesta"),
                "citations": enriched_citations,
                "confidence": parsed.get("confidence", 0.0),
                "refusal": parsed.get("refusal", False),
                "notes": parsed.get("notes"),
                "sources_used": len(context_chunks),
                "model": used_model,
                "provider": used_provider,
                "latency_ms": latency_ms,
            }

        except Exception as e:
            yield f"Error: {str(e)}"
            return self._error_response(
                str(e), time.time() - start_time, used_model, used_provider
            )

    def _try_fallback(
        self, prompt: str, max_tokens: int, exclude: str
    ) -> Optional[dict]:
        """Intenta usar un provider alternativo"""
        for provider_name in self._fallback_providers:
            if provider_name == exclude:
                continue
            try:
                provider = get_provider(provider_name)
                if provider.is_available():
                    response = provider.generate(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=0.2,
                    )
                    print(f"⚠️ Fallback a {provider_name} exitoso")
                    return {
                        "text": response.text,
                        "provider": response.provider,
                        "model": response.model,
                    }
            except Exception:
                continue
        return None

    def _parse_json_response(self, text: str) -> dict:
        """Extrae y parsea JSON de la respuesta del LLM"""
        # Intentar parsear directamente
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Buscar bloques de código markdown
        markdown_patterns = [
            r"```json\s*([\s\S]*?)\s*```",
            r"```\s*([\s\S]*?)\s*```",
        ]

        for pattern in markdown_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    json_content = match.group(1).strip()
                    return json.loads(json_content)
                except json.JSONDecodeError:
                    continue

        # Buscar JSON directo
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Fallback: retornar respuesta como texto plano
        return {
            "answer": text,
            "citations": [],
            "confidence": 0.5,
            "refusal": False,
            "_parse_error": True,
        }

    def _enrich_citations(
        self, llm_citations: list[dict], context_chunks: list[dict]
    ) -> list[dict]:
        """Enriquece las citas del LLM con metadata de los chunks"""
        enriched = []

        for citation in llm_citations:
            enriched_citation = {
                "quote": citation.get("quote", ""),
                "source": citation.get("source", "Desconocido"),
                "page": citation.get("page"),
                "source_uri": None,
                "relevance_score": 0.0,
            }

            source_name = citation.get("source", "").lower()
            for chunk in context_chunks:
                chunk_source = chunk["metadata"].get("source", "").lower()
                if source_name in chunk_source or chunk_source in source_name:
                    enriched_citation["source_uri"] = chunk["metadata"].get(
                        "source_path"
                    )
                    enriched_citation["relevance_score"] = chunk.get("score", 0)
                    break

            enriched.append(enriched_citation)

        # Si no hay citas del LLM, crear citas de los chunks usados
        if not enriched and context_chunks:
            for chunk in context_chunks[:3]:
                enriched.append(
                    {
                        "quote": chunk["content"][:150] + "...",
                        "source": chunk["metadata"].get("source", "Desconocido"),
                        "page": chunk["metadata"].get("page"),
                        "source_uri": chunk["metadata"].get("source_path"),
                        "relevance_score": chunk.get("score", 0),
                    }
                )

        return enriched

    def _calculate_confidence(self, chunks: list[dict]) -> float:
        """Calcula score de confianza basado en chunks recuperados"""
        if not chunks:
            return 0.0

        scores = [chunk.get("score", 0) for chunk in chunks]
        avg_score = sum(scores) / len(scores)

        high_quality_chunks = sum(1 for s in scores if s > 0.7)
        quality_bonus = min(high_quality_chunks * 0.05, 0.15)

        confidence = min(avg_score + quality_bonus, 1.0)
        return round(confidence, 2)

    def _fallback_answer_from_chunks(self, chunks: list[dict]) -> str:
        """Construye un resumen simple cuando el LLM no devuelve respuesta."""
        if not chunks:
            return "No se encontro informacion relevante en los documentos disponibles."

        snippets = []
        for chunk in chunks[:3]:
            text = (chunk.get("content") or "").strip()
            if not text:
                continue
            clean = re.sub(r"\s+", " ", text)
            snippet = clean[:280]
            cut = snippet.rfind(".")
            if cut >= 80:
                snippet = snippet[: cut + 1]
            snippets.append(snippet)

        if not snippets:
            return "No se encontro informacion relevante en los documentos disponibles."

        return "Resumen basado en documentos: " + " ".join(snippets)

    def _error_response(
        self,
        error_msg: str,
        elapsed: float,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> dict:
        """Respuesta en caso de error"""
        return {
            "answer": f"Error al generar respuesta: {error_msg}",
            "citations": [],
            "confidence": 0.0,
            "refusal": True,
            "notes": "Error interno del sistema",
            "sources_used": 0,
            "model": model or "unknown",
            "provider": provider or "unknown",
            "latency_ms": int(elapsed * 1000),
            "error": error_msg,
        }

    @staticmethod
    def get_available_providers() -> list[str]:
        """Retorna los providers disponibles"""
        return get_available_providers()


# Alias para compatibilidad hacia atrás
GeminiGenerator = MultiProviderGenerator
