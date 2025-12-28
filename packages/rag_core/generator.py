"""
Generator - Generación de respuestas con Gemini y citas (JSON estructurado)
Soporta múltiples modelos y streaming.
"""
import json
import re
import time
from typing import Generator

import google.generativeai as genai

from .config import get_settings

# Prompt que fuerza output JSON estructurado
SYSTEM_PROMPT = """Eres un asistente experto en normativa pública peruana.
Tu tarea es responder preguntas usando la información de los documentos proporcionados.

INSTRUCCIONES:
1. Responde basándote en los documentos proporcionados
2. Incluye citas relevantes de los documentos cuando sea posible
3. Si la información está parcialmente disponible, responde con lo que encuentres
4. Solo usa "refusal": true si NO hay absolutamente ninguna información relevante
5. Sé útil y proporciona la mejor respuesta posible
6. Responde en español

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


class GeminiGenerator:
    """Generador de respuestas usando Google Gemini con output JSON"""

    def __init__(self, model_name: str | None = None):
        settings = get_settings()
        genai.configure(api_key=settings.google_api_key)
        self.default_model_name = model_name or settings.gemini_model
        self._models: dict[str, genai.GenerativeModel] = {}

    def _get_model(self, model_name: str | None = None) -> genai.GenerativeModel:
        """Obtiene o crea una instancia del modelo especificado"""
        name = model_name or self.default_model_name
        if name not in self._models:
            self._models[name] = genai.GenerativeModel(name)
        return self._models[name]

    @property
    def model(self) -> genai.GenerativeModel:
        """Modelo por defecto (compatibilidad hacia atrás)"""
        return self._get_model()

    @property
    def model_name(self) -> str:
        """Nombre del modelo por defecto"""
        return self.default_model_name

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
        model_override: str | None = None
    ) -> dict:
        """
        Genera una respuesta estructurada basada en la query y el contexto.

        Args:
            query: Pregunta del usuario
            context_chunks: Lista de chunks recuperados con content y metadata
            max_tokens: Máximo de tokens en la respuesta
            model_override: Modelo específico a usar (opcional, para routing)

        Returns:
            dict con answer, citations, confidence, refusal, latency_ms, etc.
        """
        start_time = time.time()
        used_model = model_override or self.default_model_name

        # Construir prompt
        prompt = self._build_prompt(query, context_chunks)

        # Generar respuesta
        try:
            model = self._get_model(used_model)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.2,  # Más baja para JSON consistente
                )
            )
            raw_response = response.text
        except Exception as e:
            return self._error_response(str(e), time.time() - start_time, used_model)

        # Parsear JSON de la respuesta
        parsed = self._parse_json_response(raw_response)

        # Calcular latencia
        latency_ms = int((time.time() - start_time) * 1000)

        # Enriquecer con metadata de chunks
        enriched_citations = self._enrich_citations(
            parsed.get("citations", []),
            context_chunks
        )

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
            "latency_ms": latency_ms,
            "raw_llm_response": raw_response if parsed.get("_parse_error") else None
        }

    def generate_stream(
        self,
        query: str,
        context_chunks: list[dict],
        max_tokens: int = 1024,
        model_override: str | None = None
    ) -> Generator[str, None, dict]:
        """
        Genera respuesta en modo streaming (para UX mejorada).

        Yields:
            Chunks de texto de la respuesta mientras se genera

        Returns:
            dict final con la respuesta completa parseada
        """
        start_time = time.time()
        used_model = model_override or self.default_model_name

        prompt = self._build_prompt(query, context_chunks)

        try:
            model = self._get_model(used_model)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.2,
                ),
                stream=True
            )

            full_response = ""
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    yield chunk.text

            # Parsear respuesta completa al final
            parsed = self._parse_json_response(full_response)
            latency_ms = int((time.time() - start_time) * 1000)

            enriched_citations = self._enrich_citations(
                parsed.get("citations", []),
                context_chunks
            )

            if parsed.get("confidence") is None:
                parsed["confidence"] = self._calculate_confidence(context_chunks)

            # Retornar resultado final
            return {
                "answer": parsed.get("answer", "Error al procesar respuesta"),
                "citations": enriched_citations,
                "confidence": parsed.get("confidence", 0.0),
                "refusal": parsed.get("refusal", False),
                "notes": parsed.get("notes"),
                "sources_used": len(context_chunks),
                "model": used_model,
                "latency_ms": latency_ms,
            }

        except Exception as e:
            yield f"Error: {str(e)}"
            return self._error_response(str(e), time.time() - start_time, used_model)

    def _parse_json_response(self, text: str) -> dict:
        """
        Extrae y parsea JSON de la respuesta del LLM.
        Maneja casos donde el LLM agrega texto extra.
        """
        # Intentar parsear directamente
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Primero: buscar bloques de código markdown (prioridad más alta)
        # Estos patrones capturan el contenido DENTRO de los backticks
        markdown_patterns = [
            r'```json\s*([\s\S]*?)\s*```',  # ```json ... ```
            r'```\s*([\s\S]*?)\s*```',      # ``` ... ```
        ]

        for pattern in markdown_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    json_content = match.group(1).strip()
                    return json.loads(json_content)
                except json.JSONDecodeError:
                    continue

        # Segundo: buscar JSON directo (sin markdown)
        # Buscar el objeto JSON más externo
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Si no se pudo parsear, retornar respuesta como texto plano
        return {
            "answer": text,
            "citations": [],
            "confidence": 0.5,
            "refusal": False,
            "_parse_error": True
        }

    def _enrich_citations(
        self,
        llm_citations: list[dict],
        context_chunks: list[dict]
    ) -> list[dict]:
        """
        Enriquece las citas del LLM con metadata de los chunks.
        """
        enriched = []

        for citation in llm_citations:
            enriched_citation = {
                "quote": citation.get("quote", ""),
                "source": citation.get("source", "Desconocido"),
                "page": citation.get("page"),
                "source_uri": None,
                "relevance_score": 0.0
            }

            # Buscar chunk correspondiente para agregar metadata
            source_name = citation.get("source", "").lower()
            for chunk in context_chunks:
                chunk_source = chunk["metadata"].get("source", "").lower()
                if source_name in chunk_source or chunk_source in source_name:
                    enriched_citation["source_uri"] = chunk["metadata"].get("source_path")
                    enriched_citation["relevance_score"] = chunk.get("score", 0)
                    break

            enriched.append(enriched_citation)

        # Si no hay citas del LLM, crear citas de los chunks usados
        if not enriched and context_chunks:
            for chunk in context_chunks[:3]:  # Top 3 chunks
                enriched.append({
                    "quote": chunk["content"][:150] + "...",
                    "source": chunk["metadata"].get("source", "Desconocido"),
                    "page": chunk["metadata"].get("page"),
                    "source_uri": chunk["metadata"].get("source_path"),
                    "relevance_score": chunk.get("score", 0)
                })

        return enriched

    def _calculate_confidence(self, chunks: list[dict]) -> float:
        """
        Calcula un score de confianza basado en los chunks recuperados.
        """
        if not chunks:
            return 0.0

        # Promedio de scores de similitud
        scores = [chunk.get("score", 0) for chunk in chunks]
        avg_score = sum(scores) / len(scores)

        # Ajustar por cantidad de chunks con buen score
        high_quality_chunks = sum(1 for s in scores if s > 0.7)
        quality_bonus = min(high_quality_chunks * 0.05, 0.15)

        confidence = min(avg_score + quality_bonus, 1.0)
        return round(confidence, 2)

    def _error_response(self, error_msg: str, elapsed: float, model: str | None = None) -> dict:
        """Respuesta en caso de error"""
        return {
            "answer": f"Error al generar respuesta: {error_msg}",
            "citations": [],
            "confidence": 0.0,
            "refusal": True,
            "notes": "Error interno del sistema",
            "sources_used": 0,
            "model": model or self.default_model_name,
            "latency_ms": int(elapsed * 1000),
            "error": error_msg
        }
