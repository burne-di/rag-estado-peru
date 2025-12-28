"""
Refusal Policy - Política de rechazo cuando no hay evidencia suficiente
"""

from dataclasses import dataclass
from enum import Enum


class RefusalReason(Enum):
    """Razones de rechazo"""

    NO_CONTEXT = "no_context"
    LOW_RELEVANCE = "low_relevance"
    UNGROUNDED = "ungrounded"
    OFF_TOPIC = "off_topic"
    AMBIGUOUS = "ambiguous"
    NONE = "none"


@dataclass
class RefusalResult:
    """Resultado de la evaluación de refusal"""

    should_refuse: bool
    reason: RefusalReason
    message: str
    suggestion: str | None = None


# Mensajes de rechazo en español
REFUSAL_MESSAGES = {
    RefusalReason.NO_CONTEXT: (
        "No encontré información relevante en los documentos indexados "
        "para responder esta pregunta."
    ),
    RefusalReason.LOW_RELEVANCE: (
        "Los documentos disponibles no contienen información suficientemente "
        "relacionada con tu consulta."
    ),
    RefusalReason.UNGROUNDED: (
        "No puedo proporcionar una respuesta verificable basada en los "
        "documentos disponibles."
    ),
    RefusalReason.OFF_TOPIC: (
        "Tu pregunta parece estar fuera del alcance de los documentos "
        "indexados en este sistema."
    ),
    RefusalReason.AMBIGUOUS: (
        "La pregunta es ambigua. Por favor, reformúlala con más detalle."
    ),
}

SUGGESTIONS = {
    RefusalReason.NO_CONTEXT: "Intenta indexar más documentos relacionados con el tema.",
    RefusalReason.LOW_RELEVANCE: "Prueba reformulando la pregunta con términos más específicos.",
    RefusalReason.UNGROUNDED: "Verifica que los documentos correctos estén indexados.",
    RefusalReason.OFF_TOPIC: "Este sistema solo responde sobre normativa pública peruana.",
    RefusalReason.AMBIGUOUS: "Incluye más contexto o especifica el documento de interés.",
}


class RefusalPolicy:
    """
    Evalúa si se debe rechazar una respuesta basándose en
    múltiples señales de calidad.
    """

    def __init__(
        self,
        min_relevance_score: float = 0.1,  # Bajado para ser menos estricto
        min_grounding_score: float = 0.3,  # Bajado para ser menos estricto
        min_chunks_required: int = 1,
    ):
        """
        Args:
            min_relevance_score: Score mínimo de relevancia de chunks
            min_grounding_score: Score mínimo de grounding
            min_chunks_required: Mínimo de chunks para responder
        """
        self.min_relevance_score = min_relevance_score
        self.min_grounding_score = min_grounding_score
        self.min_chunks_required = min_chunks_required

    def evaluate(
        self,
        chunks: list[dict],
        grounding_score: float | None = None,
        query: str | None = None,
    ) -> RefusalResult:
        """
        Evalúa si se debe rechazar la respuesta.

        Args:
            chunks: Chunks recuperados con scores
            grounding_score: Score de grounding (0-1)
            query: Query original (para análisis adicional)

        Returns:
            RefusalResult con la decisión
        """
        # Check 1: ¿Hay suficientes chunks?
        if len(chunks) < self.min_chunks_required:
            return RefusalResult(
                should_refuse=True,
                reason=RefusalReason.NO_CONTEXT,
                message=REFUSAL_MESSAGES[RefusalReason.NO_CONTEXT],
                suggestion=SUGGESTIONS[RefusalReason.NO_CONTEXT],
            )

        # Check 2: ¿Los chunks son relevantes?
        avg_score = sum(c.get("score", 0) for c in chunks) / len(chunks)
        if avg_score < self.min_relevance_score:
            return RefusalResult(
                should_refuse=True,
                reason=RefusalReason.LOW_RELEVANCE,
                message=REFUSAL_MESSAGES[RefusalReason.LOW_RELEVANCE],
                suggestion=SUGGESTIONS[RefusalReason.LOW_RELEVANCE],
            )

        # Check 3: ¿La respuesta está grounded?
        if grounding_score is not None and grounding_score < self.min_grounding_score:
            return RefusalResult(
                should_refuse=True,
                reason=RefusalReason.UNGROUNDED,
                message=REFUSAL_MESSAGES[RefusalReason.UNGROUNDED],
                suggestion=SUGGESTIONS[RefusalReason.UNGROUNDED],
            )

        # Check 4: ¿La query es válida?
        if query and self._is_off_topic(query):
            return RefusalResult(
                should_refuse=True,
                reason=RefusalReason.OFF_TOPIC,
                message=REFUSAL_MESSAGES[RefusalReason.OFF_TOPIC],
                suggestion=SUGGESTIONS[RefusalReason.OFF_TOPIC],
            )

        # Todo OK - no rechazar
        return RefusalResult(
            should_refuse=False, reason=RefusalReason.NONE, message="", suggestion=None
        )

    def _is_off_topic(self, query: str) -> bool:
        """Detecta queries claramente fuera de tema"""
        off_topic_patterns = [
            # Temas claramente no relacionados
            "receta de cocina",
            "película",
            "canción",
            "clima",
            "deportes",
            "fútbol",
            "chiste",
            "poema",
        ]

        query_lower = query.lower()
        return any(pattern in query_lower for pattern in off_topic_patterns)

    def format_refusal_response(self, result: RefusalResult) -> dict:
        """
        Formatea una respuesta de rechazo en el formato estándar.
        """
        return {
            "answer": result.message,
            "citations": [],
            "confidence": 0.0,
            "refusal": True,
            "refusal_reason": result.reason.value,
            "suggestion": result.suggestion,
        }
