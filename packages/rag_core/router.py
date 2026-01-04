"""
Model Router - Selecciona el modelo y provider óptimo según la complejidad de la query.

Soporta múltiples providers:
- Groq: llama-3.3-70b (complejo) / llama-3.1-8b (simple)
- Gemini: gemini-2.5-flash (complejo) / gemini-2.0-flash-lite (simple)

Criterios de complejidad:
- Longitud de la pregunta
- Palabras clave que indican complejidad
- Número de conceptos/entidades
- Presencia de comparaciones, análisis, etc.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .providers import get_available_providers


class ModelTier(Enum):
    """Niveles de modelo disponibles"""

    LITE = "lite"  # Rápido, económico
    STANDARD = "standard"  # Balanceado, más capaz


# Mapeo de provider -> modelos por tier
PROVIDER_MODELS = {
    "groq": {
        ModelTier.LITE: "openai/gpt-oss-120b",
        ModelTier.STANDARD: "openai/gpt-oss-120b",
    },
    "gemini": {
        ModelTier.LITE: "gemini-2.0-flash-lite",
        ModelTier.STANDARD: "gemini-2.5-flash",
    },
}


@dataclass
class RoutingDecision:
    """Resultado del routing"""

    model: str
    provider: str
    tier: ModelTier
    complexity_score: float  # 0.0 a 1.0
    reason: str


class ModelRouter:
    """
    Router inteligente que selecciona el modelo y provider óptimo
    basándose en la complejidad de la query.
    """

    # Palabras que indican queries complejas
    COMPLEX_INDICATORS = [
        # Análisis profundo
        r"\b(analiza|analizar|análisis|compara|comparar|comparación)\b",
        r"\b(diferencia|diferencias|similitud|similitudes)\b",
        r"\b(ventajas?|desventajas?|pros?|contras?)\b",
        r"\b(implicaciones?|consecuencias?|efectos?)\b",
        # Razonamiento
        r"\b(por qué|porque|razón|razones|motivo|motivos)\b",
        r"\b(cómo funciona|cómo se|de qué manera)\b",
        r"\b(explica|explicar|explicación|detalla|detallar)\b",
        # Múltiples elementos
        r"\b(todos los|todas las|cuáles son|enumera|lista)\b",
        r"\b(requisitos|condiciones|pasos|procedimiento)\b",
        # Legal/técnico complejo
        r"\b(jurisprudencia|doctrina|interpretación)\b",
        r"\b(excepciones?|casos especiales|supuestos)\b",
        r"\b(artículo \d+|inciso|literal|numeral)\b",
    ]

    # Palabras que indican queries simples
    SIMPLE_INDICATORS = [
        r"\b(qué es|que es|definición|define)\b",
        r"\b(cuándo|cuando|fecha|plazo)\b",
        r"\b(dónde|donde|lugar)\b",
        r"\b(cuánto|cuanto|monto|cantidad)\b",
        r"\b(quién|quien|responsable)\b",
        r"^(es|son|está|hay)\b",
    ]

    def __init__(
        self,
        complexity_threshold: float = 0.5,
        preferred_provider: Optional[str] = None,
    ):
        """
        Args:
            complexity_threshold: Umbral para usar modelo estándar (0.0-1.0)
            preferred_provider: Provider preferido ("groq", "gemini", o None para auto)
        """
        self.complexity_threshold = complexity_threshold
        self.preferred_provider = preferred_provider

        # Compilar patrones
        self.complex_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.COMPLEX_INDICATORS
        ]
        self.simple_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.SIMPLE_INDICATORS
        ]

    def _get_active_provider(self) -> str:
        """Obtiene el provider activo"""
        if self.preferred_provider:
            return self.preferred_provider

        # Auto-detect: preferir Groq si está disponible
        available = get_available_providers()
        if "groq" in available:
            return "groq"
        elif "gemini" in available:
            return "gemini"
        else:
            raise ValueError("No hay providers disponibles")

    def route(
        self,
        query: str,
        context_chunks: Optional[list[dict]] = None,
        provider_override: Optional[str] = None,
    ) -> RoutingDecision:
        """
        Determina qué modelo y provider usar para una query.

        Args:
            query: Pregunta del usuario
            context_chunks: Chunks recuperados (opcional)
            provider_override: Forzar un provider específico

        Returns:
            RoutingDecision con el modelo y provider seleccionado
        """
        complexity_score = self._calculate_complexity(query, context_chunks)

        # Determinar provider
        provider = provider_override or self._get_active_provider()

        # Determinar tier basado en complejidad
        if complexity_score >= self.complexity_threshold:
            tier = ModelTier.STANDARD
            reason = self._get_complexity_reason(query, complexity_score)
        else:
            tier = ModelTier.LITE
            reason = "Query simple, usando modelo lite"

        # Obtener modelo específico del provider
        model = PROVIDER_MODELS.get(provider, {}).get(tier)

        if not model:
            # Fallback si el provider no tiene el tier
            model = list(PROVIDER_MODELS.get(provider, {}).values())[0]

        return RoutingDecision(
            model=model,
            provider=provider,
            tier=tier,
            complexity_score=complexity_score,
            reason=reason,
        )

    def _calculate_complexity(
        self, query: str, context_chunks: Optional[list[dict]] = None
    ) -> float:
        """
        Calcula un score de complejidad para la query.

        Factores:
        1. Longitud de la query
        2. Indicadores de complejidad
        3. Indicadores de simplicidad (resta)
        4. Cantidad de contexto necesario
        """
        scores = []

        # Factor 1: Longitud (normalizado)
        word_count = len(query.split())
        length_score = min(word_count / 30, 1.0)  # Máximo a 30 palabras
        scores.append(length_score * 0.2)

        # Factor 2: Indicadores de complejidad
        complex_matches = sum(
            1 for pattern in self.complex_patterns if pattern.search(query)
        )
        complex_score = min(complex_matches / 3, 1.0)  # Máximo a 3 matches
        scores.append(complex_score * 0.5)

        # Factor 3: Indicadores de simplicidad (resta puntos)
        simple_matches = sum(
            1 for pattern in self.simple_patterns if pattern.search(query)
        )
        simple_penalty = min(simple_matches / 2, 0.3)  # Máximo penalty 0.3
        scores.append(-simple_penalty)

        # Factor 4: Contexto (si hay muchos chunks, puede ser complejo)
        if context_chunks:
            context_score = min(len(context_chunks) / 10, 1.0) * 0.2
            scores.append(context_score)

        # Calcular score final (clamped a 0-1)
        final_score = max(0.0, min(1.0, sum(scores)))

        return round(final_score, 3)

    def _get_complexity_reason(self, query: str, score: float) -> str:
        """Genera una explicación del routing"""
        reasons = []

        for pattern in self.complex_patterns:
            match = pattern.search(query)
            if match:
                reasons.append(f"'{match.group()}'")
                if len(reasons) >= 2:
                    break

        if reasons:
            return f"Query compleja (score={score:.2f}): contiene {', '.join(reasons)}"
        else:
            return f"Query moderadamente compleja (score={score:.2f})"


# Singleton global
_router_instance: Optional[ModelRouter] = None


def get_router() -> ModelRouter:
    """Obtiene la instancia singleton del router"""
    global _router_instance
    if _router_instance is None:
        _router_instance = ModelRouter()
    return _router_instance
