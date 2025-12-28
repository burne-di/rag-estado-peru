"""
Métricas de evaluación RAG
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MetricsResult:
    """Resultado de métricas para un item"""

    question: str
    hit_at_k: bool  # ¿Fuente esperada en top-k?
    retrieval_precision: float  # Precisión de retrieval
    retrieval_recall: float  # Recall de retrieval
    faithfulness: float  # ¿Respuesta fiel al contexto?
    answer_relevance: float  # ¿Respuesta relevante a pregunta?
    latency_ms: int
    sources_retrieved: list[str]
    sources_expected: list[str]
    grounding_score: float = 0.0
    confidence: float = 0.0
    was_refusal: bool = False

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "hit_at_k": self.hit_at_k,
            "retrieval_precision": self.retrieval_precision,
            "retrieval_recall": self.retrieval_recall,
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "latency_ms": self.latency_ms,
            "grounding_score": self.grounding_score,
            "confidence": self.confidence,
            "was_refusal": self.was_refusal,
        }


@dataclass
class AggregatedMetrics:
    """Métricas agregadas del dataset"""

    total_items: int = 0
    hit_at_k_rate: float = 0.0
    avg_precision: float = 0.0
    avg_recall: float = 0.0
    avg_faithfulness: float = 0.0
    avg_answer_relevance: float = 0.0
    avg_latency_ms: float = 0.0
    avg_grounding_score: float = 0.0
    avg_confidence: float = 0.0
    refusal_rate: float = 0.0
    results: list[MetricsResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_items": self.total_items,
            "hit_at_k_rate": round(self.hit_at_k_rate, 4),
            "avg_precision": round(self.avg_precision, 4),
            "avg_recall": round(self.avg_recall, 4),
            "avg_faithfulness": round(self.avg_faithfulness, 4),
            "avg_answer_relevance": round(self.avg_answer_relevance, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "avg_grounding_score": round(self.avg_grounding_score, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "refusal_rate": round(self.refusal_rate, 4),
        }


class RAGMetrics:
    """Calcula métricas de calidad RAG"""

    def __init__(self, pipeline):
        """
        Args:
            pipeline: Instancia de RAGPipeline
        """
        self.pipeline = pipeline

    def evaluate_item(
        self,
        question: str,
        expected_sources: list[str],
        gold_answer: Optional[str] = None,
        top_k: int = 5,
    ) -> MetricsResult:
        """
        Evalúa un item individual.

        Args:
            question: Pregunta a evaluar
            expected_sources: Fuentes esperadas en la respuesta
            gold_answer: Respuesta esperada (opcional)
            top_k: Número de chunks a recuperar

        Returns:
            MetricsResult con todas las métricas
        """
        # Ejecutar query
        result = self.pipeline.query(question, top_k=top_k)

        # Extraer fuentes recuperadas
        sources_retrieved = []
        for citation in result.get("citations", []):
            source = citation.get("source", "")
            if source and source not in sources_retrieved:
                sources_retrieved.append(source)

        # Calcular Hit@K
        hit_at_k = any(
            any(
                exp.lower() in src.lower() or src.lower() in exp.lower()
                for src in sources_retrieved
            )
            for exp in expected_sources
        )

        # Calcular Precision y Recall
        if sources_retrieved and expected_sources:
            matches = sum(
                1
                for src in sources_retrieved
                if any(
                    exp.lower() in src.lower() or src.lower() in exp.lower()
                    for exp in expected_sources
                )
            )
            precision = matches / len(sources_retrieved)
            recall = matches / len(expected_sources)
        else:
            precision = 0.0
            recall = 0.0

        # Faithfulness (usar grounding_score si está disponible)
        guardrails = result.get("guardrails", {})
        faithfulness = guardrails.get("grounding_score", 0.5)

        # Answer Relevance (simplificado: usar confidence)
        answer_relevance = result.get("confidence", 0.5)

        return MetricsResult(
            question=question,
            hit_at_k=hit_at_k,
            retrieval_precision=precision,
            retrieval_recall=recall,
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            latency_ms=result.get("latency_ms", 0),
            sources_retrieved=sources_retrieved,
            sources_expected=expected_sources,
            grounding_score=guardrails.get("grounding_score", 0.0),
            confidence=result.get("confidence", 0.0),
            was_refusal=result.get("refusal", False),
        )

    def evaluate_dataset(self, dataset, top_k: int = 5) -> AggregatedMetrics:
        """
        Evalúa un dataset completo.

        Args:
            dataset: EvalDataset con items a evaluar
            top_k: Número de chunks a recuperar

        Returns:
            AggregatedMetrics con métricas agregadas
        """
        results = []

        for item in dataset:
            print(f"Evaluando: {item.question[:50]}...")
            result = self.evaluate_item(
                question=item.question,
                expected_sources=item.expected_sources,
                gold_answer=item.gold_answer,
                top_k=top_k,
            )
            results.append(result)

        # Calcular agregados
        if not results:
            return AggregatedMetrics()

        n = len(results)

        return AggregatedMetrics(
            total_items=n,
            hit_at_k_rate=sum(1 for r in results if r.hit_at_k) / n,
            avg_precision=sum(r.retrieval_precision for r in results) / n,
            avg_recall=sum(r.retrieval_recall for r in results) / n,
            avg_faithfulness=sum(r.faithfulness for r in results) / n,
            avg_answer_relevance=sum(r.answer_relevance for r in results) / n,
            avg_latency_ms=sum(r.latency_ms for r in results) / n,
            avg_grounding_score=sum(r.grounding_score for r in results) / n,
            avg_confidence=sum(r.confidence for r in results) / n,
            refusal_rate=sum(1 for r in results if r.was_refusal) / n,
            results=results,
        )
