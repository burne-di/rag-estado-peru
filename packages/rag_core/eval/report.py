"""
Generación de reportes de evaluación
"""
import json
from datetime import datetime
from pathlib import Path

from .metrics import AggregatedMetrics


class EvalReporter:
    """Genera reportes de evaluación en diferentes formatos"""

    def __init__(self, metrics: AggregatedMetrics, metadata: dict = None):
        """
        Args:
            metrics: Métricas agregadas de la evaluación
            metadata: Información adicional (modelo, config, etc.)
        """
        self.metrics = metrics
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()

    def to_json(self) -> dict:
        """Genera reporte en formato JSON"""
        report = {
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "summary": self.metrics.to_dict(),
            "details": [r.to_dict() for r in self.metrics.results],
            "thresholds": {
                "hit_at_k_target": 0.70,
                "faithfulness_target": 0.70,
                "passed": self._check_thresholds()
            }
        }
        return report

    def _check_thresholds(self) -> bool:
        """Verifica si se cumplen los umbrales mínimos"""
        return (
            self.metrics.hit_at_k_rate >= 0.70 and
            self.metrics.avg_faithfulness >= 0.70
        )

    def to_markdown(self) -> str:
        """Genera reporte en formato Markdown"""
        m = self.metrics
        passed = "✅ PASSED" if self._check_thresholds() else "❌ FAILED"

        md = f"""# Reporte de Evaluación RAG

**Fecha:** {self.timestamp}
**Estado:** {passed}

## Resumen

| Métrica | Valor | Target |
|---------|-------|--------|
| Hit@K Rate | {m.hit_at_k_rate:.2%} | ≥70% |
| Faithfulness | {m.avg_faithfulness:.2%} | ≥70% |
| Precision | {m.avg_precision:.2%} | - |
| Recall | {m.avg_recall:.2%} | - |
| Answer Relevance | {m.avg_answer_relevance:.2%} | - |
| Avg Latency | {m.avg_latency_ms:.0f}ms | - |
| Refusal Rate | {m.refusal_rate:.2%} | - |

## Configuración

| Parámetro | Valor |
|-----------|-------|
| Total Items | {m.total_items} |
| Modelo | {self.metadata.get('model', 'N/A')} |
| Embedding | {self.metadata.get('embedding_model', 'N/A')} |
| Chunk Size | {self.metadata.get('chunk_size', 'N/A')} |
| Top K | {self.metadata.get('top_k', 'N/A')} |

## Detalle por Pregunta

| # | Pregunta | Hit | Faith | Conf | Latency |
|---|----------|-----|-------|------|---------|
"""
        for i, r in enumerate(m.results, 1):
            hit = "✓" if r.hit_at_k else "✗"
            question = r.question[:40] + "..." if len(r.question) > 40 else r.question
            md += f"| {i} | {question} | {hit} | {r.faithfulness:.2f} | {r.confidence:.2f} | {r.latency_ms}ms |\n"

        md += """
## Análisis

### Fortalezas
"""
        if m.hit_at_k_rate >= 0.7:
            md += "- ✅ Retrieval efectivo (Hit@K ≥ 70%)\n"
        if m.avg_faithfulness >= 0.7:
            md += "- ✅ Respuestas fieles al contexto\n"
        if m.avg_latency_ms < 3000:
            md += "- ✅ Latencia aceptable\n"
        if m.refusal_rate < 0.3:
            md += "- ✅ Tasa de rechazo baja\n"

        md += "\n### Áreas de Mejora\n"
        if m.hit_at_k_rate < 0.7:
            md += "- ⚠️ Mejorar retrieval (ajustar embeddings o chunk size)\n"
        if m.avg_faithfulness < 0.7:
            md += "- ⚠️ Ajustar prompts para mayor fidelidad\n"
        if m.avg_latency_ms >= 3000:
            md += "- ⚠️ Optimizar latencia\n"
        if m.refusal_rate >= 0.3:
            md += "- ⚠️ Alta tasa de rechazo - revisar cobertura documental\n"

        md += """
---
*Generado automáticamente por RAG Estado Peru Evaluator*
"""
        return md

    def save(self, output_dir: str | Path, name_prefix: str = "eval"):
        """
        Guarda reporte en JSON y Markdown.

        Args:
            output_dir: Directorio de salida
            name_prefix: Prefijo para nombres de archivo
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Guardar JSON
        json_path = output_dir / f"{name_prefix}_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=2)
        print(f"✓ Reporte JSON guardado: {json_path}")

        # Guardar Markdown
        md_path = output_dir / f"{name_prefix}_{timestamp}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(self.to_markdown())
        print(f"✓ Reporte Markdown guardado: {md_path}")

        return json_path, md_path
