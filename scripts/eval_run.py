"""
Script para ejecutar evaluación offline del sistema RAG
"""

import argparse
import sys
from pathlib import Path

# Agregar root al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from packages.rag_core import RAGPipeline
from packages.rag_core.eval import EvalDataset, EvalReporter, RAGMetrics


def main():
    parser = argparse.ArgumentParser(
        description="Ejecuta evaluación offline del sistema RAG"
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=None,
        help="Ruta al dataset JSONL (default: genera dataset de ejemplo)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./reports",
        help="Directorio de salida para reportes",
    )
    parser.add_argument(
        "--top-k", "-k", type=int, default=5, help="Número de chunks a recuperar"
    )
    parser.add_argument(
        "--report", action="store_true", help="Generar y guardar reporte"
    )
    parser.add_argument(
        "--create-sample", action="store_true", help="Crear dataset de ejemplo y salir"
    )

    args = parser.parse_args()

    # Crear dataset de ejemplo si se solicita
    if args.create_sample:
        dataset = EvalDataset.create_sample()
        output_path = Path("./data/eval_dataset.jsonl")
        dataset.save(output_path)
        print(f"✓ Dataset de ejemplo creado: {output_path}")
        print(f"  Items: {len(dataset)}")
        print(f"  Stats: {dataset.get_stats()}")
        return

    # Inicializar pipeline
    print("=== Inicializando Pipeline RAG ===")
    pipeline = RAGPipeline()

    stats = pipeline.get_stats()
    print(f"Chunks indexados: {stats['total_chunks']}")
    print(f"Modelo: {stats['llm_model']}")

    if stats["total_chunks"] == 0:
        print("\n⚠ No hay documentos indexados. Ejecuta primero:")
        print("  python scripts/ingest.py")
        return

    # Cargar dataset
    print("\n=== Cargando Dataset ===")
    if args.dataset:
        dataset = EvalDataset.load(args.dataset)
        print(f"Dataset cargado: {args.dataset}")
    else:
        print("Usando dataset de ejemplo...")
        dataset = EvalDataset.create_sample()

    print(f"Items a evaluar: {len(dataset)}")
    print(f"Stats: {dataset.get_stats()}")

    # Ejecutar evaluación
    print("\n=== Ejecutando Evaluación ===")
    metrics_calc = RAGMetrics(pipeline)
    aggregated = metrics_calc.evaluate_dataset(dataset, top_k=args.top_k)

    # Mostrar resultados
    print("\n" + "=" * 50)
    print("RESULTADOS DE EVALUACIÓN")
    print("=" * 50)

    print("\n📊 Métricas Agregadas:")
    print(f"   Hit@{args.top_k} Rate:     {aggregated.hit_at_k_rate:.2%}")
    print(f"   Avg Precision:    {aggregated.avg_precision:.2%}")
    print(f"   Avg Recall:       {aggregated.avg_recall:.2%}")
    print(f"   Avg Faithfulness: {aggregated.avg_faithfulness:.2%}")
    print(f"   Avg Relevance:    {aggregated.avg_answer_relevance:.2%}")
    print(f"   Avg Latency:      {aggregated.avg_latency_ms:.0f}ms")
    print(f"   Refusal Rate:     {aggregated.refusal_rate:.2%}")

    # Verificar umbrales
    print("\n📋 Verificación de Umbrales:")
    hit_ok = aggregated.hit_at_k_rate >= 0.70
    faith_ok = aggregated.avg_faithfulness >= 0.70

    print(
        f"   Hit@K ≥ 70%:      {'✅ PASS' if hit_ok else '❌ FAIL'} ({aggregated.hit_at_k_rate:.2%})"
    )
    print(
        f"   Faithfulness ≥ 70%: {'✅ PASS' if faith_ok else '❌ FAIL'} ({aggregated.avg_faithfulness:.2%})"
    )

    overall = "✅ PASSED" if (hit_ok and faith_ok) else "❌ FAILED"
    print(f"\n🎯 Resultado General: {overall}")

    # Generar reporte si se solicita
    if args.report:
        print("\n=== Generando Reporte ===")
        reporter = EvalReporter(
            metrics=aggregated,
            metadata={
                "model": stats["llm_model"],
                "embedding_model": stats["embedding_model"],
                "chunk_size": stats["chunk_size"],
                "top_k": args.top_k,
                "total_chunks": stats["total_chunks"],
            },
        )
        reporter.save(args.output)

    print("\n=== Evaluación Completada ===")


if __name__ == "__main__":
    main()
