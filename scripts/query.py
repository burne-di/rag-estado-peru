"""
Script CLI para hacer consultas RAG
"""

import sys
from pathlib import Path

# Agregar root al path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from packages.rag_core import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Realiza consultas al sistema RAG")
    parser.add_argument("question", type=str, nargs="?", help="Pregunta a realizar")
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Modo interactivo"
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Número de chunks a recuperar (default: 5)",
    )

    args = parser.parse_args()

    pipeline = RAGPipeline()
    stats = pipeline.get_stats()

    print(f"RAG Estado Peru - {stats['total_chunks']} chunks indexados")
    print("-" * 50)

    if stats["total_chunks"] == 0:
        print("⚠ No hay documentos indexados. Ejecuta primero:")
        print("  python scripts/ingest.py")
        return

    if args.interactive:
        print("Modo interactivo. Escribe 'salir' para terminar.\n")
        while True:
            question = input("\n📝 Tu pregunta: ").strip()
            if question.lower() in ["salir", "exit", "quit"]:
                print("¡Hasta luego!")
                break
            if not question:
                continue
            process_question(pipeline, question, args.top_k)
    elif args.question:
        process_question(pipeline, args.question, args.top_k)
    else:
        parser.print_help()


def process_question(pipeline: RAGPipeline, question: str, top_k: int):
    """Procesa una pregunta y muestra la respuesta"""
    print(f"\n🔍 Buscando en {top_k} chunks...")

    result = pipeline.query(question, top_k=top_k)

    print("\n" + "=" * 50)
    print("📌 RESPUESTA:")
    print("=" * 50)
    print(result["answer"])

    print("\n" + "-" * 50)
    print(f"📚 FUENTES ({result['sources_used']} documentos):")
    print("-" * 50)

    for i, citation in enumerate(result.get("citations", []), 1):
        print(
            f"\n[{i}] {citation.get('source', 'Desconocido')} - Página {citation.get('page', '?')}"
        )
        print(f"    Relevancia: {citation.get('relevance_score', 0):.2%}")
        excerpt = citation.get("excerpt", citation.get("text", ""))
        if excerpt:
            print(f"    Extracto: {excerpt[:150]}...")


if __name__ == "__main__":
    main()
