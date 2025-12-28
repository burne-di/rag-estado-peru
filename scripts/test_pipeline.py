"""
Script para probar el pipeline completo
"""
import sys
from pathlib import Path

# Agregar root al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from packages.rag_core import RAGPipeline


def main():
    print("=" * 60)
    print("TEST DEL PIPELINE RAG - ESTADO PERU")
    print("=" * 60)

    # 1. Inicializar pipeline
    print("\n1. Inicializando pipeline...")
    pipeline = RAGPipeline()
    print("   ✓ Pipeline inicializado")

    # 2. Verificar stats
    stats = pipeline.get_stats()
    print("\n2. Estado actual:")
    print(f"   - Chunks indexados: {stats['total_chunks']}")
    print(f"   - Modelo embeddings: {stats['embedding_model']}")
    print(f"   - Modelo LLM: {stats['llm_model']}")

    # 3. Ingestar documentos si está vacío
    if stats['total_chunks'] == 0:
        print("\n3. Ingesta de documentos...")
        result = pipeline.ingest_directory("./data/raw")
        if result["status"] == "success":
            print(f"   ✓ Ingesta exitosa: {result['chunks']} chunks")
        else:
            print(f"   ✗ Error: {result.get('message')}")
            return

    # 4. Hacer una consulta de prueba
    print("\n4. Consulta de prueba...")
    test_questions = [
        "¿Qué es el Código Tributario?",
        "¿Cuáles son las obligaciones tributarias?",
    ]

    for question in test_questions:
        print(f"\n   Pregunta: {question}")
        print("   " + "-" * 50)

        result = pipeline.query(question, top_k=3)

        # Mostrar respuesta truncada
        answer = result["answer"]
        if len(answer) > 300:
            answer = answer[:300] + "..."
        print(f"   Respuesta: {answer}")
        print(f"   Fuentes: {result['sources_used']}")

    print("\n" + "=" * 60)
    print("TEST COMPLETADO")
    print("=" * 60)


if __name__ == "__main__":
    main()
