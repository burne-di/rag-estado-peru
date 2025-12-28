"""
Script para construir dataset de evaluación de forma interactiva
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from packages.rag_core.eval import EvalDataset, EvalItem


def main():
    print("=" * 60)
    print("CONSTRUCTOR DE DATASET DE EVALUACIÓN")
    print("=" * 60)
    print("\nEste script te ayuda a crear un dataset de evaluación")
    print("para medir la calidad del sistema RAG.\n")

    output_path = Path("./data/eval_dataset.jsonl")

    # Cargar dataset existente o crear nuevo
    if output_path.exists():
        print(f"Dataset existente encontrado: {output_path}")
        choice = input("¿Agregar items (a) o crear nuevo (n)? [a/n]: ").strip().lower()
        if choice == "a":
            dataset = EvalDataset.load(output_path)
            print(f"Cargado dataset con {len(dataset)} items existentes")
        else:
            dataset = EvalDataset()
            print("Creando nuevo dataset...")
    else:
        dataset = EvalDataset()
        print("Creando nuevo dataset...")

    print("\nIngresa los items de evaluación.")
    print("Escribe 'done' cuando termines.\n")

    categories = [
        "definicion",
        "plazos",
        "obligaciones",
        "procedimientos",
        "instituciones",
        "otro",
    ]
    difficulties = ["easy", "medium", "hard"]

    while True:
        print("-" * 40)
        question = input("\nPregunta (o 'done' para terminar): ").strip()

        if question.lower() == "done":
            break

        if len(question) < 10:
            print("Pregunta muy corta, intenta de nuevo.")
            continue

        # Fuentes esperadas
        sources = input("Fuentes esperadas (separadas por coma): ").strip()
        expected_sources = [s.strip() for s in sources.split(",") if s.strip()]

        if not expected_sources:
            print("Debes indicar al menos una fuente esperada.")
            continue

        # Respuesta esperada (opcional)
        gold_answer = input(
            "Respuesta esperada (opcional, Enter para omitir): "
        ).strip()
        gold_answer = gold_answer if gold_answer else None

        # Categoría
        print(f"Categorías: {', '.join(categories)}")
        category = input("Categoría: ").strip().lower()
        category = category if category in categories else "otro"

        # Dificultad
        print(f"Dificultades: {', '.join(difficulties)}")
        difficulty = input("Dificultad: ").strip().lower()
        difficulty = difficulty if difficulty in difficulties else "medium"

        # Crear item
        item = EvalItem(
            question=question,
            expected_sources=expected_sources,
            gold_answer=gold_answer,
            category=category,
            difficulty=difficulty,
        )

        dataset.add(item)
        print(f"✓ Item agregado. Total: {len(dataset)}")

    # Guardar
    if len(dataset) > 0:
        dataset.save(output_path)
        print(f"\n✓ Dataset guardado: {output_path}")
        print(f"  Total items: {len(dataset)}")
        print(f"  Stats: {dataset.get_stats()}")
    else:
        print("\nNo se agregaron items.")


if __name__ == "__main__":
    main()
