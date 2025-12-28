"""
Script CLI para ingesta de documentos
"""
import sys
from pathlib import Path

# Agregar root al path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from packages.rag_core import RAGPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Ingesta documentos PDF al vector store"
    )
    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        default="./data/raw",
        help="Directorio con PDFs a ingestar (default: ./data/raw)"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Limpiar el vector store antes de ingestar"
    )

    args = parser.parse_args()

    pipeline = RAGPipeline()

    if args.clear:
        print("Limpiando vector store...")
        pipeline.clear()

    result = pipeline.ingest_directory(args.directory)

    if result["status"] == "success":
        print("\n✓ Ingesta exitosa!")
    else:
        print(f"\n✗ Error: {result.get('message', 'Unknown error')}")


if __name__ == "__main__":
    main()
