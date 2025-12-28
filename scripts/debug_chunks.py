"""Debug script para ver contenido de chunks"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from packages.rag_core import VectorStore

vs = VectorStore()
query = "¿Cuándo nace la obligación tributaria?"
results = vs.search(query, top_k=3)

print(f"Query: {query}\n")
for i, r in enumerate(results, 1):
    print(f"=== CHUNK {i} (score: {r['score']:.3f}) ===")
    print(f"Página: {r['metadata'].get('page')}")
    print(r["content"][:800])
    print()
