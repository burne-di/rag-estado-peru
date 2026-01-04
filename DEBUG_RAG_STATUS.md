# Estado actual del troubleshooting RAG

Fecha: 2026-01-03

## Problema principal
- Las respuestas RAG no describen el contenido correcto (ej. NORMA XV), aunque el PDF sí contiene el texto.
- El retrieval devolvía chunks irrelevantes y el LLM respondía “no hay información”.

## Hallazgos confirmados
- El PDF contiene “NORMA XV: UNIDAD IMPOSITIVA TRIBUTARIA” (página 7) según `/debug/pdf-search`.
- El vector store inicialmente NO recuperaba ese chunk con consultas tipo “NORMA XV…”.
- La query de debug mostró que el retrieval estaba trayendo secciones no relacionadas.
- En `/debug/chunks`, las métricas `score_keyword`, `token_hits`, `phrase_matches` salen como null, lo que indica que el código nuevo de keyword/híbrido no se está aplicando en runtime o que `HYBRID_SEARCH` no está activo.

## Cambios realizados hasta ahora
### Infra/configuración
- `rag-estado-peru/docker-compose.yml`
  - Se agregó `env_file: .env`.
  - Se expusieron variables: `LLM_PROVIDER`, `GROQ_API_KEY`, `GROQ_MODEL`, `HYBRID_SEARCH`, `VECTOR_WEIGHT`, `KEYWORD_WEIGHT`.
  - Se cambió el default de `GROQ_MODEL` a `openai/gpt-oss-120b`.
- `rag-estado-peru/.env.example`
  - Default de `GROQ_MODEL` a `openai/gpt-oss-120b`.
  - Default de `CHUNK_SIZE=1000`, `CHUNK_OVERLAP=200`.
  - Nuevas variables de búsqueda híbrida.

### Pipeline / API
- `rag-estado-peru/services/api/main.py`
  - Nuevo endpoint: `DELETE /cache/clear`.
  - Nuevo endpoint: `POST /debug/chunks` (devuelve chunks y scores).
  - Nuevo endpoint: `POST /debug/pdf-search` (busca texto literal dentro del PDF).
  - `POST /debug/chunks` ahora devuelve métricas de keyword (`score_keyword`, `token_hits`, `phrase_matches`, `exact_match`).
- `rag-estado-peru/services/api/schemas.py`
  - `StatsResponse` ajustado para nuevos campos.
  - Nuevo schema `DebugSearchRequest` para `/debug/pdf-search`.

### RAG core
- `rag-estado-peru/packages/rag_core/providers/groq.py`
  - Se agregó el modelo `openai/gpt-oss-120b`.
  - Default complejo/simple apuntando a `openai/gpt-oss-120b`.
- `rag-estado-peru/packages/rag_core/router.py`
  - Modelo `STANDARD` de Groq apuntando a `openai/gpt-oss-120b`.
- `rag-estado-peru/packages/rag_core/config.py`
  - Nuevas variables: `hybrid_search`, `vector_weight`, `keyword_weight`.
- `rag-estado-peru/packages/rag_core/generator.py`
  - Prompt reforzado para exigir `answer` no vacío.
  - Fallback de respuesta desde chunks si el LLM devuelve vacío.
- `rag-estado-peru/packages/rag_core/guardrails/refusal_policy.py`
  - Normalización de texto para comparar resúmenes.
  - Relajado el rechazo en consultas de resumen.
- `rag-estado-peru/packages/rag_core/vectorstore.py`
  - Búsqueda híbrida (vector + keyword).
  - Normalización mejorada (quita acentos y puntuación, normaliza espacios).
  - Keyword search paginado para leer toda la colección.
  - Boost por frases y tokens; soporte para números romanos.

## Resultado actual
- `/debug/pdf-search` encuentra “NORMA XV” en el PDF (página 7).
- `/debug/chunks` aún retorna resultados vectoriales sin señal keyword (score_keyword=0.0 y campos null), por lo que el primer chunk relevante no llega a `/query`.
- El `/query` sigue respondiendo que no hay información, aun con el texto en el PDF.

## Hipótesis pendiente
- La app en runtime no está usando el código actualizado del vectorstore (posible cache de imagen o volumen que no refresca).
- `HYBRID_SEARCH` podría estar desactivado o no estar leyendo el `.env` correcto dentro del contenedor.

## Próximos pasos sugeridos
1. Verificar que `.env` tenga `HYBRID_SEARCH=true` y pesos definidos.
2. Hacer rebuild limpio:
   - `docker compose down`
   - `docker compose build --no-cache`
   - `docker compose up`
3. Reingestar:
   - `DELETE /clear`
   - `POST /ingest` con el PDF
4. Repetir `POST /debug/chunks` y validar que aparezcan:
   - `score_keyword` > 0
   - `token_hits` y `phrase_matches` con valores
   - `exact_match: true` para el chunk de NORMA XV.

## Comandos/Endpoints útiles
- `POST /debug/pdf-search`:
  - Body: `{ "file_path": "/app/data/raw/Codigo-Tributario-Sunat.pdf", "term": "NORMA XV" }`
- `POST /debug/chunks`:
  - Body: `{ "question": "NORMA XV: UNIDAD IMPOSITIVA TRIBUTARIA", "top_k": 15 }`
- `DELETE /cache/clear`
- `DELETE /clear`
- `POST /ingest`

