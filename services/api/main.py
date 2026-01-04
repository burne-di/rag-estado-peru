"""
FastAPI Application - RAG Estado Peru API
"""

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Agregar packages al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from packages.rag_core import RAGPipeline, __version__  # noqa: E402

from .schemas import (  # noqa: E402
    Citation,
    DebugSearchRequest,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    StatsResponse,
)

# Pipeline global
pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicialización y cleanup del pipeline"""
    global pipeline
    print("Inicializando RAG Pipeline...")
    pipeline = RAGPipeline()
    print(f"Pipeline listo. Chunks indexados: {pipeline.get_stats()['total_chunks']}")
    yield
    print("Cerrando aplicación...")


app = FastAPI(
    title="RAG Estado Peru API",
    description="Sistema de Preguntas y Respuestas sobre normativa pública peruana",
    version=__version__,
    lifespan=lifespan,
)

# CORS para desarrollo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Verifica el estado del servicio"""
    return HealthResponse(status="healthy", version=__version__)


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_stats():
    """Obtiene estadísticas del sistema RAG"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline no inicializado")
    return StatsResponse(**pipeline.get_stats())


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(request: QueryRequest):
    """
    Realiza una consulta RAG.

    Busca en los documentos indexados y genera una respuesta
    con citas verificables.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline no inicializado")

    if pipeline.get_stats()["total_chunks"] == 0:
        raise HTTPException(
            status_code=400, detail="No hay documentos indexados. Use /ingest primero."
        )

    try:
        result = pipeline.query(request.question, top_k=request.top_k)

        # Convertir citations al schema
        citations = [
            Citation(
                source=c.get("source", "Desconocido"),
                page=c.get("page"),
                quote=c.get("quote", c.get("excerpt", "")),
                relevance_score=c.get("relevance_score", 0),
            )
            for c in result.get("citations", [])
        ]

        return QueryResponse(
            answer=result["answer"],
            citations=citations,
            sources_used=result["sources_used"],
            model=result.get("model"),
            confidence=result.get("confidence"),
            latency_ms=result.get("latency_ms"),
            from_cache=result.get("from_cache", False),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug/chunks", tags=["Debug"])
async def debug_chunks(request: QueryRequest):
    """
    Devuelve los chunks recuperados para una query (debug).
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline no inicializado")

    normalized = request.question
    try:
        from packages.rag_core.pipeline import normalize_query

        normalized = normalize_query(request.question)
    except Exception:
        pass

    top_k = request.top_k or pipeline.settings.top_k_results
    chunks = pipeline.vector_store.search(normalized, top_k=top_k)
    return {
        "question": request.question,
        "normalized": normalized,
        "top_k": top_k,
        "chunks": [
            {
                "score": c.get("score", 0),
                "score_vector": c.get("score_vector"),
                "score_keyword": c.get("score_keyword"),
                "exact_match": c.get("exact_match"),
                "token_hits": c.get("token_hits"),
                "phrase_matches": c.get("phrase_matches"),
                "content": c.get("content", ""),
                "source": c.get("metadata", {}).get("source"),
                "page": c.get("metadata", {}).get("page"),
                "source_path": c.get("metadata", {}).get("source_path"),
            }
            for c in chunks
        ],
    }


@app.post("/debug/pdf-search", tags=["Debug"])
async def debug_pdf_search(request: DebugSearchRequest):
    """
    Busca un termino literal dentro de un PDF y devuelve paginas con match.
    """
    from packages.rag_core.loaders import PDFLoader

    try:
        loader = PDFLoader(request.file_path)
        documents = loader.load()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    term = request.term.strip()
    if not term:
        raise HTTPException(status_code=400, detail="term vacio")

    term_lower = term.lower()
    results = []
    for doc in documents:
        content = doc.content or ""
        content_lower = content.lower()
        idx = content_lower.find(term_lower)
        if idx == -1:
            continue

        start = max(0, idx - 120)
        end = min(len(content), idx + 200)
        snippet = content[start:end]
        results.append(
            {
                "page": doc.metadata.get("page"),
                "source": doc.metadata.get("source"),
                "snippet": snippet,
            }
        )
        if request.max_results and len(results) >= request.max_results:
            break

    return {
        "file_path": request.file_path,
        "term": request.term,
        "matches": results,
        "total_matches": len(results),
    }


@app.post("/query/stream", tags=["RAG"])
async def query_stream(request: QueryRequest):
    """
    Realiza una consulta RAG con streaming.

    Retorna la respuesta en chunks mientras se genera,
    mejorando la experiencia de usuario.

    El stream envía eventos SSE (Server-Sent Events):
    - data: {"type": "chunk", "content": "..."} - Chunks de texto
    - data: {"type": "done", "result": {...}} - Resultado final con metadata
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline no inicializado")

    if pipeline.get_stats()["total_chunks"] == 0:
        raise HTTPException(
            status_code=400, detail="No hay documentos indexados. Use /ingest primero."
        )

    async def generate():
        try:
            # Primero verificar caché
            from packages.rag_core.pipeline import normalize_query

            normalized = normalize_query(request.question)

            if pipeline.enable_cache:
                cached = pipeline.cache.get(normalized)
                if cached:
                    # Enviar respuesta cacheada de inmediato
                    cached["from_cache"] = True
                    yield f"data: {json.dumps({'type': 'cached', 'result': cached})}\n\n"
                    return

            # Obtener chunks relevantes
            relevant_chunks = pipeline.vector_store.search(
                normalized, top_k=request.top_k or pipeline.settings.top_k_results
            )

            # Hacer routing si está habilitado
            model_override = None
            if pipeline.enable_routing:
                routing_decision = pipeline.router.route(
                    request.question, relevant_chunks
                )
                model_override = routing_decision.model
                yield f"data: {json.dumps({'type': 'routing', 'model': model_override})}\n\n"

            # Generar con streaming
            stream = pipeline.generator.generate_stream(
                request.question, relevant_chunks, model_override=model_override
            )

            full_response = ""
            for chunk in stream:
                full_response += chunk
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                await asyncio.sleep(0)  # Permitir que el event loop procese

            # Parsear resultado final
            result = pipeline.generator._parse_json_response(full_response)

            # Guardar en caché
            if pipeline.enable_cache and not result.get("refusal"):
                pipeline.cache.set(normalized, result)

            # Enviar resultado final
            yield f"data: {json.dumps({'type': 'done', 'result': result})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post("/ingest", response_model=IngestResponse, tags=["RAG"])
async def ingest(request: IngestRequest):
    """
    Ingesta documentos PDF al vector store.

    Puede ingestar un directorio completo o un archivo específico.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline no inicializado")

    # Validar input antes del try-except
    if not request.directory and not request.file_path:
        raise HTTPException(
            status_code=400, detail="Debe especificar 'directory' o 'file_path'"
        )

    try:
        if request.directory:
            result = pipeline.ingest_directory(request.directory)
        else:
            result = pipeline.ingest_file(request.file_path)

        return IngestResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise  # Re-raise HTTPExceptions sin convertir a 500
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/clear", tags=["RAG"])
async def clear_index():
    """Elimina todos los documentos del vector store"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline no inicializado")

    pipeline.clear()
    return {"status": "success", "message": "Vector store limpiado"}


@app.delete("/cache/clear", tags=["System"])
async def clear_cache():
    """Limpia el cache de respuestas"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline no inicializado")
    if not pipeline.enable_cache or pipeline.cache is None:
        return {"status": "success", "message": "Cache no habilitado"}

    pipeline.cache.clear()
    return {"status": "success", "message": "Cache limpiado"}


@app.get("/debug/settings", tags=["Debug"])
async def debug_settings():
    """
    Debug: Muestra la configuración actual del sistema.
    Útil para verificar que las variables de entorno se cargaron correctamente.
    """
    from packages.rag_core.config import get_settings

    settings = get_settings()
    return {
        "hybrid_search": settings.hybrid_search,
        "vector_weight": settings.vector_weight,
        "keyword_weight": settings.keyword_weight,
        "llm_provider": settings.llm_provider,
        "groq_model": settings.groq_model,
        "gemini_model": settings.gemini_model,
        "top_k_results": settings.top_k_results,
        "chunk_size": settings.chunk_size,
        "embedding_model": settings.embedding_model,
        "chroma_persist_dir": settings.chroma_persist_dir,
    }


@app.post("/debug/llm", tags=["Debug"])
async def debug_llm(request: QueryRequest):
    """
    Debug: Prueba el LLM directamente sin guardrails.
    Muestra la respuesta cruda del modelo.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline no inicializado")

    from packages.rag_core.config import get_settings
    from packages.rag_core.pipeline import normalize_query

    settings = get_settings()

    try:
        # Normalizar y buscar chunks
        normalized = normalize_query(request.question)
        chunks = pipeline.vector_store.search(normalized, top_k=request.top_k or 5)

        # Hacer routing
        routing_decision = None
        if pipeline.enable_routing:
            routing_decision = pipeline.router.route(request.question, chunks)

        # Generar respuesta SIN guardrails
        response = pipeline.generator.generate(
            request.question,
            chunks,
            model_override=routing_decision.model if routing_decision else None,
            provider_override=routing_decision.provider if routing_decision else None,
        )

        return {
            "question": request.question,
            "normalized": normalized,
            "settings": {
                "hybrid_search": settings.hybrid_search,
                "vector_weight": settings.vector_weight,
                "keyword_weight": settings.keyword_weight,
            },
            "routing": {
                "provider": routing_decision.provider if routing_decision else None,
                "model": routing_decision.model if routing_decision else None,
                "complexity_score": routing_decision.complexity_score
                if routing_decision
                else None,
            },
            "chunks_count": len(chunks),
            "chunks_scores": [c.get("score", 0) for c in chunks[:5]],
            "chunks_details": [
                {
                    "score": c.get("score", 0),
                    "score_vector": c.get("score_vector"),
                    "score_keyword": c.get("score_keyword"),
                    "exact_match": c.get("exact_match"),
                    "page": c.get("metadata", {}).get("page"),
                }
                for c in chunks[:5]
            ],
            "llm_response": response,
            "raw_answer": response.get("answer", ""),
            "raw_llm_response": response.get("raw_llm_response"),
        }
    except Exception as e:
        import traceback

        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# Servir interfaz web
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.get("/", include_in_schema=False)
async def root():
    """Sirve la interfaz web"""
    index_file = static_path / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "RAG Estado Peru API", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn

    from packages.rag_core.config import get_settings

    settings = get_settings()
    uvicorn.run("main:app", host=settings.api_host, port=settings.api_port, reload=True)
