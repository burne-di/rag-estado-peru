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

    try:
        if request.directory:
            result = pipeline.ingest_directory(request.directory)
        elif request.file_path:
            result = pipeline.ingest_file(request.file_path)
        else:
            raise HTTPException(
                status_code=400, detail="Debe especificar 'directory' o 'file_path'"
            )

        return IngestResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/clear", tags=["RAG"])
async def clear_index():
    """Elimina todos los documentos del vector store"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline no inicializado")

    pipeline.clear()
    return {"status": "success", "message": "Vector store limpiado"}


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
