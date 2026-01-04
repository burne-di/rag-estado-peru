"""
Pydantic schemas para la API
"""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request para consulta RAG"""

    question: str = Field(..., min_length=3, description="Pregunta a responder")
    top_k: int | None = Field(
        None, ge=1, le=20, description="Número de chunks a recuperar"
    )


class Citation(BaseModel):
    """Una cita de un documento"""

    source: str
    page: int | None = None
    quote: str | None = None
    relevance_score: float = 0.0


class QueryResponse(BaseModel):
    """Response de consulta RAG"""

    answer: str
    citations: list[Citation]
    sources_used: int
    model: str | None = None
    confidence: float | None = None
    latency_ms: int | None = None
    from_cache: bool = False


class IngestRequest(BaseModel):
    """Request para ingesta de documentos"""

    directory: str | None = Field(None, description="Directorio con PDFs a ingestar")
    file_path: str | None = Field(None, description="Ruta a un PDF específico")


class IngestResponse(BaseModel):
    """Response de ingesta"""

    status: str
    documents: int | None = None
    pages: int | None = None
    chunks: int | None = None
    total_indexed: int | None = None
    message: str | None = None


class CacheStats(BaseModel):
    """Estadísticas del caché"""

    total_entries: int = 0
    hits: int = 0
    misses: int = 0
    hit_rate_percent: float = 0.0
    estimated_savings: str = ""


class StatsResponse(BaseModel):
    """Estadísticas del sistema"""

    total_chunks: int
    embedding_model: str
    llm_model: str
    llm_provider: str | None = None
    chunk_size: int
    top_k: int
    cache_enabled: bool = False
    cache_stats: CacheStats | None = None
    routing_enabled: bool = False
    available_models: dict[str, list[str]] | None = None


class DebugSearchRequest(BaseModel):
    """Request para buscar texto literal en un PDF"""

    file_path: str
    term: str
    max_results: int | None = Field(10, ge=1, le=100)


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    version: str
