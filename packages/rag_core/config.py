"""
Configuración central del proyecto
"""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM Providers (al menos uno debe estar configurado)
    google_api_key: str = ""  # Gemini
    groq_api_key: str = ""  # Groq

    # Modelos por defecto
    gemini_model: str = "gemini-2.5-flash"
    groq_model: str = "llama-3.3-70b-versatile"

    # Provider preferido ("groq", "gemini", o "auto" para fallback automático)
    llm_provider: str = "groq"

    # Embeddings
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma"

    # RAG Parameters
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k_results: int = 5

    # Hybrid Search (vector + keyword)
    hybrid_search: bool = True
    vector_weight: float = 0.7
    keyword_weight: float = 0.3

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
