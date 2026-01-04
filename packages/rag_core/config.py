"""
Configuración central del proyecto
"""

import os
from functools import lru_cache

from pydantic_settings import BaseSettings


def _parse_bool_env(key: str, default: bool = True) -> bool:
    """Parse boolean from environment variable correctly."""
    val = os.environ.get(key, "").lower()
    if val in ("true", "1", "yes", "on"):
        return True
    if val in ("false", "0", "no", "off"):
        return False
    return default


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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Force re-read HYBRID_SEARCH from env to ensure correct boolean parsing
        self.hybrid_search = _parse_bool_env("HYBRID_SEARCH", default=True)
        print(
            f"[Config] hybrid_search={self.hybrid_search} (from env: {os.environ.get('HYBRID_SEARCH', 'not set')})"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
