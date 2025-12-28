"""
Configuración central del proyecto
"""
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Gemini
    google_api_key: str
    gemini_model: str = "gemini-2.5-flash"

    # Embeddings
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma"

    # RAG Parameters
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k_results: int = 5

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
