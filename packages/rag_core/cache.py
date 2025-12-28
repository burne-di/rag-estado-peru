"""
Sistema de caché para respuestas del LLM.
Evita llamadas repetidas a la API para preguntas similares.
"""
import hashlib
import json
import re
import threading
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CacheEntry:
    """Entrada en el caché"""
    question_hash: str
    question: str
    answer: dict
    timestamp: float
    hits: int = 0


class ResponseCache:
    """
    Caché de respuestas LLM con persistencia en disco.

    Features:
    - Hash de preguntas normalizadas para matching exacto
    - TTL (Time To Live) configurable
    - Persistencia en JSON para sobrevivir reinicios
    - Thread-safe para uso concurrente
    - Estadísticas de uso
    """

    def __init__(
        self,
        cache_dir: str = "./data/cache",
        ttl_hours: int = 24,
        max_entries: int = 1000
    ):
        """
        Args:
            cache_dir: Directorio donde guardar el caché
            ttl_hours: Tiempo de vida de las entradas en horas
            max_entries: Número máximo de entradas en caché
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "response_cache.json"

        self.ttl_seconds = ttl_hours * 3600
        self.max_entries = max_entries

        self._cache: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._stats = {"hits": 0, "misses": 0, "saves": 0}

        # Cargar caché existente
        self._load_cache()

    def _normalize_question(self, question: str) -> str:
        """Normaliza la pregunta para mejor matching"""
        # Lowercase, remover espacios extra, remover puntuación
        text = question.lower().strip()
        # Remover acentos
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        # Remover puntuación
        text = re.sub(r'[^\w\s]', '', text)
        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text)

        return text

    def _hash_question(self, question: str) -> str:
        """Genera hash único para una pregunta normalizada"""
        normalized = self._normalize_question(question)
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def get(self, question: str) -> Optional[dict]:
        """
        Busca una respuesta en caché.

        Args:
            question: Pregunta del usuario

        Returns:
            Respuesta cacheada o None si no existe/expiró
        """
        question_hash = self._hash_question(question)

        with self._lock:
            if question_hash not in self._cache:
                self._stats["misses"] += 1
                return None

            entry = self._cache[question_hash]

            # Verificar TTL
            if time.time() - entry.timestamp > self.ttl_seconds:
                del self._cache[question_hash]
                self._stats["misses"] += 1
                return None

            # Cache hit!
            entry.hits += 1
            self._stats["hits"] += 1

            print(f"📦 Cache HIT para: '{question[:50]}...' (hits: {entry.hits})")

            return entry.answer

    def set(self, question: str, answer: dict) -> None:
        """
        Guarda una respuesta en caché.

        Args:
            question: Pregunta del usuario
            answer: Respuesta generada
        """
        question_hash = self._hash_question(question)

        with self._lock:
            # Limpiar entradas antiguas si excedemos el límite
            if len(self._cache) >= self.max_entries:
                self._evict_oldest()

            self._cache[question_hash] = CacheEntry(
                question_hash=question_hash,
                question=question,
                answer=answer,
                timestamp=time.time(),
                hits=0
            )

            self._stats["saves"] += 1
            self._save_cache()

            print(f"💾 Cache SAVE para: '{question[:50]}...'")

    def _evict_oldest(self) -> None:
        """Elimina las entradas más antiguas (LRU simple)"""
        if not self._cache:
            return

        # Ordenar por timestamp y eliminar el 20% más antiguo
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].timestamp
        )

        entries_to_remove = max(1, len(sorted_entries) // 5)

        for key, _ in sorted_entries[:entries_to_remove]:
            del self._cache[key]

        print(f"🗑️ Cache eviction: eliminadas {entries_to_remove} entradas antiguas")

    def _load_cache(self) -> None:
        """Carga el caché desde disco"""
        if not self.cache_file.exists():
            return

        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            current_time = time.time()
            loaded = 0

            for entry_data in data.get("entries", []):
                # Verificar TTL al cargar
                if current_time - entry_data["timestamp"] <= self.ttl_seconds:
                    entry = CacheEntry(**entry_data)
                    self._cache[entry.question_hash] = entry
                    loaded += 1

            print(f"📂 Cache cargado: {loaded} entradas válidas")

        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ Error cargando caché: {e}")
            self._cache = {}

    def _save_cache(self) -> None:
        """Guarda el caché a disco"""
        try:
            data = {
                "version": 1,
                "saved_at": time.time(),
                "entries": [asdict(entry) for entry in self._cache.values()]
            }

            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except IOError as e:
            print(f"⚠️ Error guardando caché: {e}")

    def get_stats(self) -> dict:
        """Retorna estadísticas del caché"""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = (
                self._stats["hits"] / total_requests * 100
                if total_requests > 0 else 0
            )

            return {
                "total_entries": len(self._cache),
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "saves": self._stats["saves"],
                "hit_rate_percent": round(hit_rate, 2),
                "estimated_savings": f"{self._stats['hits']} llamadas a API evitadas"
            }

    def clear(self) -> None:
        """Limpia todo el caché"""
        with self._lock:
            self._cache = {}
            self._stats = {"hits": 0, "misses": 0, "saves": 0}

            if self.cache_file.exists():
                self.cache_file.unlink()

            print("🧹 Cache limpiado completamente")


# Singleton global
_cache_instance: Optional[ResponseCache] = None


def get_cache() -> ResponseCache:
    """Obtiene la instancia singleton del caché"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = ResponseCache()
    return _cache_instance
