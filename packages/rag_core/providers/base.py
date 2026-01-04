"""
Base LLM Provider - Interfaz abstracta para proveedores de LLM
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator, Optional


@dataclass
class LLMResponse:
    """Respuesta estandarizada de cualquier LLM"""

    text: str
    model: str
    provider: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class LLMProvider(ABC):
    """Interfaz abstracta para proveedores de LLM"""

    provider_name: str = "base"

    @abstractmethod
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """
        Genera una respuesta del LLM.

        Args:
            prompt: El prompt completo a enviar
            model: Modelo específico a usar (opcional)
            max_tokens: Máximo de tokens en la respuesta
            temperature: Temperatura para la generación

        Returns:
            LLMResponse con el texto y metadata
        """
        pass

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> Generator[str, None, None]:
        """
        Genera una respuesta en modo streaming.

        Yields:
            Chunks de texto mientras se generan
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Verifica si el provider está configurado y disponible"""
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Retorna el modelo por defecto del provider"""
        pass

    @property
    @abstractmethod
    def available_models(self) -> list[str]:
        """Retorna la lista de modelos disponibles"""
        pass
