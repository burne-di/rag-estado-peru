"""
Groq Provider - Groq API (OpenAI-compatible, ultra-fast inference)
"""

from typing import Generator, Optional

from .base import LLMProvider, LLMResponse


class GroqProvider(LLMProvider):
    """Provider para Groq (LPU inference)"""

    provider_name = "groq"

    # Modelos disponibles en Groq
    MODELS = {
        # Llama 3.3 (más reciente)
        "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
        # OpenAI OSS en Groq
        "openai/gpt-oss-120b": "openai/gpt-oss-120b",
        # Llama 3.1
        "llama-3.1-70b-versatile": "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant": "llama-3.1-8b-instant",
        # Mixtral
        "mixtral-8x7b-32768": "mixtral-8x7b-32768",
        # Gemma
        "gemma2-9b-it": "gemma2-9b-it",
    }

    # Mapeo de modelos simples/complejos para routing
    SIMPLE_MODEL = "openai/gpt-oss-120b"
    COMPLEX_MODEL = "openai/gpt-oss-120b"

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key
        self._client = None

    def _ensure_initialized(self):
        """Inicializa el cliente de Groq si no está inicializado"""
        if self._client is None:
            from openai import OpenAI

            from ..config import get_settings

            settings = get_settings()
            api_key = self._api_key or settings.groq_api_key

            if not api_key:
                raise ValueError("GROQ_API_KEY no configurada")

            self._client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
            )

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """Genera respuesta con Groq"""
        self._ensure_initialized()

        model_name = model or self.default_model

        try:
            response = self._client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            return LLMResponse(
                text=response.choices[0].message.content,
                model=model_name,
                provider=self.provider_name,
                prompt_tokens=response.usage.prompt_tokens if response.usage else None,
                completion_tokens=(
                    response.usage.completion_tokens if response.usage else None
                ),
                total_tokens=response.usage.total_tokens if response.usage else None,
            )
        except Exception as e:
            raise RuntimeError(f"Error en Groq: {str(e)}") from e

    def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> Generator[str, None, None]:
        """Genera respuesta en streaming con Groq"""
        self._ensure_initialized()

        model_name = model or self.default_model

        try:
            stream = self._client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            yield f"Error: {str(e)}"

    def is_available(self) -> bool:
        """Verifica si Groq está disponible"""
        try:
            from ..config import get_settings

            settings = get_settings()
            return bool(self._api_key or settings.groq_api_key)
        except Exception:
            return False

    @property
    def default_model(self) -> str:
        return self.COMPLEX_MODEL

    @property
    def available_models(self) -> list[str]:
        return list(self.MODELS.keys())
