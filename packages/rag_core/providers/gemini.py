"""
Gemini Provider - Google Gemini API
"""

from typing import Generator, Optional

from .base import LLMProvider, LLMResponse


class GeminiProvider(LLMProvider):
    """Provider para Google Gemini"""

    provider_name = "gemini"

    # Modelos disponibles
    MODELS = {
        "gemini-2.5-flash": "gemini-2.5-flash",
        "gemini-2.0-flash-lite": "gemini-2.0-flash-lite",
        "gemini-1.5-flash": "gemini-1.5-flash",
        "gemini-1.5-pro": "gemini-1.5-pro",
    }

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key
        self._genai = None
        self._models: dict = {}

    def _ensure_initialized(self):
        """Inicializa el cliente de Gemini si no está inicializado"""
        if self._genai is None:
            import google.generativeai as genai

            from ..config import get_settings

            settings = get_settings()
            api_key = self._api_key or settings.google_api_key

            if not api_key:
                raise ValueError("GOOGLE_API_KEY no configurada")

            genai.configure(api_key=api_key)
            self._genai = genai

    def _get_model(self, model_name: str):
        """Obtiene o crea una instancia del modelo"""
        self._ensure_initialized()

        if model_name not in self._models:
            self._models[model_name] = self._genai.GenerativeModel(model_name)

        return self._models[model_name]

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """Genera respuesta con Gemini"""
        model_name = model or self.default_model
        gemini_model = self._get_model(model_name)

        try:
            response = gemini_model.generate_content(
                prompt,
                generation_config=self._genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            )

            return LLMResponse(
                text=response.text,
                model=model_name,
                provider=self.provider_name,
                prompt_tokens=getattr(
                    response.usage_metadata, "prompt_token_count", None
                ),
                completion_tokens=getattr(
                    response.usage_metadata, "candidates_token_count", None
                ),
                total_tokens=getattr(
                    response.usage_metadata, "total_token_count", None
                ),
            )
        except Exception as e:
            raise RuntimeError(f"Error en Gemini: {str(e)}") from e

    def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> Generator[str, None, None]:
        """Genera respuesta en streaming con Gemini"""
        model_name = model or self.default_model
        gemini_model = self._get_model(model_name)

        try:
            response = gemini_model.generate_content(
                prompt,
                generation_config=self._genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
                stream=True,
            )

            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            yield f"Error: {str(e)}"

    def is_available(self) -> bool:
        """Verifica si Gemini está disponible"""
        try:
            from ..config import get_settings

            settings = get_settings()
            return bool(self._api_key or settings.google_api_key)
        except Exception:
            return False

    @property
    def default_model(self) -> str:
        return "gemini-2.5-flash"

    @property
    def available_models(self) -> list[str]:
        return list(self.MODELS.keys())
