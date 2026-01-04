"""
Provider Factory - Gestiona la creación y selección de providers
"""

from typing import Optional

from .base import LLMProvider
from .gemini import GeminiProvider
from .groq import GroqProvider

# Registro de providers disponibles
PROVIDERS = {
    "gemini": GeminiProvider,
    "groq": GroqProvider,
}

# Instancias singleton
_provider_instances: dict[str, LLMProvider] = {}


def get_provider(name: Optional[str] = None) -> LLMProvider:
    """
    Obtiene una instancia del provider especificado.

    Args:
        name: Nombre del provider ("gemini", "groq").
              Si es None, retorna el primer provider disponible.

    Returns:
        Instancia del LLMProvider

    Raises:
        ValueError: Si el provider no existe o no está disponible
    """
    # Si no se especifica, buscar el primer disponible
    if name is None:
        for provider_name in ["groq", "gemini"]:  # Groq primero (más rápido)
            try:
                provider = get_provider(provider_name)
                if provider.is_available():
                    return provider
            except ValueError:
                continue
        raise ValueError(
            "No hay providers disponibles. Configura GROQ_API_KEY o GOOGLE_API_KEY"
        )

    # Normalizar nombre
    name = name.lower()

    # Verificar que existe
    if name not in PROVIDERS:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(f"Provider '{name}' no existe. Disponibles: {available}")

    # Crear instancia si no existe
    if name not in _provider_instances:
        _provider_instances[name] = PROVIDERS[name]()

    provider = _provider_instances[name]

    # Verificar disponibilidad
    if not provider.is_available():
        raise ValueError(f"Provider '{name}' no está configurado. Falta API key.")

    return provider


def get_available_providers() -> list[str]:
    """
    Retorna la lista de providers disponibles y configurados.

    Returns:
        Lista de nombres de providers disponibles
    """
    available = []
    for name, provider_class in PROVIDERS.items():
        try:
            provider = provider_class()
            if provider.is_available():
                available.append(name)
        except Exception:
            continue
    return available


def get_provider_with_fallback(
    primary: str, fallback: Optional[str] = None
) -> LLMProvider:
    """
    Obtiene un provider con fallback automático.

    Args:
        primary: Provider primario a intentar
        fallback: Provider de fallback (opcional)

    Returns:
        El provider disponible
    """
    try:
        provider = get_provider(primary)
        if provider.is_available():
            return provider
    except ValueError:
        pass

    if fallback:
        try:
            return get_provider(fallback)
        except ValueError:
            pass

    # Último recurso: cualquier provider disponible
    return get_provider(None)
