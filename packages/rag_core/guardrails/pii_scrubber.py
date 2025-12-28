"""
PII Scrubber - Detecta y remueve información personal identificable
"""

import re
from dataclasses import dataclass


@dataclass
class PIIMatch:
    """Representa una coincidencia de PII"""

    type: str
    value: str
    start: int
    end: int
    replacement: str


class PIIScrubber:
    """
    Detecta y remueve PII (Información Personal Identificable)
    de textos para logs y contextos.

    Patrones soportados para Perú:
    - DNI (8 dígitos)
    - RUC (11 dígitos)
    - Teléfonos
    - Emails
    - Nombres (heurístico)
    """

    # Patrones de PII comunes en Perú
    PATTERNS = {
        "dni": {
            "pattern": r"\b\d{8}\b",
            "replacement": "[DNI_REDACTED]",
            "description": "DNI peruano (8 dígitos)",
        },
        "ruc": {
            "pattern": r"\b(10|15|17|20)\d{9}\b",
            "replacement": "[RUC_REDACTED]",
            "description": "RUC peruano (11 dígitos)",
        },
        "phone": {
            "pattern": r"\b(?:\+51\s?)?(?:9\d{8}|[1-9]\d{6})\b",
            "replacement": "[PHONE_REDACTED]",
            "description": "Teléfono peruano",
        },
        "email": {
            "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "replacement": "[EMAIL_REDACTED]",
            "description": "Correo electrónico",
        },
        "credit_card": {
            "pattern": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
            "replacement": "[CARD_REDACTED]",
            "description": "Número de tarjeta",
        },
        "ip_address": {
            "pattern": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            "replacement": "[IP_REDACTED]",
            "description": "Dirección IP",
        },
    }

    def __init__(self, patterns_to_use: list[str] | None = None):
        """
        Args:
            patterns_to_use: Lista de tipos de PII a detectar.
                            Si es None, usa todos los patrones.
        """
        if patterns_to_use:
            self.active_patterns = {
                k: v for k, v in self.PATTERNS.items() if k in patterns_to_use
            }
        else:
            self.active_patterns = self.PATTERNS.copy()

        # Compilar patrones
        self.compiled_patterns = {
            pii_type: re.compile(config["pattern"], re.IGNORECASE)
            for pii_type, config in self.active_patterns.items()
        }

    def detect(self, text: str) -> list[PIIMatch]:
        """
        Detecta PII en el texto sin modificarlo.

        Returns:
            Lista de PIIMatch con las coincidencias encontradas
        """
        matches = []

        for pii_type, pattern in self.compiled_patterns.items():
            for match in pattern.finditer(text):
                matches.append(
                    PIIMatch(
                        type=pii_type,
                        value=match.group(),
                        start=match.start(),
                        end=match.end(),
                        replacement=self.active_patterns[pii_type]["replacement"],
                    )
                )

        # Ordenar por posición
        matches.sort(key=lambda m: m.start)
        return matches

    def scrub(self, text: str) -> tuple[str, list[PIIMatch]]:
        """
        Detecta y reemplaza PII en el texto.

        Returns:
            (texto_limpio, lista_de_matches)
        """
        matches = self.detect(text)

        if not matches:
            return text, []

        # Reemplazar de atrás hacia adelante para no alterar índices
        scrubbed = text
        for match in reversed(matches):
            scrubbed = (
                scrubbed[: match.start] + match.replacement + scrubbed[match.end :]
            )

        return scrubbed, matches

    def scrub_for_logs(self, data: dict) -> dict:
        """
        Limpia PII de un diccionario para logging seguro.
        Procesa strings en cualquier nivel del dict.
        """
        return self._scrub_dict(data)

    def _scrub_dict(self, obj):
        """Recursivamente limpia PII de estructuras de datos"""
        if isinstance(obj, str):
            scrubbed, _ = self.scrub(obj)
            return scrubbed
        elif isinstance(obj, dict):
            return {k: self._scrub_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._scrub_dict(item) for item in obj]
        else:
            return obj

    def get_stats(self, text: str) -> dict:
        """
        Retorna estadísticas de PII encontrado.
        """
        matches = self.detect(text)

        stats = {"total": len(matches), "by_type": {}}
        for match in matches:
            if match.type not in stats["by_type"]:
                stats["by_type"][match.type] = 0
            stats["by_type"][match.type] += 1

        return stats
