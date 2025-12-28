"""
Grounding Check - Verifica que la respuesta esté fundamentada en el contexto
"""
import re
from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass
class GroundingResult:
    """Resultado del chequeo de grounding"""
    is_grounded: bool
    score: float  # 0.0 a 1.0
    ungrounded_claims: list[str]
    evidence_found: list[dict]
    details: str


class GroundingChecker:
    """
    Verifica que las afirmaciones en la respuesta estén respaldadas
    por los chunks del contexto.
    """

    def __init__(
        self,
        min_similarity: float = 0.3,  # Reducido para ser menos estricto
        min_grounding_ratio: float = 0.5  # Reducido: 50% de afirmaciones respaldadas
    ):
        """
        Args:
            min_similarity: Similitud mínima para considerar una afirmación respaldada
            min_grounding_ratio: Proporción mínima de afirmaciones que deben estar respaldadas
        """
        self.min_similarity = min_similarity
        self.min_grounding_ratio = min_grounding_ratio

    def check(
        self,
        answer: str,
        context_chunks: list[dict]
    ) -> GroundingResult:
        """
        Verifica si la respuesta está fundamentada en el contexto.

        Args:
            answer: Respuesta generada por el LLM
            context_chunks: Chunks recuperados con 'content' key

        Returns:
            GroundingResult con el análisis
        """
        # Extraer afirmaciones de la respuesta
        claims = self._extract_claims(answer)

        if not claims:
            return GroundingResult(
                is_grounded=True,
                score=1.0,
                ungrounded_claims=[],
                evidence_found=[],
                details="No se encontraron afirmaciones verificables"
            )

        # Combinar todo el contexto
        full_context = " ".join(
            chunk.get("content", "") for chunk in context_chunks
        ).lower()

        # Verificar cada afirmación
        grounded_claims = []
        ungrounded_claims = []
        evidence_found = []

        for claim in claims:
            is_supported, similarity, evidence = self._check_claim(
                claim, full_context, context_chunks
            )

            if is_supported:
                grounded_claims.append(claim)
                if evidence:
                    evidence_found.append(evidence)
            else:
                ungrounded_claims.append(claim)

        # Calcular score
        grounding_ratio = len(grounded_claims) / len(claims) if claims else 1.0
        is_grounded = grounding_ratio >= self.min_grounding_ratio

        return GroundingResult(
            is_grounded=is_grounded,
            score=grounding_ratio,
            ungrounded_claims=ungrounded_claims,
            evidence_found=evidence_found,
            details=f"{len(grounded_claims)}/{len(claims)} afirmaciones respaldadas"
        )

    def _extract_claims(self, text: str) -> list[str]:
        """
        Extrae afirmaciones verificables del texto.
        Divide en oraciones y filtra las que parecen afirmaciones.
        """
        # Dividir en oraciones
        sentences = re.split(r'[.!?]\s+', text)

        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Filtrar oraciones muy cortas o que son preguntas
            if len(sentence) < 20:
                continue
            if sentence.endswith('?'):
                continue
            # Filtrar frases de cortesía/conexión
            skip_patterns = [
                r'^según los documentos',
                r'^de acuerdo con',
                r'^en resumen',
                r'^en conclusión',
                r'^cabe mencionar',
                r'^es importante',
                r'^no encontré',
                r'^no tengo información',
            ]
            if any(re.match(p, sentence.lower()) for p in skip_patterns):
                continue

            claims.append(sentence)

        return claims

    def _check_claim(
        self,
        claim: str,
        full_context: str,
        chunks: list[dict]
    ) -> tuple[bool, float, dict | None]:
        """
        Verifica si una afirmación está respaldada por el contexto.

        Returns:
            (is_supported, similarity_score, evidence_dict)
        """
        claim_lower = claim.lower()

        # Estrategia 1: Buscar fragmentos clave del claim en el contexto
        key_phrases = self._extract_key_phrases(claim_lower)

        matches_found = 0
        for phrase in key_phrases:
            if phrase in full_context:
                matches_found += 1

        phrase_match_ratio = matches_found / len(key_phrases) if key_phrases else 0

        # Estrategia 2: Similitud con chunks individuales
        best_similarity = 0.0
        best_chunk = None

        for chunk in chunks:
            chunk_text = chunk.get("content", "").lower()
            similarity = self._calculate_similarity(claim_lower, chunk_text)

            if similarity > best_similarity:
                best_similarity = similarity
                best_chunk = chunk

        # Combinar scores
        combined_score = (phrase_match_ratio * 0.6) + (best_similarity * 0.4)

        is_supported = combined_score >= self.min_similarity

        evidence = None
        if is_supported and best_chunk:
            evidence = {
                "source": best_chunk.get("metadata", {}).get("source", "unknown"),
                "page": best_chunk.get("metadata", {}).get("page"),
                "similarity": best_similarity,
                "excerpt": best_chunk.get("content", "")[:150]
            }

        return is_supported, combined_score, evidence

    def _extract_key_phrases(self, text: str) -> list[str]:
        """Extrae frases clave (sustantivos, números, términos técnicos)"""
        # Eliminar stopwords comunes
        stopwords = {
            'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
            'de', 'del', 'al', 'a', 'en', 'con', 'por', 'para',
            'que', 'se', 'es', 'son', 'como', 'más', 'pero', 'si',
            'su', 'sus', 'este', 'esta', 'estos', 'estas', 'ese',
            'y', 'o', 'e', 'u', 'ni', 'no', 'sí', 'también'
        }

        words = re.findall(r'\b\w{3,}\b', text.lower())
        key_words = [w for w in words if w not in stopwords]

        # Crear n-gramas (2-3 palabras)
        phrases = []
        for i in range(len(key_words) - 1):
            phrases.append(f"{key_words[i]} {key_words[i+1]}")

        # Agregar palabras individuales importantes
        phrases.extend(key_words[:5])

        return phrases[:10]  # Limitar a 10 frases clave

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calcula similitud entre dos textos usando SequenceMatcher"""
        # Para textos largos, usar una muestra
        if len(text2) > 1000:
            # Buscar la mejor ventana de texto2 que coincida con text1
            window_size = min(len(text1) * 2, len(text2))
            best_ratio = 0.0

            for i in range(0, len(text2) - window_size + 1, 100):
                window = text2[i:i + window_size]
                ratio = SequenceMatcher(None, text1, window).ratio()
                best_ratio = max(best_ratio, ratio)

            return best_ratio
        else:
            return SequenceMatcher(None, text1, text2).ratio()
