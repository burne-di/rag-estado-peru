"""
Text Chunker - División de documentos en chunks con metadata
"""
import uuid
from dataclasses import dataclass

from .loaders import Document


@dataclass
class Chunk:
    """Representa un chunk de texto con su metadata"""
    chunk_id: str
    content: str
    metadata: dict


class TextChunker:
    """
    Divide documentos en chunks de tamaño fijo con overlap.
    Preserva metadata del documento origen.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        length_function: callable = len
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function

    def split_document(self, document: Document) -> list[Chunk]:
        """
        Divide un documento en chunks
        """
        text = document.content
        chunks = []

        # Dividir por caracteres con overlap
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size

            # Intentar cortar en un espacio para no partir palabras
            if end < len(text):
                # Buscar el último espacio antes del límite
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk = Chunk(
                    chunk_id=self._generate_chunk_id(document, chunk_index),
                    content=chunk_text,
                    metadata={
                        **document.metadata,
                        "chunk_index": chunk_index,
                        "chunk_start": start,
                        "chunk_end": end,
                    }
                )
                chunks.append(chunk)
                chunk_index += 1

            # Mover el inicio considerando overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break

        return chunks

    def split_documents(self, documents: list[Document]) -> list[Chunk]:
        """
        Divide múltiples documentos en chunks
        """
        all_chunks = []
        for doc in documents:
            chunks = self.split_document(doc)
            all_chunks.extend(chunks)
        return all_chunks

    def _generate_chunk_id(self, document: Document, chunk_index: int) -> str:
        """Genera un ID único para el chunk"""
        source = document.metadata.get("source", "unknown")
        page = document.metadata.get("page", 0)
        # ID corto pero único
        unique_part = uuid.uuid4().hex[:8]
        return f"{source}::p{page}::c{chunk_index}::{unique_part}"


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50
) -> list[Chunk]:
    """
    Función de conveniencia para dividir documentos en chunks
    """
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.split_documents(documents)
