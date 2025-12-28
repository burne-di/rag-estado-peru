"""
Document Loaders - Carga y extracción de texto de PDFs y HTML
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import pdfplumber
import requests
from bs4 import BeautifulSoup


@dataclass
class Document:
    """Representa un documento cargado"""

    content: str
    metadata: dict

    def __post_init__(self):
        # Generar hash del contenido para trazabilidad
        self.metadata["content_hash"] = hashlib.sha256(
            self.content.encode()
        ).hexdigest()[:16]


class PDFLoader:
    """Carga documentos PDF y extrae texto por página"""

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        if self.file_path.suffix.lower() != ".pdf":
            raise ValueError(f"El archivo debe ser PDF: {file_path}")

    def load(self) -> list[Document]:
        """
        Carga el PDF y retorna una lista de Documents (uno por página)
        """
        documents = []

        with pdfplumber.open(self.file_path) as pdf:
            total_pages = len(pdf.pages)

            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""

                # Limpiar texto básico
                text = self._clean_text(text)

                if text.strip():  # Solo agregar si hay contenido
                    doc = Document(
                        content=text,
                        metadata={
                            "source": self.file_path.name,
                            "source_path": str(self.file_path),
                            "source_type": "pdf",
                            "page": page_num,
                            "total_pages": total_pages,
                        },
                    )
                    documents.append(doc)

        return documents

    def _clean_text(self, text: str) -> str:
        """Limpieza básica del texto extraído"""
        # Normalizar espacios múltiples
        text = re.sub(r"\s+", " ", text)
        # Eliminar caracteres de control
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return text.strip()


class HTMLLoader:
    """Carga documentos HTML (archivo local o URL) y extrae texto"""

    # Tags a ignorar
    IGNORE_TAGS = {
        "script",
        "style",
        "meta",
        "link",
        "noscript",
        "header",
        "footer",
        "nav",
        "aside",
        "form",
        "button",
        "input",
    }

    # Tags que indican secciones importantes
    CONTENT_TAGS = {
        "article",
        "main",
        "section",
        "div",
        "p",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "td",
        "th",
    }

    def __init__(self, source: str, is_url: bool = False):
        """
        Args:
            source: Ruta al archivo HTML o URL
            is_url: Si True, trata source como URL
        """
        self.source = source
        self.is_url = is_url or self._is_url(source)

    def _is_url(self, source: str) -> bool:
        """Detecta si el source es una URL"""
        try:
            result = urlparse(source)
            return all([result.scheme in ("http", "https"), result.netloc])
        except (ValueError, AttributeError):
            return False

    def load(self) -> list[Document]:
        """
        Carga el HTML y retorna una lista de Documents.
        Para HTML, retorna un solo documento con todo el contenido.
        """
        # Obtener contenido HTML
        if self.is_url:
            html_content = self._fetch_url()
        else:
            html_content = self._read_file()

        if not html_content:
            return []

        # Parsear y extraer texto
        soup = BeautifulSoup(html_content, "lxml")

        # Extraer metadata
        title = self._extract_title(soup)

        # Extraer texto limpio
        text = self._extract_text(soup)

        if not text.strip():
            return []

        # Dividir en secciones si es posible
        documents = self._split_by_sections(soup, text)

        if not documents:
            # Si no hay secciones, crear un documento único
            doc = Document(
                content=text,
                metadata={
                    "source": self._get_source_name(),
                    "source_path": self.source,
                    "source_type": "html",
                    "title": title,
                    "section": None,
                },
            )
            documents = [doc]

        return documents

    def _fetch_url(self) -> str | None:
        """Descarga contenido de una URL"""
        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"}
            response = requests.get(self.source, headers=headers, timeout=30)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            return response.text
        except Exception as e:
            print(f"Error descargando {self.source}: {e}")
            return None

    def _read_file(self) -> str | None:
        """Lee contenido de archivo local"""
        try:
            path = Path(self.source)
            if not path.exists():
                raise FileNotFoundError(f"Archivo no encontrado: {self.source}")

            # Intentar diferentes encodings
            for encoding in ["utf-8", "latin-1", "cp1252"]:
                try:
                    return path.read_text(encoding=encoding)
                except UnicodeDecodeError:
                    continue

            return path.read_text(errors="ignore")
        except Exception as e:
            print(f"Error leyendo {self.source}: {e}")
            return None

    def _extract_title(self, soup: BeautifulSoup) -> str | None:
        """Extrae el título del documento"""
        # Intentar tag title
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            return title_tag.string.strip()

        # Intentar h1
        h1_tag = soup.find("h1")
        if h1_tag:
            return h1_tag.get_text(strip=True)

        return None

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extrae texto limpio del HTML"""
        # Remover tags no deseados
        for tag in soup(self.IGNORE_TAGS):
            tag.decompose()

        # Intentar encontrar contenido principal
        main_content = soup.find("main") or soup.find("article") or soup.find("body")

        if main_content:
            text = main_content.get_text(separator=" ", strip=True)
        else:
            text = soup.get_text(separator=" ", strip=True)

        # Limpiar texto
        text = self._clean_text(text)
        return text

    def _clean_text(self, text: str) -> str:
        """Limpia el texto extraído"""
        # Normalizar espacios
        text = re.sub(r"\s+", " ", text)
        # Eliminar caracteres especiales problemáticos
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        # Eliminar secuencias de caracteres repetidos excesivos
        text = re.sub(r"(.)\1{10,}", r"\1\1\1", text)
        return text.strip()

    def _split_by_sections(self, soup: BeautifulSoup, full_text: str) -> list[Document]:
        """Intenta dividir el documento por secciones (h1, h2, h3)"""
        documents = []

        # Buscar encabezados como divisores de sección
        headings = soup.find_all(["h1", "h2", "h3"])

        if len(headings) < 2:
            return []  # No hay suficientes secciones

        for i, heading in enumerate(headings):
            section_title = heading.get_text(strip=True)

            # Obtener contenido hasta el siguiente heading
            content_parts = []
            for sibling in heading.find_next_siblings():
                if sibling.name in ["h1", "h2", "h3"]:
                    break
                text = sibling.get_text(strip=True)
                if text:
                    content_parts.append(text)

            section_content = " ".join(content_parts)

            if section_content and len(section_content) > 50:
                doc = Document(
                    content=f"{section_title}\n\n{section_content}",
                    metadata={
                        "source": self._get_source_name(),
                        "source_path": self.source,
                        "source_type": "html",
                        "section": section_title,
                        "section_index": i + 1,
                    },
                )
                documents.append(doc)

        return documents

    def _get_source_name(self) -> str:
        """Obtiene un nombre legible para el source"""
        if self.is_url:
            parsed = urlparse(self.source)
            return f"{parsed.netloc}{parsed.path[:30]}"
        else:
            return Path(self.source).name


def load_documents_from_directory(directory: str | Path) -> list[Document]:
    """
    Carga todos los PDFs y HTMLs de un directorio
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directorio no encontrado: {directory}")

    documents = []

    # Cargar PDFs (minúsculas y mayúsculas)
    pdf_files = list(directory.glob("*.pdf")) + list(directory.glob("*.PDF"))
    for pdf_path in pdf_files:
        try:
            loader = PDFLoader(pdf_path)
            docs = loader.load()
            documents.extend(docs)
            print(f"✓ Cargado: {pdf_path.name} ({len(docs)} páginas)")
        except Exception as e:
            print(f"✗ Error cargando {pdf_path.name}: {e}")

    # Cargar HTMLs
    html_files = list(directory.glob("*.html")) + list(directory.glob("*.htm"))
    for html_path in html_files:
        try:
            loader = HTMLLoader(str(html_path))
            docs = loader.load()
            documents.extend(docs)
            print(f"✓ Cargado: {html_path.name} ({len(docs)} secciones)")
        except Exception as e:
            print(f"✗ Error cargando {html_path.name}: {e}")

    return documents


def load_from_url(url: str) -> list[Document]:
    """
    Carga documentos desde una URL
    """
    loader = HTMLLoader(url, is_url=True)
    return loader.load()
