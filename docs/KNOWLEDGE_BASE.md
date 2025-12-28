# RAG Estado Peru - Base de Conocimientos

> Documentación técnica completa del proyecto para aprendizaje

## Tabla de Contenidos

1. [Arquitectura General](#arquitectura-general)
2. [Conceptos Fundamentales de RAG](#conceptos-fundamentales-de-rag)
3. [Estructura del Proyecto](#estructura-del-proyecto)
4. [Componentes Core (`packages/rag_core/`)](#componentes-core)
5. [API (`services/api/`)](#api)
6. [Scripts Utilitarios](#scripts-utilitarios)
7. [Tests](#tests)
8. [GitHub Actions CI/CD](#github-actions-cicd)
9. [Dependencias y Por Qué Cada Una](#dependencias-y-por-qué-cada-una)
10. [Flujo de Datos Completo](#flujo-de-datos-completo)
11. [Optimizaciones Implementadas](#optimizaciones-implementadas)

---

## Arquitectura General

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USUARIO                                         │
│                    (Pregunta: "¿Qué dice el artículo 5?")                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FASTAPI (services/api/)                            │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │   /query    │  │ /query/stream│  │   /ingest    │  │  /health /stats │  │
│  └─────────────┘  └──────────────┘  └──────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE (packages/rag_core/)                    │
│                                                                              │
│  ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌──────────┐   ┌──────────┐ │
│  │  Loader  │──▶│ Chunker  │──▶│VectorStore│──▶│Generator │──▶│Guardrails│ │
│  │  (PDF)   │   │ (texto)  │   │ (ChromaDB)│   │ (Gemini) │   │(validar) │ │
│  └──────────┘   └──────────┘   └───────────┘   └──────────┘   └──────────┘ │
│                                      │                                       │
│                              ┌───────┴───────┐                              │
│                              │   Embeddings  │                              │
│                              │(sentence-tfrs)│                              │
│                              └───────────────┘                              │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  OPTIMIZACIONES:  Cache │ Model Routing │ Streaming │ Normalization  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Conceptos Fundamentales de RAG

### ¿Qué es RAG?

**RAG (Retrieval-Augmented Generation)** es una técnica que combina:

1. **Retrieval (Recuperación)**: Buscar información relevante en una base de datos
2. **Augmented (Aumentada)**: Usar esa información como contexto adicional
3. **Generation (Generación)**: El LLM genera respuestas basándose en ese contexto

### ¿Por qué RAG y no solo un LLM?

| Problema del LLM Solo | Solución con RAG |
|----------------------|------------------|
| No conoce tus documentos privados | Busca en TUS documentos |
| Puede "alucinar" información | Se basa en contexto real |
| Conocimiento desactualizado | Siempre usa datos actuales |
| No cita fuentes | Puede citar exactamente de dónde sacó la info |

### ¿Qué son los Embeddings?

Los **embeddings** son representaciones numéricas (vectores) del significado de un texto.

```
Texto: "El código tributario establece..."
         │
         ▼ (modelo de embeddings)

Vector: [0.23, -0.45, 0.78, 0.12, ..., 0.56]  ← 384 dimensiones
```

**¿Por qué vectores?**
- Los vectores permiten calcular **similitud semántica**
- Textos con significado similar tienen vectores cercanos
- Podemos buscar "los 5 textos más similares a la pregunta"

```
"¿Cuál es el plazo para declarar impuestos?"
    │
    ▼ (vector de la pregunta)

[0.21, -0.43, 0.80, ...]  ← Cercano a chunks sobre plazos tributarios
                           Lejano de chunks sobre contratos laborales
```

### ¿Qué es un Vector Store?

Es una base de datos especializada en:
1. **Almacenar** vectores junto con el texto original
2. **Buscar** los K vectores más cercanos a uno dado (búsqueda de similitud)

En este proyecto usamos **ChromaDB** porque:
- Es fácil de usar (Python nativo)
- No requiere servidor separado
- Persistencia en disco automática
- Perfecto para desarrollo y proyectos medianos

---

## Estructura del Proyecto

```
rag-estado-peru/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions - CI/CD automático
│
├── data/
│   └── pdfs/                   # PDFs a indexar (no va al repo)
│
├── docs/
│   └── KNOWLEDGE_BASE.md       # Este archivo
│
├── packages/                   # Lógica reutilizable (como una librería)
│   └── rag_core/
│       ├── __init__.py         # Exporta todo lo público
│       ├── cache.py            # Cache de respuestas
│       ├── chunker.py          # Divide documentos en chunks
│       ├── config.py           # Configuración con Pydantic
│       ├── generator.py        # Genera respuestas con Gemini
│       ├── loaders.py          # Carga PDFs y HTML
│       ├── pipeline.py         # Orquesta todo el flujo
│       ├── router.py           # Selecciona modelo según complejidad
│       ├── vectorstore.py      # Embeddings + ChromaDB
│       ├── eval/               # Sistema de evaluación
│       │   ├── dataset.py
│       │   ├── metrics.py
│       │   └── report.py
│       └── guardrails/         # Validaciones de seguridad
│           ├── grounding_check.py
│           ├── pii_scrubber.py
│           └── refusal_policy.py
│
├── scripts/                    # Scripts ejecutables
│   ├── ingest.py              # Indexa documentos
│   ├── query.py               # Consulta desde CLI
│   ├── debug_chunks.py        # Debug de chunks
│   ├── build_eval_set.py      # Construye dataset de evaluación
│   ├── eval_run.py            # Ejecuta evaluación
│   └── test_pipeline.py       # Prueba rápida
│
├── services/                   # Servicios desplegables
│   └── api/
│       ├── __init__.py
│       ├── main.py            # FastAPI app
│       ├── schemas.py         # Modelos Pydantic para API
│       └── static/
│           └── index.html     # Interfaz web
│
├── tests/                      # Tests automatizados
│   ├── test_api_smoke.py
│   ├── test_chunker.py
│   └── test_guardrails.py
│
├── .env.example               # Variables de entorno ejemplo
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml             # Configuración del proyecto Python
└── README.md
```

### ¿Por qué esta estructura?

| Carpeta | Propósito | ¿Por qué separada? |
|---------|-----------|-------------------|
| `packages/` | Código reutilizable | Puede importarse desde cualquier lugar |
| `services/` | Apps desplegables | Un servicio = una responsabilidad |
| `scripts/` | Tareas de CLI | No es código importable, son ejecutables |
| `tests/` | Tests automáticos | Separados del código de producción |
| `.github/` | CI/CD | Convención de GitHub |

---

## Componentes Core

### `config.py` - Configuración Centralizada

```python
class Settings(BaseSettings):
    google_api_key: str           # API key de Gemini
    gemini_model: str = "gemini-2.5-flash"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_results: int = 5
```

**¿Por qué Pydantic Settings?**
- Lee automáticamente de variables de entorno
- Lee de archivo `.env`
- Validación de tipos automática
- Valores por defecto
- Un solo lugar para toda la configuración

### `loaders.py` - Carga de Documentos

**PDFLoader**: Extrae texto de PDFs página por página

```python
# ¿Por qué pypdf + pdfplumber?
# pypdf: Rápido, funciona con la mayoría de PDFs
# pdfplumber: Fallback para PDFs complejos con tablas
```

**HTMLLoader**: Para cargar páginas web

**`Document`**: Estructura estándar
```python
@dataclass
class Document:
    content: str           # El texto extraído
    metadata: dict         # {source: "archivo.pdf", page: 1, ...}
```

### `chunker.py` - División en Chunks

**¿Por qué dividir en chunks?**
1. Los LLMs tienen límite de contexto (tokens)
2. Chunks pequeños = búsqueda más precisa
3. Evita "ahogar" al modelo con texto irrelevante

```python
# Configuración típica:
chunk_size = 500      # ~500 caracteres por chunk
chunk_overlap = 50    # 50 caracteres se repiten entre chunks

# Ejemplo:
# Documento: "AAAAA BBBBB CCCCC DDDDD EEEEE"
# Chunk 1: "AAAAA BBBBB"
# Chunk 2: "BBBBB CCCCC"  ← Overlap preserva contexto
# Chunk 3: "CCCCC DDDDD"
```

**¿Por qué overlap?**
- Si una idea cruza dos chunks, el overlap asegura que no se pierda
- Mejora la recuperación de información en los bordes

### `vectorstore.py` - Embeddings y Búsqueda

```python
class VectorStore:
    def __init__(self):
        # 1. Modelo de embeddings (convierte texto → vectores)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # 2. Base de datos vectorial (almacena y busca)
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection("rag_chunks")
```

**¿Por qué `all-MiniLM-L6-v2`?**
| Característica | Valor |
|----------------|-------|
| Dimensiones | 384 (pequeño, rápido) |
| Idiomas | Multilingüe (incluye español) |
| Velocidad | Muy rápido |
| Calidad | Buena para propósito general |
| Tamaño | ~80MB |

**¿Por qué ChromaDB?**
- **Cero configuración**: No necesita servidor externo
- **Persistencia**: Guarda en disco automáticamente
- **Python nativo**: Se integra fácilmente
- **Gratis y open source**

Alternativas para producción: Pinecone, Weaviate, Qdrant, Milvus

### `generator.py` - Generación con Gemini

```python
class GeminiGenerator:
    def generate(self, query, context_chunks, model_override=None):
        # 1. Construir prompt con contexto
        prompt = f"""
        Eres un experto en normativa peruana.

        DOCUMENTOS:
        {context_chunks}

        PREGUNTA:
        {query}

        Responde en JSON: {{"answer": "...", "citations": [...]}}
        """

        # 2. Llamar a Gemini
        response = model.generate_content(prompt)

        # 3. Parsear JSON de la respuesta
        return self._parse_json_response(response.text)
```

**¿Por qué JSON estructurado?**
- Facilita extraer la respuesta y las citas
- El frontend puede procesar la respuesta fácilmente
- Permite validar que el modelo respondió correctamente

**Streaming (`generate_stream`)**:
- Envía tokens mientras se generan
- El usuario ve la respuesta "escribiéndose"
- Mejor experiencia de usuario

### `pipeline.py` - Orquestador Principal

```python
class RAGPipeline:
    def query(self, question):
        # 1. Normalizar pregunta (quitar acentos, limpiar)
        normalized = normalize_query(question)

        # 2. Verificar caché
        if cached := self.cache.get(normalized):
            return cached

        # 3. Buscar chunks relevantes
        chunks = self.vector_store.search(normalized, top_k=5)

        # 4. Evaluar si debemos rechazar (muy poca relevancia)
        if self.refusal_policy.evaluate(chunks).should_refuse:
            return {"refusal": True, "message": "..."}

        # 5. Seleccionar modelo (routing)
        model = self.router.route(question).model

        # 6. Generar respuesta
        response = self.generator.generate(question, chunks, model)

        # 7. Verificar grounding (¿está basada en los documentos?)
        grounding = self.grounding_checker.check(response, chunks)

        # 8. Guardar en caché
        self.cache.set(normalized, response)

        return response
```

### `router.py` - Model Routing

**¿Por qué routing?**
- Queries simples → modelo económico (`gemini-2.0-flash-lite`)
- Queries complejas → modelo más capaz (`gemini-2.5-flash`)
- **Ahorra costos** sin sacrificar calidad

```python
# Indicadores de complejidad:
COMPLEX = ["analiza", "compara", "diferencias", "por qué", "explica"]
SIMPLE = ["qué es", "cuándo", "dónde", "cuánto"]

# Score de complejidad (0.0 a 1.0):
# - Longitud de la pregunta (20%)
# - Palabras complejas encontradas (50%)
# - Palabras simples (-30% penalty)
# - Cantidad de contexto necesario (10%)
```

### `cache.py` - Cache de Respuestas

**¿Por qué cachear?**
- Las llamadas a Gemini cuestan dinero y tiempo
- Preguntas repetidas no necesitan regenerarse
- Reduce latencia dramáticamente

```python
class ResponseCache:
    def __init__(self, ttl_hours=24, max_entries=1000):
        self.ttl = ttl_hours * 3600  # Tiempo de vida
        self.max_entries = max_entries  # Límite (LRU eviction)
        self.storage = {}

    def get(self, key):
        # Verificar expiración
        if entry.timestamp + self.ttl < now:
            del self.storage[key]
            return None
        return entry.value
```

### `guardrails/` - Validaciones de Seguridad

#### `grounding_check.py`
Verifica que la respuesta esté **basada en los documentos**:
- Extrae afirmaciones de la respuesta
- Busca evidencia en los chunks
- Si <50% está respaldado → rechazar

#### `refusal_policy.py`
Decide cuándo **rechazar** responder:
- No hay chunks relevantes
- Score de relevancia muy bajo
- Pregunta fuera de tema

#### `pii_scrubber.py`
Detecta y oculta **información personal**:
- DNI, RUC
- Emails, teléfonos
- Importante para logs y privacidad

---

## API

### `services/api/main.py`

```python
app = FastAPI(title="RAG Estado Peru API")

@app.get("/health")      # ¿El servicio está vivo?
@app.get("/stats")       # Estadísticas del sistema
@app.post("/query")      # Consulta normal
@app.post("/query/stream")  # Consulta con streaming
@app.post("/ingest")     # Indexar documentos
@app.delete("/clear")    # Limpiar índice
```

### `schemas.py` - Modelos de Request/Response

```python
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int | None = Field(None, ge=1, le=20)

class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float
    from_cache: bool
```

**¿Por qué Pydantic?**
- Validación automática de inputs
- Documentación OpenAPI generada automáticamente
- Serialización/deserialización JSON
- Type hints + runtime validation

### `static/index.html` - Interfaz Web

Una interfaz simple que:
- Envía preguntas a la API
- Muestra respuestas con streaming
- Muestra citas y confianza
- Indica cuando viene de caché

---

## Scripts Utilitarios

### `scripts/ingest.py`
```bash
python scripts/ingest.py data/pdfs/
```
Indexa todos los PDFs de un directorio.

### `scripts/query.py`
```bash
python scripts/query.py "¿Qué es el código tributario?"
```
Hace una consulta desde la línea de comandos.

### `scripts/debug_chunks.py`
Muestra los chunks almacenados y sus embeddings.
Útil para debugging.

### `scripts/eval_run.py`
Ejecuta evaluación automática del sistema RAG.
Compara respuestas con un dataset de referencia.

---

## Tests

### Estructura de Tests

```python
# tests/test_chunker.py
def test_basic_chunking():
    """Verifica que el chunking funciona"""
    doc = Document(content="texto largo...", metadata={})
    chunks = chunk_documents([doc], chunk_size=100)
    assert len(chunks) > 1

# tests/test_guardrails.py
def test_detect_dni():
    """Verifica detección de DNI"""
    scrubber = PIIScrubber()
    _, pii = scrubber.scrub("Mi DNI es 12345678")
    assert len(pii) == 1

# tests/test_api_smoke.py
def test_health_endpoint(client):
    """Verifica que /health responde"""
    response = client.get("/health")
    assert response.status_code == 200
```

### Ejecutar Tests
```bash
pytest tests/ -v           # Todos los tests
pytest tests/test_chunker.py  # Solo un archivo
pytest -k "test_dni"       # Solo tests que contengan "dni"
```

---

## GitHub Actions CI/CD

### ¿Cómo funciona?

El archivo `.github/workflows/ci.yml` le dice a GitHub:

> "Cada vez que alguien haga push o PR, ejecuta estos pasos"

```yaml
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
```

### Jobs Configurados

#### 1. `lint` - Verificación de Código
```yaml
- name: Run ruff check      # ¿Hay errores de estilo?
- name: Run ruff format     # ¿El código está formateado?
```

#### 2. `test` - Tests Automáticos
```yaml
needs: lint                 # Solo corre si lint pasó
- name: Install dependencies
- name: Run tests
```

#### 3. `docker` - Build de Docker
```yaml
needs: test                 # Solo corre si tests pasaron
- name: Build Docker image
```

#### 4. `evaluate` - Evaluación RAG (solo en PRs a main)
```yaml
if: github.event_name == 'pull_request'
- name: Run evaluation
- name: Upload report
```

### ¿Por qué CI/CD?

| Sin CI/CD | Con CI/CD |
|-----------|-----------|
| "Funciona en mi máquina" | Funciona en todas partes |
| Errores descubiertos tarde | Errores detectados al instante |
| Código inconsistente | Estilo uniforme garantizado |
| Deploy manual | Deploy automático |

### Secrets (Variables Secretas)

En GitHub → Settings → Secrets:
```
GOOGLE_API_KEY = "tu-api-key"
```

El CI las usa así:
```yaml
env:
  GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
```

---

## Dependencias y Por Qué Cada Una

### Core
| Librería | Propósito | ¿Por qué esta? |
|----------|-----------|----------------|
| `fastapi` | Framework web | Rápido, moderno, async, autodocs |
| `uvicorn` | Servidor ASGI | El servidor recomendado para FastAPI |
| `pydantic` | Validación de datos | Type hints + validación runtime |

### RAG
| Librería | Propósito | ¿Por qué esta? |
|----------|-----------|----------------|
| `chromadb` | Vector store | Fácil, embebido, persistente |
| `sentence-transformers` | Embeddings | Modelos pre-entrenados de calidad |
| `google-generativeai` | API de Gemini | El LLM que usamos |

### Procesamiento de Documentos
| Librería | Propósito | ¿Por qué esta? |
|----------|-----------|----------------|
| `pypdf` | Leer PDFs | Rápido, puro Python |
| `pdfplumber` | PDFs complejos | Mejor con tablas y layouts |
| `beautifulsoup4` | Parsear HTML | El estándar para web scraping |
| `lxml` | Parser HTML/XML | Rápido, usado por BS4 |

### Desarrollo
| Librería | Propósito | ¿Por qué esta? |
|----------|-----------|----------------|
| `pytest` | Testing | El framework de tests más popular |
| `ruff` | Linting + formatting | Reemplaza flake8+black+isort, 10-100x más rápido |
| `httpx` | Cliente HTTP async | Para tests de API |

---

## Flujo de Datos Completo

### 1. Ingesta de Documentos

```
PDF → Loader → Document → Chunker → Chunks → Embeddings → ChromaDB
                  │                    │           │
                  ▼                    ▼           ▼
              {content,           [texto1,    [vector1,
               metadata}           texto2]     vector2]
```

### 2. Consulta

```
Pregunta del Usuario
         │
         ▼
    ┌─────────┐
    │ Cache?  │──── Sí ───▶ Retornar respuesta cacheada
    └────┬────┘
         │ No
         ▼
    Normalizar query (quitar acentos)
         │
         ▼
    Generar embedding de la pregunta
         │
         ▼
    Buscar K chunks más similares en ChromaDB
         │
         ▼
    ┌────────────────┐
    │ ¿Relevantes?   │──── No ───▶ Rechazar con mensaje
    └───────┬────────┘
            │ Sí
            ▼
    ┌────────────────┐
    │ Model Routing  │ ──▶ Seleccionar gemini-lite o gemini-flash
    └───────┬────────┘
            │
            ▼
    Enviar a Gemini: prompt + chunks
            │
            ▼
    Parsear respuesta JSON
            │
            ▼
    ┌────────────────┐
    │ ¿Grounded?     │──── No ───▶ Rechazar (alucinación)
    └───────┬────────┘
            │ Sí
            ▼
    Guardar en caché
            │
            ▼
    Retornar respuesta al usuario
```

---

## Optimizaciones Implementadas

### 1. Response Cache
- **Problema**: Llamar a Gemini cada vez es lento y costoso
- **Solución**: Cachear respuestas por 24 horas
- **Resultado**: Respuestas repetidas en <10ms vs ~2s

### 2. Model Routing
- **Problema**: Usar siempre el modelo más caro
- **Solución**: Analizar complejidad y elegir modelo apropiado
- **Resultado**: ~40% ahorro en costos API

### 3. Query Normalization
- **Problema**: "¿Qué es?" y "que es" se tratan como diferentes
- **Solución**: Normalizar (quitar acentos, signos)
- **Resultado**: Mejor hit rate de caché

### 4. Streaming UX
- **Problema**: Usuario espera 3-5 segundos sin feedback
- **Solución**: Server-Sent Events, mostrar tokens mientras llegan
- **Resultado**: Percepción de respuesta inmediata

### 5. Guardrails
- **Problema**: El LLM puede alucinar o dar información no basada en docs
- **Solución**: Verificar grounding, rechazar si no está respaldado
- **Resultado**: Respuestas más confiables

---

## Próximos Pasos Sugeridos

1. **Mejorar embeddings**: Probar modelos más grandes o fine-tuned para español legal
2. **Reranking**: Usar un modelo de reranking después de la búsqueda inicial
3. **Hybrid Search**: Combinar búsqueda semántica con keyword search (BM25)
4. **Evaluación continua**: Implementar métricas de RAGAS en producción
5. **Multi-tenancy**: Soportar múltiples usuarios/organizaciones
6. **Observabilidad**: Agregar logging estructurado, métricas, tracing

---

## Glosario

| Término | Definición |
|---------|------------|
| **Chunk** | Fragmento de texto de tamaño fijo |
| **Embedding** | Vector numérico que representa el significado de un texto |
| **Vector Store** | Base de datos optimizada para buscar vectores similares |
| **Grounding** | Verificar que una respuesta está basada en los documentos |
| **LLM** | Large Language Model (ej: Gemini, GPT, Claude) |
| **RAG** | Retrieval-Augmented Generation |
| **Token** | Unidad de texto para el LLM (~4 caracteres en español) |
| **Top-K** | Los K resultados más relevantes |
| **TTL** | Time To Live - tiempo de expiración del caché |
| **SSE** | Server-Sent Events - streaming del servidor al cliente |
