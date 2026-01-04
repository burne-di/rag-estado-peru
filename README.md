# RAG Estado Perú

### Sistema de Preguntas y Respuestas con citas verificables sobre normativa pública peruana

[![CI](https://github.com/Ruben-Q/rag-estado-peru/actions/workflows/ci.yml/badge.svg)](https://github.com/Ruben-Q/rag-estado-peru/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Sistema **RAG (Retrieval-Augmented Generation)** end-to-end para responder preguntas sobre **documentos públicos del Estado Peruano** (normativa tributaria, resoluciones, comunicados en PDF/HTML), retornando respuestas **fundamentadas con citas verificables**.

---

## Competencias Demostradas

### GenAI / AI Engineering
- Pipeline RAG completo **desde cero** (sin LangChain) - demuestra comprensión profunda
- Prompt engineering con output JSON estructurado
- **Guardrails**: anti-alucinación (grounding check), política de rechazo, sanitización PII
- Evaluación offline con métricas de calidad RAG (Hit@K, Faithfulness)
- **Optimizaciones**: Response Cache, Model Routing, Streaming UX

### Ingeniería de Software
- Arquitectura modular y reutilizable (`packages/rag_core`)
- API REST con FastAPI + Pydantic + SSE Streaming
- Docker/Compose para despliegue
- CI/CD con GitHub Actions (lint, test, docker, eval)
- Testing unitario y smoke tests
- Documentación técnica completa

---

## Demo Rápida

```bash
# 1. Clonar e instalar
git clone https://github.com/Ruben-Q/rag-estado-peru.git
cd rag-estado-peru
pip install -e .

# 2. Configurar API keys (al menos una)
cp .env.example .env
# Editar .env con GROQ_API_KEY y/o GOOGLE_API_KEY

# 3. Ingestar documentos
python scripts/ingest.py

# 4. Hacer consultas
python scripts/query.py -i
```

**Ejemplo de consulta:**
```
📝 Tu pregunta: ¿Cuál es el plazo para presentar una reclamación tributaria?

📌 RESPUESTA:
El plazo para presentar una reclamación tributaria es de 20 días hábiles
contados desde el día siguiente de la notificación del acto administrativo.

📚 FUENTES:
[1] Codigo-Tributario-Sunat.pdf - Página 45
    Relevancia: 92%
```

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENTE                                 │
│                 (Web UI / API REST / CLI)                       │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Service                            │
│    /health   /query   /query/stream   /ingest   /stats          │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                       RAG Pipeline                               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Cache → PII Scrub → Retrieval → Router → Generator → Guard │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  OPTIMIZACIONES: Cache │ Model Routing │ Streaming │ Norm   ││
│  └─────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────┘
          │                   │                    │
   ┌──────┴──────┐    ┌───────┴───────┐    ┌───────┴───────┐
   ▼             ▼    ▼               ▼    ▼               ▼
┌──────────┐  ┌──────────────────┐  ┌─────────────┐  ┌─────────────┐
│ ChromaDB │  │   Multi-Provider │  │  Sentence   │  │   Cache     │
│  Vector  │  │ ┌──────┐┌──────┐ │  │ Transformers│  │   (JSON)    │
│  Store   │  │ │ Groq ││Gemini│ │  │ (Embeddings)│  │             │
└──────────┘  │ └──────┘└──────┘ │  └─────────────┘  └─────────────┘
              │   (auto-fallback)│
              └──────────────────┘
```

### Flujo de Consulta con Guardrails

1. **Query** → Recibe pregunta del usuario
2. **Cache Check** → Verifica si hay respuesta cacheada
3. **PII Scrubber** → Detecta y redacta información sensible (DNI, RUC, emails)
4. **Refusal Policy (pre)** → Rechaza queries fuera de tema
5. **Retrieval** → Busca chunks relevantes en ChromaDB
6. **Model Router** → Selecciona modelo óptimo (lite vs flash) según complejidad
7. **Generator** → Genera respuesta JSON estructurada con Gemini (streaming opcional)
8. **Grounding Check** → Verifica que respuesta esté fundamentada
9. **Refusal Policy (post)** → Rechaza si grounding < 50%
10. **Cache Save** → Guarda respuesta exitosa en caché
11. **Response** → Retorna answer + citations + confidence + from_cache

---

## Estructura del Proyecto

```
rag-estado-peru/
├── packages/rag_core/           # Lógica central RAG (sin LangChain)
│   ├── config.py                # Configuración con pydantic-settings
│   ├── loaders.py               # Carga PDF y HTML
│   ├── chunker.py               # División en chunks con overlap
│   ├── vectorstore.py           # ChromaDB + embeddings
│   ├── generator.py             # Multi-provider generator + streaming
│   ├── pipeline.py              # Orquestador principal
│   ├── cache.py                 # Response cache con TTL
│   ├── router.py                # Model routing por complejidad
│   ├── providers/               # Abstracción multi-provider LLM
│   │   ├── base.py              # Interfaz abstracta LLMProvider
│   │   ├── groq.py              # Provider Groq (Llama 3.3)
│   │   ├── gemini.py            # Provider Gemini
│   │   └── factory.py           # Factory con auto-detect y fallback
│   ├── guardrails/              # Validación y seguridad
│   │   ├── grounding_check.py   # Anti-alucinación
│   │   ├── refusal_policy.py    # Política de rechazo
│   │   └── pii_scrubber.py      # Sanitización PII
│   └── eval/                    # Evaluación de calidad
│       ├── dataset.py           # Dataset de evaluación
│       ├── metrics.py           # Hit@K, Faithfulness, etc.
│       └── report.py            # Generación de reportes
│
├── services/api/                # API FastAPI
│   ├── main.py                  # Endpoints + SSE streaming
│   ├── schemas.py               # Pydantic models
│   └── static/index.html        # Interfaz web
│
├── scripts/                     # CLI utilities
│   ├── ingest.py                # Ingesta de documentos
│   ├── query.py                 # Consultas interactivas
│   ├── eval_run.py              # Ejecutar evaluación
│   └── build_eval_set.py        # Crear dataset de eval
│
├── tests/                       # Tests
│   ├── test_chunker.py
│   ├── test_guardrails.py
│   └── test_api_smoke.py
│
├── docs/                        # Documentación
│   └── KNOWLEDGE_BASE.md        # Base de conocimientos completa
│
├── data/
│   ├── raw/                     # PDFs/HTMLs originales
│   ├── samples/                 # PDFs de ejemplo (va al repo)
│   ├── cache/                   # Response cache persistido
│   └── chroma/                  # Vector store persistido
│
├── .github/workflows/ci.yml     # GitHub Actions (lint, test, docker, eval)
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Stack Tecnológico

| Componente | Tecnología | Justificación |
|------------|------------|---------------|
| **LLM** | Groq (Llama 3.3) + Gemini 2.5 Flash | Multi-provider con fallback automático |
| **Embeddings** | sentence-transformers (multilingual) | 100% local, sin costos |
| **Vector Store** | ChromaDB | Simple, persistencia local |
| **API** | FastAPI + Pydantic | Async, validación automática |
| **Contenedores** | Docker + Compose | Reproducibilidad |
| **CI/CD** | GitHub Actions | Lint, tests, build |

### Providers LLM Soportados

| Provider | Modelos | Velocidad | Notas |
|----------|---------|-----------|-------|
| **Groq** | llama-3.3-70b, llama-3.1-8b | Ultra-rápido | Recomendado (LPU inference) |
| **Gemini** | gemini-2.5-flash, gemini-2.0-flash-lite | Rápido | Tier gratuito disponible |

El sistema selecciona automáticamente el provider disponible y hace fallback si uno falla.

---

## API Endpoints

### `GET /health`
Health check del servicio.

### `GET /stats`
Estadísticas del sistema (chunks indexados, modelo, config).

### `POST /query`
Consulta RAG con citas.

**Request:**
```json
{
  "question": "¿Cuál es el plazo para presentar una reclamación?",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "El plazo es de 20 días hábiles...",
  "citations": [
    {
      "quote": "veinte (20) días hábiles",
      "source": "Codigo-Tributario-Sunat.pdf",
      "page": 45,
      "relevance_score": 0.92
    }
  ],
  "confidence": 0.85,
  "refusal": false,
  "latency_ms": 1234,
  "guardrails": {
    "grounding_score": 0.88,
    "is_grounded": true
  }
}
```

### `POST /query/stream`
Consulta RAG con **streaming SSE** (Server-Sent Events).

Retorna tokens en tiempo real mientras se genera la respuesta:
```
data: {"type": "chunk", "content": "El plazo"}
data: {"type": "chunk", "content": " es de 20"}
data: {"type": "done", "result": {...}}
```

### `POST /ingest`
Ingesta documentos al vector store.

---

## Guardrails Implementados

### 1. Grounding Check (Anti-alucinación)
Verifica que cada afirmación en la respuesta esté respaldada por el contexto recuperado.
- Extrae claims de la respuesta
- Compara con chunks del contexto
- Calcula `grounding_score` (0-1)

### 2. Refusal Policy
Rechaza respuestas cuando:
- No hay chunks relevantes (score < 0.3)
- Query fuera de tema (recetas, deportes, etc.)
- Grounding insuficiente (< 0.5)

### 3. PII Scrubber
Detecta y redacta información sensible:
- DNI peruano (8 dígitos)
- RUC (11 dígitos)
- Teléfonos, emails, tarjetas

---

## Evaluación de Calidad

### Métricas
- **Hit@K**: ¿Fuente correcta en top-k?
- **Faithfulness**: ¿Respuesta fiel al contexto?
- **Answer Relevance**: ¿Responde la pregunta?
- **Latency**: Tiempo de respuesta

### Ejecutar Evaluación
```bash
# Crear dataset de ejemplo
python scripts/eval_run.py --create-sample

# Ejecutar evaluación
python scripts/eval_run.py --report
```

### Umbrales de Aceptación
- Hit@K ≥ 70%
- Faithfulness ≥ 70%

---

## Ejecución

### Desarrollo Local
```bash
# Instalar
pip install -e ".[dev]"

# Ingestar documentos
python scripts/ingest.py --directory ./data/raw

# Consultas interactivas
python scripts/query.py -i

# API
uvicorn services.api.main:app --reload
```

### Docker
```bash
# Construir y levantar
docker compose up --build

# Swagger UI
open http://localhost:8000/docs
```

### Makefile
```bash
make install      # Instalar dependencias
make ingest       # Ingestar documentos
make query        # Modo interactivo
make run-api      # Levantar API
make test         # Ejecutar tests
make docker-up    # Docker compose up
make eval         # Ejecutar evaluación
```

---

## Documentación

| Documento | Descripción |
|-----------|-------------|
| [KNOWLEDGE_BASE.md](docs/KNOWLEDGE_BASE.md) | **Base de conocimientos completa** - Arquitectura, conceptos RAG, flujos, decisiones técnicas |

---

## Roadmap

- [x] **Hito 0**: Skeleton + /health
- [x] **Hito 1**: Ingesta + ChromaDB
- [x] **Hito 2**: /query con citas JSON
- [x] **Hito 3**: Guardrails + evaluación
- [x] **Hito 4**: CI + Docker + documentación
- [x] **Hito 5**: Response Cache + Model Routing + Streaming UX

### Optimizaciones Implementadas

| Feature | Descripción | Beneficio |
|---------|-------------|-----------|
| **Multi-Provider** | Groq + Gemini con fallback automático | Alta disponibilidad, evita rate limits |
| **Response Cache** | Caché LRU con TTL de 24h | ~40% ahorro en llamadas API |
| **Model Routing** | Selección automática de modelo según complejidad | Queries simples → modelo económico |
| **Streaming UX** | Server-Sent Events para respuestas en tiempo real | Mejor experiencia de usuario |
| **Query Normalization** | Normaliza queries para mejor cache hit rate | Mayor eficiencia de caché |

### Backlog Futuro

- [ ] Reranking con cross-encoder
- [ ] Filtros por entidad/fecha
- [ ] Multi-tenancy
- [ ] Observabilidad (métricas, tracing)

---

## Contribuir

1. Fork el repositorio
2. Crear branch (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -m 'Agregar funcionalidad'`)
4. Push (`git push origin feature/nueva-funcionalidad`)
5. Abrir Pull Request

---

## Licencia

MIT License - ver [LICENSE](LICENSE)

---

## Autor

Desarrollado como proyecto de portafolio para demostrar competencias en **AI Engineering / GenAI**.

**Contacto:** [rubendqv@gmail.com - Ruben Quispe]
