# RAG Estado Perú  
### Sistema de Preguntas y Respuestas con citas sobre normativa pública

Este proyecto implementa un sistema **RAG (Retrieval-Augmented Generation)** end-to-end para responder preguntas utilizando **documentos públicos del Estado Peruano** (normativa, resoluciones, comunicados en PDF/HTML), retornando respuestas **fundamentadas con citas verificables**.

El objetivo del repositorio es demostrar competencias reales de un **AI Engineer / GenAI Engineer**, cubriendo todo el ciclo: **ingesta de datos, embeddings, vector databases, retrieval, guardrails, evaluación, APIs y despliegue local**.

---

## 🎯 Objetivos del proyecto

- Construir un pipeline RAG completo con datos reales y públicos.
- Responder preguntas en lenguaje natural con **evidencia documental**.
- Implementar **guardrails mínimos** (anti-alucinación, sanitización básica).
- Evaluar la calidad del sistema RAG de forma **offline**.
- Exponer una **API productizable** (FastAPI).
- Mantener un stack **cloud-agnostic y sin costos** (free/local).

---

## 📌 Motivación

En entornos regulados (gobierno, banca, legal), un sistema de IA **no puede “inventar” respuestas**.  
Este proyecto prioriza:

- Trazabilidad  
- Explicabilidad  
- Evaluación de calidad  
- Buenas prácticas de ingeniería  

Más allá de una demo, apunta a un **diseño production-ready**.

---

## 🗂️ Fuentes de datos (Data pública)

Ejemplos de fuentes objetivo:

- Normativa y leyes (PDF):
  - Congreso del Perú
  - SUNAT (normas tributarias)
  - Ministerios y entidades públicas
- Comunicados y resoluciones en HTML/PDF

Las fuentes específicas se documentan en `docs/dataset_sources.md`.

---

## 🧠 Alcance del MVP

### Incluido
- Ingesta de documentos PDF/HTML
- Limpieza y normalización de texto
- Chunking con metadata
- Embeddings
- Vector Store local (Chroma por defecto)
- Retrieval top-k
- Generación de respuestas con citas
- API REST
- Evaluación offline (RAG quality)
- Tests básicos

### Fuera de alcance (por diseño)
- UI frontend
- Autenticación / OAuth
- Persistencia cloud
- Observabilidad avanzada distribuida

---

## 🏗️ Arquitectura (alto nivel)

### 1. Ingesta
```
PDF / HTML
   ↓
Loader
   ↓
Cleaner
   ↓
Chunker + Metadata
   ↓
Embeddings
   ↓
Vector Database
```

### 2. Consulta
```
Pregunta
   ↓
Embedding de query
   ↓
Retriever (top-k)
   ↓
Prompt controlado + contexto
   ↓
LLM
   ↓
Respuesta + citas
```

Las decisiones de diseño se documentan en `docs/decisions.md`.

---

## 📁 Estructura del repositorio

```
rag-estado-peru/
├─ packages/
│  └─ rag_core/        # Lógica central RAG (reutilizable)
├─ services/
│  └─ api/             # API FastAPI
├─ scripts/            # CLI de ingesta y evaluación
├─ data/
│  ├─ raw/             # Documentos originales
│  ├─ processed/       # Texto limpio / chunks
│  └─ samples/         # Muestras pequeñas versionadas
├─ docs/               # Arquitectura, fuentes, decisiones
├─ tests/              # Tests unitarios y smoke tests
└─ README.md
```

---

## ⚙️ Stack tecnológico

**Core**
- Python 3.11
- FastAPI
- Pydantic

**RAG**
- LangChain / LlamaIndex
- Chroma (local)
- Qdrant (opcional)

**Evaluación**
- RAGAS / métricas custom
- pytest

**Infra**
- Docker
- Docker Compose
- GitHub Actions (CI)

---

## ▶️ Ejecución local

```bash
docker compose up --build
```

Swagger:
- http://localhost:8000/docs

---

## 📜 Licencia
MIT
