FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero para cache de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY packages/ ./packages/
COPY services/ ./services/
COPY scripts/ ./scripts/

# Crear directorios necesarios
RUN mkdir -p data/raw data/processed data/chroma data/samples

# Copiar samples si existen (opcional para demos)
COPY data/samples/. ./data/samples/

# Variables de entorno por defecto
ENV PYTHONPATH=/app
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Exponer puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Comando por defecto
CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
