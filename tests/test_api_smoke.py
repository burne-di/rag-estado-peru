"""
Smoke tests para la API
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient


# Solo importar si las dependencias están disponibles
try:
    from services.api.main import app
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False


@pytest.mark.skipif(not API_AVAILABLE, reason="API dependencies not available")
class TestAPISmoke:
    """Smoke tests básicos para la API"""

    @pytest.fixture
    def client(self):
        """Cliente de test para FastAPI con lifespan"""
        with TestClient(app) as client:
            yield client

    def test_health_endpoint(self, client):
        """GET /health responde correctamente"""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_stats_endpoint(self, client):
        """GET /stats responde correctamente"""
        response = client.get("/stats")

        assert response.status_code == 200
        data = response.json()
        assert "total_chunks" in data
        assert "embedding_model" in data
        assert "llm_model" in data

    def test_query_without_documents(self, client):
        """POST /query sin documentos indexados"""
        response = client.post(
            "/query",
            json={"question": "¿Qué es el código tributario?"}
        )

        # Debería retornar error porque no hay documentos
        assert response.status_code in [200, 400]

    def test_query_validation(self, client):
        """POST /query valida input"""
        # Query muy corta
        response = client.post(
            "/query",
            json={"question": "ab"}  # min_length=3
        )

        assert response.status_code == 422  # Validation error

    def test_ingest_validation(self, client):
        """POST /ingest valida input"""
        # Sin directory ni file_path
        response = client.post(
            "/ingest",
            json={}
        )

        assert response.status_code == 400

    def test_openapi_docs(self, client):
        """Documentación OpenAPI disponible"""
        response = client.get("/openapi.json")

        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data
        assert "/query" in data["paths"]
        assert "/health" in data["paths"]
