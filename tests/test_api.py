"""Tests for the FastAPI server endpoints."""

from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self) -> None:
        """GET /health should return HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self) -> None:
        """GET /health response should contain status: ok."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_returns_service_name(self) -> None:
        """GET /health response should identify the service."""
        response = client.get("/health")
        data = response.json()
        assert data["service"] == "agent-orchestration"


class TestRunEndpoint:
    """Tests for POST /run."""

    def test_run_returns_200(self) -> None:
        """POST /run with a valid topic should return HTTP 200."""
        response = client.post("/run", json={"topic": "unit testing"})
        assert response.status_code == 200

    def test_run_returns_expected_fields(self) -> None:
        """Response should contain all RunResponse fields."""
        response = client.post("/run", json={"topic": "unit testing"})
        data = response.json()
        expected_fields = {"topic", "research", "draft", "review_feedback", "iteration", "status", "elapsed_seconds"}
        assert expected_fields.issubset(set(data.keys()))

    def test_run_preserves_topic(self) -> None:
        """The returned topic should match the request."""
        response = client.post("/run", json={"topic": "machine learning"})
        data = response.json()
        assert data["topic"] == "machine learning"

    def test_run_completes_pipeline(self) -> None:
        """The pipeline should reach 'complete' status."""
        response = client.post("/run", json={"topic": "data engineering"})
        data = response.json()
        assert data["status"] == "complete"

    def test_run_has_elapsed_time(self) -> None:
        """elapsed_seconds should be a positive number."""
        response = client.post("/run", json={"topic": "testing"})
        data = response.json()
        assert data["elapsed_seconds"] >= 0

    def test_run_missing_topic_returns_422(self) -> None:
        """POST /run without a topic should return HTTP 422 (validation error)."""
        response = client.post("/run", json={})
        assert response.status_code == 422

    def test_run_empty_body_returns_422(self) -> None:
        """POST /run with no body should return HTTP 422."""
        response = client.post("/run")
        assert response.status_code == 422


class TestOpenAPISchema:
    """Tests for auto-generated API documentation."""

    def test_openapi_schema_available(self) -> None:
        """GET /openapi.json should return the schema."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "Agent Orchestration API"

    def test_docs_endpoint_available(self) -> None:
        """GET /docs should return the Swagger UI page."""
        response = client.get("/docs")
        assert response.status_code == 200
