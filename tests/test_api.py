"""
tests/test_api.py

Exercises the API end-to-end against a small, synthetic dataset instead
of the real Online Retail II file, so the test suite runs in under a
second and doesn't depend on a multi-megabyte Excel file being present.

Run with:
    pytest
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import main as main_module


def _fake_rfm_dataset(n_customers: int = 40) -> pd.DataFrame:
    """A small, deterministic RFM + Return_Rate table shaped like the real pipeline's output."""
    rng = np.random.default_rng(seed=42)
    return pd.DataFrame(
        {
            "Customer ID": np.arange(1, n_customers + 1),
            "Recency": rng.integers(1, 365, n_customers),
            "Frequency": rng.integers(1, 50, n_customers),
            "Monetary": rng.uniform(10, 5000, n_customers).round(2),
            "Return_Rate": rng.uniform(0, 0.5, n_customers).round(3),
        }
    )


@pytest.fixture
def client(monkeypatch):
    """
    A TestClient whose pipeline runs against the synthetic dataset above.
    Patching build_full_dataset (as imported inside main.py) means the
    rest of the real pipeline code — scaling, KMeans, PCA, labeling — all
    still runs for real; only the Excel read is swapped out.
    """
    monkeypatch.setattr(main_module, "build_full_dataset", lambda path: _fake_rfm_dataset())
    with TestClient(main_module.app) as test_client:
        yield test_client


def test_health_ok_after_startup(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["customers_loaded"] == 40


def test_clusters_summary_has_segment_names(client):
    resp = client.get("/api/clusters/summary")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == main_module.settings.default_k
    for row in rows:
        assert row["segment_name"]
        assert row["Customers"] > 0


def test_clusters_sizes_percentages_sum_to_100(client):
    resp = client.get("/api/clusters/sizes")
    assert resp.status_code == 200
    sizes = resp.json()
    assert round(sum(s["percentage"] for s in sizes)) == 100
    assert sum(s["count"] for s in sizes) == 40


def test_clusters_pca_returns_one_point_per_customer(client):
    resp = client.get("/api/clusters/pca")
    assert resp.status_code == 200
    assert len(resp.json()) == 40


def test_customer_detail_found(client):
    resp = client.get("/api/customers/1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["Customer ID"] == 1
    assert "Cluster" in body


def test_customer_detail_not_found(client):
    resp = client.get("/api/customers/9999")
    assert resp.status_code == 404


def test_list_customers_pagination(client):
    resp = client.get("/api/customers", params={"limit": 10, "offset": 0})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 40
    assert len(body["results"]) == 10


def test_list_customers_rejects_oversized_limit(client):
    resp = client.get("/api/customers", params={"limit": 1000})
    assert resp.status_code == 422  # caught by Query(..., le=200) validation


def test_elbow_endpoint_shape(client):
    resp = client.get("/api/clusters/elbow", params={"k_min": 2, "k_max": 5})
    assert resp.status_code == 200
    points = resp.json()
    assert len(points) == 4
    assert all("silhouette" in p for p in points)


def test_refresh_recomputes_with_new_k(client):
    resp = client.post("/api/refresh", params={"k": 3})
    assert resp.status_code == 200
    assert resp.json()["k"] == 3

    summary = client.get("/api/clusters/summary").json()
    assert len(summary) == 3
