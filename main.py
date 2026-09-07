"""
main.py

FastAPI app that serves customer segmentation results as JSON endpoints,
consumed by the dashboard in templates/dashboard.html (or any other
frontend — React, Streamlit, Power BI, etc.)

Run locally with:
    uvicorn main:app --reload

Then open:
    http://127.0.0.1:8000/          -> the dashboard
    http://127.0.0.1:8000/docs      -> interactive API docs
    http://127.0.0.1:8000/api/health -> pipeline status
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from clustering import cluster_summary, label_segments, prepare_features, run_kmeans, run_pca, find_best_k
from config import get_settings
from data_pipeline import build_full_dataset
from schemas import (
    ClusterSize,
    ClusterSummary,
    Customer,
    CustomerPage,
    ElbowPoint,
    HealthStatus,
    PCAPoint,
    RefreshResponse,
)

settings = get_settings()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("customer_segmentation")


# ---------------------------------------------------------------------------
# In-memory pipeline cache.
#
# The clustering pipeline (load Excel -> clean -> RFM -> scale -> KMeans)
# is not cheap, so it runs once at startup and again only when explicitly
# refreshed via POST /api/refresh — never on every request.
# ---------------------------------------------------------------------------
class PipelineCache:
    def __init__(self) -> None:
        self.segmented = None
        self.summary = None
        self.segment_names: dict[int, str] = {}
        self.k: Optional[int] = None
        self.error: Optional[str] = None

    def compute(self, k: int) -> None:
        logger.info("Running segmentation pipeline with k=%d", k)
        rfm_behavior = build_full_dataset(settings.data_path)
        scaled_df, _ = prepare_features(rfm_behavior)
        labels, _ = run_kmeans(scaled_df, k)

        segmented = rfm_behavior.copy()
        segmented["Cluster"] = labels

        self.segmented = segmented
        self.summary = cluster_summary(segmented)
        self.segment_names = label_segments(self.summary)
        self.k = k
        self.error = None
        logger.info("Pipeline ready: %d customers across %d clusters", len(segmented), k)

    def ensure_ready(self) -> None:
        if self.segmented is None:
            if self.error:
                raise HTTPException(status_code=503, detail=self.error)
            raise HTTPException(status_code=503, detail="Pipeline has not been computed yet.")


cache = PipelineCache()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: compute once so the first real request isn't the slow one.
    # A bad data file shouldn't crash the whole app — /api/health and the
    # dashboard's error banner surface the problem instead.
    try:
        cache.compute(settings.default_k)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: surface any failure via /api/health
        cache.error = str(exc)
        logger.error("Startup pipeline computation failed: %s", exc)
    yield
    # No teardown needed — nothing external is held open.


app = FastAPI(title=settings.api_title, version=settings.api_version, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chart.js is vendored locally under /static instead of loaded from a CDN.
# Corporate networks, ad-blockers, and antivirus tools frequently block
# cdnjs.cloudflare.com and similar third-party script hosts, which silently
# breaks any dashboard that depends on it — serving the file ourselves
# means the dashboard works with zero external dependencies at runtime.
static_dir = settings.base_dir / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    """Serve the dashboard itself — this is what you open in the browser."""
    dashboard_file = settings.templates_dir / "dashboard.html"
    if not dashboard_file.exists():
        return HTMLResponse(
            "<h1>dashboard.html not found</h1>"
            "<p>Make sure templates/dashboard.html exists next to main.py.</p>",
            status_code=500,
        )
    return HTMLResponse(dashboard_file.read_text(encoding="utf-8"))


@app.get("/api", response_class=HTMLResponse, include_in_schema=False)
def api_info():
    return "<p>Customer Segmentation API. Docs at <a href='/docs'>/docs</a>. Dashboard at <a href='/'>/</a>.</p>"


@app.get("/api/health", response_model=HealthStatus, tags=["meta"])
def health():
    """Reports whether the pipeline is loaded and usable — check this before debugging further."""
    if cache.segmented is None:
        return HealthStatus(status="error", detail=cache.error or "Pipeline not yet computed")
    return HealthStatus(
        status="ok",
        customers_loaded=len(cache.segmented),
        active_k=cache.k,
    )


@app.post("/api/refresh", response_model=RefreshResponse, tags=["meta"])
def refresh_data(k: int = Query(default=None, ge=2, le=20, description="Optionally re-cluster with a new K")):
    """Recompute the full pipeline from the source file. Use this after the underlying data changes."""
    target_k = k or cache.k or settings.default_k
    try:
        cache.compute(target_k)
    except Exception as exc:  # noqa: BLE001
        cache.error = str(exc)
        raise HTTPException(status_code=500, detail=f"Failed to refresh pipeline: {exc}") from exc
    return RefreshResponse(status="recomputed", k=target_k)


@app.get("/api/clusters/summary", response_model=list[ClusterSummary], tags=["clusters"])
def clusters_summary():
    """Average Recency/Frequency/Monetary/Return_Rate per cluster, plus size and segment name."""
    cache.ensure_ready()
    rows = cache.summary.reset_index().to_dict(orient="records")
    for row in rows:
        row["segment_name"] = cache.segment_names.get(int(row["Cluster"]), f"Segment {row['Cluster']}")
    return rows


@app.get("/api/clusters/sizes", response_model=list[ClusterSize], tags=["clusters"])
def clusters_sizes():
    """Number and percentage of customers per cluster — for a pie/doughnut chart."""
    cache.ensure_ready()
    counts = cache.segmented["Cluster"].value_counts().sort_index()
    percentages = (counts / counts.sum() * 100).round(2)

    return [
        ClusterSize(
            cluster=int(c),
            segment_name=cache.segment_names.get(int(c), f"Segment {c}"),
            count=int(counts[c]),
            percentage=float(percentages[c]),
        )
        for c in counts.index
    ]


@app.get("/api/clusters/pca", response_model=list[PCAPoint], tags=["clusters"])
def clusters_pca():
    """2D PCA coordinates + cluster label per customer — for a scatter plot."""
    cache.ensure_ready()
    scaled_df, _ = prepare_features(cache.segmented)
    coords, _ = run_pca(scaled_df, n_components=2)

    return [
        PCAPoint(customer_id=int(cid), pc1=float(x), pc2=float(y), cluster=int(cluster))
        for cid, (x, y), cluster in zip(
            cache.segmented["Customer ID"], coords, cache.segmented["Cluster"]
        )
    ]


@app.get("/api/clusters/elbow", response_model=list[ElbowPoint], tags=["clusters"])
def clusters_elbow(
    k_min: int = Query(default=2, ge=2, le=19),
    k_max: int = Query(default=10, ge=3, le=20),
):
    """
    Inertia + silhouette score across a range of K values, so you can
    justify *why* the chosen K is a reasonable one (the classic elbow
    method) instead of picking it arbitrarily.
    """
    cache.ensure_ready()
    if k_min >= k_max:
        raise HTTPException(status_code=400, detail="k_min must be less than k_max")

    scaled_df, _ = prepare_features(cache.segmented)
    result = find_best_k(scaled_df, k_range=range(k_min, k_max + 1))
    return [
        ElbowPoint(k=k, inertia=i, silhouette=s)
        for k, i, s in zip(result["k_range"], result["inertia"], result["silhouette"])
    ]


@app.get("/api/customers/{customer_id}", response_model=Customer, tags=["customers"])
def customer_detail(customer_id: int):
    """Look up a single customer's RFM + Return_Rate + assigned cluster."""
    cache.ensure_ready()
    row = cache.segmented[cache.segmented["Customer ID"] == customer_id]

    if row.empty:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

    record = row.iloc[0].to_dict()
    record["segment_name"] = cache.segment_names.get(int(record["Cluster"]))
    return record


@app.get("/api/customers", response_model=CustomerPage, tags=["customers"])
def list_customers(
    cluster: Optional[int] = Query(default=None, description="Filter to a single cluster id"),
    limit: int = Query(default=50, ge=1, le=200, description="Rows per page (max 200)"),
    offset: int = Query(default=0, ge=0, description="Rows to skip"),
):
    """List customers, optionally filtered by cluster — for a searchable, paginated table."""
    cache.ensure_ready()
    segmented = cache.segmented

    if cluster is not None:
        segmented = segmented[segmented["Cluster"] == cluster]

    total = len(segmented)
    page = segmented.iloc[offset : offset + limit].copy()
    page["segment_name"] = page["Cluster"].map(lambda c: cache.segment_names.get(int(c)))

    return CustomerPage(
        total=total,
        limit=limit,
        offset=offset,
        results=page.to_dict(orient="records"),
    )
