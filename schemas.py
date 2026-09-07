"""
schemas.py

Pydantic models describing the exact shape of every request and response
in the API. Declaring these — instead of returning raw dicts, as the
original version did — buys three things for free:

1. Automatic validation: FastAPI rejects malformed input before your
   code ever sees it.
2. Automatic serialization: your endpoint can return a dict or a
   dataframe row and FastAPI coerces it to the declared shape, dropping
   anything that isn't defined here (no accidental data leaks).
3. Accurate, interactive docs at /docs — anyone consuming this API can
   see the exact fields and types without reading your source code.
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ClusterSummary(BaseModel):
    """One row of the per-cluster RFM profile, used for the bar chart and KPI cards."""

    cluster: int = Field(..., alias="Cluster")
    segment_name: str = Field(..., description="Human-readable label, e.g. 'Champions'")
    customers: int = Field(..., alias="Customers")
    avg_recency: float = Field(..., alias="Avg_Recency")
    avg_frequency: float = Field(..., alias="Avg_Frequency")
    avg_monetary: float = Field(..., alias="Avg_Monetary")
    avg_return_rate: float = Field(..., alias="Avg_Return_Rate")

    model_config = ConfigDict(populate_by_name=True)


class ClusterSize(BaseModel):
    """One row of the segment-size breakdown, used for the pie/doughnut chart."""

    cluster: int
    segment_name: str
    count: int
    percentage: float


class PCAPoint(BaseModel):
    """A single customer projected onto 2 principal components, for the scatter plot."""

    customer_id: int
    pc1: float
    pc2: float
    cluster: int


class ElbowPoint(BaseModel):
    """Inertia + silhouette score for one candidate value of K."""

    k: int
    inertia: float
    silhouette: float


class Customer(BaseModel):
    """A single customer's RFM + Return_Rate + assigned cluster."""

    customer_id: int = Field(..., alias="Customer ID")
    recency: int = Field(..., alias="Recency")
    frequency: int = Field(..., alias="Frequency")
    monetary: float = Field(..., alias="Monetary")
    return_rate: float = Field(..., alias="Return_Rate")
    cluster: int = Field(..., alias="Cluster")
    segment_name: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)


class CustomerPage(BaseModel):
    """A page of the customer table, with pagination metadata."""

    total: int
    limit: int
    offset: int
    results: List[Customer]


class HealthStatus(BaseModel):
    """Reported by /api/health so a monitoring tool (or the dashboard) can tell if the
    pipeline is actually usable, not just whether the process is alive."""

    status: str = Field(..., description="'ok' or 'error'")
    customers_loaded: Optional[int] = None
    active_k: Optional[int] = None
    detail: Optional[str] = None


class RefreshResponse(BaseModel):
    status: str
    k: int
