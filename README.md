# Customer Segmentation API + Dashboard

RFM (Recency, Frequency, Monetary) + Return-Rate customer segmentation on
the [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
dataset, served through a FastAPI backend and visualized in a live,
zero-build-step HTML dashboard.

## What it does

1. **Loads and cleans** raw transaction data (`data_pipeline.py`).
2. **Engineers RFM + Return_Rate features** per customer, log-transforms
   and scales them, and clusters customers with K-Means
   (`clustering.py`).
3. **Labels each cluster** with a human-readable RFM segment name
   (Champions, At Risk, Hibernating, ...) using a rank-based heuristic —
   no hard-coded thresholds.
4. **Serves everything as JSON** through a documented, validated FastAPI
   app (`main.py`).
5. **Visualizes it** in `templates/dashboard.html`: KPIs, segment sizes,
   segment profiles, a PCA scatter plot, an elbow-method chart for
   choosing K, and a searchable/sortable customer table.

## Project structure

```
.
├── main.py              # FastAPI app: routes, caching, lifespan startup
├── config.py             # Environment-driven settings (Settings/get_settings)
├── schemas.py            # Pydantic request/response models
├── data_pipeline.py      # Excel -> clean -> RFM + Return_Rate
├── clustering.py         # Feature prep, KMeans, PCA, elbow method, segment naming
├── templates/
│   └── dashboard.html    # Frontend (vanilla JS + Chart.js, no build step)
├── tests/
│   └── test_api.py       # Pytest suite against a synthetic dataset
├── requirements.txt
├── .env.example
├── Dockerfile
└── .gitignore
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # optional — defaults work out of the box
```

Download the dataset and place it at the project root as
`online_retail_II.xlsx` (or point `DATA_PATH` in `.env` somewhere else).

## Run

```bash
uvicorn main:app --reload
```

- Dashboard: <http://127.0.0.1:8000/>
- Interactive API docs: <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/api/health>

## API reference

| Method | Path                     | Description                                          |
|--------|--------------------------|-------------------------------------------------------|
| GET    | `/`                      | Dashboard UI                                          |
| GET    | `/api/health`            | Pipeline status, customer count, active K             |
| POST   | `/api/refresh?k=`        | Recompute the pipeline, optionally with a new K       |
| GET    | `/api/clusters/summary`  | Avg RFM + Return_Rate per cluster + segment name      |
| GET    | `/api/clusters/sizes`    | Customer count/% per cluster                          |
| GET    | `/api/clusters/pca`      | 2D PCA coordinates per customer                       |
| GET    | `/api/clusters/elbow`    | Inertia + silhouette score across a K range           |
| GET    | `/api/customers/{id}`    | Single customer's RFM + segment                       |
| GET    | `/api/customers`         | Paginated, filterable customer table                  |

Full request/response schemas are in `/docs` (auto-generated from
`schemas.py`).

## Testing

```bash
pytest
```

The test suite runs against a small synthetic dataset (see
`tests/test_api.py`) so it doesn't need the real Excel file and finishes
in under a second — useful in CI.

## Troubleshooting: `pip install` tries to build scikit-learn/pandas from source

If `pip install -r requirements.txt` starts downloading a `.tar.gz` and
running `meson`/`ninja`/`cl.exe` (Windows) instead of just installing,
it means pip couldn't find a **pre-built wheel** for your Python version
and is falling back to compiling the package from source — which then
needs a full C/C++ toolchain and usually fails.

This almost always means your Python version is newer than what the
data-science packages currently ship pre-built wheels for (e.g. Python
3.13/3.14 the week they're released). Fix:

1. Check your Python version: `python --version`.
2. Install Python **3.11 or 3.12** from [python.org](https://www.python.org/downloads/)
   if you're on something newer — these have full wheel support for
   pandas, scikit-learn, and everything else here.
3. Recreate the virtual environment with that version and reinstall:
   ```bash
   py -3.12 -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Deployment notes

- The `Dockerfile` builds a self-contained image; `docker build -t
  customer-segmentation . && docker run -p 8000:8000 customer-segmentation`
  runs it locally exactly as it would run on a host like Render or
  Fly.io.
- Before deploying publicly, set `CORS_ORIGINS` in `.env` to the exact
  frontend origin(s) instead of `["*"]`.
- The pipeline currently reads from a local Excel file at startup; for a
  larger dataset, swap `data_pipeline.load_data()` for a database query
  without touching anything downstream.

## Possible next steps

- Persist computed segments to a small database instead of recomputing
  in memory on every `/api/refresh` call.
- Add authentication if this is ever exposed outside a trusted network.
- Swap the heuristic `label_segments()` for named RFM-score buckets
  (e.g. the classic 1–5 RFM scoring grid) if stakeholders want segment
  definitions that stay stable across refreshes.
