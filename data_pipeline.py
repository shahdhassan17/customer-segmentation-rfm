"""
data_pipeline.py

Handles everything from raw Excel data to a clean RFM (+ Return Rate)
table. This is a refactor of the original notebook cells that load,
clean, and aggregate the Online Retail II dataset into small, testable,
independently-reusable functions.

Each function does exactly one step, so:
- you can unit-test cleaning logic without touching Excel I/O,
- you can swap the data source (e.g. a database instead of Excel) by
  only changing load_data(), and
- errors point you to the exact stage that failed, instead of a stack
  trace buried inside one giant function.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "Invoice",
    "Customer ID",
    "InvoiceDate",
    "Quantity",
    "Price",
}


def load_data(path: str) -> pd.DataFrame:
    """
    Read every sheet from the Excel file and combine them into one
    DataFrame.

    Raises:
        FileNotFoundError: if `path` doesn't exist, with a message that
            tells you exactly what to fix (this is the #1 setup error
            people hit when they clone the repo without the dataset).
        ValueError: if the file is missing columns the rest of the
            pipeline depends on.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find the data file at '{path}'. "
            "Download 'Online Retail II' from the UCI ML Repository, "
            "place it in the project root, or point DATA_PATH (in your "
            ".env file) at wherever you saved it."
        )

    logger.info("Loading data from %s", path)
    sheets = pd.read_excel(file_path, sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"The data file is missing required column(s): {sorted(missing)}. "
            f"Expected at least: {sorted(REQUIRED_COLUMNS)}."
        )

    logger.info("Loaded %d raw rows across %d sheet(s)", len(df), len(sheets))
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows without a Customer ID — they can't be attributed to a customer."""
    before = len(df)
    df_clean = df.dropna(subset=["Customer ID"]).copy()
    df_clean["Customer ID"] = df_clean["Customer ID"].astype(int)
    logger.info(
        "Dropped %d row(s) without a Customer ID (%d remaining)",
        before - len(df_clean),
        len(df_clean),
    )
    return df_clean


def get_purchases_only(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only genuine purchase transactions:
    - Quantity > 0 (excludes cancellations / returns)
    - Price > 0 (excludes invalid or free entries)
    """
    purchases = df_clean[(df_clean["Quantity"] > 0) & (df_clean["Price"] > 0)].copy()
    purchases["TotalPrice"] = purchases["Quantity"] * purchases["Price"]
    logger.info("%d purchase-only rows after filtering", len(purchases))
    return purchases


def build_rfm(purchases: pd.DataFrame) -> pd.DataFrame:
    """Aggregate purchase data at the customer level into Recency/Frequency/Monetary."""
    if purchases.empty:
        raise ValueError("No purchase rows left to build RFM from — check the source data.")

    reference_date = purchases["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        purchases.groupby("Customer ID")
        .agg(
            Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
            Frequency=("Invoice", "nunique"),
            Monetary=("TotalPrice", "sum"),
        )
        .reset_index()
    )
    return rfm


def build_return_rate(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Compute each customer's Return_Rate:
        Return_Rate = (# cancelled invoices) / (total # invoices)

    Cancelled invoices are identified by the 'C' prefix that this
    dataset's invoice numbers use for credit notes.
    """
    returns = df_clean[df_clean["Invoice"].astype(str).str.startswith("C")].copy()

    return_counts = (
        returns.groupby("Customer ID").agg(Return_Transactions=("Invoice", "count")).reset_index()
    )
    transaction_counts = (
        df_clean.groupby("Customer ID").agg(Total_Transactions=("Invoice", "count")).reset_index()
    )

    return_behavior = transaction_counts.merge(return_counts, on="Customer ID", how="left")
    return_behavior["Return_Transactions"] = return_behavior["Return_Transactions"].fillna(0)
    return_behavior["Return_Rate"] = (
        return_behavior["Return_Transactions"] / return_behavior["Total_Transactions"]
    )

    return return_behavior[["Customer ID", "Return_Rate"]]


def build_full_dataset(path: str) -> pd.DataFrame:
    """
    Run the full pipeline end to end: load -> clean -> RFM -> merge
    Return_Rate. This is the single entry point the API (or any other
    caller) needs.
    """
    df = load_data(path)
    df_clean = clean_data(df)

    purchases = get_purchases_only(df_clean)
    rfm = build_rfm(purchases)
    return_behavior = build_return_rate(df_clean)

    rfm_behavior = rfm.merge(return_behavior, on="Customer ID", how="left")
    rfm_behavior["Return_Rate"] = rfm_behavior["Return_Rate"].fillna(0)

    logger.info("Built RFM + Return_Rate table for %d customers", len(rfm_behavior))
    return rfm_behavior


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data = build_full_dataset("online_retail_II.xlsx")
    print(data.shape)
    print(data.head())
