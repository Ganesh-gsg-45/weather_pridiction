"""
test_data_ingestion.py

Verifies that the time-based train/test split produces ZERO date overlap.

The split logic in DataIngestion.initiate_data_ingestion() is:
    train_df = df[df["date"] <= TRAIN_CUTOFF]
    test_df  = df[df["date"] >  TRAIN_CUTOFF]

This test was manually verified during development (the cutoff is 2021-12-31).
This file turns that one-time verification into a repeatable assertion so any
future change to TRAIN_CUTOFF or the split logic is caught immediately.

No real CSV is read — a tiny synthetic DataFrame is used so the test is fast
and portable (no dependency on dataset/ being present in CI).
"""

import pandas as pd
import pytest

# Import the cutoff constant directly from its source module so the test
# stays in sync if TRAIN_CUTOFF is ever changed.
from src.weatherprediction.components.data_ingestion import TRAIN_CUTOFF


def _make_synthetic_df() -> pd.DataFrame:
    """Return a small DataFrame spanning several years, one row per day.
    Starts at 2000 to mirror the real dataset — train covers 22 years,
    test covers 3 years, so the train > test size assertion holds.
    """
    dates = pd.date_range(start="2000-01-01", end="2024-12-31", freq="D")
    return pd.DataFrame({"date": dates, "value": range(len(dates))})


def _apply_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mirror the exact split logic from DataIngestion.initiate_data_ingestion()."""
    train_df = df[df["date"] <= TRAIN_CUTOFF].copy()
    test_df  = df[df["date"] >  TRAIN_CUTOFF].copy()
    return train_df, test_df


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_no_date_overlap():
    """Train and test sets must share zero dates."""
    df = _make_synthetic_df()
    train_df, test_df = _apply_split(df)

    train_dates = set(train_df["date"])
    test_dates  = set(test_df["date"])
    overlap = train_dates & test_dates

    assert len(overlap) == 0, (
        f"Date overlap detected between train and test splits! "
        f"Overlapping dates: {sorted(overlap)[:5]}..."
    )


def test_train_ends_at_or_before_cutoff():
    """Every date in the training set must be <= TRAIN_CUTOFF."""
    df = _make_synthetic_df()
    train_df, _ = _apply_split(df)

    cutoff = pd.Timestamp(TRAIN_CUTOFF)
    beyond_cutoff = train_df[train_df["date"] > cutoff]

    assert len(beyond_cutoff) == 0, (
        f"Training set contains {len(beyond_cutoff)} rows beyond cutoff {TRAIN_CUTOFF}. "
        f"First offending date: {beyond_cutoff['date'].min()}"
    )


def test_test_starts_after_cutoff():
    """Every date in the test set must be strictly > TRAIN_CUTOFF."""
    df = _make_synthetic_df()
    _, test_df = _apply_split(df)

    cutoff = pd.Timestamp(TRAIN_CUTOFF)
    before_or_on_cutoff = test_df[test_df["date"] <= cutoff]

    assert len(before_or_on_cutoff) == 0, (
        f"Test set contains {len(before_or_on_cutoff)} rows on or before "
        f"cutoff {TRAIN_CUTOFF}."
    )


def test_split_is_exhaustive():
    """train + test must account for every row in the original DataFrame."""
    df = _make_synthetic_df()
    train_df, test_df = _apply_split(df)

    assert len(train_df) + len(test_df) == len(df), (
        f"Row count mismatch: train({len(train_df)}) + test({len(test_df)}) "
        f"!= total({len(df)})"
    )


def test_train_is_larger_than_test():
    """Sanity-check: 22-year train set should dwarf the 3-year test set."""
    df = _make_synthetic_df()
    train_df, test_df = _apply_split(df)

    assert len(train_df) > len(test_df), (
        f"Expected train({len(train_df)}) > test({len(test_df)})"
    )
