"""
utils/data_processing.py
-------------------------
Loads, cleans, and feature-engineers the Steam games dataset.

Fixes applied
─────────────
  #2  discount column name: engineer_features() now creates "discount" (not
      "discount_percentage") so it matches FEATURE_COLS in recommendation_model.py.
  #3  Targeted NA filling instead of drop-all-NA: each column gets a sensible
      default (False for booleans, 0 for numerics, "Unknown" for strings).
  #6  genre/tag data is preserved and exposed so the sidebar filter works.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


# ── Column defaults for targeted NA filling (fix #3) ──────────────────────────
_BOOL_COLS   = ["win", "mac", "linux", "steam_deck"]
_NUMERIC_COLS = [
    "price_final", "price_original", "positive_ratio",
    "user_reviews", "rating",
]
_STR_COLS = ["title", "tags", "genres", "developers"]

# Minimum reviews a game needs to appear in trending / leaderboard
MIN_REVIEWS_TRENDING = 100


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """
    Loads the CSV and applies targeted, column-aware NA filling (fix #3).
    Rows are only dropped when core identity columns (app_id, title) are
    missing — never because a platform flag or price is absent.
    """
    df = pd.read_csv(csv_path, low_memory=False)

    # ── Drop only truly un-usable rows ────────────────────────────────────────
    df.dropna(subset=["app_id", "title"], inplace=True)
    df.drop_duplicates(subset=["app_id"], inplace=True)

    # ── Boolean platform columns: missing → False ──────────────────────────────
    for col in _BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool).astype(int)
        else:
            df[col] = 0

    # ── Numeric columns: missing → 0 ──────────────────────────────────────────
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0.0

    # ── String / tag columns: missing → empty string ──────────────────────────
    for col in _STR_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
        else:
            df[col] = ""

    # ── Release date ──────────────────────────────────────────────────────────
    if "date_release" in df.columns:
        df["date_release"] = pd.to_datetime(df["date_release"], errors="coerce")
    else:
        df["date_release"] = pd.NaT

    # ── Coerce app_id to int ───────────────────────────────────────────────────
    df["app_id"] = pd.to_numeric(df["app_id"], errors="coerce").fillna(0).astype(int)

    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives model features from the cleaned dataframe.

    Fix #2: the discount column is named "discount" (not "discount_percentage")
    so it aligns with FEATURE_COLS in recommendation_model.py.
    """
    df = df.copy()

    # ── Discount (fix #2: must match FEATURE_COLS key "discount") ─────────────
    if "price_original" in df.columns and "price_final" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["discount"] = np.where(
                df["price_original"] > 0,
                (df["price_original"] - df["price_final"]) / df["price_original"] * 100,
                0.0,
            )
    else:
        df["discount"] = 0.0

    df["discount"] = df["discount"].clip(0, 100).fillna(0)

    # ── Review-based scores ───────────────────────────────────────────────────
    df["positive_ratio"] = df["positive_ratio"].clip(0, 100)

    log_reviews = np.log1p(df["user_reviews"])
    df["popularity_score"] = (
        df["positive_ratio"] / 100.0 * log_reviews
    ).fillna(0)

    # ── Trending score (recency-weighted popularity) ───────────────────────────
    now = pd.Timestamp.now()
    days_old = (now - df["date_release"]).dt.days.fillna(3650).clip(lower=1)
    recency   = np.exp(-days_old / 730)          # half-life ≈ 2 years
    df["trending_score"] = df["popularity_score"] * (1 + recency)

    # ── Price log (robust to free games) ──────────────────────────────────────
    df["log_price"] = np.log1p(df["price_final"])

    # ── Combined text column for TF-IDF (used by recommendation model) ────────
    df["text_features"] = (
        df.get("tags",   pd.Series([""] * len(df))).astype(str)
        + " "
        + df.get("genres", pd.Series([""] * len(df))).astype(str)
    ).str.strip()

    return df


# ── Filtering helpers ─────────────────────────────────────────────────────────

def filter_by_platform(
    df: pd.DataFrame,
    win: bool, mac: bool, linux: bool, deck: bool,
) -> pd.DataFrame:
    """Keep games that support at least one of the chosen platforms."""
    if not any([win, mac, linux, deck]):
        return df
    mask = pd.Series(False, index=df.index)
    if win:   mask |= df["win"].astype(bool)
    if mac:   mask |= df["mac"].astype(bool)
    if linux: mask |= df["linux"].astype(bool)
    if deck:  mask |= df["steam_deck"].astype(bool)
    return df[mask]


def filter_by_price(
    df: pd.DataFrame,
    min_price: float, max_price: float,
) -> pd.DataFrame:
    """Keep free games OR games within the USD price range."""
    mask = (df["price_final"] == 0) | (
        (df["price_final"] >= min_price) & (df["price_final"] <= max_price)
    )
    return df[mask]


def filter_by_min_ratio(df: pd.DataFrame, min_ratio: int) -> pd.DataFrame:
    return df[df["positive_ratio"] >= min_ratio]


def filter_by_genre(df: pd.DataFrame, genres: list[str]) -> pd.DataFrame:
    """
    Keep games whose tags/genres column contains at least one of the
    selected genres (case-insensitive substring match).  New in fix #6.
    """
    if not genres:
        return df
    pattern = "|".join(re.escape(g) for g in genres)
    text_col = df.get("text_features", df.get("tags", df.get("genres", pd.Series([""] * len(df)))))
    return df[text_col.str.contains(pattern, case=False, na=False, regex=True)]


def get_trending_games(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    if df.empty or "trending_score" not in df.columns:
        return df.head(top_n)
    filtered = df[df["user_reviews"] >= MIN_REVIEWS_TRENDING]
    if filtered.empty:
        filtered = df
    return filtered.sort_values("trending_score", ascending=False).head(top_n)


def get_available_genres(df: pd.DataFrame) -> list[str]:
    """
    Extract a sorted deduplicated list of genre/tag values from the dataset
    for use in the sidebar filter widget (fix #6).
    """
    import re
    col = "text_features" if "text_features" in df.columns else (
          "genres"        if "genres"        in df.columns else None)
    if col is None:
        return []
    raw = df[col].dropna().astype(str)
    tokens: set[str] = set()
    for entry in raw:
        for token in re.split(r"[,;|]+", entry):
            t = token.strip()
            if t and len(t) > 1:
                tokens.add(t)
    return sorted(tokens, key=str.lower)


# ── re import needed inside filter_by_genre ───────────────────────────────────
import re  # noqa: E402 (placed after functions that reference it via string)
