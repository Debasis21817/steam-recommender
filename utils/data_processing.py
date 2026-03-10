"""
utils/data_processing.py
------------------------
Handles all data loading, cleaning, and feature engineering for the
Steam AI Game Recommendation Engine.
"""

import pandas as pd
import numpy as np
from datetime import datetime


# ─────────────────────────────────────────────
# 1.  LOAD & CLEAN
# ─────────────────────────────────────────────

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load the games CSV, clean it, and return a tidy DataFrame.

    Steps
    -----
    - Drop rows with any missing values
    - Convert date_release to datetime
    - Cast boolean platform columns to int (0 / 1)
    - Remove exact duplicate titles (keep first occurrence)
    - Reset the index so it runs 0 … N-1
    """
    df = pd.read_csv(filepath)

    # ── Drop rows that are incomplete ──────────────────────────────────
    df.dropna(inplace=True)

    # ── Parse release date ─────────────────────────────────────────────
    df["date_release"] = pd.to_datetime(df["date_release"], errors="coerce")
    df.dropna(subset=["date_release"], inplace=True)          # remove unparseable dates

    # ── Boolean / string → integer for platform columns ───────────────
    bool_cols = ["win", "mac", "linux", "steam_deck"]
    for col in bool_cols:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
        else:
            # handle string "true"/"false" that some CSV readers leave as str
            df[col] = df[col].astype(str).str.lower().map({"true": 1, "false": 0})

    # ── Remove duplicate game titles ───────────────────────────────────
    df.drop_duplicates(subset="title", keep="first", inplace=True)

    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────
# 2.  FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived numeric features used by the recommendation model.

    New columns
    -----------
    popularity_score   : weighted blend of positive_ratio and log(user_reviews)
    discount_percentage: fraction of original price saved
    game_age           : years since release
    trending_score     : positive_ratio × log(user_reviews)
    """
    current_year = datetime.now().year

    # Guard against log(0) by clipping reviews to at least 1
    reviews_safe = df["user_reviews"].clip(lower=1)

    # ── popularity_score ──────────────────────────────────────────────
    df["popularity_score"] = (
        (df["positive_ratio"] / 100) * 0.7          # normalise ratio to [0,1]
        + np.log(reviews_safe) * 0.3
    )

    # ── discount_percentage ───────────────────────────────────────────
    df["discount_percentage"] = np.where(
        df["price_original"] > 0,
        (df["price_original"] - df["price_final"]) / df["price_original"],
        0.0,
    )

    # ── game_age ──────────────────────────────────────────────────────
    df["game_age"] = current_year - df["date_release"].dt.year

    # ── trending_score ────────────────────────────────────────────────
    df["trending_score"] = (df["positive_ratio"] / 100) * np.log(reviews_safe)

    return df


# ─────────────────────────────────────────────
# 3.  FILTER HELPERS
# ─────────────────────────────────────────────

def filter_by_platform(df: pd.DataFrame,
                        windows: bool = True,
                        mac: bool = False,
                        linux: bool = False,
                        steam_deck: bool = False) -> pd.DataFrame:
    """
    Return rows that support *at least one* of the selected platforms.
    If no platform is selected the original DataFrame is returned unchanged.
    """
    conditions = []
    if windows:
        conditions.append(df["win"] == 1)
    if mac:
        conditions.append(df["mac"] == 1)
    if linux:
        conditions.append(df["linux"] == 1)
    if steam_deck:
        conditions.append(df["steam_deck"] == 1)

    if not conditions:
        return df

    combined = conditions[0]
    for cond in conditions[1:]:
        combined = combined | cond

    return df[combined]


def filter_by_price(df: pd.DataFrame,
                    min_price: float = 0.0,
                    max_price: float = 60.0) -> pd.DataFrame:
    """Keep only games whose final price falls within [min_price, max_price]."""
    return df[(df["price_final"] >= min_price) & (df["price_final"] <= max_price)]


def filter_by_min_ratio(df: pd.DataFrame, min_ratio: float = 0.0) -> pd.DataFrame:
    """Keep only games with positive_ratio >= min_ratio (0–100 scale)."""
    return df[df["positive_ratio"] >= min_ratio]


# ─────────────────────────────────────────────
# 4.  TRENDING GAMES
# ─────────────────────────────────────────────

def get_trending_games(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Return the top-N games ranked by trending_score.
    Assumes engineer_features() has already been called.
    """
    return (
        df.sort_values("trending_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
