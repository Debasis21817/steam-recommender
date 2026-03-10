"""
model/recommendation_model.py
-------------------------------
Cosine-similarity recommender with hybrid numeric + TF-IDF text features.

Fixes applied
─────────────
  #1  Text-based similarity: genres/tags are vectorised with TF-IDF and
      blended (40 %) with the numeric feature similarity (60 %).  RPGs now
      recommend other RPGs, not just games with a similar price.
  #2  FEATURE_COLS now contains "discount" (matching engineer_features output).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Numeric features (fix #2: "discount" not "discount_percentage") ───────────
FEATURE_COLS = [
    "price_final",
    "positive_ratio",
    "user_reviews",
    "win",
    "mac",
    "linux",
    "steam_deck",
    "discount",          # fix #2
    "log_price",
]

# Weight of text (TF-IDF) vs numeric similarity
TEXT_WEIGHT    = 0.40
NUMERIC_WEIGHT = 0.60


class GameRecommender:
    """
    Hybrid cosine-similarity recommender.

    fit(df)          → scales numeric features + fits TF-IDF on text_features
    recommend(title) → blended numeric+text cosine similarity
    """

    def __init__(self) -> None:
        self._df:           pd.DataFrame | None = None
        self._titles:       list[str]            = []
        self._title_idx:    dict[str, int]       = {}
        self._num_matrix:   np.ndarray | None    = None
        self._text_matrix:  np.ndarray | None    = None
        self._scaler       = MinMaxScaler()
        self._tfidf        = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )
        self._has_text = False

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "GameRecommender":
        self._df     = df.reset_index(drop=True)
        self._titles = self._df["title"].tolist()
        self._title_idx = {t: i for i, t in enumerate(self._titles)}

        # ── Numeric matrix ────────────────────────────────────────────────────
        available = [c for c in FEATURE_COLS if c in self._df.columns]
        num_data  = self._df[available].fillna(0).values.astype(float)
        self._num_matrix = self._scaler.fit_transform(num_data)

        # ── TF-IDF text matrix (fix #1) ───────────────────────────────────────
        if "text_features" in self._df.columns:
            texts = self._df["text_features"].fillna("").tolist()
            non_empty = sum(1 for t in texts if t.strip())
            if non_empty > 10:
                try:
                    self._text_matrix = self._tfidf.fit_transform(texts).toarray()
                    self._has_text    = True
                except Exception:
                    self._has_text = False

        return self

    # ── Recommend ─────────────────────────────────────────────────────────────

    def recommend(
        self,
        title:     str,
        top_n:     int = 10,
        filter_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError("Call fit() before recommend().")
        if title not in self._title_idx:
            raise ValueError(f'"{title}" not found in the dataset.')

        idx = self._title_idx[title]

        # ── Numeric similarity ─────────────────────────────────────────────────
        num_sim = cosine_similarity(
            self._num_matrix[idx : idx + 1], self._num_matrix
        )[0]

        # ── Text similarity (fix #1) ───────────────────────────────────────────
        if self._has_text and self._text_matrix is not None:
            txt_sim  = cosine_similarity(
                self._text_matrix[idx : idx + 1], self._text_matrix
            )[0]
            sim_scores = NUMERIC_WEIGHT * num_sim + TEXT_WEIGHT * txt_sim
        else:
            sim_scores = num_sim

        # ── Candidate mask ────────────────────────────────────────────────────
        candidate_mask = np.ones(len(self._df), dtype=bool)
        candidate_mask[idx] = False        # exclude the query game itself

        if filter_df is not None and len(filter_df) < len(self._df):
            allowed = set(filter_df.index.tolist())
            for i in range(len(self._df)):
                if self._df.index[i] not in allowed:
                    candidate_mask[i] = False

        # ── Top-N ─────────────────────────────────────────────────────────────
        masked_scores = np.where(candidate_mask, sim_scores, -1.0)
        top_indices   = np.argpartition(masked_scores, -top_n)[-top_n:]
        top_indices   = top_indices[np.argsort(masked_scores[top_indices])[::-1]]

        result = self._df.iloc[top_indices].copy()
        result["similarity_score"] = sim_scores[top_indices]
        return result.reset_index(drop=True)

    # ── Utility ───────────────────────────────────────────────────────────────

    def get_all_titles(self) -> list[str]:
        return self._titles
