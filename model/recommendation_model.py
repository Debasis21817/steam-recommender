"""
model/recommendation_model.py
------------------------------
Improved content-based recommendation engine using:

1. Numeric feature similarity
2. Text similarity via TF-IDF (genres / tags / description)

Both representations are concatenated and cosine similarity
is computed on the combined vector space.
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# Corrected feature list
FEATURE_COLS = [
    "positive_ratio",
    "user_reviews",
    "price_final",
    "discount_percentage",   # FIXED (was "discount")
    "popularity_score",
    "win",
    "mac",
    "linux",
    "steam_deck",
]


TEXT_COLUMNS = ["genres", "tags", "description"]


class GameRecommender:
    """
    Content-based recommender using both numeric and textual similarity.
    """

    def __init__(self):

        self.scaler = StandardScaler()

        self.tfidf = TfidfVectorizer(
            stop_words="english",
            max_features=5000
        )

        self.feature_matrix = None
        self.df = None

    # ─────────────────────────────────────────────
    # TEXT PREPARATION
    # ─────────────────────────────────────────────

    def _build_text_corpus(self, df: pd.DataFrame):

        available_cols = [c for c in TEXT_COLUMNS if c in df.columns]

        if not available_cols:
            return None

        text_series = df[available_cols].fillna("").astype(str)

        corpus = text_series.apply(lambda row: " ".join(row.values), axis=1)

        return corpus

    # ─────────────────────────────────────────────
    # FIT
    # ─────────────────────────────────────────────

    def fit(self, df: pd.DataFrame):

        self.df = df.reset_index(drop=True)

        # -------- Numeric features --------
        features = self.df[FEATURE_COLS].copy()

        for col in FEATURE_COLS:
            features[col] = pd.to_numeric(features[col], errors="coerce")
            features[col] = features[col].fillna(features[col].median())

        numeric_matrix = self.scaler.fit_transform(features.values)

        # -------- Text features --------
        corpus = self._build_text_corpus(self.df)

        if corpus is not None:

            tfidf_matrix = self.tfidf.fit_transform(corpus)

            numeric_sparse = np.asarray(numeric_matrix)

            self.feature_matrix = np.hstack(
                [numeric_sparse, tfidf_matrix.toarray()]
            )

        else:

            self.feature_matrix = numeric_matrix

        return self

    # ─────────────────────────────────────────────
    # RECOMMEND
    # ─────────────────────────────────────────────

    def recommend(
        self,
        game_title: str,
        top_n: int = 10,
        filter_df: pd.DataFrame | None = None
    ) -> pd.DataFrame:

        if self.df is None or self.feature_matrix is None:
            raise RuntimeError("Call .fit(df) before .recommend()")

        matches = self.df[self.df["title"].str.lower() == game_title.lower()]

        if matches.empty:
            matches = self.df[
                self.df["title"].str.lower().str.contains(
                    game_title.lower(),
                    na=False
                )
            ]

        if matches.empty:
            raise ValueError(f"Game '{game_title}' not found.")

        query_idx = matches.index[0]

        query_vector = self.feature_matrix[query_idx].reshape(1, -1)

        similarities = cosine_similarity(query_vector, self.feature_matrix)[0]

        results = self.df.copy()
        results["similarity_score"] = similarities

        results = results.drop(index=query_idx)

        if filter_df is not None and not filter_df.empty:
            results = results.loc[filter_df.index]

        results = results.sort_values(
            "similarity_score",
            ascending=False
        )

        return results.head(top_n)
