"""
model/recommendation_model.py
------------------------------
Content-based recommendation engine using cosine similarity.

The model is intentionally kept lightweight so it works fast even
with 50 000+ rows on a standard laptop.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# Features used to represent each game as a numeric vector
FEATURE_COLS = [
    "positive_ratio",
    "user_reviews",
    "price_final",
    "discount",
    "popularity_score",
    "win",
    "mac",
    "linux",
    "steam_deck",
]


class GameRecommender:
    """
    Content-based recommender that finds games most similar to a query game
    by computing cosine similarity over normalised numeric features.

    Usage
    -----
    >>> rec = GameRecommender()
    >>> rec.fit(df)                         # df must have engineer_features() applied
    >>> similar = rec.recommend("Portal 2")
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_matrix: np.ndarray | None = None  # scaled feature matrix
        self.df: pd.DataFrame | None = None             # reference to training data

    # ─────────────────────────────────────────────
    # FIT
    # ─────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "GameRecommender":
        """
        Build the scaled feature matrix from the supplied DataFrame.

        Parameters
        ----------
        df : DataFrame that already has engineer_features() columns added.

        Returns
        -------
        self  (so you can chain:  rec.fit(df).recommend(...))
        """
        self.df = df.reset_index(drop=True)

        # Fill any missing feature values with the column median
        features = self.df[FEATURE_COLS].copy()
        for col in FEATURE_COLS:
            features[col] = pd.to_numeric(features[col], errors="coerce")
            features[col] = features[col].fillna(features[col].median())

        # Scale to zero-mean / unit-variance so no single feature dominates
        self.feature_matrix = self.scaler.fit_transform(features.values)
        return self

    # ─────────────────────────────────────────────
    # RECOMMEND
    # ─────────────────────────────────────────────

    def recommend(self,
                  game_title: str,
                  top_n: int = 10,
                  filter_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Return the top-N games most similar to *game_title*.

        Parameters
        ----------
        game_title : Exact title as it appears in the dataset.
        top_n      : Number of recommendations to return.
        filter_df  : Optional pre-filtered DataFrame (e.g. after platform /
                     price filters).  Recommendations are restricted to rows
                     that appear in this subset.

        Returns
        -------
        DataFrame with columns from self.df plus a "similarity_score" column,
        sorted by similarity descending.
        """
        if self.df is None or self.feature_matrix is None:
            raise RuntimeError("Call .fit(df) before .recommend()")

        # ── Find the query game's row index ───────────────────────────
        matches = self.df[self.df["title"].str.lower() == game_title.lower()]
        if matches.empty:
            # Fuzzy fall-back: find first title that *contains* the query string
            matches = self.df[
                self.df["title"].str.lower().str.contains(game_title.lower(), na=False)
            ]
        if matches.empty:
            raise ValueError(f"Game '{game_title}' not found in dataset.")

        query_idx = matches.index[0]
        query_vector = self.feature_matrix[query_idx].reshape(1, -1)

        # ── Determine which indices are in the allowed pool ───────────
        if filter_df is not None and not filter_df.empty:
            allowed_indices = filter_df.index.tolist()
        else:
            allowed_indices = list(range(len(self.df)))

        # Always exclude the query game itself from results
        allowed_indices = [i for i in allowed_indices if i != query_idx]

        if not allowed_indices:
            return pd.DataFrame()

        # ── Compute cosine similarity only for allowed games ──────────
        candidate_matrix = self.feature_matrix[allowed_indices]
        scores = cosine_similarity(query_vector, candidate_matrix)[0]

        # ── Pick top-N ────────────────────────────────────────────────
        top_local = np.argsort(scores)[::-1][:top_n]
        top_global = [allowed_indices[i] for i in top_local]
        top_scores = scores[top_local]

        result = self.df.loc[top_global].copy()
        result["similarity_score"] = top_scores
        result.sort_values("similarity_score", ascending=False, inplace=True)
        result.reset_index(drop=True, inplace=True)
        return result

    # ─────────────────────────────────────────────
    # SEARCH HELPERS
    # ─────────────────────────────────────────────

    def search_titles(self, query: str, max_results: int = 20) -> list[str]:
        """
        Return a list of game titles whose names contain *query* (case-insensitive).
        Used to power the autocomplete dropdown in the Streamlit app.
        """
        if self.df is None:
            return []
        mask = self.df["title"].str.lower().str.contains(query.lower(), na=False)
        return self.df.loc[mask, "title"].head(max_results).tolist()

    def get_all_titles(self) -> list[str]:
        """Return every game title, sorted alphabetically."""
        if self.df is None:
            return []
        return sorted(self.df["title"].tolist())
