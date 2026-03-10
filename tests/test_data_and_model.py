"""
tests/test_data_and_model.py
-----------------------------
Unit tests for data_processing.py and recommendation_model.py.

Fix #10: covers filter_by_platform, filter_by_price, engineer_features,
         recommend(), and the discount column name alignment.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def minimal_df() -> pd.DataFrame:
    """Tiny 6-row DataFrame with all columns the pipeline expects."""
    return pd.DataFrame({
        "app_id":         [1, 2, 3, 4, 5, 6],
        "title":          ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"],
        "price_final":    [0.0, 9.99, 19.99, 4.99, 0.0, 29.99],
        "price_original": [0.0, 14.99, 19.99, 9.99, 0.0, 29.99],
        "positive_ratio": [85.0, 72.0, 45.0, 91.0, 60.0, 30.0],
        "user_reviews":   [5000, 1200, 300, 800, 200, 50],
        "win":            [1, 1, 1, 0, 1, 1],
        "mac":            [0, 1, 0, 1, 0, 0],
        "linux":          [0, 0, 1, 1, 1, 0],
        "steam_deck":     [1, 0, 0, 0, 1, 0],
        "date_release":   pd.to_datetime(
            ["2020-01-01", "2021-06-15", "2019-03-10",
             "2022-11-01", "2023-07-04", "2018-08-20"]
        ),
        "tags":           ["RPG Action", "Strategy", "RPG", "Puzzle", "Action", "Horror"],
        "genres":         ["RPG", "Strategy", "RPG", "Puzzle", "Action", "Horror"],
        "log_price":      [0.0, 2.35, 3.04, 1.79, 0.0, 3.43],
        "popularity_score": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "trending_score": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "discount":       [0.0, 33.4, 0.0, 50.1, 0.0, 0.0],
        "text_features":  ["RPG Action RPG", "Strategy Strategy",
                           "RPG RPG", "Puzzle Puzzle",
                           "Action Action", "Horror Horror"],
    })


@pytest.fixture()
def engineered_df(minimal_df) -> pd.DataFrame:
    """Re-run engineer_features on minimal_df to get derived columns."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from utils.data_processing import engineer_features
    return engineer_features(minimal_df.drop(
        columns=["log_price", "popularity_score", "trending_score",
                 "discount", "text_features"],
        errors="ignore",
    ))


# ---------------------------------------------------------------------------
# data_processing — filter_by_platform
# ---------------------------------------------------------------------------

class TestFilterByPlatform:

    def _f(self, df, **kwargs):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from utils.data_processing import filter_by_platform
        return filter_by_platform(df, **kwargs)

    def test_windows_only(self, minimal_df):
        result = self._f(minimal_df, win=True, mac=False, linux=False, deck=False)
        assert all(result["win"] == 1)

    def test_mac_only(self, minimal_df):
        result = self._f(minimal_df, win=False, mac=True, linux=False, deck=False)
        assert all(result["mac"] == 1)

    def test_linux_only(self, minimal_df):
        result = self._f(minimal_df, win=False, mac=False, linux=True, deck=False)
        assert all(result["linux"] == 1)

    def test_steam_deck_only(self, minimal_df):
        result = self._f(minimal_df, win=False, mac=False, linux=False, deck=True)
        assert all(result["steam_deck"] == 1)

    def test_no_filters_returns_all(self, minimal_df):
        result = self._f(minimal_df, win=False, mac=False, linux=False, deck=False)
        assert len(result) == len(minimal_df)

    def test_win_and_mac(self, minimal_df):
        result = self._f(minimal_df, win=True, mac=True, linux=False, deck=False)
        mask = (minimal_df["win"] == 1) | (minimal_df["mac"] == 1)
        assert set(result.index) == set(minimal_df[mask].index)


# ---------------------------------------------------------------------------
# data_processing — filter_by_price
# ---------------------------------------------------------------------------

class TestFilterByPrice:

    def _f(self, df, lo, hi):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from utils.data_processing import filter_by_price
        return filter_by_price(df, lo, hi)

    def test_free_games_always_included(self, minimal_df):
        result = self._f(minimal_df, 5.0, 60.0)
        free = minimal_df[minimal_df["price_final"] == 0]
        for idx in free.index:
            assert idx in result.index

    def test_paid_games_in_range(self, minimal_df):
        result = self._f(minimal_df, 5.0, 15.0)
        paid_in = result[result["price_final"] > 0]
        assert all((paid_in["price_final"] >= 5.0) & (paid_in["price_final"] <= 15.0))

    def test_zero_range_only_free(self, minimal_df):
        result = self._f(minimal_df, 0.0, 0.0)
        assert all(result["price_final"] == 0)

    def test_full_range_returns_all(self, minimal_df):
        result = self._f(minimal_df, 0.0, 100.0)
        assert len(result) == len(minimal_df)


# ---------------------------------------------------------------------------
# data_processing — engineer_features (fix #2 regression test)
# ---------------------------------------------------------------------------

class TestEngineerFeatures:

    def test_discount_column_exists(self, engineered_df):
        """Fix #2: must be 'discount', never 'discount_percentage'."""
        assert "discount" in engineered_df.columns, (
            "engineer_features() must create a column named 'discount', "
            "matching FEATURE_COLS in recommendation_model.py"
        )

    def test_discount_no_discount_percentage(self, engineered_df):
        assert "discount_percentage" not in engineered_df.columns

    def test_discount_range(self, engineered_df):
        assert engineered_df["discount"].between(0, 100).all()

    def test_discount_free_game_is_zero(self, engineered_df):
        free = engineered_df[engineered_df["price_final"] == 0]
        assert (free["discount"] == 0).all()

    def test_discount_paid_game_with_original_higher(self, engineered_df):
        """Game 4 has price_original=9.99, price_final=4.99 → ~50% discount."""
        row = engineered_df[engineered_df["title"] == "Delta"].iloc[0]
        assert row["discount"] == pytest.approx(50.05, abs=0.5)

    def test_log_price_exists(self, engineered_df):
        assert "log_price" in engineered_df.columns

    def test_log_price_free_is_zero(self, engineered_df):
        free = engineered_df[engineered_df["price_final"] == 0]
        assert (free["log_price"] == 0).all()

    def test_popularity_score_non_negative(self, engineered_df):
        assert (engineered_df["popularity_score"] >= 0).all()

    def test_trending_score_non_negative(self, engineered_df):
        assert (engineered_df["trending_score"] >= 0).all()

    def test_text_features_column_exists(self, engineered_df):
        assert "text_features" in engineered_df.columns


# ---------------------------------------------------------------------------
# data_processing — load_and_clean_data (fix #3 regression test)
# ---------------------------------------------------------------------------

class TestLoadAndClean:

    def _load(self, path):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from utils.data_processing import load_and_clean_data
        return load_and_clean_data(path)

    def test_missing_linux_flag_does_not_drop_row(self, tmp_path):
        """Fix #3: a game missing its linux column must not be purged."""
        csv = tmp_path / "games.csv"
        csv.write_text(
            "app_id,title,price_final,price_original,positive_ratio,"
            "user_reviews,win,mac,linux,steam_deck\n"
            "42,TestGame,9.99,9.99,80,500,1,0,,0\n"   # linux is empty
        )
        df = self._load(str(csv))
        assert len(df) == 1, "Row should survive even though linux is missing"
        assert df.iloc[0]["linux"] == 0

    def test_completely_empty_title_is_dropped(self, tmp_path):
        csv = tmp_path / "games.csv"
        csv.write_text(
            "app_id,title,price_final,price_original,positive_ratio,user_reviews\n"
            "1,,9.99,9.99,80,500\n"    # title is empty → should be dropped
            "2,ValidGame,0,0,90,100\n"
        )
        df = self._load(str(csv))
        assert len(df) == 1
        assert df.iloc[0]["title"] == "ValidGame"


# ---------------------------------------------------------------------------
# recommendation_model — recommend()
# ---------------------------------------------------------------------------

class TestRecommender:

    @pytest.fixture()
    def fitted_rec(self, engineered_df):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from model.recommendation_model import GameRecommender
        rec = GameRecommender()
        rec.fit(engineered_df)
        return rec, engineered_df

    def test_recommend_returns_correct_count(self, fitted_rec):
        rec, _ = fitted_rec
        result = rec.recommend("Alpha", top_n=3)
        assert len(result) == 3

    def test_recommend_excludes_query_game(self, fitted_rec):
        rec, _ = fitted_rec
        result = rec.recommend("Alpha", top_n=5)
        assert "Alpha" not in result["title"].values

    def test_similarity_scores_in_range(self, fitted_rec):
        rec, _ = fitted_rec
        result = rec.recommend("Alpha", top_n=4)
        assert result["similarity_score"].between(0, 1).all()

    def test_unknown_title_raises(self, fitted_rec):
        rec, _ = fitted_rec
        with pytest.raises(ValueError, match="not found"):
            rec.recommend("NonExistentGame1234")

    def test_recommend_with_filter_df(self, fitted_rec):
        rec, df = fitted_rec
        # Only allow RPG games
        allowed = df[df["text_features"].str.contains("RPG", na=False)]
        result = rec.recommend("Alpha", top_n=2, filter_df=allowed)
        for _, row in result.iterrows():
            assert row["title"] in allowed["title"].values

    def test_get_all_titles(self, fitted_rec):
        rec, df = fitted_rec
        titles = rec.get_all_titles()
        assert set(titles) == set(df["title"].tolist())

    def test_text_similarity_rpg_ranks_higher_than_horror(self, fitted_rec):
        """
        Fix #1 regression: the other RPG game (Gamma) should have a higher
        similarity score than the Horror game (Zeta) when recommending from
        Alpha (also an RPG).  This validates TF-IDF has non-zero influence
        even on a tiny dataset.
        """
        rec, _ = fitted_rec
        result = rec.recommend("Alpha", top_n=5)
        scores = result.set_index("title")["similarity_score"]
        # Gamma (RPG) must outscore Zeta (Horror)
        assert "Gamma" in scores.index, "Gamma not in results"
        assert "Zeta" in scores.index, "Zeta not in results"
        assert scores["Gamma"] > scores["Zeta"], (
            f"TF-IDF should make Gamma (RPG) score higher than Zeta (Horror), "
            f"but got Gamma={scores['Gamma']:.3f} Zeta={scores['Zeta']:.3f}"
        )
