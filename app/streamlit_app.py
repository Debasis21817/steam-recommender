"""
Steam AI Game Recommendation Engine
Fast Streamlit Version
"""

import os
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
import pandas as pd

from utils.data_processing import (
    load_and_clean_data,
    engineer_features,
    filter_by_platform,
    filter_by_price,
    filter_by_min_ratio,
    get_trending_games,
)

from model.recommendation_model import GameRecommender


# ─────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────

st.set_page_config(
    page_title="Steam AI Recommender",
    page_icon="🎮",
    layout="wide"
)


# ─────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────

@st.cache_data
def load_data():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, "games.csv")

    df = load_and_clean_data(data_path)
    df = engineer_features(df)

    return df


@st.cache_resource
def build_model(df):

    model = GameRecommender()
    model.fit(df)

    return model


df = load_data()
recommender = build_model(df)


# ─────────────────────────────────────
# STEAM API
# ─────────────────────────────────────

STEAM_API = "https://store.steampowered.com/api/appdetails"


@st.cache_data(ttl=3600)
def fetch_steam_price(app_id):

    try:

        url = f"{STEAM_API}?appids={app_id}&cc=in&l=en&filters=price_overview"

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())

        inner = data[str(app_id)]

        if not inner["success"]:
            return None

        info = inner["data"]

        if info.get("is_free"):
            return "Free"

        price = info.get("price_overview", {}).get("final_formatted")

        return price

    except Exception:
        return None


def fetch_prices_parallel(app_ids):

    prices = {}

    with ThreadPoolExecutor(max_workers=10) as executor:

        futures = {
            executor.submit(fetch_steam_price, i): i
            for i in app_ids
        }

        for future in futures:

            app = futures[future]

            try:
                prices[app] = future.result()
            except Exception:
                prices[app] = None

    return prices


# ─────────────────────────────────────
# UI HEADER
# ─────────────────────────────────────

st.title("🎮 Steam AI Game Recommender")
st.caption("Find similar games using machine learning")


# ─────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────

with st.sidebar:

    st.header("Filters")

    f_win = st.checkbox("Windows", True)
    f_mac = st.checkbox("Mac")
    f_linux = st.checkbox("Linux")
    f_deck = st.checkbox("Steam Deck")

    price_range = st.slider(
        "Max price (USD)",
        0,
        60,
        (0, 60)
    )

    min_ratio = st.slider(
        "Minimum positive %",
        0,
        100,
        0
    )

    top_n = st.slider(
        "Recommendations",
        5,
        20,
        10
    )


filtered_df = filter_by_platform(df, f_win, f_mac, f_linux, f_deck)
filtered_df = filter_by_price(filtered_df, price_range[0], price_range[1])
filtered_df = filter_by_min_ratio(filtered_df, min_ratio)


# ─────────────────────────────────────
# SEARCH
# ─────────────────────────────────────

st.header("Search Game")

titles = recommender.get_all_titles()

query = st.text_input("Type a game name")

matches = [t for t in titles if query.lower() in t.lower()][:200]

selected = None

if matches:

    selected = st.selectbox(
        "Select game",
        matches
    )


# ─────────────────────────────────────
# RECOMMENDATIONS
# ─────────────────────────────────────

if selected:

    if st.button("Recommend Games"):

        recs = recommender.recommend(
            selected,
            top_n=top_n,
            filter_df=filtered_df
        )

        st.subheader("Recommended Games")

        app_ids = recs["app_id"].tolist()

        price_map = fetch_prices_parallel(app_ids)

        for _, row in recs.iterrows():

            title = row["title"]
            app_id = row["app_id"]
            ratio = row["positive_ratio"]
            reviews = int(row["user_reviews"])

            price = price_map.get(app_id, "Unknown")

            with st.container():

                st.markdown(f"### {title}")

                st.write(
                    f"⭐ {ratio}% positive | "
                    f"{reviews:,} reviews | "
                    f"Price: {price}"
                )

                st.link_button(
                    "View on Steam",
                    f"https://store.steampowered.com/app/{app_id}"
                )

                st.divider()


# ─────────────────────────────────────
# TRENDING
# ─────────────────────────────────────

st.header("🔥 Trending Games")

trending = get_trending_games(filtered_df, top_n=10)

if not trending.empty:

    app_ids = trending["app_id"].tolist()

    price_map = fetch_prices_parallel(app_ids)

    for i, (_, row) in enumerate(trending.iterrows()):

        price = price_map.get(row["app_id"], "Unknown")

        st.write(
            f"{i+1}. {row['title']} — "
            f"{row['positive_ratio']}% positive — "
            f"Price: {price}"
        )


# ─────────────────────────────────────
# ANALYTICS
# ─────────────────────────────────────

st.header("📊 Dataset Stats")

col1, col2, col3 = st.columns(3)

col1.metric("Total Games", len(df))
col2.metric("Free Games", int((df["price_final"] == 0).sum()))
col3.metric("Average Rating", f"{df['positive_ratio'].mean():.1f}%")
