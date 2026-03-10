
"""
streamlit_app.py

Improved Steam Game Recommender UI

Improvements:
- Batch Steam API requests (ThreadPoolExecutor)
- Genre sidebar filter
- Plotly interactive charts
- Search-first UX
"""

import streamlit as st
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor

import plotly.express as px

from recommendation_model import GameRecommender

STEAM_API = "https://store.steampowered.com/api/appdetails"

USD_TO_INR = 84.0


@st.cache_data
def load_data():

    df = pd.read_csv("games.csv")

    df.fillna({
        "win": 0,
        "mac": 0,
        "linux": 0,
        "steam_deck": 0
    }, inplace=True)

    return df


def fetch_steam_details(app_id):

    try:

        r = requests.get(STEAM_API, params={"appids": app_id}, timeout=5)

        data = r.json()

        if not data[str(app_id)]["success"]:
            return None

        return data[str(app_id)]["data"]

    except Exception:
        return None


def fetch_prices_batch(app_ids):

    results = {}

    with ThreadPoolExecutor(max_workers=10) as executor:

        futures = {executor.submit(fetch_steam_details, i): i for i in app_ids}

        for future in futures:

            app_id = futures[future]

            data = future.result()

            if data and "price_overview" in data:

                price = data["price_overview"]["final"] / 100

                results[app_id] = price

            else:

                results[app_id] = None

    return results


st.title("🎮 Steam Game Recommender")

df = load_data()

recommender = GameRecommender().fit(df)


st.sidebar.header("Filters")

max_price = st.sidebar.slider(
    "Max Price (USD)",
    0,
    100,
    50
)

if "genres" in df.columns:

    genres = sorted(
        set("|".join(df["genres"].dropna()).split("|"))
    )

    selected_genre = st.sidebar.selectbox(
        "Genre",
        ["All"] + genres
    )

else:

    selected_genre = "All"


search_query = st.text_input("Search for a game")


if search_query:

    matches = df[
        df["title"].str.contains(search_query, case=False, na=False)
    ]

    if len(matches):

        game = st.selectbox("Select Game", matches["title"].tolist())

    else:

        st.warning("No matches found")
        st.stop()

else:

    st.info("Type a game name to search")
    st.stop()


results = recommender.recommend(game, top_n=10)


if selected_genre != "All":

    results = results[
        results["genres"].str.contains(selected_genre, na=False)
    ]


results = results[results["price_final"] <= max_price]


st.subheader("Recommended Games")

app_ids = results["app_id"].tolist() if "app_id" in results.columns else []

price_map = fetch_prices_batch(app_ids) if app_ids else {}


for _, row in results.iterrows():

    st.markdown(f"### {row['title']}")

    if "app_id" in row:

        price = price_map.get(row["app_id"])

        if price:
            st.write(f"Price: ${price} (~₹{round(price*USD_TO_INR)})")

    st.write(f"Similarity Score: {round(row['similarity_score'],3)}")

    st.divider()


st.header("Analytics")

fig = px.histogram(
    df,
    x="price_final",
    nbins=30,
    title="Game Price Distribution"
)

st.plotly_chart(fig, use_container_width=True)
