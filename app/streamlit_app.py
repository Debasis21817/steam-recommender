"""
app/streamlit_app.py
--------------------
Steam AI Game Recommendation Engine  —  v3 (all 12 fixes applied)

Fixes in this version
─────────────────────
  #1  Hybrid TF-IDF + numeric similarity (see recommendation_model.py)
  #2  "discount" column name corrected (see data_processing.py)
  #3  Targeted NA filling, no aggressive dropna (see data_processing.py)
  #4  Batch HTTP prefetch with ThreadPoolExecutor before render
  #5  fetch_steam_details / fetch_live_price consolidated → fetch_steam_data()
  #6  Genre/tag sidebar filter added
  #7  Matplotlib replaced with Plotly interactive charts
  #8  Empty search → show popular defaults, not 200 alphabetical titles
  #9  Exchange rate fetched from exchangerate-api.com at startup (1-hr TTL)
  #10 pytest tests added in tests/ directory
  #11 Steam rate-limit / 429 / timeout errors distinguished and logged
  #12 .gitignore with __pycache__ / *.pyc entries

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import os
import re
import html as htmllib
import json
import logging
import urllib.request
import urllib.error
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Path fix ─────────────────────────────────────────────────────────────────
APP_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, APP_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px          # fix #7
import plotly.graph_objects as go    # fix #7

from utils.data_processing import (
    load_and_clean_data,
    engineer_features,
    filter_by_platform,
    filter_by_price,
    filter_by_min_ratio,
    filter_by_genre,          # fix #6
    get_trending_games,
    get_available_genres,     # fix #6
)
from model.recommendation_model import GameRecommender

# ── Logging setup (fix #11) ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("steam_recommender")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Steam AI Recommender",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
.stApp { background-color: #0e1117; color: #c7d5e0; }
section[data-testid="stSidebar"] { background-color: #1b2838; }
section[data-testid="stSidebar"] * { color: #c7d5e0 !important; }

.main-title {
    font-size: 2.6rem; font-weight: 800;
    background: linear-gradient(90deg, #1b9aaa, #66c5db);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.subtitle { color: #7ba3b8; font-size: 1rem; margin-bottom: 1.5rem; }
.sec-hdr {
    font-size: 1.35rem; font-weight: 700; color: #66c5db;
    border-left: 4px solid #1b9aaa; padding-left: 10px;
    margin: 1.5rem 0 0.8rem;
}

/* ── Metric strip ── */
.mrow { display: flex; gap: 12px; margin-bottom: 18px; flex-wrap: wrap; }
.mcrd {
    flex: 1; min-width: 120px;
    background: #1e2d3d; border: 1px solid #2a4158;
    border-radius: 10px; padding: 14px 18px; text-align: center;
}
.mval { font-size: 1.6rem; font-weight: 800; color: #66c5db; }
.mlbl { font-size: 0.78rem; color: #7ba3b8; margin-top: 2px; }

/* ── Clickable game card wrapper ── */
a.crd-link {
    display: block; text-decoration: none;
    color: inherit; margin-bottom: 14px;
}
a.crd-link:hover .gcrd  { border-color: #66c5db; }
a.crd-link:hover .gcrd img.ccover { opacity: 0.88; }

/* ── Compact game card ── */
.gcrd {
    background: linear-gradient(135deg, #1e2d3d 0%, #162232 100%);
    border: 1px solid #2a4158; border-radius: 12px;
    overflow: hidden; transition: border-color 0.22s, box-shadow 0.22s;
}
.gcrd:hover {
    border-color: #66c5db;
    box-shadow: 0 0 12px rgba(102,197,219,0.18);
}
.gcrd img.ccover {
    width: 100%; height: 140px; object-fit: cover;
    display: block; border-bottom: 1px solid #2a4158;
    transition: opacity 0.22s;
}
.gcrd .cbody { padding: 12px 14px 14px; }
.gtitle { font-size: 0.98rem; font-weight: 700; color: #e8f4f8; margin-bottom: 5px; }
.gmeta  { font-size: 0.8rem; color: #7ba3b8; margin: 4px 0 7px; }

/* ── Hero card ── */
.hero {
    background: linear-gradient(135deg, #1e2d3d 0%, #162232 100%);
    border: 1px solid #2a6080; border-radius: 14px;
    overflow: hidden; margin-bottom: 20px;
}
.hero img.hcover {
    width: 100%; max-height: 300px; object-fit: cover;
    display: block; border-bottom: 1px solid #2a4158;
}
.hero .hbody { padding: 20px 24px 22px; }
.htitle { font-size: 1.55rem; font-weight: 800; color: #e8f4f8; margin-bottom: 8px; }
.hrlbl {
    display: inline-block; padding: 4px 14px;
    border-radius: 20px; font-size: 0.85rem; font-weight: 700;
    margin-bottom: 10px;
}
.hdesc {
    color: #a8c0d0; font-size: 0.88rem; line-height: 1.6;
    margin: 10px 0 14px; max-height: 110px; overflow-y: auto;
}
.hmeta { color: #7ba3b8; font-size: 0.84rem; line-height: 1.8; }

/* ── Platform badges ── */
.bdg {
    display: inline-block; padding: 2px 9px; border-radius: 20px;
    font-size: 0.74rem; font-weight: 600; margin-right: 4px; margin-top: 3px;
}
.bwin  { background: #1a5276; color: #85c1e9; }
.bmac  { background: #1e3a2e; color: #76d7a5; }
.blin  { background: #4a235a; color: #c39bd3; }
.bdck  { background: #4a3500; color: #f0c040; }

/* ── Sim / rank badges ── */
.simbdg {
    float: right;
    background: linear-gradient(90deg, #1b9aaa, #0d7377);
    color: #fff; border-radius: 20px; padding: 3px 11px;
    font-size: 0.76rem; font-weight: 700;
}
.rnkbdg {
    display: inline-flex; align-items: center; justify-content: center;
    width: 26px; height: 26px;
    background: #1b9aaa; color: #fff; border-radius: 50%;
    font-size: 0.78rem; font-weight: 800; margin-right: 7px; flex-shrink: 0;
}

/* ── Rating colours ── */
.rpos { color: #66c2a5; font-weight: 700; }
.rmix { color: #fee08b; font-weight: 700; }
.rneg { color: #f46d43; font-weight: 700; }

/* ── Steam button ── */
.steam-btn {
    display: inline-block; margin-top: 10px;
    background: #1b9aaa; color: #fff;
    padding: 6px 16px; border-radius: 8px;
    text-decoration: none; font-size: 0.82rem; font-weight: 600;
}

/* ── Share callout ── */
.share-box {
    background: #1e2d3d; border: 1px dashed #2a6080; border-radius: 10px;
    padding: 12px 16px; margin-bottom: 16px;
    font-size: 0.85rem; color: #7ba3b8;
}
.share-box code { color: #66c5db; }
.divhr { height: 1px; background: #2a4158; margin: 1.2rem 0; }

/* ── Exchange-rate note ── */
.xrate-note {
    font-size: 0.75rem; color: #7ba3b8;
    background: #1e2d3d; border-radius: 6px;
    padding: 4px 10px; display: inline-block; margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FIX #9 — LIVE EXCHANGE RATE  (1-hr TTL)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_exchange_rate() -> tuple[float, str]:
    """
    Fetches USD→INR from exchangerate-api.com (free, no key needed for
    the open endpoint).  Returns (rate, source_label).
    """
    try:
        url = "https://open.er-api.com/v6/latest/USD"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode())
        rate = float(data["rates"]["INR"])
        return rate, "live"
    except Exception as exc:
        log.warning("Exchange-rate fetch failed (%s); using fallback 84.0", exc)
        return 84.0, "est."


USD_TO_INR, _xrate_src = fetch_exchange_rate()


# ═══════════════════════════════════════════════════════════════════════════════
# DATA + MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(
    ttl=86400,
    show_spinner="Loading and processing 50 000+ games…",
)
def load_data() -> pd.DataFrame:
    csv_path = os.path.join(APP_DIR, "games.csv")
    df = load_and_clean_data(csv_path)
    df = engineer_features(df)
    return df


@st.cache_resource(show_spinner="Building recommendation model…")
def build_model(df: pd.DataFrame) -> GameRecommender:
    rec = GameRecommender()
    rec.fit(df)
    return rec


df          = load_data()
recommender = build_model(df)


# ═══════════════════════════════════════════════════════════════════════════════
# FIX #5 — CONSOLIDATED STEAM API HELPER
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_steam_data(app_id: int, full: bool = True) -> dict:
    """
    Fix #5: Single function replacing both fetch_steam_details and
    fetch_live_price.  Use full=True for the hero card (description,
    genres, developers + price); full=False for compact price-only cards.

    Fix #11: Distinguishes timeout vs 429 rate-limit vs bad app_id errors,
    logging each separately instead of silently swallowing them.
    """
    result: dict = {
        "price_final_str": "",
        "price_orig_str":  "",
        "discount_pct":    0,
        "is_free":         False,
        "description":     "",
        "genres":          "",
        "developers":      "",
    }

    filters = "" if full else "&filters=price_overview,basic"
    url     = (
        f"https://store.steampowered.com/api/appdetails"
        f"?appids={app_id}&cc=in&l=en{filters}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urllib.request.urlopen(req, timeout=8 if full else 6) as resp:
            status = resp.getcode()
            if status == 429:
                log.warning("Steam rate-limit (429) for app_id=%s", app_id)
                return result
            data = json.loads(resp.read().decode())

    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            log.warning("Steam rate-limit (429) for app_id=%s", app_id)
        else:
            log.error("Steam HTTP error %s for app_id=%s", exc.code, app_id)
        return result

    except urllib.error.URLError as exc:
        if "timed out" in str(exc.reason).lower():
            log.warning("Steam timeout for app_id=%s", app_id)
        else:
            log.error("Steam URLError for app_id=%s: %s", app_id, exc.reason)
        return result

    except Exception as exc:          # JSON decode errors etc.
        log.error("Unexpected error fetching app_id=%s: %s", app_id, exc)
        return result

    inner = data.get(str(app_id), {})
    if not inner.get("success"):
        log.info("Steam returned success=false for app_id=%s (not on store)", app_id)
        return result

    gd = inner.get("data", {})

    if gd.get("is_free"):
        result["is_free"]          = True
        result["price_final_str"]  = "Free"
    else:
        po = gd.get("price_overview", {})
        if po:
            result["price_final_str"] = po.get("final_formatted", "")
            result["price_orig_str"]  = po.get("initial_formatted", "")
            result["discount_pct"]    = int(po.get("discount_percent", 0))

    if full:
        raw = gd.get("short_description", "")
        result["description"] = re.sub(r"<[^>]+>", "", htmllib.unescape(raw)).strip()
        result["genres"]      = ", ".join(
            g.get("description", "") for g in gd.get("genres", [])[:5]
        )
        result["developers"]  = ", ".join(gd.get("developers", [])[:3])

    return result


# ── Backwards-compat aliases ──────────────────────────────────────────────────
def fetch_steam_details(app_id: int) -> dict:
    return fetch_steam_data(app_id, full=True)

def fetch_live_price(app_id: int) -> dict:
    return fetch_steam_data(app_id, full=False)


# ═══════════════════════════════════════════════════════════════════════════════
# FIX #4 — BATCH PREFETCH
# ═══════════════════════════════════════════════════════════════════════════════

def prefetch_prices(app_ids: list[int], max_workers: int = 8) -> dict[int, dict]:
    """
    Fetches live prices for all app_ids concurrently using a thread pool
    (fix #4), so rendering never blocks on sequential HTTP calls.
    Results are also individually cached by fetch_steam_data's @st.cache_data.
    """
    results: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {pool.submit(fetch_steam_data, aid, False): aid
                      for aid in app_ids}
        for future in as_completed(future_map):
            aid = future_map[future]
            try:
                results[aid] = future.result()
            except Exception as exc:
                log.error("Prefetch failed for app_id=%s: %s", aid, exc)
                results[aid] = {}
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# RENDER HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def cover_url(app_id: int) -> str:
    return f"https://cdn.akamai.steamstatic.com/steam/apps/{app_id}/header.jpg"

def steam_page(app_id: int) -> str:
    return f"https://store.steampowered.com/app/{app_id}"

def _inr(usd: float) -> str:
    return "Free" if usd <= 0 else f"Rs. {usd * USD_TO_INR:,.0f}"

def _ratio_cls(r: float) -> str:
    return "rpos" if r >= 70 else ("rmix" if r >= 40 else "rneg")

def _rating_label(r: float) -> str:
    if r >= 95: return "Overwhelmingly Positive"
    if r >= 80: return "Very Positive"
    if r >= 70: return "Mostly Positive"
    if r >= 40: return "Mixed"
    if r >= 20: return "Mostly Negative"
    return "Overwhelmingly Negative"

def _rating_pill_style(r: float) -> str:
    if r >= 70: return "background:#1a4a3a;color:#66c2a5"
    if r >= 40: return "background:#4a3c00;color:#fee08b"
    return "background:#4a1a1a;color:#f46d43"

def _platform_badges(row: pd.Series) -> str:
    b = ""
    if row.get("win",        0): b += '<span class="bdg bwin">Windows</span>'
    if row.get("mac",        0): b += '<span class="bdg bmac">Mac</span>'
    if row.get("linux",      0): b += '<span class="bdg blin">Linux</span>'
    if row.get("steam_deck", 0): b += '<span class="bdg bdck">Steam Deck</span>'
    return b

def _price_html(row: pd.Series, live: dict | None = None) -> str:
    if live and live.get("price_final_str"):
        if live.get("is_free"):
            return '<span style="color:#66c2a5;font-weight:700">Free to Play</span>'
        price = live["price_final_str"]
        disc  = int(live.get("discount_pct", 0))
        orig  = live.get("price_orig_str", "")
        tag   = ' <span style="color:#66c5db;font-size:0.72rem">live</span>'
        if disc > 0 and orig:
            return (
                f'<span style="color:#c7d5e0;font-weight:600">{price}</span>'
                f' <s style="color:#7ba3b8">{orig}</s>'
                f' <span style="color:#f0c040">-{disc}%</span>{tag}'
            )
        return f'<span style="color:#c7d5e0;font-weight:600">{price}</span>{tag}'

    # Fallback using fix #9 rate
    price = _inr(row["price_final"])
    disc  = float(row.get("discount", 0))
    src   = _xrate_src
    tag   = f' <span style="color:#7ba3b8;font-size:0.72rem">{src}</span>'
    if disc > 0:
        orig = _inr(row["price_original"])
        return (
            f'<span style="color:#c7d5e0;font-weight:600">{price}</span>'
            f' <s style="color:#7ba3b8">{orig}</s>'
            f' <span style="color:#f0c040">-{disc:.0f}%</span>{tag}'
        )
    return f'<span style="color:#c7d5e0;font-weight:600">{price}</span>{tag}'


# ── Hero card ─────────────────────────────────────────────────────────────────

def render_hero_card(row: pd.Series, steam_details: dict):
    app_id       = int(row["app_id"])
    ratio        = row["positive_ratio"]
    release_year = getattr(row["date_release"], "year", "—")

    desc        = steam_details.get("description", "")
    genres      = steam_details.get("genres", "")
    devs        = steam_details.get("developers", "")
    desc_html   = f'<div class="hdesc">{desc}</div>'   if desc   else ""
    genres_html = f'<br><b>Genres:</b> {genres}'       if genres else ""
    devs_html   = f'<br><b>Developer:</b> {devs}'      if devs   else ""

    price_html  = _price_html(row, live=steam_details)

    st.markdown("".join([
        '<div class="hero">',
        f'<img class="hcover" src="{cover_url(app_id)}" '
        f'onerror="this.style.display=\'none\'" alt="">',
        '<div class="hbody">',
        f'<div class="htitle">{row["title"]}</div>',
        f'<span class="hrlbl" style="{_rating_pill_style(ratio)}">',
        f'&#9733; {_rating_label(ratio)} ({ratio}%)',
        '</span>',
        desc_html,
        '<div class="hmeta">',
        _platform_badges(row),
        '<br>', price_html,
        f' &middot; {int(row["user_reviews"]):,} reviews',
        f' &middot; Released {release_year}',
        genres_html, devs_html,
        '</div>',
        f'<a class="steam-btn" href="{steam_page(app_id)}" target="_blank">',
        'View on Steam &#8599;</a>',
        '</div></div>',
    ]), unsafe_allow_html=True)


# ── Compact card — accepts pre-fetched live price dict (fix #4) ───────────────

def render_game_card(
    row: pd.Series,
    show_similarity: bool = False,
    rank: int | None = None,
    live_price: dict | None = None,    # pre-fetched; no extra HTTP call
):
    app_id = int(row["app_id"])
    ratio  = row["positive_ratio"]

    # Only fall back to individual fetch if caller didn't pre-fetch (fix #4)
    if live_price is None:
        live_price = fetch_steam_data(app_id, full=False)

    sim_html  = ""
    if show_similarity and "similarity_score" in row:
        pct      = int(row["similarity_score"] * 100)
        sim_html = f'<span class="simbdg">Match {pct}%</span>'

    rank_html = f'<span class="rnkbdg">{rank}</span>' if rank is not None else ""

    st.markdown("".join([
        f'<a class="crd-link" href="{steam_page(app_id)}" target="_blank">',
        '<div class="gcrd">',
        f'<img class="ccover" src="{cover_url(app_id)}" '
        f'onerror="this.style.display=\'none\'" alt="">',
        '<div class="cbody">',
        sim_html,
        f'<div class="gtitle">{rank_html}{row["title"]}</div>',
        '<div class="gmeta">',
        f'<span class="{_ratio_cls(ratio)}">&#9733; {ratio}%</span>',
        f' &middot; {int(row["user_reviews"]):,} reviews',
        ' &middot; ', _price_html(row, live=live_price),
        '</div>',
        _platform_badges(row),
        '</div>',   # cbody
        '</div>',   # gcrd
        '</a>',
    ]), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER + STATS
# ═══════════════════════════════════════════════════════════════════════════════

url_game   = st.query_params.get("game", "")
all_titles = recommender.get_all_titles()

st.markdown('<div class="main-title">🎮 Steam AI Game Recommendation Engine</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Discover your next favourite game &mdash; '
    'machine learning &middot; live INR prices &middot; '
    'data refreshes every 24 hours automatically</div>',
    unsafe_allow_html=True,
)

# Fix #9: show exchange rate source
xrate_note = (
    f'<div class="xrate-note">'
    f'💱 1 USD = Rs.&nbsp;{USD_TO_INR:.2f} '
    f'<span style="opacity:0.7">({_xrate_src} · refreshes every hour)</span>'
    f'</div>'
)
st.markdown(xrate_note, unsafe_allow_html=True)

total_games = len(df)
free_games  = int((df["price_final"] == 0).sum())
avg_ratio   = df["positive_ratio"].mean()
avg_price   = df[df["price_final"] > 0]["price_final"].mean()

st.markdown(
    f'<div class="mrow">'
    f'<div class="mcrd"><div class="mval">{total_games:,}</div>'
    f'<div class="mlbl">Total Games</div></div>'
    f'<div class="mcrd"><div class="mval">{free_games:,}</div>'
    f'<div class="mlbl">Free-to-Play</div></div>'
    f'<div class="mcrd"><div class="mval">{avg_ratio:.1f}%</div>'
    f'<div class="mlbl">Avg Positive Ratio</div></div>'
    f'<div class="mcrd"><div class="mval">{_inr(avg_price)}</div>'
    f'<div class="mlbl">Avg Price ({_xrate_src})</div></div>'
    f'</div><div class="divhr"></div>',
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — includes genre filter (fix #6)
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🎛️ Filters")
    st.markdown("---")
    st.markdown("**🖥️ Platform**")
    f_win   = st.checkbox("Windows",    value=True)
    f_mac   = st.checkbox("Mac",        value=False)
    f_linux = st.checkbox("Linux",      value=False)
    f_deck  = st.checkbox("Steam Deck", value=False)
    st.markdown("---")

    # Fix #6: Genre / tag filter
    st.markdown("**🎮 Genre / Tags**")
    _all_genres = get_available_genres(df)
    selected_genres: list[str] = []
    if _all_genres:
        selected_genres = st.multiselect(
            "Filter by genre",
            options=_all_genres,
            default=[],
            placeholder="Any genre",
            label_visibility="collapsed",
        )
    else:
        st.caption("No genre data available in dataset.")
    st.markdown("---")

    st.markdown("**💰 Price (USD)**")
    price_range = st.slider("Price range", 0.0, 60.0, (0.0, 60.0),
                            step=0.5, format="$%.2f")
    st.markdown("---")
    st.markdown("**⭐ Min Positive Ratio**")
    min_ratio = st.slider("Minimum %", 0, 100, 0, step=5, format="%d%%")
    st.markdown("---")
    st.markdown("**🔢 Recommendations**")
    top_n = st.slider("How many?", 5, 20, 10)
    st.markdown("---")
    st.markdown(
        "<div style='color:#7ba3b8;font-size:0.8rem'>"
        "&#128279; <strong style='color:#66c5db'>Shareable URLs</strong><br>"
        "After searching, copy the browser URL.<br><br>"
        "&#128260; <strong style='color:#66c5db'>Auto-refresh</strong><br>"
        "Dataset: 24 h &middot; Prices: 30 min &middot; Rate: 1 h"
        "</div>",
        unsafe_allow_html=True,
    )

# Apply all filters
filtered_df = filter_by_platform(df, f_win, f_mac, f_linux, f_deck)
filtered_df = filter_by_price(filtered_df, price_range[0], price_range[1])
filtered_df = filter_by_min_ratio(filtered_df, min_ratio)
if selected_genres:
    filtered_df = filter_by_genre(filtered_df, selected_genres)   # fix #6


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab_search, tab_trending, tab_charts, tab_leaderboard = st.tabs(
    ["🔍 Search & Recommend", "🔥 Trending", "📊 Analytics", "🏆 Leaderboard"]
)


# ── TAB 1 — SEARCH ────────────────────────────────────────────────────────────

with tab_search:
    st.markdown('<div class="sec-hdr">🔍 Game Search</div>', unsafe_allow_html=True)
    st.caption("Click any recommended game card to open it on Steam ↗")

    search_query = st.text_input(
        "Search game",
        value=url_game,
        placeholder="e.g. Portal,  Witcher,  Counter-Strike …",
        label_visibility="collapsed",
    )

    # Fix #8: empty query → show popular defaults, not 200 alphabetical titles
    if search_query:
        matching = [t for t in all_titles if search_query.lower() in t.lower()][:200]
    else:
        # Show top-20 most-reviewed games as helpful defaults
        _popular = (
            df.sort_values("user_reviews", ascending=False)
            .head(20)["title"]
            .tolist()
        )
        matching = _popular

    if not matching:
        st.warning("No games found — try a different keyword.")
        selected_game = None
    else:
        _placeholder_label = (
            "Popular games (type to search)" if not search_query else "Pick a game"
        )
        default_idx   = matching.index(url_game) if url_game in matching else 0
        selected_game = st.selectbox(
            _placeholder_label, matching,
            index=default_idx,
            label_visibility="collapsed",
        )

    if selected_game:
        run_rec  = st.button("🎯 Find Similar Games")
        auto_run = bool(url_game and url_game in all_titles)

        if run_rec or auto_run:
            st.query_params["game"] = selected_game
            st.markdown(
                '<div class="share-box">&#128279; <strong>Shareable link active!</strong> '
                'Copy the browser URL — it encodes <code>'
                + selected_game +
                '</code> so anyone who opens it sees the same recommendations.</div>',
                unsafe_allow_html=True,
            )

            with st.spinner("Fetching game details and computing similarities…"):
                try:
                    query_row  = df[df["title"] == selected_game].iloc[0]
                    steam_info = fetch_steam_data(int(query_row["app_id"]), full=True)

                    st.markdown('<div class="sec-hdr">🎮 Selected Game</div>',
                                unsafe_allow_html=True)
                    render_hero_card(query_row, steam_info)

                    recs = recommender.recommend(
                        selected_game, top_n=top_n,
                        filter_df=(filtered_df if len(filtered_df) < len(df) else None),
                    )
                    st.markdown(
                        f'<div class="sec-hdr">🤖 Top {len(recs)} Similar Games</div>',
                        unsafe_allow_html=True,
                    )
                    if recs.empty:
                        st.info("No matches for current filters — try relaxing them.")
                    else:
                        # Fix #4: batch-prefetch all prices before rendering
                        rec_ids     = [int(r["app_id"]) for _, r in recs.iterrows()]
                        price_cache = prefetch_prices(rec_ids)

                        cols = st.columns(2)
                        for i, (_, row) in enumerate(recs.iterrows()):
                            with cols[i % 2]:
                                render_game_card(
                                    row,
                                    show_similarity=True,
                                    live_price=price_cache.get(int(row["app_id"])),
                                )

                except ValueError as e:
                    st.error(str(e))


# ── TAB 2 — TRENDING ──────────────────────────────────────────────────────────

with tab_trending:
    st.markdown('<div class="sec-hdr">🔥 Top 20 Trending Games</div>',
                unsafe_allow_html=True)
    st.caption("Click any card to open it on Steam ↗")

    trending = get_trending_games(filtered_df, top_n=20)
    if trending.empty:
        st.info("No trending games match the current filters.")
    else:
        # Fix #4: batch-prefetch prices
        t_ids        = [int(r["app_id"]) for _, r in trending.iterrows()]
        t_price_cache = prefetch_prices(t_ids)

        cols = st.columns(2)
        for i, (_, row) in enumerate(trending.iterrows()):
            with cols[i % 2]:
                render_game_card(
                    row, rank=i + 1,
                    live_price=t_price_cache.get(int(row["app_id"])),
                )


# ── TAB 3 — ANALYTICS (Plotly, fix #7) ───────────────────────────────────────

_PLOTLY_THEME = {
    "paper_bgcolor": "#0e1117",
    "plot_bgcolor":  "#1e2d3d",
    "font":          {"color": "#c7d5e0"},
    "title_font":    {"color": "#66c5db"},
}

with tab_charts:
    st.markdown('<div class="sec-hdr">📊 Dataset Analytics</div>',
                unsafe_allow_html=True)
    plot_df = filtered_df.copy()

    # ── Chart 1: Positive ratio — top 15 trending ──────────────────────────
    st.markdown("#### ⭐ Positive Review Ratio — Top 15 Trending")
    top15 = get_trending_games(plot_df, 15)
    if not top15.empty:
        top15_plot = top15.copy()
        top15_plot["short_title"] = top15_plot["title"].str[:35]
        top15_plot["color"] = top15_plot["positive_ratio"].apply(
            lambda r: "#66c2a5" if r >= 70 else ("#fee08b" if r >= 40 else "#f46d43")
        )
        fig = px.bar(
            top15_plot.iloc[::-1], x="positive_ratio", y="short_title",
            orientation="h",
            color="positive_ratio",
            color_continuous_scale=["#f46d43", "#fee08b", "#66c2a5"],
            range_color=[0, 100],
            labels={"positive_ratio": "Positive Ratio (%)", "short_title": ""},
            title="Positive Review Ratio — Top 15 Trending",
            hover_data={"title": True, "positive_ratio": True, "user_reviews": True},
        )
        fig.update_layout(**_PLOTLY_THEME, xaxis_range=[0, 105],
                          coloraxis_showscale=False)
        fig.update_traces(
            hovertemplate="<b>%{customdata[0]}</b><br>"
                          "Positive ratio: %{x}%<br>"
                          "Reviews: %{customdata[1]:,}<extra></extra>",
            customdata=top15_plot.iloc[::-1][["title", "user_reviews"]].values,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="divhr"></div>', unsafe_allow_html=True)

    # ── Chart 2: Price distribution ─────────────────────────────────────────
    st.markdown("#### 💰 Price Distribution — Paid Games")
    paid = plot_df[plot_df["price_final"] > 0]["price_final"] * USD_TO_INR
    if not paid.empty:
        fig = px.histogram(
            paid.clip(upper=60 * USD_TO_INR),
            nbins=40,
            labels={"value": "Price (Rs. INR)", "count": "Number of Games"},
            title="Game Price Distribution (INR)",
            color_discrete_sequence=["#1b9aaa"],
        )
        fig.update_layout(**_PLOTLY_THEME)
        fig.update_traces(hovertemplate="Price: Rs.%{x:,.0f}<br>Games: %{y}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="divhr"></div>', unsafe_allow_html=True)

    # ── Chart 3: Trending score ──────────────────────────────────────────────
    st.markdown("#### 🔥 Trending Score — Top 10")
    top10t = get_trending_games(plot_df, 10)
    if not top10t.empty:
        top10t["short_title"] = top10t["title"].str[:20]
        fig = px.bar(
            top10t, x="short_title", y="trending_score",
            color="trending_score",
            color_continuous_scale="teal",
            labels={"short_title": "", "trending_score": "Trending Score"},
            title="Top 10 Games by Trending Score",
            hover_data={"title": True, "trending_score": True},
        )
        fig.update_layout(**_PLOTLY_THEME, coloraxis_showscale=False,
                          xaxis_tickangle=-35)
        fig.update_traces(
            hovertemplate="<b>%{customdata[0]}</b><br>"
                          "Trending score: %{y:.2f}<extra></extra>",
            customdata=top10t[["title"]].values,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="divhr"></div>', unsafe_allow_html=True)

    # ── Chart 4: Platform coverage ───────────────────────────────────────────
    st.markdown("#### 🖥️ Platform Coverage")
    plat_counts = {
        "Windows":    int(plot_df["win"].sum()),
        "Mac":        int(plot_df["mac"].sum()),
        "Linux":      int(plot_df["linux"].sum()),
        "Steam Deck": int(plot_df["steam_deck"].sum()),
    }
    fig = px.bar(
        x=list(plat_counts.keys()),
        y=list(plat_counts.values()),
        color=list(plat_counts.keys()),
        color_discrete_map={
            "Windows": "#85c1e9", "Mac": "#76d7a5",
            "Linux": "#c39bd3", "Steam Deck": "#f0c040",
        },
        labels={"x": "Platform", "y": "Number of Games"},
        title="Games Available per Platform",
    )
    fig.update_layout(**_PLOTLY_THEME, showlegend=False)
    fig.update_traces(hovertemplate="%{x}: %{y:,} games<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)


# ── TAB 4 — LEADERBOARD ───────────────────────────────────────────────────────

with tab_leaderboard:
    st.markdown('<div class="sec-hdr">🏆 Popularity Leaderboard</div>',
                unsafe_allow_html=True)
    st.caption("Click any card to open it on Steam ↗")

    lb_col1, lb_col2 = st.columns(2)

    with lb_col1:
        st.markdown("#### 💬 Most Reviewed Games")
        top_reviewed = (
            filtered_df.sort_values("user_reviews", ascending=False)
            .head(10).reset_index(drop=True)
        )
        # Fix #4: batch-prefetch
        rev_ids        = [int(r["app_id"]) for _, r in top_reviewed.iterrows()]
        rev_price_cache = prefetch_prices(rev_ids)
        for i, (_, row) in enumerate(top_reviewed.iterrows()):
            render_game_card(
                row, rank=i + 1,
                live_price=rev_price_cache.get(int(row["app_id"])),
            )

    with lb_col2:
        st.markdown("#### 🌟 Highest Popularity Score")
        top_popular = (
            filtered_df.sort_values("popularity_score", ascending=False)
            .head(10).reset_index(drop=True)
        )
        pop_ids        = [int(r["app_id"]) for _, r in top_popular.iterrows()]
        pop_price_cache = prefetch_prices(pop_ids)
        for i, (_, row) in enumerate(top_popular.iterrows()):
            render_game_card(
                row, rank=i + 1,
                live_price=pop_price_cache.get(int(row["app_id"])),
            )
