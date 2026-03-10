"""
app/streamlit_app.py
--------------------
Steam AI Game Recommendation Engine — Streamlit front-end.

FIXES IN THIS VERSION
─────────────────────
  • Raw HTML text bug fixed — cards now render correctly
  • Prices shown in Indian Rupees (₹)
  • Leaderboard games are clickable → Steam store page
  • Cover images, shareable URLs, game detail panel all retained

Run with:
    streamlit run app/streamlit_app.py
"""

import sys, os, re, html as htmllib, json
import urllib.request
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.data_processing import (
    load_and_clean_data, engineer_features,
    filter_by_platform, filter_by_price,
    filter_by_min_ratio, get_trending_games,
)
from model.recommendation_model import GameRecommender

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


# ── Currency conversion ────────────────────────────────────────────────────────
USD_TO_INR = 84.0   # approximate; update as needed

def to_inr(usd: float) -> str:
    """Convert a USD price to a formatted Indian Rupees string."""
    if usd <= 0:
        return "Free"
    rupees = usd * USD_TO_INR
    # Use Indian number formatting (e.g. ₹1,249)
    return f"₹{rupees:,.0f}"


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
# CSS  — all card HTML is now built via st.components.v1.html or pure CSS
#         classes to avoid Streamlit's markdown parser choking on inline styles
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Global ────────────────────────────────────────── */
.stApp { background-color: #0e1117; color: #c7d5e0; }
section[data-testid="stSidebar"] { background-color: #1b2838; }
section[data-testid="stSidebar"] * { color: #c7d5e0 !important; }

/* ── Typography ────────────────────────────────────── */
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

/* ── Metric strip ──────────────────────────────────── */
.mrow { display: flex; gap: 12px; margin-bottom: 18px; flex-wrap: wrap; }
.mcrd {
    flex: 1; min-width: 120px;
    background: #1e2d3d; border: 1px solid #2a4158;
    border-radius: 10px; padding: 14px 18px; text-align: center;
}
.mval { font-size: 1.6rem; font-weight: 800; color: #66c5db; }
.mlbl { font-size: 0.78rem; color: #7ba3b8; margin-top: 2px; }

/* ── Game card (compact) ───────────────────────────── */
.gcrd {
    background: linear-gradient(135deg, #1e2d3d 0%, #162232 100%);
    border: 1px solid #2a4158; border-radius: 12px;
    overflow: hidden; margin-bottom: 14px;
    transition: border-color 0.25s;
}
.gcrd:hover { border-color: #66c5db; }
.gcrd img.ccover {
    width: 100%; height: 140px; object-fit: cover;
    display: block; border-bottom: 1px solid #2a4158;
}
.gcrd .cbody { padding: 12px 14px 14px; }
.gtitle { font-size: 0.98rem; font-weight: 700; color: #e8f4f8; margin-bottom: 5px; }
.gmeta  { font-size: 0.8rem; color: #7ba3b8; margin: 4px 0 7px; }

/* ── Hero card (selected game) ─────────────────────── */
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

/* ── Badges ────────────────────────────────────────── */
.bdg {
    display: inline-block; padding: 2px 9px; border-radius: 20px;
    font-size: 0.74rem; font-weight: 600; margin-right: 4px; margin-top: 3px;
}
.bwin  { background: #1a5276; color: #85c1e9; }
.bmac  { background: #1e3a2e; color: #76d7a5; }
.blin  { background: #4a235a; color: #c39bd3; }
.bdck  { background: #4a3500; color: #f0c040; }

/* ── Sim / rank badges ─────────────────────────────── */
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
    font-size: 0.78rem; font-weight: 800; margin-right: 7px;
    flex-shrink: 0;
}

/* ── Rating colours ────────────────────────────────── */
.rpos  { color: #66c2a5; font-weight: 700; }
.rmix  { color: #fee08b; font-weight: 700; }
.rneg  { color: #f46d43; font-weight: 700; }

/* ── Clickable leaderboard card ─────────────────────── */
.lcrd-link {
    display: block; text-decoration: none; color: inherit;
}
.lcrd-link:hover .gcrd { border-color: #66c5db; }

/* ── Steam link button ─────────────────────────────── */
.steam-btn {
    display: inline-block; margin-top: 10px;
    background: #1b9aaa; color: #fff;
    padding: 6px 16px; border-radius: 8px;
    text-decoration: none; font-size: 0.82rem; font-weight: 600;
}
.steam-btn:hover { background: #17848f; color: #fff; }

/* ── Share callout ─────────────────────────────────── */
.share-box {
    background: #1e2d3d; border: 1px dashed #2a6080; border-radius: 10px;
    padding: 12px 16px; margin-bottom: 16px;
    font-size: 0.85rem; color: #7ba3b8;
}
.share-box code { color: #66c5db; }

/* ── Divider ───────────────────────────────────────── */
.divhr { height: 1px; background: #2a4158; margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA + MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading and processing 50 000+ games…")
def load_data() -> pd.DataFrame:
    path = os.path.join(ROOT, "dataset", "games.csv")
    df   = load_and_clean_data(path)
    df   = engineer_features(df)
    return df

@st.cache_resource(show_spinner="Building recommendation model…")
def build_model(df: pd.DataFrame) -> GameRecommender:
    rec = GameRecommender()
    rec.fit(df)
    return rec

df          = load_data()
recommender = build_model(df)


# ═══════════════════════════════════════════════════════════════════════════════
# STEAM API HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def cover_url(app_id: int) -> str:
    return f"https://cdn.akamai.steamstatic.com/steam/apps/{app_id}/header.jpg"

def steam_page(app_id: int) -> str:
    return f"https://store.steampowered.com/app/{app_id}"

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_steam_details(app_id: int) -> dict:
    """
    Fetch from Steam Store API (cc=in = Indian store):
      - short description, genres, developers
      - REAL-TIME INR price via price_overview

    Price keys returned:
      price_final_str  e.g. '499'   (numeric string, paise / 100)
      price_orig_str   e.g. '999'
      price_symbol     e.g. 'Rs. '  (Steam's INR prefix)
      discount_pct     int, 0 if none
      is_free          bool
    """
    result = {
        "description":    "",
        "genres":         "",
        "developers":     "",
        "price_final_str": "",
        "price_orig_str":  "",
        "price_symbol":    "",
        "discount_pct":    0,
        "is_free":         False,
    }
    try:
        url = (
            "https://store.steampowered.com/api/appdetails"
            f"?appids={app_id}&cc=in&l=en"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())

        inner = data.get(str(app_id), {})
        if not inner.get("success"):
            return result

        gd = inner.get("data", {})

        # ── Description ───────────────────────────────────────────────
        raw = gd.get("short_description", "")
        result["description"] = re.sub(r"<[^>]+>", "", htmllib.unescape(raw)).strip()

        # ── Genres + Developers ───────────────────────────────────────
        result["genres"]     = ", ".join(
            g.get("description", "") for g in gd.get("genres", [])[:5]
        )
        result["developers"] = ", ".join(gd.get("developers", [])[:3])

        # ── Real-time INR price ───────────────────────────────────────
        if gd.get("is_free"):
            result["is_free"]         = True
            result["price_final_str"] = "Free"
        else:
            po = gd.get("price_overview", {})
            if po:
                # Steam returns final / initial in smallest unit (paise for INR)
                # divide by 100 to get rupees; also grab pre-formatted strings
                final_raw = po.get("final", 0)
                orig_raw  = po.get("initial", 0)
                result["price_final_str"] = po.get("final_formatted",
                                                    f"Rs. {final_raw / 100:.0f}")
                result["price_orig_str"]  = po.get("initial_formatted",
                                                    f"Rs. {orig_raw / 100:.0f}")
                result["discount_pct"]    = int(po.get("discount_percent", 0))

    except Exception:
        pass
    return result


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_live_price(app_id: int) -> dict:
    """
    Lightweight price-only call for compact cards (recommendations / trending).
    Returns the same price keys as fetch_steam_details.
    Using the filters param keeps the payload small and fast.
    """
    result = {
        "price_final_str": "",
        "price_orig_str":  "",
        "discount_pct":    0,
        "is_free":         False,
    }
    try:
        url = (
            "https://store.steampowered.com/api/appdetails"
            f"?appids={app_id}&cc=in&l=en&filters=price_overview,basic"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode())

        inner = data.get(str(app_id), {})
        if not inner.get("success"):
            return result

        gd = inner.get("data", {})
        if gd.get("is_free"):
            result["is_free"]         = True
            result["price_final_str"] = "Free"
        else:
            po = gd.get("price_overview", {})
            if po:
                final_raw = po.get("final", 0)
                orig_raw  = po.get("initial", 0)
                result["price_final_str"] = po.get("final_formatted",
                                                    f"Rs. {final_raw / 100:.0f}")
                result["price_orig_str"]  = po.get("initial_formatted",
                                                    f"Rs. {orig_raw / 100:.0f}")
                result["discount_pct"]    = int(po.get("discount_percent", 0))
    except Exception:
        pass
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# RENDER HELPERS
# NOTE: All HTML is built as a single contiguous string — no inline f-string
#       style attributes that contain comparison operators (< >) which confuse
#       Streamlit's markdown HTML sanitiser and cause raw-text rendering.
# ═══════════════════════════════════════════════════════════════════════════════

def _ratio_cls(ratio: float) -> str:
    """CSS class for the ratio value."""
    if ratio >= 70:
        return "rpos"
    if ratio >= 40:
        return "rmix"
    return "rneg"

def _rating_label(ratio: float) -> str:
    if ratio >= 95: return "Overwhelmingly Positive"
    if ratio >= 80: return "Very Positive"
    if ratio >= 70: return "Mostly Positive"
    if ratio >= 40: return "Mixed"
    if ratio >= 20: return "Mostly Negative"
    return "Overwhelmingly Negative"

def _rating_pill_style(ratio: float) -> str:
    """Inline style string for the rating pill — built safely, no < > chars."""
    if ratio >= 70:
        return "background:#1a4a3a;color:#66c2a5"
    if ratio >= 40:
        return "background:#4a3c00;color:#fee08b"
    return "background:#4a1a1a;color:#f46d43"

def _platform_badges(row: pd.Series) -> str:
    b = ""
    if row.get("win",        0): b += '<span class="bdg bwin">Windows</span>'
    if row.get("mac",        0): b += '<span class="bdg bmac">Mac</span>'
    if row.get("linux",      0): b += '<span class="bdg blin">Linux</span>'
    if row.get("steam_deck", 0): b += '<span class="bdg bdck">Steam Deck</span>'
    return b

def _price_html(row: pd.Series, live: dict | None = None) -> str:
    """
    Build the price HTML string.

    Priority:
      1. live dict from fetch_steam_details / fetch_live_price  (real INR from Steam)
      2. Dataset price converted to INR as fallback

    'live' is a dict with keys: price_final_str, price_orig_str, discount_pct, is_free
    """
    # ── Use real-time Steam data when available ───────────────────────
    if live and live.get("price_final_str"):
        if live.get("is_free"):
            return '<span style="color:#66c2a5;font-weight:700">Free to Play</span>'

        price = live["price_final_str"]
        disc  = int(live.get("discount_pct", 0))
        orig  = live.get("price_orig_str", "")

        if disc > 0 and orig:
            return (
                f'<span style="color:#c7d5e0;font-weight:600">{price}</span>'
                f' <s style="color:#7ba3b8">{orig}</s>'
                f' <span style="color:#f0c040">-{disc}%</span>'
                f' <span style="color:#66c5db;font-size:0.72rem">&nbsp;live</span>'
            )
        return (
            f'<span style="color:#c7d5e0;font-weight:600">{price}</span>'
            f' <span style="color:#66c5db;font-size:0.72rem">&nbsp;live</span>'
        )

    # ── Fallback: dataset price converted to INR ──────────────────────
    price = to_inr(row["price_final"])
    disc  = float(row.get("discount", 0))
    if disc > 0:
        orig = to_inr(row["price_original"])
        return (
            f'<span style="color:#c7d5e0;font-weight:600">{price}</span>'
            f' <s style="color:#7ba3b8">{orig}</s>'
            f' <span style="color:#f0c040">-{disc:.0f}%</span>'
            f' <span style="color:#7ba3b8;font-size:0.72rem">&nbsp;est.</span>'
        )
    return (
        f'<span style="color:#c7d5e0;font-weight:600">{price}</span>'
        f' <span style="color:#7ba3b8;font-size:0.72rem">&nbsp;est.</span>'
    )


# ── Hero card (selected game with full details) ───────────────────────────────

def render_hero_card(row: pd.Series, steam_details: dict):
    app_id  = int(row["app_id"])
    ratio   = row["positive_ratio"]
    r_label = _rating_label(ratio)
    r_style = _rating_pill_style(ratio)
    r_cls   = _ratio_cls(ratio)

    desc   = steam_details.get("description", "")
    genres = steam_details.get("genres", "")
    devs   = steam_details.get("developers", "")

    release_year = (
        row["date_release"].year
        if hasattr(row["date_release"], "year") else "—"
    )

    desc_html   = f'<div class="hdesc">{desc}</div>' if desc else ""
    genres_html = f'<br><b>Genres:</b> {genres}'      if genres else ""
    devs_html   = f'<br><b>Developer:</b> {devs}'     if devs else ""

    # steam_details already has real-time price — pass it as live source
    price_html = _price_html(row, live=steam_details)

    parts = [
        '<div class="hero">',
        f'<img class="hcover" src="{cover_url(app_id)}" '
        f'onerror="this.style.display=\'none\'" alt="">',
        '<div class="hbody">',
        f'<div class="htitle">{row["title"]}</div>',
        f'<span class="hrlbl" style="{r_style}">',
        f'&#9733; {r_label} ({ratio}%)',
        '</span>',
        desc_html,
        '<div class="hmeta">',
        _platform_badges(row),
        '<br>',
        price_html,
        f' &middot; {int(row["user_reviews"]):,} reviews',
        f' &middot; Released {release_year}',
        genres_html,
        devs_html,
        '</div>',
        f'<a class="steam-btn" href="{steam_page(app_id)}" target="_blank">',
        'View on Steam &#8599;',
        '</a>',
        '</div>',
        '</div>',
    ]
    st.markdown("".join(parts), unsafe_allow_html=True)


# ── Compact game card (recommendations + trending) ────────────────────────────

def render_game_card(row: pd.Series,
                     show_similarity: bool = False,
                     rank: int | None = None):
    app_id = int(row["app_id"])
    ratio  = row["positive_ratio"]
    r_cls  = _ratio_cls(ratio)

    # Fetch real-time price (cached — fast after first call per game)
    live_price = fetch_live_price(app_id)

    sim_html  = ""
    if show_similarity and "similarity_score" in row:
        pct      = int(row["similarity_score"] * 100)
        sim_html = f'<span class="simbdg">Match {pct}%</span>'

    rank_html = ""
    if rank is not None:
        rank_html = f'<span class="rnkbdg">{rank}</span>'

    parts = [
        '<div class="gcrd">',
        f'<img class="ccover" src="{cover_url(app_id)}" '
        f'onerror="this.style.display=\'none\'" alt="">',
        '<div class="cbody">',
        sim_html,
        f'<div class="gtitle">{rank_html}{row["title"]}</div>',
        '<div class="gmeta">',
        f'<span class="{r_cls}">&#9733; {ratio}%</span>',
        f' &middot; {int(row["user_reviews"]):,} reviews',
        f' &middot; ',
        _price_html(row, live=live_price),
        '</div>',
        _platform_badges(row),
        '</div>',
        '</div>',
    ]
    st.markdown("".join(parts), unsafe_allow_html=True)


# ── Leaderboard card  (clickable → Steam page) ────────────────────────────────

def render_leaderboard_card(row: pd.Series, rank: int, subtitle_html: str):
    """
    Compact card that wraps the whole thing in an <a> tag so clicking
    anywhere on the card opens the game's Steam store page.
    """
    app_id = int(row["app_id"])
    r_cls  = _ratio_cls(row["positive_ratio"])

    parts = [
        f'<a class="lcrd-link" href="{steam_page(app_id)}" target="_blank">',
        '<div class="gcrd">',
        f'<img class="ccover" src="{cover_url(app_id)}" '
        f'onerror="this.style.display=\'none\'" alt="">',
        '<div class="cbody">',
        f'<div class="gtitle">',
        f'<span class="rnkbdg">{rank}</span>',
        row["title"],
        '</div>',
        f'<div class="gmeta">{subtitle_html}</div>',
        '</div>',
        '</div>',
        '</a>',
    ]
    st.markdown("".join(parts), unsafe_allow_html=True)


# ── Matplotlib figure factory ─────────────────────────────────────────────────

def make_mpl_fig():
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#1e2d3d")
    ax.tick_params(colors="#c7d5e0")
    ax.xaxis.label.set_color("#c7d5e0")
    ax.yaxis.label.set_color("#c7d5e0")
    ax.title.set_color("#66c5db")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a4158")
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════════
# URL QUERY PARAM
# ═══════════════════════════════════════════════════════════════════════════════

url_game   = st.query_params.get("game", "")
all_titles = recommender.get_all_titles()


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER + STATS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="main-title">🎮 Steam AI Game Recommendation Engine</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Discover your next favourite game — '
    'powered by machine learning &middot; Prices in &#8377; INR</div>',
    unsafe_allow_html=True,
)

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
    f'<div class="mcrd"><div class="mval">{to_inr(avg_price)}</div>'
    f'<div class="mlbl">Avg Price (paid)</div></div>'
    f'</div>'
    f'<div class="divhr"></div>',
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
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
    st.markdown("**💰 Price (₹ INR)**")
    max_inr    = int(60 * USD_TO_INR)          # ₹5040 ≈ $60
    price_inr  = st.slider("Max price (₹)", 0, max_inr, (0, max_inr), step=100,
                            format="₹%d")
    # Convert slider value back to USD for the filter functions
    price_usd  = (price_inr[0] / USD_TO_INR, price_inr[1] / USD_TO_INR)
    st.markdown("---")
    st.markdown("**⭐ Min Positive Ratio**")
    min_ratio = st.slider("Minimum %", 0, 100, 0, step=5, format="%d%%")
    st.markdown("---")
    st.markdown("**🔢 Recommendations**")
    top_n = st.slider("How many?", 5, 20, 10)
    st.markdown("---")
    st.markdown(
        "<div style='color:#7ba3b8;font-size:0.8rem'>"
        "Data: Steam Games Dataset<br>"
        "Model: Content-based cosine similarity<br><br>"
        "<strong style='color:#66c5db'>&#128279; Shareable URLs</strong><br>"
        "After searching, your browser URL encodes the game. "
        "Copy &amp; share it!"
        "</div>",
        unsafe_allow_html=True,
    )

# Apply sidebar filters
filtered_df = filter_by_platform(df, f_win, f_mac, f_linux, f_deck)
filtered_df = filter_by_price(filtered_df, price_usd[0], price_usd[1])
filtered_df = filter_by_min_ratio(filtered_df, min_ratio)


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab_search, tab_trending, tab_charts, tab_leaderboard = st.tabs(
    ["🔍 Search & Recommend", "🔥 Trending", "📊 Analytics", "🏆 Leaderboard"]
)


# ───────────────────────────────────────────────────────────────────────────────
# TAB 1 — SEARCH & RECOMMEND
# ───────────────────────────────────────────────────────────────────────────────

with tab_search:
    st.markdown('<div class="sec-hdr">🔍 Game Search</div>', unsafe_allow_html=True)

    search_query = st.text_input(
        "Search game",
        value=url_game,
        placeholder="e.g.  Portal,  Witcher,  Counter-Strike …",
        label_visibility="collapsed",
    )

    if search_query:
        matching = [t for t in all_titles if search_query.lower() in t.lower()][:200]
    else:
        matching = all_titles[:200]

    if not matching:
        st.warning("No games found — try a different keyword.")
        selected_game = None
    else:
        default_idx = 0
        if url_game and url_game in matching:
            default_idx = matching.index(url_game)
        selected_game = st.selectbox(
            "Pick a game",
            options=matching,
            index=default_idx,
            label_visibility="collapsed",
        )

    if selected_game:
        run_rec  = st.button("🎯 Find Similar Games")
        auto_run = bool(url_game and url_game in all_titles)

        if run_rec or auto_run:
            st.query_params["game"] = selected_game
            encoded = urllib.parse.quote(selected_game)

            st.markdown(
                '<div class="share-box">'
                '&#128279; <strong>Shareable link active!</strong> '
                'Your URL now encodes <code>' + selected_game + '</code>. '
                'Copy and share it — anyone opening the link sees the same results.'
                '</div>',
                unsafe_allow_html=True,
            )

            with st.spinner("Fetching game details and computing similarities…"):
                try:
                    query_row  = df[df["title"] == selected_game].iloc[0]
                    app_id     = int(query_row["app_id"])
                    steam_info = fetch_steam_details(app_id)

                    st.markdown('<div class="sec-hdr">🎮 Selected Game</div>',
                                unsafe_allow_html=True)
                    render_hero_card(query_row, steam_info)

                    recs = recommender.recommend(
                        selected_game,
                        top_n=top_n,
                        filter_df=(filtered_df
                                   if len(filtered_df) < len(df) else None),
                    )

                    st.markdown(
                        f'<div class="sec-hdr">🤖 Top {len(recs)} Similar Games</div>',
                        unsafe_allow_html=True,
                    )

                    if recs.empty:
                        st.info("No matches for current filters — "
                                "try relaxing platform / price settings.")
                    else:
                        cols = st.columns(2)
                        for i, (_, row) in enumerate(recs.iterrows()):
                            with cols[i % 2]:
                                render_game_card(row, show_similarity=True)

                except ValueError as e:
                    st.error(str(e))


# ───────────────────────────────────────────────────────────────────────────────
# TAB 2 — TRENDING
# ───────────────────────────────────────────────────────────────────────────────

with tab_trending:
    st.markdown('<div class="sec-hdr">🔥 Top 20 Trending Games</div>',
                unsafe_allow_html=True)

    trending = get_trending_games(filtered_df, top_n=20)

    if trending.empty:
        st.info("No trending games match the current filters.")
    else:
        cols = st.columns(2)
        for i, (_, row) in enumerate(trending.iterrows()):
            with cols[i % 2]:
                render_game_card(row, rank=i + 1)


# ───────────────────────────────────────────────────────────────────────────────
# TAB 3 — ANALYTICS
# ───────────────────────────────────────────────────────────────────────────────

with tab_charts:
    st.markdown('<div class="sec-hdr">📊 Dataset Analytics</div>',
                unsafe_allow_html=True)
    plot_df = filtered_df.copy()

    # Chart 1 — Positive ratio
    st.markdown("#### ⭐ Positive Review Ratio — Top 15 Trending Games")
    top15 = get_trending_games(plot_df, 15)
    if not top15.empty:
        fig, ax = make_mpl_fig()
        colors = [
            "#66c2a5" if r >= 70 else "#fee08b" if r >= 40 else "#f46d43"
            for r in top15["positive_ratio"]
        ]
        bars = ax.barh(
            top15["title"].str[:35].tolist()[::-1],
            top15["positive_ratio"].tolist()[::-1],
            color=colors[::-1], edgecolor="#2a4158",
        )
        ax.set_xlabel("Positive Ratio (%)")
        ax.set_title("Positive Review Ratio — Top 15 Trending")
        ax.set_xlim(0, 105)
        for bar, val in zip(bars, top15["positive_ratio"].tolist()[::-1]):
            ax.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{val}%", va="center", color="#c7d5e0", fontsize=8,
            )
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown('<div class="divhr"></div>', unsafe_allow_html=True)

    # Chart 2 — Price histogram (in INR)
    st.markdown("#### 💰 Price Distribution — Paid Games (₹ INR)")
    paid_inr = plot_df[plot_df["price_final"] > 0]["price_final"] * USD_TO_INR
    if not paid_inr.empty:
        fig, ax = make_mpl_fig()
        ax.hist(paid_inr.clip(upper=60 * USD_TO_INR), bins=40,
                color="#1b9aaa", edgecolor="#0e1117", alpha=0.85)
        ax.set_xlabel("Price (₹ INR)")
        ax.set_ylabel("Number of Games")
        ax.set_title("Game Price Distribution (INR)")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown('<div class="divhr"></div>', unsafe_allow_html=True)

    # Chart 3 — Trending score
    st.markdown("#### 🔥 Trending Score — Top 10 Games")
    top10t = get_trending_games(plot_df, 10)
    if not top10t.empty:
        fig, ax = make_mpl_fig()
        palette = plt.cm.cool(np.linspace(0.3, 0.9, len(top10t)))
        ax.bar(range(len(top10t)), top10t["trending_score"].tolist(),
               color=palette, edgecolor="#0e1117")
        ax.set_xticks(range(len(top10t)))
        ax.set_xticklabels(
            [t[:18] for t in top10t["title"].tolist()],
            rotation=35, ha="right", fontsize=8,
        )
        ax.set_ylabel("Trending Score")
        ax.set_title("Top 10 Games by Trending Score")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown('<div class="divhr"></div>', unsafe_allow_html=True)

    # Chart 4 — Platform coverage
    st.markdown("#### 🖥️ Platform Coverage")
    plat_counts = {
        "Windows":    int(plot_df["win"].sum()),
        "Mac":        int(plot_df["mac"].sum()),
        "Linux":      int(plot_df["linux"].sum()),
        "Steam Deck": int(plot_df["steam_deck"].sum()),
    }
    fig, ax = make_mpl_fig()
    bars = ax.bar(
        list(plat_counts.keys()), list(plat_counts.values()),
        color=["#85c1e9","#76d7a5","#c39bd3","#f0c040"],
        edgecolor="#0e1117",
    )
    ax.set_ylabel("Number of Games")
    ax.set_title("Games Available per Platform")
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{int(bar.get_height()):,}",
            ha="center", color="#c7d5e0", fontsize=9,
        )
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)


# ───────────────────────────────────────────────────────────────────────────────
# TAB 4 — LEADERBOARD  (each card is a clickable link to Steam)
# ───────────────────────────────────────────────────────────────────────────────

with tab_leaderboard:
    st.markdown('<div class="sec-hdr">🏆 Popularity Leaderboard</div>',
                unsafe_allow_html=True)
    st.caption("Click any game card to open its Steam store page ↗")

    lb_col1, lb_col2 = st.columns(2)

    with lb_col1:
        st.markdown("#### 💬 Most Reviewed Games")
        top_reviewed = (
            filtered_df.sort_values("user_reviews", ascending=False)
            .head(10).reset_index(drop=True)
        )
        for i, (_, row) in enumerate(top_reviewed.iterrows()):
            r_cls      = _ratio_cls(row["positive_ratio"])
            live_price = fetch_live_price(int(row["app_id"]))
            price_str  = _price_html(row, live=live_price)
            subtitle = (
                f'<span class="{r_cls}">&#9733; {row["positive_ratio"]}%</span>'
                f' &middot; {int(row["user_reviews"]):,} reviews'
                f' &middot; {price_str}'
            )
            render_leaderboard_card(row, rank=i + 1, subtitle_html=subtitle)

    with lb_col2:
        st.markdown("#### 🌟 Highest Popularity Score")
        top_popular = (
            filtered_df.sort_values("popularity_score", ascending=False)
            .head(10).reset_index(drop=True)
        )
        for i, (_, row) in enumerate(top_popular.iterrows()):
            r_cls      = _ratio_cls(row["positive_ratio"])
            live_price = fetch_live_price(int(row["app_id"]))
            price_str  = _price_html(row, live=live_price)
            subtitle = (
                f'<span class="{r_cls}">&#9733; {row["positive_ratio"]}%</span>'
                f' &middot; Score: <strong style="color:#66c5db">'
                f'{row["popularity_score"]:.2f}</strong>'
                f' &middot; {price_str}'
            )
            render_leaderboard_card(row, rank=i + 1, subtitle_html=subtitle)
