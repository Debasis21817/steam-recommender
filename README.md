# 🎮 Steam AI Game Recommendation Engine

A scalable, content-based machine-learning recommendation system built with
**Python · Pandas · Scikit-learn · Streamlit · Matplotlib**.

---

## 📁 Project Structure

```
steam_recommender/
├── dataset/
│   └── games.csv                  ← 50 000+ Steam games
├── model/
│   ├── __init__.py
│   └── recommendation_model.py    ← GameRecommender class (cosine similarity)
├── app/
│   └── streamlit_app.py           ← Streamlit web interface
├── utils/
│   ├── __init__.py
│   └── data_processing.py         ← Load, clean, feature-engineer, filter
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1 · Install dependencies
```bash
pip install -r requirements.txt
```

### 2 · Run the app
```bash
streamlit run app/streamlit_app.py
```

The browser will open automatically at **http://localhost:8501**.

---

## 🧠 How It Works

### Data Pipeline (`utils/data_processing.py`)
| Step | What happens |
|------|-------------|
| Load | `pd.read_csv("games.csv")` |
| Clean | Drop nulls, parse dates, cast booleans to int, deduplicate titles |
| Features | `popularity_score`, `discount_percentage`, `game_age`, `trending_score` |
| Filters | Platform, price range, minimum positive-ratio |

### Recommendation Model (`model/recommendation_model.py`)
1. Build a **numeric feature matrix** from: `positive_ratio`, `user_reviews`,
   `price_final`, `discount`, `popularity_score`, `win`, `mac`, `linux`, `steam_deck`
2. **Normalise** with `StandardScaler` so no feature dominates
3. Compute **cosine similarity** between the query game and all candidates
4. Return the **top-N** most similar games

### Trending Score
```
trending_score = positive_ratio × log(user_reviews)
```

### Popularity Score
```
popularity_score = (positive_ratio × 0.7) + (log(user_reviews) × 0.3)
```

---

## 🖥️ App Features

| Tab | Contents |
|-----|----------|
| 🔍 Search & Recommend | Type-ahead search → select game → get top-N similar games |
| 🔥 Trending | Top 20 trending games (filtered by sidebar) |
| 📊 Analytics | 4 charts: positive ratio, price distribution, trending score, platform coverage |
| 🏆 Leaderboard | Most-reviewed & highest popularity-score games |

### Sidebar Filters
- **Platform** – Windows / Mac / Linux / Steam Deck (checkboxes)
- **Price** – $0 – $60 range slider
- **Min Positive Ratio** – 0 – 100 % slider
- **Recommendation count** – 5 – 20 slider

---

## 📦 Dataset Columns

| Column | Description |
|--------|-------------|
| `app_id` | Steam Application ID |
| `title` | Game title |
| `date_release` | Release date |
| `win / mac / linux` | Platform support flags |
| `steam_deck` | Steam Deck compatibility |
| `rating` | Textual rating (e.g. "Very Positive") |
| `positive_ratio` | % of positive reviews (0-100) |
| `user_reviews` | Total review count |
| `price_final` | Current price (USD) |
| `price_original` | Original price before discount |
| `discount` | Discount % |

---

## 🔧 Requirements

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
streamlit>=1.32.0
matplotlib>=3.7.0
```

Python **3.10+** recommended.
