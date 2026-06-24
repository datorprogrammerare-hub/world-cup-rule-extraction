import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

try:
    from anthropic import Anthropic
except Exception:
    Anthropic = None


# -------------------------------------------------------
# Config
# -------------------------------------------------------

load_dotenv()

FOOTBALL_DATA_KEY = os.getenv("FOOTBALL_DATA_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
FOOTBALL_DATA_COMPETITION = os.getenv("FOOTBALL_DATA_COMPETITION", "WC")
FOOTBALL_DATA_SEASON = int(os.getenv("FOOTBALL_DATA_SEASON", "2026"))
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

st.set_page_config(
    page_title="World Cup 2026 Rule Extraction",
    layout="wide"
)

st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f8fbff 0%, #eef6ff 45%, #f5fff8 100%);
    }

    /* Main title */
    h1 {
        color: #0b3d91;
        font-weight: 800;
    }

    h2, h3 {
        color: #123c69;
        font-weight: 700;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #e8f3ff 100%);
        border: 1px solid #b7d7ff;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 4px 14px rgba(0, 70, 140, 0.12);
    }

    [data-testid="stMetricLabel"] {
        color: #0b3d91;
        font-weight: 700;
    }

    [data-testid="stMetricValue"] {
        color: #0077b6;
        font-weight: 800;
    }

    /* Success boxes */
    .stSuccess {
        background-color: #d8f8e1;
        border-left: 6px solid #16a34a;
        border-radius: 12px;
    }

    /* Info boxes */
    .stInfo {
        background-color: #e0f2fe;
        border-left: 6px solid #0284c7;
        border-radius: 12px;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        border-radius: 14px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }

        /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b3d91 0%, #14532d 100%);
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] span {
        color: white;
    }

    /* Text inside select boxes */
    [data-testid="stSidebar"] [data-baseweb="select"] span {
        color: #111827 !important;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] div {
        color: #111827 !important;
    }

    /* Select box background */
    [data-testid="stSidebar"] [data-baseweb="select"] {
        background-color: white;
        border-radius: 10px;
    }
    
    /* Tabs */
    button[data-baseweb="tab"] {
        font-weight: 700;
        color: #0b3d91;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #dbeafe;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("World Cup 2026 Champion Predictor")
st.subheader("Machine Learning + Rule Extraction + Football-data.org + Anthropic")


# -------------------------------------------------------
# Local backup dataset
# -------------------------------------------------------

@st.cache_data
def load_backup_dataset():
    teams = [
        # Group A
        ("Mexico", "A", 15, 77, 72, 77, 75, 75, 82, 66),
        ("South Africa", "A", 58, 63, 61, 64, 62, 60, 62, 74),
        ("South Korea", "A", 23, 68, 66, 71, 71, 67, 64, 70),
        ("Czechia", "A", 34, 69, 67, 70, 69, 70, 68, 72),

        # Group B
        ("Canada", "B", 31, 70, 73, 74, 76, 68, 61, 67),
        ("Bosnia and Herzegovina", "B", 55, 65, 62, 66, 66, 64, 60, 73),
        ("Qatar", "B", 53, 64, 61, 65, 63, 63, 65, 75),
        ("Switzerland", "B", 19, 74, 70, 76, 73, 79, 76, 72),

        # Group C
        ("Brazil", "C", 5, 88, 82, 91, 91, 83, 98, 73),
        ("Morocco", "C", 13, 76, 79, 80, 79, 77, 74, 70),
        ("Haiti", "C", 85, 58, 57, 60, 59, 56, 52, 78),
        ("Scotland", "C", 39, 68, 65, 68, 67, 69, 64, 72),

        # Group D
        ("United States", "D", 16, 78, 73, 79, 80, 76, 70, 65),
        ("Paraguay", "D", 48, 66, 68, 66, 65, 69, 69, 71),
        ("Australia", "D", 27, 69, 67, 69, 68, 70, 68, 73),
        ("Turkey", "D", 26, 72, 67, 75, 77, 69, 66, 72),

        # Group E
        ("Germany", "E", 10, 86, 80, 88, 86, 84, 97, 69),
        ("Curacao", "E", 90, 57, 59, 58, 58, 56, 50, 79),
        ("Ivory Coast", "E", 42, 69, 70, 70, 71, 68, 64, 72),
        ("Ecuador", "E", 24, 71, 68, 72, 70, 73, 67, 74),

        # Group F
        ("Netherlands", "F", 7, 85, 81, 86, 85, 85, 85, 71),
        ("Japan", "F", 18, 75, 71, 77, 76, 74, 68, 72),
        ("Sweden", "F", 28, 71, 69, 72, 72, 71, 73, 73),
        ("Tunisia", "F", 41, 66, 62, 66, 65, 67, 65, 76),

        # Group G
        ("Belgium", "G", 8, 83, 76, 84, 84, 81, 80, 76),
        ("Egypt", "G", 32, 69, 68, 70, 72, 66, 69, 73),
        ("Iran", "G", 21, 70, 67, 70, 69, 70, 67, 73),
        ("New Zealand", "G", 91, 57, 58, 58, 57, 58, 55, 77),

        # Group H
        ("Spain", "H", 3, 91, 90, 90, 90, 87, 90, 72),
        ("Cape Verde", "H", 65, 62, 64, 63, 64, 61, 57, 76),
        ("Saudi Arabia", "H", 56, 63, 61, 64, 64, 62, 66, 75),
        ("Uruguay", "H", 12, 82, 78, 82, 82, 80, 91, 78),

        # Group I
        ("France", "I", 2, 92, 88, 92, 92, 88, 95, 70),
        ("Senegal", "I", 17, 74, 73, 76, 75, 75, 69, 72),
        ("Iraq", "I", 60, 62, 64, 63, 62, 63, 59, 77),
        ("Norway", "I", 25, 74, 72, 77, 80, 69, 61, 74),

        # Group J
        ("Argentina", "J", 1, 90, 84, 88, 88, 85, 96, 68),
        ("Algeria", "J", 43, 68, 67, 69, 70, 66, 65, 75),
        ("Austria", "J", 22, 73, 72, 74, 73, 74, 68, 74),
        ("Jordan", "J", 70, 60, 63, 61, 60, 61, 56, 78),

        # Group K
        ("Portugal", "K", 6, 87, 85, 90, 90, 82, 84, 74),
        ("DR Congo", "K", 61, 63, 64, 64, 65, 62, 58, 77),
        ("Uzbekistan", "K", 64, 62, 65, 63, 62, 62, 56, 77),
        ("Colombia", "K", 14, 79, 77, 81, 82, 77, 78, 74),

        # Group L
        ("England", "L", 4, 89, 86, 89, 89, 86, 88, 75),
        ("Croatia", "L", 11, 81, 75, 80, 79, 82, 92, 77),
        ("Ghana", "L", 51, 66, 65, 67, 68, 64, 69, 74),
        ("Panama", "L", 49, 65, 66, 66, 65, 64, 61, 75),
    ]

    columns = [
        "team", "group", "fifa_ranking", "elo_rating", "recent_form",
        "overall_rating", "attack_rating", "defense_rating",
        "world_cup_experience", "group_difficulty"
    ]

    df = pd.DataFrame(teams, columns=columns)

    df["points"] = 0
    df["wins"] = 0
    df["draws"] = 0
    df["losses"] = 0
    df["goals_for"] = 0
    df["goals_against"] = 0
    df["goal_difference"] = 0
    df["api_data_available"] = False

    return df

# -------------------------------------------------------
# football-data.org
# -------------------------------------------------------

@st.cache_data(ttl=60)
def fetch_football_data_matches():
    """
    Fetch World Cup matches from football-data.org.
    If the API key is missing or the API returns no data,
    the app continues using the local backup dataset.
    """
    if not FOOTBALL_DATA_KEY or FOOTBALL_DATA_KEY == "TU_TOKEN_DE_FOOTBALL_DATA_ORG":
        return pd.DataFrame(), "No FOOTBALL_DATA_KEY found. Using local backup data."

    url = f"https://api.football-data.org/v4/competitions/{FOOTBALL_DATA_COMPETITION}/matches"

    headers = {
        "X-Auth-Token": FOOTBALL_DATA_KEY
    }

    params = {
        "season": FOOTBALL_DATA_SEASON
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)

        if response.status_code != 200:
            return pd.DataFrame(), f"football-data.org error {response.status_code}: {response.text[:250]}"

        payload = response.json()
        matches = payload.get("matches", [])

        if not matches:
            return pd.DataFrame(), "football-data.org returned no matches. Using local backup data."

        rows = []

        for match in matches:
            score = match.get("score", {}).get("fullTime", {})

            rows.append({
                "home_team": match.get("homeTeam", {}).get("name", ""),
                "away_team": match.get("awayTeam", {}).get("name", ""),
                "home_goals": score.get("home"),
                "away_goals": score.get("away"),
                "status": match.get("status", ""),
                "utc_date": match.get("utcDate", "")
            })

        return pd.DataFrame(rows), "football-data.org matches loaded successfully."

    except Exception as e:
        return pd.DataFrame(), f"football-data.org request failed: {e}"


def build_team_stats_from_matches(matches_df):
    """
    Convert football-data.org match data into team-level stats:
    points, wins, draws, losses, goals for, goals against, and goal difference.
    """
    if matches_df is None or matches_df.empty:
        return pd.DataFrame()

    rows = []

    finished_matches = matches_df[
        (matches_df["status"].isin(["FINISHED", "IN_PLAY", "PAUSED", "LIVE"]))
        & matches_df["home_goals"].notna()
        & matches_df["away_goals"].notna()
    ]

    for _, match in finished_matches.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        home_goals = int(match["home_goals"])
        away_goals = int(match["away_goals"])

        if home_goals > away_goals:
            home_points, away_points = 3, 0
            home_win, home_draw, home_loss = 1, 0, 0
            away_win, away_draw, away_loss = 0, 0, 1
        elif home_goals < away_goals:
            home_points, away_points = 0, 3
            home_win, home_draw, home_loss = 0, 0, 1
            away_win, away_draw, away_loss = 1, 0, 0
        else:
            home_points, away_points = 1, 1
            home_win, home_draw, home_loss = 0, 1, 0
            away_win, away_draw, away_loss = 0, 1, 0

        rows.append({
            "team": home,
            "points": home_points,
            "wins": home_win,
            "draws": home_draw,
            "losses": home_loss,
            "goals_for": home_goals,
            "goals_against": away_goals,
            "api_data_available": True
        })

        rows.append({
            "team": away,
            "points": away_points,
            "wins": away_win,
            "draws": away_draw,
            "losses": away_loss,
            "goals_for": away_goals,
            "goals_against": home_goals,
            "api_data_available": True
        })

    if not rows:
        return pd.DataFrame()

    stats = pd.DataFrame(rows)

    stats = stats.groupby("team", as_index=False).agg({
        "points": "sum",
        "wins": "sum",
        "draws": "sum",
        "losses": "sum",
        "goals_for": "sum",
        "goals_against": "sum",
        "api_data_available": "max"
    })

    stats["goal_difference"] = stats["goals_for"] - stats["goals_against"]

    return stats


def merge_football_data(local_df, api_df):
    """
    Merge local backup dataset with football-data.org team statistics.
    """
    if api_df is None or api_df.empty:
        return local_df

    df = local_df.copy()
    api = api_df.copy()

    df["team_key"] = df["team"].str.lower().str.strip()
    api["team_key"] = api["team"].str.lower().str.strip()

    merged = df.merge(
        api[
            [
                "team_key",
                "points",
                "goals_for",
                "goals_against",
                "wins",
                "draws",
                "losses",
                "goal_difference",
                "api_data_available"
            ]
        ],
        on="team_key",
        how="left",
        suffixes=("", "_api")
    )

    for col in [
        "points",
        "goals_for",
        "goals_against",
        "wins",
        "draws",
        "losses",
        "goal_difference"
    ]:
        api_col = f"{col}_api"
        if api_col in merged.columns:
            merged[col] = merged[api_col].fillna(merged[col])
            merged.drop(columns=[api_col], inplace=True)

    if "api_data_available_api" in merged.columns:
        merged["api_data_available"] = merged["api_data_available_api"].fillna(False)
        merged.drop(columns=["api_data_available_api"], inplace=True)

    merged.drop(columns=["team_key"], inplace=True)

    return merged


def build_group_standings(data):
    """
    Build current group standings as information only.
    It uses API match statistics when available, otherwise local backup values.
    """
    standings = data.copy()

    standings["played"] = standings["wins"] + standings["draws"] + standings["losses"]

    standings = standings[
        [
            "group",
            "team",
            "played",
            "wins",
            "draws",
            "losses",
            "goals_for",
            "goals_against",
            "goal_difference",
            "points",
            "api_data_available"
        ]
    ]

    standings = standings.sort_values(
        by=[
            "group",
            "points",
            "goal_difference",
            "goals_for",
            "wins"
        ],
        ascending=[
            True,
            False,
            False,
            False,
            False
        ]
    ).reset_index(drop=True)

    return standings
# -------------------------------------------------------
# ML preparation
# -------------------------------------------------------

def create_targets(df):
    data = df.copy()

    strength_score = (
        (100 - data["fifa_ranking"]) * 0.18
        + data["elo_rating"] * 0.18
        + data["recent_form"] * 0.14
        + data["overall_rating"] * 0.13
        + data["attack_rating"] * 0.13
        + data["defense_rating"] * 0.10
        + data["world_cup_experience"] * 0.08
        + data["points"] * 1.70
        + data["goal_difference"] * 1.20
        - data["group_difficulty"] * 0.05
    )

    data["strength_score"] = strength_score

    champion_cutoff = data["strength_score"].quantile(0.84)
    semifinal_cutoff = data["strength_score"].quantile(0.75)
    final_cutoff = data["strength_score"].quantile(0.92)

    data["champion_candidate"] = (data["strength_score"] >= champion_cutoff).astype(int)
    data["semifinal_candidate"] = (data["strength_score"] >= semifinal_cutoff).astype(int)
    data["final_candidate"] = (data["strength_score"] >= final_cutoff).astype(int)

    return data


def train_models(data, target, max_depth):
    features = [
        "fifa_ranking",
        "elo_rating",
        "recent_form",
        "overall_rating",
        "attack_rating",
        "defense_rating",
        "world_cup_experience",
        "group_difficulty",
        "points",
        "goals_for",
        "goals_against",
        "goal_difference",
        "wins",
        "draws",
        "losses",
    ]

    X = data[features]
    y = data[target]

    black_box = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )
    black_box.fit(X, y)

    black_box_predictions = black_box.predict(X)
    probabilities = black_box.predict_proba(X)[:, 1]

    surrogate = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=42,
        min_samples_leaf=2
    )
    surrogate.fit(X, black_box_predictions)

    surrogate_predictions = surrogate.predict(X)

    fidelity = accuracy_score(black_box_predictions, surrogate_predictions)

    return black_box, surrogate, probabilities, black_box_predictions, surrogate_predictions, fidelity, features


# -------------------------------------------------------
# Tournament prediction logic
# -------------------------------------------------------

def build_tournament_projection(result_df):
    ranked = result_df.sort_values("champion_probability", ascending=False).reset_index(drop=True)

    top10 = ranked.head(10)
    semifinalists = ranked.head(4)
    finalists = ranked.head(2)
    champion = ranked.iloc[0]

    semifinal_1 = (semifinalists.iloc[0]["team"], semifinalists.iloc[3]["team"])
    semifinal_2 = (semifinalists.iloc[1]["team"], semifinalists.iloc[2]["team"])

    final = (finalists.iloc[0]["team"], finalists.iloc[1]["team"])

    return top10, semifinalists, finalists, champion, semifinal_1, semifinal_2, final


# -------------------------------------------------------
# Anthropic explanation
# -------------------------------------------------------

def fallback_explanation(team, probability, rules):
    return f"""
{team} aparece como candidato fuerte porque el modelo combina factores como ranking FIFA,
rating Elo, forma reciente, ataque, defensa, experiencia mundialista y rendimiento del torneo.

La probabilidad estimada es {probability:.2f}%. Las reglas extraídas muestran qué condiciones
usa el árbol de decisión sustituto para imitar al modelo Random Forest. Esto convierte una
predicción compleja en una explicación más fácil de entender.
"""


def generate_anthropic_explanation(team, probability, top10_names, rules_text):
    if not ANTHROPIC_API_KEY or Anthropic is None:
        return fallback_explanation(team, probability, rules_text)

    try:
        client = Anthropic(api_key=ANTHROPIC_API_KEY)

        prompt = f"""
You are explaining a machine learning project in simple academic English.

Project:
Predicting the FIFA World Cup 2026 champion using machine learning and rule extraction.

Predicted champion:
{team}

Model champion score:
{probability:.2f}%

Top 10 teams:
{", ".join(top10_names)}

Extracted decision-tree surrogate rules:
{rules_text}

Explain in 3 short paragraphs:
1. Why the model selected the champion.
2. How the IF-THEN rules help explain the prediction.
3. Why rule extraction is useful for interpreting black-box ML models.

Do not use Markdown headings.
Do not use bullet points.
Do not use bold text.
Keep the explanation clear, concise, and suitable for a student presentation.
"""

        message = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=600,
            temperature=0.3,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return message.content[0].text

    except Exception as e:
        return fallback_explanation(team, probability, rules_text) + f"\n\nAnthropic API fallback used because: {e}"


# -------------------------------------------------------
# Load data
# -------------------------------------------------------

local_data = load_backup_dataset()
api_matches, api_status = fetch_football_data_matches()
api_team_stats = build_team_stats_from_matches(api_matches)
data = merge_football_data(local_data, api_team_stats)
data = create_targets(data)
group_standings = build_group_standings(data)

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

st.sidebar.header("Settings")

prediction_target_label = st.sidebar.selectbox(
    "Prediction target",
    ["Champion Candidate", "Semifinal Candidate", "Final Candidate"]
)

target_map = {
    "Champion Candidate": "champion_candidate",
    "Semifinal Candidate": "semifinal_candidate",
    "Final Candidate": "final_candidate"
}

target = target_map[prediction_target_label]

max_depth = st.sidebar.slider(
    "Surrogate tree max depth",
    min_value=2,
    max_value=7,
    value=4
)

selected_team = st.sidebar.selectbox(
    "Select team",
    data["team"].sort_values().tolist()
)

use_anthropic = st.sidebar.checkbox(
    "Use Anthropic explanation",
    value=bool(ANTHROPIC_API_KEY)
)

st.sidebar.write("Anthropic activado:", use_anthropic)
st.sidebar.write("Equipo seleccionado:", selected_team)

# -------------------------------------------------------
# Train champion model for tournament projection
# -------------------------------------------------------

champion_model, champion_surrogate, champion_probs, champion_preds, champion_surr_preds, champion_fidelity, champion_features = train_models(
    data,
    "champion_candidate",
    max_depth
)

data["champion_probability"] = 5 + (champion_probs * 65)
data["champion_model_prediction"] = champion_preds
data["champion_surrogate_prediction"] = champion_surr_preds

top10, semifinalists, finalists, champion, semifinal_1, semifinal_2, final = build_tournament_projection(data)


# -------------------------------------------------------
# Train selected target model for rule view
# -------------------------------------------------------

model, surrogate, probabilities, model_predictions, surrogate_predictions, fidelity, features = train_models(
    data,
    target,
    max_depth
)

data["selected_probability"] = probabilities * 100
data["selected_model_prediction"] = model_predictions
data["selected_surrogate_prediction"] = surrogate_predictions

rules_text = export_text(surrogate, feature_names=features)


# -------------------------------------------------------
# Layout
# -------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Dashboard",
        "API Data",
        "Extracted Rules",
        "AI Explanation",
        "Project Story"
    ]
)


with tab1:
    st.header("World Cup 2026 Prediction Dashboard")

    col_status_1, col_status_2, col_status_3 = st.columns(3)

    with col_status_1:
        st.metric("Data source", "football-data.org + Backup" if not api_team_stats.empty else "Local Backup")

    with col_status_2:
        st.metric("Teams", len(data))

    with col_status_3:
        st.metric("Rule Fidelity", f"{champion_fidelity * 100:.2f}%")

    st.info(api_status)

    st.subheader("Predicted Champion")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Champion", champion["team"])

    with col2:
        st.metric("Model Champion Score", f"{champion['champion_probability']:.2f}%")

    with col3:
        st.metric("Group", champion["group"])

    st.subheader("Top 10 Favorites")

    top10.index = range(1, len(top10) + 1)

    st.dataframe(
        top10[
            [
                "team",
                "group",
                "champion_probability",
                "fifa_ranking",
                "elo_rating",
                "recent_form",
                "attack_rating",
                "defense_rating",
                "points",
                "goal_difference"
            ]
        ],
        use_container_width=True
    )

    st.subheader("Predicted Semifinals")

    c1, c2 = st.columns(2)

    with c1:
        st.write("### Semifinal 1")
        st.success(f"{semifinal_1[0]} vs {semifinal_1[1]}")

    with c2:
        st.write("### Semifinal 2")
        st.success(f"{semifinal_2[0]} vs {semifinal_2[1]}")

    st.subheader("Predicted Final")

    st.success(f"{final[0]} vs {final[1]}")

    st.subheader("Selected Country Result")

    selected_dashboard_row = data[data["team"] == selected_team].iloc[0]

    r_col1, r_col2, r_col3, r_col4 = st.columns(4)

    with r_col1:
        st.metric("Selected Country", selected_dashboard_row["team"])

    with r_col2:
        st.metric("Group", selected_dashboard_row["group"])

    with r_col3:
        st.metric(
            "Model Champion Score",
            f"{selected_dashboard_row['champion_probability']:.2f}%"
        )

    with r_col4:
        champion_label = (
            "Yes"
            if selected_dashboard_row["champion_model_prediction"] == 1
            else "No"
        )
        st.metric("Champion Candidate", champion_label)

    selected_stats = pd.DataFrame(
        [
            {
                "team": selected_dashboard_row["team"],
                "group": selected_dashboard_row["group"],
                "played": int(
                    selected_dashboard_row["wins"]
                    + selected_dashboard_row["draws"]
                    + selected_dashboard_row["losses"]
                ),
                "wins": int(selected_dashboard_row["wins"]),
                "draws": int(selected_dashboard_row["draws"]),
                "losses": int(selected_dashboard_row["losses"]),
                "goals_for": int(selected_dashboard_row["goals_for"]),
                "goals_against": int(selected_dashboard_row["goals_against"]),
                "goal_difference": int(selected_dashboard_row["goal_difference"]),
                "points": int(selected_dashboard_row["points"]),
                "model_champion_score": f"{selected_dashboard_row['champion_probability']:.2f}%"
            }
        ]
    )

    st.dataframe(
        selected_stats,
        use_container_width=True,
        hide_index=True
    )

    st.write(
        f"""
        **{selected_dashboard_row['team']}** has a model champion score of
        **{selected_dashboard_row['champion_probability']:.2f}%** according to the Random Forest model.
        The result is based on ranking, Elo rating, recent form, attack, defense,
        World Cup experience, points, goals and goal difference.
        """
    )

    st.subheader("Current Group Standings")

    st.write(
        """
        This section shows the current group information only.
        It includes played games, wins, draws, losses, goals, goal difference and points.
        """
    )

    selected_group_dashboard = st.selectbox(
        "Select group to view current standings",
        sorted(group_standings["group"].unique()),
        key="dashboard_group_selector"
    )

    group_table_dashboard = group_standings[
        group_standings["group"] == selected_group_dashboard
    ].copy()

    group_table_dashboard = group_table_dashboard.reset_index(drop=True)
    group_table_dashboard.insert(0, "position", range(1, len(group_table_dashboard) + 1))

    st.dataframe(
        group_table_dashboard[
            [
                "position",
                "team",
                "played",
                "wins",
                "draws",
                "losses",
                "goals_for",
                "goals_against",
                "goal_difference",
                "points",
                "api_data_available"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Top 12 Model Score Chart")

    chart_df = data.sort_values("champion_probability", ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(chart_df["team"], chart_df["champion_probability"])
    ax.set_ylabel("Model champion score (%)")
    ax.set_xlabel("Team")
    ax.set_title("Predicted Model Champion Score")
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)



    
with tab2:
    st.header("football-data.org Data")

    st.write("API status:")
    st.code(api_status)

    st.write(
        """
        If football-data.org returns match data, the app converts completed matches
        into team statistics such as points, goals, wins, draws, losses, and goal difference.
        If the API fails, the app automatically uses the local backup dataset.
        """
    )

    if api_matches is not None and not api_matches.empty:
        st.subheader("Raw API Matches")
        st.dataframe(api_matches, use_container_width=True, hide_index=True)

        st.subheader("Team Stats Built From Matches")
        st.dataframe(api_team_stats, use_container_width=True, hide_index=True)
    else:
        st.warning("No football-data.org data loaded. The app is using local backup data.")

    st.subheader("Final Model Dataset")
    st.dataframe(data, use_container_width=True, hide_index=True)



with tab3:
    st.header("Extracted IF-THEN Rules")

    st.write(
        """
        The Random Forest is the black-box model. The Decision Tree is the surrogate model.
        The surrogate model learns to imitate the Random Forest, then we extract readable
        IF-THEN rules from the tree.
        """
    )

    st.metric("Selected target", prediction_target_label)
    st.metric("Fidelity", f"{fidelity * 100:.2f}%")

    st.code(rules_text, language="text")

    st.subheader("Selected Team Prediction")

    selected_row = data[data["team"] == selected_team].iloc[0]

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.metric("Team", selected_team)

    with col_b:
        st.metric("Probability", f"{selected_row['selected_probability']:.2f}%")

    with col_c:
        label = "Yes" if selected_row["selected_model_prediction"] == 1 else "No"
        st.metric(prediction_target_label, label)


with tab4:
    st.header("Anthropic AI Explanation")

    st.write(
        """
        Anthropic is used only to explain the model results in natural language.
        It does not replace the machine learning model.
        """
    )

    if use_anthropic:
        with st.spinner("Generating explanation..."):
            explanation = generate_anthropic_explanation(
                champion["team"],
                champion["champion_probability"],
                top10["team"].tolist(),
                rules_text
            )
    else:
        explanation = fallback_explanation(
            champion["team"],
            champion["champion_probability"],
            rules_text
        )

    st.markdown(explanation)


with tab5:
    st.header("Project Story")

    st.write(
        """
        This project predicts possible FIFA World Cup 2026 contenders using machine learning.
        The app combines football data, a local backup dataset, a Random Forest model, and a
        Decision Tree surrogate model.

        The goal is not only to predict a champion. The main goal is to explain the prediction.
        A black-box model such as Random Forest can be accurate, but it is difficult to interpret.
        To solve this, we train a Decision Tree surrogate to imitate the Random Forest. Then we
        extract IF-THEN rules from the tree.

        This makes the prediction more transparent. Instead of only saying which team is likely
        to win, the app also explains the conditions that led to that prediction.
        """
    )

    st.subheader("System Architecture")

    st.code(
        """
Football-data.org
+ Local backup dataset
        ↓
Data processing
        ↓
Random Forest model
        ↓
Prediction:
- Top 10 favorites
- 4 semifinalists
- 2 finalists
- champion
        ↓
Decision Tree surrogate
        ↓
Extracted IF-THEN rules
        ↓
Anthropic explanation
        """,
        language="text"
    )

    st.subheader("Presentation Slides Outline")

    st.markdown(
        """
        1. Title: Predicting the World Cup 2026 Champion Using Rule Extraction  
        2. Problem: ML predictions are often hard to explain  
        3. Data sources: Football-data.org + local backup dataset  
        4. Features: ranking, Elo, form, attack, defense, points, goals  
        5. Black-box model: Random Forest  
        6. Rule extraction: Decision Tree surrogate  
        7. Metrics: model champion score and fidelity  
        8. Results: Top 10, semifinals, final, champion  
        9. Extracted IF-THEN rules  
        10. Anthropic explanation  
        11. GenAI literacy  
        12. Conclusion  
        """
    )