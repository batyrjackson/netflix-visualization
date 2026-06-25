import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
import warnings
warnings.filterwarnings("ignore")

# ── Настройки страницы ────────────────────────────────────────
st.set_page_config(
    page_title="Netflix Dashboard",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #141414; }
    h1, h2, h3 { color: #E50914; }
    .metric-card {
        background: #221F1F;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid #E50914;
    }
</style>
""", unsafe_allow_html=True)

# ── Загрузка данных ───────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/processed/netflix_cleaned.csv")
    except FileNotFoundError:
        df = pd.read_csv("data/raw/netflix_titles.csv")
        df.drop_duplicates(inplace=True)
        df["director"].fillna("Unknown", inplace=True)
        df["cast"].fillna("Unknown", inplace=True)
        df["country"].fillna("United States", inplace=True)
        df["rating"].fillna("TV-MA", inplace=True)
        df["duration"].fillna("0 min", inplace=True)
        df.dropna(subset=["date_added"], inplace=True)
        df["date_added"] = df["date_added"].str.strip()
        df["year_added"]  = pd.to_datetime(df["date_added"], errors="coerce").dt.year
        df["month_added"] = pd.to_datetime(df["date_added"], errors="coerce").dt.month
        df["country_main"] = df["country"].apply(lambda x: str(x).split(",")[0].strip())
        df["duration_val"] = df["duration"].str.extract(r"(\d+)").astype(float)
        df["genre_count"]  = df["listed_in"].apply(lambda x: len(str(x).split(",")))
    return df

df = load_data()

# ── Заголовок ────────────────────────────────────────────────
st.title("🎬 Netflix Movies & TV Shows — Аналитический Дашборд")
st.markdown("**Деректерді визуализациялау** пәні бойынша жобалық жұмыс")
st.markdown("---")

# ── Боковая панель — Фильтры ──────────────────────────────────
st.sidebar.header("🔧 Фильтры")

type_filter = st.sidebar.multiselect(
    "Тип контента",
    options=df["type"].unique().tolist(),
    default=df["type"].unique().tolist()
)

top_countries = df["country_main"].value_counts().nlargest(15).index.tolist()
country_filter = st.sidebar.multiselect(
    "Страна",
    options=top_countries,
    default=top_countries[:5]
)

year_min = int(df["year_added"].min()) if df["year_added"].notna().any() else 2008
year_max = int(df["year_added"].max()) if df["year_added"].notna().any() else 2021
year_range = st.sidebar.slider(
    "Год добавления",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max)
)

# Применение фильтров
mask = (
    df["type"].isin(type_filter) &
    df["country_main"].isin(country_filter if country_filter else df["country_main"].unique()) &
    df["year_added"].between(year_range[0], year_range[1])
)
filtered = df[mask]

# ── KPI карточки ──────────────────────────────────────────────
st.subheader("📊 Ключевые показатели (KPI)")
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.metric("🎬 Всего контента", f"{len(filtered):,}")
with k2:
    movies = len(filtered[filtered["type"] == "Movie"])
    st.metric("🎥 Фильмов", f"{movies:,}")
with k3:
    shows = len(filtered[filtered["type"] == "TV Show"])
    st.metric("📺 Сериалов", f"{shows:,}")
with k4:
    countries = filtered["country_main"].nunique()
    st.metric("🌍 Стран", f"{countries}")
with k5:
    top_genre = filtered["listed_in"].str.split(",").explode().str.strip().value_counts().idxmax() if len(filtered) else "—"
    st.metric("🏆 Топ жанр", top_genre[:15])

st.markdown("---")

# ── Визуализации ─────────────────────────────────────────────
st.subheader("📈 Визуализации")

col1, col2 = st.columns(2)

with col1:
    # Movie vs TV Show
    type_counts = filtered["type"].value_counts().reset_index()
    type_counts.columns = ["type", "count"]
    fig1 = px.pie(type_counts, values="count", names="type",
                  title="Movie vs TV Show",
                  color_discrete_sequence=["#E50914", "#564d4d"])
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # Контент по годам
    yearly = filtered.groupby(["year_added", "type"]).size().reset_index(name="count")
    fig2 = px.line(yearly, x="year_added", y="count", color="type",
                   title="Добавление контента по годам",
                   color_discrete_sequence=["#E50914", "#564d4d"],
                   markers=True)
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    # Топ-10 стран
    top10 = filtered["country_main"].value_counts().nlargest(10).reset_index()
    top10.columns = ["country", "count"]
    fig3 = px.bar(top10, x="count", y="country", orientation="h",
                  title="Топ-10 стран по контенту",
                  color_discrete_sequence=["#E50914"])
    fig3.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    # Рейтинги
    rating_counts = filtered["rating"].value_counts().nlargest(10).reset_index()
    rating_counts.columns = ["rating", "count"]
    fig4 = px.bar(rating_counts, x="rating", y="count",
                  title="Распределение по рейтингу",
                  color_discrete_sequence=["#E50914"])
    st.plotly_chart(fig4, use_container_width=True)

# Boxplot длительности
st.subheader("📦 Длительность контента (Boxplot)")
box_data = filtered.copy()
fig5 = px.box(box_data[box_data["duration_val"] > 0],
              x="type", y="duration_val", color="type",
              title="Распределение длительности по типу контента",
              color_discrete_sequence=["#E50914", "#564d4d"],
              labels={"duration_val": "Длительность", "type": "Тип"})
st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")

# ── ML Модель ─────────────────────────────────────────────────
st.subheader("🤖 Machine Learning Model — Классификация (Movie vs TV Show)")

with st.expander("ℹ️ О модели", expanded=True):
    st.markdown("""
    - **Задача:** Классификация — предсказать тип контента (Movie / TV Show)  
    - **Алгоритм:** Random Forest Classifier  
    - **Признаки:** год выхода, год добавления, рейтинг, страна, количество жанров, длительность  
    - **Метрики:** Accuracy, Precision, Recall, F1-Score
    """)

@st.cache_data
def train_model(data):
    d = data.copy()
    d["target"] = (d["type"] == "Movie").astype(int)

    rating_order = {
        "G": 1, "TV-Y": 1, "TV-Y7": 2, "TV-Y7-FV": 2,
        "PG": 3, "TV-PG": 3, "PG-13": 4, "TV-14": 4,
        "R": 5, "TV-MA": 5, "NC-17": 6, "NR": 0, "UR": 0
    }
    d["rating_enc"] = d["rating"].map(rating_order).fillna(0)

    top10c = d["country_main"].value_counts().nlargest(10).index
    d["country_clean"] = d["country_main"].apply(lambda x: x if x in top10c else "Other")
    le = LabelEncoder()
    d["country_enc"] = le.fit_transform(d["country_clean"])

    feats = ["release_year", "year_added", "rating_enc",
             "country_enc", "genre_count", "duration_val"]
    feats = [f for f in feats if f in d.columns]

    X = d[feats].fillna(0)
    y = d["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    m = {
        "Accuracy":  round(accuracy_score(y_test, preds), 4),
        "Precision": round(precision_score(y_test, preds, average="weighted"), 4),
        "Recall":    round(recall_score(y_test, preds, average="weighted"), 4),
        "F1-Score":  round(f1_score(y_test, preds, average="weighted"), 4),
    }
    cm = confusion_matrix(y_test, preds)
    fi = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    return m, cm, fi, X_test.shape[0]

with st.spinner("Обучение модели..."):
    ml_metrics, cm, fi, test_size = train_model(df)

# Метрики ML
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("✅ Accuracy",  f"{ml_metrics['Accuracy']:.1%}")
mc2.metric("🎯 Precision", f"{ml_metrics['Precision']:.1%}")
mc3.metric("🔍 Recall",    f"{ml_metrics['Recall']:.1%}")
mc4.metric("⚖️ F1-Score",  f"{ml_metrics['F1-Score']:.1%}")

ml_col1, ml_col2 = st.columns(2)

with ml_col1:
    # Confusion Matrix
    fig_cm = px.imshow(cm,
                       labels=dict(x="Предсказано", y="Реально", color="Кол-во"),
                       x=["TV Show", "Movie"],
                       y=["TV Show", "Movie"],
                       text_auto=True,
                       color_continuous_scale="Reds",
                       title="Confusion Matrix")
    st.plotly_chart(fig_cm, use_container_width=True)

with ml_col2:
    # Feature Importance
    fi_df = fi.reset_index()
    fi_df.columns = ["Признак", "Важность"]
    fig_fi = px.bar(fi_df, x="Важность", y="Признак", orientation="h",
                    title="Feature Importance",
                    color_discrete_sequence=["#E50914"])
    fig_fi.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_fi, use_container_width=True)

st.markdown("---")
st.markdown("""
**📝 Аналитические выводы:**
- Датасет Netflix содержит преимущественно фильмы (~70%)
- США является лидером по производству контента
- Пик добавления контента пришёлся на 2018–2020 годы
- Random Forest классифицирует Movie/TV Show с высокой точностью
- Длительность и рейтинг — наиболее важные признаки для модели
""")

st.caption("Жобалық зертханалық жұмыс | Деректерді визуализациялау | Астана, 2026")
