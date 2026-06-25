import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.data_cleaning import load_and_clean

os.makedirs("img", exist_ok=True)

# ── 1. Загрузка ──────────────────────────────────────────────
df = load_and_clean()
print(df.head())
print(df.info())
print(df.describe())

# ── 2. Пропуски ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing):
    missing.plot(kind="bar", ax=ax, color="salmon")
    ax.set_title("Пропущенные значения по столбцам")
    ax.set_ylabel("Количество")
else:
    ax.text(0.5, 0.5, "Пропусков нет ✅", ha="center", va="center",
            fontsize=14, transform=ax.transAxes)
    ax.set_title("Пропущенные значения")
plt.tight_layout()
plt.savefig("img/missing_values.png", dpi=150)
plt.show()

# ── 3. Movie vs TV Show ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
type_counts = df["type"].value_counts()
axes[0].pie(type_counts, labels=type_counts.index, autopct="%1.1f%%",
            colors=["#E50914", "#221F1F"], startangle=90,
            textprops={"color": "black"})
axes[0].set_title("Movie vs TV Show")

type_counts.plot(kind="bar", ax=axes[1], color=["#E50914", "#221F1F"])
axes[1].set_title("Количество контента по типу")
axes[1].set_ylabel("Количество")
axes[1].tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig("img/type_distribution.png", dpi=150)
plt.show()

# ── 4. Топ-10 стран ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
top_countries = df["country_main"].value_counts().nlargest(10)
top_countries.plot(kind="bar", ax=ax, color="#E50914")
ax.set_title("Топ-10 стран по количеству контента")
ax.set_ylabel("Количество")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("img/top_countries.png", dpi=150)
plt.show()

# ── 5. Контент по годам ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
yearly = df.groupby(["year_added", "type"]).size().unstack(fill_value=0)
yearly.plot(kind="line", ax=ax, marker="o", color=["#E50914", "#564d4d"])
ax.set_title("Добавление контента по годам")
ax.set_ylabel("Количество")
ax.set_xlabel("Год")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("img/content_by_year.png", dpi=150)
plt.show()

# ── 6. Топ-10 рейтингов ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
df["rating"].value_counts().nlargest(10).plot(kind="bar", ax=ax, color="#E50914")
ax.set_title("Распределение по рейтингу")
ax.set_ylabel("Количество")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("img/rating_distribution.png", dpi=150)
plt.show()

# ── 7. Boxplot длительности ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
movies = df[df["type"] == "Movie"]["duration_val"].dropna()
shows  = df[df["type"] == "TV Show"]["duration_val"].dropna()
ax.boxplot([movies, shows], labels=["Movie (мин)", "TV Show (сезоны)"], patch_artist=True,
           boxprops=dict(facecolor="#E50914", alpha=0.6))
ax.set_title("Распределение длительности (Boxplot)")
ax.set_ylabel("Значение")
plt.tight_layout()
plt.savefig("img/boxplot_duration.png", dpi=150)
plt.show()

# ── 8. Корреляционная матрица ────────────────────────────────
num_cols = ["release_year", "year_added", "duration_val", "genre_count"]
num_cols = [c for c in num_cols if c in df.columns]
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Reds", ax=ax)
ax.set_title("Корреляционная матрица")
plt.tight_layout()
plt.savefig("img/correlation_matrix.png", dpi=150)
plt.show()

print("\n✅ Все графики EDA сохранены в папку img/")
