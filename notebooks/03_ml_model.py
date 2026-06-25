import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)
import warnings
warnings.filterwarnings("ignore")

os.makedirs("img", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ── 1. Загрузка ──────────────────────────────────────────────
df = pd.read_csv("data/processed/netflix_cleaned.csv")

# ── 2. Feature Engineering ───────────────────────────────────
df["target"] = (df["type"] == "Movie").astype(int)

rating_order = {
    "G": 1, "TV-Y": 1, "TV-Y7": 2, "TV-Y7-FV": 2,
    "PG": 3, "TV-PG": 3, "PG-13": 4, "TV-14": 4,
    "R": 5, "TV-MA": 5, "NC-17": 6, "NR": 0, "UR": 0
}
df["rating_enc"] = df["rating"].map(rating_order).fillna(0)

top10 = df["country_main"].value_counts().nlargest(10).index
df["country_clean"] = df["country_main"].apply(lambda x: x if x in top10 else "Other")
le = LabelEncoder()
df["country_enc"] = le.fit_transform(df["country_clean"])

features = ["release_year", "year_added", "rating_enc",
            "country_enc", "genre_count", "duration_val"]
features = [f for f in features if f in df.columns]

X = df[features].fillna(0)
y = df["target"]

# ── 3. Train / Test split ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

# ── 4. Обучение моделей ───────────────────────────────────────
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

lr = LogisticRegression(max_iter=500, random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# ── 5. Метрики ────────────────────────────────────────────────
def metrics(y_true, y_pred):
    return {
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, average="weighted"), 4),
        "Recall":    round(recall_score(y_true, y_pred, average="weighted"), 4),
        "F1-Score":  round(f1_score(y_true, y_pred, average="weighted"), 4),
    }

rf_m = metrics(y_test, rf_pred)
lr_m = metrics(y_test, lr_pred)

print("\n📊 Random Forest:", rf_m)
print("📊 Logistic Regression:", lr_m)
print("\n", classification_report(y_test, rf_pred, target_names=["TV Show", "Movie"]))

# Сохранение метрик
results = pd.DataFrame({
    "Метрика": list(rf_m.keys()),
    "Random Forest": list(rf_m.values()),
    "Logistic Regression": list(lr_m.values()),
})
results.to_csv("data/processed/ml_metrics.csv", index=False)

# ── 6. Визуализации ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Netflix ML — Результаты классификации", fontsize=14, fontweight="bold")

# Confusion Matrix
cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=axes[0],
            xticklabels=["TV Show", "Movie"],
            yticklabels=["TV Show", "Movie"])
axes[0].set_title("Confusion Matrix (Random Forest)")
axes[0].set_xlabel("Предсказано")
axes[0].set_ylabel("Реально")

# Feature Importance
fi = pd.Series(rf.feature_importances_, index=features).sort_values()
fi.plot(kind="barh", ax=axes[1], color="#E50914")
axes[1].set_title("Feature Importance")
axes[1].set_xlabel("Важность")

# Сравнение моделей
x = np.arange(len(rf_m))
w = 0.35
axes[2].bar(x - w/2, rf_m.values(), w, label="Random Forest", color="#E50914")
axes[2].bar(x + w/2, lr_m.values(), w, label="Logistic Regression", color="#564d4d")
axes[2].set_xticks(x)
axes[2].set_xticklabels(rf_m.keys())
axes[2].set_ylim(0, 1.1)
axes[2].set_title("Сравнение моделей")
axes[2].legend()
axes[2].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("img/model_results.png", dpi=150, bbox_inches="tight")
print("\n✅ Сохранено: img/model_results.png")
plt.show()

# Feature importance CSV
fi_df = fi.reset_index()
fi_df.columns = ["Признак", "Важность"]
fi_df.to_csv("data/processed/feature_importance.csv", index=False)
print("✅ Сохранено: data/processed/feature_importance.csv")
