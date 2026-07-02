# 🎬 Netflix Movies & TV Shows — Деректерді Визуализациялау Жобасы

> **Пән:** Деректерді визуализациялау | **Астана, 2026**

---

## 📌 Жоба сипаттамасы

Бұл жоба Netflix контент датасетін талдауға арналған. EDA, статистикалық талдау, интерактивті дашборд және ML классификация моделі арқылы контент тенденцияларын зерттейміз.

---

## 📦 Датасет

| Параметр | Мән |
|----------|-----|
| Атауы | Netflix Movies and TV Shows |
| Дереккөзі | [Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows) |
| Жол саны | ~8 800 |
| Белгілер саны | 12 |

---

## ❓ Зерттеу сұрақтары

1. Netflix-те Movie және TV Show арасындағы қатынас қандай?
2. Қай елдер ең көп контент шығарады?
3. Контент саны жылдар бойынша қалай өзгерді?
4. Рейтинг пен контент типі арасында байланыс бар ма?
5. ML моделі Movie/TV Show-ды қаншалықты дәл болжай алады?

---

## 🗂️ Жоба құрылымы

```
netflix-visualization/
│
├── data/
│   ├── raw/                  # Бастапқы датасет (Kaggle-дан)
│   └── processed/            # Тазаланған деректер, ML метрикалары
│
├── notebooks/
│   ├── 01_eda.py             # EDA және деректерді тазалау
│   └── 03_ml_model.py        # ML классификация моделі
│
├── src/
│   └── data_cleaning.py      # Деректерді тазалау функциялары
│
├── app/
│   └── streamlit_app.py      # Интерактивті дашборд
│
├── img/                      # Графиктер
├── README.md
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Қолданылған технологиялар

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pandas](https://img.shields.io/badge/pandas-2.0-green)
![Scikit--learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red)

- **Python 3.11+**
- pandas, numpy — деректермен жұмыс
- matplotlib, seaborn, plotly — визуализациялар
- scikit-learn — ML модельдер
- Streamlit — интерактивті дашборд

---

## 📊 Негізгі визуализациялар

| EDA | ML Нәтижелері |
|-----|--------------|
| ![EDA](img/type_distribution.png) | ![ML](img/model_results.png) |

---

## 🚀 Жобаны іске қосу

```bash
# 1. Репозиторийді клондау
git clone https://github.com/batyrjackson/netflix-visualization.git
cd netflix-visualization

# 2. Тәуелділіктерді орнату
pip3 install -r requirements.txt

# 3. Файлдарды іске қосу (ретімен)
python src/data_cleaning.py
python notebooks/01_eda.py
python notebooks/03_ml_model.py

# 4. Дашбордты іске қосу
streamlit run app/streamlit_app.py
```

---

## 🤖 ML Модель нәтижелері

| Параметр | Мән |
|----------|-----|
| Міндет | Классификация (Movie vs TV Show) |
| Алгоритм | Random Forest Classifier |
| Белгілер | release_year, rating, country, genre_count, duration |
| Train / Test бөлу | 80% / 20% |

### Random Forest Classifier

| Метрика | Мән |
|---------|-----|
| ✅ Accuracy | ~0.86 |
| 🎯 Precision | ~0.85 |
| 🔍 Recall | ~0.86 |
| ⚖️ F1-Score | ~0.85 |

### Logistic Regression (салыстыру үшін)

| Метрика | Мән |
|---------|-----|
| ✅ Accuracy | ~0.79 |
| 🎯 Precision | ~0.78 |
| 🔍 Recall | ~0.79 |
| ⚖️ F1-Score | ~0.78 |

> **Қорытынды:** Random Forest моделі Logistic Regression-ға қарағанда ~7% жоғары дәлдік көрсетті. Ең маңызды белгілер: `duration_val` және `rating_enc`.

---

## 🔢 Математикалық талдау

### Негізгі статистика

| Көрсеткіш | release_year | year_added | duration_val |
|-----------|-------------|------------|--------------|
| Орташа (Mean) | 2014.3 | 2018.6 | 89.4 |
| Медиана (Median) | 2017.0 | 2019.0 | 98.0 |
| Стд. ауытқу (Std) | 8.7 | 1.8 | 31.2 |
| Мин (Min) | 1925 | 2008 | 1 |
| Макс (Max) | 2021 | 2021 | 312 |
| IQR | 9.0 | 3.0 | 52.0 |

### Корреляциялық талдау

| | release_year | year_added | duration_val | genre_count |
|--|-------------|------------|--------------|-------------|
| **release_year** | 1.00 | 0.71 | 0.08 | 0.03 |
| **year_added** | 0.71 | 1.00 | 0.05 | 0.02 |
| **duration_val** | 0.08 | 0.05 | 1.00 | -0.12 |
| **genre_count** | 0.03 | 0.02 | -0.12 | 1.00 |

> **Түсініктеме:** `release_year` мен `year_added` арасындағы корреляция жоғары (0.71) — жаңа фильмдер тезірек Netflix-ке қосылады.

---

## 📈 Негізгі қорытындылар

- Netflix контентінің **~70%** — Movie, **~30%** — TV Show
- Ең көп контент шығаратын ел: **АҚШ**
- 2018–2020 жылдары контент саны күрт өсті
- Random Forest моделі **~86%** дәлдікпен Movie/TV Show-ды ажыратады
- `duration_val` және `rating_enc` — ең маңызды белгілер

---

## 👤 Автор

**Аты-жөні:** Набиев Батырбек  
**GitHub:** [@batyrjackson](https://github.com/batyrjackson)
