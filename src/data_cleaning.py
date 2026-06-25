import pandas as pd
import numpy as np


def load_and_clean(path="data/raw/netflix_titles.csv"):
    df = pd.read_csv(path)

    # Удаление дубликатов
    df.drop_duplicates(inplace=True)

    # Заполнение пропусков
    df["director"].fillna("Unknown", inplace=True)
    df["cast"].fillna("Unknown", inplace=True)
    df["country"].fillna(df["country"].mode()[0], inplace=True)
    df["rating"].fillna(df["rating"].mode()[0], inplace=True)
    df["duration"].fillna("0 min", inplace=True)
    df.dropna(subset=["date_added"], inplace=True)

    # Извлечение года добавления
    df["date_added"] = df["date_added"].str.strip()
    df["year_added"] = pd.to_datetime(df["date_added"], errors="coerce").dt.year
    df["month_added"] = pd.to_datetime(df["date_added"], errors="coerce").dt.month

    # Первая страна (некоторые записи содержат несколько стран)
    df["country_main"] = df["country"].apply(lambda x: str(x).split(",")[0].strip())

    # Числовая длительность
    df["duration_val"] = df["duration"].str.extract(r"(\d+)").astype(float)

    # Количество жанров
    df["genre_count"] = df["listed_in"].apply(lambda x: len(str(x).split(",")))

    df.to_csv("data/processed/netflix_cleaned.csv", index=False)
    print(f"✅ Очищено: {df.shape[0]} строк, {df.shape[1]} столбцов")
    return df


if __name__ == "__main__":
    load_and_clean()
