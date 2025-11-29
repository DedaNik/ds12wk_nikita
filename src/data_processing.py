import pandas as pd
import numpy as np

def load_data(path: str) -> pd.DataFrame:
    """Загружает CSV файл."""
    return pd.read_csv(path)

def clean_data(EDA: pd.DataFrame) -> pd.DataFrame:
    """Заменяет '?' на NaN и удаляет строки с пропусками."""
    EDA = EDA.copy()
    EDA.replace('?', np.nan, inplace=True)
    EDA.dropna(inplace=True)
    return EDA

def encode_categorical(EDA: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """One-hot кодирование указанных категориальных колонок."""
    return pd.get_dummies(EDA, columns=cols)

def encode_categorical_top_10(EDA: pd.DataFrame, target_col: str, top_n: int = 10) -> pd.DataFrame:
    """
    Кодирование категориального признака:
    - Оставляем top_n самых частых категорий
    - Остальные объединяем в 'Other'
    - Применяем One-hot
    """
    EDA = EDA.copy()

    # 1. Находим топ категорий
    top_categories = EDA[target_col].value_counts().nlargest(top_n).index

    # 2. Остальные → "Other"
    EDA[target_col] = EDA[target_col].apply(
        lambda x: x if x in top_categories else 'Other'
    )

    # 3. One-hot encoding
    EDA = pd.get_dummies(EDA, columns=[target_col])

    return EDA
