def load_data(path: str) -> pd.DataFrame:
  EDA = pd.read_csv(path)
  EDA.head()
  return EDA

def clean_data(EDA: pd.DataFrame) -> pd.DataFrame:
  EDA.replace('?',np.nan,inplace=True)
  EDA.dropna(inplace=True)
  return EDA

def encode_categorical(EDA: pd.DataFrame,cols: list[str]) -> pd.DataFrame:
  return pd.get_dummies(EDA, columns=cols)

def encode_categorical_top_10(EDA: pd.DataFrame,target_col: str) -> pd.DataFrame:
  # 1️⃣ Находим топ-N самых частых категорий
  top_categories = EDA[target_col].value_counts().nlargest(top_n).index
  # 2️⃣ Заменяем все остальные на "Other"
  EDA[target_col] = EDA[target_col].apply(lambda x: x if x in top_categories else 'Other')
  EDA = pd.get_dummies(EDA, target_col)
  return EDA

