
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def data_separation(EDA: pd.DataFrame,y_name: str):
  X = EDA.drop(columns=[y_name],axis=1)
  y = EDA[y_name]
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
  return X_train, X_test, y_train, y_test

def train_catboost(X_train, X_test, y_train, y_test):
  clf = CatBoostClassifier(
     verbose=100,   # выводит прогресс каждые 100 итераций
    random_state=42
  )

  clf.fit(X_train, y_train,)
  y_pred_proba = clf.predict_proba(X_test)[:, 1]
  
  auc = roc_auc_score(y_test, y_pred_proba)
  print(f"AUC: {auc:.4f}")

  clf.save_model('models/catboost_baseline.cbm')
  
  return clf

