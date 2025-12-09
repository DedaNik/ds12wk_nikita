
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import mlflow

def data_separation(EDA: pd.DataFrame,y_name: str):
  X = EDA.drop(columns=[y_name],axis=1)
  y = EDA[y_name]
  return X, y

def train_catboost(X,y):
  
  skf = StratifiedKFold(n_splits=5, shuffle = True, random_state=42)
  auc_scores = []
  
  for fold, (train_idx, val_idx) in enumerate(skf.split(X,y),1):

    X_train, X_test = X.iloc[train_idx] , X.iloc[val_idx]
    y_train, y_test = y.iloc[train_idx] , y.iloc[val_idx]
        
    with mlflow.start_run(run_name=f"fold_{fold}"):
             
      # 1️⃣ Параметры модели
      params = {"iterations": 1000, "learning_rate": 0.1, "random_state": 42}   
      for k, v in params.items():
        mlflow.log_param(k, v)

    clf = CatBoostClassifier(**params, verbose=100)
    
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_pred_proba)
    
    mlflow.log_metric("AUC", auc)
    mlflow.catboost.log_model(clf, artifact_path="catboost_model")
    
    print(f"Fold {fold} AUC: {auc:.4f}")
    auc_scores.append(auc)
    clf.save_model(f'models/catboost_fold{fold}.cbm')

  results = pd.DataFrame({
    'Fold': list(range(1, 6)),
    'AUC': auc_scores
  })
  
  mean_row = pd.DataFrame([{'Fold':'Mean', 'AUC': np.mean(auc_scores)}])
  results = pd.concat([results, mean_row], ignore_index=True)
  
  print(results)
    
  return clf

