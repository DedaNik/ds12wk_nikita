import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

def  target_encode_oof(df:pd.DataFrame,col_cat,target,n_splits):

  kf = KFold(n_splits = n_splits, shuffle=True, random_state=42)

  df = df.copy()

  col = col_cat+"_te"

  df[col]= np.nan
  
  for fold, (train_indx, val_indx) in enumerate(kf.split(df)):
  
    train_df = df.iloc[train_indx]
    val_df = df.iloc[val_indx]
  
    means = train_df.groupby(col_cat)[target].mean()
  
    global_mean = train_df[target].mean()
  
    train_enc = train_df[col_cat].map(means)
    val_enc = val_df[col_cat].map(means).fillna(global_mean)
  
    train_labels = df.index[train_indx]
    val_labels = df.index[val_indx]
  
    df.loc[train_labels, col] = train_enc.values
    df.loc[val_labels, col] = val_enc.values
  return df
