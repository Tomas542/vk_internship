import json
import pickle

import numpy as np
import pandas as pd

import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')

def feat(df1: pd.DataFrame, df2: pd.DataFrame, features: list[int]) -> pd.DataFrame:
    geo1 = np.atleast_2d(df1[['lat', 'lon']]).reshape(1,-1,2)
    geo2 = np.atleast_2d(df2[['lat', 'lon']]).reshape(-1,1,2)

    p = np.power(geo1-geo2, 2).sum(axis=2).argmin(axis = 0) # рассчитываем евклидово расстояние без корня

    df1.loc[:, features] = df2.loc[p, features].reset_index(drop=True)
    
    return df1

def scale(sc, df:pd.DataFrame, features: list[int]) -> pd.DataFrame:
    df_scaled = pd.DataFrame(sc.transform(df[features]))
    df_scaled.loc[:, ['lon', 'lat']] = df.loc[:, ['lon', 'lat']].reset_index(drop=True)
    return df_scaled

# считываем конфиг для LightGBM
with open('./config/lgb.conf', "r") as f_json:
    lgb_config = json.loads(f_json.read())

# считываем параметры для скейлера
with open('./config/scaler.pkl','rb') as f:
    sc = pickle.load(f)

# считывание данных
train_df = pd.read_csv('./datasets/train.csv')
test_df = pd.read_csv('./datasets/test.csv')
features_df = pd.read_csv('./datasets/features.csv')

features = list((str(x) for x in range(363)))

# размещение фич в train и test
train_features = feat(train_df, features_df, features)
test_features = feat(test_df, features_df, features)

# скейлинг фичей
train_scaled = scale(sc, train_features, features)
test_scaled = scale(sc, test_features, features)

# расчёт предсказаний
lgbr = lgb.LGBMRegressor(**lgb_config)
lgbr.fit(train_scaled, train_df['score'])
preds = lgbr.predict(test_scaled)

# создание серии score и конкатенации с id из test
score = pd.Series(preds, name='score')
pd.concat([test_df.loc[:, 'id'], score], axis=1).to_csv('./datasets/submission.csv', index=False, header=True)