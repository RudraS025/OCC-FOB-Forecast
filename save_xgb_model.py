import xgboost as xgb
import joblib
import pandas as pd
import numpy as np
import sqlite3

def create_lag_features(series, lags=[1, 12]):
    df_feat = pd.DataFrame({'y': series})
    for lag in lags:
        df_feat[f'lag_{lag}'] = series.shift(lag)
    df_feat = df_feat.dropna()
    return df_feat

# Load data from SQLite for consistency with app
conn = sqlite3.connect('occ_fob_data.db')
df = pd.read_sql_query('SELECT * FROM occ_fob', conn, parse_dates=['Date'])
conn.close()
df = df.set_index('Date').sort_index()

log_series = np.log(df['OCC_FOB_USD_ton'])
lags = [1, 12]
xgb_data = create_lag_features(log_series, lags)
train_size = int(len(log_series) * 0.85)
xgb_train = xgb_data.iloc[:train_size-lags[-1]]
X_train = xgb_train.drop('y', axis=1)
y_train = xgb_train['y']

# Train XGBoost model with CPU only
model_xgb = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, tree_method='hist', n_jobs=1, enable_categorical=False, predictor='auto', verbosity=0)
model_xgb.fit(X_train.values, y_train.values)

# Save the model using joblib (compatible with Heroku CPU XGBoost)
joblib.dump(model_xgb, 'xgboost_occ_fob_model.pkl')
print('Model retrained and saved for CPU-only XGBoost.')
