from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from pandas.tseries.offsets import DateOffset

app = Flask(__name__)

MODEL_PATH = 'xgboost_occ_fob_model.pkl'
DATA_PATH = 'Paper_TimeSeries.xlsx'
RESULTS_PATH = 'OCC_FOB_Results.xlsx'

# Load model and data
def load_model():
    return joblib.load(MODEL_PATH)

def load_data():
    df = pd.read_excel(DATA_PATH)
    df.columns = ['Date', 'OCC_FOB_USD_ton']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    return df

def invert_log_transform(log_preds):
    return np.exp(log_preds)

def create_lag_features(series, lags=[1, 12]):
    df_feat = pd.DataFrame({'y': series})
    for lag in lags:
        df_feat[f'lag_{lag}'] = series.shift(lag)
    df_feat = df_feat.dropna()
    return df_feat

@app.route('/', methods=['GET', 'POST'])
def index():
    df = load_data()
    log_series = np.log(df['OCC_FOB_USD_ton'])
    lags = [1, 12]
    model = load_model()
    forecast = None
    forecast_dates = None
    last_month = df.index[-1].strftime('%B %Y')
    message = None
    if request.method == 'POST':
        try:
            new_value = float(request.form['new_value'])
            new_date = pd.to_datetime(request.form['new_date'])
            # Update data
            df.loc[new_date] = new_value
            df = df.sort_index()
            df.to_excel(DATA_PATH)
            log_series = np.log(df['OCC_FOB_USD_ton'])
            # Save updated data
            message = f"Added {new_value} for {new_date.strftime('%B %Y')}!"
        except Exception as e:
            message = f"Error: {e}"
    # Prepare lags for forecast
    last_logs = list(np.log(df['OCC_FOB_USD_ton'])[-max(lags):])
    preds = []
    for i in range(6):
        features = [last_logs[-lag] for lag in lags]
        pred_log = model.predict(np.array(features).reshape(1, -1))[0]
        pred = invert_log_transform(pred_log)
        preds.append(pred)
        last_logs.append(pred_log)
    forecast_dates = [df.index[-1] + DateOffset(months=i+1) for i in range(6)]
    forecast = list(zip([d.strftime('%b %Y') for d in forecast_dates], preds))
    return render_template('index.html', forecast=forecast, last_month=last_month, message=message)

@app.route('/download')
def download():
    if os.path.exists(RESULTS_PATH):
        return send_file(RESULTS_PATH, as_attachment=True)
    return 'Results file not found.'

if __name__ == '__main__':
    app.run(debug=True)
