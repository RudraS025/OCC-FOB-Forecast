from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import numpy as np
import joblib
import os
import sqlite3
from datetime import datetime
from pandas.tseries.offsets import DateOffset
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

MODEL_PATH = 'xgboost_occ_fob_model.json'
DATA_PATH = 'Paper_TimeSeries.xlsx'
RESULTS_PATH = 'OCC_FOB_Results.xlsx'
DB_PATH = 'occ_fob_data.db'

# Load model and data
def load_model():
    booster = xgb.Booster()
    booster.load_model('xgboost_occ_fob_model.json')
    return booster

def init_db_from_excel():
    if not os.path.exists(DB_PATH):
        df = pd.read_excel(DATA_PATH)
        df.columns = ['Date', 'OCC_FOB_USD_ton']
        df['Date'] = pd.to_datetime(df['Date'])
        conn = sqlite3.connect(DB_PATH)
        df.to_sql('occ_fob', conn, if_exists='replace', index=False)
        conn.close()

def load_data():
    init_db_from_excel()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('SELECT * FROM occ_fob', conn, parse_dates=['Date'])
    conn.close()
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

def add_new_price(new_date, new_value):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('INSERT OR REPLACE INTO occ_fob (Date, OCC_FOB_USD_ton) VALUES (?, ?)', (new_date.strftime('%Y-%m-%d'), new_value))
    conn.commit()
    conn.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        df = load_data()
        log_series = np.log(df['OCC_FOB_USD_ton'])
        lags = [1, 12]
        model = load_model()
        forecast = None
        forecast_dates = None
        last_month = df.index[-1]
        if pd.isnull(last_month):
            last_month_str = "No Data"
        else:
            last_month_str = last_month.strftime('%B %Y')
        message = None
        if request.method == 'POST':
            try:
                new_value = float(request.form['new_value'])
                new_date = pd.to_datetime(request.form['new_date'])
                add_new_price(new_date, new_value)
                df = load_data()
                log_series = np.log(df['OCC_FOB_USD_ton'])
                # Update last_month and last_month_str after adding new value
                last_month = df.index.max()
                if pd.isnull(last_month):
                    last_month_str = "No Data"
                else:
                    last_month_str = last_month.strftime('%B %Y')
                message = f"Added {new_value} for {new_date.strftime('%B %Y')}!"
            except Exception as e:
                message = f"Error: {e}"
        # Always update last_month and last_month_str after any changes
        last_month = df.index.max()
        if pd.isnull(last_month):
            last_month_str = "No Data"
        else:
            last_month_str = last_month.strftime('%B %Y')
        # Prepare lags for forecast
        last_logs = list(np.log(df['OCC_FOB_USD_ton'])[-max(lags):])
        preds = []
        forecast_dates = []
        # Forecast always starts from the month AFTER the latest actual value
        forecast_start = (last_month + DateOffset(months=1)).replace(day=1)
        for i in range(6):
            features = [last_logs[-lag] for lag in lags]
            dtest = xgb.DMatrix(np.array(features).reshape(1, -1))
            pred_log = model.predict(dtest)[0]
            pred = invert_log_transform(pred_log)
            forecast_date = (forecast_start + DateOffset(months=i)).replace(day=1)
            preds.append(pred)
            forecast_dates.append(forecast_date)
            last_logs.append(pred_log)
        forecast = list(zip([d.strftime('%b %Y') if not pd.isnull(d) else "No Date" for d in forecast_dates], preds))

        # Prepare data for graph: last 4 months actual + 6 months forecast
        actual_dates = df.index[-4:]
        actual_values = df['OCC_FOB_USD_ton'][-4:]
        valid_forecast = [(d, v) for d, v in zip(forecast_dates, preds) if not pd.isnull(d)]
        if valid_forecast:
            forecast_months, forecast_values = zip(*valid_forecast)
        else:
            forecast_months, forecast_values = [], []
        plt.figure(figsize=(10,5))
        if len(actual_dates) > 0:
            plt.plot(actual_dates, actual_values, marker='o', label='Actual (last 4 months)')
        if len(forecast_months) > 0:
            plt.plot(forecast_months, forecast_values, marker='x', linestyle='--', label='Forecast (next 6 months)')
        plt.xlabel('Date')
        plt.ylabel('OCC FOB (USD/ton)')
        plt.title('OCC FOB (USD/ton): Last 4 Actual & Next 6 Forecasted')
        plt.legend()
        plt.tight_layout()
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return render_template('index.html', forecast=forecast, last_month=last_month_str, message=message, plot_url=plot_url)
    except Exception as e:
        return f"Internal Server Error: {e}", 500

@app.route('/download')
def download():
    try:
        df = load_data()
        lags = [1, 12]
        model = load_model()
        last_month = df.index.max()
        last_logs = list(np.log(df['OCC_FOB_USD_ton'])[-max(lags):])
        preds = []
        forecast_dates = []
        # Forecast always starts from the month AFTER the latest actual value
        forecast_start = (last_month + DateOffset(months=1)).replace(day=1)
        for i in range(6):
            features = [last_logs[-lag] for lag in lags]
            dtest = xgb.DMatrix(np.array(features).reshape(1, -1))
            pred_log = model.predict(dtest)[0]
            pred = invert_log_transform(pred_log)
            forecast_date = (forecast_start + DateOffset(months=i)).replace(day=1)
            preds.append(pred)
            forecast_dates.append(forecast_date)
            last_logs.append(pred_log)
        forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast_OCC_FOB_USD_ton': preds})
        forecast_df['Date'] = forecast_df['Date'].dt.strftime('%b %Y')
        actual_df = df.reset_index().rename(columns={'Date': 'Date', 'OCC_FOB_USD_ton': 'Actual_OCC_FOB_USD_ton'})
        actual_df['Date'] = actual_df['Date'].dt.strftime('%b %Y')
        with pd.ExcelWriter(RESULTS_PATH, engine='openpyxl') as writer:
            actual_df.to_excel(writer, sheet_name='Actual', index=False)
            forecast_df.to_excel(writer, sheet_name='Forecast', index=False)
        if os.path.exists(RESULTS_PATH):
            return send_file(RESULTS_PATH, as_attachment=True)
        return 'Results file not found.'
    except Exception as e:
        return f"Internal Server Error: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)
