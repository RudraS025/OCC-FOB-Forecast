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
import stat

app = Flask(__name__)

MODEL_PATH = 'xgboost_occ_fob_model.json'
DATA_PATH = 'Paper_TimeSeries.xlsx'
RESULTS_PATH = 'OCC_FOB_Results.xlsx'
DB_PATH = '/data/occ_fob_data.db'  # Change to 'occ_fob_data.db' for local dev if needed

# Load model and data
def load_model():
    booster = xgb.Booster()
    booster.load_model('xgboost_occ_fob_model.json')
    return booster

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('PRAGMA journal_mode=WAL;')
    conn.execute('PRAGMA synchronous=NORMAL;')
    return conn

def ensure_table_schema():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS occ_fob (
        Date TEXT PRIMARY KEY,
        OCC_FOB_USD_ton REAL
    )''')
    conn.commit()
    conn.close()

def init_db_from_excel():
    if not os.path.exists(DB_PATH):
        df = pd.read_excel(DATA_PATH)
        df.columns = ['Date', 'OCC_FOB_USD_ton']
        df['Date'] = pd.to_datetime(df['Date'])
        ensure_table_schema()
        conn = get_db_connection()
        for _, row in df.iterrows():
            conn.execute('INSERT OR REPLACE INTO occ_fob (Date, OCC_FOB_USD_ton) VALUES (?, ?)', (row['Date'].strftime('%Y-%m-%d'), row['OCC_FOB_USD_ton']))
        conn.commit()
        conn.close()

def load_data():
    # Only initialize DB from Excel if DB does not exist (prevents overwriting user data)
    if not os.path.exists(DB_PATH):
        init_db_from_excel()
    ensure_table_schema()
    conn = get_db_connection()
    df = pd.read_sql_query('SELECT * FROM occ_fob', conn, parse_dates=['Date'])
    conn.close()
    # Drop rows with NaT or null dates
    df = df[df['Date'].notnull()]
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
    print(f"[DEBUG] add_new_price called with new_date={new_date} (type: {type(new_date)}), new_value={new_value}")
    if pd.isnull(new_date) or not isinstance(new_date, pd.Timestamp):
        print(f"[DEBUG] Invalid new_date: {new_date}")
        raise ValueError(f"Invalid date: {new_date}")
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        # Print all rows with this date for debugging
        cur.execute('SELECT * FROM occ_fob WHERE Date = ?', (new_date.strftime('%Y-%m-%d'),))
        rows = cur.fetchall()
        print(f"[DEBUG] Rows with Date={new_date.strftime('%Y-%m-%d')}: {rows}")
        # If any exist, forcibly delete them
        if rows:
            print(f"[DEBUG] Forcibly deleting all rows with Date={new_date.strftime('%Y-%m-%d')}")
            cur.execute('DELETE FROM occ_fob WHERE Date = ?', (new_date.strftime('%Y-%m-%d'),))
            conn.commit()
        # Now insert the new value
        cur.execute('INSERT INTO occ_fob (Date, OCC_FOB_USD_ton) VALUES (?, ?)', (new_date.strftime('%Y-%m-%d'), new_value))
        conn.commit()
        conn.close()
        # Only clean up rows with invalid dates, not all others
        cleanup_db()
        log_all_dates()
    except Exception as e:
        print(f"[DEBUG] DB write error: {e}")
        raise

def cleanup_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        # Remove rows where Date is NULL, empty, or OCC_FOB_USD_ton is NULL
        cur.execute("DELETE FROM occ_fob WHERE Date IS NULL OR Date = '' OR OCC_FOB_USD_ton IS NULL")
        conn.commit()
        # Remove rows where Date cannot be parsed as a valid date or not in YYYY-MM-DD format
        df = pd.read_sql_query('SELECT Date FROM occ_fob', conn)
        for d in df['Date']:
            try:
                dt = pd.to_datetime(d, errors='raise')
                # Only keep if string matches YYYY-MM-DD
                if not (isinstance(d, str) and len(d) == 10 and d[4] == '-' and d[7] == '-'): 
                    raise ValueError('Bad format')
            except Exception:
                cur.execute('DELETE FROM occ_fob WHERE Date = ? OR Date IS NULL OR Date = ""', (d if d is not None else '',))
        conn.commit()
        # Vacuum DB to clean up deleted rows
        cur.execute('VACUUM')
        conn.commit()
        # Print all dates and their types for debugging
        df2 = pd.read_sql_query('SELECT Date FROM occ_fob ORDER BY Date', conn)
        print("[DEBUG] Dates/types after cleanup:", [(d, type(d)) for d in df2['Date']])
        conn.close()
        print("[DEBUG] DB cleaned of invalid dates.")
    except Exception as e:
        print(f"[DEBUG] Error cleaning DB: {e}")

# Utility: Log all dates in DB for debugging
def log_all_dates():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query('SELECT Date FROM occ_fob ORDER BY Date', conn, parse_dates=['Date'])
        conn.close()
        print(f"[DEBUG] All dates in DB after insert: {list(df['Date'])}")
    except Exception as e:
        print(f"[DEBUG] Error reading all dates: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        df = load_data()
        log_series = np.log(df['OCC_FOB_USD_ton'])
        lags = [1, 12]
        model = load_model()
        forecast = None
        forecast_dates = None
        last_month = df.index[-1] if len(df.index) > 0 else None
        if pd.isnull(last_month):
            last_month_str = "No Data"
        else:
            last_month_str = last_month.strftime('%B %Y')
        message = None
        if request.method == 'POST':
            try:
                print(f"[DEBUG] POST form: {request.form}")
                new_value = float(request.form['new_value'])
                new_date = pd.to_datetime(request.form['new_date'], errors='coerce')
                add_new_price(new_date, new_value)
                # Force reload from DB after adding new value
                df = load_data()
                log_series = np.log(df['OCC_FOB_USD_ton'])
                last_month = df.index.max() if len(df.index) > 0 else None
                if pd.isnull(last_month):
                    last_month_str = "No Data"
                else:
                    last_month_str = last_month.strftime('%B %Y')
                message = f"Added {new_value} for {new_date.strftime('%B %Y')}!"
                print(f"[DEBUG] After add: Latest date in DB: {last_month}")
            except Exception as e:
                message = f"Error: {e}"
        # Always update last_month and last_month_str after any changes
        last_month = df.index.max() if len(df.index) > 0 else None
        if pd.isnull(last_month):
            last_month_str = "No Data"
        else:
            last_month_str = last_month.strftime('%B %Y')
        # Prepare lags for forecast
        last_logs = list(np.log(df['OCC_FOB_USD_ton'])[-max(lags):])
        if len(last_logs) < max(lags):
            message = (message or "") + " Not enough data to forecast. Please ensure at least 12 months of data are present."
            forecast = []
            plot_url = None
            return render_template('index.html', forecast=forecast, last_month=last_month_str, message=message, plot_url=plot_url)
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

@app.route('/db_status')
def db_status():
    try:
        exists = os.path.exists(DB_PATH)
        try:
            st = os.stat(DB_PATH)
            perms = stat.filemode(st.st_mode)
            owner = st.st_uid
            size = st.st_size
        except Exception as e:
            perms = f"Error: {e}"
            owner = 'N/A'
            size = 'N/A'
        try:
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql_query('SELECT Date FROM occ_fob ORDER BY Date', conn, parse_dates=['Date'])
            conn.close()
            dates = [str(d) for d in df['Date']]
        except Exception as e:
            dates = [f"Error: {e}"]
        return f"DB_PATH: {DB_PATH}<br>Exists: {exists}<br>Perms: {perms}<br>Owner: {owner}<br>Size: {size}<br>Dates in DB:<br>{'<br>'.join(dates)}"
    except Exception as e:
        return f"Error in /db_status: {e}", 500

# --- Admin route to reset DB from Excel ---
@app.route('/admin/reset_db')
def admin_reset_db():
    import os
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    init_db_from_excel()
    return redirect(url_for('db_status'))

# NOTE: On Render, use a persistent disk for occ_fob_data.db or switch to a managed DB for true persistence.

# At app startup, print DB path and permissions
print(f"[DEBUG] App startup. DB path: {DB_PATH}")
try:
    st = os.stat(DB_PATH)
    print(f"[DEBUG] DB path: {DB_PATH}, mode: {oct(st.st_mode)}, owner: {st.st_uid}, perms: {stat.filemode(st.st_mode)}")
except Exception as e:
    print(f"[DEBUG] Could not stat DB file at startup: {e}")

if __name__ == '__main__':
    app.run(debug=True)
