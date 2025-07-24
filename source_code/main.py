import sys, os

# (optional, but good if you're double-checking path priorities)
this_dir = os.path.dirname(__file__) or '.'
sys.path.insert(0, os.path.join(this_dir, 'python_libs/lib/python3.8/site-packages'))

import os
import tempfile
import atexit
import shutil

from flask import Flask, render_template, request, send_file, session, redirect, url_for
from flask_session import Session
import pandas as pd
import io, csv, base64, zipfile, socket, webbrowser
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from threading import Timer
from waitress import serve


REQUIRED_COLUMNS = ['Altitude', 'Temperature', 'Pressure', 'Humidity', 'Heading', 'Speed']

ISRO_COLUMN_MAP = {
    'Altitude(m)': 'Altitude',
    'Temp ext(c)': 'Temperature',
    'Pressure(mbar)': 'Pressure',
    'Humidity(%rh)': 'Humidity',
    'Dir(?)': 'Heading',
    'Speed(m/s)': 'Speed'
}
ISRO_COLUMN_LETTERS = {
    'Altitude':    'C',
    'Temperature': 'D',
    'Pressure':    'F',
    'Humidity':    'G',
    # 'Speed':       'N',   # not "WindSpeed"
    # 'Heading':     'O',   # not "WindDirection"
    'Speed': 'O',     # Picks 6.2
    'Heading': 'N'    # Picks 250.0

}

KOMOLINE_COLUMN_MAP = {
    'Temperature': 'Temperature',
    'Humidity': 'Humidity',
    'Pressure': 'Pressure',
    'Altitude': 'Altitude',
    'Heading': 'Heading',
    'Speed': 'Speed'
}

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Use a writable temporary session directory
SESSION_DIR = os.path.join(tempfile.gettempdir(), "atmo_sessions")
os.makedirs(SESSION_DIR, exist_ok=True)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = SESSION_DIR
app.config['SESSION_COOKIE_NAME'] = 'session'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_FILE_THRESHOLD'] = 500
Session(app)

APP_VERSION = "v4.0"
DEVELOPER = {"team": "MET Grp-3", "organization": "DRDO"}

def map_columns(df, dataset_type):
    col_map = ISRO_COLUMN_MAP if dataset_type == 'ISRO' else KOMOLINE_COLUMN_MAP
    renamed = {}
    col_map_lower = {k.strip().lower(): v for k, v in col_map.items()}

    print("📋 Original columns:", df.columns.tolist())  # Debugging

    for col in df.columns:
        normalized = col.strip().lower()
        if normalized in col_map_lower:
            renamed[col] = col_map_lower[normalized]

    df = df.rename(columns=renamed)

    print("✅ Renamed columns:", df.columns.tolist())  # Debugging
    return df
# def safe_read_csv(file_stream):
#     content = file_stream.read()
#     try:
#         dialect = csv.Sniffer().sniff(content.decode('utf-8'), delimiters=',;')
#         file_stream.seek(0)
#         return pd.read_csv(io.StringIO(content.decode('utf-8')), delimiter=dialect.delimiter)
#     except Exception:
#         file_stream.seek(0)
#         return pd.read_csv(file_stream)
def safe_read_csv(file_stream, dataset_type=None):
    content = file_stream.read().decode('utf-8', errors='replace')
    file_stream.seek(0)

    if dataset_type == 'ISRO':
        try:
            raw = pd.read_csv(
                io.StringIO(content),
                skiprows=53,
                header=None,
                on_bad_lines='skip'  # Uses default engine='c'
            )

            expected_indices = [ord(letter.upper()) - ord('A') for letter in ISRO_COLUMN_LETTERS.values()]
            if raw.shape[1] < max(expected_indices) + 1:
                print("[ERROR] safe_read_csv ISRO: Not enough columns. Got", raw.shape[1])
                print("First few rows:\n", raw.head())
                return None

            idx_map = {
                field: ord(letter.upper()) - ord('A')
                for field, letter in ISRO_COLUMN_LETTERS.items()
            }

            selected = raw.iloc[:, list(idx_map.values())].copy()
            selected.columns = REQUIRED_COLUMNS
            print("[INFO] safe_read_csv ISRO: Successfully parsed with shape", selected.shape)
            return selected

        except Exception as e:
            print("[ERROR] safe_read_csv ISRO exception:", e)
            return None

    # --- Komoline or other fallback ---
    try:
        # Attempt to detect delimiter
        dialect = csv.Sniffer().sniff(content, delimiters=',;')
        delimiter = dialect.delimiter
        df = pd.read_csv(io.StringIO(content), delimiter=delimiter, on_bad_lines='skip')
        print(f"[INFO] safe_read_csv fallback: Loaded with shape {df.shape} using delimiter '{delimiter}'")
        return df

    except Exception as e:
        print("[ERROR] safe_read_csv fallback exception:", e)
        return None

def create_analysis_pdf(df):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        for col, color, label in [('Temperature', 'red', 'Temperature (K)'),
                                  ('Pressure', 'blue', 'Pressure (hPa)'),
                                  ('Humidity', 'green', 'Humidity (%)'),
                                  ('Speed', 'purple', 'Wind Speed (m/s)'),
                                  ('Heading', 'orange', 'Wind Direction (°)')]:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(df[col], df['Altitude'], color=color)
            ax.set_title(f"{label} vs Altitude")
            ax.set_xlabel(label)
            ax.set_ylabel("Altitude (m)")
            ax.grid(True)
            pdf.savefig(fig)
            plt.close(fig)

        try:
            wind_df = df[['Heading', 'Speed']].dropna()
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, polar=True)
            wind_df['DirBin'] = pd.cut(wind_df['Heading'], bins=np.arange(0, 361, 30),
                                       labels=np.arange(15, 360, 30), include_lowest=True)
            rose_data = df.groupby('DirBin', observed=True).size()
            if not rose_data.empty:
                theta = np.radians(rose_data.index.astype(float))
                radii = rose_data.values
                colors = plt.cm.viridis(radii / max(radii))
                ax.bar(theta, radii, width=np.radians(30), color=colors, edgecolor='black')
                ax.set_theta_zero_location('N')
                ax.set_theta_direction(-1)
                ax.set_title("Wind Rose", va='bottom', y=1.1)
                pdf.savefig(fig)
            plt.close(fig)
        except Exception as e:
            print(f"Error adding wind rose to PDF: {e}")

    buf.seek(0)
    return buf

def create_wind_rose(df):
    try:
        df = df[['WindDirection_deg', 'WindSpeed_knots']].copy()
        df.rename(columns={'WindDirection_deg': 'Heading', 'WindSpeed_knots': 'Speed'}, inplace=True)
        df['Heading'] = pd.to_numeric(df['Heading'], errors='coerce')
        df['Speed'] = pd.to_numeric(df['Speed'], errors='coerce')
        df = df.dropna()

        if df.empty:
            return None

        fig = plt.figure(figsize=(6, 6), facecolor='none')
        ax = fig.add_subplot(111, polar=True, facecolor='none')
        df['DirBin'] = pd.cut(df['Heading'], bins=np.arange(0, 361, 30),
                              labels=np.arange(15, 360, 30), include_lowest=True)
        rose_data = df.groupby('DirBin', observed=True).size()

        if rose_data.empty:
            return None

        theta = np.radians(rose_data.index.astype(float))
        radii = rose_data.values
        colors = plt.cm.viridis(radii / max(radii))
        ax.bar(theta, radii, width=np.radians(30), color=colors, edgecolor='black')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_title("Wind Rose (Frequency by Direction)", va='bottom', y=1.1)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return image_base64

    except Exception as e:
        print(f"Wind Rose generation failed: {e}")
        return None

# Remaining Flask route logic will be added below here (index, download_csv, download_zip, clear_session, etc)
@app.route('/clear')
def clear_session():
    session.clear()
    return redirect(url_for('index'))

@app.route('/download/csv')
def download_csv():
    if 'result' not in session:
        return "No processed data available to download.", 400

    csv_data = session['result']
    return send_file(
        io.BytesIO(csv_data.encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='processed_data.csv'
    )

@app.route('/download/zip')
def download_zip():
    if 'result' not in session and 'csv_data' not in session:
        return "No data available to download.", 400

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        if 'result' in session:
            zipf.writestr("processed_data.csv", session['result'])
        if 'csv_data' in session:
            for filename, csv_str in session['csv_data'].items():
                try:
                    df = pd.read_csv(io.StringIO(csv_str))
                    if not set(REQUIRED_COLUMNS).issubset(df.columns):
                        continue
                    pdf = create_analysis_pdf(df)
                    zipf.writestr(f"analysis_{filename.rsplit('.',1)[0]}.pdf", pdf.read())
                except Exception as e:
                    print(f"Error with {filename}: {e}")

    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name='processed_output.zip'
    )

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def open_browser(port):
    webbrowser.open_new(f"http://127.0.0.1:{port}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        dataset_type = request.form.get('dataset_type')
        auto_generate = request.form.get('auto_generate')  # "1" if auto button clicked
        altitude_step = request.form.get('altitude', type=float)
        uploaded_files = request.files.getlist('files')

        if not uploaded_files or not dataset_type:
            return render_template('index.html', error="Please upload files and select dataset type.")

        dataframes = []
        csv_data = {}

        for file in uploaded_files:
            # Pass dataset_type so safe_read_csv can handle ISRO specially
            df = safe_read_csv(file.stream, dataset_type=dataset_type)
            if df is None:
                return render_template('index.html', error="Failed to parse CSV. Please check file format.")

            # Only run map_columns for non-ISRO (ISRO columns are already set)
            if dataset_type != 'ISRO':
                df = map_columns(df, dataset_type)

            # ——— CLEAN & NORMALIZE ———
            for col in REQUIRED_COLUMNS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=REQUIRED_COLUMNS)
            df = df[df['Altitude'] > 0]
            df = df.sort_values(by='Altitude').reset_index(drop=True)
            if dataset_type == 'Komoline' and 'Speed' in df.columns:
                df['Speed'] = df['Speed'] / 2
            # ————————————————


            # ✅ Only accept if required columns present
            if set(REQUIRED_COLUMNS).issubset(df.columns):
                dataframes.append((file.filename, df))
                csv_data[file.filename] = df.to_csv(index=False)
                if df.empty:
                    continue


        if not dataframes:
            return render_template('index.html', error="No valid CSV files found after processing.")

        result_rows = []

        if auto_generate == "1":
            all_data = pd.concat([df for _, df in dataframes])
            if all_data.empty or all_data['Altitude'].dropna().empty:
                return render_template('index.html', error="All data files are empty or contain invalid altitude values.")

            max_alt = all_data['Altitude'].max()
            if pd.isna(max_alt):
                return render_template('index.html', error="Altitude data is invalid or missing.")
            max_alt = int(max_alt)
            altitudes = list(range(0, 1001, 100)) + list(range(1500, max_alt + 1, 500))

            for fname, df in dataframes:
                for input_alt in altitudes:
                    nearest = df.iloc[(df['Altitude'] - input_alt).abs().argsort()[:1]]
                    matched_alt = nearest['Altitude'].values[0]
                    if abs(matched_alt - input_alt) <= 99:
                        matched = matched_alt
                    else:
                        matched = 0
                    result_rows.append({
                        'Dataset': fname,
                        'Input_Altitude': input_alt,
                        'Matched_Altitude_m': matched,
                        'Temperature_K': round(nearest['Temperature'].values[0], 1),
                        'Pressure_hPa': int(round(val)) if (val := nearest['Pressure'].values[0]) % 1 == 0 else round(val, 1),
                        'Humidity_percent': int(round(nearest['Humidity'].values[0])),
                        'WindSpeed_mps': round(nearest['Speed'].values[0], 1),
                        'WindSpeed_knots': round(nearest['Speed'].values[0] * 1.94384, 1),
                        'WindDirection_deg': int(round(nearest['Heading'].values[0]))
                    })

        else:
            if altitude_step is None or altitude_step <= 0:
                return render_template('index.html', error="Please provide a valid altitude step.")
            for fname, df in dataframes:
                max_alt = df['Altitude'].max()
                altitudes = list(np.arange(0, max_alt + altitude_step, altitude_step))

                for input_alt in altitudes:
                    nearest = df.iloc[(df['Altitude'] - input_alt).abs().argsort()[:1]]

                    if nearest.empty:
                        matched = 0
                        temperature = pressure = humidity = speed = heading = 0
                    else:
                        matched = nearest['Altitude'].values[0]
                        temperature = nearest['Temperature'].values[0]
                        pressure = nearest['Pressure'].values[0]
                        humidity = nearest['Humidity'].values[0]
                        speed = nearest['Speed'].values[0]
                        heading = nearest['Heading'].values[0]

                    result_rows.append({
                        'Dataset': fname,
                        'Input_Altitude': input_alt,
                        'Matched_Altitude_m': matched,
                        'Temperature_K': round(temperature, 1),
                        'Pressure_hPa': int(round(pressure)) if pressure % 1 == 0 else round(pressure, 1),
                        'Humidity_percent': int(round(humidity)),
                        'WindSpeed_mps': round(speed, 1),
                        'WindSpeed_knots': round(speed * 1.94384, 1),
                        'WindDirection_deg': int(round(heading))
                    })


        result_df = pd.DataFrame(result_rows)

        # ✅ Convert DataFrame to pure Python types
        result_records = result_df.astype(object).to_dict(orient='records')
        print(result_records[0])

        for col in ['Temperature_K', 'WindSpeed_mps', 'WindSpeed_knots']:
            result_df[col] = result_df[col].map(lambda x: f"{x:.1f}")

        for col in ['Pressure_hPa']:
            result_df[col] = result_df[col].map(lambda x: str(int(x)) if float(x).is_integer() else f"{x:.1f}")

        for col in ['Humidity_percent', 'WindDirection_deg']:
            result_df[col] = result_df[col].map(lambda x: str(int(round(x))))

        result_records = result_df.astype(object).to_dict(orient='records')
        print(result_records[0])
        # Store CSV version for download
        session['result'] = result_df.to_csv(index=False)
        session['csv_data'] = csv_data

        # Generate wind rose using the raw DataFrame
        wind_rose = create_wind_rose(result_df)

        # ✅ Send only native Python types to template
        return render_template('index.html',
                            result=result_records,
                            wind_rose_image=wind_rose,
                            metadata_map={fname: {'rows': len(df), 'columns': list(df.columns)} for fname, df in dataframes})

    return render_template('index.html')
@atexit.register
def cleanup_sessions():
    try:
        shutil.rmtree(SESSION_DIR)
    except Exception as e:
        print(f"Cleanup failed: {e}")
if __name__ == '__main__':
    port = find_free_port()
    Timer(1, open_browser, args=(port,)).start()
    serve(app, host="127.0.0.1", port=port)
