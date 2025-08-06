import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file
import joblib
import numpy as np
import matplotlib.pyplot as plt
import io
import requests
import datetime
import time
import tensorflow as tf

app = Flask(__name__)

# --- Muat Model dan Komponen yang sudah dilatih ---
try:
    nn_model = tf.keras.models.load_model('model/rainfall_nn_model.h5', compile=False)
    rf_model = joblib.load('model/rainfall_rf_model.pkl')
    gb_model = joblib.load('model/rainfall_gb_model.pkl')
    scaler = joblib.load('model/rainfall_scaler.pkl')
    model_info = joblib.load('model/rainfall_model_info.pkl')
    expected_features = model_info['features']
    ensemble_weights = model_info['ensemble_weights']
    print("Semua model berhasil dimuat.")
    print(f"Features yang diperlukan: {len(expected_features)}")
except Exception as e:
    print(f"Error: Gagal memuat model: {e}")
    nn_model, rf_model, gb_model, scaler, model_info = None, None, None, None, None
    expected_features, ensemble_weights = None, None

OPENWEATHERMAP_API_KEY = "260a47451396ec0bbb39dd28fd1670cc"

def ensemble_predict(X_data):
    nn_pred = nn_model.predict(X_data, verbose=0).flatten()
    rf_pred = rf_model.predict(X_data)
    gb_pred = gb_model.predict(X_data)
    ensemble_pred = (ensemble_weights[0] * nn_pred +
                     ensemble_weights[1] * rf_pred +
                     ensemble_weights[2] * gb_pred)
    return ensemble_pred

def create_features(weather_data):
    temp = weather_data['temperature']
    humidity = weather_data['humidity']
    pressure = weather_data['pressure']
    wind_speed = weather_data['wind_speed']
    dew_point = temp - ((100 - humidity) / 5)
    ground_pressure = pressure + 5
    clouds = weather_data.get('clouds', 50)
    wind_deg = weather_data.get('wind_deg', 180)
    snow = 0.0
    ice = 0.0
    fr_rain = weather_data.get('rainfall', 0.0)
    convective = 0.0
    snow_depth = 0.0
    accumulated = 0.0
    hours = 1.0
    rate = fr_rain
    current_time = datetime.datetime.now()
    hour = current_time.hour
    month = current_time.month
    day_of_week = current_time.weekday()
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    is_weekend = 1 if day_of_week >= 5 else 0
    temp_humidity = temp * humidity / 100
    pressure_diff = pressure - ground_pressure
    dew_point_spread = temp - dew_point
    wind_pressure = wind_speed * pressure / 1000
    cloud_humidity = clouds * humidity / 100
    features_dict = {
        'temperature': temp,
        'dew_point': dew_point,
        'pressure': pressure,
        'ground_pressure': ground_pressure,
        'humidity': humidity,
        'clouds': clouds,
        'wind_speed': wind_speed,
        'wind_deg': wind_deg,
        'snow': snow,
        'ice': ice,
        'fr_rain': fr_rain,
        'convective': convective,
        'snow_depth': snow_depth,
        'accumulated': accumulated,
        'hours': hours,
        'rate': rate,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'month_sin': month_sin,
        'month_cos': month_cos,
        'is_weekend': is_weekend,
        'temp_humidity': temp_humidity,
        'pressure_diff': pressure_diff,
        'dew_point_spread': dew_point_spread,
        'wind_pressure': wind_pressure,
        'cloud_humidity': cloud_humidity
    }
    feature_values = []
    for feature in expected_features:
        feature_values.append(features_dict.get(feature, 0.0))
    return np.array(feature_values).reshape(1, -1)

def validate_weather_input(data):
    errors = {}
    if not -50 <= data.get('temperature', 0) <= 60:
        errors['temperature'] = 'Suhu harus antara -50°C dan 60°C.'
    if not 0 <= data.get('humidity', 0) <= 100:
        errors['humidity'] = 'Kelembaban harus antara 0% dan 100%.'
    if not 900 <= data.get('pressure', 0) <= 1100:
        errors['pressure'] = 'Tekanan Udara harus antara 900 hPa dan 1100 hPa.'
    if not 0 <= data.get('wind_speed', 0) <= 50:
        errors['wind_speed'] = 'Kecepatan Angin harus antara 0 m/s dan 50 m/s.'
    if not 0 <= data.get('rainfall', 0) <= 500:
        errors['rainfall'] = 'Curah Hujan harus antara 0 mm dan 500 mm.'
    return errors

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_rainfall', methods=['POST'])
def predict_rainfall():
    if None in [nn_model, rf_model, gb_model, scaler]:
        return jsonify({'error': 'Model belum dimuat dengan benar.'}), 500
    try:
        data = request.get_json(force=True)
        errors = validate_weather_input(data)
        if errors:
            return jsonify({'error': 'Input tidak valid', 'details': errors}), 400
        features = create_features(data)
        features_scaled = scaler.transform(features)
        prediction = ensemble_predict(features_scaled)
        output = round(float(prediction[0]), 3)
        if output < 0.1:
            category = "Tidak ada hujan"
        elif output < 1.0:
            category = "Hujan ringan"
        elif output < 5.0:
            category = "Hujan sedang"
        else:
            category = "Hujan lebat"
        return jsonify({
            'prediksi_curah_hujan_mm': output,
            'kategori': category,
            'input_data': data
        })
    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan saat prediksi: {e}'}), 500

@app.route('/predict_from_coords', methods=['POST'])
def predict_from_coords():
    if None in [nn_model, rf_model, gb_model, scaler]:
        return jsonify({'error': 'Model belum dimuat dengan benar.'}), 500
    try:
        data = request.get_json(force=True)
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        if latitude is None or longitude is None:
            return jsonify({'error': 'Koordinat lintang dan bujur diperlukan.'}), 400
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={OPENWEATHERMAP_API_KEY}&units=metric"
        max_retries = 3
        retry_delay = 1
        weather_response_json = None
        for i in range(max_retries):
            try:
                response = requests.get(weather_url, timeout=10)
                response.raise_for_status()
                weather_response_json = response.json()
                break
            except requests.exceptions.RequestException:
                time.sleep(retry_delay * (2 ** i))
                if i == max_retries - 1:
                    raise
        if not weather_response_json:
            return jsonify({'error': 'Gagal mendapatkan data cuaca dari API eksternal.'}), 500
        weather_data = {
            'temperature': weather_response_json['main']['temp'],
            'humidity': weather_response_json['main']['humidity'],
            'pressure': weather_response_json['main']['pressure'],
            'wind_speed': weather_response_json['wind']['speed'],
            'rainfall': weather_response_json.get('rain', {}).get('1h', 0.0),
            'clouds': weather_response_json.get('clouds', {}).get('all', 50),
            'wind_deg': weather_response_json.get('wind', {}).get('deg', 180)
        }
        errors = validate_weather_input(weather_data)
        if errors:
            return jsonify({'error': 'Data cuaca dari API tidak valid', 'details': errors}), 400
        features = create_features(weather_data)
        features_scaled = scaler.transform(features)
        prediction = ensemble_predict(features_scaled)
        output = round(float(prediction[0]), 3)
        if output < 0.1:
            category = "Tidak ada hujan"
        elif output < 1.0:
            category = "Hujan ringan"
        elif output < 5.0:
            category = "Hujan sedang"
        else:
            category = "Hujan lebat"
        response_data = {
            'prediksi_curah_hujan_mm': output,
            'kategori': category,
            'lokasi': {
                'latitude': latitude,
                'longitude': longitude
            },
            'data_cuaca': weather_data
        }
        return jsonify(response_data)
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Gagal terhubung ke API cuaca: {e}'}), 500
    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan saat prediksi dari koordinat: {e}'}), 500

@app.route('/model_info', methods=['GET'])
def get_model_info():
    if model_info is None:
        return jsonify({'error': 'Informasi model tidak tersedia.'}), 500
    return jsonify({
        'model_type': 'Ensemble (Neural Network + Random Forest + Gradient Boosting)',
        'target': model_info['target'],
        'features_count': len(model_info['features']),
        'test_r2_score': round(model_info['test_r2'], 4),
        'test_mae': round(model_info['test_mae'], 3),
        'ensemble_weights': {
            'neural_network': ensemble_weights[0],
            'random_forest': ensemble_weights[1],
            'gradient_boosting': ensemble_weights[2]
        }
    })

@app.route('/export_csv', methods=['POST'])
def export_csv():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])
    csv_file = io.StringIO()
    df.to_csv(csv_file, index=False)
    output = io.BytesIO()
    output.write(csv_file.getvalue().encode('utf-8'))
    output.seek(0)
    return send_file(output,
                     mimetype='text/csv',
                     as_attachment=True,
                     download_name='prediksi_curah_hujan.csv')

@app.route('/plot.png')
def plot_png():
    fig, ax = plt.subplots(figsize=(8, 6))
    categories = ['Tidak ada\nhujan', 'Hujan\nringan', 'Hujan\nsedang', 'Hujan\nlebat']
    values = [65, 20, 10, 5]
    colors = ['#87CEEB', '#4682B4', '#1E90FF', '#0000CD']
    bars = ax.bar(categories, values, color=colors, alpha=0.8)
    ax.set_ylabel('Probabilitas (%)')
    ax.set_title('Distribusi Prediksi Curah Hujan', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value}%', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    img = io.BytesIO()
    fig.savefig(img, format='png', dpi=150, bbox_inches='tight')
    img.seek(0)
    plt.close(fig)
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)