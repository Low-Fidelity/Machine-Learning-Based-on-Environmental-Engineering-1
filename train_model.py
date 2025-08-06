import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
import os

# Create model directory
os.makedirs('model', exist_ok=True)

# Load data
df = pd.read_csv('air_quality_data.csv')
print(f"Dataset shape: {df.shape}")

# Preprocessing - Fix datetime parsing
df['forecast_dt'] = pd.to_datetime(df['forecast dt iso'].str.replace(' UTC', '').str.replace(' +0000', ''), errors='coerce')
df['slice_dt'] = pd.to_datetime(df['slice dt iso'].str.replace(' UTC', '').str.replace(' +0000', ''), errors='coerce')

# Remove rows with invalid dates
df = df.dropna(subset=['forecast_dt', 'slice_dt'])
print(f"Dataset shape after removing invalid dates: {df.shape}")

# Time features
df['hour'] = df['forecast_dt'].dt.hour
df['month'] = df['forecast_dt'].dt.month
df['day_of_week'] = df['forecast_dt'].dt.dayofweek

# Cyclical features
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Weather interaction features
df['temp_humidity'] = df['temperature'] * df['humidity'] / 100
df['pressure_diff'] = df['pressure'] - df['ground_pressure']
df['dew_point_spread'] = df['temperature'] - df['dew_point']
df['wind_pressure'] = df['wind_speed'] * df['pressure'] / 1000
df['cloud_humidity'] = df['clouds'] * df['humidity'] / 100

# Is weekend
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Features selection
features = [
    'temperature', 'dew_point', 'pressure', 'ground_pressure', 'humidity', 
    'clouds', 'wind_speed', 'wind_deg', 'snow', 'ice', 'fr_rain', 
    'convective', 'snow_depth', 'accumulated', 'hours', 'rate',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'is_weekend',
    'temp_humidity', 'pressure_diff', 'dew_point_spread', 'wind_pressure',
    'cloud_humidity'
]

target = 'rain'

# Clean data
for col in features:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

df[target] = pd.to_numeric(df[target], errors='coerce')
df = df.dropna(subset=[target])

# Available features only
available_features = [f for f in features if f in df.columns]
print(f"Using {len(available_features)} features")

# Prepare data
X = df[available_features].values.astype(np.float32)
y = df[target].values.astype(np.float32)

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Neural Network Model
nn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    Dropout(0.1),
    
    Dense(1, activation='linear')
])

nn_model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Train Neural Network
print("Training Neural Network...")
nn_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

# Random Forest
print("Training Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Gradient Boosting
print("Training Gradient Boosting...")
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)
gb_model.fit(X_train, y_train)

# Ensemble predictions
def ensemble_predict(X_data):
    nn_pred = nn_model.predict(X_data, verbose=0).flatten()
    rf_pred = rf_model.predict(X_data)
    gb_pred = gb_model.predict(X_data)
    
    # Weighted ensemble
    ensemble_pred = 0.4 * nn_pred + 0.35 * rf_pred + 0.25 * gb_pred
    return ensemble_pred

# Predictions
train_pred = ensemble_predict(X_train)
val_pred = ensemble_predict(X_val)
test_pred = ensemble_predict(X_test)

# Evaluation
def evaluate_model(y_true, y_pred, set_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Custom accuracy for rainfall (within reasonable tolerance)
    tolerance_1mm = np.mean(np.abs(y_true - y_pred) <= 1.0) * 100
    tolerance_2mm = np.mean(np.abs(y_true - y_pred) <= 2.0) * 100
    
    print(f"\n{set_name} Results:")
    print(f"  MAE: {mae:.3f} mm")
    print(f"  RMSE: {rmse:.3f} mm")
    print(f"  R²: {r2:.4f}")
    print(f"  Within ±1mm: {tolerance_1mm:.1f}%")
    print(f"  Within ±2mm: {tolerance_2mm:.1f}%")
    
    return r2, mae

print("\n" + "="*50)
print("RAINFALL PREDICTION RESULTS")
print("="*50)

train_r2, train_mae = evaluate_model(y_train, train_pred, "Training")
val_r2, val_mae = evaluate_model(y_val, val_pred, "Validation")
test_r2, test_mae = evaluate_model(y_test, test_pred, "Test")

print(f"\nFinal Test Performance:")
print(f"R² Score: {test_r2:.4f}")
print(f"MAE: {test_mae:.3f} mm")

# Model performance assessment
if test_r2 > 0.7:
    print("🎉 EXCELLENT! Model performance is great!")
elif test_r2 > 0.5:
    print("🚀 GOOD! Strong predictive performance!")
else:
    print("👍 FAIR! Baseline model established!")

# Save models
nn_model.save('model/rainfall_nn_model.h5')
joblib.dump(rf_model, 'model/rainfall_rf_model.pkl')
joblib.dump(gb_model, 'model/rainfall_gb_model.pkl')
joblib.dump(scaler, 'model/rainfall_scaler.pkl')

# Save model info
model_info = {
    'features': available_features,
    'target': target,
    'test_r2': test_r2,
    'test_mae': test_mae,
    'ensemble_weights': [0.4, 0.35, 0.25]
}
joblib.dump(model_info, 'model/rainfall_model_info.pkl')

print(f"\nModels saved successfully!")
print(f"✅ Neural Network: model/rainfall_nn_model.h5")
print(f"✅ Random Forest: model/rainfall_rf_model.pkl") 
print(f"✅ Gradient Boosting: model/rainfall_gb_model.pkl")
print(f"✅ Scaler: model/rainfall_scaler.pkl")
print(f"✅ Model Info: model/rainfall_model_info.pkl")

# Simple prediction example
print(f"\n" + "="*50)
print("PREDICTION EXAMPLE")
print("="*50)

# Take first 5 test samples for demo
sample_predictions = test_pred[:5]
sample_actual = y_test[:5]

print("Sample Predictions vs Actual:")
for i in range(5):
    print(f"  Sample {i+1}: Predicted={sample_predictions[i]:.2f}mm, Actual={sample_actual[i]:.2f}mm")

print(f"\n🎯 Model ready for use!")
print(f"Dataset: {len(df):,} samples")
print(f"Features: {len(available_features)}")
print(f"Target: Rainfall prediction (mm)")