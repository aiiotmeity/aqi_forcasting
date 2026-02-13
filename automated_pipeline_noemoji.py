"""
Multi-Device Air Quality Forecasting Pipeline - CUMULATIVE VERSION

This module provides device-specific forecasting for:
- lora-v1 (Station 1) 
- loradev2 (Station 2)

FEATURES:
1. Uses FULL historical cumulative dataset for training
2. Better forecasting with more data over time
3. Device-specific S3 paths: data/air_quality/lora-v1/ and data/air_quality/loradev2/
4. Enhanced error handling for both devices
"""

import os
import time
import boto3
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# AWS Configuration
AWS_REGION = 'us-east-1'
DYNAMO_TABLE = 'ttn_aws_db'
S3_BUCKET = 'ai-model-bucket-output'

# Constants
GASES = ['SO2', 'PM2.5', 'PM10', 'NO2', 'NH3', 'CO', 'O3']
TIME_WINDOW = 30
FUTURE_STEPS = 4
RETRAIN_INTERVAL_DAYS = 4

# Realistic outlier thresholds for pollution data
OUTLIER_THRESHOLDS = {
    'SO2': {'min': 0, 'max': 1000},
    'PM2.5': {'min': 0, 'max': 500},
    'PM10': {'min': 0, 'max': 500},
    'NO2': {'min': 0, 'max': 1000},
    'NH3': {'min': 0, 'max': 100},
    'CO': {'min': 0, 'max': 50},
    'O3': {'min': 0, 'max': 300},
    'Temperature': {'min': -40, 'max': 60},
    'Humidity': {'min': 0, 'max': 100},
    'Pressure': {'min': 900, 'max': 1100},
    'AQI': {'min': 0, 'max': 500}
}

class AirQualityForecastingPipeline:
    def __init__(self, config=None):
        """
        Initialize device-specific forecasting pipeline
        
        Parameters:
        -----------
        config : dict, optional
            Configuration including device_id and device-specific settings
        """
        self.config = config or {}
        self.device_id = self.config.get('device_id', 'lora-v1')
        self.station_name = self.config.get('station_name', f'Station-{self.device_id}')
        self.region = self.config.get('region', AWS_REGION)
        self.dynamo_table = self.config.get('dynamo_table', DYNAMO_TABLE)
        self.s3_bucket = self.config.get('s3_bucket', S3_BUCKET)
        
        # Device-specific paths
        self.model_path = f'models/air_quality/{self.device_id}'
        self.data_path = f'data/air_quality/{self.device_id}'
        
        # Initialize AWS clients
        self.init_aws_clients()
        
        # Load device-specific cumulative dataset
        self.csv_file = self.config.get('csv_file', f'updated_real4_{self.device_id.replace("-", "_")}.csv')
        self.df = self.load_dataset()
        
        # Initialize models dictionary
        self.models = {}
        self.scalers = {}
        
        print(f" Initialized pipeline for device: {self.device_id} ({self.station_name})")
        print(f" Cumulative dataset loaded: {len(self.df)} records")
    
    def init_aws_clients(self):
        """Initialize AWS service clients"""
        self.s3_client = boto3.client('s3', region_name=self.region)
        self.dynamo_client = boto3.client('dynamodb', region_name=self.region)
        self.sagemaker_client = boto3.client('sagemaker', region_name=self.region)
    
    def load_dataset(self):
        """Load device-specific cumulative dataset"""
        try:
            # Try to load from local file first
            if os.path.exists(self.csv_file):
                df = pd.read_csv(self.csv_file)
                print(f" Loaded {self.device_id} cumulative dataset from local file: {self.csv_file}")
            else:
                # Try to load from S3
                try:
                    obj = self.s3_client.get_object(
                        Bucket=self.s3_bucket, 
                        Key=f"{self.data_path}/{self.csv_file}"
                    )
                    df = pd.read_csv(obj['Body'])
                    print(f" Loaded {self.device_id} dataset from S3")
                except:
                    print(f" No existing dataset found for {self.device_id}, creating empty dataset")
                    return self.create_empty_dataset()
            
            # Ensure timestamp column exists
            if 'timestamp' not in df.columns:
                df['timestamp'] = pd.to_datetime(df['DATE'], format='%d-%m-%Y')
            
            # Apply outlier detection
            df = self.remove_domain_outliers(df, method='cap')
            
            print(f" Loaded {len(df)} historical records for device {self.device_id}")
            return df
            
        except Exception as e:
            print(f" Error loading dataset for {self.device_id}: {e}")
            return self.create_empty_dataset()
    
    def create_empty_dataset(self):
        """Create empty dataset with correct structure"""
        return pd.DataFrame(columns=[
            'DATE', 'PM2.5', 'PM10', 'O3', 'CO', 'NO2', 'SO2', 'NH3',
            'Temperature', 'Humidity', 'Pressure', 'SampleCount', 'AQI', 
            'timestamp'
        ])
    
    def remove_domain_outliers(self, df, method='cap'):
        """Apply domain-specific outlier detection"""
        if df.empty:
            return df
            
        print(f" Applying outlier detection for {self.device_id} (Method: {method})")
        
        df_clean = df.copy()
        original_length = len(df_clean)
        
        for column, thresholds in OUTLIER_THRESHOLDS.items():
            if column in df_clean.columns:
                below_min = df_clean[column] < thresholds['min']
                above_max = df_clean[column] > thresholds['max']
                outliers = below_min | above_max
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    print(f"   {column}: Found {outlier_count} outliers")
                    
                    if method == 'cap':
                        if below_min.sum() > 0:
                            df_clean.loc[below_min, column] = thresholds['min']
                        if above_max.sum() > 0:
                            df_clean.loc[above_max, column] = thresholds['max']
        
        final_length = len(df_clean)
        print(f" Outlier detection complete for {self.device_id}: {original_length} -> {final_length} records")
        
        return df_clean
    
    def add_rolling_averages(self, df, target_column):
        """Add rolling averages for specific target column"""
        df = df.copy()
        
        # Define rolling windows based on dataset size
        max_window = min(7, len(df) // 2)
        windows = [3, max_window] if max_window >= 3 else [min(3, len(df) - 1)]
        
        for window in windows:
            if window > 0:
                df[f'{target_column}_rolling_{window}d'] = df[target_column].rolling(
                    window=window,
                    min_periods=1,
                    center=True
                ).mean()
        
        return df
    
    def prepare_forecasting_data(self, df, target_column):
        """Prepare device-specific data for forecasting using FULL historical dataset"""
        print(f" Preparing forecasting data for {target_column} using {len(df)} historical records")
        
        # Add rolling averages
        df = self.add_rolling_averages(df, target_column)
        
        # Define features based on available data
        base_features = [target_column]
        rolling_features = [col for col in df.columns if col.startswith(f'{target_column}_rolling_')]
        features = base_features + rolling_features
        
        print(f" Using features: {features}")
        
        # Convert to numeric and handle missing values
        for feature in features:
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
            df[feature].fillna(df[feature].mean(), inplace=True)
        
        # Scale features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[features])
        
        return df, features, scaled_data, scaler
    
    def create_time_windowed_data(self, data, target_idx, time_window=30):
        """Create time-windowed data for training"""
        # Adjust time window based on available data
        actual_window = min(time_window, len(data) - 2)
        
        if actual_window < 1:
            return np.array([]), np.array([])
        
        X = np.array([data[i:i + actual_window, :] for i in range(len(data) - actual_window)])
        y = np.array([data[i + actual_window, target_idx] for i in range(len(data) - actual_window)])
        return X, y.reshape(-1, 1)
    
    def create_model(self, input_shape, num_layers=2):
        """Create GRU model for device-specific forecasting"""
        model = Sequential()
        regularizer = l2(0.001)
        
        # Adjust model complexity based on data size
        if input_shape[0] < 20:
            units = 32
            dropout = 0.2
        else:
            units = 64
            dropout = 0.3
        
        # First GRU layer
        model.add(GRU(units=units, return_sequences=num_layers > 1,
                      kernel_regularizer=regularizer,
                      input_shape=input_shape))
        model.add(Dropout(dropout))
        
        # Additional layers if enough data
        if num_layers > 1 and input_shape[0] > 10:
            model.add(GRU(units=units//2, return_sequences=False,
                          kernel_regularizer=regularizer))
            model.add(Dropout(dropout))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mae')
        
        return model
    
    def generate_simple_forecast(self, target_column):
        """Generate simple trend-based forecast when not enough data for ML model"""
        print(f" Generating simple trend-based forecast for {target_column} ({self.device_id})")
        
        if len(self.df) == 0:
            # Use default values if no data
            default_values = {
                'PM2.5': 50, 'PM10': 60, 'SO2': 100, 'NO2': 50, 
                'NH3': 5, 'CO': 0.5, 'O3': 30
            }
            forecast_values = [default_values.get(target_column, 50)] * FUTURE_STEPS
        else:
            # Use last value or average with trend
            last_values = self.df[target_column].tail(min(7, len(self.df)))
            base_value = last_values.mean()
            
            # Calculate simple trend
            if len(last_values) > 1:
                trend = (last_values.iloc[-1] - last_values.iloc[0]) / len(last_values)
            else:
                trend = 0
            
            # Generate forecast with trend and variation
            forecast_values = []
            for i in range(FUTURE_STEPS):
                # Apply trend
                trend_value = base_value + (trend * (i + 1))
                
                # Add small random variation (5% of base value)
                variation = np.random.normal(0, base_value * 0.05)
                forecast_value = max(0, trend_value + variation)
                
                # Cap to reasonable limits
                if target_column in OUTLIER_THRESHOLDS:
                    limits = OUTLIER_THRESHOLDS[target_column]
                    forecast_value = max(limits['min'], min(limits['max'], forecast_value))
                
                forecast_values.append(forecast_value)
        
        # Create forecast DataFrame
        last_date = datetime.now()
        if not self.df.empty and 'timestamp' in self.df.columns:
            last_date = self.df['timestamp'].iloc[-1]
        
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=FUTURE_STEPS)
        
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            f'Forecasted_{target_column}': forecast_values
        })
        
        print(f" Simple forecast generated for {target_column} ({self.device_id}):")
        for _, row in forecast_df.iterrows():
            print(f"   {row['Date'].strftime('%Y-%m-%d')}: {row[f'Forecasted_{target_column}']:.2f}")
        
        return None, None, forecast_df
    
    def train_gru_model(self, target_column, time_window=30, epochs=50, batch_size=32):
        """Train GRU model for specific gas and device using FULL historical dataset"""
        print(f"\n Training {target_column} model for {self.device_id} ({self.station_name})")
        print(f" Using {len(self.df)} historical records for training")
        
        # Adjust time window based on available data
        actual_window = min(time_window, len(self.df) // 3)
        
        # Check if we have enough data
        if len(self.df) < actual_window + 10:
            print(f" Not enough data for {target_column} ML training. Need at least {actual_window + 10} records, have {len(self.df)}")
            print(f" Using simple forecasting instead for {target_column}")
            return self.generate_simple_forecast(target_column)
        
        # Prepare data using FULL historical dataset
        df_prepared, features, scaled_data, scaler = self.prepare_forecasting_data(
            self.df, target_column
        )
        
        # Get target feature index
        target_idx = features.index(target_column)
        
        # Create windowed data
        X, y = self.create_time_windowed_data(scaled_data, target_idx, actual_window)
        
        if len(X) < 10:
            print(f" Not enough windowed data for {target_column}. Need at least 10 samples, have {len(X)}")
            print(f" Using simple forecasting instead for {target_column}")
            return self.generate_simple_forecast(target_column)
        
        # Split data
        test_size = min(0.2, max(0.1, 5 / len(X)))  # Adaptive test size
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        
        print(f" Training data shape: {X_train.shape}")
        print(f" Test data shape: {X_test.shape}")
        print(f" Using time window: {actual_window}")
        
        # Create and train model
        model = self.create_model((X_train.shape[1], X_train.shape[2]))
        
        # Adjust training parameters based on data size
        actual_epochs = min(epochs, max(20, len(X_train) // 2))
        actual_batch_size = min(batch_size, max(8, len(X_train) // 4))
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=min(10, actual_epochs // 3),
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            epochs=actual_epochs,
            batch_size=actual_batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate model
        predictions = model.predict(X_test)
        
        # Inverse transform for evaluation
        pred_array = np.zeros((len(predictions), len(features)))
        pred_array[:, target_idx] = predictions.flatten()
        predictions_rescaled = scaler.inverse_transform(pred_array)[:, target_idx]
        
        y_test_array = np.zeros((len(y_test), len(features)))
        y_test_array[:, target_idx] = y_test.flatten()
        y_test_rescaled = scaler.inverse_transform(y_test_array)[:, target_idx]
        
        # Calculate metrics
        mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
        rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled))
        r2 = r2_score(y_test_rescaled, predictions_rescaled)
        
        print(f' Model Performance for {target_column} ({self.device_id}):')
        print(f'   MAE: {mae:.4f}')
        print(f'   RMSE: {rmse:.4f}')
        print(f'   R Score: {r2:.4f}')
        
        # Generate forecast
        forecast_df = self.generate_forecast_for_gas(
            target_column, model, scaler, scaled_data, target_idx, FUTURE_STEPS, actual_window
        )
        
        # Store model and scaler
        self.models[target_column] = model
        self.scalers[target_column] = scaler
        
        # Save model
        self.save_model(target_column, model)
        
        return model, scaler, forecast_df
    
    def generate_forecast_for_gas(self, target_column, model, scaler, scaled_data, target_idx, future_steps, time_window):
        """Generate forecast for specific gas"""
        print(f" Generating {future_steps}-day forecast for {target_column} ({self.device_id})")
        
        # Get last sequence
        last_sequence = scaled_data[-time_window:]
        features_count = last_sequence.shape[1]
        future_predictions = []
        
        for step in range(future_steps):
            # Make prediction
            last_sequence_reshaped = last_sequence.reshape(1, time_window, features_count)
            next_prediction = model.predict(last_sequence_reshaped, verbose=0)
            
            # Get actual values for rolling average calculation
            last_actual_values = scaler.inverse_transform(last_sequence)[:, 0]
            
            # Get predicted value in original scale
            next_value = scaler.inverse_transform(
                np.array([[next_prediction[0, 0]] + [0] * (features_count - 1)])
            )[0, 0]
            
            # Calculate rolling averages if we have multiple features
            if features_count > 1:
                values_for_3d = np.append(last_actual_values[-2:], next_value)[-3:]
                rolling_3d = np.mean(values_for_3d)
                
                if features_count > 2:
                    values_for_7d = np.append(last_actual_values[-6:], next_value)[-7:]
                    rolling_7d = np.mean(values_for_7d)
                    unscaled_features = np.array([[next_value, rolling_3d, rolling_7d]])
                else:
                    unscaled_features = np.array([[next_value, rolling_3d]])
            else:
                unscaled_features = np.array([[next_value]])
            
            next_features = scaler.transform(unscaled_features)
            
            # Update sequence
            future_predictions.append(next_features[0])
            last_sequence = np.vstack([last_sequence[1:], next_features])
        
        # Process predictions
        future_predictions = np.array(future_predictions)
        future_predictions_rescaled = scaler.inverse_transform(future_predictions)[:, target_idx]
        
        # Create forecast DataFrame
        last_date = self.df['timestamp'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps)
        
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            f'Forecasted_{target_column}': future_predictions_rescaled
        })
        
        # Apply outlier capping
        forecast_df = self.cap_forecast_outliers(forecast_df, target_column)
        
        print(f" Forecast generated for {target_column} ({self.device_id}):")
        for _, row in forecast_df.iterrows():
            print(f"   {row['Date'].strftime('%Y-%m-%d')}: {row[f'Forecasted_{target_column}']:.2f}")
        
        return forecast_df
    
    def cap_forecast_outliers(self, forecast_df, gas_name):
        """Cap forecast values to domain limits"""
        if gas_name in OUTLIER_THRESHOLDS:
            thresholds = OUTLIER_THRESHOLDS[gas_name]
            column = f'Forecasted_{gas_name}'
            
            if column in forecast_df.columns:
                below_min = forecast_df[column] < thresholds['min']
                above_max = forecast_df[column] > thresholds['max']
                
                if below_min.sum() > 0:
                    forecast_df.loc[below_min, column] = thresholds['min']
                    print(f"   Capped {below_min.sum()} {gas_name} values to minimum {thresholds['min']}")
                
                if above_max.sum() > 0:
                    forecast_df.loc[above_max, column] = thresholds['max']
                    print(f"   Capped {above_max.sum()} {gas_name} values to maximum {thresholds['max']}")
        
        return forecast_df
    
    def save_model(self, target_column, model):
        """Save device-specific model"""
        try:
            # Create device-specific directory
            device_model_dir = f'saved_models/{self.device_id}'
            os.makedirs(device_model_dir, exist_ok=True)
            
            # Save locally
            local_path = f"{device_model_dir}/{target_column}_model.keras"
            model.save(local_path)
            print(f" Model saved locally: {local_path}")
            
            # Save to S3
            with open(local_path, 'rb') as model_file:
                self.s3_client.upload_fileobj(
                    model_file,
                    self.s3_bucket,
                    f"{self.model_path}/{target_column}_model.keras"
                )
            print(f" Model uploaded to S3: {self.model_path}/{target_column}_model.keras")
            
        except Exception as e:
            print(f" Error saving model for {target_column} ({self.device_id}): {e}")
    
    def load_model(self, target_column):
        """Load device-specific model"""
        try:
            device_model_dir = f'saved_models/{self.device_id}'
            local_path = f"{device_model_dir}/{target_column}_model.keras"
            
            if os.path.exists(local_path):
                model = load_model(local_path)
                print(f" Loaded model from: {local_path}")
                return model
            
            # Try downloading from S3
            try:
                os.makedirs(device_model_dir, exist_ok=True)
                self.s3_client.download_file(
                    self.s3_bucket,
                    f"{self.model_path}/{target_column}_model.keras",
                    local_path
                )
                model = load_model(local_path)
                print(f" Downloaded and loaded model from S3 for {self.device_id}")
                return model
            except Exception as s3_err:
                print(f" Could not load model from S3: {s3_err}")
                return None
                
        except Exception as e:
            print(f" Error loading model for {target_column} ({self.device_id}): {e}")
            return None
    
    def forecast_gases(self, gases=None, load_saved_models=True):
        """Generate forecasts for all gases for this device using CUMULATIVE dataset"""
        gases = gases or GASES
        forecasts = {}
        
        print(f" Starting forecasting for {self.device_id} ({self.station_name})")
        print(f" Available cumulative data: {len(self.df)} historical records")
        
        for gas in gases:
            try:
                print(f"\n Processing {gas} for {self.device_id}...")
                model = None
                
                # Try to load existing model if we have enough data
                if load_saved_models and len(self.df) >= 20:
                    model = self.load_model(gas)
                
                if model is None or len(self.df) < 20:
                    # Train new model or use simple forecast
                    print(f" Training/Generating forecast for {gas}")
                    actual_window = min(TIME_WINDOW, max(5, len(self.df) // 3))
                    model, scaler, forecast_df = self.train_gru_model(
                        target_column=gas,
                        time_window=actual_window,
                        epochs=30,
                        batch_size=16
                    )
                    forecasts[gas] = forecast_df
                else:
                    # Use existing model
                    print(f" Using existing model for {gas}")
                    df_prepared, features, scaled_data, scaler = self.prepare_forecasting_data(
                        self.df, gas
                    )
                    self.models[gas] = model
                    self.scalers[gas] = scaler
                    
                    target_idx = features.index(gas)
                    actual_window = min(TIME_WINDOW, len(scaled_data) - 1)
                    forecast_df = self.generate_forecast_for_gas(
                        gas, model, scaler, scaled_data, target_idx, FUTURE_STEPS, actual_window
                    )
                    forecasts[gas] = forecast_df
                    
                print(f" Successfully forecasted {gas} for {self.device_id}")
                    
            except Exception as e:
                print(f" Error forecasting {gas} for {self.device_id}: {e}")
                # Generate fallback simple forecast
                try:
                    print(f" Generating fallback forecast for {gas}")
                    _, _, forecast_df = self.generate_simple_forecast(gas)
                    forecasts[gas] = forecast_df
                    print(f" Fallback forecast generated for {gas}")
                except Exception as fallback_error:
                    print(f" Complete failure for {gas}: {fallback_error}")
        
        # Save forecasts to S3 with device-specific paths
        self.save_forecasts(forecasts)
        
        print(f" Forecasting summary for {self.device_id}: {len(forecasts)}/{len(gases)} gases completed")
        
        return forecasts
    
    def save_forecasts(self, forecasts):
        """Save device-specific forecasts to S3 with SEPARATE PATHS"""
        try:
            if not forecasts:
                print(f" No forecasts to save for {self.device_id}")
                return
            
            print(f" Saving forecasts for {self.device_id}...")
            
            # Create combined forecast
            combined_forecast = None
            
            for gas, forecast_df in forecasts.items():
                if combined_forecast is None:
                    combined_forecast = pd.DataFrame({'Date': forecast_df['Date']})
                combined_forecast[f'Forecasted_{gas}'] = forecast_df[f'Forecasted_{gas}']
            
            if combined_forecast is not None:
                # Format dates consistently
                save_df = combined_forecast.copy()
                save_df['Date'] = save_df['Date'].dt.strftime('%Y-%m-%d')
                
                # Save CSV locally
                forecast_filename = f'forecast_{self.device_id.replace("-", "_")}.csv'
                save_df.to_csv(forecast_filename, index=False)
                print(f" Forecast saved locally: {forecast_filename}")
                
                # Create device-specific dashboard data
                dashboard_data = {
                    "device_id": self.device_id,
                    "station_name": self.station_name,
                    "updated_at": datetime.now().isoformat(),
                    "forecast_days": len(combined_forecast),
                    "historical_data_days": len(self.df),
                    "gases": {},
                    "dates": save_df['Date'].tolist()
                }
                
                # Add gas data
                for column in combined_forecast.columns:
                    if column.startswith('Forecasted_'):
                        gas_name = column.replace('Forecasted_', '')
                        dashboard_data["gases"][gas_name] = {
                            "values": combined_forecast[column].tolist(),
                            "unit": "g/m"
                        }
                
                # Save JSON locally
                json_filename = f'forecast_{self.device_id.replace("-", "_")}.json'
                with open(json_filename, 'w') as f:
                    json.dump(dashboard_data, f, indent=2)
                print(f" Dashboard JSON saved: {json_filename}")
                
                # UPLOAD TO S3 WITH DEVICE-SPECIFIC PATHS
                try:
                    csv_buffer = save_df.to_csv(index=False).encode()
                    json_buffer = json.dumps(dashboard_data).encode()
                    
                    # Path 1: Device-specific folder
                    csv_key_1 = f"data/air_quality/{self.device_id}/latest_forecast.csv"
                    json_key_1 = f"data/air_quality/{self.device_id}/latest_forecast.json"
                    
                    # Path 2: Root level with device suffix
                    if self.device_id == "lora-v1":
                        device_suffix = "lora_v1"
                    elif self.device_id == "lora-v3":
                        device_suffix = "lora-v3"
                    else:
                        # This will handle "loradev2" and any other default cases
                        device_suffix = "loradev2"
                    csv_key_2 = f"data/air_quality/latest_forecast_{device_suffix}.csv"
                    json_key_2 = f"data/air_quality/latest_forecast_{device_suffix}.json"
                    
                    # Upload CSV files
                    self.s3_client.put_object(
                        Body=csv_buffer,
                        Bucket=self.s3_bucket,
                        Key=csv_key_1,
                        ContentType='text/csv'
                    )
                    
                    self.s3_client.put_object(
                        Body=csv_buffer,
                        Bucket=self.s3_bucket,
                        Key=csv_key_2,
                        ContentType='text/csv'
                    )
                    
                    # Upload JSON files
                    self.s3_client.put_object(
                        Body=json_buffer,
                        Bucket=self.s3_bucket,
                        Key=json_key_1,
                        ContentType='application/json'
                    )
                    
                    self.s3_client.put_object(
                        Body=json_buffer,
                        Bucket=self.s3_bucket,
                        Key=json_key_2,
                        ContentType='application/json'
                    )
                    
                    print(f" Forecasts uploaded to S3 for {self.device_id}")
                    print(f"    Device folder: {csv_key_1}")
                    print(f"    Device folder: {json_key_1}") 
                    print(f"    Root level: {csv_key_2}")
                    print(f"    Root level: {json_key_2}")
                    
                except Exception as s3_err:
                    print(f" Error uploading forecasts to S3: {s3_err}")
                    
        except Exception as e:
            print(f" Error saving forecasts for {self.device_id}: {e}")
    
    def should_retrain(self):
        """Check if device-specific models should be retrained"""
        try:
            last_train_file = f'last_training_{self.device_id.replace("-", "_")}.txt'
            
            if os.path.exists(last_train_file):
                with open(last_train_file, 'r') as f:
                    last_train_time = datetime.fromisoformat(f.read().strip())
                
                current_time = datetime.now()
                time_diff = current_time - last_train_time
                
                print(f" Time since last training for {self.device_id}: {time_diff.days} days")
                return time_diff.days >= RETRAIN_INTERVAL_DAYS
            else:
                print(f" No training record for {self.device_id}, retraining needed")
                return True
                
        except Exception as e:
            print(f" Error checking retraining schedule for {self.device_id}: {e}")
            return True
    
    def update_training_timestamp(self, device_id=None):
        """Update device-specific training timestamp"""
        try:
            device_id = device_id or self.device_id
            timestamp_file = f'last_training_{device_id.replace("-", "_")}.txt'
            
            with open(timestamp_file, 'w') as f:
                f.write(datetime.now().isoformat())
            
            print(f" Training timestamp updated for {device_id}")
            
        except Exception as e:
            print(f" Error updating training timestamp for {device_id}: {e}")
    
    def run_pipeline(self):
        """Run complete pipeline for this device using CUMULATIVE dataset"""
        try:
            print(f" Starting pipeline for {self.device_id} ({self.station_name})")
            print(f" Using cumulative dataset with {len(self.df)} historical records")
            
            # The dataset is already updated by update_dataset.py
            # So we just need to reload it and run forecasting
            self.df = self.load_dataset()
            
            # Check retraining
            if self.should_retrain():
                print(f" Retraining models for {self.device_id} with {len(self.df)} records")
                self.forecast_gases(load_saved_models=False)
                self.update_training_timestamp()
            else:
                print(f" Using existing models for {self.device_id}")
                self.forecast_gases(load_saved_models=True)
            
            print(f" Pipeline completed for {self.device_id}")
            
        except Exception as e:
            print(f" Error in pipeline for {self.device_id}: {e}")
            import traceback
            traceback.print_exc()
