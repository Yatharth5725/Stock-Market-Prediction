
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os
import warnings
import tensorflow as tf

try:
    import xgboost as xgb
except ImportError:
    xgb = None

warnings.filterwarnings('ignore')

from data_processor import DataProcessor  # Moved import to top for clarity

class StockPredictor:
    def __init__(self, model_path='models/'):
        self.lstm_model = None
        self.xgb_model = None
        self.scaler = None
        self.model_path = model_path
        self.processor = DataProcessor()
        
    def train_lstm(self, data, lookback=60, epochs=50, batch_size=32):
        """Train LSTM model for stock prediction"""
        # Prepare data
        X, y = self.processor.prepare_lstm_data(data, lookback)
        
        if len(X) < 100:
            raise ValueError("Insufficient data for LSTM training. Need at least 100 samples.")
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Reshape for LSTM
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Build LSTM model
        self.lstm_model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(25),
            tf.keras.layers.Dense(1)
        ])
        
        self.lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        # Early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train model
        history = self.lstm_model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Calculate accuracy
        predictions = self.lstm_model.predict(X_test, verbose=0)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Save model
        os.makedirs(self.model_path, exist_ok=True)
        self.lstm_model.save(f'{self.model_path}lstm_model.h5')
        self.scaler = self.processor.scaler
        
        with open(f'{self.model_path}scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        accuracy = max(0, r2 * 100) if r2 > 0 else 0.0
        
        return {
            "mse": float(mse),
            "mae": float(mae),
            "r2_score": float(r2),
            "accuracy": float(accuracy),
            "epochs_trained": len(history.history['loss'])
        }
    
    def train_xgboost(self, data):
        """Train XGBoost model for stock prediction"""
        if xgb is None:
            raise ImportError("xgboost module is not installed. Please install it to use this feature.")
        # Prepare data
        X, y = self.processor.prepare_xgboost_data(data)
        
        if len(X) < 50:
            raise ValueError("Insufficient data for XGBoost training. Need at least 50 samples.")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Train XGBoost model
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            subsample=0.8,
            colsample_bytree=0.8
        )
        
        self.xgb_model.fit(X_train, y_train)
        
        # Calculate accuracy
        predictions = self.xgb_model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Save model
        os.makedirs(self.model_path, exist_ok=True)
        with open(f'{self.model_path}xgb_model.pkl', 'wb') as f:
            pickle.dump(self.xgb_model, f)
        
        # Ensure X is a DataFrame for feature importance keys
        feature_importance = {}
        if hasattr(X, 'columns'):
            feature_importance = dict(zip(X.columns, self.xgb_model.feature_importances_))
        else:
            feature_importance = {f'feature_{i}': imp for i, imp in enumerate(self.xgb_model.feature_importances_)}
        
        return {
            "mse": float(mse),
            "mae": float(mae),
            "r2_score": float(r2),
            "accuracy": float(max(0, r2 * 100)),
            "feature_importance": feature_importance
        }
    
    def predict_lstm(self, data, days_ahead=30, lookback=60):
        """Make predictions using LSTM model"""
        # Load model and scaler
        try:
            self.lstm_model = tf.keras.models.load_model(f'{self.model_path}lstm_model.h5')
            with open(f'{self.model_path}scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
        except Exception as e:
            print(f"No saved LSTM model found or error loading it: {e}. Training new model...")
            self.train_lstm(data)
            # Reload model after training
            self.lstm_model = tf.keras.models.load_model(f'{self.model_path}lstm_model.h5')
            with open(f'{self.model_path}scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
        
        # Prepare data
        scaled_data = self.scaler.transform(data[['Close']])
        
        # Get last sequence
        if len(scaled_data) < lookback:
            raise ValueError(f"Need at least {lookback} data points for LSTM prediction")
        
        last_sequence = scaled_data[-lookback:]
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days_ahead):
            # Reshape for prediction
            pred_input = current_sequence.reshape((1, lookback, 1))
            
            # Make prediction
            next_pred = self.lstm_model.predict(pred_input, verbose=0)[0, 0]
            predictions.append(next_pred)
            
            # Update sequence
            current_sequence = np.append(current_sequence[1:], next_pred).reshape(-1, 1)
        
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        return predictions.flatten()
    
    def predict_xgboost(self, data, days_ahead=30):
        """Make predictions using XGBoost model"""
        if xgb is None:
            raise ImportError("xgboost module is not installed. Please install it to use this feature.")
        # Load model
        try:
            with open(f'{self.model_path}xgb_model.pkl', 'rb') as f:
                self.xgb_model = pickle.load(f)
        except Exception as e:
            print(f"No saved XGBoost model found or error loading it: {e}. Training new model...")
            self.train_xgboost(data)
            # Reload model after training
            with open(f'{self.model_path}xgb_model.pkl', 'rb') as f:
                self.xgb_model = pickle.load(f)
        
        # Prepare features for the last available data point
        X, _ = self.processor.prepare_xgboost_data(data)
        
        if len(X) == 0:
            raise ValueError("No features available for XGBoost prediction")
        
        last_features = X.iloc[-1:].copy()
        
        predictions = []
        
        for i in range(days_ahead):
            # Make prediction
            next_pred = self.xgb_model.predict(last_features)[0]
            predictions.append(next_pred)
            
            # Update features (simplified approach)
            # In a real scenario, you'd need more sophisticated feature engineering
            last_features = last_features.copy()
            if 'Close' in last_features.columns:
                last_features.loc[last_features.index[0], 'Close'] = next_pred
            
            # Add some noise to prevent identical predictions
            noise_factor = 0.001
            for col in last_features.columns:
                if col != 'Close':
                    last_features.loc[last_features.index[0], col] *= (1 + np.random.normal(0, noise_factor))
        
        return np.array(predictions)
    
    def get_model_info(self):
        """Get information about available models"""
        lstm_exists = os.path.exists(f'{self.model_path}lstm_model.h5')
        xgb_exists = os.path.exists(f'{self.model_path}xgb_model.pkl')
        
        return {
            "lstm_available": lstm_exists,
            "xgboost_available": xgb_exists,
            "models_path": self.model_path
        }
    
    def evaluate_model(self, data, model_type='lstm'):
        """Evaluate model performance"""
        if model_type == 'lstm':
            X, y = self.processor.prepare_lstm_data(data)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            if self.lstm_model is None:
                try:
                    self.lstm_model = tf.keras.models.load_model(f'{self.model_path}lstm_model.h5')
                except Exception as e:
                    return {"error": f"No trained LSTM model found: {e}"}
            
            predictions = self.lstm_model.predict(X, verbose=0)
            
        else:  # xgboost
            X, y = self.processor.prepare_xgboost_data(data)
            
            if self.xgb_model is None:
                try:
                    with open(f'{self.model_path}xgb_model.pkl', 'rb') as f:
                        self.xgb_model = pickle.load(f)
                except Exception as e:
                    return {"error": f"No trained XGBoost model found: {e}"}
            
            predictions = self.xgb_model.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        accuracy = max(0, r2 * 100) if r2 > 0 else 0.0
        
        return {
            "mse": float(mse),
            "mae": float(mae),
            "r2_score": float(r2),
            "accuracy": float(accuracy)
        }
