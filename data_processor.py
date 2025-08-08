import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ta
from datetime import datetime, timedelta
import pickle
import os

class DataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.data_save_path = 'data/'
        
    def fetch_stock_data(self, symbol, period='1y', interval='1d'):
        """Fetch stock data from Yahoo Finance"""
        try:
            # Create cache filename
            cache_file = f"{self.data_save_path}{symbol}_{period}_{interval}.pkl"
            
            # Check if cached data exists and is recent (less than 1 hour old)
            if os.path.exists(cache_file):
                mod_time = os.path.getmtime(cache_file)
                if (datetime.now().timestamp() - mod_time) < 3600:  # 1 hour
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
            
            # Fetch fresh data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Cache the data
            os.makedirs(self.data_save_path, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            return data
            
        except Exception as e:
            raise Exception(f"Error fetching data for {symbol}: {str(e)}")
    
    def preprocess_data(self, data):
        """Preprocess stock data for ML models"""
        # Remove any missing values
        data = data.dropna()
        
        # Add technical indicators
        data = self.add_technical_indicators(data)
        
        # Remove any rows with NaN values created by indicators
        data = data.dropna()
        
        return data
    
    def add_technical_indicators(self, data):
        """Add technical indicators to the dataset"""
        # Moving averages
        data['MA_10'] = data['Close'].rolling(window=10).mean()
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        
        # RSI
        # Ensure data['Close'] is 1-dimensional Series
        close_series = data['Close']
        if len(close_series.shape) > 1:
            close_series = close_series.squeeze()
        data['RSI'] = ta.momentum.RSIIndicator(close_series).rsi()
        
        # MACD
        close_series = data['Close']
        if len(close_series.shape) > 1:
            close_series = close_series.squeeze()
        macd = ta.trend.MACD(close_series)
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        data['MACD_Histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(data['Close'])
        data['BB_Upper'] = bollinger.bollinger_hband()
        data['BB_Lower'] = bollinger.bollinger_lband()
        data['BB_Middle'] = bollinger.bollinger_mavg()
        
        # Volume indicators
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
        
        # Price changes
        data['Price_Change'] = data['Close'].pct_change()
        data['High_Low_Ratio'] = data['High'] / data['Low']
        
        return data
    
    def prepare_lstm_data(self, data, lookback=60, target_col='Close'):
        """Prepare data for LSTM model"""
        # Scale the data
        scaled_data = self.scaler.fit_transform(data[[target_col]])
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def prepare_xgboost_data(self, data, target_col='Close'):
        """Prepare data for XGBoost model"""
        # Create features
        features = ['Open', 'High', 'Low', 'Volume', 'MA_10', 'MA_20', 'MA_50', 
                   'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 
                   'Volume_Ratio', 'Price_Change', 'High_Low_Ratio']
        
        # Remove any columns that don't exist
        available_features = [col for col in features if col in data.columns]
        
        X = data[available_features].copy()
        y = data[target_col].copy()
        
        # Remove any NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        return X, y
    
    def calculate_technical_analysis(self, data):
        """Calculate technical analysis metrics"""
        current_price = data['Close'].iloc[-1]
        
        analysis = {
            "current_price": float(current_price),
            "ma_10": float(data['MA_10'].iloc[-1]) if 'MA_10' in data.columns else None,
            "ma_20": float(data['MA_20'].iloc[-1]) if 'MA_20' in data.columns else None,
            "ma_50": float(data['MA_50'].iloc[-1]) if 'MA_50' in data.columns else None,
            "rsi": float(data['RSI'].iloc[-1]) if 'RSI' in data.columns else None,
            "macd": float(data['MACD'].iloc[-1]) if 'MACD' in data.columns else None,
            "volume_trend": "increasing" if data['Volume'].iloc[-1] > data['Volume'].iloc[-2] else "decreasing",
            "volatility": float(data['Close'].pct_change().std() * np.sqrt(252))
        }
        
        # Add trading signals
        if analysis['rsi']:
            if analysis['rsi'] > 70:
                analysis['rsi_signal'] = "overbought"
            elif analysis['rsi'] < 30:
                analysis['rsi_signal'] = "oversold"
            else:
                analysis['rsi_signal'] = "neutral"
        
        # MA crossover signals
        if analysis['ma_10'] and analysis['ma_20']:
            if analysis['ma_10'] > analysis['ma_20']:
                analysis['ma_signal'] = "bullish"
            else:
                analysis['ma_signal'] = "bearish"
        
        return analysis
    
    def get_market_sentiment(self, data):
        """Calculate market sentiment indicators"""
        price_change = data['Close'].pct_change().iloc[-1]
        volume_change = data['Volume'].pct_change().iloc[-1]
        
        sentiment = {
            "price_momentum": "bullish" if price_change > 0 else "bearish",
            "volume_momentum": "increasing" if volume_change > 0 else "decreasing",
            "overall_sentiment": "positive" if price_change > 0 and volume_change > 0 else "negative",
            "confidence_score": min(abs(price_change) * 100, 100)
        }
        
        return sentiment
    
    def save_processed_data(self, data, symbol, suffix="processed"):
        """Save processed data to file"""
        filename = f"{self.data_save_path}{symbol}_{suffix}.pkl"
        os.makedirs(self.data_save_path, exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        return filename
    
    def load_processed_data(self, symbol, suffix="processed"):
        """Load processed data from file"""
        filename = f"{self.data_save_path}{symbol}_{suffix}.pkl"
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)
        return None