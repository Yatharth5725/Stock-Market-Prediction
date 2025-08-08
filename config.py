import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    
    # Database settings (if needed later)
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'sqlite:///stock_data.db'
    
    # API settings
    API_RATE_LIMIT = os.environ.get('API_RATE_LIMIT') or 100
    
    # ML Model settings
    MODEL_SAVE_PATH = 'models/'
    DATA_SAVE_PATH = 'data/'
    
    # Stock data settings
    DEFAULT_PERIOD = '1y'
    DEFAULT_INTERVAL = '1d'
    
    # LSTM settings
    LSTM_LOOKBACK = 60
    LSTM_EPOCHS = 50
    LSTM_BATCH_SIZE = 32
    
    # XGBoost settings
    XGB_N_ESTIMATORS = 100
    XGB_MAX_DEPTH = 6
    XGB_LEARNING_RATE = 0.1
    
    # Popular stock tickers
    POPULAR_STOCKS = [
        {"symbol": "AAPL", "name": "Apple Inc."},
        {"symbol": "GOOGL", "name": "Alphabet Inc."},
        {"symbol": "MSFT", "name": "Microsoft Corporation"},
        {"symbol": "AMZN", "name": "Amazon.com Inc."},
        {"symbol": "TSLA", "name": "Tesla Inc."},
        {"symbol": "META", "name": "Meta Platforms Inc."},
        {"symbol": "NVDA", "name": "NVIDIA Corporation"},
        {"symbol": "NFLX", "name": "Netflix Inc."},
        {"symbol": "JPM", "name": "JPMorgan Chase & Co."},
        {"symbol": "V", "name": "Visa Inc."}
    ]