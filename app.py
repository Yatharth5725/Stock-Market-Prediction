# app.py - Main Flask Application with Frontend and API
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
import os
from data_processor import DataProcessor
from ml_models import StockPredictor
from config import Config

app = Flask(__name__)
CORS(app)
app.config.from_object(Config)

# Initialize components
data_processor = DataProcessor()
stock_predictor = StockPredictor()

# Frontend Routes
@app.route('/')
def index():
    """Serve the main frontend page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Serve dashboard page (if you create one)"""
    return render_template('dashboard.html')

# API Routes
@app.route('/api/')
def api_index():
    """API documentation endpoint"""
    return jsonify({
        "message": "Stock Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "GET /api/stocks/<symbol>": "Get stock data",
            "POST /api/predict": "Get stock predictions",
            "GET /api/analyze/<symbol>": "Get technical analysis",
            "POST /api/models/train": "Train ML models",
            "GET /api/health": "Health check",
            "GET /api/popular-stocks": "Get popular stocks list",
            "GET /api/search-stocks": "Search stocks by symbol/name"
        }
    })

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    model_info = stock_predictor.get_model_info()
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "models_available": model_info
    })

@app.route('/api/stocks/<symbol>')
def get_stock_data(symbol):
    """Get stock data for a specific symbol"""
    try:
        period = request.args.get('period', '1y')
        interval = request.args.get('interval', '1d')
        
        stock_data = data_processor.fetch_stock_data(symbol, period, interval)
        
        if stock_data.empty:
            return jsonify({"error": "No data found for symbol"}), 404
        
        # Convert to JSON format
        data = stock_data.reset_index().to_dict('records')
        
        # Convert Timestamp objects to strings
        for record in data:
            if 'Date' in record:
                record['Date'] = record['Date'].strftime('%Y-%m-%d')
        
        # Calculate basic statistics
        current_price = float(stock_data['Close'].iloc[-1])
        high_52w = float(stock_data['High'].max())
        low_52w = float(stock_data['Low'].min())
        
        stats = {
            "current_price": current_price,
            "high_52w": high_52w,
            "low_52w": low_52w,
            "volume_avg": float(stock_data['Volume'].mean()),
            "volatility": float(stock_data['Close'].pct_change().std() * np.sqrt(252)),
            "change_1d": float(stock_data['Close'].pct_change().iloc[-1] * 100),
            "change_1w": float(stock_data['Close'].pct_change(5).iloc[-1] * 100) if len(stock_data) >= 5 else 0,
            "performance_vs_52w_high": float(((current_price - high_52w) / high_52w) * 100),
            "performance_vs_52w_low": float(((current_price - low_52w) / low_52w) * 100)
        }
        
        return jsonify({
            "symbol": symbol.upper(),
            "data": data,
            "statistics": stats,
            "total_records": len(data),
            "last_updated": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict_stock():
    """Make stock price predictions"""
    try:
        data = request.json
        symbol = data.get('symbol')
        model_type = data.get('model_type', 'lstm')
        days_ahead = data.get('days_ahead', 30)
        
        if not symbol:
            return jsonify({"error": "Symbol is required"}), 400
        
        if days_ahead > 90:
            return jsonify({"error": "Maximum prediction period is 90 days"}), 400
        
        # Get and preprocess data
        raw_data = data_processor.fetch_stock_data(symbol, period='2y')
        processed_data = data_processor.preprocess_data(raw_data)
        
        # Make predictions
        if model_type.lower() == 'lstm':
            predictions = stock_predictor.predict_lstm(processed_data, days_ahead)
        else:
            predictions = stock_predictor.predict_xgboost(processed_data, days_ahead)
        
        # Generate future dates (business days only)
        last_date = processed_data.index[-1]
        future_dates = pd.bdate_range(
            start=last_date + timedelta(days=1),
            periods=days_ahead
        )
        
        # Calculate confidence scores (simplified)
        current_price = float(processed_data['Close'].iloc[-1])
        volatility = float(processed_data['Close'].pct_change().std())
        
        # Prepare response
        prediction_data = []
        for i, (date, pred) in enumerate(zip(future_dates, predictions)):
            # Confidence decreases with time
            confidence = max(0.5, 0.9 - (i * 0.01))
            
            prediction_data.append({
                "date": date.strftime('%Y-%m-%d'),
                "predicted_price": float(pred),
                "confidence": float(confidence),
                "change_from_current": float(((pred - current_price) / current_price) * 100)
            })
        
        return jsonify({
            "symbol": symbol.upper(),
            "model_type": model_type,
            "predictions": prediction_data,
            "current_price": current_price,
            "prediction_range": f"{days_ahead} days",
            "volatility": volatility,
            "last_updated": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze/<symbol>')
def analyze_stock(symbol):
    """Get technical analysis for a stock"""
    try:
        # Fetch data
        stock_data = data_processor.fetch_stock_data(symbol, period='1y')
        processed_data = data_processor.preprocess_data(stock_data)
        
        # Technical analysis
        analysis = data_processor.calculate_technical_analysis(processed_data)
        
        # Market sentiment
        sentiment = data_processor.get_market_sentiment(processed_data)
        
        return jsonify({
            "symbol": symbol.upper(),
            "technical_analysis": analysis,
            "market_sentiment": sentiment,
            "last_updated": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/models/train', methods=['POST'])
def train_models():
    """Train ML models"""
    try:
        data = request.json
        symbol = data.get('symbol')
        model_types = data.get('model_types', ['lstm', 'xgboost'])
        
        if not symbol:
            return jsonify({"error": "Symbol is required"}), 400
        
        # Fetch and preprocess data
        raw_data = data_processor.fetch_stock_data(symbol, period='5y')
        processed_data = data_processor.preprocess_data(raw_data)
        
        results = {}
        
        for model_type in model_types:
            try:
                if model_type == 'lstm':
                    accuracy = stock_predictor.train_lstm(processed_data)
                elif model_type == 'xgboost':
                    accuracy = stock_predictor.train_xgboost(processed_data)
                
                results[model_type] = {
                    "status": "success",
                    "metrics": accuracy,
                    "trained_on": symbol
                }
            except Exception as e:
                results[model_type] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return jsonify({
            "symbol": symbol.upper(),
            "training_results": results,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/models/evaluate', methods=['POST'])
def evaluate_models():
    """Evaluate trained models"""
    try:
        data = request.json
        symbol = data.get('symbol')
        model_type = data.get('model_type', 'lstm')
        
        if not symbol:
            return jsonify({"error": "Symbol is required"}), 400
        
        # Fetch and preprocess data
        raw_data = data_processor.fetch_stock_data(symbol, period='5y')
        processed_data = data_processor.preprocess_data(raw_data)
        
        # Evaluate model
        results = stock_predictor.evaluate_model(processed_data, model_type=model_type)
        
        return jsonify({
            "symbol": symbol.upper(),
            "model_type": model_type,
            "evaluation_results": results,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
