// DOM Elements
const form = document.getElementById('controlForm');
const analyzeBtn = document.getElementById('analyzeBtn');
const btnText = document.getElementById('btnText');
const btnSpinner = document.getElementById('btnSpinner');

// Initialize Charts
function initCharts() {
    Plotly.newPlot('mainChart', [{
        x: [],
        y: [],
        type: 'line',
        line: { color: '#0d6efd' }
    }], {
        margin: { t: 30, l: 50, r: 30, b: 50 },
        xaxis: { title: 'Date' },
        yaxis: { title: 'Price (USD)' }
    });

    Plotly.newPlot('predictionChart', [{
        x: [],
        y: [],
        type: 'line',
        name: 'Predicted',
        line: { color: '#dc3545' }
    }], {
        margin: { t: 30, l: 50, r: 30, b: 50 },
        xaxis: { title: 'Date' },
        yaxis: { title: 'Price (USD)' }
    });
}

// Form Submission Handler
form.addEventListener('submit', async (e) => {
    e.preventDefault();

    // UI Loading State
    analyzeBtn.disabled = true;
    btnText.textContent = 'Analyzing...';
    btnSpinner.classList.remove('d-none');

    // Get Form Data
    const symbol = document.getElementById('stockSymbol').value;
    const modelType = document.getElementById('modelType').value;

    try {
        // Fetch historical stock data
        const historicalResponse = await fetch(`/api/stocks/${symbol}`);
        if (!historicalResponse.ok) throw new Error('Failed to fetch historical data');
        const historicalData = await historicalResponse.json();

        // Fetch prediction data
        const predictionResponse = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                symbol: symbol,
                model_type: modelType,
                days_ahead: 30
            })
        });
        if (!predictionResponse.ok) throw new Error('Failed to fetch prediction data');
        const predictionData = await predictionResponse.json();

        // Prepare historical chart data
        const histDates = historicalData.data.map(item => item.Date);
        const histPrices = historicalData.data.map(item => item.Close);

        // Prepare prediction chart data
        const predDates = predictionData.predictions.map(item => item.date);
        const predPrices = predictionData.predictions.map(item => item.predicted_price);

        // Update Charts
        updateChart('mainChart', {
            x: histDates,
            y: histPrices,
            name: 'Historical'
        });

        updateChart('predictionChart', {
            x: predDates,
            y: predPrices,
            name: 'Forecast'
        });

        // Update Stats
        document.getElementById('currentPrice').textContent = `$${predictionData.current_price.toFixed(2)}`;
        document.getElementById('weekHigh').textContent = `$${historicalData.statistics.high_52w.toFixed(2)}`;
        document.getElementById('weekLow').textContent = `$${historicalData.statistics.low_52w.toFixed(2)}`;
        document.getElementById('marketCap').textContent = `-`; // Market cap not provided by backend

        // Update Titles
        document.getElementById('chartTitle').textContent = `${symbol} Historical Prices`;
        document.getElementById('predictionTitle').textContent = `${symbol} 30-Day Forecast (${modelType.toUpperCase()})`;

    } catch (error) {
        console.error('Analysis failed:', error);
        alert('Analysis failed. Please try again.');
    } finally {
        // Reset UI
        analyzeBtn.disabled = false;
        btnText.textContent = 'Run Analysis';
        btnSpinner.classList.add('d-none');
    }
});

// Chart Update Function
function updateChart(chartId, data) {
    Plotly.react(chartId, [{
        x: data.x,
        y: data.y,
        type: 'line',
        name: data.name,
        line: { color: chartId === 'mainChart' ? '#0d6efd' : '#dc3545' }
    }], {
        margin: { t: 30, l: 50, r: 30, b: 50 }
    });
}

// Initialize App
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
});