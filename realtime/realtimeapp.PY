import os
import sys
import time
import subprocess
import threading
import json
import pandas as pd
import datetime
import plotly
from flask import Flask, render_template, jsonify, request, send_file
from collections import Counter, deque
import paho.mqtt.client as mqtt
import logging
from queue import Queue
import io

# Global variables
app = Flask(__name__, template_folder='templates', static_folder='static')
transactions = deque(maxlen=1000)
fraud_transactions = deque(maxlen=50)
transaction_stats = {
    "total_transactions": 0,
    "fraud_transactions": 0,
    "legitimate_transactions": 0,
    "fraud_amount_total": 0,
    "legitimate_amount_total": 0,
    "transaction_history": {"timestamps": [], "counts": [], "fraud_counts": []},
    "merchant_stats": Counter(),
    "category_stats": Counter(),
    "hourly_distribution": [0] * 24
}
running = False
publisher_process = None
mqtt_client = None

# Setup logging
logging.basicConfig(filename="system.log", level=logging.INFO, format="%(asctime)s - %(message)s")

def check_requirements():
    required_packages = ["flask", "pandas", "plotly", "paho-mqtt", "numpy", "joblib", "xgboost", "scikit-learn", "matplotlib", "streamlit"]
    print("🔍 Checking required packages...")
    print(f"Using Python executable: {sys.executable}")
    missing_packages = []
    for package in required_packages:
        try:
            if package == "paho-mqtt":
                __import__("paho.mqtt.client")
                print(f"✅ {package} is installed")
            elif package == "scikit-learn":
                __import__("sklearn")
                print(f"✅ {package} is installed")
            elif package == "plotly":
                __import__("plotly")
                print(f"✅ {package} is installed")
            else:
                __import__(package)
                print(f"✅ {package} is installed")
        except ImportError as e:
            missing_packages.append(package)
            print(f"❌ {package} is missing: {e}")
    if missing_packages:
        print("\n📦 Installing missing packages...")
        try:
            python_executable = sys.executable
            subprocess.check_call([python_executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir"] + missing_packages)
            print("✅ All packages installed successfully!")
            time.sleep(1)
            for package in missing_packages:
                try:
                    if package == "paho-mqtt":
                        __import__("paho.mqtt.client")
                    elif package == "scikit-learn":
                        __import__("sklearn")
                    elif package == "plotly":
                        __import__("plotly")
                    else:
                        __import__(package)
                    print(f"✅ {package} installed and verified")
                except ImportError as e:
                    print(f"❌ Failed to verify {package} after installation: {e}")
                    return False
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing packages: {e}")
            return False
    return True

def setup_dashboard():
    print("🔧 Setting up dashboard files...")
    if not os.path.exists("templates"):
        os.makedirs("templates")
        print("✅ Created templates directory")
    if not os.path.exists("static"):
        os.makedirs("static")
        print("✅ Created static directory")
    with open("templates/index.html", "w") as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fraud Detection Dashboard</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.24.2/plotly.min.js"></script>
    <style>
        :root {
            --primary-bg: #1a1a1a;
            --card-bg: #2d2d2d;
            --success-color: #28a745;
            --warning-color: #ffc107;
            --danger-color: #dc3545;
            --info-color: #17a2b8;
            --dark-color: #121212;
            --text-color: #ffffff;
        }
        
        body {
            background-color: var(--primary-bg);
            color: var(--text-color);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .dashboard-header {
            background-color: var(--dark-color);
            color: var(--text-color);
            padding: 1rem 0;
            margin-bottom: 2rem;
        }
        
        .stat-card {
            background-color: var(--card-bg);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        
        .table-container, .chart-container {
            background-color: var(--card-bg);
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            margin-bottom: 20px;
        }
        
        .table {
            color: var(--text-color);
        }
        
        .navbar-brand img {
            height: 60px;
            margin-right: 15px;
        }
        
        .navbar-brand span {
            color: #ffffff;
            font-size: 1.5rem;
        }
        
        .navbar-subtitle {
            color: #ffffff;
            font-size: 0.9rem;
        }
        
        .table-responsive {
            max-height: 400px;
            overflow-y: auto;
        }
    </style>
</head>
<body>
    <nav class="navbar dashboard-header">
        <div class="container">
            <a class="navbar-brand" href="#">
                <img src="/static/logo.png" alt="Logo">
                <div>
                    <span>Fraud Detection Dashboard</span>
                    <div class="navbar-subtitle">Done by Prithwin and Akshay</div>
                </div>
            </a>
            <div class="badge bg-info">
                <i class="fas fa-clock me-1"></i>
                <span id="last-updated">Updating...</span>
            </div>
        </div>
    </nav>

    <div class="container">
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="stat-card">
                    <p class="stat-label">Total Transactions</p>
                    <p class="stat-value" id="total-transactions">0</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card">
                    <p class="stat-label">Fraud Rate</p>
                    <p class="stat-value fraud-indicator" id="fraud-rate">0%</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card">
                    <p class="stat-label">Fraud Amount</p>
                    <p class="stat-value fraud-indicator" id="fraud-amount">$0</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card">
                    <p class="stat-label">Avg. Fraud Amount</p>
                    <p class="stat-value fraud-indicator" id="avg-fraud-amount">$0</p>
                </div>
            </div>
        </div>

        <div class="row mb-4">
            <div class="col-md-8">
                <div class="chart-container">
                    <h5>Transaction Activity</h5>
                    <div id="transaction-chart" style="height: 280px;"></div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="chart-container">
                    <h5>Transaction Status</h5>
                    <div id="status-chart" style="height: 280px;"></div>
                </div>
            </div>
        </div>

        <div class="row mb-4">
            <div class="col-md-6">
                <div class="chart-container">
                    <h5>Top Merchants</h5>
                    <div id="merchants-chart" style="height: 280px;"></div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="chart-container">
                    <h5>Hourly Distribution</h5>
                    <div id="hourly-chart" style="height: 280px;"></div>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col-md-12">
                <div class="table-container">
                    <div class="d-flex justify-content-between align-items-center mb-3">
                        <h5><i class="fas fa-history me-2"></i>All Transactions</h5>
                        <button class="btn btn-primary" onclick="downloadTransactions()">
                            <i class="fas fa-download me-2"></i>Download CSV
                        </button>
                    </div>
                    <div class="table-responsive">
                        <table class="table table-striped">
                            <thead>
                                <tr>
                                    <th>Time</th>
                                    <th>Merchant</th>
                                    <th>Amount</th>
                                    <th>Status</th>
                                    <th>Category</th>
                                </tr>
                            </thead>
                            <tbody id="all-transactions-table">
                                <tr>
                                    <td colspan="5" class="text-center">Loading...</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="table-container">
                    <h5><i class="fas fa-exclamation-triangle me-2"></i>Recent Fraud Alerts</h5>
                    <div class="table-responsive">
                        <table class="table table-striped">
                            <thead>
                                <tr>
                                    <th>Time</th>
                                    <th>Merchant</th>
                                    <th>Amount</th>
                                    <th>Probability</th>
                                </tr>
                            </thead>
                            <tbody id="fraud-alerts-table">
                                <tr>
                                    <td colspan="4" class="text-center">Loading...</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <div class="container mb-4">
            <div class="row justify-content-center">
                <div class="col-md-6 text-center">
                    <button id="start-btn" class="btn btn-success me-2" onclick="startTransactions()">Generate Transactions</button>
                    <button id="stop-btn" class="btn btn-danger" onclick="stopTransactions()" disabled>Stop Transactions</button>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/js/bootstrap.bundle.min.js"></script>
    <script>
        const REFRESH_INTERVAL = 1000;

        function updateStats() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('total-transactions').textContent = data.total_transactions.toLocaleString();
                    document.getElementById('fraud-rate').textContent = data.fraud_rate + '%';
                    document.getElementById('fraud-amount').textContent = '$' + data.fraud_amount_total.toLocaleString();
                    document.getElementById('avg-fraud-amount').textContent = '$' + data.avg_fraud_amount.toLocaleString();
                    document.getElementById('last-updated').textContent = new Date().toLocaleTimeString();
                    updateTransactionChart(data.transaction_history);
                    updateStatusChart(data.fraud_transactions, data.legitimate_transactions);
                    updateHourlyChart(data.hourly_distribution);
                });
        }

        function updateAllTransactions() {
            fetch('/api/recent_transactions?limit=1000')
                .then(response => response.json())
                .then(data => {
                    const tableBody = document.getElementById('all-transactions-table');
                    tableBody.innerHTML = '';
                    data.reverse().forEach(transaction => {
                        const row = document.createElement('tr');
                        row.innerHTML = `
                            <td>${transaction.timestamp.split(' ')[1] || transaction.timestamp}</td>
                            <td>${transaction.merchant}</td>
                            <td>$${parseFloat(transaction.amt).toFixed(2)}</td>
                            <td><span class="badge ${transaction.is_fraud ? 'bg-danger' : 'bg-success'}">
                                ${transaction.is_fraud ? 'Fraud' : 'Legitimate'}</span></td>
                            <td>${transaction.category}</td>
                        `;
                        tableBody.appendChild(row);
                    });
                });
        }

        function updateFraudAlerts() {
            fetch('/api/recent_frauds?limit=5')
                .then(response => response.json())
                .then(data => {
                    const tableBody = document.getElementById('fraud-alerts-table');
                    tableBody.innerHTML = '';
                    if (data.length === 0) {
                        tableBody.innerHTML = '<tr><td colspan="4" class="text-center">No fraud alerts yet</td></tr>';
                        return;
                    }
                    data.reverse().forEach(transaction => {
                        const row = document.createElement('tr');
                        const timestamp = transaction.timestamp || new Date().toLocaleTimeString();
                        const probValue = transaction.fraud_probability || '0.95';
                        row.innerHTML = `
                            <td>${timestamp.split(' ')[1] || timestamp}</td>
                            <td>${transaction.merchant}</td>
                            <td class="text-danger fw-bold">$${parseFloat(transaction.amt).toFixed(2)}</td>
                            <td>${(parseFloat(probValue) * 100).toFixed(1)}%</td>
                        `;
                        tableBody.appendChild(row);
                    });
                });
        }

        function updateMerchantsChart() {
            fetch('/api/top_merchants')
                .then(response => response.json())
                .then(data => {
                    const merchants = data.map(item => item[0]);
                    const counts = data.map(item => item[1]);
                    Plotly.newPlot('merchants-chart', [{
                        x: merchants,
                        y: counts,
                        type: 'bar',
                        marker: { color: '#17a2b8' }
                    }], {
                        margin: { t: 10, r: 10, l: 50, b: 80 },
                        xaxis: { tickangle: -45 },
                        yaxis: { title: 'Transaction Count' }
                    });
                });
        }

        function updateTransactionChart(historyData) {
            const timestamps = historyData.timestamps;
            const counts = historyData.counts;
            const fraudCounts = historyData.fraud_counts;
            const windowSize = 5;
            const smoothedCounts = [];
            const smoothedFraudCounts = [];
            
            for (let i = 0; i < counts.length; i++) {
                let sum = 0, fraudSum = 0, count = 0;
                for (let j = Math.max(0, i - windowSize + 1); j <= i; j++) {
                    sum += counts[j] || 0;
                    fraudSum += fraudCounts[j] || 0;
                    count++;
                }
                smoothedCounts.push(sum);
                smoothedFraudCounts.push(fraudSum);
            }
            
            Plotly.newPlot('transaction-chart', [
                { x: timestamps, y: smoothedCounts, type: 'scatter', mode: 'lines', name: 'All Transactions', line: { color: '#17a2b8', width: 3 } },
                { x: timestamps, y: smoothedFraudCounts, type: 'scatter', mode: 'lines', name: 'Fraud Transactions', line: { color: '#dc3545', width: 3 } }
            ], {
                margin: { t: 10, r: 10, l: 40, b: 40 },
                legend: { orientation: 'h', x: 0.5, xanchor: 'center', y: 1.02 },
                xaxis: { showgrid: false },
                yaxis: { title: 'Transaction Count' }
            });
        }

        function updateStatusChart(fraudCount, legitimateCount) {
            Plotly.newPlot('status-chart', [{
                values: [fraudCount, legitimateCount],
                labels: ['Fraud', 'Legitimate'],
                type: 'pie',
                hole: 0.4,
                marker: { colors: ['#dc3545', '#28a745'] },
                textinfo: 'label+percent',
                insidetextorientation: 'radial'
            }], {
                margin: { t: 10, r: 10, l: 10, b: 10 },
                showlegend: false
            });
        }

        function updateHourlyChart(hourlyData) {
            const hours = Array.from({length: 24}, (_, i) => i);
            Plotly.newPlot('hourly-chart', [{
                x: hours,
                y: hourlyData,
                type: 'bar',
                marker: { color: '#6f42c1' }
            }], {
                margin: { t: 10, r: 10, l: 40, b: 40 },
                xaxis: {
                    title: 'Hour of Day',
                    tickmode: 'array',
                    tickvals: hours.filter(h => h % 2 === 0),
                    ticktext: hours.filter(h => h % 2 === 0).map(h => h + ':00')
                },
                yaxis: { title: 'Transaction Count' }
            });
        }

        function downloadTransactions() {
            window.location.href = '/api/download_transactions';
        }

        function initDashboard() {
            updateStats();
            updateAllTransactions();
            updateFraudAlerts();
            updateMerchantsChart();
            setInterval(updateStats, REFRESH_INTERVAL);
            setInterval(updateAllTransactions, REFRESH_INTERVAL);
            setInterval(updateFraudAlerts, REFRESH_INTERVAL);
            setInterval(updateMerchantsChart, REFRESH_INTERVAL * 2);
        }

        document.addEventListener('DOMContentLoaded', initDashboard);

        function startTransactions() {
            fetch('/start_transactions', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'started') {
                        document.getElementById('start-btn').disabled = true;
                        document.getElementById('stop-btn').disabled = false;
                    }
                });
        }

        function stopTransactions() {
            fetch('/stop_transactions', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'stopped') {
                        document.getElementById('start-btn').disabled = false;
                        document.getElementById('stop-btn').disabled = true;
                    }
                });
        }
    </script>
</body>
</html>
        """)
        print("✅ Updated index.html with white title, credits, and larger logo")

def train_model():
    print("\n🧠 Training fraud detection model...")
    try:
        subprocess.run([sys.executable, "bigtrain.py"], check=True)
        if os.path.exists("fraud_model.pkl"):
            print("✅ Model training completed successfully!")
            return True
        else:
            print("❌ Model training failed!")
            return False
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        return False

def start_mqtt_broker():
    print("\n🔄 Checking MQTT broker status...")
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        def on_connect(client, userdata, flags, rc, properties=None):
            if rc == 0:
                print("✅ MQTT broker is running!")
            else:
                print(f"❌ MQTT broker connection failed with code {rc}")
        client.on_connect = on_connect
        client.connect("localhost", 1883, 5)
        client.loop_start()
        time.sleep(2)
        client.loop_stop()
        client.disconnect()
        return True
    except Exception as e:
        print(f"❌ MQTT broker error: {e}")
        print("Please install and start a Mosquitto MQTT broker:")
        print("- On Windows: Download from https://mosquitto.org/download/")
        return False

def on_connect(client, userdata, flags, rc, properties=None):
    print(f"Connected with result code {rc}")
    if rc == 0:
        logging.info("Successfully subscribed to credit_card/transactions")
        client.subscribe("credit_card/transactions")
    else:
        logging.error(f"Failed to connect with code {rc}")

def on_message(client, userdata, msg):
    try:
        transaction = json.loads(msg.payload.decode())
        process_transaction(transaction)
    except Exception as e:
        logging.error(f"Error processing message: {e}")

def process_transaction(transaction):
    global transactions, fraud_transactions, transaction_stats
    if "timestamp" not in transaction:
        transaction["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    logging.info(f"Processing transaction: {transaction}")
    transaction_stats["total_transactions"] += 1
    if transaction.get("is_fraud", 0) == 1:
        transaction_stats["fraud_transactions"] += 1
        transaction_stats["fraud_amount_total"] += float(transaction["amt"])
        fraud_transactions.append(transaction)
    else:
        transaction_stats["legitimate_transactions"] += 1
        transaction_stats["legitimate_amount_total"] += float(transaction["amt"])
    transactions.append(transaction)
    transaction_stats["merchant_stats"][transaction["merchant"]] += 1
    transaction_stats["category_stats"][transaction["category"]] += 1
    hour = int(transaction.get("transaction_hour", datetime.datetime.now().hour))
    transaction_stats["hourly_distribution"][hour] += 1
    current_timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    transaction_stats["transaction_history"]["timestamps"].append(current_timestamp)
    if len(transaction_stats["transaction_history"]["timestamps"]) > 60:
        transaction_stats["transaction_history"]["timestamps"].pop(0)
        transaction_stats["transaction_history"]["counts"].pop(0)
        transaction_stats["transaction_history"]["fraud_counts"].pop(0)
    transaction_stats["transaction_history"]["counts"].append(1)
    transaction_stats["transaction_history"]["fraud_counts"].append(1 if transaction.get("is_fraud", 0) == 1 else 0)
    logging.info(f"Updated stats: {transaction_stats}")

def load_transaction_data():
    try:
        if os.path.exists("detected_frauds.csv"):
            fraud_df = pd.read_csv("detected_frauds.csv")
            for _, row in fraud_df.iterrows():
                transaction = row.to_dict()
                if "timestamp" not in transaction and "datetime" in transaction:
                    transaction["timestamp"] = transaction["datetime"]
                transaction["is_fraud"] = 1
                process_transaction(transaction)
        if os.path.exists("simulated_transactions.csv"):
            trans_df = pd.read_csv("simulated_transactions.csv")
            for _, row in trans_df.sample(min(100, len(trans_df))).iterrows():
                process_transaction(row.to_dict())
    except Exception as e:
        logging.error(f"Error loading transaction data: {e}")

def start_mqtt_client():
    global mqtt_client
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    try:
        mqtt_client.connect("localhost", 1883, 60)
        mqtt_client.loop_start()
        logging.info("✅ MQTT client started for classification")
    except Exception as e:
        logging.error(f"Error connecting to MQTT broker: {e}")

def stop_mqtt_client():
    global mqtt_client
    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        mqtt_client = None
        logging.info("✅ MQTT client stopped")

def start_publisher():
    global publisher_process
    if sys.platform.startswith('win'):
        publisher_process = subprocess.Popen(['cmd.exe', '/k', sys.executable, "mqtt_publisher.py"], creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        publisher_process = subprocess.Popen(["gnome-terminal", "--", sys.executable, "mqtt_publisher.py"])
    logging.info("✅ Publisher started")

def stop_publisher():
    global publisher_process
    if publisher_process:
        publisher_process.terminate()
        publisher_process.wait()
        publisher_process = None
        logging.info("✅ Publisher stopped")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    stats = {
        "total_transactions": transaction_stats["total_transactions"],
        "fraud_transactions": transaction_stats["fraud_transactions"],
        "legitimate_transactions": transaction_stats["legitimate_transactions"],
        "fraud_rate": round(transaction_stats["fraud_transactions"] / transaction_stats["total_transactions"] * 100, 2) if transaction_stats["total_transactions"] > 0 else 0,
        "fraud_amount_total": round(transaction_stats["fraud_amount_total"], 2),
        "legitimate_amount_total": round(transaction_stats["legitimate_amount_total"], 2),
        "avg_fraud_amount": round(transaction_stats["fraud_amount_total"] / transaction_stats["fraud_transactions"], 2) if transaction_stats["fraud_transactions"] > 0 else 0,
        "avg_legitimate_amount": round(transaction_stats["legitimate_amount_total"] / transaction_stats["legitimate_transactions"], 2) if transaction_stats["legitimate_transactions"] > 0 else 0,
        "transaction_history": transaction_stats["transaction_history"],
        "hourly_distribution": transaction_stats["hourly_distribution"]
    }
    logging.info(f"API stats returned: {stats}")
    return jsonify(stats)

@app.route('/api/recent_transactions')
def get_recent_transactions():
    limit = min(int(request.args.get('limit', 10)), 1000)
    recent = list(transactions)[-limit:]
    logging.info(f"API recent_transactions returned: {len(recent)} transactions")
    return jsonify(recent)

@app.route('/api/recent_frauds')
def get_recent_frauds():
    limit = min(int(request.args.get('limit', 10)), 50)
    recent = list(fraud_transactions)[-limit:]
    logging.info(f"API recent_frauds returned: {len(recent)} frauds")
    return jsonify(recent)

@app.route('/api/top_merchants')
def get_top_merchants():
    top_merchants = transaction_stats["merchant_stats"].most_common(10)
    logging.info(f"API top_merchants returned: {top_merchants}")
    return jsonify(top_merchants)

@app.route('/api/download_transactions')
def download_transactions():
    df = pd.DataFrame(list(transactions))
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return send_file(
        io.BytesIO(buffer.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'transactions_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )

@app.route('/start_transactions', methods=['POST'])
def start_transactions():
    global running
    if not running:
        running = True
        start_mqtt_client()
        time.sleep(1)
        start_publisher()
        logging.info("Transactions generation started")
    return jsonify({"status": "started"})

@app.route('/stop_transactions', methods=['POST'])
def stop_transactions():
    global running
    if running:
        stop_publisher()
        stop_mqtt_client()
        running = False
        logging.info("Transactions generation stopped")
    return jsonify({"status": "stopped"})

if __name__ == '__main__':
    if not check_requirements():
        sys.exit(1)
    setup_dashboard()
    if not os.path.exists("fraud_model.pkl"):
        if not train_model():
            sys.exit(1)
    if not start_mqtt_broker():
        sys.exit(1)
    load_transaction_data()
    app.run(debug=True, use_reloader=False, port=5000)