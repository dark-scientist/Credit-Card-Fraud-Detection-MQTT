import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import time
import paho.mqtt.client as mqtt
import threading
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime, timedelta
import os
import random
from collections import deque

# Set page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
def initialize_session_state():
    if 'transactions' not in st.session_state:
        st.session_state.transactions = deque(maxlen=100)  # Store last 100 transactions
    if 'fraud_count' not in st.session_state:
        st.session_state.fraud_count = 0
    if 'legitimate_count' not in st.session_state:
        st.session_state.legitimate_count = 0
    if 'mqtt_client' not in st.session_state:
        st.session_state.mqtt_client = None
    if 'processing_active' not in st.session_state:
        st.session_state.processing_active = False
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = datetime.now()
    if 'transaction_history' not in st.session_state:
        st.session_state.transaction_history = []
    if 'fraud_probabilities' not in st.session_state:
        st.session_state.fraud_probabilities = []
    if 'amounts_by_category' not in st.session_state:
        st.session_state.amounts_by_category = {}
    if 'fraud_by_category' not in st.session_state:
        st.session_state.fraud_by_category = {}
    if 'transactions_by_hour' not in st.session_state:
        st.session_state.transactions_by_hour = {hour: 0 for hour in range(24)}
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'label_encoders' not in st.session_state:
        st.session_state.label_encoders = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = None

initialize_session_state()

# Apply custom CSS for professional look and better visibility
st.markdown("""
<style>
    .fraud-alert {
        background-color: #ffe6e6;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #cc0000;
        margin: 10px 0;
        color: #333333;
    }
    .legitimate-transaction {
        background-color: #e6ffe6;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #006600;
        margin: 10px 0;
        color: #333333;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        color: #333333;
    }
    .stButton>button {
        width: 100%;
        background-color: #007bff;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .dashboard-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 5px;
        color: #003087;
    }
    .dashboard-subtitle {
        text-align: center;
        font-size: 1.2rem;
        font-style: italic;
        color: #555555;
        margin-bottom: 20px;
    }
    .stDataFrame {
        background-color: #ffffff;
        color: #000000;
    }
    .logo-container {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Display logo at the top, centered and smaller
st.markdown('<div class="logo-container">', unsafe_allow_html=True)
st.image("logo.png", width=200)  # Reduced from 300 to 200 for a smaller size
st.markdown('</div>', unsafe_allow_html=True)

# Dashboard layout
st.markdown('<h1 class="dashboard-title">Real-Time Fraud Detection Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="dashboard-subtitle">Done by Prithwin & Akshay</p>', unsafe_allow_html=True)

# Helper functions for model loading and preprocessing
def load_model_and_encoders():
    """Load the fraud detection model and prepare encoders"""
    try:
        if not os.path.exists("fraud_model.pkl"):
            st.error("Model file 'fraud_model.pkl' not found! Run bigtrain.py first.")
            return False
        
        if not os.path.exists("fraudTrain.csv"):
            st.error("Training data file 'fraudTrain.csv' not found!")
            return False
        
        model = joblib.load("fraud_model.pkl")
        df = pd.read_csv("fraudTrain.csv", nrows=10000)
        df = df.dropna()
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        df['dob'] = pd.to_datetime(df['dob'])
        
        df['transaction_hour'] = df['trans_date_trans_time'].dt.hour
        df['transaction_day'] = df['trans_date_trans_time'].dt.day
        df['transaction_month'] = df['trans_date_trans_time'].dt.month
        df['age'] = df['trans_date_trans_time'].dt.year - df['dob'].dt.year
        df['geo_distance'] = np.sqrt((df['lat'] - df['merch_lat'])**2 + (df['long'] - df['merch_long'])**2)
        
        drop_cols = ['trans_date_trans_time', 'dob', 'Unnamed: 0', 'first', 'last', 'street', 'trans_num', 'unix_time', 'cc_num']
        df = df.drop(columns=drop_cols, errors='ignore')
        
        categorical_cols = ['merchant', 'category', 'gender', 'city', 'state', 'job']
        label_encoders = {}
        for col in categorical_cols:
            label_encoders[col] = LabelEncoder()
            label_encoders[col].fit(df[col].astype(str))
        
        selected_features = ['amt', 'geo_distance', 'transaction_hour', 'transaction_day', 
                            'transaction_month', 'age', 'city_pop', 'merchant', 'category', 
                            'gender', 'city', 'state', 'job']
        
        df_encoded = df.copy()
        for col in categorical_cols:
            df_encoded[col] = label_encoders[col].transform(df_encoded[col].astype(str))
        
        scaler = StandardScaler()
        scaler.fit(df_encoded[selected_features])
        
        st.session_state.model = model
        st.session_state.label_encoders = label_encoders
        st.session_state.scaler = scaler
        st.session_state.selected_features = selected_features
        st.session_state.model_loaded = True
        return True
    
    except Exception as e:
        st.error(f"Error loading model and encoders: {e}")
        return False

def preprocess_transaction(transaction):
    """Process a transaction for prediction"""
    if not st.session_state.model_loaded:
        return None
    
    try:
        data = {}
        for key in st.session_state.selected_features:
            if key in transaction:
                data[key] = [transaction[key]]
            else:
                return None
        
        df_trans = pd.DataFrame(data)
        
        categorical_cols = ['merchant', 'category', 'gender', 'city', 'state', 'job']
        for col in categorical_cols:
            if col in df_trans.columns:
                try:
                    df_trans[col] = st.session_state.label_encoders[col].transform(df_trans[col].astype(str))
                except ValueError:
                    df_trans[col] = -1
        
        X_scaled = st.session_state.scaler.transform(df_trans[st.session_state.selected_features])
        df_scaled = pd.DataFrame(X_scaled, columns=st.session_state.selected_features)
        
        return df_scaled
    
    except Exception as e:
        st.error(f"Error preprocessing transaction: {e}")
        return None

def generate_transaction(fraud_probability=0.05):
    """Generate a random transaction"""
    if not os.path.exists("fraudTrain.csv"):
        st.error("Training data file 'fraudTrain.csv' not found!")
        return None
    
    try:
        df = pd.read_csv("fraudTrain.csv", nrows=10000)
        
        MERCHANTS = df['merchant'].unique().tolist()
        CATEGORIES = df['category'].unique().tolist()
        CITIES = df['city'].unique().tolist()
        STATES = df['state'].unique().tolist()
        JOBS = df['job'].unique().tolist()
        GENDERS = df['gender'].unique().tolist()
        
        AMT_MIN, AMT_MAX = df['amt'].min(), df['amt'].max()
        CITY_POP_MIN, CITY_POP_MAX = df['city_pop'].min(), df['city_pop'].max()
        LAT_MIN, LAT_MAX = df['lat'].min(), df['lat'].max()
        LONG_MIN, LONG_MAX = df['long'].min(), df['long'].max()
        MERCH_LAT_MIN, MERCH_LAT_MAX = df['merch_lat'].min(), df['merch_lat'].max()
        MERCH_LONG_MIN, MERCH_LONG_MAX = df['merch_long'].min(), df['merch_long'].max()
        
        df['geo_distance'] = np.sqrt((df['lat'] - df['merch_lat'])**2 + (df['long'] - df['merch_long'])**2)
        GEO_DISTANCE_MIN, GEO_DISTANCE_MAX = df['geo_distance'].min(), df['geo_distance'].max()
        
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        df['dob'] = pd.to_datetime(df['dob'])
        df['age'] = df['trans_date_trans_time'].dt.year - df['dob'].dt.year
        AGE_MIN, AGE_MAX = df['age'].min(), df['age'].max()
        
        is_fraud = random.random() < fraud_probability
        current_time = datetime.now()
        
        merchant = random.choice(MERCHANTS)
        category = random.choice(CATEGORIES)
        gender = random.choice(GENDERS)
        city = random.choice(CITIES)
        state = random.choice(STATES)
        job = random.choice(JOBS)
        
        if is_fraud:
            amt = round(random.uniform(AMT_MAX * 0.7, AMT_MAX), 2)
        else:
            amt = round(random.uniform(AMT_MIN, AMT_MAX * 0.7), 2)
        
        lat = random.uniform(LAT_MIN, LAT_MAX)
        long = random.uniform(LONG_MIN, LONG_MAX)
        merch_lat = random.uniform(MERCH_LAT_MIN, MERCH_LAT_MAX)
        merch_long = random.uniform(MERCH_LONG_MIN, MERCH_LONG_MAX)
        
        geo_distance = np.sqrt((lat - merch_lat)**2 + (long - merch_long)**2)
        if is_fraud and random.random() < 0.7:
            geo_distance = random.uniform(GEO_DISTANCE_MAX * 0.8, GEO_DISTANCE_MAX * 1.2)
        
        transaction_hour = current_time.hour
        transaction_day = current_time.day
        transaction_month = current_time.month
        age = random.randint(int(AGE_MIN), int(AGE_MAX))
        city_pop = random.randint(int(CITY_POP_MIN), int(CITY_POP_MAX))
        
        transaction = {
            "merchant": merchant,
            "category": category,
            "amt": amt,
            "gender": gender,
            "city": city,
            "state": state,
            "city_pop": city_pop,
            "job": job,
            "lat": lat,
            "long": long,
            "merch_lat": merch_lat,
            "merch_long": merch_long,
            "is_fraud": int(is_fraud),
            "geo_distance": geo_distance,
            "transaction_hour": transaction_hour,
            "transaction_day": transaction_day,
            "transaction_month": transaction_month,
            "age": age,
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return transaction
    
    except Exception as e:
        st.error(f"Error generating transaction: {e}")
        return None

# MQTT callbacks
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        st.session_state.mqtt_connected = True
    else:
        st.session_state.mqtt_connected = False

def on_message(client, userdata, msg):
    try:
        transaction = json.loads(msg.payload.decode())
        process_transaction(transaction)
    except Exception as e:
        st.error(f"Error processing message: {e}")

def process_transaction(transaction):
    """Process a transaction for the dashboard"""
    if not st.session_state.model_loaded:
        return
    
    if "timestamp" not in transaction:
        transaction["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    processed_transaction = preprocess_transaction(transaction)
    if processed_transaction is None:
        return
    
    fraud_probability = st.session_state.model.predict_proba(processed_transaction)[0, 1]
    prediction = 1 if fraud_probability > 0.5 else 0
    
    transaction["fraud_probability"] = fraud_probability
    transaction["fraud_predicted"] = prediction
    
    st.session_state.transactions.append(transaction)
    
    if prediction == 1:
        st.session_state.fraud_count += 1
    else:
        st.session_state.legitimate_count += 1
    
    category = transaction["category"]
    if category not in st.session_state.amounts_by_category:
        st.session_state.amounts_by_category[category] = 0
        st.session_state.fraud_by_category[category] = 0
    
    st.session_state.amounts_by_category[category] += transaction["amt"]
    if prediction == 1:
        st.session_state.fraud_by_category[category] += 1
    
    hour = transaction["transaction_hour"]
    st.session_state.transactions_by_hour[hour] += 1
    
    st.session_state.transaction_history.append({
        "timestamp": datetime.strptime(transaction["timestamp"], "%Y-%m-%d %H:%M:%S"),
        "amount": transaction["amt"],
        "fraud": prediction
    })
    
    st.session_state.fraud_probabilities.append(fraud_probability)
    st.session_state.last_update_time = datetime.now()

def simulate_transactions(count, interval):
    """Simulate a batch of transactions"""
    if not st.session_state.model_loaded:
        st.error("Please load the model first!")
        return
    
    for _ in range(count):
        transaction = generate_transaction(fraud_probability=0.1)
        if transaction:
            process_transaction(transaction)
            time.sleep(interval)
    st.rerun()

def start_mqtt_client():
    """Initialize and connect the MQTT client"""
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        client.connect("localhost", 1883, 60)
        client.subscribe("credit_card/transactions")
        client.loop_start()
        st.session_state.mqtt_client = client
        return True
    except Exception as e:
        st.error(f"Failed to connect to MQTT broker: {e}")
        return False

def stop_mqtt_client():
    """Stop and disconnect the MQTT client"""
    if st.session_state.mqtt_client:
        st.session_state.mqtt_client.loop_stop()
        st.session_state.mqtt_client.disconnect()
        st.session_state.mqtt_client = None

# Top row metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Transactions Processed", len(st.session_state.transactions))
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Fraudulent Transactions", st.session_state.fraud_count)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Legitimate Transactions", st.session_state.legitimate_count)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    fraud_rate = 0 if len(st.session_state.transactions) == 0 else (st.session_state.fraud_count / len(st.session_state.transactions) * 100)
    st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar controls
st.sidebar.title("Control Panel")

st.sidebar.subheader("Setup")
setup_button = st.sidebar.button("Load Model & Encoders")

if setup_button:
    with st.sidebar:
        with st.spinner("Loading model and encoders..."):
            if load_model_and_encoders():
                st.success("Model and encoders loaded successfully!")
            else:
                st.error("Failed to load model and encoders!")

st.sidebar.subheader("Connection")
mqtt_button = st.sidebar.button("Connect to MQTT Broker")

if mqtt_button:
    with st.sidebar:
        with st.spinner("Connecting to MQTT broker..."):
            if start_mqtt_client():
                st.success("Connected to MQTT broker!")
            else:
                st.error("Failed to connect to MQTT broker!")

st.sidebar.subheader("Simulation")
col1, col2 = st.sidebar.columns(2)
with col1:
    num_transactions = st.number_input("Number of transactions", min_value=1, max_value=100, value=10)
with col2:
    interval = st.number_input("Interval (seconds)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)

simulate_button = st.sidebar.button("Generate Transactions")

if simulate_button:
    with st.sidebar:
        with st.spinner(f"Generating {num_transactions} transactions..."):
            simulate_transactions(num_transactions, interval)
            st.success(f"Generated {num_transactions} transactions!")

if st.sidebar.button("Reset Dashboard"):
    st.session_state.transactions = deque(maxlen=100)
    st.session_state.fraud_count = 0
    st.session_state.legitimate_count = 0
    st.session_state.transaction_history = []
    st.session_state.fraud_probabilities = []
    st.session_state.amounts_by_category = {}
    st.session_state.fraud_by_category = {}
    st.session_state.transactions_by_hour = {hour: 0 for hour in range(24)}
    st.sidebar.success("Dashboard reset!")

# Visualizations
col1, col2 = st.columns(2)

with col1:
    st.subheader("Recent Transactions")
    if len(st.session_state.transactions) > 0:
        recent_df = pd.DataFrame(list(st.session_state.transactions)[-10:])
        if not recent_df.empty and 'amt' in recent_df.columns and 'fraud_predicted' in recent_df.columns:
            recent_df = recent_df[['timestamp', 'merchant', 'category', 'amt', 'fraud_probability', 'fraud_predicted']]
            recent_df.columns = ['Timestamp', 'Merchant', 'Category', 'Amount', 'Fraud Probability', 'Fraud Predicted']
            
            def highlight_fraud(row):
                if row['Fraud Predicted'] == 1:
                    return ['background-color: #ffe6e6; color: #000000'] * len(row)
                return ['background-color: #e6ffe6; color: #000000'] * len(row)
            
            styled_df = recent_df.style.apply(highlight_fraud, axis=1)
            st.dataframe(styled_df, use_container_width=True)
    else:
        st.info("No transactions yet. Generate some transactions or connect to MQTT.")

with col2:
    st.subheader("Fraud Probability Distribution")
    if len(st.session_state.fraud_probabilities) > 0:
        fig = px.histogram(
            x=st.session_state.fraud_probabilities,
            nbins=20,
            color_discrete_sequence=['#003087'],
            opacity=0.8
        )
        fig.update_layout(
            xaxis_title="Fraud Probability",
            yaxis_title="Count",
            bargap=0.1,
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No fraud probability data yet.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Transaction Amount vs Time")
    if len(st.session_state.transaction_history) > 0:
        time_df = pd.DataFrame(st.session_state.transaction_history)
        fig = px.scatter(
            time_df,
            x="timestamp",
            y="amount",
            color="fraud",
            color_discrete_map={1: "#cc0000", 0: "#006600"},
            hover_data=["amount", "fraud"],
            labels={"fraud": "Fraud Detected"},
            height=300,
        )
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Transaction Amount ($)",
            showlegend=True,
            legend_title="Transaction Type",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No transaction history yet.")

with col2:
    st.subheader("Fraud by Category")
    if st.session_state.fraud_by_category:
        category_df = pd.DataFrame({
            'Category': list(st.session_state.fraud_by_category.keys()),
            'Fraud Count': list(st.session_state.fraud_by_category.values()),
            'Total Amount': list(st.session_state.amounts_by_category.values())
        })
        category_df = category_df.sort_values('Fraud Count', ascending=False).head(10)
        fig = px.bar(
            category_df,
            x='Category',
            y='Fraud Count',
            color='Total Amount',
            color_continuous_scale='Blues',
            height=300,
        )
        fig.update_layout(
            xaxis_title="Category",
            yaxis_title="Number of Fraudulent Transactions",
            coloraxis_colorbar_title="Total Amount ($)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No category data yet.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Live Fraud Detection Feed")
    live_container = st.container()
    with live_container:
        for tx in list(st.session_state.transactions)[-5:]:
            if 'fraud_predicted' in tx and tx['fraud_predicted'] == 1:
                st.markdown(f"""
                <div class="fraud-alert">
                    <h4>FRAUD DETECTED</h4>
                    <p>Amount: <b>${tx['amt']:.2f}</b> | Merchant: {tx['merchant']} | Category: {tx['category']}</p>
                    <p>Probability: <b>{tx.get('fraud_probability', 0):.2%}</b> | Time: {tx.get('timestamp', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="legitimate-transaction">
                    <h4>LEGITIMATE TRANSACTION</h4>
                    <p>Amount: <b>${tx['amt']:.2f}</b> | Merchant: {tx['merchant']} | Category: {tx['category']}</p>
                    <p>Probability: <b>{tx.get('fraud_probability', 0):.2%}</b> | Time: {tx.get('timestamp', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)

with col2:
    st.subheader("Transactions by Hour")
    hours = list(st.session_state.transactions_by_hour.keys())
    counts = list(st.session_state.transactions_by_hour.values())
    fig = px.line(
        x=hours,
        y=counts,
        markers=True,
        line_shape="spline",
        color_discrete_sequence=['#003087'],
    )
    fig.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Number of Transactions",
        height=300,
    )
    fig.update_xaxes(tickmode='linear', tick0=0, dtick=2)
    st.plotly_chart(fig, use_container_width=True)

st.markdown(f"""
<div style="text-align: center; padding: 10px; color: #555555;">
    Last updated: {st.session_state.last_update_time.strftime("%Y-%m-%d %H:%M:%S")}
</div>
""", unsafe_allow_html=True)