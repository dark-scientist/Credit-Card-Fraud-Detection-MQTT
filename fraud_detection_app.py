import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import threading
import os
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import paho.mqtt.client as mqtt
import logging
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🔍",
    layout="wide"
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Initialize session state for transaction history
if 'transactions' not in st.session_state:
    st.session_state.transactions = []
if 'fraud_transactions' not in st.session_state:
    st.session_state.fraud_transactions = []
if 'legitimate_transactions' not in st.session_state:
    st.session_state.legitimate_transactions = []
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'mqtt_client' not in st.session_state:
    st.session_state.mqtt_client = None
if 'stats' not in st.session_state:
    st.session_state.stats = {
        'total': 0,
        'fraud': 0,
        'legitimate': 0
    }

# MQTT Settings
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "credit_card/transactions"

# Model and data paths
MODEL_PATH = "fraud_model.pkl"
TRAIN_DATA_PATH = "fraudTrain.csv"

# Check if model and data exist
@st.cache_resource
def check_files():
    files_exist = True
    files_to_check = [MODEL_PATH, TRAIN_DATA_PATH]
    missing_files = []
    
    for file in files_to_check:
        if not os.path.exists(file):
            files_exist = False
            missing_files.append(file)
    
    return files_exist, missing_files

# Load model and prepare encoders
@st.cache_resource
def load_model_and_encoders():
    try:
        # Load model
        xgb_model = joblib.load(MODEL_PATH)
        
        # Load sample of training data for encoders
        df = pd.read_csv(TRAIN_DATA_PATH, nrows=10000)
        df = df.dropna()
        
        # Date conversions
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        df['dob'] = pd.to_datetime(df['dob'])
        
        # Feature Engineering
        df['transaction_hour'] = df['trans_date_trans_time'].dt.hour
        df['transaction_day'] = df['trans_date_trans_time'].dt.day
        df['transaction_month'] = df['trans_date_trans_time'].dt.month
        df['age'] = df['trans_date_trans_time'].dt.year - df['dob'].dt.year
        df['geo_distance'] = np.sqrt((df['lat'] - df['merch_lat'])**2 + (df['long'] - df['merch_long'])**2)
        
        # Drop unnecessary columns
        drop_cols = ['trans_date_trans_time', 'dob', 'Unnamed: 0', 'first', 'last', 'street', 'trans_num', 'unix_time', 'cc_num']
        df = df.drop(columns=drop_cols, errors='ignore')
        
        # Prepare label encoders
        categorical_cols = ['merchant', 'category', 'gender', 'city', 'state', 'job']
        label_encoders = {}
        for col in categorical_cols:
            label_encoders[col] = LabelEncoder()
            label_encoders[col].fit(df[col].astype(str))
        
        # Define selected features for the model
        selected_features = ['amt', 'geo_distance', 'transaction_hour', 'transaction_day', 
                            'transaction_month', 'age', 'city_pop', 'merchant', 'category', 
                            'gender', 'city', 'state', 'job']
        
        # Encode categorical features
        df_encoded = df.copy()
        for col in categorical_cols:
            df_encoded[col] = label_encoders[col].transform(df_encoded[col].astype(str))
        
        # Fit the scaler
        scaler = StandardScaler()
        scaler.fit(df_encoded[selected_features])
        
        return xgb_model, label_encoders, scaler, selected_features, categorical_cols
    
    except Exception as e:
        st.error(f"Error loading model or encoders: {e}")
        return None, None, None, None, None

# Extract unique values from training data
@st.cache_data
def get_reference_data():
    df = pd.read_csv(TRAIN_DATA_PATH, nrows=10000)
    
    # Extract unique values for categorical features
    merchants = df['merchant'].unique().tolist()
    categories = df['category'].unique().tolist()
    cities = df['city'].unique().tolist()
    states = df['state'].unique().tolist()
    jobs = df['job'].unique().tolist()
    genders = df['gender'].unique().tolist()
    
    # Get ranges for numerical features
    amt_min, amt_max = df['amt'].min(), df['amt'].max()
    city_pop_min, city_pop_max = df['city_pop'].min(), df['city_pop'].max()
    lat_min, lat_max = df['lat'].min(), df['lat'].max()
    long_min, long_max = df['long'].min(), df['long'].max()
    merch_lat_min, merch_lat_max = df['merch_lat'].min(), df['merch_lat'].max()
    merch_long_min, merch_long_max = df['merch_long'].min(), df['merch_long'].max()
    
    # Calculate ranges for geo_distance
    df['geo_distance'] = np.sqrt((df['lat'] - df['merch_lat'])**2 + (df['long'] - df['merch_long'])**2)
    geo_distance_min, geo_distance_max = df['geo_distance'].min(), df['geo_distance'].max()
    
    # Calculate age range
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = df['trans_date_trans_time'].dt.year - df['dob'].dt.year
    age_min, age_max = df['age'].min(), df['age'].max()
    
    return {
        'merchants': merchants,
        'categories': categories,
        'cities': cities,
        'states': states,
        'jobs': jobs,
        'genders': genders,
        'amt_min': amt_min,
        'amt_max': amt_max,
        'city_pop_min': city_pop_min,
        'city_pop_max': city_pop_max,
        'lat_min': lat_min,
        'lat_max': lat_max,
        'long_min': long_min,
        'long_max': long_max,
        'merch_lat_min': merch_lat_min,
        'merch_lat_max': merch_lat_max,
        'geo_distance_min': geo_distance_min,
        'geo_distance_max': geo_distance_max,
        'age_min': age_min,
        'age_max': age_max
    }

# Function to generate a random transaction
def generate_transaction(ref_data, fraud_probability=0.05):
    # Determine if this transaction is fraudulent
    is_fraud = random.random() < fraud_probability
    
    # Current timestamp
    current_time = datetime.now()
    trans_date_trans_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    unix_time = int(current_time.timestamp())
    
    # Person info
    gender = random.choice(ref_data['genders'])
    dob = (current_time - timedelta(days=random.randint(int(ref_data['age_min']*365), int(ref_data['age_max']*365)))).strftime("%Y-%m-%d")
    
    # Random credit card number
    cc_num = f"{''.join([str(random.randint(0, 9)) for _ in range(16)])}"
    
    # Transaction info
    merchant = random.choice(ref_data['merchants'])
    category = random.choice(ref_data['categories'])
    
    # Location info
    city = random.choice(ref_data['cities'])
    state = random.choice(ref_data['states'])
    job = random.choice(ref_data['jobs'])
    
    # Amount logic
    if is_fraud:
        amt = round(random.uniform(ref_data['amt_max'] * 0.7, ref_data['amt_max']), 2)
    else:
        amt = round(random.uniform(ref_data['amt_min'], ref_data['amt_max'] * 0.7), 2)
    
    # Coordinates
    lat = random.uniform(ref_data['lat_min'], ref_data['lat_max'])
    long = random.uniform(ref_data['long_min'], ref_data['long_max'])
    merch_lat = random.uniform(ref_data['merch_lat_min'], ref_data['merch_lat_max'])
    merch_long = random.uniform(ref_data['merch_long_min'], ref_data['merch_long_max'])
    
    # Calculated geo_distance
    geo_distance = np.sqrt((lat - merch_lat)**2 + (long - merch_long)**2)
    
    # For fraudulent transactions, sometimes make geo_distance very large
    if is_fraud and random.random() < 0.7:
        geo_distance = random.uniform(ref_data['geo_distance_max'] * 0.8, ref_data['geo_distance_max'] * 1.2)
    
    # Calculate transaction hour, day, month
    transaction_hour = current_time.hour
    transaction_day = current_time.day
    transaction_month = current_time.month
    
    # Calculate age from DOB
    age = current_time.year - datetime.strptime(dob, "%Y-%m-%d").year
    
    # Other fields
    zip_code = f"{random.randint(10000, 99999)}"
    city_pop = random.randint(int(ref_data['city_pop_min']), int(ref_data['city_pop_max']))
    first = "SimFirst"
    last = "SimLast"
    street = "123 Sim Street"
    trans_num = f"T{random.randint(100000, 999999)}"
    
    # Create transaction dictionary
    transaction = {
        "trans_date_trans_time": trans_date_trans_time,
        "cc_num": cc_num,
        "merchant": merchant,
        "category": category,
        "amt": amt,
        "first": first,
        "last": last,
        "gender": gender,
        "street": street,
        "city": city,
        "state": state,
        "zip": zip_code,
        "lat": lat,
        "long": long,
        "city_pop": city_pop,
        "job": job,
        "dob": dob,
        "trans_num": trans_num,
        "unix_time": unix_time,
        "merch_lat": merch_lat,
        "merch_long": merch_long,
        "is_fraud": int(is_fraud)
    }
    
    # Add calculated features
    transaction_with_features = transaction.copy()
    transaction_with_features.update({
        "geo_distance": geo_distance,
        "transaction_hour": transaction_hour, 
        "transaction_day": transaction_day,
        "transaction_month": transaction_month,
        "age": age
    })
    
    return transaction_with_features

# Function to preprocess transaction for the model
def preprocess_transaction(transaction, label_encoders, scaler, selected_features, categorical_cols):
    try:
        # Create a dataframe with the transaction
        data = {}
        for key in selected_features:
            if key in transaction:
                data[key] = [transaction[key]]
            else:
                logging.warning(f"Missing feature: {key}")
                return None
        
        df_trans = pd.DataFrame(data)
        
        # Apply label encoding to categorical features
        for col in categorical_cols:
            if col in df_trans.columns:
                try:
                    df_trans[col] = label_encoders[col].transform(df_trans[col].astype(str))
                except ValueError:
                    # Use -1 for unknown values
                    logging.warning(f"Unknown value in {col}: {df_trans[col].iloc[0]}")
                    df_trans[col] = -1
        
        # Apply scaling
        X_scaled = scaler.transform(df_trans[selected_features])
        df_scaled = pd.DataFrame(X_scaled, columns=selected_features)
        
        return df_scaled
    
    except Exception as e:
        logging.error(f"Error preprocessing transaction: {e}")
        return None

# MQTT Callback when a message is received
def on_message(client, userdata, msg):
    try:
        # Parse the JSON message
        transaction = json.loads(msg.payload.decode())
        
        # Get processed transaction
        processed_transaction = preprocess_transaction(
            transaction, 
            st.session_state.label_encoders, 
            st.session_state.scaler, 
            st.session_state.selected_features, 
            st.session_state.categorical_cols
        )
        
        if processed_transaction is None:
            logging.warning("Failed to process transaction")
            return
            
        # Make a prediction
        fraud_probability = st.session_state.model.predict_proba(processed_transaction)[0, 1]
        prediction = 1 if fraud_probability > 0.5 else 0
        
        # Add to transaction history
        transaction_info = {
            'merchant': transaction['merchant'],
            'category': transaction['category'],
            'amount': transaction['amt'],
            'time': transaction['trans_date_trans_time'],
            'actual_fraud': transaction['is_fraud'],
            'predicted_fraud': prediction,
            'probability': fraud_probability
        }
        
        st.session_state.transactions.append(transaction_info)
        
        # Update stats
        st.session_state.stats['total'] += 1
        
        if prediction == 1:
            st.session_state.fraud_transactions.append(transaction_info)
            st.session_state.stats['fraud'] += 1
        else:
            st.session_state.legitimate_transactions.append(transaction_info)
            st.session_state.stats['legitimate'] += 1
            
    except Exception as e:
        logging.error(f"Error processing message: {e}")

# MQTT connection callback
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        logging.info("Connected to MQTT Broker!")
        client.subscribe(MQTT_TOPIC)
        logging.info(f"Subscribed to topic: {MQTT_TOPIC}")
    else:
        logging.error(f"Connection failed with code {rc}")

# Function to start MQTT client
def start_mqtt_client():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
        return client
    except Exception as e:
        st.error(f"Failed to connect to MQTT Broker: {e}")
        return None

# Function to publish transaction to MQTT
def publish_transaction(client, transaction):
    if client is None:
        st.error("MQTT client not connected")
        return False
    
    try:
        # Convert transaction to JSON
        transaction_json = json.dumps(transaction)
        
        # Publish to MQTT topic
        result = client.publish(MQTT_TOPIC, transaction_json, qos=1)
        return result.rc == 0
    except Exception as e:
        st.error(f"Error publishing transaction: {e}")
        return False

# Function to continuously generate transactions
def generate_transactions_continuously():
    try:
        while st.session_state.is_running:
            # Generate a transaction
            transaction = generate_transaction(st.session_state.ref_data, fraud_probability=0.05)
            
            # Publish to MQTT
            success = publish_transaction(st.session_state.mqtt_client, transaction)
            
            if not success:
                st.error("Failed to publish transaction")
                st.session_state.is_running = False
                break
                
            # Wait before generating the next transaction
            time.sleep(0.5)
    except Exception as e:
        st.error(f"Error in transaction generation: {e}")
        st.session_state.is_running = False

# Main app
def main():
    # App title and header
    st.title("Credit Card Fraud Detection System")
    st.markdown("### Real-time transaction monitoring with ML-based fraud detection")
    
    # Check if required files exist
    files_exist, missing_files = check_files()
    if not files_exist:
        st.error(f"Missing required files: {', '.join(missing_files)}")
        st.stop()
    
    # Load model and encoders if not already loaded
    if 'model' not in st.session_state:
        with st.spinner("Loading model and encoders..."):
            model, label_encoders, scaler, selected_features, categorical_cols = load_model_and_encoders()
            
            if model is None:
                st.error("Failed to load model and encoders")
                st.stop()
                
            st.session_state.model = model
            st.session_state.label_encoders = label_encoders
            st.session_state.scaler = scaler
            st.session_state.selected_features = selected_features
            st.session_state.categorical_cols = categorical_cols
    
    # Load reference data if not already loaded
    if 'ref_data' not in st.session_state:
        with st.spinner("Loading reference data..."):
            st.session_state.ref_data = get_reference_data()
    
    # Initialize MQTT client if not already initialized
    if st.session_state.mqtt_client is None:
        with st.spinner("Connecting to MQTT broker..."):
            st.session_state.mqtt_client = start_mqtt_client()
            if st.session_state.mqtt_client is None:
                st.error("Failed to connect to MQTT broker")
                st.stop()
    
    # Create dashboard layout
    col1, col2 = st.columns([1, 3])
    
    # Control panel
    with col1:
        st.subheader("Control Panel")
        
        # Generate button
        if not st.session_state.is_running:
            if st.button("🚀 Generate Transactions", type="primary"):
                st.session_state.is_running = True
                threading.Thread(target=generate_transactions_continuously, daemon=True).start()
        else:
            if st.button("⏹️ Stop Generation", type="secondary"):
                st.session_state.is_running = False
        
        # Stats
        st.subheader("Statistics")
        total_col, fraud_col, legit_col = st.columns(3)
        with total_col:
            st.metric("Total", st.session_state.stats['total'])
        with fraud_col:
            st.metric("Fraud", st.session_state.stats['fraud'])
        with legit_col:
            st.metric("Legitimate", st.session_state.stats['legitimate'])
        
        # Fraud rate
        if st.session_state.stats['total'] > 0:
            fraud_rate = (st.session_state.stats['fraud'] / st.session_state.stats['total']) * 100
            st.progress(fraud_rate / 100, text=f"Fraud Rate: {fraud_rate:.1f}%")
    
    # Transaction monitor
    with col2:
        st.subheader("Transaction Monitor")
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["All Transactions", "Fraud Alerts", "Legitimate"])
        
        # All transactions tab
        with tab1:
            if st.session_state.transactions:
                # Take the last 10 transactions for display
                display_transactions = st.session_state.transactions[-10:][::-1]
                
                for tx in display_transactions:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        emoji = "🚨" if tx['predicted_fraud'] == 1 else "✅"
                        st.write(f"{emoji} **{tx['merchant']}** - {tx['category']}")
                        st.caption(f"Time: {tx['time']}")
                    
                    with col2:
                        st.write(f"**${tx['amount']:.2f}**")
                    
                    with col3:
                        color = "red" if tx['predicted_fraud'] == 1 else "green"
                        st.markdown(f"<p style='color:{color};font-weight:bold;'>{tx['probability']*100:.1f}% Risk</p>", unsafe_allow_html=True)
                    
                    st.divider()
            else:
                st.info("No transactions yet. Click 'Generate Transactions' to start.")
        
        # Fraud alerts tab
        with tab2:
            if st.session_state.fraud_transactions:
                # Take the last 10 fraudulent transactions
                display_frauds = st.session_state.fraud_transactions[-10:][::-1]
                
                for tx in display_frauds:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.write(f"🚨 **{tx['merchant']}** - {tx['category']}")
                        st.caption(f"Time: {tx['time']}")
                    
                    with col2:
                        st.write(f"**${tx['amount']:.2f}**")
                    
                    with col3:
                        st.markdown(f"<p style='color:red;font-weight:bold;'>{tx['probability']*100:.1f}% Risk</p>", unsafe_allow_html=True)
                    
                    st.divider()
            else:
                st.info("No fraud alerts yet.")
        
        # Legitimate transactions tab
        with tab3:
            if st.session_state.legitimate_transactions:
                # Take the last 10 legitimate transactions
                display_legitimate = st.session_state.legitimate_transactions[-10:][::-1]
                
                for tx in display_legitimate:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.write(f"✅ **{tx['merchant']}** - {tx['category']}")
                        st.caption(f"Time: {tx['time']}")
                    
                    with col2:
                        st.write(f"**${tx['amount']:.2f}**")
                    
                    with col3:
                        st.markdown(f"<p style='color:green;font-weight:bold;'>{tx['probability']*100:.1f}% Risk</p>", unsafe_allow_html=True)
                    
                    st.divider()
            else:
                st.info("No legitimate transactions yet.")

if __name__ == "__main__":
    # Check for missing imports
    import random
    from datetime import timedelta
    
    main()