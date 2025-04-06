import os
import json
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
import time  # Added this import to fix the undefined variable error
from joblib import load
import paho.mqtt.client as mqtt

# Setup logging
logging.basicConfig(filename="mqtt_subscriber.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Also log to a separate file for fraud events
fraud_logger = logging.getLogger('fraud_logger')
fraud_logger.setLevel(logging.WARNING)
fraud_handler = logging.FileHandler('fraud_events.log')
fraud_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
fraud_logger.addHandler(fraud_handler)

# Load the trained fraud detection model
MODEL_PATH = "fraud_model.pkl"

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model file '{MODEL_PATH}' not found! Run bigtrain.py first.")
    exit(1)

try:
    xgb_model = load(MODEL_PATH)
    print("✅ Fraud detection model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading fraud detection model: {e}")
    exit(1)

# Load a small sample of training data to fit the encoders
TRAIN_DATA_PATH = "fraudTrain.csv"
if not os.path.exists(TRAIN_DATA_PATH):
    print(f"❌ Training data file '{TRAIN_DATA_PATH}' not found!")
    exit(1)

try:
    # Load only the first 10000 rows to save memory
    df = pd.read_csv(TRAIN_DATA_PATH, nrows=10000)
    print("✅ Training data sample loaded successfully!")
except Exception as e:
    print(f"❌ Error loading training data: {e}")
    exit(1)

# Preprocess the training data to fit the encoders (same as in bigtrain.py)
print("🔧 Preparing encoders and scalers...")
df = df.dropna()
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
    print(f"✅ Label encoder fitted for {col}")

# Define features that the model expects
selected_features = ['amt', 'geo_distance', 'transaction_hour', 'transaction_day', 
                    'transaction_month', 'age', 'city_pop', 'merchant', 'category', 
                    'gender', 'city', 'state', 'job']

# Encode categorical features
df_encoded = df.copy()
for col in categorical_cols:
    df_encoded[col] = label_encoders[col].transform(df_encoded[col].astype(str))

# Fit the scaler on the training data
scaler = StandardScaler()
scaler.fit(df_encoded[selected_features])
print("✅ Scaler fitted on training data")

# MQTT Settings
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "credit_card/transactions"

# Open a file to log detected frauds in CSV format
fraud_csv = open('detected_frauds.csv', 'w')
fraud_csv.write("timestamp,merchant,category,amount,gender,city,state,job,is_fraud_actual,is_fraud_predicted,fraud_probability\n")
fraud_csv.flush()

# Function to process a transaction for the model
def preprocess_transaction(transaction):
    try:
        # Create a dataframe with the incoming transaction
        data = {}
        for key in selected_features:
            if key in transaction:
                data[key] = [transaction[key]]
            else:
                print(f"⚠️ Missing feature: {key}")
                return None
        
        df_trans = pd.DataFrame(data)
        
        # Apply label encoding to categorical features
        for col in categorical_cols:
            if col in df_trans.columns:
                try:
                    df_trans[col] = label_encoders[col].transform(df_trans[col].astype(str))
                except ValueError:
                    # If value not in encoder, use -1 as fallback
                    print(f"⚠️ Unknown value in {col}: {df_trans[col].iloc[0]}")
                    df_trans[col] = -1
        
        # Apply scaling to all features
        X_scaled = scaler.transform(df_trans[selected_features])
        df_scaled = pd.DataFrame(X_scaled, columns=selected_features)
        
        return df_scaled
    
    except Exception as e:
        print(f"⚠️ Error preprocessing transaction: {e}")
        return None

# MQTT Callback when a message is received
def on_message(client, userdata, msg):
    try:
        # Parse the JSON message
        transaction = json.loads(msg.payload.decode())
        
        # For fraud detection, we need the derived features
        # Preprocess the transaction for the model
        processed_transaction = preprocess_transaction(transaction)
        
        if processed_transaction is None:
            print("⚠️ Failed to process transaction, skipping...")
            return
            
        # Make a prediction
        fraud_probability = xgb_model.predict_proba(processed_transaction)[0, 1]
        prediction = 1 if fraud_probability > 0.5 else 0
        
        # Get actual fraud value if available
        actual_fraud = transaction.get("is_fraud", "unknown")
        
        # Format timestamp nicely
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Log the transaction result
        if prediction == 1:
            print(f"🚨 FRAUD DETECTED: ${transaction['amt']:.2f} - {transaction['merchant']} - {fraud_probability:.4f} probability")
            fraud_logger.warning(f"FRAUD: ${transaction['amt']:.2f} - {transaction['merchant']} - {fraud_probability:.4f} probability")
            
            # Log to CSV
            fraud_csv.write(f"{timestamp},{transaction['merchant']},{transaction['category']},{transaction['amt']},{transaction['gender']},{transaction['city']},{transaction['state']},{transaction['job']},{actual_fraud},{prediction},{fraud_probability:.4f}\n")
            fraud_csv.flush()
        else:
            print(f"✅ LEGITIMATE: ${transaction['amt']:.2f} - {transaction['merchant']} - {fraud_probability:.4f} probability")
            logging.info(f"OK: ${transaction['amt']:.2f} - {transaction['merchant']}")
        
        # Check if our prediction matches actual fraud status (for testing)
        if actual_fraud != "unknown":
            if int(actual_fraud) == prediction:
                print(f"✓ CORRECT PREDICTION: Actual={actual_fraud}, Predicted={prediction}")
            else:
                print(f"✗ INCORRECT PREDICTION: Actual={actual_fraud}, Predicted={prediction}")
                
    except Exception as e:
        print(f"⚠️ Error processing message: {e}")

# MQTT connection callback
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("✅ Connected to MQTT Broker!")
        client.subscribe(MQTT_TOPIC)
        print(f"✅ Subscribed to topic: {MQTT_TOPIC}")
    else:
        print(f"❌ Connection failed with code {rc}")

# Initialize MQTT client
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message

# Connect to the MQTT broker
try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    print(f"🔌 Connecting to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}...")
except Exception as e:
    print(f"❌ Failed to connect to MQTT Broker: {e}")
    exit(1)

print("📡 Fraud detection service is running... (Press Ctrl+C to stop)")

try:
    client.loop_forever()
except KeyboardInterrupt:
    print("\n⛔ Stopping fraud detection service...")
    fraud_csv.close()
    print("✅ Fraud CSV file closed.")