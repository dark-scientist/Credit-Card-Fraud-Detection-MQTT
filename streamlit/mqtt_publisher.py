import random
import time
import json
from datetime import datetime, timedelta
import logging
import paho.mqtt.client as mqtt
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(filename="mqtt_publisher.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Load training data to get valid categories and ranges
df = pd.read_csv("fraudTrain.csv")

# Extract unique values for categorical features
MERCHANTS = df['merchant'].unique().tolist()
CATEGORIES = df['category'].unique().tolist()
CITIES = df['city'].unique().tolist()
STATES = df['state'].unique().tolist()
JOBS = df['job'].unique().tolist()
GENDERS = df['gender'].unique().tolist()

# Get ranges for numerical features to make simulation more realistic
AMT_MIN, AMT_MAX = df['amt'].min(), df['amt'].max()
CITY_POP_MIN, CITY_POP_MAX = df['city_pop'].min(), df['city_pop'].max()
LAT_MIN, LAT_MAX = df['lat'].min(), df['lat'].max()
LONG_MIN, LONG_MAX = df['long'].min(), df['long'].max()
MERCH_LAT_MIN, MERCH_LAT_MAX = df['merch_lat'].min(), df['merch_lat'].max()
MERCH_LONG_MIN, MERCH_LONG_MAX = df['merch_long'].min(), df['merch_long'].max()

# Calculate ranges for derived features
df['geo_distance'] = np.sqrt((df['lat'] - df['merch_lat'])**2 + (df['long'] - df['merch_long'])**2)
GEO_DISTANCE_MIN, GEO_DISTANCE_MAX = df['geo_distance'].min(), df['geo_distance'].max()

df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['dob'] = pd.to_datetime(df['dob'])
df['age'] = df['trans_date_trans_time'].dt.year - df['dob'].dt.year
AGE_MIN, AGE_MAX = df['age'].min(), df['age'].max()

# MQTT Settings
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "credit_card/transactions"

# Initialize MQTT Client
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

# Try connecting to MQTT broker
try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    print("✅ Connected to MQTT Broker successfully!")
except Exception as e:
    print(f"❌ Failed to connect to MQTT Broker: {e}")
    exit(1)

# Function to generate a random fraud or legitimate transaction
def generate_transaction(fraud_probability=0.05):
    # Determine if this transaction is fraudulent based on probability
    is_fraud = random.random() < fraud_probability
    
    # Current timestamp
    current_time = datetime.now()
    trans_date_trans_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    unix_time = int(current_time.timestamp())
    
    # Person info
    gender = random.choice(GENDERS)
    dob = (current_time - timedelta(days=random.randint(int(AGE_MIN*365), int(AGE_MAX*365)))).strftime("%Y-%m-%d")
    
    # Create a random, believable cc_num (not real)
    cc_num = f"{''.join([str(random.randint(0, 9)) for _ in range(16)])}"
    
    # Transaction info
    merchant = random.choice(MERCHANTS)
    category = random.choice(CATEGORIES)
    
    # Location info
    city = random.choice(CITIES)
    state = random.choice(STATES)
    job = random.choice(JOBS)
    
    # Amount logic (fraudulent transactions tend to be larger)
    if is_fraud:
        amt = round(random.uniform(AMT_MAX * 0.7, AMT_MAX), 2)  # Biased toward higher amounts
    else:
        amt = round(random.uniform(AMT_MIN, AMT_MAX * 0.7), 2)
    
    # Coordinates
    lat = random.uniform(LAT_MIN, LAT_MAX)
    long = random.uniform(LONG_MIN, LONG_MAX)
    merch_lat = random.uniform(MERCH_LAT_MIN, MERCH_LAT_MAX)
    merch_long = random.uniform(MERCH_LONG_MIN, MERCH_LONG_MAX)
    
    # Calculated geo_distance
    geo_distance = np.sqrt((lat - merch_lat)**2 + (long - merch_long)**2)
    
    # For fraud transactions, sometimes make the geo_distance very large
    if is_fraud and random.random() < 0.7:
        geo_distance = random.uniform(GEO_DISTANCE_MAX * 0.8, GEO_DISTANCE_MAX * 1.2)
    
    # Calculate transaction hour, day, month for the model
    transaction_hour = current_time.hour
    transaction_day = current_time.day
    transaction_month = current_time.month
    
    # Calculate age from DOB
    age = current_time.year - datetime.strptime(dob, "%Y-%m-%d").year
    
    # Other fields
    zip_code = f"{random.randint(10000, 99999)}"
    city_pop = random.randint(int(CITY_POP_MIN), int(CITY_POP_MAX))
    first = "SimFirst"  # dummy first name
    last = "SimLast"    # dummy last name
    street = "123 Sim Street"  # dummy street
    trans_num = f"T{random.randint(100000, 999999)}"
    
    # Create transaction dictionary in the SAME format as the training data
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
    
    # Also add the calculated features that the model expects
    transaction_with_features = transaction.copy()
    transaction_with_features.update({
        "geo_distance": geo_distance,
        "transaction_hour": transaction_hour, 
        "transaction_day": transaction_day,
        "transaction_month": transaction_month,
        "age": age
    })
    
    return transaction_with_features

# Start streaming transactions
print("🔹 Streaming transactions to MQTT... (Press Ctrl+C to stop)")
print("🔹 Every 20th transaction will be logged to CSV as well")

# Create/open a CSV file to log some transactions
csv_file = open('simulated_transactions.csv', 'w')
csv_headers = ','.join(list(df.columns))
csv_file.write(f"{csv_headers}\n")
csv_file.flush()

transaction_count = 0

try:
    while True:
        # Generate transaction with 5% fraud probability
        transaction = generate_transaction(fraud_probability=0.05)
        
        # Separate base transaction from calculated features
        base_transaction = {k: v for k, v in transaction.items() if k in df.columns}
        
        # Convert to JSON format for MQTT
        transaction_json = json.dumps(transaction)
        
        # Publish to MQTT topic with QoS=1
        result = client.publish(MQTT_TOPIC, transaction_json, qos=1)
        
        # Check if publish was successful
        if result.rc == 0:
            is_fraud_str = "🚨 FRAUD" if transaction["is_fraud"] == 1 else "✅ LEGITIMATE"
            print(f"Published: {is_fraud_str} - Amount: ${transaction['amt']:.2f} - {transaction['merchant']}")
            logging.info(f"Published: {transaction_json}")
            
            # Log every 20th transaction to CSV
            transaction_count += 1
            if transaction_count % 20 == 0:
                csv_line = ','.join([str(base_transaction.get(col, '')) for col in df.columns])
                csv_file.write(f"{csv_line}\n")
                csv_file.flush()
                print(f"💾 Logged transaction #{transaction_count} to CSV")
        else:
            print("⚠️ Failed to publish message.")
        
        # Simulate delay for real-time effect (faster than 1 sec)
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\n⛔ Stopping transaction stream...")
    client.disconnect()
    csv_file.close()
    print("✅ MQTT Client Disconnected. CSV file closed.")