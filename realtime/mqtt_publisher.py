




import random
import time
import json
from datetime import datetime, timedelta
import logging
import paho.mqtt.client as mqtt
import pandas as pd
import numpy as np
from queue import Queue
import threading

# Setup logging
logging.basicConfig(filename="mqtt_publisher.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Load training data
df = pd.read_csv("fraudTrain.csv")

# Extract unique values and ranges
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

# MQTT Settings
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "credit_card/transactions"
BUFFER_SIZE = 1000

message_queue = Queue(maxsize=BUFFER_SIZE)
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

def generate_transaction(fraud_probability=0.05):
    is_fraud = random.random() < fraud_probability
    current_time = datetime.now()
    trans_date_trans_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    unix_time = int(current_time.timestamp())

    gender = random.choice(GENDERS)
    dob = (current_time - timedelta(days=random.randint(int(AGE_MIN*365), int(AGE_MAX*365)))).strftime("%Y-%m-%d")
    cc_num = f"{''.join([str(random.randint(0, 9)) for _ in range(16)])}"
    merchant = random.choice(MERCHANTS)
    category = random.choice(CATEGORIES)
    city = random.choice(CITIES)
    state = random.choice(STATES)
    job = random.choice(JOBS)

    amt = round(random.uniform(AMT_MAX * 0.7, AMT_MAX) if is_fraud else random.uniform(AMT_MIN, AMT_MAX * 0.7), 2)
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
    age = current_time.year - datetime.strptime(dob, "%Y-%m-%d").year

    transaction = {
        "trans_date_trans_time": trans_date_trans_time,
        "cc_num": cc_num,
        "merchant": merchant,
        "category": category,
        "amt": amt,
        "gender": gender,
        "city": city,
        "state": state,
        "lat": lat,
        "long": long,
        "merch_lat": merch_lat,
        "merch_long": merch_long,
        "is_fraud": int(is_fraud),
        "geo_distance": geo_distance,
        "transaction_hour": transaction_hour,
        "transaction_day": transaction_day,
        "transaction_month": transaction_month,
        "age": age
    }
    return transaction

def publish_messages():
    transaction_count = 0
    csv_file = open('simulated_transactions.csv', 'w')
    csv_headers = ','.join(list(pd.read_csv("fraudTrain.csv").columns))
    csv_file.write(f"{csv_headers}\n")
    csv_file.flush()

    while True:
        try:
            transaction = generate_transaction()
            transaction_json = json.dumps(transaction)
            if message_queue.full():
                message_queue.get()  # Drop oldest if buffer full
            message_queue.put(transaction_json)
            print(f"📤 Queued transaction: ${transaction['amt']:.2f} - {transaction['merchant']}")
            time.sleep(0.1)  # Faster rate for testing

            transaction_count += 1
            if transaction_count % 20 == 0:
                base_transaction = {k: transaction[k] for k in pd.read_csv("fraudTrain.csv").columns}
                csv_line = ','.join([str(base_transaction.get(col, '')) for col in pd.read_csv("fraudTrain.csv").columns])
                csv_file.write(f"{csv_line}\n")
                csv_file.flush()
                print(f"💾 Logged transaction #{transaction_count} to CSV")
        except Exception as e:
            print(f"⚠️ Error generating transaction: {e}")

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("✅ Connected to MQTT Broker!")
        client.subscribe(MQTT_TOPIC)
    else:
        print(f"❌ Connection failed with code {rc}")

def on_disconnect(client, userdata, rc):
    print("⚠️ Disconnected from MQTT Broker. Attempting to reconnect...")
    while rc != 0:
        try:
            client.reconnect()
            break
        except Exception:
            time.sleep(5)

def mqtt_publish_loop():
    while True:
        if not message_queue.empty():
            message = message_queue.get()
            result = client.publish(MQTT_TOPIC, message, qos=2)
            if result.rc == 0:
                print(f"✅ Published message")
            else:
                print(f"⚠️ Failed to publish message, rc={result.rc}")
                message_queue.put(message)  # Re-queue if failed
        time.sleep(0.01)

try:
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    publish_thread = threading.Thread(target=publish_messages, daemon=True)
    publish_thread.start()
    mqtt_publish_thread = threading.Thread(target=mqtt_publish_loop, daemon=True)
    mqtt_publish_thread.start()
    client.loop_forever()
except KeyboardInterrupt:
    print("\n⛔ Stopping transaction stream...")
    client.disconnect()
    print("✅ MQTT Client Disconnected.")