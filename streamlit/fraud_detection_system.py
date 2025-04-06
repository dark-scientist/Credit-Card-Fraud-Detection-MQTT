import os
import time
import subprocess
import sys

def check_requirements():
    """Check that all required files and libraries are available"""
    print("🔍 Checking requirements...")
    
    # Check if Python packages are installed
    try:
        import pandas
        import numpy
        import joblib
        import paho.mqtt.client
        import xgboost
        import sklearn
        import matplotlib
        print("✅ All required Python packages are installed.")
    except ImportError as e:
        print(f"❌ Missing Python package: {e}")
        print("Try running: pip install pandas numpy joblib paho-mqtt xgboost scikit-learn matplotlib")
        return False
    
    # Check if training data exists
    if not os.path.exists("fraudTrain.csv"):
        print("❌ Training data file 'fraudTrain.csv' not found!")
        print("Download the dataset from Kaggle: https://www.kaggle.com/datasets/kartik2112/fraud-detection")
        return False
    
    # Check if Python files exist
    if not os.path.exists("mqtt_publisher.py"):
        print("❌ File 'mqtt_publisher.py' not found!")
        return False
    
    if not os.path.exists("mqtt_subscriber.py"):
        print("❌ File 'mqtt_subscriber.py' not found!")
        return False
    
    if not os.path.exists("bigtrain.py"):
        print("❌ File 'bigtrain.py' not found!")
        return False
    
    # Check if model exists, if not we need to train it
    if not os.path.exists("fraud_model.pkl"):
        print("⚠️ Model file 'fraud_model.pkl' not found! Will train a new model.")
    else:
        print("✅ Existing model 'fraud_model.pkl' found.")
    
    return True

def train_model():
    """Train the fraud detection model"""
    print("\n🧠 Training fraud detection model...")
    print("This may take several minutes depending on your computer's performance.\n")
    
    try:
        # Run the training script
        subprocess.run([sys.executable, "bigtrain.py"], check=True)
        
        # Check if model was created
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
    """Start the MQTT broker if it's not running"""
    print("\n🔄 Checking MQTT broker status...")
    
    # Try to check if broker is running
    try:
        import paho.mqtt.client as mqtt
        
        def on_connect(client, userdata, flags, rc, properties=None):
            if rc == 0:
                print("✅ MQTT broker is running!")
            else:
                print(f"❌ MQTT broker connection failed with code {rc}")
        
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
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
        print("- On Ubuntu: sudo apt-get install mosquitto")
        print("- On Windows: Download from https://mosquitto.org/download/")
        print("- On macOS: brew install mosquitto")
        return False

def run_fraud_detection_system():
    """Run the complete fraud detection system"""
    if not check_requirements():
        return

    # Train model if needed
    if not os.path.exists("fraud_model.pkl"):
        if not train_model():
            return
    
    # Check MQTT broker
    if not start_mqtt_broker():
        return
    
    print("\n🚀 Starting Fraud Detection System...")
    print("This will start two processes in separate terminals:")
    print("1. Transaction Publisher - Generates simulated transactions")
    print("2. Fraud Detector - Processes transactions and detects fraud")
    
    try:
        # Start the subscriber in a new terminal
        if sys.platform.startswith('win'):
            subscriber_cmd = f'start cmd /k "{sys.executable} mqtt_subscriber.py"'
        else:  # Linux/macOS
            subscriber_cmd = f"gnome-terminal -- {sys.executable} mqtt_subscriber.py"
        
        print(f"\n📡 Starting Fraud Detection Service...")
        os.system(subscriber_cmd)
        
        # Give the subscriber time to initialize
        time.sleep(2)
        
        # Start the publisher in a new terminal
        if sys.platform.startswith('win'):
            publisher_cmd = f'start cmd /k "{sys.executable} mqtt_publisher.py"'
        else:  # Linux/macOS
            publisher_cmd = f"gnome-terminal -- {sys.executable} mqtt_publisher.py"
        
        print(f"\n📤 Starting Transaction Generator...")
        os.system(publisher_cmd)
        
        print("\n✅ Fraud Detection System is now running!")
        print("Check the separate terminal windows to see the system in action.")
        print("Press Ctrl+C in each terminal to stop the processes.")
        
    except Exception as e:
        print(f"❌ Error starting system: {e}")

if __name__ == "__main__":
    run_fraud_detection_system()