# FraudShield: AI-Powered Credit Card Fraud Detection

FraudShield is an AI-driven credit card fraud detection system that utilizes machine learning and real-time transaction monitoring to identify fraudulent activities. The system leverages MQTT messaging for real-time transaction processing and XGBoost for fraud prediction.

## 📌 Features
- **Machine Learning Model**: Uses XGBoost for fraud classification
- **Real-time Transaction Monitoring**: Listens to transactions via MQTT
- **Streamlit Dashboard**: (Coming soon) For real-time fraud visualization
- **Logging & Alerting**: Logs fraudulent transactions with probability scores
- **Scalable**: Easily deployable in production environments

---

## 🏗️ Project Structure
```
FraudShield/
│-- bigtrain.py              # Model training script
│-- fraud_detection_app.py   # Streamlit dashboard (UI for visualization)
│-- fraud_detection_system.py # Core fraud detection logic
│-- mqtt_publisher.py        # Publishes transactions via MQTT
│-- mqtt_subscriber.py       # Listens for transactions, detects fraud
│-- detected_frauds.csv      # Logs of fraud transactions
│-- fraud_model.pkl          # Trained ML model
│-- README.md                # Project Documentation
```

---

## 🚀 Setup Instructions
### 1️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 2️⃣ Train the Model
```sh
python bigtrain.py
```

### 3️⃣ Run the Fraud Detection System
```sh
python mqtt_subscriber.py
```

### 4️⃣ Publish Transactions (Simulated)
```sh
python mqtt_publisher.py
```

---

## 📊 Streamlit Dashboard (Coming Soon)
📌 This section will showcase fraud detection insights. (Screenshots to be added)

---

## 🤝 Contributing
Feel free to open issues and submit pull requests to enhance this project.

---

## 📜 License
MIT License. Free to use and modify.

