# 💳 Credit Card Fraud Detection via MQTT + Streamlit

This project presents two modes of deploying a credit card fraud detection system:

---

## 📦 1. Streamlit App (User-Friendly Dashboard)

This version allows users to **upload CSV files** or simulate random fraud data, process it through the model, and visualize the output with elegant charts.

### 🧠 Features:
- Upload historical transaction data
- Fraud prediction using trained model
- Summary statistics & fraud rate
- Interactive Streamlit-based dashboard

### 🚀 Deployment:
- Run with `streamlit run streamlit_app.py`
- Customize thresholds, visualize fraud distribution

### 📸 Screenshot Preview:
> *(Insert Streamlit UI screenshots here)*

---

## 🌐 2. Real-Time Infinite Random Data Simulation (via MQTT)

A simulation pipeline for **real-time fraud detection** using:
- `paho-mqtt` for broker-based message communication
- Infinite data generation mimicking card transactions
- Live classification using the trained ML model
- Console alerts + logging for detected frauds

### 🔧 Components:
- `simulate.py` — generates and publishes data
- `subscribe.py` — listens and detects fraud in real-time
- `broker` — Mosquitto setup

### 📸 Screenshot Preview:
> *(Insert CLI/MQTT simulation screenshot here)*

---

## 🧠 Model Info
- Algorithm: Random Forest (with SMOTE for imbalance)
- Evaluation: Accuracy, Precision, Recall, Confusion Matrix
- Model is saved as: `fraud_model.pkl`

---

## 📁 Project Structure (post-update)
