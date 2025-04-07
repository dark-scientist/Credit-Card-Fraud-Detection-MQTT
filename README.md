# Credit Card Fraud Detection System

The Credit Card Fraud Detection System combines real-time message streaming with machine learning to identify and respond to fraudulent credit card transactions instantly. Built on the lightweight MQTT protocol, the system enables immediate analysis of transaction data as it occurs, rather than relying on traditional batch processing methods. The core of the system is an XGBoost-based classifier trained on historical credit card transaction data, which classifies incoming transactions and triggers appropriate responses—updating credit limits for legitimate transactions or flagging and blocking cards in case of fraudulent activity. The system features two visualization interfaces: a Streamlit dashboard for historical analysis and a Flask-based real-time simulation for monitoring live transaction flows.

## Table of Contents
- [Overview](#overview)
- [Project Architecture](#project-architecture)
- [Features](#features)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Streamlit Dashboard](#streamlit-dashboard)
  - [Real-time Flask Simulation](#real-time-flask-simulation)
- [Results](#results)
- [Technical Details](#technical-details)
- [Contributors](#contributors)
- [License](#license)


## Project Architecture

The system is composed of the following components:
- Data Producer: Generates transaction data (either from CSV or random generation)
- MQTT Broker: Facilitates communication between components
- Fraud Detection Service: ML-based service that processes transactions and flags potential fraud
- Visualization Dashboards:
  - Streamlit dashboard for historical analysis
  - Flask-based Real-time Simulation Dashboard for live monitoring

## Features

- Real-time credit card transaction processing
- Machine learning-based fraud detection with XGBoost algorithm and hyperparameter fine-tuning
- MQTT protocol implementation for efficient message handling
- Two visualization options:
  - Streamlit dashboard for interactive data analysis and configurable transaction volume
  - Flask-based real-time simulation dashboard generating random transactions
- Configurable fraud detection thresholds
- Performance metrics and visualization

## File Structure

```
Credit-Card-Fraud-Detection-MQTT-Worldline/
├── realtime/
│   ├── static/
│   │   └── logo.png/
│   ├── templates/
│   │   ├── bigtrain.py
│   │   ├── fraud_events.log
│   │   ├── fraud_model.pkl
│   │   ├── mqtt_publisher.py
│   │   ├── mqtt_subscriber.py
│   │   ├── realtimeapp.py
│   │   └── system.log
│   ├── resources/
│   │   ├── LICENSE
│   │   ├── bigtraim.py
│   │   ├── fraud_dashboard.py
│   │   ├── fraud_detect_system.py
│   │   ├── fraud_events.log
│   │   ├── fraud_model.pkl
│   │   ├── beginning/
│   │   ├── mqtt_publisher.py
│   │   ├── mqtt_subscriber.py
│   │   └── randomHistory.log
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/dark-scientist/Credit-Card-Fraud-Detection-MQTT-Worldline.git
cd Credit-Card-Fraud-Detection-MQTT-Worldline
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up MQTT broker (if not using a cloud service):
```bash
# For Mosquitto on Ubuntu
sudo apt-get install mosquitto mosquitto-clients
sudo systemctl start mosquitto
```

## Usage

### Streamlit Dashboard

The Streamlit dashboard provides an interactive way to analyze transaction data with configurable transaction volumes.

1. Start the Streamlit application:
```bash
cd realtime/resources
streamlit run fraud_dashboard.py
```

2. Access the dashboard at `http://localhost:8501` in your web browser.

3. Use the dashboard to:
   - Configure the number of transactions to process
   - View fraud detection metrics and visualizations
   - Analyze transaction patterns and anomalies
   - Export reports and insights

### Real-time Flask Simulation

The real-time simulation generates infinite random transaction data through a Flask application.

1. Start the MQTT broker:
```bash
# If using Mosquitto locally
mosquitto -v
```

2. Start the fraud detection system:
```bash
cd realtime/resources
python fraud_detect_system.py
```

3. Start the Flask application with random data generation:
```bash
cd realtime/templates
python beginning.py
```

4. Access the real-time dashboard at `http://localhost:5000` in your web browser.

5. The system will automatically:
   - Generate random transaction combinations via randomGenerator.py
   - Process them through the fraud detection model
   - Display results in real-time on the dashboard

## Results

### streamlit(output)
![image-3](https://github.com/user-attachments/assets/f7b77d25-2df7-4060-a6cf-5e6ffa6840ad)
![image-4](https://github.com/user-attachments/assets/787c85bc-faec-436c-9416-fd1a45e89ff7)
![image-1](https://github.com/user-attachments/assets/f0720fbe-846c-4f38-aba0-80692c75aa75)
![image-2](https://github.com/user-attachments/assets/8f7d1ed8-0aa3-48c8-a76c-ae026e1eab0b)

### flask(local host)
![Screenshot_2025-04-06_225935](https://github.com/user-attachments/assets/5be96d0c-ea31-4919-aee2-a7690d72e932)
![Screenshot_2025-04-06_230555](https://github.com/user-attachments/assets/2a30b3ca-d001-4298-97d6-a27ea6e99b0b)
![Screenshot_2025-04-06_230609](https://github.com/user-attachments/assets/1ee8a782-0659-4881-ac49-354faa134077)



## Technical Details

### Machine Learning Model
The fraud detection system utilizes an XGBoost classifier optimized through extensive hyperparameter tuning via RandomizedSearchCV. To address the inherent class imbalance in credit card fraud datasets, the SMOTE (Synthetic Minority Oversampling Technique) algorithm was employed during model training. This approach, combined with manual feature engineering, resulted in a highly effective model with the following performance metrics:

High classification accuracy on test data
Strong ROC AUC score for distinguishing between fraud and non-fraud cases
Balanced precision and recall, critical for minimizing both false positives and false negatives

Key features that contributed significantly to the model's decision-making include:

Transaction amount
Geo-distance between user and merchant locations
Transaction hour, day, and month (derived from timestamps)
User age and demographic information

The preprocessing pipeline includes date conversion, handling missing values, categorical encoding, and feature selection based on domain knowledge.
### MQTT Implementation
The system leverages MQTT (Message Queuing Telemetry Transport) for lightweight, efficient, and low-latency data transmission between components. This architecture enables continuous real-time analysis of transactions as they occur, allowing for immediate intervention when fraudulent activity is detected.
### The MQTT communication structure includes:

transactions/new: For publishing new transaction data
transactions/processed: For publishing processed transaction results
alerts/fraud: For sending fraud alerts to subscribers

### System activity is tracked through comprehensive logging:

mqtt_publisher.log: Records all published messages
mqtt_subscriber.log: Records all received messages
fraud_events.log: Documents detected fraudulent transactions

### Data Processing Pipeline

Transaction data is generated either from historical CSV files or through random generation
Data undergoes preprocessing and feature engineering before being published via MQTT
The fraud detection model processes incoming transactions, applying the trained XGBoost classifier
Classification results determine system responses (transaction approval or fraud alerts)
Results are published to relevant MQTT topics and logged for analysis
Visualization components subscribe to these topics and update their displays in real-time


## Contributors

- Prithwin [dark-scientist](https://github.com/dark-scientist)
- Akshay [Akshay Benedict](https://github.com/akvoid1)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
