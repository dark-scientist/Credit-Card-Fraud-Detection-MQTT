import joblib

# Load the saved model
xgb_model = joblib.load("fraud_model.pkl")

print("✅ Model loaded successfully!")
 