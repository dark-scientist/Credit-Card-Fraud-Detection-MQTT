import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv("fraudTrain.csv")

# Data Cleaning and Preprocessing
print("Initial Data Info:")
print(df.info())

# Drop missing values
df = df.dropna()

# Convert date columns to datetime format
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['dob'] = pd.to_datetime(df['dob'])

# Feature Engineering
df['transaction_hour'] = df['trans_date_trans_time'].dt.hour
df['transaction_day'] = df['trans_date_trans_time'].dt.day
df['transaction_month'] = df['trans_date_trans_time'].dt.month
df['age'] = df['trans_date_trans_time'].dt.year - df['dob'].dt.year

# Calculate distance between transaction location and merchant location
df['geo_distance'] = np.sqrt((df['lat'] - df['merch_lat'])**2 + (df['long'] - df['merch_long'])**2)

# Drop unnecessary columns
drop_cols = ['trans_date_trans_time', 'dob', 'Unnamed: 0', 'first', 'last', 'street', 'trans_num', 'unix_time', 'cc_num']
df = df.drop(columns=drop_cols, errors='ignore')

# Label Encoding for Categorical Variables
categorical_cols = ['merchant', 'category', 'gender', 'city', 'state', 'job']
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# Define Features and Target
X = df.drop(columns=['is_fraud'])
y = df['is_fraud']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE for Class Balancing
smote = SMOTE(sampling_strategy=0.3, k_neighbors=5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
print(f"Class distribution after SMOTE: {pd.Series(y_resampled).value_counts()}")

# Feature Selection
selected_features = ['amt', 'geo_distance', 'transaction_hour', 'transaction_day', 'transaction_month', 'age', 'city_pop', 'merchant', 'category', 'gender', 'city', 'state', 'job']
X_resampled_selected = X_resampled[selected_features]
X_test_selected = X_test[selected_features]

# Apply Feature Scaling *after* feature selection
scaler = StandardScaler()
X_resampled_selected = scaler.fit_transform(X_resampled_selected)
X_test_selected = scaler.transform(X_test_selected)

# Hyperparameter Tuning with Randomized Search
param_grid = {
    'n_estimators': [200, 400, 600],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0.1, 0.5, 1.0],
    'reg_lambda': [0.1, 0.5, 1.0]
}
xgb_model = xgb.XGBClassifier(scale_pos_weight=7500/1289169, eval_metric="logloss", random_state=42)

random_search = RandomizedSearchCV(xgb_model, param_grid, cv=3, n_iter=10, n_jobs=-1, scoring='roc_auc', random_state=42)
random_search.fit(X_resampled_selected, y_resampled)

best_model = random_search.best_estimator_
print(f"Best Hyperparameters: {random_search.best_params_}")

# Model Training
best_model.fit(X_resampled_selected, y_resampled)

# Predict and Evaluate
y_pred_proba = best_model.predict_proba(X_test_selected)[:, 1]
y_test_pred = best_model.predict(X_test_selected)
print("Classification Report:")
print(classification_report(y_test, y_test_pred))
print(f"AUC: {roc_auc_score(y_test, y_pred_proba)}")

# Save the Final Model (only the model, not the encoders)
joblib.dump(best_model, "fraud_model.pkl")
print("✅ Model saved successfully!")

# Plot Feature Importance
xgb.plot_importance(best_model)
plt.show()