# Import necessary libraries for data manipulation, preprocessing, and machine learning
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# ----------------------------------------------
# 1. Load and Preprocess the Dataset
# ----------------------------------------------
# Load the credit card dataset
df = pd.read_csv(r'C:\Users\pawar\data_science\Project\Resume Project\new-res-proj\Credit_Card_Fraud_Detection_Dataset\creditcard.csv')

# Remove duplicate rows
df = df.drop_duplicates()

# Separate features (X) and target (y)
X = df.drop('Class', axis=1)
y = df['Class']
feature_names = X.columns

# ----------------------------------------------
# 2. Handle Skewness and Normalize Features
# ----------------------------------------------
# Identify highly skewed columns (absolute skewness > 1)
skewness = X.skew()
skewed_columns = skewness[abs(skewness) > 1].index

# Apply Yeo-Johnson transformation to reduce skewness
pt = PowerTransformer(method='yeo-johnson')
X[skewed_columns] = pt.fit_transform(X[skewed_columns])

# Save the PowerTransformer for later use
joblib.dump(pt, 'power_transformer.pkl')

# Apply RobustScaler to normalize the 'Amount' column
scaler = RobustScaler()
X[['Amount']] = scaler.fit_transform(X[['Amount']])

# Save the RobustScaler for later use
joblib.dump(scaler, 'robust_scaler.pkl')

# ----------------------------------------------
# 3. Handle Class Imbalance with SMOTE
# ----------------------------------------------
smote = SMOTE(sampling_strategy=0.1, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# ----------------------------------------------
# 4. Train and Save Random Forest Model
# ----------------------------------------------
# Split the resampled dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rf = RandomForestClassifier(n_estimators=30, max_depth=30, max_samples=0.2, bootstrap=True, random_state=42)
rf.fit(X_train, y_train)

# Evaluate the model
y_predict = rf.predict(X_test)
y_pred_prob = rf.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_predict)
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"Random Forest Classifier Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Save the trained model
joblib.dump(rf, 'random_forest_model.pkl')