# Import necessary libraries for data manipulation, visualization, and machine learning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer, RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.inspection import PartialDependenceDisplay
from collections import Counter
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# ----------------------------------------------
# 1. Load and Inspect the Dataset
# ----------------------------------------------
# Load the credit card dataset from a CSV file
# The dataset contains 31 columns: Time, V1-V28 (anonymized features), Amount, and Class (0 = normal, 1 = fraud)
df = pd.read_csv(r'C:\Users\pawar\data_science\Project\Resume Project\new-res-proj\Credit_Card_Fraud_Detection_Dataset\creditcard.csv')

# Display the first 5 rows to understand the structure of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Display dataset info (column names, data types, and non-null counts)
print("\nDataset Info:")
df.info()

# Check for missing values in each column
print("\nMissing Values:")
print(df.isna().sum())

# Display the total number of rows in the dataset
print("\nTotal number of rows:", len(df))

# Calculate missing values and their percentage
missing_value = df.isnull().sum()
percentage_missing_value = (missing_value / len(df)) * 100
missing_df = pd.DataFrame({
    'missing_value': missing_value,
    'percentage_missing_value': percentage_missing_value
})
print("\nMissing Values Summary:")
print(missing_df)

# Check the shape of the dataset (rows, columns)
print("\nDataset Shape:", df.shape)

# ----------------------------------------------
# 2. Handle Duplicate Rows
# ----------------------------------------------
# Check for duplicate rows in the dataset
num_duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {num_duplicates}")

# Remove duplicate rows to ensure data integrity
df = df.drop_duplicates()
print(f"Number of rows after dropping duplicates: {df.shape[0]}")

# Verify the updated shape after removing duplicates
print("\nUpdated Dataset Shape:", df.shape)

# ----------------------------------------------
# 3. Analyze Class Distribution (Imbalanced Data)
# ----------------------------------------------
# Check the distribution of the target variable 'Class' (0 = normal, 1 = fraud)
classes = df['Class'].value_counts()
print("\nClass Distribution:")
print(classes)
print(f"Normal Transactions: {classes[0]}")
print(f"Fraudulent Transactions: {classes[1]}")
print(f"Percentage of Normal Transactions: {(classes[0] / df['Class'].count()) * 100:.2f}%")
print(f"Percentage of Fraudulent Transactions: {(classes[1] / df['Class'].count()) * 100:.2f}%")

# Visualize class distribution using a bar plot
plt.figure(figsize=(5, 5))
bars = plt.bar(['Normal', 'Fraud'], [classes[0], classes[1]], color='lightblue')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 20, str(int(yval)),
             ha='center', va='bottom', fontweight='bold')
plt.title('Class Distribution')
plt.xlabel('Transaction Type')
plt.ylabel('Count')
plt.show()

# ----------------------------------------------
# 4. Data Preprocessing
# ----------------------------------------------
# Separate features (X) and target (y)
X = df.drop('Class', axis=1)  # Features: All columns except 'Class'
y = df['Class']  # Target: 'Class' column
feature_names = X.columns

# Visualize feature distributions using histograms
fig, axes = plt.subplots(10, 3, figsize=(30, 45))
axes = axes.flatten()
for i, ax in enumerate(axes):
    sns.histplot(X[feature_names[i]], ax=ax)
    ax.set_title(feature_names[i])
plt.tight_layout()
plt.show()

# ----------------------------------------------
# 5. Handle Skewness in Features
# ----------------------------------------------
# Calculate skewness for each feature
skewness = X.skew()
skew_df = pd.DataFrame({'Features': feature_names, 'Skewness': skewness})
print("\nSkewness of Features:")
print(skew_df)

# Identify highly skewed columns (absolute skewness > 1)
skewed_columns = skew_df.loc[abs(skew_df['Skewness']) > 1, 'Features']
print("\nHighly Skewed Columns:", list(skewed_columns))

# Apply Yeo-Johnson transformation to reduce skewness
pt = PowerTransformer(method='yeo-johnson', copy=False)
X[skewed_columns] = pt.fit_transform(X[skewed_columns])

# ----------------------------------------------
# 6. Visualize Relationships
# ----------------------------------------------
# Scatter plot: Time vs. Class
plt.figure(figsize=(7, 5))
plt.scatter(X['Time'], y, c=y, cmap='coolwarm')
plt.xlabel('Time')
plt.ylabel('Class (0 = Normal, 1 = Fraud)')
plt.title('Scatter Plot of Time vs. Class')
plt.show()

# Scatter plot: Amount vs. Class
plt.figure(figsize=(10, 5))
plt.scatter(X['Amount'], y, c=y, cmap='coolwarm')
plt.xlabel('Amount')
plt.ylabel('Class (0 = Normal, 1 = Fraud)')
plt.title('Scatter Plot of Amount vs. Class')
plt.show()

# ----------------------------------------------
# 7. Normalize Features
# ----------------------------------------------
# Apply RobustScaler to normalize the 'Amount' column
scaler = RobustScaler()
X[['Amount']] = scaler.fit_transform(X[['Amount']])
print("\nFeatures after normalization (first 5 rows):")
print(X.head())

# ----------------------------------------------
# 8. Handle Class Imbalance with SMOTE
# ----------------------------------------------
# Check class distribution before SMOTE
print("\nClass Distribution Before SMOTE:", Counter(y))

# Apply SMOTE to oversample the minority class (fraud)
smote = SMOTE(sampling_strategy=0.1, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Check class distribution after SMOTE
print("Class Distribution After SMOTE:", Counter(y_resampled))

# ----------------------------------------------
# 9. Model Building: Logistic Regression with Cross-Validation
# ----------------------------------------------
# Initialize StratifiedKFold for cross-validation (preserves class distribution)
skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

# Test different values of the regularization parameter C for Logistic Regression
C_values = [0.01, 0.1, 0.5, 1, 2]
for c in C_values:
    mean_accuracy_list = []
    mean_roc_auc_list = []
    mean_precision_list = []
    accur_list = []
    roc_auc_list = []
    precision_list = []
    
    # Perform cross-validation
    for train_index, val_index in skf.split(X_resampled, y_resampled):
        X_train, X_val = X_resampled.iloc[train_index], X_resampled.iloc[val_index]
        y_train, y_val = y_resampled.iloc[train_index], y_resampled.iloc[val_index]
        
        # Train Logistic Regression model
        model = LogisticRegression(C=c)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_val)
        y_pred_prob = model.predict_proba(X_val)[:, 1]  # Probability scores for ROC-AUC
        
        # Calculate metrics
        accur_list.append(accuracy_score(y_val, y_pred))
        precision_list.append(precision_score(y_val, y_pred))
        roc_auc_list.append(roc_auc_score(y_val, y_pred_prob))
    
    # Compute mean metrics across folds
    mean_accuracy_list = np.mean(accur_list)
    mean_roc_auc_list = np.mean(roc_auc_list)
    mean_precision_list = np.mean(precision_list)
    
    # Print results
    print("===============================================")
    print(f'C_value: {c}')
    print(f'Mean Accuracy: {mean_accuracy_list:.4f}')
    print(f'Mean ROC-AUC: {mean_roc_auc_list:.4f}')
    print(f'Mean Precision: {mean_precision_list:.4f}')
    print("===============================================\n")

# ----------------------------------------------
# 10. Model Building: Random Forest Classifier
# ----------------------------------------------
# Split the resampled dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Initialize and train Random Forest Classifier
rf = RandomForestClassifier(n_estimators=30, max_depth=30, max_samples=0.2, bootstrap=True, random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_predict = rf.predict(X_test)
y_pred_prob = rf.predict_proba(X_test)[:, 1]  # Probability scores for ROC-AUC

# Calculate and print performance metrics
accuracy = accuracy_score(y_test, y_predict)
roc_auc = roc_auc_score(y_test, y_pred_prob)
print("\nRandom Forest Classifier Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

# ----------------------------------------------
# 11. Partial Dependence Plots
# ----------------------------------------------
# Visualize the partial dependence of the Random Forest model on selected features
# Features [0, 1, 2] correspond to 'Time', 'V1', and 'V2'
PartialDependenceDisplay.from_estimator(rf, X_resampled, features=[0, 1, 2])
plt.show()