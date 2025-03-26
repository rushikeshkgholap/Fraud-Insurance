import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_excel(r"E:\Rushikesh Gholap\fraud insurance.xlsx")  # Ensure correct path
print(df.head())  # Display first few rows

# ---- Step 1: Data Cleaning ----
# Convert DateTime columns to numeric
for col in df.select_dtypes(include=['datetime64']).columns:
    df[col + '_year'] = df[col].dt.year
    df[col + '_month'] = df[col].dt.month
    df[col + '_day'] = df[col].dt.day
    df.drop(columns=[col], inplace=True)  # Drop original datetime column

# Drop unnecessary columns
df.drop(columns=["policy_number", "insured_zip"], errors='ignore', inplace=True)

# Fill missing values
df.fillna(df.select_dtypes(include=['number']).median(), inplace=True)
df.fillna(df.select_dtypes(include=['object']).mode().iloc[0], inplace=True)

# Encode categorical columns
encoder = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str)  # Ensure all categorical columns are strings
    df[col] = encoder.fit_transform(df[col])

# ---- Step 2: Define Features & Target ----
X = df.drop(columns=["fraud_reported"], errors='ignore')  # Features
y = df["fraud_reported"].astype(int)  # Convert target variable to integer

# ---- Step 3: Handle Imbalance with SMOTE ----
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# ---- Step 4: Train-Test Split ----
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# ---- Step 5: Train Model ----
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# ---- Step 6: Evaluate Model ----
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ---- Step 7: Hyperparameter Tuning ----
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)

# Save cleaned data before SMOTE
df.to_excel(r"E:\Rushikesh Gholap\cleaned_fraud_data.xlsx", index=False)

# Save resampled (balanced) data after SMOTE
resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
resampled_df['fraud_reported'] = y_resampled  # Add target column
resampled_df.to_excel(r"E:\Rushikesh Gholap\resampled_fraud_data.xlsx", index=False)

# Save predictions
output_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
output_df.to_excel(r"E:\Rushikesh Gholap\fraud_predictions.xlsx", index=False)

print("All data saved successfully!")
