## Python Example: Predicting CI/CD Pipeline Failures Using Logistic Regression
This Python script demonstrates how logistic regression can be used to predict whether a CI/CD pipeline will fail or succeed based on historical build data.

## Step 1: Install Dependencies
```
pip install pandas numpy scikit-learn matplotlib seaborn
```
## Python Code for CI/CD Failure Prediction

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample Data: CI/CD Pipeline Runs
data = {
    "lines_of_code_changed": [50, 200, 150, 400, 10, 30, 500, 1000, 300, 700],
    "failed_tests": [1, 5, 3, 10, 0, 1, 12, 20, 8, 15],
    "previous_build_failed": [0, 1, 0, 1, 0, 0, 1, 1, 1, 1],  # 1 = Previous Build Failed, 0 = Succeeded
    "cpu_load": [40, 85, 60, 90, 20, 30, 95, 99, 70, 98],  # Server Load during Build (%)
    "build_failed": [0, 1, 0, 1, 0, 0, 1, 1, 1, 1],  # 1 = Build Failed, 0 = Build Succeeded (Target Variable)
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Define Features (X) and Target Variable (y)
X = df.drop(columns=["build_failed"])  # Features
y = df["build_failed"]  # Target variable (Build Failure: 1, Success: 0)

# Split Data into Training and Testing Sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize Features for Better Performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report)

# Visualizing the Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Success", "Failure"], yticklabels=["Success", "Failure"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("CI/CD Pipeline Failure Prediction - Confusion Matrix")
plt.show()

```
