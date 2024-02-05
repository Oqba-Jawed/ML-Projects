import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load the data
data = pd.read_csv('2.csv')

# Data Preprocessing
data['LPD'] = pd.to_datetime(data['LPD'].str.strip(), errors='coerce')
data['Cash'] = pd.to_numeric(data['Cash'].astype(str).str.replace('[^\d.]', '', regex=True), errors='coerce')
data['Sturbs (No. of Payments)'] = pd.to_numeric(data['Sturbs (No. of Payments)'].fillna(0))
data['Net Amount'] = pd.to_numeric(data['Net Amount'].astype(str).str.replace('[^\d.]', '', regex=True), errors='coerce')
data['Calendar Year'] = data['LPD'].dt.year

# Advanced Feature Engineering
# Example: Summarize each year's data for each customer into new features
features = data.groupby(['Contract Number', 'Calendar Year']).agg({
    'Net Amount': ['sum', 'mean', 'max'],
    'Cash': ['sum', 'mean', 'max'],
    'Sturbs (No. of Payments)': 'sum'
}).reset_index()

# Flatten the multi-level column structure
features.columns = ['_'.join(col).strip() for col in features.columns.values]
# print(features)
# Classify each customer-year based on your criteria
def classify_customer(row):
    # Apply your criteria for classification
    # Example:
    if row['Sturbs (No. of Payments)_sum'] < 5:
        return 'Defaulter'
    elif 5 <= row['Sturbs (No. of Payments)_sum'] <= 8:
        return 'Dodger'
    elif row['Sturbs (No. of Payments)_sum'] > 8:
        return 'Star Customer'
    else:
        return 'Unclassified'

features['Customer_Classification'] = features.apply(classify_customer, axis=1)
# print(features)
# Encoding and Dataset Preparation
label_encoder = LabelEncoder()
features['Customer_Classification_Encoded'] = label_encoder.fit_transform(features['Customer_Classification'])
X = features.drop(['Contract Number_', 'Customer_Classification', 'Customer_Classification_Encoded'], axis=1)
# print(X)
# print(X)
y = features['Customer_Classification_Encoded']
print(y)
# Splitting the Dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the RandomForestClassifier
model = LogisticRegression(random_state=42, max_iter=1000)

# model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Prediction and Evaluation
y_pred = model.predict_proba(X)
# prob_class_0_first_instance = y_pred[0][0]
prob_df = pd.DataFrame(y_pred, columns=label_encoder.classes_)
mean_probabilities = prob_df.mean(axis=0)
print(mean_probabilities)
# print(y_pred)
mean_value = np.mean(y_pred)
# print(y_pred)
# print(mean_value)
unique_classes = sorted(set(y))

# Map these unique classes back to their string names using the label encoder
target_names = label_encoder.inverse_transform(unique_classes)
# print(target_names)
# Generate the classification report
report = classification_report(y, y_pred, target_names=target_names)
print(report)
label_encoder.fit(features['Customer_Classification'])
features['Customer_Classification_Encoded'] = label_encoder.transform(features['Customer_Classification'])

# Check the mapping of encoded labels
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
for k in label_mapping:
    if label_mapping[k] == mean_value:
        print("This customer is classified as a", k)
# print(label_mapping['Defaulter'])
# print(label_mapping)
