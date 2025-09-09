import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  
data_path= os.path.join(BASE_DIR, 'data', 'Mall_Customers.csv')

# Load the mall customer dataset
data = pd.read_csv(data_path)

# Drop CustomerID
data = data.drop('CustomerID', axis=1)

# Convert categorical variable 'Gender' to numerical values
data['Genre'] = data['Genre'].map({'Male': 0, 'Female': 1})

# Preprocess the dataset
X = data.drop('CustomerID', axis=1)
y = data['CustomerID']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'mall_customer_model.pkl')