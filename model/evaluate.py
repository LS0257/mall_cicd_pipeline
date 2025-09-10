import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # go one level up
data_path= os.path.join(BASE_DIR, 'data', 'Mall_Customers.csv')
 
# Load the dataset
data = pd.read_csv(data_path)

# Convert categorical variable 'Gender' to numerical values
data['Genre'] = data['Genre'].map({'Male': 0, 'Female': 1})

# Preprocess the dataset
X = data.drop('CustomerID', axis=1)
y = data['CustomerID']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the saved model
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # go one level up
data_path= os.path.join(BASE_DIR, 'mall_customers_model.pkl')
model = joblib.load(data_path)

# Make predictions
y_pred= model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Modelaccuracy: {accuracy:.2f}')