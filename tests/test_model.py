import unittest
import joblib
from sklearn.ensemble import RandomForestClassifier
import os

BASE_DIR = os.path.dirname(os.path.dirname(_file_))  # go one level up
data_path= os.path.join(BASE_DIR, 'mall_customer_model.pkl')
model = joblib.load(data_path)

class TestModelTraining(unittest.TestCase):
    def test_model_training(self):
        # model = joblib.load('mall_customer_model.pkl')
        self.assertIsInstance(model, RandomForestClassifier)
        self.assertGreaterEqual(len(model.feature_importances_), 4)
if _name_ == '_main_':
    unittest.main()