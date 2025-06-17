import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your PCOS dataset
data = pd.read_csv('pcos_data.csv')

# Define features and target
X = data[['Age', 'BMI', 'FSH', 'LH', 'CycleIrregular', 'HairGrowth', 'Acne']]
y = data['PCOS']

# Train a model
model = RandomForestClassifier()
model.fit(X, y)

# Save model to pcos_model.pkl
joblib.dump(model, 'pcos_model.pkl')

print("Model trained and saved as pcos_model.pkl")
