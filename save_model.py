
# save_model.py
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import os

# Create sample data
X, y = make_classification(n_samples=100, n_features=5, n_informative=3, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Make sure 'model' folder exists
os.makedirs('model', exist_ok=True)

# Save the trained model
with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved to model/model.pkl")
