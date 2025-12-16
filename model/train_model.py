import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# Load training data
train_data = pd.read_csv('data/train.csv')

# Separate features and target
X = train_data.drop('price_range', axis=1)
y = train_data['price_range']

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
accuracy = model.score(X_val, y_val)
print(f"Model trained successfully with validation accuracy: {accuracy:.2f}")

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Save the trained model
joblib.dump(model, 'model/mobile_price_model.pkl')
print("Model saved as mobile_price_model.pkl")

# Print feature importance for reference
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))
