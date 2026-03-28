import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

# Load dataset
data = pd.read_csv("heart.csv")

# Features & target
X = data.drop("target", axis=1)
y = data["target"]

# Scale features (IMPORTANT for better accuracy)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print(f"✅ Model Accuracy: {accuracy:.2f}")

# Save model + scaler
with open("model.pkl", "wb") as f:
    pickle.dump((model, scaler), f)

print("✅ Model & scaler saved as model.pkl")