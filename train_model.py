import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pickle

# Load and preprocess the data
print("Loading data...")
df = pd.read_csv('ipl_data.csv')

# Select relevant features
X = df[['bat_team', 'bowl_team', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5']]
y = df['total']

# Convert categorical variables to numerical using Label Encoding
le = LabelEncoder()
X['bat_team'] = le.fit_transform(X['bat_team'])
X['bowl_team'] = le.fit_transform(X['bowl_team'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
print("Training model...")
model = XGBRegressor(
    n_estimators=100,
    max_depth=8,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# Save the model
print("Saving model...")
filename = 'ml_model.pkl'
pickle.dump(model, open(filename, 'wb'))
print(f"Model saved as {filename}")

# Save the label encoders
print("Saving label encoders...")
encoders = {
    'bat_team': le,
    'bowl_team': le
}
pickle.dump(encoders, open('label_encoders.pkl', 'wb'))
print("Label encoders saved as label_encoders.pkl")