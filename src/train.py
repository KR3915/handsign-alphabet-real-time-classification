import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from datetime import datetime
import mediapipe as mp

# --- 1. Load Data ---
DATA_FILE = '../data/data.csv'
MODELS_DIR = '../models'

if not os.path.exists(DATA_FILE):
    print(f"Error: Data file not found at {DATA_FILE}")
    exit()

# --- Define Header Manually to ensure consistency ---
# This matches the header created in capture.py
mp_hands = mp.solutions.hands
landmark_names = [lm.name for lm in mp_hands.HandLandmark if lm != mp_hands.HandLandmark.WRIST]
header = [f'{side}_{lm_name}_{axis}' for side in ['right', 'left'] for lm_name in landmark_names for axis in ['x', 'y']] + ['label']

# Load the CSV, assuming no header and assigning our own
df = pd.read_csv(DATA_FILE, header=None, names=header)
print(f"Loaded {len(df)} rows from {DATA_FILE}")

# --- 2. Preprocess and Validate Data ---
if df.isnull().values.any():
    print("Warning: Missing values found in the dataset. Consider cleaning the data.")
    df.dropna(inplace=True) # Drop rows with missing values
    print(f"Proceeding with {len(df)} rows after dropping NaN values.")

X = df.drop('label', axis=1) # Features
y = df['label']             # Target

# --- 3. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Train Model ---
print("Training RandomForestClassifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print("Training complete.")

# --- 5. Evaluate Model ---
y_pred = clf.predict(X_test)
print("\n--- Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# --- 6. Save Model ---
os.makedirs(MODELS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = os.path.join(MODELS_DIR, f"model_{timestamp}.joblib")
joblib.dump(clf, model_path)
print(f"\nModel saved to {model_path}")