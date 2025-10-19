import mediapipe as mp
import math
from typing import Tuple
from utils import depth_factor, get_relative_position
import cv2
import csv
import os
from typing import List
import numpy as np
import joblib

mp_hands = mp.solutions.hands # Alias the hands solution module for easier reference
cap = cv2.VideoCapture(0)
# Use confidence thresholds for more stable detection
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
drawing = mp.solutions.drawing_utils

# --- Load the most recent model ---
MODELS_DIR = '../models'
if not os.path.exists(MODELS_DIR) or not os.listdir(MODELS_DIR):
    print(f"Error: Models directory '{MODELS_DIR}' is empty or does not exist.")
    print("Please run train.py to create a model.")
    exit()

latest_model_file = max(
    [os.path.join(MODELS_DIR, f) for f in os.listdir(MODELS_DIR) if f.endswith('.joblib')],
    key=os.path.getctime
)

print(f"Loading model: {latest_model_file}")
model = joblib.load(latest_model_file)



def process_hand_landmarks(hand_landmarks) -> List[float]:
    """
    Processes a single hand's landmarks to create a flat list of 40 coordinates
    relative to the wrist and adjusted for depth.
    """
    coords = []
    wrist_lm = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    # Define the landmarks to be processed, excluding the wrist
    landmark_order = [
        # Thumb
        mp_hands.HandLandmark.THUMB_CMC, mp_hands.HandLandmark.THUMB_MCP,
        mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.THUMB_TIP,
        # Index Finger
        mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.INDEX_FINGER_DIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
        # Middle Finger
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_DIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        # Ring Finger
        mp_hands.HandLandmark.RING_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_DIP, mp_hands.HandLandmark.RING_FINGER_TIP,
        # Pinky
        mp_hands.HandLandmark.PINKY_MCP, mp_hands.HandLandmark.PINKY_PIP,
        mp_hands.HandLandmark.PINKY_DIP, mp_hands.HandLandmark.PINKY_TIP,
    ]

    for landmark_enum in landmark_order:
        landmark = hand_landmarks.landmark[landmark_enum]
        relative_coords = (landmark.x - wrist_lm.x, landmark.y - wrist_lm.y)
        processed_x, processed_y = depth_factor(relative_coords, depth=landmark.z)
        coords.extend([processed_x, processed_y])

    return coords

# ---check if the webcam is capturing correctly---
while True:
    ret, frame = cap.read()
    H, W, _ = frame.shape
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Initialize coordinates with zeros for missing hands
    right_hand_coords = [0.0] * 40
    left_hand_coords = [0.0] * 40

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[i].classification[0].label
            processed_coords = process_hand_landmarks(hand_landmarks)
            
            if handedness == 'Right':
                right_hand_coords = processed_coords
            elif handedness == 'Left':
                left_hand_coords = processed_coords
            
            # Draw landmarks
            drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) # Use mp_hands alias here

    # --- Prepare data and predict ---
    feature_vector = right_hand_coords + left_hand_coords
    X_input = np.array([feature_vector])

    prediction = model.predict(X_input)[0]
    prediction_proba = model.predict_proba(X_input)[0]
    confidence = max(prediction_proba)

    # --- Display the prediction on the frame ---

    display_text = ':)' if prediction == 'A' else ':('
    cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Hand Sign Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
hands.close()