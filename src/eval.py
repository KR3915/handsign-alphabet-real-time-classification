import mediapipe as mp
import math
from typing import Tuple, List
from utils import depth_factor, get_relative_position
import cv2
import csv
import os
import numpy as np
import joblib
import pandas as pd

mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
drawing = mp.solutions.drawing_utils

# --- Load the most recent model/artifacts ---
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
if not os.path.exists(MODELS_DIR) or not os.listdir(MODELS_DIR):
    print(f"Error: Models directory '{MODELS_DIR}' is empty or does not exist.")
    print("Please run train.py to create a model.")
    exit()

latest_model_file = max(
    [os.path.join(MODELS_DIR, f) for f in os.listdir(MODELS_DIR) if f.endswith('.joblib')],
    key=os.path.getctime
)

print(f"Loading model/artifacts: {latest_model_file}")
artifacts = joblib.load(latest_model_file)

# Support both old saved estimator and new artifact-dict format
if isinstance(artifacts, dict):
    model = artifacts.get("model")
    imputer = artifacts.get("imputer")
    feature_columns = artifacts.get("feature_columns")
    label_encoder = artifacts.get("label_encoder")
else:
    # older files that saved estimator directly
    model = artifacts
    imputer = None
    feature_columns = None
    label_encoder = None

if model is None or not hasattr(model, "predict"):
    print("Error: loaded object does not contain a fitted estimator under key 'model'.")
    exit()

def process_hand_landmarks(hand_landmarks) -> List[float]:
    coords = []
    wrist_lm = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    landmark_order = [
        mp_hands.HandLandmark.THUMB_CMC, mp_hands.HandLandmark.THUMB_MCP,
        mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.INDEX_FINGER_DIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_DIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_DIP, mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_MCP, mp_hands.HandLandmark.PINKY_PIP,
        mp_hands.HandLandmark.PINKY_DIP, mp_hands.HandLandmark.PINKY_TIP,
    ]

    for landmark_enum in landmark_order:
        landmark = hand_landmarks.landmark[landmark_enum]
        relative_coords = (landmark.x - wrist_lm.x, landmark.y - wrist_lm.y)
        processed_x, processed_y = depth_factor(relative_coords, depth=landmark.z)
        coords.extend([processed_x, processed_y])

    return coords

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        H, W, _ = frame.shape

        frame = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        right_hand_coords = [0.0] * 40
        left_hand_coords = [0.0] * 40
        right_hand_pos = None   # pixel (x, y) for right hand wrist
        left_hand_pos = None    # pixel (x, y) for left hand wrist

        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[i].classification[0].label
                processed_coords = process_hand_landmarks(hand_landmarks)

                # compute wrist pixel position to place text near the hand
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                wrist_x_px = int(wrist.x * W)
                wrist_y_px = int(wrist.y * H)

                if handedness == 'Right':
                    right_hand_coords = processed_coords
                    right_hand_pos = (wrist_x_px, wrist_y_px)
                elif handedness == 'Left':
                    left_hand_coords = processed_coords
                    left_hand_pos = (wrist_x_px, wrist_y_px)

                drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                        drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                                        drawing.DrawingSpec(color=(255,255,255), thickness=2))

        feature_vector = right_hand_coords + left_hand_coords
        X_input = None

        # If saved feature column names exist, align and pad/truncate as needed
        if feature_columns:
            expected_len = len(feature_columns)
            fv = list(feature_vector)
            if len(fv) < expected_len:
                fv = fv + [0.0] * (expected_len - len(fv))
            elif len(fv) > expected_len:
                fv = fv[:expected_len]

            X_df = pd.DataFrame([fv], columns=feature_columns)
            # If imputer exists, use it; otherwise rely on raw values
            if imputer is not None:
                # keep as DataFrame to preserve feature names
                X_input = pd.DataFrame(imputer.transform(X_df), columns=feature_columns)
            else:
                X_input = X_df.astype(float)
        else:
            # No feature metadata saved — pass raw vector (shape must match model input)
            X_input = np.array([feature_vector], dtype=float)

        # Predict
        try:
            pred_raw = model.predict(X_input)
            pred = pred_raw[0]
        except Exception as e:
            print("Prediction error:", e)
            pred = None

        # Predict confidence if supported
        confidence = None
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_input)[0]
                confidence = float(np.max(proba))
            else:
                confidence = 1.0
        except Exception:
            confidence = None

        # Decode label if label encoder provided
        display_label = pred
        if label_encoder is not None and pred is not None:
            try:
                display_label = label_encoder.inverse_transform([int(pred)])[0]
            except Exception:
                # If inverse_transform fails, keep raw pred
                pass

        conf_text = f"{confidence:.2f}" if confidence is not None else "N/A"

        # Place predicted text on the detected hand:
        # prefer right hand (if present), else left, else fallback to top-left.
        if right_hand_pos is not None:
            pos_x, pos_y = right_hand_pos
            # offset above wrist
            text_pos = (max(0, pos_x + 10), max(20, pos_y - 20))
        elif left_hand_pos is not None:
            pos_x, pos_y = left_hand_pos
            text_pos = (max(0, pos_x + 10), max(20, pos_y - 20))
        else:
            text_pos = (10, 30)

        display_text = f"{display_label} ({conf_text})"
        # draw a filled rectangle background for better readability
        (text_w, text_h), baseline = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        pad = 6
        rect_tl = (text_pos[0] - pad, text_pos[1] - text_h - pad)
        rect_br = (text_pos[0] + text_w + pad, text_pos[1] + baseline + pad)
        cv2.rectangle(frame, rect_tl, rect_br, (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, display_text, (text_pos[0], text_pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Hand Sign Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()