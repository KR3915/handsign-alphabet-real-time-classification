import cv2
import mediapipe as mp
import joblib
import numpy as np
from utils import process_hand_landmarks # Import the processing function

# Načti uložený model
model = joblib.load("../models/model_01.joblib")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape

    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # --- Process landmarks to create the feature vector ---
            # This ensures the input for the model is in the exact same format as the training data.
            coords = process_hand_landmarks(hand_landmarks, W, H)
            X_input = np.array([coords])

            # --- predikce ---
            prediction = model.predict(X_input)[0]

            # --- vypiš výsledek nad rukou ---
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            wrist_x, wrist_y = int(wrist.x * W), int(wrist.y * H)
            cv2.putText(frame, f"{prediction}", (wrist_x, wrist_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # --- vykresli ruce ---
            drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Prediction Overlay", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()