import cv2
import mediapipe as mp
import math
import csv
import os
from typing import List
from utils import depth_factor # Import from utils

# Initialize MediaPipe Hands and OpenCV Video Capture
mp_hands = mp.solutions.hands # Alias the hands solution module for easier reference
cap = cv2.VideoCapture(0)
# Use confidence thresholds for more stable detection
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
drawing = mp.solutions.drawing_utils

DATA_FILE = '../data/data.csv'

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

# --- Create CSV header ---
landmark_names = [lm.name for lm in mp_hands.HandLandmark if lm != mp_hands.HandLandmark.WRIST]
header = [f'{side}_{lm_name}_{axis}' for side in ['right', 'left'] for lm_name in landmark_names for axis in ['x', 'y']] + ['label']

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
            
    # Display the frame
    cv2.imshow("Hand Joints", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('a'):
        # Combine right and left hand data, and add the label
        row = right_hand_coords + left_hand_coords + ['A']
        
        # Check if file exists to write header
        file_exists = os.path.isfile(DATA_FILE)
        
        with open(DATA_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header) # Write header if file is new
            writer.writerow(row)
            print(f"Saved data for label 'A'")

    if key == ord('b'):    # Combine right and left hand data, and add the label
        row = right_hand_coords + left_hand_coords + ['B']

        # Check if file exists to write header
        file_exists = os.path.isfile(DATA_FILE)
        
        with open(DATA_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header) # Write header if file is new
            writer.writerow(row)
            print(f"Saved data for label 'B'")
cap.release()
cv2.destroyAllWindows()
hands.close()
