
import math
import mediapipe as mp
from typing import List, Tuple, Optional

def depth_factor(val: Tuple[float, float], depth: float, k: float = 2.5, clamp: float = 0.25) -> Tuple[float, float]:
    """Applies a depth-based scaling factor to a 2D position."""
    val1, val2 = val

    # Clamp depth range so extreme values don’t explode
    depth_clamped = max(-clamp, min(clamp, depth))

    # Apply smooth sigmoid-like scaling
    scale = 1.0 / (1.0 + k * depth_clamped)   # base linear
    scale = scale * (1.0 - 0.5 * (1 / (1 + math.exp(-10 * (depth_clamped + clamp/2)))))  

    factor = -scale
    return (val1 / factor, val2 / factor)
def get_relative_position(wrist_lm, landmark_lm, W: int, H: int) -> Tuple[int, int]:
    """Calculates the pixel difference between a landmark and the wrist."""
    # Convert normalized coordinates to pixel coordinates
    wrist_x_px = int(wrist_lm.x * W)
    wrist_y_px = int(wrist_lm.y * H)
    landmark_x_px = int(landmark_lm.x * W)
    landmark_y_px = int(landmark_lm.y * H)

    # Calculate the pixel difference
    relative_x = landmark_x_px - wrist_x_px
    relative_y = landmark_y_px - wrist_y_px

    return relative_x, relative_y

def process_hand_landmarks(hand_landmarks, W: int, H: int) -> List[float]:
    """
    Processes hand landmarks to create a flat list of depth-adjusted coordinates relative to the wrist.

    The order of landmarks is consistent with the header created in capture.py:
    thumb (cmc, mcp, ip, tip), index (mcp, pip, dip, tip), etc.
    """
    mp_hands = mp.solutions.hands
    coords = []

    # Get the wrist landmark to use as the origin
    wrist_lm = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    # Define the landmarks to be processed in the correct order
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

        # 1. Calculate coordinates relative to the wrist
        relative_x = landmark.x - wrist_lm.x
        relative_y = landmark.y - wrist_lm.y

        # 2. Apply the depth factor to the relative coordinates
        processed_x, processed_y = depth_factor((relative_x, relative_y), landmark.z)
        coords.extend([processed_x, processed_y])

    return coords
