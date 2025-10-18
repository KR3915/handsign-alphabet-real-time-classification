import cv2
import mediapipe as mp
import math
# Initialize MediaPipe Hands and OpenCV Video Capture
mp_hands = mp.solutions.hands # Alias the hands solution module for easier reference
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands()
drawing = mp.solutions.drawing_utils

# Function to calculate depth factor


def depth_factor(val, depth, k=2.5, clamp=0.25):
    val1, val2 = val

    # Clamp depth range so extreme values don’t explode
    depth_clamped = max(-clamp, min(clamp, depth))

    # Apply smooth sigmoid-like scaling (stable near 0, limited when close)
    scale = 1.0 / (1.0 + k * depth_clamped)   # base linear
    scale = scale * (1.0 - 0.5 * (1 / (1 + math.exp(-10 * (depth_clamped + clamp/2)))))  

    factor = -scale
    return (val1 / factor, val2 / factor)

# Function to get relative position in pixels
def get_relative_position(wrist_lm, landmark_lm, W, H):
    # Convert normalized coordinates to pixel coordinates
    wrist_x_px = int(wrist_lm.x * W)
    wrist_y_px = int(wrist_lm.y * H)
    landmark_x_px = int(landmark_lm.x * W)
    landmark_y_px = int(landmark_lm.y * H)
    
    # Calculate the pixel difference
    relative_x = landmark_x_px - wrist_x_px
    relative_y = landmark_y_px - wrist_y_px
    
    return relative_x, relative_y

# ---check if the webcam is capturing correctly---
while True:
    ret, frame = cap.read()
    H, W, _ = frame.shape
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # 1. Check handedness (Left/Right)
            handedness = results.multi_handedness[i].classification[0].label
            
            # The WRIST landmark index is 0 for all hands (Left or Right)
            wrist_index = mp_hands.HandLandmark.WRIST # This correctly references index 0
            wrist_landmark = hand_landmarks.landmark[wrist_index] 
            #   ---thumb---
            thumb_cmc = mp_hands.HandLandmark.THUMB_CMC
            thumb_mcp = mp_hands.HandLandmark.THUMB_MCP
            thumb_ip = mp_hands.HandLandmark.THUMB_IP
            thumb_tip = mp_hands.HandLandmark.THUMB_TIP
            # --thumb landmark --
            thumb_cmc_landmark = hand_landmarks.landmark[thumb_cmc]
            thumb_mcp_landmark = hand_landmarks.landmark[thumb_mcp]
            thumb_ip_landmark = hand_landmarks.landmark[thumb_ip]
            thumb_tip_landmark = hand_landmarks.landmark[thumb_tip]
            
            thumb_cmc_X, thumb_cmc_Y = get_relative_position(wrist_landmark, thumb_cmc_landmark, W, H)
            thumb_mcp_X, thumb_mcp_Y = get_relative_position(wrist_landmark, thumb_mcp_landmark, W, H)
            thumb_ip_X, thumb_ip_Y = get_relative_position(wrist_landmark, thumb_ip_landmark, W, H)
            thumb_tip_X, thumb_tip_Y = get_relative_position(wrist_landmark, thumb_tip_landmark, W, H)
            
            #----- INDEX -----
            index_mcp = mp_hands.HandLandmark.INDEX_FINGER_MCP
            index_pip = mp_hands.HandLandmark.INDEX_FINGER_PIP
            index_dip = mp_hands.HandLandmark.INDEX_FINGER_DIP
            index_tip = mp_hands.HandLandmark.INDEX_FINGER_TIP

            index_mcp_landmark = hand_landmarks.landmark[index_mcp]
            index_pip_landmark = hand_landmarks.landmark[index_pip]
            index_dip_landmark = hand_landmarks.landmark[index_dip]
            index_tip_landmark = hand_landmarks.landmark[index_tip]


            index_mcp_X, index_mcp_Y = depth_factor(get_relative_position(wrist_landmark, index_mcp_landmark, W, H), depth=index_dip_landmark.z)
            index_pip_X, index_pip_Y = get_relative_position(wrist_landmark, index_pip_landmark, W, H)
            index_dip_X, index_dip_Y = get_relative_position(wrist_landmark, index_dip_landmark, W, H)
            index_tip_X, index_tip_Y = get_relative_position(wrist_landmark, index_tip_landmark, W, H)
            # Separate print statements based on handedness
            if handedness == 'Left':
                print(f'LEFT HAND THUMB: CMC: [{thumb_cmc_X},{thumb_cmc_Y}] | MCP: [{thumb_mcp_X},{thumb_mcp_Y}] | IP: [{thumb_ip_X},{thumb_ip_Y}] | TIP: [{thumb_tip_X},{thumb_tip_Y}]')
                print(f'LEFT HAND INDEX: CMC: [{index_mcp_X},{index_mcp_Y}] | MCP: [{index_pip_X},{index_pip_Y}] | IP: [{index_dip_X},{index_dip_Y}] | TIP: [{index_tip_X},{index_tip_Y}]')
            elif handedness == 'Right':
                print(f'RIGHT HAND THUMB: CMC: [{thumb_cmc_X},{thumb_cmc_Y}] | MCP: [{thumb_mcp_X},{thumb_mcp_Y}] | IP: [{thumb_ip_X},{thumb_ip_Y}] | TIP: [{thumb_tip_X},{thumb_tip_Y}]')
                print(f'RIGHT HAND INDEX: CMC: [{index_mcp_X},{index_mcp_Y}] | MCP: [{index_pip_X},{index_pip_Y}] | IP: [{index_dip_X},{index_dip_Y}] | TIP: [{index_tip_X},{index_tip_Y}, Y: {index_dip_landmark.z:.4f}]')

            wrist_position = (wrist_landmark.x, wrist_landmark.y)

            #relative_position = get_relative_position(wrist_position, frame_center)


            # 2. Print coordinates based on handedness
            #print(f"{handedness} Wrist: X: {wrist_landmark.x:.4f} | Y: {wrist_landmark.y:.4f}")
            #print(f'{handedness} Thumb_cmc: X {thumb_cmc_landmark.x:.4f} | Y: {thumb_cmc_landmark.y:.4f}')
            # Draw landmarks
            drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) # Use mp_hands alias here
            
    # Display the frame
    cv2.imshow("Hand Joints", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
