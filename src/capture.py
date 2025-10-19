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
            
            thumb_cmc_X, thumb_cmc_Y = depth_factor((thumb_cmc_landmark.x - wrist_landmark.x, thumb_cmc_landmark.y - wrist_landmark.y), depth=thumb_cmc_landmark.z)
            thumb_mcp_X, thumb_mcp_Y = depth_factor((thumb_mcp_landmark.x - wrist_landmark.x, thumb_mcp_landmark.y - wrist_landmark.y), depth=thumb_mcp_landmark.z)
            thumb_ip_X, thumb_ip_Y = depth_factor((thumb_ip_landmark.x - wrist_landmark.x, thumb_ip_landmark.y - wrist_landmark.y), depth=thumb_ip_landmark.z)
            thumb_tip_X, thumb_tip_Y = depth_factor((thumb_tip_landmark.x - wrist_landmark.x, thumb_tip_landmark.y - wrist_landmark.y), depth=thumb_tip_landmark.z)
            
            #----- INDEX -----
            index_mcp = mp_hands.HandLandmark.INDEX_FINGER_MCP
            index_pip = mp_hands.HandLandmark.INDEX_FINGER_PIP
            index_dip = mp_hands.HandLandmark.INDEX_FINGER_DIP
            index_tip = mp_hands.HandLandmark.INDEX_FINGER_TIP
            # --index landmark --
            index_mcp_landmark = hand_landmarks.landmark[index_mcp]
            index_pip_landmark = hand_landmarks.landmark[index_pip]
            index_dip_landmark = hand_landmarks.landmark[index_dip]
            index_tip_landmark = hand_landmarks.landmark[index_tip]
            #-- calculate depth factor for index finger --
            index_mcp_X, index_mcp_Y = depth_factor((index_mcp_landmark.x - wrist_landmark.x, index_mcp_landmark.y - wrist_landmark.y), depth=index_mcp_landmark.z)
            index_pip_X, index_pip_Y = depth_factor((index_pip_landmark.x - wrist_landmark.x, index_pip_landmark.y - wrist_landmark.y), depth=index_pip_landmark.z)
            index_dip_X, index_dip_Y = depth_factor((index_dip_landmark.x - wrist_landmark.x, index_dip_landmark.y - wrist_landmark.y), depth=index_dip_landmark.z)
            index_tip_X, index_tip_Y = depth_factor((index_tip_landmark.x - wrist_landmark.x, index_tip_landmark.y - wrist_landmark.y), depth=index_tip_landmark.z)

            #---- MIDDLE ------
            middle_mcp = mp_hands.HandLandmark.MIDDLE_FINGER_MCP
            middle_pip = mp_hands.HandLandmark.MIDDLE_FINGER_PIP
            middle_dip = mp_hands.HandLandmark.MIDDLE_FINGER_DIP
            middle_tip = mp_hands.HandLandmark.MIDDLE_FINGER_TIP
            # --middle landmark --
            middle_mcp_landmark = hand_landmarks.landmark[middle_mcp]
            middle_pip_landmark = hand_landmarks.landmark[middle_pip]
            middle_dip_landmark = hand_landmarks.landmark[middle_dip]
            middle_tip_landmark = hand_landmarks.landmark[middle_tip]
            #-- calculate depth factor for middle finger --
            middle_mcp_X, middle_mcp_Y = depth_factor((middle_mcp_landmark.x - wrist_landmark.x, middle_mcp_landmark.y - wrist_landmark.y), depth=middle_mcp_landmark.z)
            middle_pip_X, middle_pip_Y = depth_factor((middle_pip_landmark.x - wrist_landmark.x, middle_pip_landmark.y - wrist_landmark.y), depth=middle_pip_landmark.z)
            middle_dip_X, middle_dip_Y = depth_factor((middle_dip_landmark.x - wrist_landmark.x, middle_dip_landmark.y - wrist_landmark.y), depth=middle_dip_landmark.z)
            middle_tip_X, middle_tip_Y = depth_factor((middle_tip_landmark.x - wrist_landmark.x, middle_tip_landmark.y - wrist_landmark.y), depth=middle_tip_landmark.z)
            # Separate print statements based on handedness
            # RING FINGER
            ring_mcp = mp_hands.HandLandmark.RING_FINGER_MCP
            ring_pip = mp_hands.HandLandmark.RING_FINGER_PIP
            ring_dip = mp_hands.HandLandmark.RING_FINGER_DIP
            ring_tip = mp_hands.HandLandmark.RING_FINGER_TIP
            # --ring landmark --
            ring_mcp_landmark = hand_landmarks.landmark[ring_mcp]
            ring_pip_landmark = hand_landmarks.landmark[ring_pip]
            ring_dip_landmark = hand_landmarks.landmark[ring_dip]
            ring_tip_landmark = hand_landmarks.landmark[ring_tip]
            #-- calculate depth factor for ring finger --
            ring_mcp_X, ring_mcp_Y = depth_factor((ring_mcp_landmark.x - wrist_landmark.x, ring_mcp_landmark.y - wrist_landmark.y), depth=ring_mcp_landmark.z)
            ring_pip_X, ring_pip_Y = depth_factor((ring_pip_landmark.x - wrist_landmark.x, ring_pip_landmark.y - wrist_landmark.y), depth=ring_pip_landmark.z)
            ring_dip_X, ring_dip_Y = depth_factor((ring_dip_landmark.x - wrist_landmark.x, ring_dip_landmark.y - wrist_landmark.y), depth=ring_dip_landmark.z)
            ring_tip_X, ring_tip_Y = depth_factor((ring_tip_landmark.x - wrist_landmark.x, ring_tip_landmark.y - wrist_landmark.y), depth=ring_tip_landmark.z)
            # PINKY FINGER
            pinky_mcp = mp_hands.HandLandmark.PINKY_MCP
            pinky_pip = mp_hands.HandLandmark.PINKY_PIP
            pinky_dip = mp_hands.HandLandmark.PINKY_DIP
            pinky_tip = mp_hands.HandLandmark.PINKY_TIP
            # --pinky landmark --
            pinky_mcp_landmark = hand_landmarks.landmark[pinky_mcp]
            pinky_pip_landmark = hand_landmarks.landmark[pinky_pip]
            pinky_dip_landmark = hand_landmarks.landmark[pinky_dip]
            pinky_tip_landmark = hand_landmarks.landmark[pinky_tip]
            #-- calculate depth factor for pinky finger --
            pinky_mcp_X, pinky_mcp_Y = depth_factor((pinky_mcp_landmark.x - wrist_landmark.x, pinky_mcp_landmark.y - wrist_landmark.y), depth=pinky_mcp_landmark.z)
            pinky_pip_X, pinky_pip_Y = depth_factor((pinky_pip_landmark.x - wrist_landmark.x, pinky_pip_landmark.y - wrist_landmark.y), depth=pinky_pip_landmark.z)
            pinky_dip_X, pinky_dip_Y = depth_factor((pinky_dip_landmark.x - wrist_landmark.x, pinky_dip_landmark.y - wrist_landmark.y), depth=pinky_dip_landmark.z)
            pinky_tip_X, pinky_tip_Y = depth_factor((pinky_tip_landmark.x - wrist_landmark.x, pinky_tip_landmark.y - wrist_landmark.y), depth=pinky_tip_landmark.z)        
            
            if handedness == 'Left':
                print(f'LEFT HAND THUMB: CMC: [{thumb_cmc_X},{thumb_cmc_Y}] | MCP: [{thumb_mcp_X},{thumb_mcp_Y}] | IP: [{thumb_ip_X},{thumb_ip_Y}] | TIP: [{thumb_tip_X},{thumb_tip_Y}]')
                print(f'LEFT HAND INDEX: CMC: [{index_mcp_X},{index_mcp_Y}] | MCP: [{index_pip_X},{index_pip_Y}] | IP: [{index_dip_X},{index_dip_Y}] | TIP: [{index_tip_X},{index_tip_Y}]')
                print(f'LEFT HAND MIDDLE: MCP: [{middle_mcp_X},{middle_mcp_Y}] | PIP: [{middle_pip_X},{middle_pip_Y}] | DIP: [{middle_dip_X},{middle_dip_Y}] | TIP: [{middle_tip_X},{middle_tip_Y}]')
                print(f'LEFT HAND RING: MCP: [{ring_mcp_X},{ring_mcp_Y}] | PIP: [{ring_pip_X},{ring_pip_Y}] | DIP: [{ring_dip_X},{ring_dip_Y}] | TIP: [{ring_tip_X},{ring_tip_Y}]')
                print(f'LEFT HAND PINKY: MCP: [{pinky_mcp_X},{pinky_mcp_Y}] | PIP: [{pinky_pip_X},{pinky_pip_Y}] | DIP: [{pinky_dip_X},{pinky_dip_Y}] | TIP: [{pinky_tip_X},{pinky_tip_Y}]')
            elif handedness == 'Right':
                print(f'RIGHT HAND THUMB: CMC: [{thumb_cmc_X},{thumb_cmc_Y}] | MCP: [{thumb_mcp_X},{thumb_mcp_Y}] | IP: [{thumb_ip_X},{thumb_ip_Y}] | TIP: [{thumb_tip_X},{thumb_tip_Y}]')
                print(f'RIGHT HAND INDEX: CMC: [{index_mcp_X},{index_mcp_Y}] | MCP: [{index_pip_X},{index_pip_Y}] | IP: [{index_dip_X},{index_dip_Y}] | TIP: [{index_tip_X},{index_tip_Y}, Y: {index_dip_landmark.z:.4f}]')
                print(f'RIGHT HAND MIDDLE: MCP: [{middle_mcp_X},{middle_mcp_Y}] | PIP: [{middle_pip_X},{middle_pip_Y}] | DIP: [{middle_dip_X},{middle_dip_Y}] | TIP: [{middle_tip_X},{middle_tip_Y}]')
                print(f'RIGHT HAND RING: MCP: [{ring_mcp_X},{ring_mcp_Y}] | PIP: [{ring_pip_X},{ring_pip_Y}] | DIP: [{ring_dip_X},{ring_dip_Y}] | TIP: [{ring_tip_X},{ring_tip_Y}]')
                print(f'RIGHT HAND PINKY: MCP: [{pinky_mcp_X},{pinky_mcp_Y}] | PIP: [{pinky_pip_X},{pinky_pip_Y}] | DIP: [{pinky_dip_X},{pinky_dip_Y}] | TIP: [{pinky_tip_X},{pinky_tip_Y}]')
            # Get wrist position
            wrist_position = (wrist_landmark.x, wrist_landmark.y)

            #relative_position = get_relative_position(wrist_position, frame_center)


            # 2. Print coordinates based on handedness
            #print(f"{handedness} Wrist: X: {wrist_landmark.x:.4f} | Y: {wrist_landmark.y:.4f}")
            #print(f'{handedness} Thumb_cmc: X {thumb_cmc_landmark.x:.4f} | Y: {thumb_cmc_landmark.y:.4f}')
            # Draw landmarks
            drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) # Use mp_hands alias here
            
    # Display the frame
    cv2.imshow("Hand Joints", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
 

    
cap.release()
cv2.destroyAllWindows()
hands.close()
