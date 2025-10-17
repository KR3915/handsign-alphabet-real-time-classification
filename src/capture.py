import cv2
import mediapipe as mp
# --- Initialize MediaPipe Hands and OpenCV Video Capture ---
cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands()
drawing = mp.solutions.drawing_utils
# ---check if the webcam is capturing correctly---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            drawing.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
    # Display the frame
    cv2.imshow("Hand Joints", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
