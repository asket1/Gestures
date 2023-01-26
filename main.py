import cv2
import mediapipe as mp
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def steer_left(x):
    print("left", x)
def steer_right(x):
    print("right", x)
def drive_forward():
    print("forward")
def extra_function():
    print("extra")

def control_function():
    INDEX_BASE = (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x - hand_landmarks.landmark[
        mp_hands.HandLandmark.INDEX_FINGER_MCP].x) ** 2 + \
                 (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y - hand_landmarks.landmark[
                     mp_hands.HandLandmark.INDEX_FINGER_MCP].y) ** 2 + \
                 (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z - hand_landmarks.landmark[
                     mp_hands.HandLandmark.INDEX_FINGER_MCP].z) ** 2
    INDEX_TIP = (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x - hand_landmarks.landmark[
        mp_hands.HandLandmark.INDEX_FINGER_TIP].x) ** 2 + \
                (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y - hand_landmarks.landmark[
                    mp_hands.HandLandmark.INDEX_FINGER_TIP].y) ** 2 + \
                (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z - hand_landmarks.landmark[
                    mp_hands.HandLandmark.INDEX_FINGER_TIP].z) ** 2
    INDEX_COEF = INDEX_TIP/INDEX_BASE-1
    MIDDLE_BASE = (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x - hand_landmarks.landmark[
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x) ** 2 + \
                 (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y - hand_landmarks.landmark[
                     mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y) ** 2 + \
                 (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z - hand_landmarks.landmark[
                     mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z) ** 2
    MIDDLE_TIP = (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x - hand_landmarks.landmark[
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x) ** 2 + \
                (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y - hand_landmarks.landmark[
                    mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y) ** 2 + \
                (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z - hand_landmarks.landmark[
                    mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z) ** 2
    MIDDLE_COEF = MIDDLE_TIP/MIDDLE_BASE-1

    RING_BASE = (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x - hand_landmarks.landmark[
        mp_hands.HandLandmark.RING_FINGER_MCP].x) ** 2 + \
                  (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y - hand_landmarks.landmark[
                      mp_hands.HandLandmark.RING_FINGER_MCP].y) ** 2 + \
                  (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z - hand_landmarks.landmark[
                      mp_hands.HandLandmark.RING_FINGER_MCP].z) ** 2
    RING_TIP = (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x - hand_landmarks.landmark[
        mp_hands.HandLandmark.RING_FINGER_TIP].x) ** 2 + \
                 (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y - hand_landmarks.landmark[
                     mp_hands.HandLandmark.RING_FINGER_TIP].y) ** 2 + \
                 (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z - hand_landmarks.landmark[
                     mp_hands.HandLandmark.RING_FINGER_TIP].z) ** 2
    RING_COEF = RING_TIP / RING_BASE - 1

    PINKY_BASE = (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x - hand_landmarks.landmark[
        mp_hands.HandLandmark.PINKY_MCP].x) ** 2 + \
                (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y - hand_landmarks.landmark[
                    mp_hands.HandLandmark.PINKY_MCP].y) ** 2 + \
                (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z - hand_landmarks.landmark[
                    mp_hands.HandLandmark.PINKY_MCP].z) ** 2
    PINKY_TIP = (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x - hand_landmarks.landmark[
        mp_hands.HandLandmark.PINKY_TIP].x) ** 2 + \
               (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y - hand_landmarks.landmark[
                   mp_hands.HandLandmark.PINKY_TIP].y) ** 2 + \
               (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z - hand_landmarks.landmark[
                   mp_hands.HandLandmark.PINKY_TIP].z) ** 2
    PINKY_COEF = PINKY_TIP / PINKY_BASE - 1

    if INDEX_COEF > 0 and MIDDLE_COEF < 0 and RING_COEF < 0 and PINKY_COEF < 0:
        INDEX_LENGTH = (hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x - hand_landmarks.landmark[
        mp_hands.HandLandmark.INDEX_FINGER_TIP].x) ** 2 + \
                 (hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y - hand_landmarks.landmark[
                     mp_hands.HandLandmark.INDEX_FINGER_TIP].y) ** 2
        INDEX_X = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x - hand_landmarks.landmark[
        mp_hands.HandLandmark.INDEX_FINGER_TIP].x
        INDEX_ANGLE = INDEX_X/INDEX_LENGTH
        if INDEX_ANGLE > 2:
            steer_right(INDEX_ANGLE)
        elif INDEX_ANGLE < -2:
            steer_left(abs(INDEX_ANGLE))
    if INDEX_COEF < 0 and MIDDLE_COEF < 0 and RING_COEF < 0 and PINKY_COEF < 0:
        drive_forward()
    if INDEX_COEF > 0 and MIDDLE_COEF < 0 and RING_COEF < 0 and PINKY_COEF > 0:
        extra_function()

cap = cv2.VideoCapture(1)
with mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        control_function()

    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
