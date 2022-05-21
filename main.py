import cv2
import mediapipe as mp
import os

def function1():
    # depth = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z - \
    #          hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z

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

    if MIDDLE_COEF < 0 and INDEX_COEF > 0:
        # INDEX_ANGLE_X = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x - hand_landmarks.landmark[
        # mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
        # INDEX_ANGLE_Y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y - hand_landmarks.landmark[
        #     mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
        # INDEX_ANGLE = INDEX_ANGLE_X/INDEX_ANGLE_Y
        INDEX_LENGTH = (hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x - hand_landmarks.landmark[
        mp_hands.HandLandmark.INDEX_FINGER_TIP].x) ** 2 + \
                 (hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y - hand_landmarks.landmark[
                     mp_hands.HandLandmark.INDEX_FINGER_TIP].y) ** 2
        INDEX_X = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x - hand_landmarks.landmark[
        mp_hands.HandLandmark.INDEX_FINGER_TIP].x
        INDEX_ANGLE = INDEX_X/INDEX_LENGTH
        if INDEX_ANGLE > 2:
            print('left')
        elif INDEX_ANGLE < -2:
            print('right')

    if INDEX_COEF < 0 and MIDDLE_COEF < 0 and RING_COEF < 0 and PINKY_COEF < 0:
        print('stop')
    #print(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x)
    #print(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y)
    # print(INDEX_COEF)
    # print(MIDDLE_COEF)
    # print(RING_COEF)
    # print(PINKY_COEF)
    #
    # if height > 0.22:
    #     print('close')
    # if height < 0.18:
    #     print('far')
    length1 = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
IMAGE_FILES = []
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    cv2.imwrite(
        '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    # Draw hand world landmarks.
    if not results.multi_hand_world_landmarks:
      continue
    for hand_world_landmarks in results.multi_hand_world_landmarks:
      mp_drawing.plot_landmarks(
        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)


# For webcam input:
if os.path.exists("1.txt"):
  os.remove("1.txt")
cap = cv2.VideoCapture(1)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
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
        function1()

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
