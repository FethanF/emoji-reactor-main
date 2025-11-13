#!/usr/bin/env python3
"""
Real-time emoji display based on camera pose and facial expression detection.
"""
import traceback
import cv2
import numpy as np
import mediapipe as mp
# NEW IMPORTS FOR GESTURE RECOGNIZER
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.python.solutions import drawing_utils


# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

# Configuration
SMILE_THRESHOLD = 0.0155
SAD_THRESHOLD = -0.001
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 450
EMOJI_WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)
textToggle = 1

# --- NEW: GESTURE RECOGNIZER SETUP ---
MODEL_PATH = 'gesture_recognizer.task'
try:
    # BaseOptions: Path to the model file
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    
    # GestureRecognizerOptions: Settings for the recognizer (using IMAGE mode for simplicity)
    options = vision.GestureRecognizerOptions(base_options=base_options,
                                               running_mode=vision.RunningMode.IMAGE,
                                               num_hands=2) # Detect up to two hands

    # Create the recognizer instance
    recognizer = vision.GestureRecognizer.create_from_options(options)

except Exception as e:
    print(f"Error initializing Gesture Recognizer. Ensure '{MODEL_PATH}' is in the directory and MediaPipe is updated.")
    print(f"Details: {e}")
    exit()
# -------------------------------------

# Load emoji images
try:
    smiling_emoji = cv2.imread("smile.jpg")
    straight_face_emoji = cv2.imread("plain.png")
    hands_up_emoji = cv2.imread("air.jpg")
    sad_emoji = cv2.imread("sad.jpg")
    thumbs_up_emoji = cv2.imread("thumb.jpg")

    if smiling_emoji is None:
        raise FileNotFoundError("smile.jpg not found")
    if straight_face_emoji is None:
        raise FileNotFoundError("plain.png not found")
    if hands_up_emoji is None:
        raise FileNotFoundError("air.jpg not found")
    if sad_emoji is None:
        raise FileNotFoundError("sad.jpg not found")
    if thumbs_up_emoji is None:
        raise FileNotFoundError("thumb.jpg not found")

    # Resize emojis
    smiling_emoji = cv2.resize(smiling_emoji, EMOJI_WINDOW_SIZE)
    straight_face_emoji = cv2.resize(straight_face_emoji, EMOJI_WINDOW_SIZE)
    hands_up_emoji = cv2.resize(hands_up_emoji, EMOJI_WINDOW_SIZE)
    thumbs_up_emoji = cv2.resize(thumbs_up_emoji, EMOJI_WINDOW_SIZE)
    sad_emoji = cv2.resize(sad_emoji, EMOJI_WINDOW_SIZE)
    
except Exception as e:
    print("Error loading emoji images!")
    print(f"Details: {e}")
    print("\nExpected files:")
    print("- smile.jpg (smiling face)")
    print("- plain.png (straight face)")
    print("- air.jpg (hands up)")
    exit()

blank_emoji = np.zeros((EMOJI_WINDOW_SIZE[0], EMOJI_WINDOW_SIZE[1], 3), dtype=np.uint8)

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cv2.namedWindow('Emoji Output', cv2.WINDOW_NORMAL)
cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera Feed', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.resizeWindow('Emoji Output', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.moveWindow('Camera Feed', 100, 100)
cv2.moveWindow('Emoji Output', WINDOW_WIDTH + 150, 100)

print("Controls:")
print("  Press 'q' to quit")
print("  Raise hands above shoulders for hands up")
print("  Perform 'thumbs up' for thumbs up emoji")
print("  Smile for smiling emoji")
print("  Straight face for neutral emoji")

# mp_pose is kept running ONLY for the 'Hands Up' body pose check
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh, \
     mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        current_state = "STRAIGHT_FACE"
        mouth_aspect_ratio = 0.0 # Initialize for printing

        body_pose = pose.process(image_rgb)

        # --- 0. DRAW HAND LANDMARKS ---
        try:
            results = hands.process(image_rgb)
            image_rgb.flags.writeable = True

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            
        except Exception as e:
            print("Error: Failed to draw pose:", e)
            traceback.print_exc()


        # --- 1. CHECK FOR THUMBS UP (USING NEW MODEL) ---
        # Convert the frame to a MediaPipe Image object
        recognition_result = recognizer.recognize(mp_image)

        if recognition_result.gestures:
            # The result contains a list of gestures, one list per hand
            for gesture_list in recognition_result.gestures:
                # Loop through all detected gestures (usually only one per hand)
                for gesture in gesture_list:
                    # Check for the standardized gesture name
                    if gesture.category_name == "Thumb_Up":
                        current_state = "THUMBS_UP"
                        break
                if current_state == "THUMBS_UP":
                    break
        # ------------------------------------------------

        # --- 2. CHECK FOR HANDS UP (USING BODY POSE) ---
        # Only check this if THUMBS_UP wasn't detected
        if current_state != "THUMBS_UP" and body_pose.pose_landmarks:
            landmarks = body_pose.pose_landmarks.landmark

            # This logic checks if wrists are above shoulders
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            if (left_wrist.y < left_shoulder.y) and (right_wrist.y < right_shoulder.y):
                current_state = "HANDS_UP"
        
        # --- 3. CHECK FACIAL EXPRESSION ---
        # Only check if neither hand gesture/pose was detected
        if current_state != "HANDS_UP" and current_state != "THUMBS_UP":
            results_face = face_mesh.process(image_rgb)
            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    left_corner = face_landmarks.landmark[291]
                    right_corner = face_landmarks.landmark[61]
                    upper_lip = face_landmarks.landmark[13]
                    lower_lip = face_landmarks.landmark[14]

                    # check if corners of lips are higher or lower than center of lips
                    y_of_corners = (left_corner.y + right_corner.y) / 2
                    y_of_centers = (lower_lip.y + upper_lip.y) / 2

                    # positive -> corners are above lips
                    # negative -> corners are under lips
                    y_diff = y_of_centers - y_of_corners

                    if y_diff > SMILE_THRESHOLD:
                        current_state = "SMILING"
                    elif y_diff < SAD_THRESHOLD:
                        current_state = "SAD"
                    else:
                        current_state = "STRAIGHT_FACE"
        
        # --- 4. SELECT EMOJI BASED ON FINAL STATE ---
        if current_state == "SMILING":
            emoji_to_display = smiling_emoji
            emoji_name = "😊"
        elif current_state == "STRAIGHT_FACE":
            emoji_to_display = straight_face_emoji
            emoji_name = "😐"
        elif current_state == "HANDS_UP":
            emoji_to_display = hands_up_emoji
            emoji_name = "🙌"
        elif current_state == "SAD":
            emoji_to_display = sad_emoji
            emoji_name = "😢"
        elif current_state == "THUMBS_UP":
            emoji_to_display = thumbs_up_emoji
            emoji_name = "👍"
        else:
            emoji_to_display = blank_emoji
            emoji_name = "❓"

        camera_frame_resized = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        
        if textToggle > 0:
            cv2.putText(camera_frame_resized, f'STATE: {current_state}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(camera_frame_resized, 'Press "q" to quit', (10, WINDOW_HEIGHT - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Camera Feed', camera_frame_resized)
        cv2.imshow('Emoji Output', emoji_to_display)

        key = cv2.waitKey(5) & 0xFF

        if key == ord('q'):
            break # exit
        elif key == ord('t'):
            textToggle = textToggle * -1

cap.release()
cv2.destroyAllWindows()