import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Finger tip landmark IDs (Index, Middle, Ring, Pinky)
tip_ids = [8, 12, 16, 20]

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    # Convert BGR to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            landmarks = []
            for id, lm in enumerate(hand_lms.landmark):
                # Convert normalized coordinates to pixel coordinates
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append([id, cx, cy])

            if landmarks:
                fingers = []

                # Thumb: Compare x-coordinates (assuming right hand)
                if landmarks[4][1] > landmarks[3][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # Other 4 fingers: Compare y-coordinates (tip vs joint below it)
                for tip in tip_ids:
                    if landmarks[tip][2] < landmarks[tip - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                total_fingers = fingers.count(1)
                cv2.putText(img, f'Fingers: {total_fingers}', (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw hand landmarks on the image
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Finger Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
