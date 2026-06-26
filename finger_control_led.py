import cv2
import mediapipe as mp
import serial
import time

# Serial port setup — change 'COM3' to your Arduino port
# On Linux/Mac it will be something like '/dev/ttyUSB0' or '/dev/ttyACM0'
# ser = serial.Serial('COM8', 9600, timeout=1)
time.sleep(2)  # Wait for Arduino to reset after serial connection

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

tip_ids = [8, 12, 16, 20]

# Finger names for display
finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']

# Track previous state to avoid sending repeated commands
prev_fingers = [-1, -1, -1, -1, -1]

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            landmarks = []
            for id, lm in enumerate(hand_lms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append([id, cx, cy])

            if landmarks:
                fingers = []

                # Thumb (x-axis)
                if landmarks[4][1] > landmarks[3][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # Other 4 fingers (y-axis)
                for tip in tip_ids:
                    if landmarks[tip][2] < landmarks[tip - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # Send serial command only when finger state changes
                # Commands: A0# to A4# = LED ON  (finger index 0–4)
                #           B0# to B4# = LED OFF (finger index 0–4)
                for i in range(5):
                    if fingers[i] != prev_fingers[i]:
                        if fingers[i] == 1:
                            cmd = f'A{i}#'   # ON
                        else:
                            cmd = f'B{i}#'   # OFF
                        # ser.write(cmd.encode())
                        # print(f'Sent: {cmd} ({finger_names[i]} {"ON" if fingers[i] else "OFF"})')

                prev_fingers = fingers[:]

                total_fingers = fingers.count(1)
                cv2.putText(img, f'Fingers: {total_fingers}', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Show individual finger states on screen
                for i, name in enumerate(finger_names):
                    state = 'ON' if fingers[i] else 'OFF'
                    color = (0, 255, 0) if fingers[i] else (0, 0, 255)
                    cv2.putText(img, f'{name}: {state}', (50, 90 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Finger LED Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Turn off all LEDs on exit
# for i in range(5):
#     ser.write(f'B{i}#'.encode())

cap.release()
cv2.destroyAllWindows()
# ser.close()