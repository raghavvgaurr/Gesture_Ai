import cv2
import mediapipe as mp
import csv
import os
import warnings
warnings.filterwarnings("ignore")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

DATA_DIR = "sign_data"
os.makedirs(DATA_DIR, exist_ok=True)

gesture_name = input("Enter gesture name (yes, no, hello, bathroom, none): ").lower()
csv_path = os.path.join(DATA_DIR, f"{gesture_name}.csv")

# Write header
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([f"lm{i}_{axis}" for i in range(21) for axis in ["x", "y", "z"]])

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    print("Press 'q' to stop recording")

    while True:
        ret, img = cap.read()
        if not ret:
            continue

        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0]

            # ✅ DRAW LANDMARKS
            mp_draw.draw_landmarks(
                img,
                lm,
                mp_hands.HAND_CONNECTIONS
            )

            # save data
            row = []
            for p in lm.landmark:
                row += [p.x, p.y, p.z]

            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow(row)

        cv2.imshow("Collecting Gesture Data", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print(f"Saved data to {csv_path}")


