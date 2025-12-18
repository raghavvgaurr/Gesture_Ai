import cv2
import mediapipe as mp
import numpy as np
import joblib

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

model = joblib.load("sign_rf_model.pkl")
le = joblib.load("sign_label_encoder.pkl")

cap = cv2.VideoCapture(0)

def extract_features(lm):
    row = []
    for p in lm.landmark:
        row += [p.x, p.y, p.z]
    return np.array(row).reshape(1, -1)

with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
    while True:
        ret, img = cap.read()
        if not ret:
            continue

        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        gesture_text = ""

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(img, lm, mp_hands.HAND_CONNECTIONS)

            feat = extract_features(lm)
            pred = model.predict(feat)[0]
            gesture_text = le.inverse_transform([pred])[0].upper()

        cv2.putText(img, gesture_text, (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

        cv2.imshow("Sign Language -> Text", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
