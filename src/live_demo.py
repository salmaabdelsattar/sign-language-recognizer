import time
from collections import Counter, deque

import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from features import normalize

MODEL_PATH = "models/hand_landmarker.task"
CLASSIFIER_PATH = "models/sign_model.joblib"
MIN_CONF = 0.6        # ignore low-confidence predictions
VOTE_NEEDED = 7       # out of the last 10 frames must agree
COMMIT_SECONDS = 1.0  # hold a sign this long to type it

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

clf = joblib.load(CLASSIFIER_PATH)
detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=1,
        min_hand_detection_confidence=0.6,
    )
)

history = deque(maxlen=10)
text = ""
hold_letter, hold_start, committed = None, 0.0, False

cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    smoothed, conf = None, 0.0
    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        h, w = frame.shape[:2]
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for p in pts:
            cv2.circle(frame, p, 4, (0, 0, 255), -1)

        row = [v for p in lm for v in (p.x, p.y, p.z)]
        proba = clf.predict_proba(normalize(row).reshape(1, -1))[0]
        i = proba.argmax()
        conf = proba[i]
        history.append(clf.classes_[i] if conf >= MIN_CONF else None)

        votes = Counter(l for l in history if l is not None)
        if votes:
            letter, count = votes.most_common(1)[0]
            if count >= VOTE_NEEDED:
                smoothed = letter
    else:
        history.clear()

    # Hold a steady sign for COMMIT_SECONDS to type it
    now = time.time()
    if smoothed != hold_letter:
        hold_letter, hold_start, committed = smoothed, now, False
    elif smoothed is not None and not committed and now - hold_start >= COMMIT_SECONDS:
        text += smoothed
        committed = True

    shown = smoothed if smoothed else "?"
    cv2.putText(frame, f"{shown}  ({conf:.2f})", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
    cv2.putText(frame, f"Text: {text}", (20, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, "SPACE=space  C=clear  Esc=quit", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.imshow("sign language demo", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == 32:
        text += " "
    elif key == ord("c"):
        text = ""

cap.release()
cv2.destroyAllWindows()