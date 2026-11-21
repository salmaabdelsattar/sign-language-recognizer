import argparse
import csv
import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/hand_landmarker.task"
CSV_PATH = "data/landmarks_none.csv"
WIN = "collect NONE"
SESSIONS = [
    "hand CLOSE to the camera, centered",
    "hand FAR from the camera, centered",
    "hand on the LEFT of the frame, tilted a little",
    "hand on the RIGHT of the frame, tilted the other way",
]
POSES = [
    "OPEN palm, fingers spread wide",
    "PEACE sign (index + middle fingers up)",
    "POINT with your index finger",
    "CURL: slowly open and close your fingers",
]

parser = argparse.ArgumentParser()
parser.add_argument("--signer", required=True)
parser.add_argument("--lang", default="asl")
parser.add_argument("--samples", type=int, default=50)
parser.add_argument("--start-session", type=int, default=1)
args = parser.parse_args()

options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=1,
    min_hand_detection_confidence=0.6,
)
detector = vision.HandLandmarker.create_from_options(options)

os.makedirs("data", exist_ok=True)
new_file = not os.path.exists(CSV_PATH)
f = open(CSV_PATH, "a", newline="")
writer = csv.writer(f)
if new_file:
    header = ["lang", "signer", "session", "label"]
    for i in range(21):
        header += [f"x{i}", f"y{i}", f"z{i}"]
    writer.writerow(header)

cap = cv2.VideoCapture(0)


def read_frame():
    ok, frame = cap.read()
    if not ok:
        return None, None
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if result.hand_landmarks:
        h, w = frame.shape[:2]
        for p in result.hand_landmarks[0]:
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 4, (0, 0, 255), -1)
    return frame, result


def show(frame, line1, line2=""):
    cv2.putText(frame, line1, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if line2:
        cv2.putText(frame, line2, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow(WIN, frame)
    return cv2.waitKey(1) & 0xFF


quit_all = False
for s_idx in range(args.start_session - 1, len(SESSIONS)):
    session = s_idx + 1
    for pose in POSES:
        while True:
            frame, result = read_frame()
            if frame is None:
                quit_all = True
                break
            key = show(frame,
                       f"Session {session}/{len(SESSIONS)}: {SESSIONS[s_idx]}",
                       f"{pose} - SPACE to start (Esc quits)")
            if key == 27:
                quit_all = True
                break
            if key == 32:
                break
        if quit_all:
            break

        t0 = time.time()
        while time.time() - t0 < 2:
            frame, result = read_frame()
            if frame is None:
                break
            show(frame, f"Get ready: {pose}", f"{2 - int(time.time() - t0)}")

        collected, frame_count = 0, 0
        while collected < args.samples:
            frame, result = read_frame()
            if frame is None:
                quit_all = True
                break
            if result.hand_landmarks:
                frame_count += 1
                if frame_count % 2 == 0:
                    row = [args.lang, args.signer, session, "NONE"]
                    for p in result.hand_landmarks[0]:
                        row += [p.x, p.y, p.z]
                    writer.writerow(row)
                    collected += 1
            key = show(frame, f"Recording {pose}: {collected}/{args.samples}",
                       "Do NOT make A, B, C, L or Y")
            if key == 27:
                quit_all = True
                break
        f.flush()
        print(f"session {session} | {pose}: {collected} samples")
        if quit_all:
            break
    if quit_all:
        break

f.close()
cap.release()
cv2.destroyAllWindows()