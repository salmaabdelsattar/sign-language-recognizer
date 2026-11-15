import argparse
import csv
import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/hand_landmarker.task"
CSV_PATH = "data/landmarks.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True, help="sign name, e.g. A or alif")
parser.add_argument("--lang", required=True, help="asl or arsl")
parser.add_argument("--signer", required=True, help="who is signing, e.g. salma")
parser.add_argument("--samples", type=int, default=200)
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
    header = ["lang", "signer", "label", "hand"]
    for i in range(21):
        header += [f"x{i}", f"y{i}", f"z{i}"]
    writer.writerow(header)

cap = cv2.VideoCapture(0)
start = time.time()
collected, frame_count = 0, 0

while collected < args.samples:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    hand_found = bool(result.hand_landmarks)
    counting_down = time.time() - start < 3

    if hand_found:
        h, w = frame.shape[:2]
        for p in result.hand_landmarks[0]:
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 4, (0, 0, 255), -1)

    if not counting_down and hand_found:
        frame_count += 1
        if frame_count % 3 == 0:  # skip frames so samples differ a bit
            row = [args.lang, args.signer, args.label,
                   result.handedness[0][0].category_name]
            for p in result.hand_landmarks[0]:
                row += [p.x, p.y, p.z]
            writer.writerow(row)
            collected += 1

    if counting_down:
        text = f"Get ready: {args.label}  {3 - int(time.time() - start)}"
    else:
        text = f"Recording {args.label}: {collected}/{args.samples}"
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("collect (Esc to quit)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

f.close()
cap.release()
cv2.destroyAllWindows()
print(f"Saved {collected} samples for {args.label}")