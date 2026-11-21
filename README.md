# Real-Time Sign Language Recognizer

A webcam app that recognizes sign language handshapes in real time using
MediaPipe hand landmarks and machine learning. Currently supports 5 ASL letters
(A, B, C, L, Y), with the full alphabet and Arabic Sign Language planned.

<!-- Add a demo GIF here later -->

## How it works

1. **MediaPipe Hand Landmarker** extracts 21 hand landmarks per frame.
2. **Features:** 210 pairwise distances between landmarks, scaled by palm size.
   These don't change with hand position, tilt, distance from the camera, or
   left/right hand.
3. **Classifier:** Random Forest.
4. **Live demo:** prediction smoothing over recent frames, and holding a sign
   for 1 second types the letter.

## Results

Evaluation is leave-one-session-out: the model is trained on 3 recording
sessions (different hand positions) and tested on the 4th, which it never saw.

| Step | What changed | Accuracy |
|---|---|---|
| 1 | Raw landmarks, single recording sweep, time-ordered split | ~0.50 |
| 2 | Rotation- and mirror-invariant distance features | ~0.73 |
| 3 | Data recorded in 4 separate sessions (close, far, left, right) | 0.945 |
| 4 | Removed one mislabeled recording (verified visually) | 0.992 |

A random train/test split scored about 0.92 even at step 1, because
near-identical neighboring frames end up in both sets. This is why the
session-based split matters.

## Run it

```bash
python3.11 -m venv slr-env
source slr-env/bin/activate
pip install mediapipe opencv-python numpy pandas scikit-learn scipy joblib matplotlib

python src/hand_demo.py                       # first run downloads the hand model
python src/collect_sessions.py --signer yourname
python src/train_final.py
python src/live_demo.py
```

## Limitations

- Data from one signer and one camera setup so far; results on new people are
  not yet measured.
- Static fingerspelled letters only, not full sign language (grammar, facial
  expression, motion).
- The classifier always answers with one of its known letters, even for
  unfamiliar handshapes.

## Roadmap

- [ ] Record more signers and report leave-one-signer-out accuracy
- [ ] Full ASL alphabet
- [ ] Arabic Sign Language
- [ ] Dynamic signs (short words) with a sequence model


### Model comparison (leave-one-session-out, 5 letters)

| Model | Session 1 | Session 2 | Session 3 | Session 4 | Mean |
|---|---|---|---|---|---|
| Logistic Regression | 1.000 | 0.996 | 0.987 | 0.990 | 0.993 |
| SVM (RBF) | 0.997 | 0.993 | 0.989 | 0.988 | 0.992 |
| Random Forest | 0.997 | 0.992 | 0.986 | 0.991 | 0.992 |
| MLP (PyTorch) | 1.000 | 0.996 | 0.986 | 0.991 | 0.993 |

All four models score within 0.1 percentage points, so the features and data
collection matter more than the choice of model here. With one signer and five
distinct letters this test cannot separate them; harder letters and new signers should.