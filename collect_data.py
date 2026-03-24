"""
collect_data.py  —  Step 1
Full alphabet A-Z + SPACE gesture + DELETE gesture
Works with MediaPipe 0.10.13+
"""

import cv2
import os
import csv
import mediapipe as mp

LETTERS    = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["SPACE", "DELETE"]
SAMPLES    = 200
CSV_FILE   = "landmarks.csv"
MODEL_PATH = "hand_landmarker.task"

# ── Auto-download model ───────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmark model (~9 MB) ...")
    import urllib.request
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )
    print("Download complete!\n")

# ── MediaPipe setup ───────────────────────────────────────────────────────────
BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)

# ── Check existing CSV data ───────────────────────────────────────────────────
existing_counts = {letter: 0 for letter in LETTERS}
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0] in existing_counts:
                existing_counts[row[0]] += 1
    print("Existing data found:")
    for l, c in existing_counts.items():
        status = "DONE" if c >= SAMPLES else f"{c}/{SAMPLES}"
        print(f"  {l:7s}: {status}")
    print()

# ── Gesture hints ─────────────────────────────────────────────────────────────
HINTS = {
    "A": "Fist, thumb on side",
    "B": "4 fingers up, thumb folded in",
    "C": "Curved hand like letter C",
    "D": "Index up, others curl to thumb",
    "E": "All fingers curl down, thumb tucked",
    "F": "Index+thumb touch, others spread up",
    "G": "Index+thumb point sideways",
    "H": "Index+middle point sideways",
    "I": "Pinky up, others folded",
    "J": "Pinky up (like I but trace J)",
    "K": "Index up, middle angled, thumb out",
    "L": "Index up + thumb out = L shape",
    "M": "3 fingers folded over thumb",
    "N": "2 fingers folded over thumb",
    "O": "All fingers+thumb form an O",
    "P": "Like K pointing downward",
    "Q": "Like G pointing downward",
    "R": "Index+middle fingers crossed",
    "S": "Fist, thumb over fingers",
    "T": "Thumb tucked between index+middle",
    "U": "Index+middle up together",
    "V": "Index+middle up spread (peace sign)",
    "W": "Index+middle+ring up spread",
    "X": "Index finger bent/hooked",
    "Y": "Thumb+pinky out, others folded",
    "Z": "Index traces Z shape",
    "SPACE":  "Open flat hand, palm facing camera",
    "DELETE": "Fist with thumb pointing down",
}

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 200), -1)
    for a_i, b_i in HAND_CONNECTIONS:
        a, b = landmarks[a_i], landmarks[b_i]
        cv2.line(frame,
                 (int(a.x*w), int(a.y*h)),
                 (int(b.x*w), int(b.y*h)),
                 (255, 220, 0), 2)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

BOX_X, BOX_Y, BOX_W, BOX_H = 200, 70, 260, 260
total = len(LETTERS)
done  = sum(1 for l in LETTERS if existing_counts[l] >= SAMPLES)

print(f"===  Full Alphabet Collector  ===")
print(f"Letters  : A-Z + SPACE + DELETE = {total} total")
print(f"Complete : {done}/{total}")
print(f"Remaining: {total - done}\n")
print("TIP: Yellow dots must appear on hand before pressing SPACE!\n")

with HandLandmarker.create_from_options(options) as detector:
    with open(CSV_FILE, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)

        for idx, letter in enumerate(LETTERS):
            count = existing_counts[letter]
            if count >= SAMPLES:
                continue

            hint = HINTS.get(letter, "")
            print(f"\n[{idx+1}/{total}]  '{letter}'  —  {hint}")
            state = "waiting"

            while count < SAMPLES:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                h, w  = frame.shape[:2]

                mp_img = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )
                result     = detector.detect(mp_img)
                hand_found = bool(result.hand_landmarks)

                if hand_found:
                    draw_hand(frame, result.hand_landmarks[0])

                box_color = (0, 210, 0) if hand_found else (0, 165, 255)
                cv2.rectangle(frame,
                              (BOX_X, BOX_Y),
                              (BOX_X+BOX_W, BOX_Y+BOX_H),
                              box_color, 3)

                # top banner
                cv2.rectangle(frame, (0, 0), (w, 65), (20, 20, 20), -1)
                cv2.putText(frame,
                            f"[{idx+1}/{total}]  Letter: {letter}   {count}/{SAMPLES}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                status_txt = "Hand detected!" if hand_found else "No hand — move closer!"
                status_col = (0, 255, 120) if hand_found else (0, 100, 255)
                cv2.putText(frame, status_txt,
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_col, 1)

                # hint above box
                cv2.putText(frame, hint,
                            (BOX_X, BOX_Y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1)

                # bottom bar
                cv2.rectangle(frame, (0, h-60), (w, h), (20, 20, 20), -1)
                if state == "waiting":
                    cv2.putText(frame,
                                "Wait for yellow dots  |  SPACE = start collecting  |  Q = quit",
                                (8, h-35), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 220, 255), 1)
                    # overall progress bar
                    prog_w = int((done / total) * (w - 20))
                    cv2.rectangle(frame, (10, h-20), (w-10, h-8), (60,60,60), -1)
                    cv2.rectangle(frame, (10, h-20), (10+prog_w, h-8), (0,180,80), -1)
                    cv2.putText(frame, f"Overall: {done}/{total} letters done",
                                (10, h-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (150,150,150), 1)
                else:
                    bar_w = int((count / SAMPLES) * (w - 20))
                    cv2.rectangle(frame, (10, h-45), (w-10, h-25), (60,60,60), -1)
                    cv2.rectangle(frame, (10, h-45), (10+bar_w, h-25), (0, 200, 80), -1)
                    cv2.putText(frame, f"Collecting '{letter}'... hold steady  {count}/{SAMPLES}",
                                (10, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (180, 255, 180), 1)

                cv2.imshow("Data Collection — Full Alphabet", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\nStopped. Run collect_data.py again to continue from where you left off.")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

                if state == "waiting" and key == ord(' '):
                    if not hand_found:
                        print("  No hand detected! Show yellow dots on your hand first.")
                        continue
                    state = "collecting"
                    print(f"  Recording '{letter}'...")

                if state == "collecting" and hand_found:
                    lm    = result.hand_landmarks[0]
                    wrist = lm[0]
                    row   = [letter]
                    for p in lm:
                        row += [round(p.x - wrist.x, 6),
                                round(p.y - wrist.y, 6),
                                round(p.z - wrist.z, 6)]
                    writer.writerow(row)
                    csv_file.flush()
                    count += 1

            done += 1
            print(f"  '{letter}' done!  ({done}/{total} complete)")

cap.release()
cv2.destroyAllWindows()
print(f"\nAll {done} letters collected!")
print("Next:  python train.py\n")
