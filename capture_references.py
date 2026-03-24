"""
capture_references.py  —  Reference photo capture
Saves ONE clean hand photo per letter — NO dots, NO lines, just your hand.
Saves to: reference_signs/ folder
Does NOT affect landmarks.csv or training data.

Run:  python capture_references.py
"""

import cv2
import os
import mediapipe as mp

LETTERS    = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["SPACE", "DELETE"]
SAVE_DIR   = "reference_signs"
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
    print("Done!\n")

BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.4
)

os.makedirs(SAVE_DIR, exist_ok=True)

existing = {l for l in LETTERS
            if os.path.exists(os.path.join(SAVE_DIR, f"{l}.jpg"))}
if existing:
    print(f"Already captured: {sorted(existing)}")
    print(f"Remaining: {len(LETTERS)-len(existing)}\n")

HINTS = {
    "A":"Fist, thumb on side",
    "B":"4 fingers up, thumb folded",
    "C":"Curved hand like C",
    "D":"Index up, others curl to thumb",
    "E":"All fingers curl down",
    "F":"Index+thumb touch, others up",
    "G":"Index+thumb point sideways",
    "H":"Index+middle point sideways",
    "I":"Pinky up only",
    "J":"Pinky up (like I)",
    "K":"Index up, middle angled, thumb out",
    "L":"Index up + thumb out = L",
    "M":"3 fingers over thumb",
    "N":"2 fingers over thumb",
    "O":"All fingers form an O",
    "P":"Like K pointing down",
    "Q":"Like G pointing down",
    "R":"Index+middle crossed",
    "S":"Fist, thumb over fingers",
    "T":"Thumb between index+middle",
    "U":"Index+middle up together",
    "V":"Index+middle spread (peace sign)",
    "W":"Index+middle+ring spread up",
    "X":"Index finger bent/hooked",
    "Y":"Thumb+pinky out",
    "Z":"Index traces Z",
    "SPACE": "Open flat hand, palm facing camera",
    "DELETE":"Fist, thumb pointing down",
}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

total = len(LETTERS)
done  = len(existing)

print("=" * 45)
print("   Reference Photo Capture")
print("=" * 45)
print(f" {total} letters  |  {done} already done")
print(" SPACE = capture photo")
print(" S     = skip this letter")
print(" Q     = quit (resumes next time)")
print("=" * 45 + "\n")

with HandLandmarker.create_from_options(options) as detector:
    for idx, letter in enumerate(LETTERS):

        if letter in existing:
            print(f"  [{idx+1}/{total}] {letter} — skipping (already done)")
            continue

        hint     = HINTS.get(letter, "")
        captured = False
        print(f"\n[{idx+1}/{total}]  Sign: {letter}  —  {hint}")

        while not captured:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            # detect hand (for bounding box only — NOT drawn on frame)
            mp_img = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
            result     = detector.detect(mp_img)
            hand_found = bool(result.hand_landmarks)

            # display frame — clean, no landmarks drawn
            display = frame.copy()

            # header overlay
            cv2.rectangle(display, (0,0), (w,70), (20,20,26), -1)
            cv2.putText(display,
                        f"[{idx+1}/{total}]  Sign: {letter}   {done}/{total} done",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (82, 196, 255), 2, cv2.LINE_AA)
            cv2.putText(display, hint,
                        (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                        (160, 160, 180), 1, cv2.LINE_AA)

            # hand detected status (top right)
            status     = "Hand detected!" if hand_found else "No hand — move closer"
            status_col = (72, 210, 130)   if hand_found else (80, 80, 220)
            cv2.putText(display, status,
                        (w - 230, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        status_col, 1, cv2.LINE_AA)

            # green border when hand found
            if hand_found:
                cv2.rectangle(display, (2, 2), (w-2, h-2), (72, 210, 130), 3)

            # bottom bar
            cv2.rectangle(display, (0, h-50), (w, h), (20,20,26), -1)
            cv2.putText(display,
                        "SPACE = capture   S = skip   Q = quit",
                        (12, h-18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 200, 255), 1, cv2.LINE_AA)

            cv2.imshow("Reference Capture — ASL Signs", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nStopped. Run again to continue.")
                cap.release()
                cv2.destroyAllWindows()
                exit()

            elif key == ord('s'):
                print(f"  Skipped {letter}")
                captured = True

            elif key == ord(' '):
                if not hand_found:
                    print("  No hand detected! Move your hand closer first.")
                    continue

                # crop tightly around hand using landmark bounding box
                lms = result.hand_landmarks[0]
                xs  = [int(lm.x * w) for lm in lms]
                ys  = [int(lm.y * h) for lm in lms]
                pad = 45
                x1  = max(0, min(xs) - pad)
                y1  = max(0, min(ys) - pad)
                x2  = min(w, max(xs) + pad)
                y2  = min(h, max(ys) + pad)

                # make it square
                cw, ch   = x2-x1, y2-y1
                side     = max(cw, ch)
                cx_mid   = (x1+x2)//2
                cy_mid   = (y1+y2)//2
                x1s      = max(0, cx_mid - side//2)
                y1s      = max(0, cy_mid - side//2)
                x2s      = min(w, x1s + side)
                y2s      = min(h, y1s + side)

                # crop from CLEAN frame (no drawings)
                crop = frame[y1s:y2s, x1s:x2s]
                crop = cv2.resize(crop, (200, 200))

                save_path = os.path.join(SAVE_DIR, f"{letter}.jpg")
                cv2.imwrite(save_path, crop)
                done += 1
                existing.add(letter)
                captured = True

                # flash green confirmation
                flash = display.copy()
                cv2.rectangle(flash, (0,0), (w,h), (72,210,130), 8)
                cv2.putText(flash, f"SAVED  {letter}!",
                            (w//2 - 100, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                            (72, 210, 130), 4, cv2.LINE_AA)
                cv2.imshow("Reference Capture — ASL Signs", flash)
                cv2.waitKey(700)
                print(f"  Saved {letter}.jpg  ({done}/{total} done)")

cap.release()
cv2.destroyAllWindows()

print(f"\n{'='*45}")
print(f"  Done! {done}/{total} photos saved to {SAVE_DIR}/")
print(f"  Next: run  python app.py")
print(f"{'='*45}\n")
