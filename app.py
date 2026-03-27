"""
app.py  —  Black & White theme
- Full color photos (no greyscale)
- White buttons with black text
- Clean bold title
Works with MediaPipe 0.10.13+
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import os
import math
import threading

# ── Text-to-Speech (Windows SAPI via pyttsx3) ────────────────────────────────
try:
    import pyttsx3
    _tts = pyttsx3.init()
    _tts.setProperty("rate", 160)
    _tts_ok = True
except Exception as _e:
    print(f"TTS unavailable: {_e}\nRun:  pip install pyttsx3")
    _tts_ok = False

_tts_lock = threading.Lock()

def speak(text):
    """Speak text in a background thread — UI never blocks."""
    if not text or not text.strip():
        return
    def _run():
        with _tts_lock:
            try:
                if _tts_ok:
                    _tts.say(text)
                    _tts.runAndWait()
            except Exception as e:
                print(f"TTS error: {e}")
    threading.Thread(target=_run, daemon=True).start()

MODEL_FILE  = "model.pkl"
MODEL_PATH  = "hand_landmarker.task"
REF_DIR     = "reference_signs"
CONF_THRESH = 0.45
HOLD_SEC    = 0.8
BUFFER      = 7

# ── Load classifier ───────────────────────────────────────────────────────────
if not os.path.exists(MODEL_FILE):
    print(f"ERROR: {MODEL_FILE} not found. Run  python train.py  first.")
    exit()

with open(MODEL_FILE, "rb") as f:
    saved = pickle.load(f)
model = saved["model"]
le    = saved["label_encoder"]

# ── Download landmark model if needed ────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmark model (~9 MB) ...")
    import urllib.request
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )
    print("Done!\n")

# ── MediaPipe ─────────────────────────────────────────────────────────────────
BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5
)

# ── Colours ───────────────────────────────────────────────────────────────────
C_BG        = (18,  18,  18)
C_PANEL     = (24,  24,  24)
C_CARD      = (34,  34,  34)
C_CARD_HI   = (60,  60,  60)
C_WHITE     = (255, 255, 255)
C_LIGHT     = (200, 200, 200)
C_MUTED     = ( 90,  90,  90)
C_BORDER    = ( 65,  65,  65)
C_BLACK     = (  0,   0,   0)
C_BTN_WHITE = (240, 240, 240)

# ── Layout ────────────────────────────────────────────────────────────────────
PANEL_W      = 420
CAM_W        = 520
CAM_H        = 390
WIN_W        = PANEL_W + CAM_W
WIN_H        = 780

CAM_X        = PANEL_W
CAM_Y        = 0

BOX_X        = CAM_X + 140
BOX_Y        = 70
BOX_W        = 200
BOX_H        = 200

OUT_Y        = CAM_H + 10
OUT_H        = WIN_H - CAM_H - 80
OUT_X        = CAM_X + 10
OUT_W        = CAM_W - 20

BTN_H  = 44
BTN_W  = 108
GAP    = 10
BTN_Y  = WIN_H - 60
BTN_SX = CAM_X + (CAM_W - (4*BTN_W + 3*GAP)) // 2

BTN_CLEAR = (BTN_SX,               BTN_Y, BTN_W, BTN_H)
BTN_SAVE  = (BTN_SX+BTN_W+GAP,     BTN_Y, BTN_W, BTN_H)
BTN_SPEAK = (BTN_SX+2*(BTN_W+GAP), BTN_Y, BTN_W, BTN_H)
BTN_QUIT  = (BTN_SX+3*(BTN_W+GAP), BTN_Y, BTN_W, BTN_H)

# ── Reference panel layout ────────────────────────────────────────────────────
ALL_KEYS  = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["DELETE", "SPACE"]
DISP_LBL  = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["Del",    "Spc"]
GCOLS     = 4
CELL_W    = PANEL_W // GCOLS
CELL_H    = 92
PHOTO_SZ  = 66
ROWS      = math.ceil(len(ALL_KEYS) / GCOLS)

# ── Load reference photos ─────────────────────────────────────────────────────
ref_photos = {}
for key in ALL_KEYS:
    path = os.path.join(REF_DIR, f"{key}.jpg")
    if os.path.exists(path):
        img = cv2.imread(path)
        if img is not None:
            ref_photos[key] = cv2.resize(img, (PHOTO_SZ, PHOTO_SZ))

print(f"Loaded {len(ref_photos)}/28 reference photos from '{REF_DIR}/'")

# ── Build reference panel ─────────────────────────────────────────────────────
def make_panel(highlight=None):
    ph    = max(WIN_H, 52 + ROWS*CELL_H + 20)
    panel = np.full((ph, PANEL_W, 3), C_PANEL, dtype=np.uint8)

    # header
    cv2.rectangle(panel, (0,0), (PANEL_W,46), C_BG, -1)
    cv2.putText(panel, "ASL Hand Signs",
                (14, 31), cv2.FONT_HERSHEY_DUPLEX, 0.72, C_WHITE, 1, cv2.LINE_AA)
    cv2.line(panel, (0,46), (PANEL_W,46), C_BORDER, 1)

    known = set(le.classes_)

    for idx, key in enumerate(ALL_KEYS):
        row    = idx // GCOLS
        col    = idx % GCOLS
        cx     = col * CELL_W
        cy     = 52 + row * CELL_H
        lbl    = DISP_LBL[idx]
        is_hi  = (key == highlight)
        trained = key in known

        # cell background
        bg = C_CARD_HI if is_hi else (C_CARD if trained else (28,28,28))
        cv2.rectangle(panel, (cx+1,cy+1), (cx+CELL_W-1,cy+CELL_H-1), bg, -1)

        # cell border — bright white when active
        bd_col = C_WHITE  if is_hi else (C_BORDER if trained else (40,40,40))
        bd_w   = 2        if is_hi else 1
        cv2.rectangle(panel, (cx+1,cy+1), (cx+CELL_W-1,cy+CELL_H-1), bd_col, bd_w)

        # photo — always FULL COLOR
        photo_x = cx + (CELL_W - PHOTO_SZ) // 2
        photo_y = cy + 5

        if key in ref_photos:
            thumb = ref_photos[key].copy()   # full color, no greyscale
            panel[photo_y:photo_y+PHOTO_SZ, photo_x:photo_x+PHOTO_SZ] = thumb
            # photo border
            pb_col = C_WHITE if is_hi else C_BORDER
            cv2.rectangle(panel,
                          (photo_x,   photo_y),
                          (photo_x+PHOTO_SZ, photo_y+PHOTO_SZ),
                          pb_col, 1)
        else:
            cv2.rectangle(panel, (photo_x,photo_y),
                          (photo_x+PHOTO_SZ, photo_y+PHOTO_SZ), (40,40,40), -1)
            cv2.putText(panel, "?",
                        (photo_x+PHOTO_SZ//2-8, photo_y+PHOTO_SZ//2+8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_MUTED, 2)

        # letter label
        lbl_col = C_WHITE if is_hi else (C_LIGHT if trained else C_MUTED)
        fs      = 0.40 if len(lbl)>1 else 0.46
        tw      = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, fs, 1)[0][0]
        cv2.putText(panel, lbl,
                    (cx+(CELL_W-tw)//2, cy+CELL_H-5),
                    cv2.FONT_HERSHEY_DUPLEX, fs, lbl_col, 1, cv2.LINE_AA)

    cv2.line(panel, (PANEL_W-1,0), (PANEL_W-1,ph), C_BORDER, 1)
    return panel

# ── Helpers ───────────────────────────────────────────────────────────────────
HAND_CONN = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

def draw_live_hand(frame, landmarks):
    h, w = frame.shape[:2]
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 5, C_BLACK, -1, cv2.LINE_AA)
    for a, b in HAND_CONN:
        pa, pb = landmarks[a], landmarks[b]
        cv2.line(frame,
                 (int(pa.x*w), int(pa.y*h)),
                 (int(pb.x*w), int(pb.y*h)),
                 C_LIGHT, 2, cv2.LINE_AA)

def extract_features(landmarks):
    wrist = landmarks[0]
    feats = []
    for p in landmarks:
        feats += [p.x-wrist.x, p.y-wrist.y, p.z-wrist.z]
    return feats

def rounded_rect(img, x, y, w, h, r, color, t=-1):
    if w < 2*r or h < 2*r:
        r = max(1, min(w,h)//2)
    cv2.rectangle(img, (x+r,y),   (x+w-r,y+h),   color, t)
    cv2.rectangle(img, (x,  y+r), (x+w,  y+h-r), color, t)
    for cx2,cy2 in [(x+r,y+r),(x+w-r,y+r),(x+r,y+h-r),(x+w-r,y+h-r)]:
        cv2.circle(img, (cx2,cy2), r, color, t)

def draw_btn(canvas, rect, label, bg, text_col):
    x, y, w, h = rect
    # white fill
    rounded_rect(canvas, x, y, w, h, 8, bg)
    # subtle dark border
    cv2.rectangle(canvas, (x,y), (x+w,y+h), (180,180,180), 1)
    tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)[0]
    cv2.putText(canvas, label,
                (x+(w-tw)//2, y+(h+th)//2-2),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, text_col, 1, cv2.LINE_AA)

def in_rect(px, py, rect):
    x, y, w, h = rect
    return x<=px<=x+w and y<=py<=y+h

def bar(canvas, x, y, w, h, pct, fg, bg=(45,45,45)):
    rounded_rect(canvas, x, y, w, h, h//2, bg)
    fw = int(w * pct)
    if fw > h:
        rounded_rect(canvas, x, y, fw, h, h//2, fg)

# ── State ─────────────────────────────────────────────────────────────────────
phrase         = ""
frame_buffer   = []
last_stable    = ""
lock_start     = 0.0
locked         = False
last_confirmed = 0.0
mouse_click    = None
last_hi        = None
ref_panel      = make_panel(None)
flash_start    = 0.0   # time of last confirmed letter (for snap flash)

cv2.namedWindow("Sign Language to Text", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Sign Language to Text", WIN_W, WIN_H)

def on_mouse(e, x, y, f, p):
    global mouse_click
    if e == cv2.EVENT_LBUTTONDOWN:
        mouse_click = (x, y)

cv2.setMouseCallback("Sign Language to Text", on_mouse)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

print("App running!  Hold a sign in the box for 0.8s to confirm.")
# print("SPACE = space  |  BACKSPACE = delete  |  Q = quit\n")

with HandLandmarker.create_from_options(options) as detector:
    running = True
    while running:
        ret, raw = cap.read()
        if not ret:
            break

        raw = cv2.flip(raw, 1)
        raw = cv2.resize(raw, (CAM_W, CAM_H))

        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        )
        result = detector.detect(mp_img)

        predicted_now = ""
        conf_now      = 0.0

        if result.hand_landmarks:
            draw_live_hand(raw, result.hand_landmarks[0])
            feats    = extract_features(result.hand_landmarks[0])
            proba    = model.predict_proba([feats])[0]
            conf_now = float(max(proba))
            if conf_now >= CONF_THRESH:
                predicted_now = le.classes_[int(np.argmax(proba))]

        frame_buffer.append(predicted_now)
        if len(frame_buffer) > BUFFER:
            frame_buffer.pop(0)

        stable = ""
        if frame_buffer.count(predicted_now) >= int(BUFFER*0.8) and predicted_now:
            stable = predicted_now

        now = time.time()
        if stable and stable == last_stable:
            if not locked and (now-lock_start) >= HOLD_SEC:
                if now-last_confirmed > HOLD_SEC:
                    if stable == "SPACE":    phrase += " "
                    elif stable == "DELETE": phrase = phrase[:-1]
                    else:                    phrase += stable
                    last_confirmed = now
                    locked         = True
                    flash_start    = now   # trigger snap flash
        elif stable != last_stable:
            last_stable = stable
            lock_start  = now
            locked      = False

        hold_pct = 0.0
        if stable and not locked:
            hold_pct = min(1.0, (now-lock_start)/HOLD_SEC)

        if stable != last_hi:
            ref_panel = make_panel(stable if stable else None)
            last_hi   = stable

        # ── Canvas ────────────────────────────────────────────────────────
        canvas = np.full((WIN_H, WIN_W, 3), C_BG, dtype=np.uint8)
        canvas[CAM_Y:CAM_Y+CAM_H, CAM_X:CAM_X+CAM_W] = raw

        # dark gradient bottom of camera
        for i in range(80):
            a  = (i/80) * 0.75
            yp = CAM_H-80+i
            if 0 <= yp < CAM_H:
                canvas[yp, CAM_X:CAM_X+CAM_W] = (
                    canvas[yp, CAM_X:CAM_X+CAM_W]*(1-a)
                ).astype(np.uint8)

        # bounding box + corner accents
        bc = C_WHITE if locked else (C_LIGHT if stable else C_MUTED)
        cv2.rectangle(canvas,(BOX_X,BOX_Y),(BOX_X+BOX_W,BOX_Y+BOX_H),bc,2,cv2.LINE_AA)
        ca = 18
        for sx,sy,dx,dy in [(BOX_X,BOX_Y,1,1),(BOX_X+BOX_W,BOX_Y,-1,1),
                             (BOX_X,BOX_Y+BOX_H,1,-1),(BOX_X+BOX_W,BOX_Y+BOX_H,-1,-1)]:
            cv2.line(canvas,(sx,sy),(sx+dx*ca,sy),bc,3,cv2.LINE_AA)
            cv2.line(canvas,(sx,sy),(sx,sy+dy*ca),bc,3,cv2.LINE_AA)

        # ── Title — bold white, clean font ───────────────────────────────
        title = "SIGN LANGUAGE"
        subtitle = " TO TEXT"

        x = CAM_X + 12
        y = 40 # Single baseline

        # main title 
        cv2.putText(canvas, title,
                    (x,y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, C_BLACK, 2, cv2.LINE_AA)
        
        # get width of main title
        (title_w, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)

        # subtitle on SAME line
        cv2.putText(canvas, subtitle,
                    (x + title_w + 6, y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, C_LIGHT, 1, cv2.LINE_AA)
        # Underline
        cv2.line(canvas,
                (x, y + 10),
                (x + title_w + 120, y + 10),
                C_BORDER, 1)
    


        # big predicted letter
        if stable:
            disp = stable if len(stable)==1 else stable[:3]
            col  = C_WHITE if locked else C_LIGHT
            fs   = 5.0 if len(disp)==1 else 2.0
            tw   = cv2.getTextSize(disp, cv2.FONT_HERSHEY_DUPLEX, fs, 4)[0][0]
            px   = BOX_X+BOX_W+20
            py   = BOX_Y+BOX_H//2+int(fs*20)
            if px+tw < CAM_X+CAM_W-10:
                cv2.putText(canvas, disp, (px,py),
                            cv2.FONT_HERSHEY_DUPLEX, fs, col, 4, cv2.LINE_AA)

        # hold bar — neon cyan glow
        bx, by = CAM_X+12, CAM_H-38
        if stable and not locked:
            cv2.putText(canvas, "Hold to confirm",
                        (bx, by-4), cv2.FONT_HERSHEY_SIMPLEX, 0.33, C_WHITE, 1, cv2.LINE_AA)
            rounded_rect(canvas, bx, by, 180, 8, 4, (45,45,45))
            fw = int(180 * hold_pct)
            if fw > 4:
                # glow pass (wider, dimmer)
                glow = canvas.copy()
                rounded_rect(glow, bx, by-2, fw, 12, 6, (0, 180, 60))
                cv2.addWeighted(glow, 0.30, canvas, 0.70, 0, canvas)
                # main neon green bar
                rounded_rect(canvas, bx, by, fw, 8, 4, (0, 255, 80))

        # output card
        rounded_rect(canvas, OUT_X, OUT_Y, OUT_W, OUT_H, 10, C_CARD)
        cv2.putText(canvas, "OUTPUT",
                    (OUT_X+16, OUT_Y+22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_MUTED, 1, cv2.LINE_AA)
        cv2.line(canvas, (OUT_X+16,OUT_Y+28), (OUT_X+OUT_W-16,OUT_Y+28), C_BORDER, 1)

        lines = [phrase[i:i+24] for i in range(0, max(1,len(phrase)), 24)]
        if not lines: lines = [""]
        for li, line in enumerate(lines[:4]):
            txt = line if line.strip() else ("Start signing..." if li==0 else "")
            col = C_WHITE if line.strip() else C_MUTED
            cv2.putText(canvas, txt,
                        (OUT_X+16, OUT_Y+56+li*44),
                        cv2.FONT_HERSHEY_DUPLEX, 0.82, col, 1, cv2.LINE_AA)

        # blinking cursor
        if int(time.time()*2)%2==0:
            ll  = lines[-1] if lines else ""
            li  = min(len(lines)-1, 3)
            tw  = cv2.getTextSize(ll, cv2.FONT_HERSHEY_DUPLEX, 0.82, 1)[0][0]
            cv2.line(canvas,
                     (OUT_X+16+tw+4, OUT_Y+56+li*44-24),
                     (OUT_X+16+tw+4, OUT_Y+56+li*44+4),
                     C_LIGHT, 2)

        # ── Buttons — white fill, black text ──────────────────────────────
        draw_btn(canvas, BTN_CLEAR, "Clear All",    C_BTN_WHITE,    C_BLACK)
        draw_btn(canvas, BTN_SAVE,  "Save in File", C_BTN_WHITE,    C_BLACK)
        draw_btn(canvas, BTN_SPEAK, "Speak [T]",   (180, 220, 180), C_BLACK)
        draw_btn(canvas, BTN_QUIT,  "Quit (Q)",     C_BTN_WHITE,    C_BLACK)

        # cv2.putText(canvas,
        #            "SPACE = space     BKSP = delete     Q = quit",
        #            (CAM_X+12, WIN_H-8),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_MUTED, 1, cv2.LINE_AA)

        # reference panel LEFT
        rph = min(WIN_H, ref_panel.shape[0])
        canvas[:rph, 0:PANEL_W] = ref_panel[:rph]
        cv2.line(canvas, (PANEL_W-1,0), (PANEL_W-1,WIN_H), C_BORDER, 1)

        # ── Camera snap flash ──────────────────────────────────────────────
        FLASH_DUR = 0.18   # seconds
        flash_age = now - flash_start
        if flash_age < FLASH_DUR:
            alpha = 1.0 - (flash_age / FLASH_DUR)   # fades out
            brightness = int(255 * alpha)
            flash_overlay = np.full((CAM_H, CAM_W, 3), brightness, dtype=np.uint8)
            canvas[CAM_Y:CAM_Y+CAM_H, CAM_X:CAM_X+CAM_W] = cv2.addWeighted(
                canvas[CAM_Y:CAM_Y+CAM_H, CAM_X:CAM_X+CAM_W], 1 - alpha,
                flash_overlay, alpha, 0
            )

        cv2.imshow("Sign Language to Text", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            running = False
        elif key == ord('t') or key == ord('T'):
            speak(phrase if phrase.strip() else "Nothing typed yet")
        elif key == ord(' '):
            phrase += ' '
            last_confirmed = time.time()
        elif key in (8, 127):
            phrase = phrase[:-1]

        mc = mouse_click
        mouse_click = None
        if mc:
            mx, my = mc
            if in_rect(mx, my, BTN_CLEAR):
                phrase = ""
            elif in_rect(mx, my, BTN_SAVE):
                with open("sign_output.txt", "w") as f:
                    f.write(phrase)
                print(f"Saved: '{phrase}'")
            elif in_rect(mx, my, BTN_SPEAK):
                speak(phrase if phrase.strip() else "Nothing typed yet")
            elif in_rect(mx, my, BTN_QUIT):
                running = False

cap.release()
cv2.destroyAllWindows()
print(f"\nFinal text: {phrase}\n")