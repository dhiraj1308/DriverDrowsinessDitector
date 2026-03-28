import cv2
import mediapipe as mp
import time
import math
import pygame
import os
import threading
from pathlib import Path
from twilio.rest import Client

# ---------- Twilio Setup ----------
BASE_DIR = Path(__file__).resolve().parent
LOCATION_FILE = BASE_DIR / "location.txt"

account_sid = os.getenv("TWILIO_ACCOUNT_SID", "AC9766a2effa1ad46db3eec8b671aed85d")
auth_token = os.getenv("TWILIO_AUTH_TOKEN", "874ae30fd32d189f2dcaceb7196a05ed")
twilio_from = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
whatsapp_to = os.getenv("ALERT_WHATSAPP_TO", "whatsapp:+918331976094")

client = Client(account_sid, auth_token)

def send_whatsapp_alert():
    try:
        with open(LOCATION_FILE, "r", encoding="utf-8") as f:
            lat, lng = f.read().strip().split(",")

        location_link = f"https://maps.google.com/?q={lat},{lng}"

        message = client.messages.create(
            body=f"⚠️ Driver drowsiness detected!\nLocation: {location_link}",
            from_=twilio_from,
            to=whatsapp_to
        )

        print("Message sent:", message.sid)

    except Exception as e:
        print("WhatsApp error:", e)
        print("Location file path:", LOCATION_FILE)
        print("WhatsApp sender:", twilio_from)
        print("WhatsApp recipient:", whatsapp_to)

def send_whatsapp_alert_async():
    threading.Thread(target=send_whatsapp_alert, daemon=True).start()

# ---------- Alarm Setup ----------
pygame.mixer.init()
pygame.mixer.music.load(str(BASE_DIR / "alarm.wav"))

# ---------- MediaPipe Setup ----------
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=2
)

cap = cv2.VideoCapture(0)

# ---------- Utility Functions ----------
def dist(a, b):
    return math.dist(a, b)

def eye_aspect_ratio(eye, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in eye]
    return (dist(pts[1], pts[5]) + dist(pts[2], pts[4])) / (2 * dist(pts[0], pts[3]))

def mouth_aspect_ratio(mouth, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in mouth]
    return dist(pts[2], pts[6]) / dist(pts[0], pts[4])

def head_down_ratio(face):
    # Ratio of nose position between eyes and mouth. Higher value indicates nodding down.
    left_eye = face[33]
    right_eye = face[263]
    nose_tip = face[1]
    upper_lip = face[13]

    eye_center_y = (left_eye.y + right_eye.y) / 2.0
    eye_to_mouth = max(1e-6, (upper_lip.y - eye_center_y))
    return (nose_tip.y - eye_center_y) / eye_to_mouth

def head_tilt_degrees(face):
    # Eye line angle from horizontal. Large absolute angle indicates side tilt.
    left_eye = face[33]
    right_eye = face[263]
    dy = right_eye.y - left_eye.y
    dx = right_eye.x - left_eye.x
    return abs(math.degrees(math.atan2(dy, dx)))

# ---------- Thresholds ----------
EAR_THR = 0.22
MAR_THR = 0.6
HEAD_DOWN_THR = 0.70
HEAD_TILT_DEG_THR = 18
HEAD_BEND_SCORE_STEP = 2
DROWSY_SCORE_LIMIT = 8
ALERT_COOLDOWN_SEC = 120
NO_FACE_DECAY = 2

# ---------- Variables ----------
score = 0
captured = False
blink_count = 0
eye_closed = False
yawn_count = 0
mouth_open = False

drowsy_start_time = None
first_drowsy_alert_sent = False
last_whatsapp_alert_time = 0.0

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = int(time.time() * 1000)

        results = landmarker.detect_for_video(mp_image, timestamp)
        face_detected = bool(results.face_landmarks)
        event_drowsy = False

        status = "ALERT"
        color = (0, 255, 0)

        if face_detected:
            face = results.face_landmarks[0]

            # ---------- Face Landmarks ----------
            right_eye = [face[i] for i in [33, 160, 158, 133, 153, 144]]
            left_eye = [face[i] for i in [362, 385, 387, 263, 373, 380]]
            mouth = [face[i] for i in [61, 81, 13, 311, 291, 308, 14]]

            # ---------- EAR / MAR ----------
            right_ear = eye_aspect_ratio(right_eye, w, h)
            left_ear = eye_aspect_ratio(left_eye, w, h)
            ear = (right_ear + left_ear) / 2
            mar = mouth_aspect_ratio(mouth, w, h)
            hdr = head_down_ratio(face)
            tilt_deg = head_tilt_degrees(face)

            # ---------- Blink Counter ----------
            if ear < EAR_THR:
                if not eye_closed:
                    blink_count += 1
                    eye_closed = True
            else:
                eye_closed = False

            # ---------- Yawn Counter ----------
            if mar > MAR_THR:
                if not mouth_open:
                    yawn_count += 1
                    mouth_open = True
            else:
                mouth_open = False

            # ---------- Drowsiness Logic ----------
            head_bent = (hdr > HEAD_DOWN_THR) or (tilt_deg > HEAD_TILT_DEG_THR)
            eyes_closed_now = ear < EAR_THR
            yawning_now = mar > MAR_THR
            event_drowsy = head_bent or eyes_closed_now or yawning_now

            if head_bent:
                score += HEAD_BEND_SCORE_STEP
            elif eyes_closed_now or yawning_now:
                score += 1
            else:
                # Normal behavior detected: reset score immediately to 0
                score = 0
                captured = False
                drowsy_start_time = None

            # Send the first WhatsApp alert immediately at first drowsy signal.
            if event_drowsy:
                now = time.time()
                if not first_drowsy_alert_sent:
                    first_drowsy_alert_sent = True
                    last_whatsapp_alert_time = now
                    send_whatsapp_alert_async()
                elif score >= DROWSY_SCORE_LIMIT and (now - last_whatsapp_alert_time) >= ALERT_COOLDOWN_SEC:
                    last_whatsapp_alert_time = now
                    send_whatsapp_alert_async()

            # ---------- Drowsy Alert ----------
            if score >= DROWSY_SCORE_LIMIT:
                status = "DROWSY – TAKE A BREAK"
                color = (0, 0, 255)

                # ---------- Screenshot ----------
                

                # ---------- Timer ----------
                if drowsy_start_time is None:
                    drowsy_start_time = time.time()

            # ---------- Display EAR ----------
            cv2.putText(frame, f"EAR: {ear:.2f}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # ---------- Display MAR ----------
            cv2.putText(frame, f"MAR: {mar:.2f}", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.putText(frame, f"HEAD DOWN: {hdr:.2f}", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.putText(frame, f"HEAD TILT: {tilt_deg:.1f}", (20, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        else:
            # ---------- Face Missing ----------
            status = "LOOK FORWARD"
            color = (0, 165, 255)
            cv2.putText(frame, "LOOK FORWARD", (20, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            # Do not treat mirror glance / temporary face loss as drowsiness.
            score = max(0, score - NO_FACE_DECAY)
            captured = False
            drowsy_start_time = None

        # ---------- Alarm Control ----------
        # Alarm ON immediately on head bend / eye close / yawn.
        alarm_needed = event_drowsy

        if alarm_needed:
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play(-1)
        else:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()

        # ---------- Status ----------
        cv2.putText(frame, f"STATUS: {status}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # ---------- Fatigue Score ----------
        score_color = (0,255,0)
        if score > 5:
            score_color = (0,255,255)
        if score > 10:
            score_color = (0,0,255)

        cv2.putText(frame, f"FATIGUE SCORE: {score}", (20, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)

        # ---------- Blink Count ----------
        cv2.putText(frame, f"BLINKS: {blink_count}", (20, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # ---------- Yawn Count ----------
        cv2.putText(frame, f"YAWNS: {yawn_count}", (20, 290),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.putText(frame, "Press T for test WhatsApp | Q to quit", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

        cv2.imshow("Driver Drowsiness Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('t'):
            print("Manual WhatsApp test triggered.")
            send_whatsapp_alert_async()
        if key == ord('q'):
            break

# ---------- Cleanup ----------
cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()