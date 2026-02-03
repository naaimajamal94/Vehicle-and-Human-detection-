import streamlit as st
import tempfile, os
import cv2, time
import imageio
from ultralytics import YOLO

st.set_page_config(page_title="Risk-Aware Navigation", layout="centered")

st.title("🚦 Risk-Aware Autonomous Navigation Demo")

# ---- CONFIG ----
MODEL_PATH = "person_vehicle_model.pt"  # user provides locally
CONF_THRESH = 0.3

# ---- LOAD MODEL ----
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

uploaded_video = st.file_uploader("Upload a video", type=["mp4"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = []

    prev_state = {}

    st.info("Processing video…")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        h, w, _ = frame.shape

        results = model(frame, conf=CONF_THRESH, verbose=False)
        action = "FORWARD"

        for i, box in enumerate(results[0].boxes):
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) / 2
            box_h = y2 - y1

            # Distance
            rel_dist = min((box_h / h) / 0.6, 1.0)

            # Speed
            speed = 0
            now = time.time()
            if i in prev_state:
                ph, pt = prev_state[i]
                speed = max((box_h - ph) / (now - pt + 1e-6), 0)
            prev_state[i] = (box_h, now)
            speed = min(speed / 200, 1.0)

            # Centrality
            central = 1 - abs(cx - w/2) / (w/2)

            # Risk (0–1)
            risk = 0.5*rel_dist + 0.3*speed + 0.2*central

            label = "PERSON" if cls_id == 0 else "VEHICLE"
            color = (0,255,0) if cls_id == 0 else (255,0,0)
            cv2.rectangle(frame, (x1,y1),(x2,y2),color,2)
            cv2.putText(frame, f"{label} | {risk:.2f}", (x1,y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if cls_id == 0 and risk > 0.6:
                action = "STOP"
            elif cls_id == 1 and risk > 0.8:
                action = "WAIT"

        cv2.putText(frame, f"ACTION: {action}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    out_path = "output.mp4"
    imageio.mimsave(out_path, frames, fps=fps)

    st.success("Done")
    st.video(out_path)
