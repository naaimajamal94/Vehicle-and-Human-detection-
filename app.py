import streamlit as st
import tempfile, os, time
import cv2
import imageio
from ultralytics import YOLO

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Risk-Aware Autonomous Navigation",
    layout="centered"
)

st.title("🚦 Risk-Aware Autonomous Navigation App")

st.markdown("""
This app demonstrates **AI-automated risk-aware decision making**.

- 👤 Human → **STOP**
- 🚗 Vehicle → **WAIT / OVERTAKE**
- Decisions use **distance + relative speed + risk**
""")

st.warning("⏳ Video processing may take some time depending on video length.")

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MODEL_PATH = "person_vehicle_model.pt"   # must be present in repo
CONF_THRESH = 0.3
MAX_APPROACH_SPEED = 200
DIST_CLOSE = 0.6

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# -------------------------------------------------
# VIDEO UPLOAD
# -------------------------------------------------
uploaded_video = st.file_uploader(
    "Upload a video file",
    type=["mp4"]
)

if uploaded_video:

    # Save uploaded video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 25

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    prev_state = {}
    processed = 0

    progress = st.progress(0)

    # -------------------------------------------------
    # PROCESS VIDEO
    # -------------------------------------------------
    with st.spinner("Processing video… please wait ⏳"):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            h, w, _ = frame.shape

            results = model(frame, conf=CONF_THRESH, verbose=False)
            action = "FORWARD"

            for i, box in enumerate(results[0].boxes):
                cls_id = int(box.cls[0])  # 0=person, 1=vehicle
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) / 2
                box_h = y2 - y1

                # ---------------- Distance (0–1)
                rel_dist = min((box_h / h) / DIST_CLOSE, 1.0)

                # ---------------- Relative Speed (0–1)
                now = time.time()
                speed = 0
                if i in prev_state:
                    ph, pt = prev_state[i]
                    speed = max((box_h - ph) / (now - pt + 1e-6), 0)
                prev_state[i] = (box_h, now)
                speed = min(speed / MAX_APPROACH_SPEED, 1.0)

                # ---------------- Centrality (0–1)
                central = 1 - abs(cx - w / 2) / (w / 2)

                # ---------------- Risk (0–1)
                risk = 0.5 * rel_dist + 0.3 * speed + 0.2 * central

                label = "PERSON" if cls_id == 0 else "VEHICLE"
                color = (0, 255, 0) if cls_id == 0 else (255, 0, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{label} | RISK {risk:.2f}",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

                # ---------------- Decisions
                if cls_id == 0 and risk >= 0.6:
                    action = "STOP"
                elif cls_id == 1 and risk >= 0.8:
                    action = "WAIT"

            cv2.putText(
                frame,
                f"ACTION: {action}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            processed += 1
            if total_frames > 0:
                progress.progress(min(processed / total_frames, 1.0))

    cap.release()

    # -------------------------------------------------
    # SAFE VIDEO SAVE (CRASH-PROOF)
    # -------------------------------------------------
    if len(frames) == 0:
        st.error("❌ No frames were processed. Please upload a valid video.")
        st.stop()

    out_path = "output.mp4"
    safe_fps = int(fps) if fps > 1 else 25

    imageio.mimsave(
        out_path,
        frames,
        fps=safe_fps,
        format="FFMPEG"
    )

    st.success("✅ Processing complete")
    st.video(out_path)
