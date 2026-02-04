import streamlit as st
import tempfile, time
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

st.title("🚦 Risk-Aware Autonomous Navigation")

st.markdown("""
**Automated AI decision system**

- 👤 Human → STOP  
- 🚗 Vehicle → WAIT  
- Decisions based on **distance + speed + risk**
""")

st.warning("⏳ Video processing may take time depending on length.")

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MODEL_PATH = "person_vehicle_model.pt"   # must be in repo
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
video = st.file_uploader("Upload a video", type=["mp4"])

if video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video.read())

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 25

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    prev_state = {}
    progress = st.progress(0)
    processed = 0

    with st.spinner("Processing video…"):
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

                # Distance (0–1)
                D = min((box_h / h) / DIST_CLOSE, 1.0)

                # Speed (0–1)
                now = time.time()
                speed = 0
                if i in prev_state:
                    ph, pt = prev_state[i]
                    speed = max((box_h - ph) / (now - pt + 1e-6), 0)
                prev_state[i] = (box_h, now)
                V = min(speed / MAX_APPROACH_SPEED, 1.0)

                # Centrality (0–1)
                P = 1 - abs(cx - w/2) / (w/2)

                # Risk (0–1)
                risk = 0.5 * D + 0.3 * V + 0.2 * P

                label = "PERSON" if cls_id == 0 else "VEHICLE"
                color = (0,255,0) if cls_id == 0 else (255,0,0)

                cv2.rectangle(frame, (x1,y1),(x2,y2),color,2)
                cv2.putText(
                    frame,
                    f"{label} | RISK {risk:.2f}",
                    (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

                if cls_id == 0 and risk >= 0.6:
                    action = "STOP"
                elif cls_id == 1 and risk >= 0.8:
                    action = "WAIT"

            cv2.putText(
                frame,
                f"ACTION: {action}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                3
            )

            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            processed += 1
            if total_frames > 0:
                progress.progress(min(processed / total_frames, 1.0))

    cap.release()

    if len(frames) == 0:
        st.error("❌ No frames processed.")
        st.stop()

    imageio.mimsave(
        "output.mp4",
        frames,
        fps=int(fps),
        format="FFMPEG"
    )

    st.success("✅ Done")
    st.video("output.mp4")
