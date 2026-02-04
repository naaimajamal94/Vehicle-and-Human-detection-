import streamlit as st
import os

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Risk-Aware Autonomous Navigation",
    layout="centered"
)

st.title("🚦 Risk-Aware Autonomous Navigation – Demo")

st.markdown("""
This application demonstrates the **final results** of a  
**risk-aware, human-centric autonomous navigation system**.

⚠️ **Note:**  
Due to cloud resource limitations, AI inference is executed locally.  
This app presents **precomputed final outputs** for demonstration.
""")

st.divider()

# -------------------------------------------------
# PROJECT OVERVIEW
# -------------------------------------------------
st.subheader("🧠 System Overview")

st.markdown("""
**Pipeline:**

Camera → Object Detection → Distance & Speed Estimation → Risk Calculation → Decision

**Decisions:**
- 👤 Human → **STOP**
- 🚗 Vehicle → **WAIT / OVERTAKE**
""")

st.divider()

# -------------------------------------------------
# DEMO VIDEOS
# -------------------------------------------------
st.subheader("🎬 Final Output Demonstrations")

demo_path = "demo"

videos = {
    "Human Detected – STOP": "human_stop.mp4",
    "Vehicle Detected – WAIT": "vehicle_wait.mp4",
    "Mixed Scene – Risk-Based Decisions": "mixed_scene.mp4"
}

for title, file in videos.items():
    video_file = os.path.join(demo_path, file)

    st.markdown(f"### {title}")

    if os.path.exists(video_file):
        st.video(video_file)
    else:
        st.warning(f"Video `{file}` not found in demo folder.")

    st.markdown("""
**Decision Logic:**
- Distance estimated via bounding box size
- Relative speed inferred from temporal changes
- Risk normalized between 0 and 1
""")

    st.divider()

# -------------------------------------------------
# RISK EXPLANATION
# -------------------------------------------------
st.subheader("⚠️ Risk Calculation (Simplified)")

st.markdown("""
Risk is calculated as:

**Risk = 0.5 × Distance + 0.3 × Speed + 0.2 × Path Alignment**

- Distance → how close the object is  
- Speed → how fast it is approaching  
- Alignment → how central it is in the path  

This ensures **human safety is always prioritized**.
""")

st.divider()

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("""
✅ **Project Status:** Completed  
📌 **Inference:** Local execution  
🌐 **Cloud App:** Visualization & explanation
""")
