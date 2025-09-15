import streamlit as st
import tempfile
import os
import time
from cover_drive_analysis_realtime import analyze_video

st.set_page_config(page_title="Cricket Cover Drive Analyzer", layout="wide")
st.title("AthleteRise: Real-Time Cricket Cover Drive Analysis")

# --- Video uploader ---
uploaded_file = st.file_uploader("Upload a cricket video (MP4)", type=["mp4"])

if uploaded_file:
    # Save uploaded file to a temporary path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
        tmp_input.write(uploaded_file.read())
        input_path = tmp_input.name

    # Prepare output paths
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, "annotated_video.mp4")
    output_eval_path = os.path.join(output_dir, "evaluation.json")

    # --- Process video with spinner ---
    with st.spinner("Processing video... This may take a while depending on length."):
        analyze_video(input_path, output_video_path)

        # Wait until the output video is fully written
        timeout = 120  # seconds
        start_time = time.time()
        while not os.path.exists(output_video_path):
            if time.time() - start_time > timeout:
                st.error("Processing timed out. Please try again.")
                break
            time.sleep(0.5)

    if os.path.exists(output_video_path):
        st.success("✅ Processing complete!")

        # --- Display video ---
        st.video(output_video_path)

        # --- Download buttons ---
        with open(output_video_path, "rb") as f:
            video_bytes = f.read()
        st.download_button(
            label="Download Annotated Video",
            data=video_bytes,
            file_name="annotated_video.mp4",
            mime="video/mp4"
        )

        if os.path.exists(output_eval_path):
            with open(output_eval_path, "rb") as f:
                json_bytes = f.read()
            st.download_button(
                label="Download Evaluation JSON",
                data=json_bytes,
                file_name="evaluation.json",
                mime="application/json"
            )
        else:
            st.warning("Evaluation JSON not found.")
