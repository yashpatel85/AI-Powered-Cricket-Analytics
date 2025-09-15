import streamlit as st
import tempfile
from cover_drive_analysis_realtime import analyze_video

st.set_page_config(page_title="Cricket Cover Drive Analyzer", layout="wide")
st.title("AthleteRise: Real-Time Cricket Cover Drive Analysis")

uploaded_file = st.file_uploader("Upload a cricket video", type=["mp4", "mov", "avi"])

if uploaded_file:
    with st.spinner("Processing video..."):
        # Save uploaded video to temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        input_path = tfile.name

        # Process video
        output_path = analyze_video(input_path)

        # Display processed video
        st.video(output_path)

        # Download button
        with open(output_path, "rb") as f:
            st.download_button(
                label="Download Processed Video",
                data=f,
                file_name="annotated_video.mp4",
                mime="video/mp4"
            )
