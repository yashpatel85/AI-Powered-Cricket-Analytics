AthleteRise – AI-Powered Cricket Analytics

Real-Time Cover Drive Analysis for cricket using Python, OpenCV, and MediaPipe.
Processes full-length videos, estimates pose, computes biomechanical metrics, and outputs an annotated video and shot evaluation.

🚀 Features

Real-Time Pose Estimation: Head, shoulders, elbows, wrists, hips, knees, ankles

Biomechanical Metrics:

Front elbow angle

Spine lean

Head-over-knee alignment

Front foot direction

Live Overlays on Video: Metrics and color-coded feedback cues

Phase Detection: Stance → Stride → Downswing → Impact → Follow-through → Recovery

Final Evaluation: Multi-category scores and actionable feedback

Streamlit App: Upload any video → view annotated playback → download results

📁 Project Structure

AI-Powered Cricket Analytics/
│
├─ data/                        # Input videos
├─ output/                      # Annotated video & evaluation JSON
├─ utils/                        # pose.py, metrics.py, phases.py, evaluation.py
├─ cover_drive_analysis_realtime.py
├─ streamlit_app.py              # Streamlit demo
├─ requirements.txt
└─ README.md


⚙️ Setup

1. Clone the repo
   git clone <your-repo-url>
   cd AI-Powered-Cricket-Analytics

2. Create a virtual environment (optional but recommended)

    conda create -n cricket_env python=3.10
    conda activate cricket_env

3. Install dependencies

   pip install -r requirements.txt

▶️ Running the Analysis Script

python cover_drive_analysis_realtime.py --input data/input_video.mp4 --output output/annotated.mp4

🌐 Running the Streamlit App

streamlit run streamlit_app.py

🛠️ Notes

Lightweight MediaPipe pose model for real-time performance

Metrics normalized for different video resolutions

Gracefully handles missing/occluded joints

Designed for fresher-level portfolio demo; modular and extendable

🔮 Future Enhancements

Automatic bat tracking & swing path analysis

Streamlit sidebar for real-time metrics & phase charts

Multi-shot video analysis & skill grade prediction

Export HTML/PDF report summarizing metrics
