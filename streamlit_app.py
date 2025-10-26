import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import av
import cv2

st.set_page_config(page_title="Deteksi Sampah YOLO", layout="wide")

st.title("♻️ Deteksi Jenis Sampah (YOLOv8)")
st.markdown("Gunakan kamera Anda untuk mendeteksi jenis sampah secara langsung menggunakan model YOLOv8.")

# ====== Load YOLO model ======
model_path = "models/best9.pt"
model = YOLO(model_path)

# ====== Kelas untuk proses frame video ======
class YOLOVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model(img)
        annotated = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# ====== Komponen kamera ======
webrtc_streamer(
    key="yolo-detection",
    video_transformer_factory=YOLOVideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
)

st.markdown("---")
st.info("🟢 Pastikan Anda mengizinkan akses kamera di browser agar deteksi dapat berjalan.")
