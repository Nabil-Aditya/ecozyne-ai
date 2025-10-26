import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
from ultralytics import YOLO
import cv2

st.set_page_config(page_title="Deteksi Sampah YOLOv8 Realtime", layout="wide")

st.title("♻️ Deteksi Sampah Realtime (YOLOv8 + Webcam Browser)")

# ====== Load model YOLO ======
model = YOLO("models/best9.pt")

# ====== Konfigurasi WebRTC ======
rtc_configuration = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# ====== Video Processor ======
class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Prediksi YOLO
        results = self.model(img)
        annotated = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# ====== Jalankan kamera browser ======
webrtc_streamer(
    key="yolo-realtime",
    mode="sendrecv",
    rtc_configuration=rtc_configuration,
    video_processor_factory=YOLOVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

st.markdown("---")
st.info("🔹 Kamera akan aktif setelah kamu klik 'Start' dan memberi izin kamera di browser.")
st.caption("⚠️ Pastikan koneksi stabil, karena YOLO jalan di server Streamlit Cloud.")
