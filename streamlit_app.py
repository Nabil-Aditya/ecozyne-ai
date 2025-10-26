import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from ultralytics import YOLO
import numpy as np

st.set_page_config(page_title="Deteksi Sampah YOLO", layout="wide")

st.title("♻️ Deteksi Jenis Sampah Real-time (YOLOv8)")

# ====== Load Model ======
@st.cache_resource
def load_model():
    model_path = "models/best9.pt"
    return YOLO(model_path)

with st.spinner("🔄 Loading model..."):
    model = load_model()

st.success("✅ Model berhasil dimuat!")

# ====== Pengaturan ======
st.sidebar.header("⚙️ Pengaturan Deteksi")
confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Cara Penggunaan:**
    1. Klik tombol "START" di bawah
    2. Izinkan akses kamera di browser
    3. Deteksi akan berjalan otomatis
    4. Klik "STOP" untuk menghentikan
    """
)

# ====== Video Processor Class ======
class WasteDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.confidence = confidence
    
    def recv(self, frame):
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Run YOLO detection
        results = self.model(img, conf=self.confidence, verbose=False)
        
        # Get annotated frame
        annotated_frame = results[0].plot()
        
        # Convert back to av.VideoFrame
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# ====== WebRTC Streamer ======
st.subheader("🎥 Live Camera Detection")

webrtc_ctx = webrtc_streamer(
    key="waste-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=WasteDetectionProcessor,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    },
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1280},
            "height": {"ideal": 720},
        },
        "audio": False,
    },
    async_processing=True,
)

# ====== Status Info ======
if webrtc_ctx.state.playing:
    st.success("🟢 **Deteksi Aktif** - Kamera sedang berjalan")
else:
    st.info("⚪ Klik tombol **START** untuk mulai deteksi")

# ====== Info Tambahan ======
st.markdown("---")
st.markdown(
    """
    ### 📝 Informasi:
    - ✅ **Deteksi Real-time**: Langsung dari kamera browser Anda
    - 🔒 **Privacy**: Video diproses secara lokal, tidak disimpan
    - 🎯 **Model**: YOLOv8 custom trained untuk deteksi sampah
    - 🌐 **Browser Support**: Chrome, Firefox, Edge (terbaru)
    
    ### ⚠️ Troubleshooting:
    - Jika kamera tidak muncul, coba refresh halaman
    - Pastikan browser sudah izinkan akses kamera
    - Gunakan browser modern (Chrome/Firefox recommended)
    """
)

# ====== Footer ======
st.markdown("---")
st.markdown(
    "<div style='text-align: center'><p>Powered by YOLOv8 🚀 | Built with Streamlit</p></div>",
    unsafe_allow_html=True
)