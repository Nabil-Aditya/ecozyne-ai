import streamlit as st
import cv2
from ultralytics import YOLO

st.set_page_config(page_title="Deteksi Sampah YOLO", layout="wide")

st.title("♻️ Deteksi Jenis Sampah (YOLOv8)")

# ====== Load model ======
model_path = "models/best9.pt"
model = YOLO(model_path)

# ====== Pilihan Kamera ======
st.subheader("Pilih Kamera yang Ingin Digunakan")
camera_option = st.selectbox(
    "Pilih sumber kamera:",
    options=[0, 1, 2],
    format_func=lambda x: f"Kamera {x} (index {x})"
)
st.caption("💡 Jika kamera tidak tampil, coba ganti index (0 biasanya kamera laptop, 1 untuk webcam eksternal).")

# ====== Tombol kamera ======
start_cam = st.checkbox("🎥 Aktifkan Kamera")

# Placeholder tampilan video
frame_placeholder = st.empty()

if start_cam:
    cap = cv2.VideoCapture(camera_option)
    if not cap.isOpened():
        st.error("❌ Kamera tidak terdeteksi! Pastikan kamera aktif dan index benar.")
    else:
        st.success(f"✅ Kamera {camera_option} aktif — deteksi sedang berjalan...")

        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Gagal membaca kamera.")
                break

            # Jalankan prediksi YOLO
            results = model(frame)
            annotated_frame = results[0].plot()

            # Tampilkan hasil deteksi
            frame_placeholder.image(annotated_frame, channels="BGR")

            # Cek status kamera (kalau user uncheck)
            if not st.session_state.get("🎥 Aktifkan Kamera", True):
                break

        cap.release()
        st.info("Kamera dimatikan.")
else:
    st.info("Centang kotak di atas untuk menyalakan kamera dan mulai deteksi.")
