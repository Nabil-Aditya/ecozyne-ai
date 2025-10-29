import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2

st.set_page_config(page_title="Deteksi Sampah YOLO", layout="wide")

st.title("♻️ Deteksi Jenis Sampah Real-time (YOLOv8)")

# ====== Class Mapping ======
CLASS_MAPPING = {
    'Non_Organics_Metal': 'Non_Organics',
    'Non_Organics_Paper': 'Non_Organics',
    'Non_Organics_Glass': 'Non_Organics',
    'Non_Organics_Plastic': 'Non_Organics',
    'Non_Organics_Textile': 'Non_Organics',
    'Non_Organics_Miscellaneous': 'Non_Organics',
    'Non_Organics_Cardboard': 'Non_Organics',
    'Organics_Vegetation': 'Organics_NonEco',
    'Organics_Food': 'Organics_NonEco',
    'Organics_Eco': 'Organics_Eco',
}

def map_class(original_class):
    """Map original class to simplified category"""
    return CLASS_MAPPING.get(original_class, original_class)

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
confidence = 0.25  # Fixed confidence threshold

show_original_class = st.sidebar.checkbox("Tampilkan Kelas Asli", value=False)

# ====== Mode Selection ======
mode = st.radio(
    "Pilih Mode:",
    ["📸 Snapshot Mode (Ambil Foto)", "🖼️ Upload Gambar"],
    horizontal=True
)

st.markdown("---")

# ====== MODE 1: Camera Snapshot ======
if mode == "📸 Snapshot Mode (Ambil Foto)":
    st.subheader("📸 Ambil Foto dari Kamera")
    st.info("💡 Klik tombol kamera di bawah untuk mengambil foto, lalu deteksi akan otomatis berjalan")
    
    # Enable continuous detection
    enable_continuous = st.checkbox("🔄 Mode Continuous (auto-refresh setiap foto)")
    
    camera_photo = st.camera_input("Ambil foto dari kamera Anda")
    
    if camera_photo is not None:
        # Load image
        image = Image.open(camera_photo)
        img_array = np.array(image)
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📷 Foto Asli")
            st.image(image, use_container_width=True)
        
        # Run detection
        with st.spinner("🔍 Mendeteksi sampah..."):
            results = model(img_array, conf=confidence, verbose=False)
            annotated_img = results[0].plot()
            
            # Get detection info
            detections = results[0].boxes
        
        with col2:
            st.markdown("#### ✅ Hasil Deteksi")
            st.image(annotated_img, channels="BGR", use_container_width=True)
        
        # Show detection details
        st.markdown("---")
        st.markdown("### 📊 Detail Deteksi")
        
        if len(detections) > 0:
            # Count by mapped category
            category_counts = {}
            for box in detections:
                cls = int(box.cls[0])
                original_label = model.names[cls]
                mapped_label = map_class(original_label)
                category_counts[mapped_label] = category_counts.get(mapped_label, 0) + 1
            
            # Show category summary
            st.markdown("#### 🎯 Ringkasan Kategori")
            summary_cols = st.columns(len(category_counts))
            for idx, (category, count) in enumerate(category_counts.items()):
                with summary_cols[idx]:
                    st.metric(
                        label=category,
                        value=f"{count} objek"
                    )
            
            st.markdown("---")
            
            # Create metrics for individual detections
            st.markdown("#### 🔍 Detail Objek Terdeteksi")
            cols = st.columns(min(len(detections), 4))
            
            for i, box in enumerate(detections):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                original_label = model.names[cls]
                mapped_label = map_class(original_label)
                
                with cols[i % 4]:
                    if show_original_class:
                        st.metric(
                            label=f"Objek {i+1}",
                            value=mapped_label,
                            delta=f"{conf:.1%}",
                            help=f"Kelas asli: {original_label}"
                        )
                    else:
                        st.metric(
                            label=f"Objek {i+1}",
                            value=mapped_label,
                            delta=f"{conf:.1%}"
                        )
            
            # Detailed table
            with st.expander("📋 Lihat Detail Lengkap"):
                for i, box in enumerate(detections):
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    original_label = model.names[cls]
                    mapped_label = map_class(original_label)
                    coords = box.xyxy[0].cpu().numpy()
                    
                    st.markdown(f"""
                    **Deteksi {i+1}:**
                    - 🏷️ Kategori: `{mapped_label}`
                    - 🔖 Kelas Asli: `{original_label}`
                    - 📊 Confidence: `{conf:.2%}`
                    - 📍 Koordinat: `x1={coords[0]:.0f}, y1={coords[1]:.0f}, x2={coords[2]:.0f}, y2={coords[3]:.0f}`
                    """)
        else:
            st.warning("⚠️ Tidak ada sampah terdeteksi dalam gambar")
            st.info("💡 Coba ambil foto lagi dengan objek yang lebih jelas")
        
        # Auto-refresh untuk continuous mode
        if enable_continuous:
            st.markdown("---")
            st.info("🔄 Mode Continuous aktif - Ambil foto baru untuk deteksi berikutnya")

# ====== MODE 2: Upload Image ======
elif mode == "🖼️ Upload Gambar":
    st.subheader("🖼️ Upload Gambar untuk Deteksi")
    
    uploaded_file = st.file_uploader(
        "Pilih gambar (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📷 Gambar Asli")
            st.image(image, use_container_width=True)
        
        # Run detection
        with st.spinner("🔍 Mendeteksi sampah..."):
            results = model(img_array, conf=confidence, verbose=False)
            annotated_img = results[0].plot()
            
            # Get detection info
            detections = results[0].boxes
        
        with col2:
            st.markdown("#### ✅ Hasil Deteksi")
            st.image(annotated_img, channels="BGR", use_container_width=True)
        
        # Show detection details
        st.markdown("---")
        st.markdown("### 📊 Detail Deteksi")
        
        if len(detections) > 0:
            # Count by mapped category
            category_counts = {}
            for box in detections:
                cls = int(box.cls[0])
                original_label = model.names[cls]
                mapped_label = map_class(original_label)
                category_counts[mapped_label] = category_counts.get(mapped_label, 0) + 1
            
            # Show category summary
            st.markdown("#### 🎯 Ringkasan Kategori")
            summary_cols = st.columns(len(category_counts))
            for idx, (category, count) in enumerate(category_counts.items()):
                with summary_cols[idx]:
                    st.metric(
                        label=category,
                        value=f"{count} objek"
                    )
            
            st.markdown("---")
            
            # Create metrics for individual detections
            st.markdown("#### 🔍 Detail Objek Terdeteksi")
            cols = st.columns(min(len(detections), 4))
            
            for i, box in enumerate(detections):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                original_label = model.names[cls]
                mapped_label = map_class(original_label)
                
                with cols[i % 4]:
                    if show_original_class:
                        st.metric(
                            label=f"Objek {i+1}",
                            value=mapped_label,
                            delta=f"{conf:.1%}",
                            help=f"Kelas asli: {original_label}"
                        )
                    else:
                        st.metric(
                            label=f"Objek {i+1}",
                            value=mapped_label,
                            delta=f"{conf:.1%}"
                        )
            
            # Summary
            st.success(f"✅ Terdeteksi **{len(detections)} objek sampah**")
            
            # Detailed table
            with st.expander("📋 Lihat Detail Lengkap"):
                for i, box in enumerate(detections):
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    original_label = model.names[cls]
                    mapped_label = map_class(original_label)
                    
                    st.markdown(f"**{i+1}.** {mapped_label} ({original_label}) — Confidence: {conf:.2%}")
        else:
            st.warning("⚠️ Tidak ada sampah terdeteksi")

# ====== Info & Help ======
st.markdown("---")

with st.expander("ℹ️ Informasi & Bantuan"):
    st.markdown("""
    ### 🎯 Cara Penggunaan:
    
    **Mode Snapshot:**
    - Klik tombol kamera untuk mengambil foto
    - Sistem akan otomatis mendeteksi sampah
    - Aktifkan "Mode Continuous" untuk deteksi berulang
    
    **Mode Upload:**
    - Upload gambar dari galeri/file
    - Sistem akan mendeteksi semua sampah dalam gambar
    
    ### 🏷️ Kategori Sampah:
    - **Non_Organics**: Metal, Paper, Glass, Plastic, Textile, Cardboard, Miscellaneous
    - **Organics_NonEco**: Vegetation, Food
    - **Organics_Eco**: Eco-friendly organic waste
    
    ### ⚙️ Tips untuk Hasil Terbaik:
    - 💡 Gunakan pencahayaan yang cukup
    - 📏 Jarak objek tidak terlalu jauh
    - 🎯 Fokus pada objek sampah
    - 🔍 Pastikan objek terlihat jelas
    
    ### 🔒 Privacy:
    - Gambar diproses secara real-time
    - Tidak ada data yang disimpan di server
    - 100% privasi terjaga
    """)

# ====== Footer ======
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 1rem;'>
        <p style='color: #666;'>
            Made by Ecozyne Team Development 🚀 | Built with ❤️ 
        </p>
    </div>
    """,
    unsafe_allow_html=True
)