import streamlit as st
from pathlib import Path
import PIL
import settings
import helper
import base64
from datetime import datetime
from sqlalchemy.orm import Session
from settings import DetectionResult, get_db
import io

st.set_page_config(
    page_title="Pendeteksi Penyakit Daun Padi Menggunakan YOLOv8",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Judul utama aplikasi
st.title("🌾 Pendeteksi Penyakit Pada Daun Padi Menggunakan YOLOv8")

# Sidebar untuk navigasi halaman
st.sidebar.header("Halooo👋🏼, Selamat Datang")
page = st.sidebar.selectbox("Pilih Halaman", ["[Home] 🍃", "[Detection] 🔬", "[History Image] 📚"])

# Penjelasan pada halaman home
if page == "[Home] 🍃":
    st.header("Selamat Datang di Aplikasi Deteksi Penyakit Daun Padi! 🌱")
    st.write(
        """
        Aplikasi ini menggunakan algoritma YOLOv8 untuk mendeteksi berbagai penyakit pada daun padi. 
        Dengan teknologi ini, Anda dapat:
        - Mengunggah gambar atau video daun padi untuk mendeteksi penyakit secara otomatis.
        - Melihat hasil deteksi dengan tingkat kepercayaan yang disesuaikan.
        - Menyimpan dan mengelola riwayat deteksi untuk referensi di masa depan.
        
        **Mulai dengan memilih halaman di sidebar dan unggah gambar atau video Anda di halaman Deteksi!** 📸
        """
    )

# Hanya tampilkan sidebar konfigurasi model di halaman Detection
if page == "[Detection] 🔬":
    # Sidebar untuk konfigurasi model
    st.sidebar.header("Konfigurasi Model")
    model_type = 'Detection'

    # Slider untuk mengatur tingkat kepercayaan model
    confidence = float(st.sidebar.slider("Tingkat Kepercayaan Model By %", 0, 100, 25)) / 100

    model_path = Path(settings.DETECTION_MODEL)
    model = helper.load_model(model_path)

    # Sidebar untuk memilih sumber gambar atau video
    st.sidebar.header("Konfigurasi Gambar/Video:")
    source_radio = st.sidebar.radio("Pilih Sumber", settings.SOURCES_LIST)

# Fungsi untuk mengubah ukuran gambar
def resize_to_fixed_height(image, height):
    aspect_ratio = image.width / image.height
    width = int(height * aspect_ratio)
    return image.resize((width, height))

fixed_height = 300  # Set a fixed height for resizing

# Tempatkan state untuk hasil deteksi dan hasil prediksi
if 'detection_result' not in st.session_state:
    st.session_state['detection_result'] = None
if 'detection_prediction' not in st.session_state:
    st.session_state['detection_prediction'] = None

def save_detection_result(object_name, confidence, image):
    db: Session = next(get_db())
    image_buffer = PIL.Image.fromarray(image)
    buffered = io.BytesIO()
    image_buffer.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    detection_result = DetectionResult(
        object_name=object_name,
        confidence=confidence,
        image_base64=image_base64,
        timestamp=datetime.now()
    )
    db.add(detection_result)
    db.commit()

def main():
    if page == "[Detection] 🔬":
        st.header("Deteksi Penyakit pada Daun Padi")
        if source_radio == settings.IMAGE:
            uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image = PIL.Image.open(uploaded_file)
                resized_image = resize_to_fixed_height(image, fixed_height)
                st.image(resized_image, caption="Gambar yang Diupload", use_column_width=False)
                
                # Perform detection
                res_plotted, res = helper.detect_objects(image, model, confidence)
                
                if res_plotted is not None:
                    # Convert to PIL Image and resize the result image to fixed height
                    result_image = PIL.Image.fromarray(res_plotted)
                    resized_result_image = resize_to_fixed_height(result_image, fixed_height)
                    st.session_state['detection_result'] = resized_result_image
                    st.session_state['detection_prediction'] = res

                    # Save detection result
                    classes_detected = [box.cls for box in res[0].boxes]
                    class_names = [model.names[int(cls)] for cls in classes_detected]
                    unique_class_names = list(set(class_names))  # Remove duplicates
                    save_detection_result(', '.join(unique_class_names), confidence, res_plotted)

                if st.session_state['detection_result'] is not None:
                    st.image(st.session_state['detection_result'], caption="Hasil Deteksi Objek", use_column_width=False)
        
        elif source_radio == settings.VIDEO:
            helper.play_stored_video(confidence, model)
            
        elif source_radio == settings.WEBCAM:
            helper.play_webcam(confidence, model)
            
        elif source_radio == settings.YOUTUBE:
            helper.play_youtube_video(confidence, model)

def show_history_page():
    st.header("Riwayat Deteksi")
    db: Session = next(get_db())
    results = db.query(DetectionResult).all()

    for result in results:
        st.write(f"**Nama Objek:** {result.object_name}")
        st.write(f"**Tingkat Kepercayaan:** {float(result.confidence):.2f}")
        st.write(f"**Timestamp:** {result.timestamp}")
        
        # Decode base64 to image
        img_data = base64.b64decode(result.image_base64)
        img = PIL.Image.open(io.BytesIO(img_data))
        
        # Resize the image to fixed height
        resized_img = resize_to_fixed_height(img, fixed_height)
        st.image(resized_img, caption="Hasil Deteksi", use_column_width=False)

        st.write("---")

if __name__ == "__main__":
    if page == "[History Image] 📚":
        show_history_page()
    else:
        main()
