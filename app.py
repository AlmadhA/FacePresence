import streamlit as st
import cv2
import face_recognition
import numpy as np
import tempfile
import os
from PIL import Image

# Fungsi untuk memuat dataset wajah (dengan caching)
@st.cache_resource
def load_dataset(dataset_path="dataset"):
    known_face_encodings = []
    known_face_names = []
    
    for file in os.listdir(dataset_path):
        if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
            image_path = os.path.join(dataset_path, file)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            
            known_face_encodings.append(encoding)
            name = os.path.splitext(file)[0]  # Nama file tanpa ekstensi
            known_face_names.append(name)
    
    return known_face_encodings, known_face_names

# Fungsi untuk pengenalan wajah
def recognize_faces(image, known_face_encodings, known_face_names):
    rgb_image = np.array(image.convert('RGB'))
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    recognized_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        recognized_names.append(name)
    return face_locations, recognized_names

# Fungsi untuk menampilkan bounding box pada wajah
def draw_bounding_boxes(image, face_locations, recognized_names):
    image_np = np.array(image)
    for (top, right, bottom, left), name in zip(face_locations, recognized_names):
        cv2.rectangle(image_np, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image_np, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return Image.fromarray(image_np)

# Streamlit UI
st.title("Absensi Wajah - FP&A 2025")
st.write("Aktifkan kamera, arahkan wajah Anda, lalu tekan tombol untuk mengambil gambar.")

# Load dataset wajah
st.info("Memuat dataset wajah...")
known_face_encodings, known_face_names = load_dataset()

# Video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Tidak dapat mengakses kamera!")

FRAME_WINDOW = st.image([])
captured_image = None

# Stream kamera di Streamlit
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Gagal membaca frame dari kamera!")
        break

    # Tampilkan kamera di Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

    # Tombol untuk menangkap gambar
    if st.button("Tangkap Gambar"):
        captured_image = frame
        st.success("Gambar berhasil ditangkap!")
        break

cap.release()

# Jika gambar berhasil ditangkap
if captured_image is not None:
    st.subheader("Gambar yang Ditangkap")
    st.image(captured_image, caption="Gambar yang Ditangkap", use_column_width=True)

    # Lakukan pengenalan wajah
    st.info("Melakukan pengenalan wajah...")
    pil_image = Image.fromarray(captured_image)
    face_locations, recognized_names = recognize_faces(pil_image, known_face_encodings, known_face_names)

    # Tampilkan hasil pengenalan
    if recognized_names:
        st.subheader("Hasil Pengenalan Wajah")
        for name in recognized_names:
            st.write(f"Wajah dikenali sebagai: **{name}**")
            if st.button(f"Konfirmasi Kehadiran untuk {name}"):
                st.success(f"Kehadiran untuk {name} telah dicatat.")
            else:
                st.warning(f"Klik tombol untuk konfirmasi kehadiran.")

        # Tampilkan bounding box di gambar
        result_image = draw_bounding_boxes(pil_image, face_locations, recognized_names)
        st.image(result_image, caption="Hasil dengan Bounding Box", use_column_width=True)
    else:
        st.warning("Tidak ada wajah yang dikenali. Coba lagi.")