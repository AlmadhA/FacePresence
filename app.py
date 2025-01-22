import streamlit as st
import face_recognition
import cv2
import numpy as np
import os

# Fungsi untuk memuat dataset wajah
def load_dataset(dataset_path="dataset"):
    known_face_encodings = []
    known_face_names = []
    
    for file in os.listdir(dataset_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            image_path = os.path.join(dataset_path, file)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            
            known_face_encodings.append(encoding)
            name = os.path.splitext(file)[0]  # Nama file tanpa ekstensi
            known_face_names.append(name)
    
    return known_face_encodings, known_face_names

# Fungsi untuk memulai kamera dan memproses wajah
def recognize_faces(known_face_encodings, known_face_names):
    cap = cv2.VideoCapture(0)
    attendance = []

    st.write("Tekan 'q' pada jendela kamera untuk keluar.")
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Gagal mengakses kamera!")
            break

        # Deteksi wajah di frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            # Tampilkan nama pada frame
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Catat kehadiran
            if name != "Unknown" and name not in attendance:
                attendance.append(name)
                st.success(f"{name} berhasil diabsen!")

        # Tampilkan kamera
        cv2.imshow("Face Recognition Attendance", frame)

        # Keluar dengan menekan 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return attendance

# Streamlit UI
st.title("OUTING FP&A 2025")
st.write("Arahkan wajah ke kamera untuk absen. Tekan tombol di bawah untuk memulai.")

# Load dataset
st.info("Memuat dataset wajah...")
known_face_encodings, known_face_names = load_dataset()

# Tombol untuk memulai pengenalan wajah
if st.button("Mulai Absensi"):
    st.write("Buka kamera dan arahkan wajah peserta.")
    attendance = recognize_faces(known_face_encodings, known_face_names)
    
    # Tampilkan hasil absensi
    st.subheader("Daftar Hadir")
    if attendance:
        st.write(attendance)
    else:
        st.warning("Tidak ada peserta yang diabsen.")

# Ekspor daftar hadir
if "attendance" in locals():
    st.download_button(
        label="Download Absensi",
        data="\n".join(attendance),
        file_name="absensi_wajah.txt",
        mime="text/plain",
    )