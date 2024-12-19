import os
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Tentukan path model (model ada di folder root proyek)
MODEL_PATH = "best_model.h5"
CLASS_NAMES = ['Fuji Apple', 'Golden Delicious Apple', 'Granny Smith Apple']

# Fungsi untuk memuat model dengan validasi tambahan
def load_classification_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"Model tidak ditemukan di path: {model_path}. Pastikan file tersebut ada di folder root proyek.")
        return None
    try:
        return load_model(model_path)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# Load model
model = load_classification_model(MODEL_PATH)

# Fungsi untuk mengklasifikasi gambar
def classify_image(image_path, model):
    try:
        input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
        input_image_array = tf.keras.utils.img_to_array(input_image)
        input_image_exp_dim = tf.expand_dims(input_image_array, 0)

        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])

        class_idx = np.argmax(result)
        confidence_scores = result.numpy()
        return CLASS_NAMES[class_idx], confidence_scores
    except Exception as e:
        return "Error", str(e)

# Fungsi untuk membuat progress bar khusus
def custom_progress_bar(confidence, colors):
    progress_html = "<div style=\"border: 1px solid #ddd; border-radius: 5px; overflow: hidden; width: 100%; font-size: 14px;\">"
    for conf, color in zip(confidence, colors):
        percentage = conf * 100
        progress_html += f"<div style=\"width: {percentage:.2f}%; background: {color}; color: white; text-align: center; height: 24px; float: left;\">{percentage:.2f}%</div>"
    progress_html += "</div>"
    st.sidebar.markdown(progress_html, unsafe_allow_html=True)

# Tampilan utama aplikasi
st.title("Klasifikasi Buah Apel - Fuji, Granny, Golden")
uploaded_files = st.file_uploader("Unggah Gambar Apel (Beberapa diperbolehkan)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if st.sidebar.button("Prediksi"):
    if not model:
        st.sidebar.error("Model tidak berhasil dimuat. Tidak dapat melanjutkan prediksi.")
    elif uploaded_files:
        st.sidebar.write("### Hasil Prediksi")
        for uploaded_file in uploaded_files:
            try:
                with open(uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                label, confidence = classify_image(uploaded_file.name, model)

                if label != "Error":
                    colors = ["#FF4136", "#2ECC40", "#FFD700"]  # Merah, Hijau, Kuning
                    st.sidebar.write(f"*Nama File:* {uploaded_file.name}")
                    st.sidebar.markdown(f"<h4 style='color: {colors[np.argmax(confidence)]};'> Prediksi: {label} </h4>", unsafe_allow_html=True)

                    st.sidebar.write("*Confidence:*")
                    for i, cls_name in enumerate(CLASS_NAMES):
                        st.sidebar.write(f"- {cls_name}: {confidence[i] * 100:.2f}%")

                    custom_progress_bar(confidence, colors)

                    st.sidebar.write("---")
                else:
                    st.sidebar.error(f"Kesalahan saat memproses gambar {uploaded_file.name}: {confidence}")
            except Exception as e:
                st.sidebar.error(f"Gagal memproses file {uploaded_file.name}: {e}")
    else:
        st.sidebar.error("Silahkan unggah setidaknya satu gambar untuk diprediksi.")

if uploaded_files:
    st.write("### Preview Gambar")
    for uploaded_file in uploaded_files:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption=uploaded_file.name, use_column_width=True)
        except Exception as e:
            st.error(f"Gagal memuat gambar {uploaded_file.name}: {e}")
