import os
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Tentukan path model (model ada di folder root proyek)
model_path = "best_model.h5"

# Pastikan model ada di dalam folder root
if not os.path.exists(model_path):
    st.error("Model tidak ditemukan. Pastikan file 'gugelnet_apel.h5' ada di folder root proyek.")
else:
    # Load model
    model = load_model(model_path)
    class_name = ['merah', 'hijau', 'kuning']

    # Fungsi untuk mengklasifikasi gambar
    def classify_image(image_path):
        try:
            input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
            input_image_array = tf.keras.utils.img_to_array(input_image)
            input_image_exp_dim = tf.expand_dims(input_image_array, 0)

            predictions = model.predict(input_image_exp_dim)
            result = tf.nn.softmax(predictions[0])

            class_idx = np.argmax(result)
            confidence_scores = result.numpy()
            return class_name[class_idx], confidence_scores
        except Exception as e:
            return "Error", str(e)

    # Fungsi untuk membuat progress bar khusus
    def custom_progress_bar(confidence, colors):
        progress_html = "<div style=\"border: 1px solid #ddd; border-radius: 5px; overflow: hidden; width: 100%; font-size: 14px;\">"
        for i, (conf, color) in enumerate(zip(confidence, colors)):
            percentage = conf * 100
            progress_html += f"<div style=\"width: {percentage:.2f}%; background: {color}; color: white; text-align: center; height: 24px; float: left;\">{percentage:.2f}%</div>"
        progress_html += "</div>"
        st.sidebar.markdown(progress_html, unsafe_allow_html=True)

    # Tampilan utama aplikasi
    st.title("Klasifikasi Buah Apel - Fuji, Granny, Golden")
    uploaded_files = st.file_uploader("Unggah Gambar Apel (Beberapa diperbolehkan)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if st.sidebar.button("Prediksi"):
        if uploaded_files:
            st.sidebar.write("### Hasil Prediksi")
            for uploaded_file in uploaded_files:
                with open(uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                label, confidence = classify_image(uploaded_file.name)

                if label != "Error":
                    colors = ["#FF4136", "#2ECC40", "#FFD700"]  # Merah, Hijau, Kuning
                    st.sidebar.write(f"*Nama File:* {uploaded_file.name}")
                    st.sidebar.markdown(f"<h4 style='color: {colors[np.argmax(confidence)]};'> Prediksi: {label} </h4>", unsafe_allow_html=True)

                    st.sidebar.write("*Confidence:*")
                    for i, cls_name in enumerate(class_name):
                        st.sidebar.write(f"- {cls_name}: {confidence[i] * 100:.2f}%")

                    custom_progress_bar(confidence, colors)

                    st.sidebar.write("---")
                else:
                    st.sidebar.write(f"Kesalahan saat memproses gambar {uploaded_file.name}: {confidence}")
        else:
            st.sidebar.error("Silahkan unggah setidaknya satu gambar untuk diprediksi.")

    if uploaded_files:
        st.write("### Preview Gambar")
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            st.image(image, caption=uploaded_file.name, use_column_width=True)
