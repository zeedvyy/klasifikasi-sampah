import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Set path relatif ke model
MODEL_PATH = os.path.join('model', 'googlenet_model.h5')

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Daftar label kelas (sesuaikan dengan modelmu)
class_names = ['Organik', 'Anorganik', 'B3', 'Campuran']

# Fungsi prediksi
def predict_image(image):
    img = image.resize((224, 224))  # Sesuaikan dengan input model
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Tambah batch dimension
    img_array = img_array / 255.0  # Normalisasi

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100
    return predicted_class, confidence

# Streamlit UI
st.title("🧠 Klasifikasi Gambar Sampah")
st.write("Upload gambar sampah untuk diklasifikasikan menggunakan model GoogLeNet.")

uploaded_file = st.file_uploader("Pilih gambar sampah", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    if st.button("Prediksi"):
        with st.spinner("Memproses gambar..."):
            label, confidence = predict_image(image)
            st.success(f"Hasil Prediksi: **{label}** ({confidence:.2f}% yakin)")
