import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.efficientnet import preprocess_input
import io

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model_mangga_efficientnetb3.tflite")
interpreter.allocate_tensors()

# Ambil input dan output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Label kelas
class_labels = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back',
                'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']

# Fungsi preprocessing dan prediksi
def load_and_preprocess(img_file, img_size=(224, 224)):
    img = keras_image.load_img(img_file, target_size=img_size)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img, img_array.astype(np.float32)  # TFLite biasanya pakai float32

# Fungsi prediksi dengan TFLite
def predict_tflite(img_array):
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# ---------------------------- UI Aplikasi ----------------------------
st.markdown("<h1 style='text-align: center;'>🍃 Deteksi Penyakit Daun Mangga</h1>", unsafe_allow_html=True)
st.markdown("### Selamat Datang!")
st.write("""
Aplikasi ini digunakan untuk mendeteksi penyakit pada daun mangga menggunakan teknologi Deep Learning berbasis EfficientNetB3 dalam format TFLite.  
Silakan unggah gambar daun mangga yang ingin Anda periksa untuk mendapatkan hasil klasifikasi.
""")

# Penjelasan Penyakit
st.markdown("### Jenis-Jenis Penyakit (Klik untuk lihat penjelasan):")
with st.expander("Anthracnose"):
    st.write("Penyakit jamur yang menyebabkan bercak hitam atau coklat pada daun dan buah.")
with st.expander("Bacterial Canker"):
    st.write("Infeksi bakteri yang menyebabkan luka berair, daun mengering, dan cabang mati.")
with st.expander("Powdery Mildew"):
    st.write("Jamur seperti bedak putih yang menghambat pertumbuhan.")
with st.expander("Cutting Weevil"):
    st.write("Serangan kumbang pemotong menyebabkan kerusakan pada jaringan daun dan pembentukan lubang-lubang kecil.")
with st.expander("Die Back"):
    st.write("Penyakit yang menyebabkan pengeringan dan kematian bertahap pada ujung ranting dan cabang tanaman.")
with st.expander("Gall Midge"):
    st.write("Serangan serangga pengisap yang menyebabkan pembengkakan (gall) pada jaringan daun muda atau tunas.")
with st.expander("Sooty Mould"):
    st.write("Lapisan hitam seperti jelaga yang tumbuh di permukaan daun akibat embun madu dari serangga seperti kutu putih.")

# Upload dan Prediksi
st.markdown("### 📷 Upload Gambar Daun")
uploaded_file = st.file_uploader("Unggah gambar daun mangga", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img, img_array = load_and_preprocess(uploaded_file)
    
    # Tampilkan gambar
    st.image(img, caption="Gambar yang Diupload", use_column_width=True)

    # Prediksi
    pred = predict_tflite(img_array)
    pred_class_idx = np.argmax(pred)
    pred_label = class_labels[pred_class_idx]
    pred_conf = pred[0][pred_class_idx] * 100

    result = f"{pred_label} ({pred_conf:.2f}%)"
    st.success(f"✅ Hasil Deteksi: {result}")

    # Detail probabilitas semua kelas
    st.markdown("#### Probabilitas Kelas:")
    for i, label in enumerate(class_labels):
        st.write(f"{label}: {pred[0][i]:.4f}")

    # Simpan ke history.txt
    with open("history.txt", "a") as f:
        f.write(f"{result}\n")
