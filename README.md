# 🌍 Sejukin – Air Quality Status Predictor

**Sejukin** adalah aplikasi berbasis Streamlit yang memprediksi status kualitas udara berdasarkan tanggal, negara, dan nilai AQI (Air Quality Index). Aplikasi ini menggunakan model Artificial Neural Network (ANN) yang telah dikonversi ke format `.tflite` untuk performa optimal dan kemudahan deployment.

---

## 🚀 Fitur

- Prediksi status kualitas udara (Good, Moderate, Unhealthy, dll.)
- Input data: tanggal, negara, dan nilai AQI
- Visualisasi peta berdasarkan nilai AQI
- Model ringan dengan TensorFlow Lite

---

## 📁 Struktur File
sejukin/
├── app.py                   # Aplikasi utama Streamlit
├── model_ann.tflite         # Model ANN dalam format TensorFlow Lite        
├── scaler.pkl               # Scaler untuk normalisasi fitur
├── country_encoder.pkl      # Label encoder untuk fitur negara
├── status_encoder.pkl       # Label encoder untuk label target (status AQI)
├── requirements.txt         # Daftar library Python yang diperlukan

---

## 🛠️ Dokumentasi Instalasi 

### 1. Clone Repository

```bash
git clone https://github.com/namauser/sejukin.git
cd sejukin
```

### 2. Buat dan Aktifkan Virtual Environment

```bash
python3 -m venv myvenv
source myvenv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4.Jalankan Aplikasi

```bash
streamlit run app.py
```

## 🧠 Tentang Model

Model dibuat menggunakan **Keras** dan disimpan dalam dua format:

- `model_ann.h5` – Model asli dalam format HDF5 (Keras standard)
- `model_ann.tflite` – Model ringan dalam format TensorFlow Lite, digunakan untuk inference di aplikasi
---

### 🔢 Fitur Input Model

Model menerima input berupa lima fitur numerik:

- Tahun (dari tanggal)
- Bulan (dari tanggal)
- Hari (dari tanggal)
- Negara (dalam bentuk label encoded menggunakan `country_encoder.pkl`)
- Nilai AQI (Air Quality Index)

Semua fitur diolah melalui **scaler (`scaler.pkl`)** sebelum dimasukkan ke model.
---

## 📌 Contoh Input

Berikut adalah contoh data yang dimasukkan ke dalam aplikasi:

| Tanggal     | Negara     | AQI Value |
|-------------|------------|-----------|
| 2023-06-15  | Indonesia  | 135       |
---

## ✅ Output

Setelah pengguna memasukkan data dan menekan tombol prediksi, aplikasi akan menghasilkan:

- **Prediksi status kualitas udara**, seperti: `Good`, `Moderate`, `Unhealthy`, dll.
- **Visualisasi peta negara** yang dipilih, dengan warna yang merepresentasikan nilai AQI menggunakan skala warna (semakin merah, semakin tidak sehat).
- **Akurasi 97%**.


