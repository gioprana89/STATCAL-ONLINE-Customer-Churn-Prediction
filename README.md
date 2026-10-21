# STATCAL ONLINE — Machine Learning Lab (Bab 2)
## Regresi Logistik: Prediksi Pelanggan Berhenti Berlangganan (Customer Churn)

Aplikasi Streamlit ini merupakan **praktikum Bab 2** dari buku **Machine Learning Berbasis Studi Kasus**. Menggunakan dataset sintetis **1.000 pelanggan** untuk memahami cara kerja prediksi Regresi Logistik dengan empat pertanyaan sederhana.

## Cara menjalankan di Windows

1. Instal Python versi 3.10 atau lebih baru.
2. Ekstrak ZIP. Pastikan `app.py`, `ml_core.py`, `dataset_pelanggan.csv`, dan `requirements.txt` berada dalam folder yang sama.
3. Buka Command Prompt/terminal di folder tersebut.
4. Jalankan:
   ```bash
   python -m pip install -r requirements.txt
   python -m streamlit run app.py
   ```
5. Buka alamat lokal yang tampil di terminal (umumnya http://localhost:8501).

## Deploy di Streamlit Community Cloud

1. Buat repository di GitHub dan unggah `app.py`, `ml_core.py`, `requirements.txt`, serta `dataset_pelanggan.csv` pada root repository.
2. Masuk ke https://share.streamlit.io/ dan buat aplikasi baru dengan `app.py` sebagai entry point.
3. Excel sumber disertakan dalam ZIP untuk bahan praktikum, tetapi tidak wajib diunggah ke GitHub karena CSV bawaan sudah ada.

## Menggunakan dataset Excel yang diunggah

Aplikasi menerima file `.csv` atau `.xlsx`. Jika menggunakan workbook yang disertakan, pilih **DATA_PELANGGAN** sebagai worksheet. Aplikasi akan otomatis menggunakan CSV bawaan jika tidak ada file yang diunggah.

## Pengaturan standar buku (untuk reprodusibilitas)

- Target (Y): `Status_Berlangganan`.
- Kelas positif (1): `Berhenti`.
- Kelas negatif (0): `Bertahan`.
- Variabel X: delapan prediktor, tanpa `ID_Pelanggan`.
- Data training: 80%; testing: 20% (stratified).
- Threshold: 0,50.
- Regularisasi C: 1,0; bobot kelas seimbang: tidak aktif.
- Random seed: 42.

## Empat pertanyaan praktikum

1. **TP** — Berapa pelanggan yang sebenarnya Berhenti dan benar diprediksi Berhenti?
2. **FN** — Berapa pelanggan yang sebenarnya Berhenti tetapi keliru diprediksi Bertahan?
3. **Recall** — Seberapa baik model menemukan pelanggan yang benar-benar Berhenti?
4. **Probabilitas** — Berapa probabilitas Berhenti dari satu profil pelanggan yang dimasukkan?

## Menu aplikasi

- **Kasus & Data:** seluruh data, pembagian training/testing, dan kamus variabel.
- **Q1 • Benar (TP):** jumlah pelanggan Berhenti yang terdeteksi dengan benar beserta observasi asal.
- **Q2 • Terlewat (FN):** jumlah pelanggan Berhenti yang keliru diprediksi Bertahan beserta observasi asal.
- **Q3 • Recall:** rasio TP/(TP+FN) dan interpretasi.
- **Q4 • Probabilitas:** input profil pelanggan baru dan hasil prediksi probabilitas serta kelas.
- **Laporan Excel:** unduh satu workbook yang berisi ringkasan jawaban, pengaturan, data lengkap, data training, data testing dan seluruh hasil prediksinya.

**Lembar hasil Excel:** Ringkasan, Pertanyaan & Jawaban, Pengaturan Model, Data Lengkap, Data Training, Data Testing, Prediksi Testing, Confusion Matrix, Metrik Evaluasi, Simulasi Profil, Kamus Variabel. Lembar Simulasi Profil baru berisi hasil apabila pengguna sudah menekan tombol hitung pada tab Q4.

## Catatan metodologi

- Prediktor berasal dari kondisi sampai akhir September 2026, sementara status target adalah Oktober 2026.
- Pemrosesan missing data, standardisasi, dan one-hot encoding dilatih pada bagian **training** dengan scikit-learn Pipeline, bukan pada seluruh data.
- TP, FN, Recall, dan angka evaluasi lain selalu dihitung pada **data testing**.
- Pengubahan threshold dapat mengubah kategori dan Recall walaupun probabilitas dasar model sama.
- Angka probabilitas bukan kepastian, bukan hubungan sebab-akibat, dan tidak otomatis terkalibrasi.
- Data dan prediksi adalah simulasi untuk pembelajaran, **bukan** dasar pengambilan keputusan terhadap pelanggan sebenarnya.

## Referensi teknis

- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- https://docs.streamlit.io/
