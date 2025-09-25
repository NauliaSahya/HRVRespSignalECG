# Proyek Analisis Sinyal Fisiologis Non-Stasioner

Proyek ini merupakan implementasi dan analisis sinyal fisiologis, khususnya sinyal **Elektrokardiogram (ECG)** dan **sinyal pernapasan**, menggunakan teknik pemrosesan sinyal non-stasioner. Tujuannya adalah untuk mendemonstrasikan aplikasi praktis dari algoritma seperti **Discrete Wavelet Transform (DWT)** dan **Filter Bank** dalam mengekstraksi informasi penting seperti **Heart Rate Variability (HRV)** dan laju pernapasan dari sinyal mentah.

## Latar Belakang

Sinyal fisiologis seperti ECG bersifat non-stasioner, yang berarti karakteristik statistiknya (misalnya, rata-rata dan varians) berubah seiring waktu. Oleh karena itu, metode analisis sinyal tradisional seperti Fourier Transform seringkali tidak memadai. Dalam proyek ini, kita menggunakan pendekatan berbasis **Wavelet Transform** yang ideal untuk menganalisis sinyal-sinyal ini karena kemampuannya untuk memberikan resolusi waktu dan frekuensi secara simultan.

## Fitur Utama

### 1. Teori dan Implementasi Filter Bank
* **Koefisien Wavelet**: Memvisualisasikan koefisien wavelet `h[n]` (low-pass filter) dan `g[n]` (high-pass filter) yang menjadi dasar dari transformasi.
* **Respons Frekuensi**: Menghitung dan memplot respons frekuensi untuk setiap level filter bank (Q1 hingga Q8) untuk menunjukkan bagaimana sinyal disaring pada pita frekuensi yang berbeda.
* **Penerapan pada ECG**: Menerapkan filter bank pada sinyal ECG mentah untuk menghasilkan sinyal terfilter pada berbagai skala DWT, menunjukkan dekomposisi sinyal secara visual.

### 2. Analisis Sinyal Non-Stasioner
* **Algoritma Mallat**: Menerapkan algoritma Mallat untuk menguraikan sinyal ECG menjadi koefisien detail (`w2f`) dan koefisien aproksimasi (`s2f`) pada berbagai level.

### 3. Analisis Heart Rate Variability (HRV)
* **Deteksi Puncak R**: Menggunakan hasil dari DWT skala 1, 2, dan 3 untuk mendeteksi puncak R pada sinyal ECG.
* **Analisis Domain Waktu**: Menghitung metrik HRV standar seperti **SDNN**, **RMSSD**, dan **pNN50** dari interval RR (waktu antar puncak R).
* **Analisis Domain Frekuensi**: Menggunakan metode Welch manual untuk menghitung **Power Spectral Density (PSD)** dan menentukan kekuatan daya pada pita frekuensi **VLF**, **LF**, dan **HF**.
* **Analisis Non-Linear**: Menghasilkan plot **Poincaré** untuk analisis non-linear HRV.

### 4. Estimasi Sinyal Pernapasan
* **Ekstraksi Sinyal Pernapasan**: Menggunakan koefisien detail pada skala 8 (pita frekuensi rendah) dari DWT ECG untuk mengestimasi sinyal pernapasan.

---

## Cara Menjalankan Aplikasi

1.  **Persiapan Lingkungan**: Pastikan Anda telah menginstal Python. Instal pustaka yang diperlukan dengan menjalankan perintah berikut:

    ```bash
    pip install streamlit numpy pandas altair pyhrv matplotlib
    ```

2.  **Siapkan Data**: Simpan kode yang diberikan sebagai file Python (misalnya, `hrv_app.py`). Letakkan file data bernama `samples5min.txt` di direktori yang sama.

3.  **Jalankan Aplikasi**: Buka terminal atau Command Prompt, navigasikan ke direktori file, dan jalankan perintah:

    ```bash
    streamlit run hrv_app.py
    ```

4.  Aplikasi akan terbuka secara otomatis di browser web Anda. Anda dapat berinteraksi dengan berbagai opsi dan melihat hasilnya secara langsung.

---

## Hasil dan Visualisasi

Anda dapat menambahkan tangkapan layar (screenshot) dari aplikasi yang sedang berjalan di bawah ini untuk mendokumentasikan hasil dari setiap fitur.

### **Analisis HRV Domain Waktu**
Berdasarkan analisis, nilai-nilai metrik domain waktu yang dihasilkan adalah sebagai berikut:
* **SDNN**: **22.10 ms** (menunjukkan variabilitas rendah)
* **pNN50**: **3.07%**
* **RMSSD**: **35.09 ms** (cenderung rendah)
* **SDSD**: **35.13 ms**

!(time_domain_hrv.png)

### **Analisis HRV Domain Frekuensi**
Rasio **LF/HF yang rendah** mengindikasikan dominasi aktivitas **parasimpatik**, yang didukung oleh **HF power yang lebih besar dari LF power**.

!(freq_domain_hrv.png)

### **Analisis HRV Non-Linear (Poincaré Plot)**
Poincaré plot menghasilkan nilai:
* **SD1**: **24.83 ms**
* **SD2**: **18.98 ms**

Ini menunjukkan variabilitas jangka pendek yang lebih dominan namun dengan distribusi titik yang terkonsentrasi.

!(poincare_plot.png)

---

### Cara Menambahkan Gambar Hasil

1.  Ambil tangkapan layar dari plot yang Anda inginkan saat aplikasi berjalan.
2.  Simpan gambar-gambar tersebut dalam format yang umum (misalnya, `.png`, `.jpg`) di direktori yang sama dengan file `README.md` ini.
3.  Ubah nama file di dalam tag gambar di atas sesuai dengan nama file gambar yang Anda simpan. Misalnya, jika Anda menyimpan gambar analisis domain waktu sebagai `hrv_time_analysis.png`, ganti `time_domain_hrv.png` menjadi `hrv_time_analysis.png`.
