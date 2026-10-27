# Discord Ecosystem 2026 — Machine Learning Dataset

## 1. Deskripsi Proyek

Proyek ini membahas penerapan Machine Learning untuk menganalisis repositori GitHub yang berkaitan dengan ekosistem Discord. Dataset mencakup informasi repositori, pemilik, bahasa pemrograman, lisensi, aktivitas pembaruan, serta indikator popularitas dan pengembangan proyek.

Fokus analisis adalah pemilihan algoritma regresi berdasarkan karakteristik data, ukuran dataset, dan pola hubungan antarvariabel. Model regresi dapat diuji untuk memprediksi jumlah `stars` berdasarkan informasi repositori yang tersedia.

Evaluasi model menggunakan Mean Absolute Error (MAE), Mean Squared Error (MSE), dan Coefficient of Determination (R²). Analisis klasifikasi tingkat popularitas dapat dikembangkan sebagai eksperimen tambahan.

## 2. Dataset

Dataset utama: [`discord_ecosystem_2026.csv`](https://github.com/Fhmsdm/Dataset-discord_ecosystem_2026-/blob/main/discord_ecosystem_2026.csv)

Notebook yang tersedia:
- [Notebook utama](https://github.com/Fhmsdm/Dataset-discord_ecosystem_2026-/blob/main/discord_ecosystem_2026.ipynb)
- [Notebook dengan penjelasan](https://github.com/Fhmsdm/Dataset-discord_ecosystem_2026-/blob/main/discord_ecosystem_2026_dengan_penjelasan.ipynb)

CSV memiliki 25 kolom. Variabel penting yang tersedia antara lain:

| Kolom | Penjelasan |
|---|---|
| `id` | Identitas repositori di GitHub |
| `name` | Nama repositori |
| `full_name` | Nama lengkap repositori beserta pemilik |
| `owner` | Nama akun pemilik |
| `owner_type` | Jenis pemilik, misalnya User atau Organization |
| `is_official_discord` | Penanda repositori organisasi resmi Discord |
| `description` | Deskripsi singkat repositori |
| `stars` | Jumlah stars repositori |
| `forks` | Jumlah fork repositori |
| `watchers` | Jumlah watchers yang tercatat |
| `open_issues` | Jumlah isu terbuka |
| `stars_per_day` | Nilai stars per hari yang tersedia pada dataset |
| `language` | Bahasa pemrograman utama |
| `license` | Lisensi repositori |
| `topics` | Daftar topik repositori |
| `queried_topic` | Topik pencarian yang digunakan |
| `created_at` | Waktu pembuatan repositori |
| `updated_at` | Waktu pembaruan metadata terakhir |
| `pushed_at` | Waktu push terakhir |
| `repo_age_days` | Usia repositori dalam hari |
| `days_since_last_push` | Jarak hari sejak push terakhir |
| `size_kb` | Ukuran repositori dalam KB |
| `is_fork` | Penanda apakah repositori merupakan fork |
| `archived` | Penanda apakah repositori diarsipkan |
| `html_url`, `homepage` | Tautan repositori dan situs terkait |

Definisi kolom tersebut mengikuti nama dan contoh isi CSV. Tipe data, nilai kosong, serta cara pembentukan setiap kolom perlu diperiksa saat analisis.

## 3. Landasan Teori

### 3.1 Machine Learning

Machine Learning adalah pendekatan komputasi yang memungkinkan sistem mempelajari pola dari data untuk menghasilkan prediksi atau keputusan. Dalam proyek ini, Machine Learning digunakan untuk mempelajari hubungan antara karakteristik repositori GitHub dan indikator popularitasnya.

### 3.2 Regresi dalam Machine Learning

Regresi merupakan metode *supervised learning* untuk memprediksi nilai numerik berdasarkan satu atau lebih variabel input.

**Y = f(X) + ε**

Keterangan:
- `Y`: variabel target yang ingin diprediksi.
- `X`: satu atau lebih variabel input.
- `f(X)`: hubungan yang dipelajari model.
- `ε`: komponen kesalahan atau variasi yang tidak dijelaskan model.

Dalam dataset ini, `stars` dapat dipilih sebagai target prediksi. Fitur harus dipilih dengan hati-hati agar tidak membocorkan nilai target.

### 3.3 Faktor Pemilihan Algoritma Regresi

**A. Jenis dan karakteristik data**

Dataset mencakup data numerik, kategorikal, tanggal, teks, dan Boolean. Fitur seperti `language`, `license`, `owner_type`, serta `queried_topic` perlu diproses sesuai algoritma. Kolom teks panjang dan identitas repositori tidak otomatis digunakan sebagai fitur.

**B. Ukuran dataset**

Jumlah sampel dan fitur memengaruhi waktu pelatihan, penggunaan memori, dan kompleksitas model. Linear Regression dapat menjadi baseline yang efisien, sedangkan model ensemble dapat diuji ketika pola data lebih kompleks. Ukuran dataset saja tidak menentukan algoritma terbaik.

**C. Linearitas hubungan antarvariabel**

Jika hubungan fitur dan target cenderung linear, Linear Regression dapat menjadi pilihan awal. Jika pola lebih kompleks atau nonlinier, Random Forest Regression dan Gradient Boosting/XGBoost dapat diuji.

Pemilihan model juga perlu mempertimbangkan interpretabilitas, waktu pelatihan, waktu prediksi, dan kemampuan generalisasi.

### 3.4 Algoritma Regresi

| Algoritma | Karakteristik dan kegunaan |
|---|---|
| Linear Regression | Model dasar untuk hubungan linear dan interpretasi koefisien |
| Ridge Regression | Regresi linear dengan regularisasi L2 |
| Lasso Regression | Regularisasi L1 yang dapat membuat sebagian koefisien menjadi nol |
| Elastic Net | Menggabungkan regularisasi L1 dan L2 |
| Polynomial Regression | Memodelkan hubungan melengkung dengan fitur polinomial |
| Decision Tree Regression | Menangkap pola nonlinier melalui aturan berbentuk pohon |
| Random Forest Regression | Menggabungkan banyak pohon keputusan |
| Gradient Boosting Regression | Membangun model bertahap untuk memperbaiki kesalahan |
| XGBoost Regression | Implementasi gradient boosting yang fleksibel dan efisien |
| Support Vector Regression (SVR) | Memodelkan hubungan linear maupun nonlinier menggunakan support vector dan kernel |
| K-Nearest Neighbors Regression | Prediksi berdasarkan target dari sampel berdekatan |
| Neural Network Regression | Mempelajari hubungan kompleks menggunakan jaringan neuron |

Tidak semua algoritma harus digunakan sekaligus. Kandidat model dipilih berdasarkan karakteristik data, sumber daya komputasi, dan tujuan prediksi.

### 3.5 Evaluasi Model Regresi

**A. Mean Absolute Error (MAE)**

MAE menghitung rata-rata selisih absolut prediksi dan nilai aktual. MAE yang lebih rendah menunjukkan kesalahan absolut yang lebih kecil. Nilainya mengikuti skala target dan tidak harus kurang dari satu.

**B. Mean Squared Error (MSE)**

MSE menghitung rata-rata kuadrat selisih prediksi dan nilai aktual. Nilai lebih rendah menunjukkan kesalahan kuadrat yang lebih kecil. MSE memberi penalti lebih besar terhadap kesalahan besar dan satuannya merupakan kuadrat dari target.

**C. Coefficient of Determination (R²)**

R² mengukur kemampuan model menjelaskan variasi target pada data evaluasi.

- `R² = 1`: prediksi sempurna pada data evaluasi.
- `R²` mendekati 1: model menjelaskan variasi target dengan baik.
- `R² = 0`: performa setara dengan baseline rata-rata target.
- `R² < 0`: performa lebih buruk daripada baseline tersebut.

R² dapat bernilai negatif. MAE dan MSE tidak harus di bawah satu karena bergantung pada skala `stars`. Ketiga metrik perlu dianalisis bersama.

## 4. Metodologi Penelitian

Tahapan penelitian yang direncanakan:

1. **Data Understanding:** memahami struktur dataset dan definisi variabel.
2. **Data Cleaning:** memeriksa nilai hilang, duplikasi, tipe data, dan nilai tidak wajar.
3. **Exploratory Data Analysis (EDA):** menganalisis distribusi, korelasi, dan pola antarvariabel.
4. **Feature Selection:** memilih fitur relevan dan mencegah kebocoran informasi target.
5. **Data Splitting:** membagi data menjadi training dan testing, misalnya 80% dan 20%.
6. **Preprocessing:** mengisi nilai kosong serta melakukan encoding fitur kategorikal.
7. **Model Training:** melatih algoritma regresi yang dipilih.
8. **Model Evaluation:** membandingkan MAE, MSE, R², dan bila diperlukan waktu pelatihan serta prediksi.
9. **Model Comparison:** menentukan model yang sesuai berdasarkan hasil evaluasi.

**Catatan data leakage:** `stars_per_day` berpotensi dihitung dari `stars` dan usia repositori. Jika targetnya `stars`, kolom `stars_per_day` sebaiknya tidak digunakan sebagai fitur kecuali rumusnya sudah diperiksa dan penggunaannya dapat dibenarkan. Kolom identitas seperti `id`, `name`, `full_name`, dan URL juga perlu dipertimbangkan dengan hati-hati.

Kolom `topics` berisi beberapa topik dalam satu kolom sehingga memerlukan strategi preprocessing tersendiri jika akan digunakan sebagai fitur. Kolom tanggal mentah dapat diubah menjadi fitur yang relevan jika diperlukan.

## 5. Rencana Eksperimen

Eksperimen dirancang untuk menjawab pertanyaan berikut:

1. Bagaimana karakteristik dataset Discord Ecosystem 2026 memengaruhi pemilihan algoritma regresi?
2. Bagaimana perbedaan performa model linear dan nonlinier dalam memprediksi `stars`?
3. Algoritma mana yang menghasilkan MAE dan MSE terendah serta R² terbaik?
4. Bagaimana preprocessing fitur kategorikal memengaruhi hasil prediksi?
5. Bagaimana kompleksitas model memengaruhi waktu pelatihan dan prediksi?
6. Algoritma mana yang paling sesuai berdasarkan akurasi, interpretabilitas, dan kebutuhan komputasi?

Target eksperimen utama yang diusulkan adalah `stars`. Model perlu dibandingkan pada data uji yang sama. Distribusi target dan pengaruh nilai ekstrem harus diperiksa sebelum menarik kesimpulan.

## 6. Landasan Penelitian dan Referensi

Referensi berikut dapat digunakan untuk memperkuat konsep metode dan evaluasi. Periksa detail bibliografi dan kesesuaian isi artikel sebelum menjadikannya dasar klaim spesifik dalam laporan akademik.

### 6.1 Uddin dan Lu (2024)

**Judul:** *Dataset meta-level and statistical features affect machine learning performance.*

Artikel ini membahas hubungan karakteristik statistik dataset dengan performa algoritma Machine Learning. Relevansinya adalah pentingnya memeriksa jumlah sampel, jumlah fitur, dan karakteristik data sebelum memilih model.

DOI: https://doi.org/10.1038/s41598-024-51825-x

### 6.2 Dokumentasi Scikit-learn

- Evaluasi model: https://scikit-learn.org/stable/modules/model_evaluation.html
- Model linear: https://scikit-learn.org/stable/modules/linear_model.html
- Random Forest dan ensemble: https://scikit-learn.org/stable/modules/ensemble.html

### 6.3 Dokumentasi XGBoost

Dokumentasi resmi untuk implementasi model XGBoost Regression:

https://xgboost.readthedocs.io/

## 7. Dokumentasi Dataset

- Repository: https://github.com/Fhmsdm/Dataset-discord_ecosystem_2026-
- CSV: https://github.com/Fhmsdm/Dataset-discord_ecosystem_2026-/blob/main/discord_ecosystem_2026.csv
- Notebook utama: https://github.com/Fhmsdm/Dataset-discord_ecosystem_2026-/blob/main/discord_ecosystem_2026.ipynb
- Notebook dengan penjelasan: https://github.com/Fhmsdm/Dataset-discord_ecosystem_2026-/blob/main/discord_ecosystem_2026_dengan_penjelasan.ipynb

## 8. Konsep Dasar Klasifikasi

Dataset ini juga dapat digunakan untuk eksperimen klasifikasi tingkat popularitas repositori. Kategori dapat dibentuk dari jumlah `stars`, misalnya rendah, sedang, dan tinggi. Batas kategori harus ditentukan berdasarkan distribusi data atau aturan penelitian yang dijelaskan secara eksplisit.

Algoritma yang dapat dipertimbangkan:
- **Decision Tree Classifier:** membuat aturan klasifikasi berbasis pohon.
- **Random Forest Classifier:** menggabungkan beberapa pohon keputusan.
- **K-Nearest Neighbors (KNN):** menentukan kelas berdasarkan tetangga terdekat.
- **Naive Bayes:** mengklasifikasikan data berdasarkan probabilitas dengan preprocessing yang sesuai.

Evaluasi klasifikasi dapat menggunakan accuracy, precision, recall, dan F1-score. Regresi dan klasifikasi adalah eksperimen berbeda sehingga target, preprocessing, dan metriknya harus dijelaskan secara terpisah.

## 9. Kesimpulan

Pemilihan algoritma regresi pada dataset Discord Ecosystem 2026 perlu mempertimbangkan jenis dan karakteristik data, ukuran dataset, serta linearitas hubungan antarvariabel. Linear Regression dapat menjadi baseline, sedangkan Ridge, Lasso, Random Forest, Gradient Boosting, dan XGBoost dapat diuji sesuai kebutuhan.

Model terbaik tidak dapat ditentukan hanya berdasarkan teori. Penentuan akhir dilakukan melalui eksperimen yang konsisten dengan membandingkan MAE, MSE, dan R² pada data uji. Jika klasifikasi popularitas dilakukan, hasilnya dilaporkan sebagai eksperimen terpisah dengan metrik klasifikasi.

## 10. Status Proyek

Dokumentasi ini merupakan landasan teori dan rencana eksperimen untuk dataset Discord Ecosystem 2026. Hasil performa dan klaim algoritma terbaik akan ditambahkan setelah pengujian model benar-benar dilakukan pada dataset.

## Penulis

**Fahmi Saddam**
