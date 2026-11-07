# HMI Generator – Motor Monitoring System

Dashboard HMI berbasis web untuk memantau **Motor 1 – Generator (Induction Motor)** secara real-time. Sistem menampilkan suhu bearing, getaran, RPM, tegangan, dan arus, menjalankan proteksi otomatis (trip) di sisi server, serta dilengkapi kamera dengan deteksi objek **YOLOv8** dan pengelola dataset untuk melatih model sendiri.

> **Status:** prototipe. Data sensor masih **simulasi** (belum terhubung ke sensor/PLC sungguhan). Lihat bagian [Status & Keterbatasan](#status--keterbatasan).

## Fitur

**Monitoring real-time**
- Suhu bearing DE & NDE, getaran, RPM, tegangan, arus, dan running time
- Data dikirim server ke browser tiap **1 detik** lewat Socket.IO
- Grafik tren suhu, RPM, dan getaran
- Status kondisi bearing dan kartu berwarna: Normal, Warning, Trip

**Proteksi & kontrol**
- Pengecekan batas (threshold) dilakukan di **server**: overtemperature DE/NDE, overspeed, getaran, undervoltage/overvoltage, overcurrent
- Saat melewati batas, motor otomatis **trip** dan terkunci sampai alarm di-reset
- Tombol **Start**, **Stop**, **Reset Alarm**, dan ganti mode AUTO/MANUAL
- Threshold dapat diubah lewat panel pengaturan dan disimpan ke `thresholds.json`
- Modal notifikasi trip yang menampilkan semua penyebab

**Riwayat & ekspor**
- Tabel **History** (rentang waktu) dan tabel **Live** tampil bersamaan
- Grafik gabungan DE / NDE / RPM×10 / Vibration
- Ekspor ke **Excel** dan **PDF**

**Kamera & YOLO**
- Live camera dengan deteksi objek YOLOv8n (model COCO) di dashboard
- **Image Dataset Manager** (`/dataset`): ambil gambar dari kamera server, gambar bounding box, simpan dalam **format YOLO** (`class x_center y_center width height`, ternormalisasi 0–1), upload gambar yang sudah ada, dan hapus gambar
- ID kelas dipegang server (`dataset_classes.json`) sehingga konsisten antar sesi

**Keamanan dasar**
- Proteksi CSRF pada semua request yang mengubah data
- Rate limiting per IP (100 request/menit)
- Header keamanan (`X-Frame-Options`, `nosniff`, dll.) dan CORS same-origin secara default

## Teknologi

| Bagian | Teknologi |
|---|---|
| Backend | Flask 3, Flask-SocketIO |
| Computer vision | OpenCV, Ultralytics YOLOv8 |
| Frontend | HTML, CSS, JavaScript, Chart.js, Socket.IO client |
| Penyimpanan cloud (sisi browser) | Firebase Realtime Database |
| Ekspor | SheetJS (Excel), jsPDF (PDF) |

## Struktur Proyek

```
HMI-Generator/
├── app.py               # Server Flask: route, Socket.IO, proteksi, kamera, YOLO, dataset
├── kamera.py            # Skrip kamera lama (tidak dipakai oleh app.py)
├── signal_engine.py     # Engine support/resistance (monitoring saja, belum terhubung ke HMI)
├── requirements.txt
├── yolov8n.pt           # Bobot YOLOv8 nano (COCO)
├── templates/
│   ├── index.html       # Dashboard utama
│   └── dataset.html     # Image Dataset Manager
└── static/dataset/      # Gambar & label YOLO hasil capture/upload
```

File yang dibuat otomatis saat aplikasi berjalan: `.flask_secret`, `thresholds.json`, `dataset_classes.json`.

## Instalasi

Prasyarat: **Python 3.10–3.12** dan webcam yang terpasang di komputer yang menjalankan server.

```bash
git clone https://github.com/payruzzz/HMI-Generator.git
cd HMI-Generator

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

## Menjalankan

```bash
python app.py
```

Buka di browser:

- Dashboard: <http://localhost:5000>
- Dataset Manager: <http://localhost:5000/dataset>
- Dari perangkat lain di jaringan yang sama: `http://<IP-server>:5000`

Kamera dibuka oleh **server** (indeks kamera `0`), bukan oleh browser. Karena itu halaman dataset bisa dipakai dari perangkat mana pun tanpa HTTPS.

### Variabel lingkungan (opsional)

| Variabel | Fungsi |
|---|---|
| `FLASK_SECRET_KEY` | Secret key session. Jika kosong, dibuat otomatis dan disimpan di `.flask_secret` |
| `ALLOWED_ORIGINS` | Daftar origin CORS dipisah koma, contoh `http://a:5000,http://b:5000` |
| `FLASK_DEBUG` | `true` untuk mode debug |

## Cara Memakai Dataset Manager

1. Buka `/dataset`. Tampilan kamera langsung muncul.
2. Isi **Class Label** (misalnya `motor`, `panel`) lalu klik **Capture Frame**.
3. Gambar bounding box pada hasil capture. **Undo Box** menghapus box terakhir.
4. Klik **Save to Dataset**. Gambar `.jpg` dan label `.txt` tersimpan di `static/dataset/`.
5. Gunakan **New Capture** untuk mengambil gambar berikutnya.

Contoh isi file label:

```
0 0.512345 0.430000 0.210000 0.380000
```

## Endpoint Utama

| Method | Endpoint | Keterangan |
|---|---|---|
| GET | `/` | Dashboard |
| GET | `/data` | Snapshot data motor (JSON) |
| GET | `/motor_status` | Status dan running time |
| GET | `/history?start=&end=` | Data riwayat (saat ini data demo) |
| POST | `/control` | Perintah `start`, `stop`, `reset`, `toggle_mode` |
| GET / POST | `/get_thresholds`, `/save_thresholds` | Baca/simpan threshold |
| GET | `/video_feed` | Stream kamera dengan deteksi YOLO |
| GET | `/raw_feed` | Stream kamera mentah |
| GET | `/snapshot` | Satu frame JPEG dari kamera |
| GET | `/dataset` | Halaman Dataset Manager |
| POST | `/capture_dataset` | Simpan gambar + anotasi |
| POST | `/upload_dataset` | Upload gambar |
| POST | `/delete_dataset_image/<file>` | Hapus gambar dan labelnya |

Semua request POST memerlukan token CSRF.

## Status & Keterbatasan

- Nilai sensor dihasilkan oleh `simulate_sensors()` di `app.py` dan belum berasal dari perangkat keras.
- Endpoint `/history` masih mengembalikan data placeholder; pembacaan riwayat riil dari Firebase belum dibuat.
- Interlock relay, alarm audio, dan fail-safe sensor belum diimplementasikan, karena proteksi saat ini hanya berupa logika perangkat lunak.
- Model `yolov8n.pt` adalah model COCO bawaan. Untuk mendeteksi objek khusus, latih model dengan dataset dari Dataset Manager.
- `signal_engine.py` adalah modul terpisah dan belum dipakai oleh aplikasi.

## Catatan Keamanan

- Aplikasi berjalan di `0.0.0.0` **tanpa autentikasi pengguna**. Jangan diekspos ke internet.
- Jangan commit `.flask_secret` ke repositori. Tambahkan ke `.gitignore`:

  ```
  .flask_secret
  __pycache__/
  thresholds.json
  dataset_classes.json
  ```

- Konfigurasi Firebase pada `templates/index.html` terlihat di sisi klien. Pastikan aturan (rules) database Anda dibatasi.

## Lisensi

Belum ditentukan. Tambahkan berkas `LICENSE` sesuai kebutuhan.
