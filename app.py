from flask import (Flask, render_template, jsonify, request, Response,
                   redirect, url_for, flash, session)
from flask_socketio import SocketIO
import random
import datetime
import time
import threading
import os
import secrets
import json
import base64
import binascii
import math
import re
from collections import defaultdict, deque
from functools import wraps

import cv2
import numpy as np
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)

# ---------------- SECRET KEY ----------------
_secret_file = '.flask_secret'
if os.getenv('FLASK_SECRET_KEY'):
    app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY')
else:
    if os.path.exists(_secret_file):
        with open(_secret_file, 'r') as f:
            app.config['SECRET_KEY'] = f.read().strip()
    else:
        _key = secrets.token_hex(32)
        with open(_secret_file, 'w') as f:
            f.write(_key)
        app.config['SECRET_KEY'] = _key


@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Cache-Control'] = 'no-store, max-age=0'
    return response


app.config['UPLOAD_FOLDER'] = 'static/dataset'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MiB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------------- RATE LIMIT ----------------
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX = 100
_rate_buckets = defaultdict(deque)

# Default: hanya same-origin (otomatis cocok untuk localhost maupun IP LAN).
# Kalau perlu origin lain: set env ALLOWED_ORIGINS="http://a:5000,http://b:5000"
_origins_env = os.getenv('ALLOWED_ORIGINS')
_cors = [o.strip() for o in _origins_env.split(',')] if _origins_env else None
socketio = SocketIO(app, cors_allowed_origins=_cors, async_mode='threading')


# ---------------- CSRF ----------------
def generate_csrf_token():
    if '_csrf_token' not in session:
        session['_csrf_token'] = secrets.token_hex(32)
    return session['_csrf_token']


def validate_csrf_token():
    token = request.headers.get('X-CSRF-Token') or request.form.get('csrf_token')
    return bool(token) and token == session.get('_csrf_token')


app.jinja_env.globals['csrf_token'] = generate_csrf_token


def csrf_protect(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if request.method in ('POST', 'PUT', 'DELETE', 'PATCH'):
            if not validate_csrf_token():
                return jsonify({"error": "CSRF token missing or invalid"}), 403
        return f(*args, **kwargs)
    return wrapper


def rate_limit(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        ip = request.remote_addr or 'unknown'
        now = time.time()
        bucket = _rate_buckets[ip]
        while bucket and bucket[0] < now - RATE_LIMIT_WINDOW:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT_MAX:
            return jsonify({"error": "Rate limit exceeded"}), 429
        bucket.append(now)
        return f(*args, **kwargs)
    return wrapper


# ---------------- YOLO ----------------
yolo_model = YOLO('yolov8n.pt')  # model COCO; kelas kustom butuh training sendiri
yolo_lock = threading.Lock()


# ---------------- KAMERA (shared, satu thread untuk semua client) ----------------
class Camera:
    IDLE_TIMEOUT = 5.0  # detik tanpa penonton -> kamera dilepas

    def __init__(self, index=0):
        self.index = index
        self._lock = threading.Lock()
        self._frame = None
        self._thread = None
        self._last_access = 0.0
        self._retry_after = 0.0

    def _run(self):
        backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.index, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        try:
            if not cap.isOpened():
                print("Kamera tidak dapat dibuka.")
                self._retry_after = time.time() + 3
                return
            while time.time() - self._last_access < self.IDLE_TIMEOUT:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                with self._lock:
                    self._frame = frame
        finally:
            cap.release()
            with self._lock:
                self._frame = None
                self._thread = None

    def get_frame(self):
        self._last_access = time.time()
        with self._lock:
            if self._thread is None and time.time() >= self._retry_after:
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()
            return None if self._frame is None else self._frame.copy()


camera = Camera(0)


def _encode_jpeg(frame, quality=85):
    ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


def _mjpeg_chunk(jpg):
    return b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n'


def generate_yolo_stream():
    """Stream untuk dashboard: kamera + deteksi YOLO (mirror agar natural)."""
    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        frame = cv2.flip(frame, 1)
        with yolo_lock:
            results = yolo_model.predict(frame, conf=0.5, verbose=False)
        frame = results[0].plot()
        cv2.putText(frame, "YOLO DETECTION MODE", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        jpg = _encode_jpeg(frame)
        if jpg:
            yield _mjpeg_chunk(jpg)
        time.sleep(0.02)


def generate_raw_stream():
    """Stream mentah (tanpa YOLO, tanpa mirror) untuk halaman dataset."""
    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        jpg = _encode_jpeg(frame)
        if jpg:
            yield _mjpeg_chunk(jpg)
        time.sleep(0.03)


@app.route('/video_feed')
def video_feed():
    return Response(generate_yolo_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/raw_feed')
def raw_feed():
    return Response(generate_raw_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/snapshot')
def snapshot():
    deadline = time.time() + 4
    frame = None
    while time.time() < deadline:
        frame = camera.get_frame()
        if frame is not None:
            break
        time.sleep(0.05)
    if frame is None:
        return jsonify({"error": "Kamera tidak tersedia"}), 503
    jpg = _encode_jpeg(frame, 92)
    if not jpg:
        return jsonify({"error": "Gagal encode gambar"}), 500
    return Response(jpg, mimetype='image/jpeg')


# ---------------- STATE MOTOR & THRESHOLD ----------------
state_lock = threading.RLock()

generator_state = {
    "temp_de": 65.0, "temp_nde": 63.0, "rpm": 1400, "vib": 2.1,
    "volt": 230, "curr": 5.1,
    "mode": "AUTO", "state": "RUNNING", "trip_lock": False,
    "trip_reason": [],
    "logs": ["[System] Motor JY Series online and ready."]
}

THRESHOLD_FILE = 'thresholds.json'
DEFAULT_THRESHOLDS = {"temp_de": 80.0, "temp_nde": 80.0, "rpm": 1500.0,
                      "vib": 5.0, "volt_min": 220.0, "volt_max": 240.0,
                      "curr": 8.0}
thresholds = dict(DEFAULT_THRESHOLDS)
if os.path.exists(THRESHOLD_FILE):
    try:
        with open(THRESHOLD_FILE) as f:
            saved = json.load(f)
        for _k in thresholds:
            if isinstance(saved.get(_k), (int, float)):
                thresholds[_k] = float(saved[_k])
    except Exception as e:
        print(f"Gagal membaca {THRESHOLD_FILE}: {e}")

start_running_time = time.time()
accumulated_runtime = 0.0


def add_log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    generator_state["logs"].insert(0, f"[{ts}] {msg}")
    del generator_state["logs"][10:]


def check_limits(g, t):
    """Kembalikan daftar alasan trip (kosong = aman)."""
    r = []
    if g["temp_de"] > t["temp_de"]:
        r.append(f"Suhu DE Bearing terlalu tinggi: {g['temp_de']:.1f}°C (Limit: {t['temp_de']:g}°C)")
    if g["temp_nde"] > t["temp_nde"]:
        r.append(f"Suhu NDE Bearing terlalu tinggi: {g['temp_nde']:.1f}°C (Limit: {t['temp_nde']:g}°C)")
    if g["rpm"] > t["rpm"]:
        r.append(f"RPM berlebihan (Overspeed): {g['rpm']} RPM (Limit: {t['rpm']:g} RPM)")
    if g["vib"] > t["vib"]:
        r.append(f"Getaran berlebihan: {g['vib']:.1f} mm/s (Limit: {t['vib']:g} mm/s)")
    if g["volt"] < t["volt_min"]:
        r.append(f"Undervoltage: {g['volt']} V (Min: {t['volt_min']:g} V)")
    if g["volt"] > t["volt_max"]:
        r.append(f"Overvoltage: {g['volt']} V (Max: {t['volt_max']:g} V)")
    if g["curr"] > t["curr"]:
        r.append(f"Overcurrent: {g['curr']:.2f} A (Limit: {t['curr']:g} A)")
    return r


def do_trip(reasons):
    global accumulated_runtime
    if generator_state["state"] == "RUNNING":
        accumulated_runtime += time.time() - start_running_time
    generator_state["state"] = "STOPPED"
    generator_state["trip_lock"] = True
    generator_state["trip_reason"] = reasons
    add_log("AUTO SHUTDOWN: Motor dipaksa mati. Reset alarm sebelum START.")


def simulate_sensors():
    g = generator_state
    if g["state"] == "RUNNING":
        r = random.random()
        if r > 0.99:      # ~1%: melewati batas -> trip
            g["temp_de"] = round(random.uniform(81.0, 90.0), 1)
            g["temp_nde"] = round(random.uniform(81.0, 90.0), 1)
            g["vib"] = round(random.uniform(5.2, 7.0), 1)
        elif r > 0.93:    # ~6%: mendekati batas -> warning
            g["temp_de"] = round(random.uniform(77.0, 79.5), 1)
            g["temp_nde"] = round(random.uniform(76.0, 79.0), 1)
            g["vib"] = round(random.uniform(4.3, 4.8), 1)
        else:
            g["temp_de"] = round(random.uniform(50.0, 70.0), 1)
            g["temp_nde"] = round(random.uniform(48.0, 68.0), 1)
            g["vib"] = round(random.uniform(1.0, 3.5), 1)
        g["rpm"] = 1400 if random.random() > 0.1 else 1405
        g["volt"] = 230
        g["curr"] = round(random.uniform(5.0, 5.3), 2)
    else:
        g["temp_de"] = round(max(25.0, g["temp_de"] - 2.0), 1)
        g["temp_nde"] = round(max(25.0, g["temp_nde"] - 2.0), 1)
        g["rpm"] = 0
        g["vib"] = 0.0
        g["volt"] = 0
        g["curr"] = 0.0


def snapshot_state():
    g = generator_state
    return {
        "temp_de": g["temp_de"], "temp_nde": g["temp_nde"], "rpm": g["rpm"],
        "vib": g["vib"], "volt": g["volt"], "curr": g["curr"],
        "mode": g["mode"], "state": g["state"], "trip_lock": g["trip_lock"],
        "trip_reason": list(g["trip_reason"]),
        "timestamp": datetime.datetime.now().isoformat()
    }


def emit_motor_data():
    """Update sensor, cek proteksi di SERVER, lalu broadcast tiap 500 ms."""
    while True:
        try:
            with state_lock:
                simulate_sensors()
                if generator_state["state"] == "RUNNING":
                    reasons = check_limits(generator_state, thresholds)
                    if reasons:
                        do_trip(reasons)
                data = snapshot_state()
            socketio.emit('motor_data', data)
        except Exception as e:
            print(f"Error emitting data: {e}")
        socketio.sleep(0.5)


socketio.start_background_task(emit_motor_data)


# ---------------- ROUTES: DASHBOARD ----------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/data')
def get_data():
    with state_lock:
        return jsonify(snapshot_state())


@app.route('/motor_status')
def motor_status():
    with state_lock:
        if generator_state["state"] == "RUNNING":
            uptime = accumulated_runtime + (time.time() - start_running_time)
        else:
            uptime = accumulated_runtime
        state = generator_state["state"]
    h = int(uptime // 3600)
    m = int((uptime % 3600) // 60)
    s = int(uptime % 60)
    return jsonify({"running_time": f"{h:02d}:{m:02d}:{s:02d}", "state": state})


@app.route('/history')
def get_history():
    start_time = request.args.get('start')
    end_time = request.args.get('end')
    if start_time and end_time:
        # TODO: ganti dengan data riil dari Firebase bila sudah siap.
        demo_table = [
            {"Date": start_time, "Time": "00:00:00", "temp_de": 50.0, "temp_nde": 48.0, "rpm": 1380, "vib": 2.1, "status": "NORMAL"},
            {"Date": "Mid-period", "Time": "12:00:00", "temp_de": 60.0, "temp_nde": 58.0, "rpm": 1400, "vib": 3.0, "status": "NORMAL"},
            {"Date": end_time, "Time": "23:59:59", "temp_de": 62.0, "temp_nde": 60.0, "rpm": 1410, "vib": 3.3, "status": "NORMAL"}
        ]
        return jsonify({
            "real_data": False, "demo": True,
            "labels": ["Start", "Q1", "Mid", "Q2", "End"],
            "temp_de": [50.0, 55.0, 60.0, 65.0, 62.0],
            "temp_nde": [48.0, 53.0, 58.0, 63.0, 60.0],
            "rpm": [1380, 1400, 1420, 1390, 1410],
            "vib": [2.1, 2.6, 3.1, 3.8, 3.3],
            "table": demo_table,
            "message": "Data placeholder (demo). Ganti ke data riil dari Firebase."
        })
    return jsonify({
        "real_data": False, "demo": False, "labels": [], "temp_de": [],
        "temp_nde": [], "rpm": [], "vib": [], "table": [],
        "message": "Pilih rentang waktu dahulu"
    })


@app.route('/control', methods=['GET', 'POST'])
@csrf_protect
@rate_limit
def control_generator():
    global start_running_time, accumulated_runtime
    if request.method == 'GET':
        return jsonify({"error": "Gunakan POST untuk perintah kontrol"}), 405
    body = request.get_json(silent=True) or {}
    cmd = body.get('cmd') if request.is_json else request.form.get('cmd')
    if not cmd or not isinstance(cmd, str):
        return jsonify({"error": "Parameter cmd diperlukan"}), 400

    with state_lock:
        if cmd == 'start':
            if generator_state["trip_lock"]:
                add_log("START ditolak. Reset alarm terlebih dahulu.")
                return jsonify({"error": "Reset alarm terlebih dahulu"}), 403
            if generator_state["state"] != "RUNNING":
                start_running_time = time.time()
            generator_state["state"] = "RUNNING"
            add_log("Motor STARTED by operator.")
        elif cmd == 'stop':
            if generator_state["state"] == "RUNNING":
                accumulated_runtime += time.time() - start_running_time
            generator_state["state"] = "STOPPED"
            add_log("Motor STOPPED.")
        elif cmd == 'reset':
            accumulated_runtime = 0.0
            start_running_time = time.time()
            generator_state["trip_lock"] = False
            generator_state["trip_reason"] = []
            add_log("Alarm RESET. Running time reset to 00:00:00.")
        elif cmd == 'toggle_mode':
            generator_state["mode"] = "MANUAL" if generator_state["mode"] == "AUTO" else "AUTO"
            add_log(f"Mode changed to {generator_state['mode']}.")
        else:
            return jsonify({"error": "Perintah tidak dikenal"}), 400
    return jsonify({"ok": True, "message": f"Perintah '{cmd}' diproses."})


@app.route('/get_thresholds')
def get_thresholds():
    with state_lock:
        return jsonify(thresholds)


@app.route('/save_thresholds', methods=['POST'])
@csrf_protect
@rate_limit
def save_thresholds():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "No data"}), 400
    params = {}
    for k in DEFAULT_THRESHOLDS:
        if k in data:
            v = data[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) \
                    or not math.isfinite(v) or v <= 0:
                return jsonify({"error": f"Nilai '{k}' tidak valid"}), 400
            params[k] = float(v)
    if not params:
        return jsonify({"error": "No valid params"}), 400
    with state_lock:
        new = {**thresholds, **params}
        if new["volt_min"] >= new["volt_max"]:
            return jsonify({"error": "volt_min harus < volt_max"}), 400
        thresholds.update(params)
        try:
            with open(THRESHOLD_FILE, 'w') as f:
                json.dump(thresholds, f, indent=2)
        except OSError as e:
            print(f"Gagal menyimpan threshold: {e}")
        add_log("Threshold diperbarui.")
    return jsonify({"ok": True, "thresholds": params})


# ---------------- DATASET ----------------
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
CLASSES_FILE = 'dataset_classes.json'
DEFAULT_CLASSES = ['motor', 'LEMARI', 'panel', 'kabel']
_classes_lock = threading.Lock()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def list_images():
    folder = app.config['UPLOAD_FOLDER']
    return sorted(f for f in os.listdir(folder)
                  if f.rsplit('.', 1)[-1].lower() in ALLOWED_EXTENSIONS)


def clean_label(label):
    label = re.sub(r'[\x00-\x1f\x7f]', '', str(label or '')).strip()[:50]
    return label or 'object'


def load_classes():
    if os.path.exists(CLASSES_FILE):
        try:
            with open(CLASSES_FILE) as f:
                data = json.load(f)
            if isinstance(data, list):
                return [str(c) for c in data]
        except Exception:
            pass
    return []


def get_class_id(label):
    """ID kelas dipegang server -> konsisten antar sesi (urutan = ID YOLO)."""
    with _classes_lock:
        classes = load_classes()
        if label not in classes:
            classes.append(label)
            with open(CLASSES_FILE, 'w') as f:
                json.dump(classes, f, indent=2)
        return classes.index(label)


def decodes_as_image(raw_bytes):
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR) is not None


@app.route('/dataset')
def dataset_page():
    images = []
    try:
        images = list_images()
    except Exception as e:
        flash(f"Error reading dataset folder: {e}", "danger")
    classes = load_classes()
    for c in DEFAULT_CLASSES:
        if c not in classes:
            classes.append(c)
    return render_template('dataset.html', images=images, classes=classes)


@app.route('/add_class', methods=['POST'])
@csrf_protect
@rate_limit
def add_class():
    data = request.get_json(silent=True) or {}
    label = clean_label(data.get('label'))
    return jsonify({"success": True, "label": label, "id": get_class_id(label)})


@app.route('/upload_dataset', methods=['POST'])
@csrf_protect
@rate_limit
def upload_dataset():
    files = request.files.getlist('images')
    if not files:
        flash('No file part in request.', "danger")
        return redirect(url_for('dataset_page'))
    saved, errors = [], []
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, f in enumerate(files):
        if not f or not f.filename:
            continue
        if not allowed_file(f.filename):
            errors.append(f"{f.filename}: Invalid extension")
            continue
        raw = f.read()
        if not decodes_as_image(raw):
            errors.append(f"{f.filename}: bukan gambar yang valid")
            continue
        base, ext = os.path.splitext(secure_filename(f.filename))
        if not base or not ext:  # nama non-ASCII -> secure_filename bisa kosong
            base = f"upload_{stamp}_{i}"
            ext = '.' + f.filename.rsplit('.', 1)[-1].lower()
        filename = base + ext
        counter = 1
        while os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
            filename = f"{base}_{counter}{ext}"
            counter += 1
        with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'wb') as out:
            out.write(raw)
        saved.append(filename)
    if saved:
        flash(f'Saved {len(saved)} image(s): {", ".join(saved)}', "success")
    if errors:
        flash(f'Errors: {"; ".join(errors)}', "warning")
    if not saved and not errors:
        flash('No valid images uploaded.', "warning")
    return redirect(url_for('dataset_page'))


@app.route('/delete_dataset_image/<filename>', methods=['POST'])
@csrf_protect
@rate_limit
def delete_dataset_image(filename):
    safe = secure_filename(filename)
    folder = app.config['UPLOAD_FOLDER']
    path = os.path.join(folder, safe)
    if not safe or not allowed_file(safe) or \
            not os.path.abspath(path).startswith(os.path.abspath(folder)):
        flash('Invalid path', "danger")
        return redirect(url_for('dataset_page'))
    if os.path.isfile(path):
        os.remove(path)
        ann = os.path.splitext(path)[0] + '.txt'  # hapus anotasi yatim
        if os.path.isfile(ann):
            os.remove(ann)
        flash(f'Deleted {safe}', "success")
    else:
        flash('File not found', "warning")
    return redirect(url_for('dataset_page'))


@app.route('/api/dataset/images')
def api_dataset_images():
    try:
        return jsonify(list_images())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/capture_dataset', methods=['POST'])
@csrf_protect
@rate_limit
def capture_dataset():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or 'image' not in data:
        return jsonify({"error": "No image data"}), 400

    img_str = str(data['image'])
    img_str = img_str.split(',', 1)[1] if ',' in img_str else img_str
    if len(img_str) > 5 * 1024 * 1024:
        return jsonify({"error": "Image too large (max 5 MiB)"}), 413
    try:
        img_bytes = base64.b64decode(img_str, validate=True)
    except (binascii.Error, ValueError) as e:
        return jsonify({"error": f"Invalid base64: {e}"}), 400
    if not decodes_as_image(img_bytes):
        return jsonify({"error": "Data bukan gambar yang valid"}), 400

    # Validasi anotasi SEBELUM menulis file apa pun
    lines = []
    boxes = data.get('boxes') or []
    if not isinstance(boxes, list):
        return jsonify({"error": "Format box tidak valid"}), 400
    try:
        for box in boxes:
            label = clean_label(box.get('label'))
            x, y, w, h = (min(max(float(box[k]), 0.0), 1.0) for k in ('x', 'y', 'w', 'h'))
            lines.append((label, x, y, w, h))
    except (KeyError, ValueError, TypeError, AttributeError):
        return jsonify({"error": "Format box tidak valid"}), 400

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"capture_{timestamp}.jpg"
    folder = app.config['UPLOAD_FOLDER']
    with open(os.path.join(folder, filename), 'wb') as f:
        f.write(img_bytes)

    if lines:
        # Format YOLO: class x_center y_center width height (ternormalisasi 0-1)
        with open(os.path.join(folder, f"capture_{timestamp}.txt"), 'w') as f:
            for label, x, y, w, h in lines:
                f.write(f"{get_class_id(label)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    return jsonify({"success": True, "filename": filename})


if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1', 'yes')
    # use_reloader=False: agar model YOLO & thread tidak dimuat dua kali
    socketio.run(app, debug=debug_mode, use_reloader=False, host='0.0.0.0',
                 port=5000, allow_unsafe_werkzeug=True)