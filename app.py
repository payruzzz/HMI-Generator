from flask import Flask, render_template, jsonify, request, Response # Pastikan Response di-import
import random
import datetime
import time
import cv2 # Import library OpenCV
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model sekali saat startup
yolo_model = YOLO('yolov8n.pt')  # ponytail: yolov8n (nano) tercepat, upgrade ke yolov8m/yolov8x untuk akurasi lebih tinggi

# --- BAGIAN KAMERA ---
def generate_normal_camera_stream():
    # Membuka Webcam (Gunakan cv2.CAP_DSHOW agar langsung terbuka di Windows)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # Resolusi diturunkan agar stream web tidak lag dan tidak membebani browser
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Gagal membaca frame dari kamera.")
            break

        # Efek cermin agar lebih natural
        frame = cv2.flip(frame, 1)

        # YOLO detection
        results = yolo_model.predict(frame, conf=0.5, verbose=False)
        frame = results[0].plot()  # Draw bounding boxes langsung ke frame

        # Tampilkan teks indikator di layar
        cv2.putText(frame, "YOLO DETECTION MODE", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        # Encode frame OpenCV menjadi format gambar (JPEG)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Gunakan 'yield' untuk mengirim data secara beruntun (streaming MJPEG)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# Endpoint untuk menampilkan stream kamera
@app.route('/video_feed')
def video_feed():
    return Response(generate_normal_camera_stream(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')
# ---------------------

# Variabel status motor global disesuaikan dengan spesifikasi JY Series 1 fasa (0.55 kW)
generator_state = {
    "temp_de": 65.0,
    "temp_nde": 63.0,
    "rpm": 1400,
    "vib": 2.1,
    "volt": 230,
    "curr": 5.1,
    "mode": "AUTO",
    "state": "RUNNING",
    "trip_lock": False,
    "logs": ["[System] Motor JY Series online and ready."]
}

# Variabel untuk melacak Running Time
start_running_time = time.time()
accumulated_runtime = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def get_data():
    if generator_state["state"] == "RUNNING":
        chance = random.random()
        if chance > 0.85:
            generator_state["temp_de"] = round(random.uniform(81.0, 90.0), 1)  # Memicu Trip Temp DE
            generator_state["temp_nde"] = round(random.uniform(81.0, 90.0), 1)  # Memicu Trip Temp NDE
            generator_state["vib"] = round(random.uniform(5.2, 7.0), 1)     # Memicu Trip Vib
        elif chance > 0.70:
            generator_state["temp_de"] = round(random.uniform(77.0, 79.5), 1)  # Memicu Warning Temp DE
            generator_state["temp_nde"] = round(random.uniform(76.0, 79.0), 1)  # Memicu Warning Temp NDE
        else:
            generator_state["temp_de"] = round(random.uniform(50.0, 70.0), 1)
            generator_state["temp_nde"] = round(random.uniform(48.0, 68.0), 1)
            generator_state["vib"] = round(random.uniform(1.0, 3.5), 1)
            
        generator_state["rpm"] = 1400 if random.random() > 0.1 else 1405
        generator_state["volt"] = 230
        generator_state["curr"] = round(random.uniform(5.0, 5.3), 2)
    else:
        generator_state["temp_de"] = round(max(25.0, generator_state["temp_de"] - 2.0), 1)
        generator_state["temp_nde"] = round(max(25.0, generator_state["temp_nde"] - 2.0), 1)
        generator_state["rpm"] = 0
        generator_state["vib"] = 0.0
        generator_state["volt"] = 0
        generator_state["curr"] = 0.0

    return jsonify(generator_state)

# Endpoint baru untuk menghitung dan mengirim Running Time ke Frontend
@app.route('/motor_status')
def motor_status():
    global start_running_time, accumulated_runtime
    
    if generator_state["state"] == "RUNNING":
        current_uptime = accumulated_runtime + (time.time() - start_running_time)
    else:
        current_uptime = accumulated_runtime

    # Format detik menjadi HH:MM:SS
    hours = int(current_uptime // 3600)
    minutes = int((current_uptime % 3600) // 60)
    seconds = int(current_uptime % 60)
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return jsonify({
        "running_time": formatted_time,
        "state": generator_state["state"]
    })

@app.route('/history')
def get_history():
    start_time = request.args.get('start', 'N/A')
    end_time = request.args.get('end', 'N/A')
    
    labels = ["Start", "Q1", "Mid", "Q2", "End"]
    seed_val = len(str(start_time))
    
    temp_de_data = [round(50 + ((i * seed_val) % 15) + random.uniform(-2, 2), 1) for i in range(len(labels))]
    temp_nde_data = [round(48 + ((i * seed_val) % 14) + random.uniform(-2, 2), 1) for i in range(len(labels))]
    rpm_data = [1380 + ((i * seed_val) % 25) for i in range(len(labels))]
    vib_data = [round(2.0 + (i % 3) + random.uniform(0, 0.8), 1) for i in range(len(labels))]

    table_data = [
        {"time": end_time, "temp_de": temp_de_data[-1], "temp_nde": temp_nde_data[-1], "rpm": rpm_data[-1], "vib": vib_data[-1], "status": "NORMAL"},
        {"time": "Mid-point Period", "temp_de": temp_de_data[2], "temp_nde": temp_nde_data[2], "rpm": rpm_data[2], "vib": vib_data[2], "status": "NORMAL"},
        {"time": start_time, "temp_de": temp_de_data[0], "temp_nde": temp_nde_data[0], "rpm": rpm_data[0], "vib": vib_data[0], "status": "NORMAL"}
    ]

    return jsonify({
        "labels": labels,
        "temp_de": temp_de_data,
        "temp_nde": temp_nde_data,
        "rpm": rpm_data,
        "vib": vib_data,
        "table": table_data
    })

@app.route('/control')
def control_generator():
    global start_running_time, accumulated_runtime
    cmd = request.args.get('cmd')
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    
    if cmd == 'start':
        if generator_state.get("trip_lock"):
            generator_state["logs"].insert(0, f"[{timestamp}] START ditolak. Reset alarm terlebih dahulu.")
            return "Blocked: reset alarm first", 403
        if generator_state["state"] != "RUNNING":
            start_running_time = time.time()  # Reset timer hit mulai dari awal
        generator_state["state"] = "RUNNING"
        generator_state["logs"].insert(0, f"[{timestamp}] Motor STARTED by operator.")
    elif cmd == 'stop':
        if generator_state["state"] == "RUNNING":
            accumulated_runtime += (time.time() - start_running_time)  # Simpan akumulasi waktu jalan
        generator_state["state"] = "STOPPED"
        generator_state["logs"].insert(0, f"[{timestamp}] Motor STOPPED.")
    elif cmd == 'trip':
        if generator_state["state"] == "RUNNING":
            accumulated_runtime += (time.time() - start_running_time)
        generator_state["state"] = "STOPPED"
        generator_state["trip_lock"] = True
        generator_state["logs"].insert(0, f"[{timestamp}] AUTO SHUTDOWN: Motor dipaksa mati. Reset alarm sebelum START.")
    elif cmd == 'reset':
        accumulated_runtime = 0
        start_running_time = time.time()
        generator_state["trip_lock"] = False
        generator_state["logs"].insert(0, f"[{timestamp}] Alarm RESET. Running time reset to 00:00:00.")
    elif cmd == 'toggle_mode':
        generator_state["mode"] = "MANUAL" if generator_state["mode"] == "AUTO" else "AUTO"
        generator_state["logs"].insert(0, f"[{timestamp}] Mode changed to {generator_state['mode']}.")
        
    if len(generator_state["logs"]) > 10:
        generator_state["logs"].pop()
        
    return f"Success command: {cmd}"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)