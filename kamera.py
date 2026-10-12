import cv2

def generate_normal_camera_stream():
    # Membuka Webcam (Gunakan cv2.CAP_DSHOW agar langsung terbuka di Windows)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # 1. Atur resolusi ke format Widescreen (16:9)
    # Gunakan 1920x1080 (Full HD) atau 1280x720 (HD) agar tampilannya melebar/wide.
    # Hindari 4000x3000 karena itu rasio 4:3 (kotak).
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Gagal membaca frame dari kamera.")
            break

        # Efek cermin agar lebih natural
        frame = cv2.flip(frame, 1)

        # ---------------------------------------------------------
        # 2. OPSI ULTRA-WIDE (Hapus tanda pagar '#' di bawah ini jika ingin efek lebih lebar)
        # Ini akan memotong (crop) bagian atas dan bawah gambar sebesar 15%
        # 
        # height, width, _ = frame.shape
        # crop_margin = int(height * 0.15) 
        # frame = frame[crop_margin:height-crop_margin, 0:width]
        # ---------------------------------------------------------

        # Tampilkan teks indikator di layar
        cv2.putText(frame, "WIDESCREEN CAMERA MODE", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        # Encode frame OpenCV menjadi format gambar (JPEG)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Gunakan 'yield' untuk mengirim data secara beruntun (streaming MJPEG)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()