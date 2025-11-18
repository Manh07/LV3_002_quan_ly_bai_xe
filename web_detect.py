from flask import Flask, render_template, Response, jsonify
import cv2
import pytesseract
import imutils
import numpy as np
import re
import time
from pyzbar import pyzbar
from datetime import datetime

app = Flask(__name__)

# Cấu hình Tesseract
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
custom_config = (
    r"--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-"
)

# Biến toàn cục
last_plate = {"text": "", "time": 0}
last_qr = {"data": "", "time": 0}
detection_history = []
PROCESS_INTERVAL = 0.5
QR_COOLDOWN = 1.0

# Khởi tạo camera
camera = cv2.VideoCapture(0)
camera.set(3, 640)
camera.set(4, 480)


def process_frame(frame):
    """Xử lý frame: nhận diện biển số và QR code"""
    global last_plate, last_qr, detection_history

    current_time = time.time()
    display = frame.copy()

    # ============ QUÉT QR CODE ============
    qr_codes = pyzbar.decode(frame)
    qr_detected = False

    if qr_codes:
        qr_detected = True
        for qr in qr_codes:
            qr_data = qr.data.decode("utf-8")
            qr_type = qr.type
            (qx, qy, qw, qh) = qr.rect

            # Vẽ khung đỏ
            cv2.rectangle(display, (qx, qy), (qx + qw, qy + qh), (0, 0, 255), 3)

            # Hiển thị dữ liệu
            qr_text = f"QR: {qr_data[:30]}"
            cv2.putText(
                display,
                qr_text,
                (qx, qy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

            # Lưu lịch sử
            if (
                qr_data != last_qr["data"]
                or (current_time - last_qr["time"]) >= QR_COOLDOWN
            ):
                last_qr = {"data": qr_data, "time": current_time}
                detection_history.insert(
                    0,
                    {
                        "type": "QR",
                        "content": qr_data,
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                    },
                )
                # Giữ tối đa 20 bản ghi
                if len(detection_history) > 20:
                    detection_history.pop()

    # ============ NHẬN DIỆN BIỂN SỐ ============
    if current_time - last_plate["time"] >= PROCESS_INTERVAL:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 13, 15, 15)
        edged = cv2.Canny(gray, 30, 200)

        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        plate_found = False
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            if len(approx) == 4:
                # Vẽ khung xanh
                cv2.drawContours(display, [approx], -1, (0, 255, 0), 3)

                # Cắt vùng biển
                mask = np.zeros(gray.shape, np.uint8)
                cv2.drawContours(mask, [approx], -1, 255, -1)
                (x, y) = np.where(mask == 255)
                if len(x) == 0 or len(y) == 0:
                    continue

                (x1, y1) = (np.min(x), np.min(y))
                (x2, y2) = (np.max(x), np.max(y))
                cropped = gray[x1 : x2 + 1, y1 : y2 + 1]

                # Tiền xử lý
                cropped = cv2.resize(
                    cropped, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC
                )
                cropped = cv2.fastNlMeansDenoising(cropped, None, 10, 7, 21)
                cropped = cv2.convertScaleAbs(cropped, alpha=1.5, beta=0)
                cropped = cv2.threshold(
                    cropped, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
                )[1]

                kernel = np.ones((2, 2), np.uint8)
                cropped = cv2.morphologyEx(cropped, cv2.MORPH_CLOSE, kernel)

                # OCR
                text = pytesseract.image_to_string(cropped, config=custom_config)
                text = re.sub(r"[^A-Z0-9\.\-]", "", text.upper()).strip()

                # Chuẩn hóa
                if len(text) >= 6:
                    if "-" not in text and "." not in text and len(text) >= 8:
                        if text[2].isalpha() and text[3].isdigit():
                            text = text[:2] + text[2] + "-" + text[3:]
                        elif text[3].isalpha() and text[4].isdigit():
                            text = text[:3] + "-" + text[3:]

                    # Lưu lịch sử
                    if text != last_plate["text"]:
                        last_plate = {"text": text, "time": current_time}
                        detection_history.insert(
                            0,
                            {
                                "type": "PLATE",
                                "content": text,
                                "timestamp": datetime.now().strftime("%H:%M:%S"),
                            },
                        )
                        if len(detection_history) > 20:
                            detection_history.pop()

                    # Hiển thị
                    cv2.putText(
                        display,
                        f"Bien so: {text}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )
                    plate_found = True
                break

        if not plate_found:
            cv2.putText(
                display,
                "Dang quet bien so...",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )
    else:
        # Hiển thị biển cũ
        if last_plate["text"] and current_time - last_plate["time"] < 2.0:
            cv2.putText(
                display,
                f"Bien so: {last_plate['text']}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
        else:
            cv2.putText(
                display,
                "Dang quet bien so...",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

    # Trạng thái QR
    if qr_detected:
        status_qr = f"QR: Phat hien ({len(qr_codes)})"
        color_qr = (0, 255, 0)
    else:
        status_qr = "QR: Dang tim..."
        color_qr = (128, 128, 128)

    cv2.putText(
        display, status_qr, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_qr, 2
    )

    return display


def generate_frames():
    """Generator để stream video"""
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = imutils.resize(frame, width=640)
        frame = process_frame(frame)

        # Encode frame thành JPEG
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        # Yield frame theo định dạng multipart
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    """Trang chủ"""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """Route để stream video"""
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/history")
def get_history():
    """API lấy lịch sử phát hiện"""
    return jsonify(
        {
            "history": detection_history,
            "last_plate": (
                last_plate["text"] if last_plate["text"] else "Chưa phát hiện"
            ),
            "last_qr": last_qr["data"] if last_qr["data"] else "Chưa phát hiện",
        }
    )


@app.route("/api/stats")
def get_stats():
    """API thống kê"""
    plate_count = sum(1 for item in detection_history if item["type"] == "PLATE")
    qr_count = sum(1 for item in detection_history if item["type"] == "QR")

    return jsonify(
        {
            "total_detections": len(detection_history),
            "plate_count": plate_count,
            "qr_count": qr_count,
        }
    )


if __name__ == "__main__":
    print("=" * 60)
    print("🚗 Hệ thống nhận diện biển số & QR Code")
    print("=" * 60)
    print("📡 Server đang chạy tại: http://localhost:5000")
    print("🌐 Hoặc truy cập: http://127.0.0.1:5000")
    print("⏹️  Nhấn Ctrl+C để dừng server")
    print("=" * 60)

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
