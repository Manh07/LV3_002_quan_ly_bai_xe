import cv2
import numpy as np
from pyzbar import pyzbar
import time

# Biến lưu trữ kết quả
last_qr_data = ""
last_qr_time = 0
SCAN_COOLDOWN = 1.0  # Chỉ quét mỗi 1 giây để tránh spam

# Mở webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    print("Không thể mở webcam!")
    exit()

print("Đang quét mã QR... (Nhấn 'q' để thoát)")
print("=" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    current_time = time.time()

    # Detect và decode QR code
    qr_codes = pyzbar.decode(frame)

    if qr_codes:
        for qr in qr_codes:
            # Lấy dữ liệu QR
            qr_data = qr.data.decode("utf-8")
            qr_type = qr.type

            # Lấy vị trí QR code
            (x, y, w, h) = qr.rect

            # Vẽ khung xanh quanh QR code
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Vẽ các điểm góc
            points = qr.polygon
            if len(points) == 4:
                pts = np.array(points, dtype=np.int32)
                cv2.polylines(display, [pts], True, (255, 0, 0), 2)

            # Hiển thị loại mã và dữ liệu
            text = f"{qr_type}: {qr_data}"
            cv2.putText(
                display,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # In ra terminal (chỉ khi khác mã cũ hoặc đủ thời gian cooldown)
            if (
                qr_data != last_qr_data
                or (current_time - last_qr_time) >= SCAN_COOLDOWN
            ):
                print(f"\n[{time.strftime('%H:%M:%S')}] Phát hiện {qr_type}:")
                print(f"  Dữ liệu: {qr_data}")
                print("-" * 50)

                last_qr_data = qr_data
                last_qr_time = current_time

        # Hiển thị trạng thái
        cv2.putText(
            display,
            f"Da quet: {len(qr_codes)} ma",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
    else:
        # Không phát hiện QR
        cv2.putText(
            display,
            "Dang tim ma QR...",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )

    # Hiển thị FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Default
    cv2.putText(
        display,
        f"FPS: {fps:.1f}",
        (10, 460),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        2,
    )

    # Hiển thị frame
    cv2.imshow("Quet ma QR - Webcam", display)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
print("\nĐã thoát chương trình.")
