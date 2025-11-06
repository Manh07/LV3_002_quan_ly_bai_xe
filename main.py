import cv2
import pytesseract
import imutils
import numpy as np
import re
import time

# Cấu hình Tesseract
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-'

# Biến kiểm soát
last_text = ""
last_time = 0
PROCESS_INTERVAL = 0.5  # Xử lý mỗi 0.5 giây (giảm tải CPU)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

print("Quét biển số liên tục... (Nhấn 'q' để thoát)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=640)
    display = frame.copy()
    
    current_time = time.time()

    # Chỉ xử lý OCR mỗi 0.5s
    if current_time - last_time >= PROCESS_INTERVAL:
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
                plate_cnt = approx
                plate_found = True

                # Vẽ khung
                cv2.drawContours(display, [plate_cnt], -1, (0, 255, 0), 3)

                # Cắt vùng biển
                mask = np.zeros(gray.shape, np.uint8)
                cv2.drawContours(mask, [plate_cnt], -1, 255, -1)
                (x, y) = np.where(mask == 255)
                if len(x) == 0 or len(y) == 0:
                    continue
                (x1, y1) = (np.min(x), np.min(y))
                (x2, y2) = (np.max(x), np.max(y))
                cropped = gray[x1:x2+1, y1:y2+1]

                # Tiền xử lý ảnh
                cropped = cv2.resize(cropped, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
                cropped = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

                # OCR
                text = pytesseract.image_to_string(cropped, config=custom_config)
                text = re.sub(r'[^A-Z0-9\.\-]', '', text.upper()).strip()

                # Chuẩn hóa biển số
                if len(text) >= 6 and text != last_text:
                    if '-' not in text and '.' not in text and len(text) >= 8:
                        if text[2].isalpha() and text[3].isdigit():
                            text = text[:2] + text[2] + '-' + text[3:]
                        elif text[3].isalpha() and text[4].isdigit():
                            text = text[:3] + '-' + text[3:]

                    print(f"Đã phát hiện: {text}")
                    last_text = text
                    last_time = current_time

                    cv2.putText(display, f"{text}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
                break  # Chỉ xử lý 1 biển số mỗi khung

        if not plate_found:
            cv2.putText(display, "Dang quet...", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    else:
        # Hiển thị biển số cũ nếu còn trong thời gian chờ
        if last_text and current_time - last_time < 2.0:
            cv2.putText(display, f"{last_text}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        else:
            cv2.putText(display, "Dang quet...", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Hiển thị FPS
    fps = 1.0 / (time.time() - current_time + 0.001)
    cv2.putText(display, f"FPS: {fps:.1f}", (10, 480-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    cv2.imshow("Bien so REALTIME - Raspi 4", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()