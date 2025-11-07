# LV3_002 - License Plate Recognition System

## Project Overview

Real-time Vietnamese license plate recognition system using OpenCV and Tesseract OCR, optimized for Raspberry Pi 4.

## Architecture

- **Single-file application**: `main.py` contains the entire pipeline
- **Processing flow**: Camera capture → Edge detection → Contour finding → License plate extraction → OCR → Normalization
- **Performance optimization**: Throttled OCR processing (0.5s intervals) to reduce CPU load on embedded hardware

## Key Technologies

- **OpenCV (cv2)**: Video capture, image preprocessing, contour detection
- **Tesseract OCR**: Character recognition with custom config for alphanumeric Vietnamese plates
- **imutils**: Image resizing utilities
- **NumPy**: Image array manipulation

## Critical Configuration

```python
# Tesseract path (system-specific)
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# OCR config: Only recognize uppercase letters, numbers, dots, dashes
custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-'

# Performance tuning
PROCESS_INTERVAL = 0.5  # Adjust for CPU load vs responsiveness trade-off
```

## Vietnamese License Plate Format

- Standard format: `XX-A XXXX` or `XXA-XXXX` (2-3 digits, letter, 4 digits)
- Normalization logic in lines 75-82: Auto-inserts hyphen between letter/number transition
- Example: `29A1234` → `29-A1234` or `29A-1234`

## Development Workflow

### Running the Application

```bash
python main.py
# Press 'q' to exit
```

### Dependencies

```bash
pip install opencv-python pytesseract imutils numpy
sudo apt-get install tesseract-ocr  # On Raspberry Pi/Debian
```

### Testing Changes

- Camera must be accessible at index 0 (`cv2.VideoCapture(0)`)
- Resolution: 640×480 (set in lines 16-17)
- Test with physical license plates in good lighting

## Project Conventions

### Image Processing Pipeline

1. **Bilateral filter** (line 30): Noise reduction while preserving edges
2. **Canny edge detection** (line 31): threshold 30-200
3. **Contour approximation** (line 37): Look for 4-corner shapes (license plates)
4. **Cropping & enhancement** (lines 46-56): Resize 1.8x, OTSU thresholding for OCR accuracy

### Performance Patterns

- **Time-gated processing**: OCR only runs every 0.5s (lines 26, 29)
- **Single plate per frame**: `break` after first valid detection (line 87)
- **Duplicate suppression**: Track `last_text` to avoid repeated prints (line 77)
- **Result persistence**: Display last detected plate for 2 seconds (line 93)

### Debugging Tips

- FPS counter displayed at bottom (lines 99-101)
- Green bounding box shows detected plate region
- "Dang quet..." status indicates no plate found
- Check Tesseract path if OCR fails silently

## Code Modification Guidelines

- When adjusting OCR accuracy, modify `custom_config` whitelist
- To change detection sensitivity, tune Canny thresholds (line 31) or contour approximation epsilon (line 37)
- For different camera hardware, update resolution (lines 16-17) and VideoCapture index (line 15)
- Vietnamese text comments describe UI elements and status messages
