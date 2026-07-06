import cv2
import pytesseract
import os

# Updated to use your exact relative data path
VIDEO_PATH = os.path.join("data", "appraise_sample.mp4")
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def debug_visual_layouts():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video at path: {VIDEO_PATH}")
        print("Double check that the 'data' folder is in the same directory as this script.")
        return

    # Read the very first frame of your video
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read a valid frame from the video.")
        return

    print(f"Video Frame loaded successfully! Dimensions: {frame.shape[1]}x{frame.shape[0]}")

    # --- CROP 1: CP Box ---
    cp_crop = frame[70:180, 250:830]
    cp_gray = cv2.cvtColor(cp_crop, cv2.COLOR_BGR2GRAY)
    _, cp_thresh = cv2.threshold(cp_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite("debug_cp_crop.png", cp_thresh)
    print("Saved 'debug_cp_crop.png' to your folder.")

    # --- CROP 2: Name Box ---
    name_crop = frame[1200:1310, 100:980]
    name_gray = cv2.cvtColor(name_crop, cv2.COLOR_BGR2GRAY)
    _, name_thresh = cv2.threshold(name_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite("debug_name_crop.png", name_thresh)
    print("Saved 'debug_name_crop.png' to your folder.")

    # Run a test OCR directly on this static frame
    cp_text = pytesseract.image_to_string(cp_thresh, config='--psm 6').strip()
    name_text = pytesseract.image_to_string(name_thresh, config='--psm 6').strip()

    print("\n--- Test OCR Output for Frame 1 ---")
    print(f"Raw CP OCR Result: '{cp_text}'")
    print(f"Raw Name OCR Result: '{name_text}'")

if __name__ == "__main__":
    debug_visual_layouts()
