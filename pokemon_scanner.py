import cv2
import pytesseract
import pandas as pd
import numpy as np
import re
import os
from tqdm import tqdm

# ==========================================
# ---- CONFIGURATION & FILE PATHS ----------
# ==========================================
VIDEO_PATH = os.path.join("data", "appraise_sample.mp4")
OUTPUT_CSV = "pokemon_iv_inventory.csv"
DEBUG_DIR = "debug_captures"

os.makedirs(DEBUG_DIR, exist_ok=True)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- YOUR PERFECT CALIBRATED COORDINATES ---
X_START = 125
X_END = 500

# Top of your bars shifted down 15 pixels to sample the middle thickness
ATTACK_Y  = 1750 + 15
DEFENSE_Y = 1850 + 15
HP_Y      = 1950 + 15

def clean_ocr_string(text):
    return re.sub(r'[^a-zA-Z\s\-]', '', text).strip()

def extract_name(frame):
    """Parses the name text from your perfectly functioning upper zone."""
    name_crop = frame[950:1050, 100:980]
    name_gray = cv2.cvtColor(name_crop, cv2.COLOR_BGR2GRAY)
    _, name_thresh = cv2.threshold(name_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    name_text = pytesseract.image_to_string(
        name_thresh, 
        config='--psm 6 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-'
    ).strip()
    return clean_ocr_string(name_text)

def calculate_single_stat(binary_mask, y_coordinate):
    """Counts high-contrast bar pixels along the horizontal line."""
    pixel_row = binary_mask[y_coordinate, X_START:X_END]
    white_pixel_count = np.sum(pixel_row == 255)
    
    # 25 pixels per IV point mapping
    iv_value = int(round(white_pixel_count / 25.0))
    return max(0, min(15, iv_value))

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at '{video_path}'")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pokemon_data = []
    captured_inventory_set = set()
    
    # Raised threshold to ensure the script registers a still screen comfortably
    MOTION_THRESHOLD = 8.0  
    ret, prev_frame = cap.read()

    print("Processing entire video track for names and stats...")
    
    for _ in tqdm(range(total_frames - 1)):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Motion detection on the background panel
        zone_curr = frame[2100:2150, 500:550]
        zone_prev = prev_frame[2100:2150, 500:550]
        
        diff = cv2.absdiff(zone_curr, zone_prev)
        motion_score = diff.mean() 
        
        if motion_score < MOTION_THRESHOLD:
            name = extract_name(frame)
            
            if len(name) >= 3:
                # Isolate the IV chart bounding box (Y: 1700 to 2100)
                iv_zone_crop = frame[1700:2100, X_START:X_END]
                iv_zone_gray = cv2.cvtColor(iv_zone_crop, cv2.COLOR_BGR2GRAY)
                
                # Dark stat bars on a light panel turn pure white (255) when inverted
                _, bar_mask_crop = cv2.threshold(iv_zone_gray, 200, 255, cv2.THRESH_BINARY_INV)
                
                # Project back to full frame size matrix
                full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                full_mask[1700:2100, X_START:X_END] = bar_mask_crop
                
                # Extract stats from line traversals
                atk = calculate_single_stat(full_mask, ATTACK_Y)
                dfn = calculate_single_stat(full_mask, DEFENSE_Y)
                hp  = calculate_single_stat(full_mask, HP_Y)
                
                identity_tuple = (name, atk, dfn, hp)
                
                if identity_tuple not in captured_inventory_set:
                    pokemon_data.append({
                        "Name": name, "Attack": atk, "Defense": dfn, "HP": hp,
                        "Total_IV_%": round(((atk + dfn + hp) / 45.0) * 100, 1)
                    })
                    captured_inventory_set.add(identity_tuple)
                    print(f"\n[Captured]: {name} | IV: {atk}/{dfn}/{hp} ({pokemon_data[-1]['Total_IV_%']}% )")
                    
                    # Generate and store tracking image verification files
                    canvas = cv2.cvtColor(full_mask, cv2.COLOR_GRAY2BGR)
                    cv2.line(canvas, (X_START, ATTACK_Y), (X_END, ATTACK_Y), (0, 255, 0), 2)
                    cv2.line(canvas, (X_START, DEFENSE_Y), (X_END, DEFENSE_Y), (0, 255, 0), 2)
                    cv2.line(canvas, (X_START, HP_Y), (X_END, HP_Y), (0, 255, 0), 2)
                    
                    proc_filename = os.path.join(DEBUG_DIR, f"{name}_static_processed.png")
                    cv2.imwrite(proc_filename, canvas)

        prev_frame = frame

    cap.release()
    
    if pokemon_data:
        df = pd.DataFrame(pokemon_data)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSuccess! Compiled your inventory sheet into '{OUTPUT_CSV}'")
    else:
        print("\nProcessed video track but zero items were saved to the dataset.")

if __name__ == "__main__":
    process_video(VIDEO_PATH)
