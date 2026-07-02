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

# Ensure the debug directory exists
os.makedirs(DEBUG_DIR, exist_ok=True)

# Windows Tesseract path alignment
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- CALIBRATED COORDINATES (1080x2340) ---
NAME_Y_START, NAME_Y_END = 950, 1050
NAME_X_START, NAME_X_END = 100, 980

# Shifted down 15 pixels from your top-edge readings to hit the bar centers
ATTACK_Y  = 1800 + 15  
DEFENSE_Y = 1900 + 15
HP_Y      = 2000 + 15

X_START = 125
X_END   = 500

def clean_ocr_string(text):
    return re.sub(r'[^a-zA-Z\s\-]', '', text).strip()

def extract_name(frame):
    """Parses the species or nickname box."""
    name_crop = frame[NAME_Y_START:NAME_Y_END, NAME_X_START:NAME_X_END]
    name_gray = cv2.cvtColor(name_crop, cv2.COLOR_BGR2GRAY)
    _, name_thresh = cv2.threshold(name_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    name_text = pytesseract.image_to_string(
        name_thresh, 
        config='--psm 6 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-'
    ).strip()
    return clean_ocr_string(name_text)

def calculate_single_stat(mask_image, y_coordinate):
    """Counts white pixels along the horizontal line and maps it to 0-15."""
    pixel_row = mask_image[y_coordinate, X_START:X_END]
    white_pixel_count = np.sum(pixel_row == 255)
    iv_value = int(round(white_pixel_count / 25.0))
    return max(0, min(15, iv_value))

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at path '{video_path}'")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pokemon_data = []
    captured_inventory_set = set()
    
    MOTION_THRESHOLD = 5.0  
    ret, prev_frame = cap.read()

    print("Extracting Names and saving visual debug frames...")
    
    for _ in tqdm(range(total_frames - 1)):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Motion detection on the neutral background card
        zone_curr = frame[1100:1150, 500:550]
        zone_prev = prev_frame[1100:1150, 500:550]
        
        diff = cv2.absdiff(zone_curr, zone_prev)
        motion_score = diff.mean() 
        
        if motion_score < MOTION_THRESHOLD:
            name = extract_name(frame)
            
            if len(name) >= 3:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower_orange = np.array([0, 100, 100])
                upper_orange = np.array([25, 255, 255])
                color_mask = cv2.inRange(hsv, lower_orange, upper_orange)
                
                atk = calculate_single_stat(color_mask, ATTACK_Y)
                dfn = calculate_single_stat(color_mask, DEFENSE_Y)
                hp  = calculate_single_stat(color_mask, HP_Y)
                
                identity_tuple = (name, atk, dfn, hp)
                
                if identity_tuple not in captured_inventory_set:
                    pokemon_data.append({
                        "Name": name, "Attack": atk, "Defense": dfn, "HP": hp,
                        "Total_IV_%": round(((atk + dfn + hp) / 45.0) * 100, 1)
                    })
                    captured_inventory_set.add(identity_tuple)
                    print(f"\n[Captured]: {name} | IV: {atk}/{dfn}/{hp}")
                    
                    # --- FIX: SAVE VISUAL DEBUG VISUALS ---
                    # 1. Save the original clean frame
                    orig_filename = os.path.join(DEBUG_DIR, f"{name}_original.png")
                    cv2.imwrite(orig_filename, frame)
                    
                    # 2. Draw target lines over the mask and save it
                    canvas = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
                    # Draw where the script checks (Green horizontal lines)
                    cv2.line(canvas, (X_START, ATTACK_Y), (X_END, ATTACK_Y), (0, 255, 0), 2)
                    cv2.line(canvas, (X_START, DEFENSE_Y), (X_END, DEFENSE_Y), (0, 255, 0), 2)
                    cv2.line(canvas, (X_START, HP_Y), (X_END, HP_Y), (0, 255, 0), 2)
                    # Draw vertical boundary marks (Blue lines)
                    cv2.line(canvas, (X_START, ATTACK_Y-20), (X_START, HP_Y+20), (255, 0, 0), 2)
                    cv2.line(canvas, (X_END, ATTACK_Y-20), (X_END, HP_Y+20), (255, 0, 0), 2)
                    
                    proc_filename = os.path.join(DEBUG_DIR, f"{name}_processed.png")
                    cv2.imwrite(proc_filename, canvas)

        prev_frame = frame

    cap.release()
    
    if pokemon_data:
        pd.DataFrame(pokemon_data).to_csv(OUTPUT_CSV, index=False)
        print(f"\nSuccess! Saved to '{OUTPUT_CSV}'. Check the '{DEBUG_DIR}' folder for diagnostic images.")

if __name__ == "__main__":
    process_video(VIDEO_PATH)
