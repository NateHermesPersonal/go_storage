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
BAR_TOTAL_WIDTH = X_END - X_START  # 375 pixels

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
    """
    Counts the number of white pixels along the horizontal line 
    from X_START to X_END and maps it to a value between 0 and 15.
    """
    # Extract the line of pixels across our bar range
    pixel_row = mask_image[y_coordinate, X_START:X_END]
    
    # White pixels in our binary mask have a value of 255
    white_pixel_count = np.sum(pixel_row == 255)
    
    # Convert pixel length to 0-15 scale (25 pixels per stat point)
    iv_value = int(round(white_pixel_count / 25.0))
    
    # Constrain values to safe limits
    return max(0, min(15, iv_value))

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at path '{video_path}'")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pokemon_data = []
    captured_inventory_set = set()
    
    # Motion configuration looking at the stable appraisal card panel background
    MOTION_THRESHOLD = 5.0  
    ret, prev_frame = cap.read()

    print("Extracting Names and 0-15 IV statistics...")
    
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
            # 1. Read the Name text
            name = extract_name(frame)
            
            if len(name) >= 3:
                # 2. Generate the validated orange stat bar color mask
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # --- FIXED: COLOR BOUNDS APPLIED ---
                lower_orange = np.array([5, 100, 100])
                upper_orange = np.array([25, 255, 255])
                color_mask = cv2.inRange(hsv, lower_orange, upper_orange)
                
                # 3. Calculate IV scores from the mask
                atk = calculate_single_stat(color_mask, ATTACK_Y)
                dfn = calculate_single_stat(color_mask, DEFENSE_Y)
                hp  = calculate_single_stat(color_mask, HP_Y)
                
                # Global Deduplication check using the composite identity tuple
                identity_tuple = (name, atk, dfn, hp)
                
                if identity_tuple not in captured_inventory_set:
                    pokemon_data.append({
                        "Name": name,
                        "Attack": atk,
                        "Defense": dfn,
                        "HP": hp,
                        "Total_IV_%": round(((atk + dfn + hp) / 45), 2) * 100
                    })
                    captured_inventory_set.add(identity_tuple)
                    print(f"\n[Captured]: {name} | IV: {atk}/{dfn}/{hp} ({pokemon_data[-1]['Total_IV_%']}% )")

        prev_frame = frame

    cap.release()
    
    # Output to structural spreadsheet format
    if pokemon_data:
        df = pd.DataFrame(pokemon_data)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSuccess! Compiled {len(pokemon_data)} items into '{OUTPUT_CSV}'")
    else:
        print("\nProcessed video track but found no valid data.")

if __name__ == "__main__":
    process_video(VIDEO_PATH)
