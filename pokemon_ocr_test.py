import cv2
import pytesseract
import pandas as pd
import re
import os
from tqdm import tqdm

# ==========================================
# ---- CONFIGURATION & FILE PATHS ----------
# ==========================================
VIDEO_PATH = os.path.join("data", "appraise_sample.mp4")
OUTPUT_CSV = "pokemon_final_inventory.csv"

# Windows Tesseract path alignment
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def clean_ocr_string(text):
    """Removes non-alphabetic characters from the parsed name."""
    return re.sub(r'[^a-zA-Z\s\-]', '', text).strip()


def extract_pokemon_data(frame):
    """
    Crops specific CP and Name regions using your visually calibrated coordinates.
    """
    # --- REGION 1: Combat Power (CP) Box (100 to 250 Y-coordinates) ---
    # Width expanded (150 to 930) to safely capture 'CP 4000+' strings centered on screen
    cp_crop = frame[100:250, 150:930]
    cp_gray = cv2.cvtColor(cp_crop, cv2.COLOR_BGR2GRAY)
    _, cp_thresh = cv2.threshold(cp_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    cp_text = pytesseract.image_to_string(
        cp_thresh, 
        config='--psm 6 -c tessedit_char_whitelist=0123456789'
    ).strip()

    # --- REGION 2: Name Box (950 to 1050 Y-coordinates) ---
    # Width kept wide to ensure long custom nicknames aren't truncated
    name_crop = frame[950:1050, 100:980]
    name_gray = cv2.cvtColor(name_crop, cv2.COLOR_BGR2GRAY)
    _, name_thresh = cv2.threshold(name_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    name_text = pytesseract.image_to_string(
        name_thresh, 
        config='--psm 6 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-'
    ).strip()

    return name_text, cp_text


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at path '{video_path}'")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Empty video file.")
        return

    pokemon_data = []
    last_seen_key = None
    
    # Motion setting for checking a small static box on the appraisal panel background
    # (Since the appraisal panel background is higher up than before, we check Y: 1100 to 1150)
    MOTION_THRESHOLD = 5.0  

    print("Running data extraction with calibrated coordinates...")
    
    for _ in tqdm(range(total_frames - 1)):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Track a tiny 50x50 block on the solid white menu background card to detect pauses
        zone_curr = frame[1100:1150, 500:550]
        zone_prev = prev_frame[1100:1150, 500:550]
        
        diff = cv2.absdiff(zone_curr, zone_prev)
        motion_score = diff.mean() 
        
        # When the screen locks into place, pull the text data
        if motion_score < MOTION_THRESHOLD:
            raw_name, raw_cp = extract_pokemon_data(frame)
            
            clean_name = clean_ocr_string(raw_name)
            clean_cp = raw_cp.strip()
            
            current_key = f"{clean_name}_{clean_cp}"
            
            # Deduplicate logic and length sanity check
            if current_key != last_seen_key:
                if len(clean_name) >= 3 and clean_cp.isdigit():
                    cp_val = int(clean_cp)
                    
                    if 10 <= cp_val <= 6000:
                        pokemon_data.append({
                            "Name": clean_name,
                            "CP": cp_val
                        })
                        last_seen_key = current_key
                        print(f"\n[Captured]: {clean_name} | CP: {cp_val}")

        prev_frame = frame

    cap.release()
    
    # Save out structured spreadsheet data
    if pokemon_data:
        df = pd.DataFrame(pokemon_data)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSuccess! Found {len(pokemon_data)} items and saved to '{OUTPUT_CSV}'")
    else:
        print("\nProcessed video but zero items passed validation.")
        print("Debugging tip: Try raising MOTION_THRESHOLD to 12.0 in the script to account for animated backgrounds.")


if __name__ == "__main__":
    process_video(VIDEO_PATH)
