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
    cp_crop = frame[100:250, 150:930]
    cp_gray = cv2.cvtColor(cp_crop, cv2.COLOR_BGR2GRAY)
    _, cp_thresh = cv2.threshold(cp_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    cp_text = pytesseract.image_to_string(
        cp_thresh, 
        config='--psm 6 -c tessedit_char_whitelist=0123456789'
    ).strip()

    # --- REGION 2: Name Box (950 to 1050 Y-coordinates) ---
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
    
    # Read the very first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Empty video file.")
        return

    pokemon_data = []
    
    # --- THE MEMORY SET ---
    # Keeps track of every unique (Name, CP) tuple we have successfully saved
    captured_inventory_set = set()
    
    # --- FIX 1: CAPTURE THE 1ST POKÉMON ---
    # Manually check frame 1 before starting the movement loop
    raw_name, raw_cp = extract_pokemon_data(first_frame)
    clean_name = clean_ocr_string(raw_name)
    clean_cp = raw_cp.strip()
    
    if len(clean_name) >= 3 and clean_cp.isdigit():
        cp_val = int(clean_cp)
        if 10 <= cp_val <= 6000:
            pokemon_data.append({"Name": clean_name, "CP": cp_val})
            captured_inventory_set.add((clean_name, cp_val))
            print(f"[Initial Frame Capture]: {clean_name} | CP: {cp_val}")

    prev_frame = first_frame
    MOTION_THRESHOLD = 5.0  

    print("Running data extraction with absolute deduplication...")
    
    # Loop through the rest of the video
    for _ in tqdm(range(total_frames - 1)):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Track 50x50 block on solid white background to detect swipe pauses
        zone_curr = frame[1100:1150, 500:550]
        zone_prev = prev_frame[1100:1150, 500:550]
        
        diff = cv2.absdiff(zone_curr, zone_prev)
        motion_score = diff.mean() 
        
        if motion_score < MOTION_THRESHOLD:
            raw_name, raw_cp = extract_pokemon_data(frame)
            
            clean_name = clean_ocr_string(raw_name)
            clean_cp = raw_cp.strip()
            
            if len(clean_name) >= 3 and clean_cp.isdigit():
                cp_val = int(clean_cp)
                
                if 10 <= cp_val <= 6000:
                    # --- FIX 2: GLOBAL DEDUPLICATION CHECK ---
                    # Only save if this specific Name + CP combination hasn't been seen yet
                    identity_tuple = (clean_name, cp_val)
                    
                    if identity_tuple not in captured_inventory_set:
                        pokemon_data.append({
                            "Name": clean_name,
                            "CP": cp_val
                        })
                        captured_inventory_set.add(identity_tuple)
                        print(f"\n[Validated Capture]: {clean_name} | CP: {cp_val}")

        prev_frame = frame

    cap.release()
    
    # Save spreadsheet data
    if pokemon_data:
        df = pd.DataFrame(pokemon_data)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSuccess! Found {len(pokemon_data)} unique items and saved to '{OUTPUT_CSV}'")
    else:
        print("\nProcessed video but zero unique items passed validation.")


if __name__ == "__main__":
    process_video(VIDEO_PATH)
