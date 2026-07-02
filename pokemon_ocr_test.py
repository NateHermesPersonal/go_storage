import cv2
import pytesseract
import pandas as pd
import re
from tqdm import tqdm

# ==========================================
# ---- CONFIGURATION & FILE PATHS ----------
# ==========================================
VIDEO_PATH = "data/appraise_sample.mp4"  
OUTPUT_CSV = "pokemon_final_inventory.csv"

# Windows Tesseract path alignment
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def clean_ocr_string(text):
    """Removes weird OCR artifacts and leaves only alpha characters."""
    return re.sub(r'[^a-zA-Z\s\-]', '', text).strip()

def extract_pokemon_data(frame):
    """
    Crops specific CP and Name regions for a 1080x2340 display.
    Applies tight adaptive thresholding to eliminate background noise.
    """
    # --- REGION 1: Combat Power (CP) Box ---
    # Shrunk slightly to 90-160 to avoid catching elements above/below text
    cp_crop = frame[90:160, 300:780]
    cp_gray = cv2.cvtColor(cp_crop, cv2.COLOR_BGR2GRAY)
    _, cp_thresh = cv2.threshold(cp_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Whitelist only digits
    cp_text = pytesseract.image_to_string(
        cp_thresh, 
        config='--psm 6 -c tessedit_char_whitelist=0123456789'
    ).strip()

    # --- REGION 2: Name Box ---
    # Shifted slightly to avoid clipping descending letters (like g, j, p, q, y)
    name_crop = frame[1210:1300, 150:930]
    name_gray = cv2.cvtColor(name_crop, cv2.COLOR_BGR2GRAY)
    _, name_thresh = cv2.threshold(name_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Whitelist only alphabet letters for names/species
    name_text = pytesseract.image_to_string(
        name_thresh, 
        config='--psm 6 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-'
    ).strip()

    return name_text, cp_text

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, prev_frame = cap.read()
    if not ret:
        return

    pokemon_data = []
    last_seen_key = None
    
    # Lowered from 5.0 to 1.8 to force the screen to be completely motionless
    MOTION_THRESHOLD = 1.8  

    print("Analyzing video with strict validation filters...")
    
    for _ in tqdm(range(total_frames - 1)):
        ret, frame = cap.read()
        if not ret:
            break
            
        diff = cv2.absdiff(frame, prev_frame)
        non_zero_mean = diff.mean() 
        
        if non_zero_mean < MOTION_THRESHOLD:
            raw_name, raw_cp = extract_pokemon_data(frame)
            
            clean_name = clean_ocr_string(raw_name)
            clean_cp = raw_cp.strip()
            
            # Create a unique tracking key
            current_key = f"{clean_name}_{clean_cp}"
            
            # --- STRICT VALIDATION FILTERS ---
            # 1. Reject if key matches the previous frame
            if current_key != last_seen_key:
                # 2. Reject if the name is too short (e.g. OCR noise like "I" or "X")
                # 3. Reject if CP is empty or realistically impossible (e.g. over 6000 CP)
                if len(clean_name) >= 3 and clean_cp.isdigit():
                    cp_val = int(clean_cp)
                    if 10 <= cp_val <= 6000:
                        
                        pokemon_data.append({
                            "Name": clean_name,
                            "CP": cp_val
                        })
                        
                        last_seen_key = current_key
                        print(f"\n[Validated Capture]: {clean_name} | CP: {cp_val}")

        prev_frame = frame

    cap.release()
    
    if pokemon_data:
        df = pd.DataFrame(pokemon_data)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSuccess! Filtered down to {len(pokemon_data)} valid entries in '{OUTPUT_CSV}'")
    else:
        print("\nNo valid entries passed the validation filters.")
        print("Adjustment tip: If it is skipping everything, raise MOTION_THRESHOLD to 3.0.")

if __name__ == "__main__":
    process_video(VIDEO_PATH)
