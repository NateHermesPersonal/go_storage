import cv2
import pytesseract
import pandas as pd
from tqdm import tqdm

# ==========================================
# ---- CONFIGURATION & FILE PATHS ----------
# ==========================================
VIDEO_PATH = "data/appraise_sample.mp4"  # 1. CHANGE THIS to match your filename!
OUTPUT_CSV = "pokemon_final_inventory.csv"

# 2. WINDOWS USERS ONLY: If Python can't find tesseract, 
# uncomment the line below and point it to your installation path:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def extract_pokemon_data(frame):
    """
    Crops the specific CP and Name regions for a 1080x2340 display,
    applies image preprocessing, and runs the OCR engine.
    """
    # --- REGION 1: Combat Power (CP) Box (Top Center) ---
    # Coordinates format: frame[y_start:y_end, x_start:x_end]
    cp_crop = frame[70:180, 250:830]
    cp_gray = cv2.cvtColor(cp_crop, cv2.COLOR_BGR2GRAY)
    _, cp_thresh = cv2.threshold(cp_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Whitelist limits OCR to digits and the letters C and P
    cp_text = pytesseract.image_to_string(
        cp_thresh, 
        config='--psm 6 -c tessedit_char_whitelist=0123456789CPcp '
    ).strip()

    # --- REGION 2: Name Box (Middle Center, Above Appraisal Chart) ---
    name_crop = frame[1200:1310, 100:980]
    name_gray = cv2.cvtColor(name_crop, cv2.COLOR_BGR2GRAY)
    _, name_thresh = cv2.threshold(name_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Standard alphanumeric text parsing
    name_text = pytesseract.image_to_string(name_thresh, config='--psm 6').strip()

    return name_text, cp_text


def process_video(video_path):
    # Load the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'. Check the file name.")
        return

    # Gather data for progress bar tracker
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Read the very first frame to establish a baseline for motion tracking
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Video file appears to be empty or corrupted.")
        return

    pokemon_data = []
    last_seen_key = None
    
    # Sensitivity Settings: Lower numbers require the screen to be perfectly still.
    # Higher numbers allow for slight screen jitter or animated backgrounds.
    MOTION_THRESHOLD = 5.0  

    print("Analyzing video and extracting Pokémon stats...")
    
    # Loop through every remaining frame of the video track
    for _ in tqdm(range(total_frames - 1)):
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Calculate frame-by-frame visual changes (Motion Detection)
        diff = cv2.absdiff(frame, prev_frame)
        non_zero_mean = diff.mean() 
        
        # 2. If motion drops below threshold, the swipe animation has finished
        if non_zero_mean < MOTION_THRESHOLD:
            
            # Extract text data from the static frame layout
            raw_name, raw_cp = extract_pokemon_data(frame)
            current_key = f"{raw_name}_{raw_cp}"
            
            # 3. Data-driven Deduplication Check
            if raw_name and raw_cp and (current_key != last_seen_key):
                
                # Filter out raw OCR artifacts/special characters during transition frames
                clean_name = "".join(c for c in raw_name if c.isalnum() or c.isspace()).strip()
                clean_cp = "".join(c for c in raw_cp if c.isdigit()).strip()
                
                # Basic string length validation before saving entry to sheet
                if len(clean_name) > 2 and len(clean_cp) > 0:
                    pokemon_data.append({
                        "Name": clean_name,
                        "CP": int(clean_cp)
                    })
                    
                    # Update key to prevent reading this same screen on the next frame loop
                    last_seen_key = current_key
                    print(f"\n[Captured]: {clean_name} | CP: {clean_cp}")

        # Set current frame as baseline for next iteration loop
        prev_frame = frame

    cap.release()
    
    # 4. Generate CSV File Output
    if pokemon_data:
        df = pd.DataFrame(pokemon_data)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSuccess! Saved {len(pokemon_data)} items to '{OUTPUT_CSV}'")
    else:
        print("\nFinished processing: No items were saved.")
        print("Tip: If it skipped everything, try raising MOTION_THRESHOLD to 10.0.")


if __name__ == "__main__":
    process_video(VIDEO_PATH)
