import cv2
import numpy as np
import os

VIDEO_PATH = os.path.join("data", "appraise_sample.mp4")

def analyze_iv_bars():
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to read video frame.")
        return

    # 1. Convert frame to HSV color space (highly reliable for catching specific colors)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 2. Define the Pokémon GO Appraisal Orange/Pink bar color range
    # These bounds target the glowing orange/red fill color of the bars
    lower_orange = np.array([0, 100, 100])
    upper_orange = np.array([25, 255, 255])
    
    # Create a binary image mask (white pixels = stat bar, black pixels = background)
    color_mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Save the color mask image so you can visually verify it
    cv2.imwrite("debug_iv_color_mask.png", color_mask)
    print("Saved 'debug_iv_color_mask.png'. Check if the 3 stat bars show up as solid white horizontal bars!")

if __name__ == "__main__":
    analyze_iv_bars()
