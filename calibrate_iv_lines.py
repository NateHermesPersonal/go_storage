import cv2
import numpy as np
import os

VIDEO_PATH = os.path.join("data", "appraise_sample.mp4")

def refine_iv_coordinates():
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not read frame.")
        return

    # Generate the same working white mask you just verified
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Using standard PoGO Appraisal Orange/Pink color bounds
    lower_orange = np.array([0, 100, 100])
    upper_orange = np.array([25, 255, 255])
    color_mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Convert mask to BGR so we can draw colored calibration lines on top of it
    debug_canvas = cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB)
    
    # 1. Estimate the exact center-lines for each bar within your bands
    # We will draw a green line through our best guess for the centers
    attack_y = 1800
    defense_y = 1900
    hp_y = 2000
    
    cv2.line(debug_canvas, (0, attack_y), (debug_canvas.shape[1], attack_y), (0, 255, 0), 2)
    cv2.line(debug_canvas, (0, defense_y), (debug_canvas.shape[1], defense_y), (0, 255, 0), 2)
    cv2.line(debug_canvas, (0, hp_y), (debug_canvas.shape[1], hp_y), (0, 255, 0), 2)
    
    # 2. Draw vertical reference markers every 50 pixels horizontally 
    # This helps find where the bars start on the left and end on the right
    for x in range(100, debug_canvas.shape[1], 50):
        cv2.line(debug_canvas, (x, 1700), (x, 2100), (255, 0, 0), 1)
        if x % 100 == 0:
            cv2.putText(debug_canvas, str(x), (x - 15, 1690), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

    cv2.imwrite("debug_iv_lines.png", cv2.cvtColor(debug_canvas, cv2.COLOR_RGB2BGR))
    print("Saved 'debug_iv_lines.png'. Let's check the line alignments!")

if __name__ == "__main__":
    refine_iv_coordinates()
