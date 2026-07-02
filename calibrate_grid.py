import cv2
import os

VIDEO_PATH = os.path.join("data", "appraise_sample.mp4")

def generate_calibration_grid():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video at path: {VIDEO_PATH}")
        return

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read a valid frame.")
        return

    height, width, _ = frame.shape
    print(f"Loaded frame resolution: {width}x{height}")

    # Draw horizontal grid lines every 50 pixels
    for y in range(0, height, 50):
        # Draw the line
        cv2.line(frame, (0, y), (width, y), (0, 0, 255), 1)
        # Add text label for the coordinate (only on the left side to keep it clean)
        if y % 100 == 0:
            cv2.putText(frame, str(y), (10, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # Save the full image with the grid overlay
    output_path = "debug_full_grid.png"
    cv2.imwrite(output_path, frame)
    print(f"Success! Open '{output_path}' to visually find your text coordinates.")

if __name__ == "__main__":
    generate_calibration_grid()
