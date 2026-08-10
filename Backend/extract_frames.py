import cv2
import os
import numpy as np

RAW_CLIPS_DIR = "../RawClips"   # RawClips/dry/, RawClips/damp/, RawClips/wet/
OUTPUT_DIR = "../Data"           # Data/dry/, Data/damp/, Data/wet/
FRAME_INTERVAL_SEC = 1.5         # check a candidate frame every N seconds
SIMILARITY_THRESHOLD = 30        # higher = allows more similar frames through; lower = stricter

def frame_difference(f1, f2):
    """Mean pixel difference between two grayscale frames."""
    gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    return np.mean(diff)

def extract_from_clip(video_path, output_folder, start_index):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * FRAME_INTERVAL_SEC)

    saved_count = start_index
    last_saved_frame = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            if last_saved_frame is None or frame_difference(frame, last_saved_frame) > SIMILARITY_THRESHOLD:
                out_path = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(out_path, frame)
                last_saved_frame = frame
                saved_count += 1

        frame_idx += 1

    cap.release()
    return saved_count

def process_class(class_name):
    clips_folder = os.path.join(RAW_CLIPS_DIR, class_name)
    output_folder = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(output_folder, exist_ok=True)

    existing = len([f for f in os.listdir(output_folder) if f.endswith('.jpg')])
    index = existing

    if not os.path.exists(clips_folder):
        print(f"No clips folder found for {class_name}, skipping.")
        return

    for filename in os.listdir(clips_folder):
        if filename.lower().endswith(('.mp4', '.mkv', '.mov', '.avi')):
            path = os.path.join(clips_folder, filename)
            print(f"Processing {filename} for class '{class_name}'...")
            index = extract_from_clip(path, output_folder, index)

    print(f"{class_name}: now has {index} total frames")

if __name__ == "__main__":
    for cls in ["dry", "damp", "wet"]:
        process_class(cls)
        