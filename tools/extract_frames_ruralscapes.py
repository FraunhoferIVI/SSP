import os
import cv2
from multiprocessing import Pool
from tqdm import tqdm

def extract_frames(seq_folder, resize=False):
    """
    Extracts all frames from images.mp4 in the given seq_folder and saves them as PNG images
    in a folder named FF_Images inside seq_folder.
    Optionally downsamples each frame to 2560x1440.
    """
    video_path = os.path.join(seq_folder, 'images.mp4')
    output_folder = os.path.join(seq_folder, 'FF_Images')
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video file {video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if resize:
            # Downsample the frame to 2560x1440 using bicubic interpolation
            frame = cv2.resize(frame, (2560, 1440), interpolation=cv2.INTER_CUBIC)

        # Construct the filename with zero-padded six-digit numbering
        filename = f"{frame_count:06d}.png"
        filepath = os.path.join(output_folder, filename)
        cv2.imwrite(filepath, frame)
        frame_count += 1
    
    cap.release()
    print(f"Extracted {frame_count} frames from {video_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract frames from videos in the UAVid dataset.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the uavid_v1.5_official_release directory')
    parser.add_argument('--resize', action='store_true', help='Downsample frames to 2560x1440 using bicubic interpolation')
    args = parser.parse_args()
    
    dataset_path = args.dataset_path
    resize = args.resize
    
    seq_folders = []
    for split in ['uavid_train', 'uavid_val']:
        split_path = os.path.join(dataset_path, split)
        # List all seq folders in the split
        seqs = [os.path.join(split_path, seq) for seq in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, seq))]
        seq_folders.extend(seqs)
    
    print(f"Found {len(seq_folders)} sequences to process.")
    
    # Wrap the function to include the resize argument
    def wrapper(seq_folder):
        extract_frames(seq_folder, resize=resize)
    
    # Use multiprocessing Pool with tqdm for progress tracking
    with Pool(processes=8) as pool:  # 8 processes in parallel
        list(tqdm(pool.imap(wrapper, seq_folders), total=len(seq_folders), desc="Processing sequences"))
    
    print("Frame extraction completed.")