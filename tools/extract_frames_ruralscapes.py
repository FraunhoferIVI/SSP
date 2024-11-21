import os
import cv2
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

def extract_and_save_frames(args):
    """
    Extracts frames from a video file at specified intervals, resizes them if needed,
    and saves them with a specific naming convention.
    """
    video_path, output_base_dir, frame_interval, resize, resize_dims = args
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(output_base_dir, video_basename)
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video file {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = list(range(0, total_frames, frame_interval))
    if frame_indices[-1] != total_frames - 1:
        frame_indices.append(total_frames - 1)  # Ensure last frame is included

    frame_set = set(frame_indices)
    frame_count = 0
    saved_count = 0

    pbar = tqdm(total=total_frames, desc=f"Processing {video_basename}", position=0, leave=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count in frame_set:
            if resize:
                frame = cv2.resize(frame, resize_dims, interpolation=cv2.INTER_CUBIC)
            filename = f"segfull_{video_basename}_{frame_count:06d}.png"
            filepath = os.path.join(output_folder, filename)
            cv2.imwrite(filepath, frame)
            saved_count += 1

        frame_count += 1
        pbar.update(1)

    cap.release()
    pbar.close()
    print(f"Extracted {saved_count} frames from {video_path}")

def remap_label_image(args):
    """
    Efficiently remaps a single label image from RGB to single-channel class indices and saves it.
    """
    label_path, output_path, mapping = args
    label_img = cv2.imread(label_path)

    # Create an array to map RGB values to class indices
    max_val = 256  # For 8-bit images
    mapping_array = np.zeros((max_val, max_val, max_val), dtype=np.uint8)
    for class_index, rgb in mapping.items():
        mapping_array[rgb[0], rgb[1], rgb[2]] = class_index

    # Remap the label image
    label_mapped = mapping_array[label_img[..., 0], label_img[..., 1], label_img[..., 2]]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, label_mapped)

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Process Ruralscapes dataset.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the Ruralscapes directory')
    parser.add_argument('--resize', action='store_true', help='Downsample frames to 2560x1440 using bicubic interpolation')
    parser.add_argument('--processes', type=int, default=8, help='Number of processes to use')
    args = parser.parse_args()

    dataset_path = args.dataset_path
    resize = args.resize
    num_processes = args.processes
    resize_dims = (2560, 1440)

    # Define the mapping from RGB values to class indices
    mapping_list = [
        ("unlabeled", (0, 0, 0)),
        ("residential", (255, 255, 0)),
        ("land", (0, 255, 0)),
        ("forest", (0, 127, 0)),
        ("sky", (0, 255, 255)),
        ("fence", (127, 127, 0)),
        ("road", (255, 255, 255)),
        ("hill", (127, 127, 63)),
        ("church", (255, 0, 255)),
        ("car", (127, 127, 127)),
        ("person", (255, 0, 0)),
        ("haystack", (255, 127, 0)),
        ("water", (0, 0, 255))
    ]

    mapping = {idx: rgb for idx, (_, rgb) in enumerate(mapping_list)}

    # Set paths
    videos_dir = os.path.join(dataset_path, 'videos')
    images_dir = os.path.join(dataset_path, 'images')
    labels_manual_dir = os.path.join(dataset_path, 'labels', 'manual_labels')
    labels_gt_dir = os.path.join(dataset_path, 'labels', 'gt_labels')

    # Ensure directories exist
    if not os.path.exists(videos_dir):
        print(f"Videos directory not found: {videos_dir}")
        return
    if not os.path.exists(labels_manual_dir):
        print(f"Manual labels directory not found: {labels_manual_dir}")
        return

    # Process videos
    video_files = [f for f in os.listdir(videos_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    print(f"Found {len(video_files)} video files to process.")

    video_args = []
    for video_file in video_files:
        video_path = os.path.join(videos_dir, video_file)
        video_args.append((video_path, images_dir, 5, resize, resize_dims))

    # Use multiprocessing to process videos
    with Pool(processes=num_processes) as pool:
        list(pool.imap_unordered(extract_and_save_frames, video_args))

    print("Frame extraction completed.")

    # Process labels
    seq_dirs = [d for d in os.listdir(labels_manual_dir) if os.path.isdir(os.path.join(labels_manual_dir, d))]
    print(f"Found {len(seq_dirs)} sequences to process for labels.")

    label_args = []
    for seq_dir in seq_dirs:
        input_seq_dir = os.path.join(labels_manual_dir, seq_dir)
        output_seq_dir = os.path.join(labels_gt_dir, seq_dir)
        label_files = [f for f in os.listdir(input_seq_dir) if f.lower().endswith('.png')]
        for label_file in label_files:
            label_path = os.path.join(input_seq_dir, label_file)
            output_path = os.path.join(output_seq_dir, label_file)
            label_args.append((label_path, output_path, mapping))

    # Use multiprocessing to process labels
    with Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap_unordered(remap_label_image, label_args), total=len(label_args), desc="Remapping labels"))

    print("Label remapping completed.")

if __name__ == "__main__":
    main()
