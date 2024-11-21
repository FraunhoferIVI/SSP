import os
import shutil
import argparse
from tqdm import tqdm
from multiprocessing import Pool
import threading

def move_files_thread(file_pairs):
    """
    Use threads to move files in parallel.
    """
    threads = []
    for src, dest in file_pairs:
        if os.path.isfile(src):
            t = threading.Thread(target=shutil.move, args=(src, dest))
            threads.append(t)
            t.start()
    # Wait for all threads to complete
    for t in threads:
        t.join()

def process_sequence(args):
    """
    Processes a single sequence and organizes files into clips.
    """
    seq, source_images_path, source_labels_path, data_dir, rural_val_seq = args

    # Prepare directories for the sequence
    origin_dir = os.path.join(data_dir, seq, "origin")
    mask_dir = os.path.join(data_dir, seq, "mask")
    os.makedirs(origin_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    images_path = os.path.join(source_images_path, seq)
    labels_path = os.path.join(source_labels_path, seq)

    if not os.path.exists(images_path) or not os.path.exists(labels_path):
        print(f"Missing images or labels for sequence {seq}. Skipping.")
        return

    image_files = sorted([f for f in os.listdir(images_path) if f.endswith(".png")])
    label_files = sorted([f for f in os.listdir(labels_path) if f.endswith(".png")])

    # Discard the first image
    image_files = image_files[1:-1]
    label_files = label_files[1:-1]

    # Move images and labels
    image_file_pairs = [
        (os.path.join(images_path, img), os.path.join(origin_dir, img))
        for img in image_files
    ]
    label_file_pairs = [
        (os.path.join(labels_path, lbl), os.path.join(mask_dir, lbl))
        for lbl in label_files
    ]

    move_files_thread(image_file_pairs)
    move_files_thread(label_file_pairs)

import shutil  # Make sure to import shutil if not already imported

def generate_clips(data_dir, train_txt_path, val_txt_path, rural_val_seq):
    """
    Generates clips for images and masks under the 'data' folder, handling sparse mask labels.
    """
    clip_length = 50  # Number of frames per clip
    min_clip_length = 10  # Minimum number of frames to consider a clip
    global_clip_index = 0

    with open(train_txt_path, 'w') as train_file, open(val_txt_path, 'w') as val_file:
        seq_list = os.listdir(data_dir)
        with tqdm(total=len(seq_list), desc="Processing sequences") as pbar:
            for seq in seq_list:
                seq_path = os.path.join(data_dir, seq)
                origin_dir = os.path.join(seq_path, "origin")
                mask_dir = os.path.join(seq_path, "mask")

                if not os.path.exists(origin_dir) or not os.path.exists(mask_dir):
                    print(f"Origin or mask directory missing for sequence {seq}. Skipping.")
                    pbar.update(1)
                    continue

                # Gather frames from origin
                frames = sorted(os.listdir(origin_dir))

                # Divide frames into clips
                clips = [frames[i:i + clip_length] for i in range(0, len(frames), clip_length)]

                for local_clip_index, clip in enumerate(clips):
                    # Check if any masks are available for the frames in the clip
                    has_mask = any(os.path.isfile(os.path.join(mask_dir, frame)) for frame in clip)

                    # Skip the clip if no masks are available
                    if not has_mask:
                        print(f"Skipping clip {global_clip_index}_{seq}_{local_clip_index} as it has no masks.")
                        continue

                    # Skip the clip if it's shorter than the minimum length
                    if len(clip) < min_clip_length:
                        print(f"Skipping clip {global_clip_index}_{seq}_{local_clip_index} with {len(clip)} frames (less than {min_clip_length}).")
                        continue

                    clip_name = f"{global_clip_index}_{seq}_{local_clip_index}"
                    clip_dir = os.path.join(data_dir, clip_name)
                    images_clip_dir = os.path.join(clip_dir, "origin")
                    masks_clip_dir = os.path.join(clip_dir, "mask")
                    os.makedirs(images_clip_dir, exist_ok=True)
                    os.makedirs(masks_clip_dir, exist_ok=True)

                    # Prepare file pairs for moving
                    image_file_pairs = [
                        (os.path.join(origin_dir, frame), os.path.join(images_clip_dir, frame))
                        for frame in clip if os.path.isfile(os.path.join(origin_dir, frame))
                    ]
                    mask_file_pairs = [
                        (os.path.join(mask_dir, frame), os.path.join(masks_clip_dir, frame))
                        for frame in clip if os.path.isfile(os.path.join(mask_dir, frame))
                    ]

                    # Multithreading for moving files
                    move_files_thread(image_file_pairs)
                    move_files_thread(mask_file_pairs)

                    # Write clip name to the appropriate file
                    if seq in rural_val_seq:
                        val_file.write(clip_name + '\n')
                    else:
                        train_file.write(clip_name + '\n')

                    global_clip_index += 1

                # Remove the sequence directory after processing
                shutil.rmtree(seq_path)

                # Update the progress bar after processing the sequence
                pbar.update(1)




def organize_ruralscapes(dataset_dir):
    """
    Organize the Ruralscapes dataset into the desired structure and process sequences.
    """
    source_images = os.path.join(dataset_dir, "Ruralscapes/images")
    source_labels = os.path.join(dataset_dir, "Ruralscapes/labels/gt_labels")
    data_dir = os.path.join(dataset_dir, "ruralscapes/data")
    txt_dir = os.path.join(dataset_dir, "ruralscapes")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)

    train_txt_path = os.path.join(txt_dir, "train.txt")
    val_txt_path = os.path.join(txt_dir, "val.txt")

    rural_val_seq = ["DJI_0051", "DJI_0056", "DJI_0061", "DJI_0086", "DJI_0088", "DJI_0089", "DJI_0116"]

    sequences = [seq for seq in os.listdir(source_images) if os.path.isdir(os.path.join(source_images, seq))]

    args = [
        (seq, source_images, source_labels, data_dir, rural_val_seq)
        for seq in sequences
    ]

    with Pool(processes=8) as pool:
        pool.map(process_sequence, args)

    # Generate clips
    generate_clips(data_dir, train_txt_path, val_txt_path, rural_val_seq)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Organize ruralscapes dataset.')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the dataset directory.')
    args = parser.parse_args()

    organize_ruralscapes(args.dataset_dir)

    # delete the original images and labels folders
    shutil.rmtree(os.path.join(args.dataset_dir, "Ruralscapes"))
