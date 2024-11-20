import os
import shutil
import numpy as np
import cv2
import argparse
from tqdm import tqdm
from multiprocessing import Pool
import threading

def copy_files_thread(file_pairs):
    """
    Use threads to copy files in parallel.
    """
    threads = []
    for src, dest in file_pairs:
        if os.path.isfile(src):
            t = threading.Thread(target=shutil.copy, args=(src, dest))
            threads.append(t)
            t.start()
    # Wait for all threads to complete
    for t in threads:
        t.join()

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

def process_and_merge_masks(label_path, vdd_label_path, dest_mask_path, dest_color_mask_path):
    """
    Process and merge masks from Labels and VDD_Labels directories.
    Saves the single-channel and colored versions of the processed masks.
    """
    # Mapping for single-channel ground truth
    label_mapping = {
        (0, 0, 0): 0,         # Background clutter
        (128, 0, 0): 7,       # Building
        (128, 64, 128): 1,    # Road
        (0, 128, 0): 3,       # Tree
        (128, 128, 0): 2,     # Low vegetation
        (64, 0, 128): 5,      # Moving car
        (192, 0, 192): 5,     # Static car (merged with Moving car)
        (64, 64, 0): 4        # Human
    }

    vdd_mapping = {
        5: 8,  # Roof
        6: 6   # Water
    }

    color_mapping = {
        0: (0, 0, 0),
        1: (128, 0, 128),
        2: (112, 148, 32),
        3: (64, 64, 0),
        4: (255, 16, 255),
        5: (0, 128, 128),
        6: (0, 0, 255),
        7: (255, 0, 0),
        8: (64, 160, 120)
    }

    # Load label and VDD label images
    label_img = cv2.imread(label_path, cv2.IMREAD_COLOR)
    label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB) # Convert to RGB
    vdd_label_img = cv2.imread(vdd_label_path, cv2.IMREAD_GRAYSCALE)

    if label_img is None or vdd_label_img is None:
        print(f"Failed to load {label_path} or {vdd_label_path}")
        return

    # Create a blank single-channel mask
    merged_mask = np.zeros(label_img.shape[:2], dtype=np.uint8)

    # Map the labels using exact matching
    for color, class_id in label_mapping.items():
        color_array = np.array(color, dtype=np.uint8)
        mask = np.all(label_img == color_array, axis=-1)  # Exact match
        merged_mask[mask] = class_id

    # Overlay VDD labels with conditions
    for vdd_class, new_class in vdd_mapping.items():
        vdd_mask = (vdd_label_img == vdd_class)
        
        if new_class == 6:  # Water
            # Overlay water only on background (class 0)
            overlay_mask = (merged_mask == 0) & vdd_mask
            merged_mask[overlay_mask] = new_class
        elif new_class == 8:  # Roof
            # Overlay roof only on building (class 7)
            overlay_mask = (merged_mask == 7) & vdd_mask
            merged_mask[overlay_mask] = new_class
        else:
            # Skip overlay for other cases
            continue

    # Save the single-channel mask
    cv2.imwrite(dest_mask_path, merged_mask)

    # Create and save the colored mask
    color_mask = np.zeros((*merged_mask.shape, 3), dtype=np.uint8)
    for class_id, color in color_mapping.items():
        color_mask[merged_mask == class_id] = color

    # Convert color_mask to BGR
    color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
    cv2.imwrite(dest_color_mask_path, color_mask)

def process_sequence(args):
    """
    Processes a single sequence and merges masks serially.
    """
    seq, dataset_path, mask_path, dest_dir, data_dir, uavid_val_seq = args

    origin_dir = os.path.join(dest_dir, seq, "origin")
    mask_dir = os.path.join(dest_dir, seq, "mask")
    color_mask_dir = os.path.join(dest_dir, seq, "mask_colored")
    os.makedirs(origin_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(color_mask_dir, exist_ok=True)

    seq_path = os.path.join(dataset_path, seq)
    image_seq_path = os.path.join(seq_path, "FF_Images")
    label_seq_path = os.path.join(seq_path, "Labels")
    vdd_label_seq_path = os.path.join(mask_path, seq, "VDD_Labels")

    # Move images using threads
    if os.path.exists(image_seq_path):
        image_files = [(os.path.join(image_seq_path, file), os.path.join(origin_dir, file))
                       for file in os.listdir(image_seq_path) if os.path.isfile(os.path.join(image_seq_path, file))]
        move_files_thread(image_files)

    # Process masks serially
    if os.path.exists(label_seq_path) and os.path.exists(vdd_label_seq_path):
        for file in os.listdir(label_seq_path):
            label_file = os.path.join(label_seq_path, file)
            vdd_file = os.path.join(vdd_label_seq_path, file)
            if os.path.isfile(label_file) and os.path.isfile(vdd_file):
                dest_mask_file = os.path.join(mask_dir, file)
                dest_color_mask_file = os.path.join(color_mask_dir, file)
                process_and_merge_masks(label_file, vdd_file, dest_mask_file, dest_color_mask_file)

def organize_uavid(dataset_dir):
    """
    Organize the UAVid dataset into the desired structure and process sequences.
    """
    source_images = os.path.join(dataset_dir, "uavid_v1.5_official_release")
    source_masks = os.path.join(dataset_dir, "UAVid7")
    dest_dir = os.path.join(dataset_dir, "UAVid/downsampled_original_data")
    data_dir = os.path.join(dataset_dir, "UAVid/data")
    txt_dir = os.path.join(dataset_dir, "UAVid")

    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)

    train_txt_path = os.path.join(txt_dir, "train.txt")
    val_txt_path = os.path.join(txt_dir, "val.txt")

    uavid_val_seq = ["seq16", "seq17", "seq18", "seq19", "seq20", "seq36", "seq37"]

    for dataset_type in ["uavid_train", "uavid_val"]:
        dataset_path = os.path.join(source_images, dataset_type)
        mask_path = os.path.join(source_masks, dataset_type)

        if not os.path.exists(dataset_path):
            print(f"Dataset path {dataset_path} does not exist. Skipping.")
            continue

        sequences = [seq for seq in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, seq))]

        args = [
            (seq, dataset_path, mask_path, dest_dir, data_dir, uavid_val_seq)
            for seq in sequences
        ]

        with Pool(processes=8) as pool:
            pool.map(process_sequence, args)

    # Generate clips
    generate_clips(dest_dir, data_dir, train_txt_path, val_txt_path)


def generate_clips(source_dir, data_dir, train_txt_path, val_txt_path):
    """
    Generates clips for images and masks under the 'data' folder.
    The structure is: UAVid/data/{global_clip_index}_{sequence_name}_{local_clip_index}.
    """
    clip_length = 100  # Number of frames per clip
    min_clip_length = 50  # Minimum number of frames to consider a clip
    global_clip_index = 0

    uavid_val_seq = ["seq16", "seq17", "seq18", "seq19", "seq20", "seq36", "seq37"]

    with open(train_txt_path, 'w') as train_file, open(val_txt_path, 'w') as val_file:
        seq_list = os.listdir(source_dir)
        with tqdm(total=len(seq_list), desc="Processing sequences") as pbar:
            for seq in seq_list:
                seq_path = os.path.join(source_dir, seq)
                origin_dir = os.path.join(seq_path, "origin")
                mask_dir = os.path.join(seq_path, "mask")

                if not os.path.exists(origin_dir) or not os.path.exists(mask_dir):
                    print(f"Origin or mask directory missing for sequence {seq}. Skipping.")
                    pbar.update(1)
                    continue

                # Gather frames and skip the first frame
                frames = sorted(os.listdir(origin_dir))[1:]  # Skip the first frame (index 0)

                # Divide frames into clips
                clips = [frames[i:i + clip_length] for i in range(0, len(frames), clip_length)]

                for local_clip_index, clip in enumerate(clips):
                    # Skip incomplete clips
                    if len(clip) < min_clip_length:
                        print(f"Skipping clip {global_clip_index}_{seq}_{local_clip_index} with {len(clip)} frames (less than {min_clip_length}).")
                        continue

                    clip_name = f"{global_clip_index}_{seq}_{local_clip_index}"
                    clip_dir = os.path.join(data_dir, clip_name)
                    images_clip_dir = os.path.join(clip_dir, "origin")
                    masks_clip_dir = os.path.join(clip_dir, "mask")
                    os.makedirs(images_clip_dir, exist_ok=True)
                    os.makedirs(masks_clip_dir, exist_ok=True)

                    # Prepare file pairs for copying
                    image_file_pairs = [
                        (os.path.join(origin_dir, frame), os.path.join(images_clip_dir, frame))
                        for frame in clip if os.path.isfile(os.path.join(origin_dir, frame))
                    ]
                    mask_file_pairs = [
                        (os.path.join(mask_dir, frame), os.path.join(masks_clip_dir, frame))
                        for frame in clip if os.path.isfile(os.path.join(mask_dir, frame))
                    ]

                    # multithreading for copying files
                    copy_files_thread(image_file_pairs)
                    copy_files_thread(mask_file_pairs)

                    # Write clip name to the appropriate file
                    if seq in uavid_val_seq:
                        val_file.write(clip_name + '\n')
                    else:
                        train_file.write(clip_name + '\n')

                    global_clip_index += 1

                # Update the progress bar after processing the sequence
                pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Organize UAVid dataset.')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the dataset directory.')
    args = parser.parse_args()

    organize_uavid(args.dataset_dir)


# def organize_uavid(dataset_dir):
#     source_images = os.path.join(dataset_dir, "uavid_v1.5_official_release")
#     source_masks = os.path.join(dataset_dir, "UAVid7")
#     dest_dir = os.path.join(dataset_dir, "UAVid/downsampled_original_data")
#     data_dir = os.path.join(dataset_dir, "UAVid/data")
#     txt_dir = os.path.join(dataset_dir, "UAVid")
    
#     # Create the base directory
#     os.makedirs(dest_dir, exist_ok=True)
#     os.makedirs(data_dir, exist_ok=True)
#     os.makedirs(txt_dir, exist_ok=True)

#     train_txt_path = os.path.join(txt_dir, "train.txt")
#     val_txt_path = os.path.join(txt_dir, "val.txt")
    
#     # Process train and validation sets
#     for dataset_type in ["uavid_train", "uavid_val"]:
#         dataset_path = os.path.join(source_images, dataset_type)
#         mask_path = os.path.join(source_masks, dataset_type)

#         # Check if dataset path exists
#         if not os.path.exists(dataset_path):
#             print(f"Dataset path {dataset_path} does not exist. Skipping.")
#             continue

#         sequences = [seq for seq in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, seq))]
#         with tqdm(total=len(sequences), desc=f"Processing {dataset_type}") as pbar:
#             # Iterate through sequences in the dataset
#             for seq in sequences:
#                 seq_path = os.path.join(dataset_path, seq)

#                 # Create destination directories
#                 origin_dir = os.path.join(dest_dir, seq, "origin")
#                 mask_dir = os.path.join(dest_dir, seq, "mask")
#                 color_mask_dir = os.path.join(dest_dir, seq, "mask_colored")
#                 os.makedirs(origin_dir, exist_ok=True)
#                 os.makedirs(mask_dir, exist_ok=True)
#                 os.makedirs(color_mask_dir, exist_ok=True)

#                 # Process image files move FF_Images to origin_dir
#                 image_seq_path = os.path.join(seq_path, "FF_Images")
#                 if os.path.exists(image_seq_path):
#                     for file in os.listdir(image_seq_path):
#                         src_img = os.path.join(image_seq_path, file)
#                         dest_img = os.path.join(origin_dir, file)
#                         if os.path.isfile(src_img):
#                             shutil.move(src_img, dest_img)

#                 # Process mask files
#                 label_seq_path = os.path.join(seq_path, "Labels")
#                 vdd_label_seq_path = os.path.join(mask_path, seq, "VDD_Labels")

#                 if os.path.exists(label_seq_path) and os.path.exists(vdd_label_seq_path):
#                     for file in os.listdir(label_seq_path):
#                         label_file = os.path.join(label_seq_path, file)
#                         vdd_file = os.path.join(vdd_label_seq_path, file)
#                         if os.path.isfile(label_file) and os.path.isfile(vdd_file):
#                             dest_mask_file = os.path.join(mask_dir, file)
#                             dest_color_mask_file = os.path.join(color_mask_dir, file)
#                             process_and_merge_masks(label_file, vdd_file, dest_mask_file, dest_color_mask_file)
#                 else:
#                     print(f"Labels or VDD_Labels folder missing for sequence {seq}. Skipping.")

#                 # Update progress bar
#                 pbar.update(1)

#     # Generate clips in the "UAVid/data" folder and save clip names to txt files
#     generate_clips(dest_dir, data_dir, train_txt_path, val_txt_path)