#!/bin/bash

# Define the dataset directory and script paths
DATASET_DIR="datasets"
TOOLS_DIR="tools"
EXTRACT_UAVID_FRAMES_SCRIPT="extract_frames_uavid.py"
ORGANIZE_UAVID_SCRIPT="organize_uavid.py"

# Navigate to the dataset directory for extraction
if [ ! -d "$DATASET_DIR" ]; then
    echo "Dataset directory $DATASET_DIR does not exist."
    exit 1
fi

# Unzip the original dataset
# unzip "$DATASET_DIR/uavid_v1.5_official_release.zip" -d "$DATASET_DIR"

# Unzip the extended uavid dataset
# unzip "$DATASET_DIR/UAVid7-20241118T183546Z-001.zip" -d "$DATASET_DIR"

# Run the Python script to extract frames
# python "$TOOLS_DIR/$EXTRACT_UAVID_FRAMES_SCRIPT" --dataset_path "$DATASET_DIR/uavid_v1.5_official_release" --resize

# Run the Python script to organize files
python "$TOOLS_DIR/$ORGANIZE_UAVID_SCRIPT" --dataset_dir "$DATASET_DIR"