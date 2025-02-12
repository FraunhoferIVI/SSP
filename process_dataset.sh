#!/bin/bash

# Define the dataset directory and script paths
DATASET_DIR="datasets"
TOOLS_DIR="tools"
EXTRACT_UAVID_FRAMES_SCRIPT="extract_frames_uavid.py"
ORGANIZE_UAVID_SCRIPT="organize_uavid.py"
EXTRACT_RURALSCAPES_FRAMES_SCRIPT="extract_frames_ruralscapes.py"
ORGANIZE_RURALSCAPES_SCRIPT="organize_ruralscapes.py"

# Define ZIP file paths
UAVID_ZIP="$DATASET_DIR/uavid_v1.5_official_release.zip"
UAVID_EXTENDED_ZIP="$DATASET_DIR/UAVid7-20241118T183546Z-001.zip" # Please replace this with the correct file name - the provided file name is just an example
RURALSCAPES_ZIP="$DATASET_DIR/Ruralscapes.zip" # Please ensure that the zip file is not corrupted
DRONESCAPES_ZIP="$DATASET_DIR/Dronescapes.zip" # Please note that the Dronescapes dataset is large and may take a while to extract - is already preprocessed

# Check if the dataset directory exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "Dataset directory $DATASET_DIR does not exist."
    exit 1
fi

# # Process UAVid datasets
if [ -f "$UAVID_ZIP" ] && [ -f "$UAVID_EXTENDED_ZIP" ]; then
    echo "UAVid ZIP files found. Proceeding with extraction and organization."

    # Unzip UAVid datasets
    unzip -o "$UAVID_ZIP" -d "$DATASET_DIR"
    unzip -o "$UAVID_EXTENDED_ZIP" -d "$DATASET_DIR"

    # Run the Python scripts for UAVid
    python "$TOOLS_DIR/$EXTRACT_UAVID_FRAMES_SCRIPT" --dataset_path "$DATASET_DIR/uavid_v1.5_official_release" --resize
    python "$TOOLS_DIR/$ORGANIZE_UAVID_SCRIPT" --dataset_dir "$DATASET_DIR"
else
    echo "UAVid ZIP files not found. Skipping UAVid processing."
fi

# Process Ruralscapes dataset
if [ -f "$RURALSCAPES_ZIP" ]; then
    echo "Ruralscapes ZIP file found. Proceeding with extraction and organization."

    # # Unzip Ruralscapes dataset
    unzip -o "$RURALSCAPES_ZIP" -d "$DATASET_DIR"

    # Run the Python scripts for Ruralscapes
    python "$TOOLS_DIR/$EXTRACT_RURALSCAPES_FRAMES_SCRIPT" --dataset_path "$DATASET_DIR/Ruralscapes" --resize
    python "$TOOLS_DIR/$ORGANIZE_RURALSCAPES_SCRIPT" --dataset_dir "$DATASET_DIR"
else
    echo "Ruralscapes ZIP file not found. Skipping Ruralscapes processing."
fi

# Process Dronescapes dataset
if [ -f "$DRONESCAPES_ZIP" ]; then
    echo "Dronescapes ZIP file found. Proceeding with extraction and organization."

    # # Unzip Dronescapes dataset
    unzip -o "$DRONESCAPES_ZIP" -d "$DATASET_DIR"
else
    echo "Dronescapes ZIP file not found. Skipping Dronescapes processing."
fi

echo "Script execution completed."
