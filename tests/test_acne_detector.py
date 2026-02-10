import os
import sys

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(_file_), '..')))

from acne_detector import detect_acne  # make sure your file is acne_detector.py


INPUT_FOLDER = "input"   # folder containing real face images
OUTPUT_FOLDER = "output"  # where processed images will be saved


def test_input_folder_exists():
    """Check if input folder exists"""
    assert os.path.exists(INPUT_FOLDER), "input folder is missing"


def test_images_are_processed():
    """Test that detect_acne processes all images successfully"""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    processed_any = False

    for file in os.listdir(INPUT_FOLDER):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            processed_any = True
            path = os.path.join(INPUT_FOLDER, file)
            success = acne_detector.detect_acne(path, OUTPUT_FOLDER)
            assert success is True, f"Processing failed for {file}"

    assert processed_any, "No test images found in input folder"


def test_output_files_created():
    """Check that output files are created"""
    for file in os.listdir(INPUT_FOLDER):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            name = os.path.splitext(file)[0]
            output_file = f"{OUTPUT_FOLDER}/{name}_acne.png"
            assert os.path.exists(output_file), f"Output file not created: {output_file}"
