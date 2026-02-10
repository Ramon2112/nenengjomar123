import os
import sys

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_processing import crack_detector

INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "test_output"


def test_input_folder_exists():
    assert os.path.exists(INPUT_FOLDER), "input_images folder is missing"


def test_images_are_processed():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    processed_any = False

    for file in os.listdir(INPUT_FOLDER):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            processed_any = True
            path = os.path.join(INPUT_FOLDER, file)
            success = crack_detector.detect_cracks(path, OUTPUT_FOLDER)
            assert success is True, f"Processing failed for {file}"

    assert processed_any, "No test images found in input_images folder"


def test_output_files_created():
    for file in os.listdir(INPUT_FOLDER):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            name = os.path.splitext(file)[0]

            assert os.path.exists(f"{OUTPUT_FOLDER}/{name}_edges.png")
            assert os.path.exists(f"{OUTPUT_FOLDER}/{name}_mask.png")
            assert os.path.exists(f"{OUTPUT_FOLDER}/{name}_detected.png")
