import os
import sys

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_processing import jigsaw

INPUT_FOLDER = "input"      # folder containing images
OUTPUT_FOLDER = "output"    # where jigsaw images will be saved
GRID_SIZE = 3               # default grid size for tests


def test_input_folder_exists():
    """Check if input folder exists"""
    assert os.path.exists(INPUT_FOLDER), f"Input folder '{INPUT_FOLDER}' is missing"


def test_images_are_processed():
    """Test that create_jigsaw processes all images successfully"""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    processed_any = False
    for file in os.listdir(INPUT_FOLDER):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            processed_any = True
            img_path = os.path.join(INPUT_FOLDER, file)
            success = jigsaw.create_jigsaw(img_path, GRID_SIZE, OUTPUT_FOLDER)
            assert success is True, f"Processing failed for {file}"

    assert processed_any, "No images found in input folder"


def test_output_files_created():
    """Check that output files are created"""
    for file in os.listdir(INPUT_FOLDER):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            name = os.path.splitext(file)[0]
            output_file = os.path.join(OUTPUT_FOLDER, f"{name}_jigsaw.png")
            assert os.path.exists(output_file), f"Output file not created: {output_file}"


def test_invalid_image_path():
    """Test create_jigsaw with invalid image path"""
    result = jigsaw.create_jigsaw("nonexistent_image.png", GRID_SIZE, OUTPUT_FOLDER)
    assert result is None
