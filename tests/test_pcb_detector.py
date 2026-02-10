import os
import sys

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_processing import pcb_fault_detector  # replace with your module name if different

INPUT_FOLDER = "input"      # folder containing PCB images
OUTPUT_FOLDER = "output"    # where processed images will be saved


def test_input_folder_exists():
    """Check if input folder exists"""
    assert os.path.exists(INPUT_FOLDER), f"Input folder '{INPUT_FOLDER}' is missing"


def test_images_are_processed():
    """Test that detect_faults processes all images successfully"""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    processed_any = False
    for file in os.listdir(INPUT_FOLDER):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            processed_any = True
            img_path = os.path.join(INPUT_FOLDER, file)
            # Use the same image as reference and test
            success = pcb_fault_detector.detect_faults(img_path, img_path, OUTPUT_FOLDER)
            assert success is True, f"Processing failed for {file}"

    assert processed_any, "No images found in input folder"


def test_output_files_created():
    """Check that output files are created"""
    for file in os.listdir(INPUT_FOLDER):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            name = os.path.splitext(file)[0]
            output_file = os.path.join(OUTPUT_FOLDER, f"{name}_faults.png")
            assert os.path.exists(output_file), f"Output file not created: {output_file}"


def test_invalid_image_path():
    """Test detect_faults with invalid image path"""
    result = pcb_fault_detector.detect_faults("nonexistent_ref.png", "nonexistent_test.png", OUTPUT_FOLDER)
    assert result is None
