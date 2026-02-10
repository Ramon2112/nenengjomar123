import cv2
import numpy as np
import os

def apply_dawn_effect(image_path, output_folder="output"):
    """
    Applies a dawn effect with light from top-left to a single image and saves it.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image not found: {image_path}")
        return False

    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_folder, f"{filename}_dawn.png")

    # Convert image to float for processing
    img_float = img.astype(np.float32) / 255.0
    rows, cols, _ = img.shape

    # Create top-left light gradient
    x = np.linspace(0, 1, cols)
    y = np.linspace(0, 1, rows)
    X, Y = np.meshgrid(x, y)
    gradient = (1 - Y) * (1 - X * 0.8)  # Light stronger at top-left

    # Dawn color overlay (BGR)
    dawn_color = np.array([1.0, 0.8, 0.6])
    overlay = np.zeros_like(img_float)
    for i in range(3):
        overlay[:, :, i] = dawn_color[i] * gradient

    # Blend original image with overlay
    result = img_float * 0.7 + overlay * 0.3
    result = np.clip(result, 0, 1)
    result = (result * 255).astype(np.uint8)

    cv2.imwrite(output_path, result)
    print(f"Dawn effect applied: {output_path}")
    return True

def process_folder(input_folder, output_folder="output"):
    """
    Processes all images in a folder to apply dawn effect.
    """
    for file in os.listdir(input_folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(input_folder, file)
            apply_dawn_effect(path, output_folder)

if __name__ == "__main__":
    folder = input("Enter folder path containing images: ")
    process_folder(folder)
