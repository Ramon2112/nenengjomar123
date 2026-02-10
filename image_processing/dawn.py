import cv2
import numpy as np
import os

def apply_dawn_effect(image_path, output_folder="output"):
    """
    Applies a dawn effect with light rays from top-left to a single image and saves it.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image not found: {image_path}")
        return False

    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_folder, f"{filename}_dawn_rays.png")

    img_float = img.astype(np.float32) / 255.0
    rows, cols, _ = img.shape

    # Create radial gradient for light rays
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    cx, cy = 0, 0  # Light source at top-left corner
    distance = np.sqrt((X - cx)**2 + (Y - cy)**2)
    distance = distance / np.max(distance)  # Normalize to 0-1

    # Create radial light intensity with streak effect
    rays = np.exp(-3 * distance)  # exponential falloff
    rays += 0.15 * np.sin(0.05 * X + 0.05 * Y)  # subtle streaks
    rays = np.clip(rays, 0, 1)

    # Dawn color overlay (BGR)
    dawn_color = np.array([1.0, 0.8, 0.6])
    overlay = np.zeros_like(img_float)
    for i in range(3):
        overlay[:, :, i] = dawn_color[i] * rays

    # Blend original image with overlay
    result = img_float * 0.7 + overlay * 0.3
    result = np.clip(result, 0, 1)
    result = (result * 255).astype(np.uint8)

    cv2.imwrite(output_path, result)
    print(f"Dawn rays effect applied: {output_path}")
    return True

def process_folder(input_folder, output_folder="output"):
    """
    Processes all images in a folder to apply dawn rays effect.
    """
    for file in os.listdir(input_folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(input_folder, file)
            apply_dawn_rays(path, output_folder)

if __name__ == "__main__":
    folder = input("Enter folder path containing images: ")
    process_folder(folder)
