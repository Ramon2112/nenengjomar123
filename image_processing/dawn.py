import cv2
import numpy as np
import os

def apply_dawn_effect(image_path, output_folder="output"):
    """
    Applies strong dawn rays from the top-left corner to an image.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image not found: {image_path}")
        return False

    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_folder, f"{filename}_strong_dawn.png")

    img_float = img.astype(np.float32) / 255.0
    rows, cols, _ = img.shape

    # Create coordinate grid
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

    # Distance from top-left (light source)
    distance = np.sqrt(X**2 + Y**2)
    distance_norm = distance / np.max(distance)

    # Strong exponential falloff for bright rays
    rays = np.exp(-4 * distance_norm)  # stronger decay
    rays = np.clip(rays, 0, 1)

    # Optional: add slight radial blur effect by smoothing
    from scipy.ndimage import gaussian_filter
    rays = gaussian_filter(rays, sigma=20)

    # Dawn color overlay (BGR)
    dawn_color = np.array([1.0, 0.85, 0.6])  # warm orange
    overlay = np.zeros_like(img_float)
    for i in range(3):
        overlay[:, :, i] = dawn_color[i] * rays

    # Blend overlay with original image
    result = img_float * 0.6 + overlay * 0.4
    result = np.clip(result, 0, 1)
    result = (result * 255).astype(np.uint8)

    cv2.imwrite(output_path, result)
    print(f"Strong dawn rays effect applied: {output_path}")
    return True

def process_folder(input_folder, output_folder="output"):
    """
    Processes all images in a folder to apply strong dawn rays.
    """
    for file in os.listdir(input_folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(input_folder, file)
            apply_dawn_effect(path, output_folder)

if __name__ == "__main__":
    folder = input("Enter folder path containing images: ")
    process_folder(folder)
