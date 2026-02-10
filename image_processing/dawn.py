import cv2
import numpy as np
import os

def apply_dawn_effect(image_path, output_folder="output"):
    """
    Applies a bright dawn effect with rays from top-left to center-right.
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

    # Create a directional light ray gradient
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Direction vector from top-left (0,0) to center-right (cols, rows/2)
    target_x = cols
    target_y = rows // 2
    dx = target_x - 0
    dy = target_y - 0
    length = np.sqrt(dx**2 + dy**2)

    # Normalize distance along ray direction
    distance_along_ray = ((X * dx + Y * dy) / (length**2))
    distance_along_ray = np.clip(distance_along_ray, 0, 1)

    # Create bright rays effect (exponential falloff)
    rays = np.exp(-3 * (1 - distance_along_ray))  # stronger near start
    rays = np.clip(rays, 0, 1)

    # Optional: add subtle streaks for realism
    rays += 0.2 * np.sin(0.05 * X + 0.03 * Y)
    rays = np.clip(rays, 0, 1)

    # Dawn color overlay (BGR)
    dawn_color = np.array([1.0, 0.85, 0.6])  # warmer and brighter
    overlay = np.zeros_like(img_float)
    for i in range(3):
        overlay[:, :, i] = dawn_color[i] * rays

    # Blend overlay with original image (adjust alpha for intensity)
    result = img_float * 0.6 + overlay * 0.4
    result = np.clip(result, 0, 1)
    result = (result * 255).astype(np.uint8)

    cv2.imwrite(output_path, result)
    print(f"Bright dawn rays effect applied: {output_path}")
    return True

def process_folder(input_folder, output_folder="output"):
    """
    Processes all images in a folder to apply bright dawn rays effect.
    """
    for file in os.listdir(input_folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(input_folder, file)
            apply_dawn_effect(path, output_folder)

if __name__ == "__main__":
    folder = input("Enter folder path containing images: ")
    process_folder(folder)
