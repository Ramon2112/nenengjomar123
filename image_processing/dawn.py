import cv2
import numpy as np
import os

def apply_dawn_effect(image_path, output_folder="output"):
    """
    Applies cinematic dawn rays effect similar to uploaded reference image.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image not found: {image_path}")
        return False

    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_folder, f"{filename}_cinematic_dawn.png")

    img_float = img.astype(np.float32) / 255.0
    rows, cols, _ = img.shape

    # Create coordinate grid
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

    # Light source at top-left
    light_x, light_y = 0, 0

    # Vector from light source to each pixel
    dx = X - light_x
    dy = Y - light_y

    # Angle of rays for directional spread
    angle = np.arctan2(dy, dx)
    
    # Radial distance from light
    distance = np.sqrt(dx**2 + dy**2)
    distance_norm = distance / np.max(distance)

    # Create light ray intensity with directional focus
    # Stronger along diagonal (~45 degrees) and exponential falloff
    target_angle = np.pi / 4  # 45 degrees diagonal
    angle_diff = np.abs(angle - target_angle)
    rays = np.exp(-5 * distance_norm) * np.exp(-5 * angle_diff)
    
    # Add subtle streaks for realism
    rays += 0.2 * np.sin(0.1 * X + 0.05 * Y)
    rays = np.clip(rays, 0, 1)

    # Warm cinematic dawn color (BGR)
    dawn_color = np.array([1.0, 0.85, 0.6])
    overlay = np.zeros_like(img_float)
    for i in range(3):
        overlay[:, :, i] = dawn_color[i] * rays

    # Blend overlay with original image
    result = img_float * 0.6 + overlay * 0.4
    result = np.clip(result, 0, 1)
    result = (result * 255).astype(np.uint8)

    cv2.imwrite(output_path, result)
    print(f"Cinematic dawn rays effect applied: {output_path}")
    return True

def process_folder(input_folder, output_folder="output"):
    """
    Process all images in a folder to apply cinematic dawn rays effect.
    """
    for file in os.listdir(input_folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(input_folder, file)
            apply_dawn_effect(path, output_folder)

if __name__ == "__main__":
    folder = input("Enter folder path containing images: ")
    process_folder(folder)
