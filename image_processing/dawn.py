import cv2
import numpy as np
import os

def apply_dawn_effect(image_path, output_folder="output"):
    """
    Applies cinematic dawn rays effect from top-left corner to center.
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
    
    # Center of the image
    center_x, center_y = cols // 2, rows // 2
    
    # Create coordinate grid
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Vector from top-left (0,0) to each pixel
    dx = X  # Distance along x from left
    dy = Y  # Distance along y from top
    
    # Normalized distance from top-left (0,0) to center direction
    # Create direction vector from top-left to center
    center_dir_x = center_x
    center_dir_y = center_y
    
    # Normalize the direction vector
    center_dir_length = np.sqrt(center_dir_x**2 + center_dir_y**2)
    center_dir_x_norm = center_dir_x / center_dir_length
    center_dir_y_norm = center_dir_y / center_dir_length
    
    # For each pixel, compute vector from top-left to pixel
    pixel_dir_x = X
    pixel_dir_y = Y
    
    # Avoid division by zero for pixel at (0,0)
    pixel_length = np.sqrt(pixel_dir_x**2 + pixel_dir_y**2)
    pixel_length[pixel_length == 0] = 1  # Avoid division by zero
    
    pixel_dir_x_norm = pixel_dir_x / pixel_length
    pixel_dir_y_norm = pixel_dir_y / pixel_length
    
    # Dot product to measure alignment with center direction
    # 1.0 means perfectly aligned, 0.0 means perpendicular
    alignment = pixel_dir_x_norm * center_dir_x_norm + pixel_dir_y_norm * center_dir_y_norm
    
    # Create a mask for rays - only in direction towards center
    # Use alignment factor with threshold
    ray_mask = np.maximum(alignment, 0)  # Keep only positive alignments (towards center)
    
    # Sharpen the mask to be more directional
    ray_mask = np.power(ray_mask, 4)  # Higher power makes it more directional
    
    # Distance-based falloff
    distance_from_corner = np.sqrt(dx**2 + dy**2)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    distance_norm = distance_from_corner / max_distance
    
    # Create rays - stronger near the center line, fading with distance
    rays = np.exp(-3 * distance_norm) * ray_mask
    
    # Add radial streaks for realism (aligned with direction)
    streak_angle = np.arctan2(center_dir_y, center_dir_x)
    streak_pattern = 0.15 * np.sin(0.05 * (X * np.cos(streak_angle) + Y * np.sin(streak_angle)))
    rays += streak_pattern
    
    # Apply vignette-like falloff at edges
    vignette_x = 1.0 - np.abs(X - center_x) / center_x
    vignette_y = 1.0 - np.abs(Y - center_y) / center_y
    vignette = vignette_x * vignette_y
    rays *= vignette
    
    rays = np.clip(rays, 0, 1)
    
    # Warm cinematic dawn color (BGR)
    dawn_color = np.array([1.0, 0.85, 0.6])  # Orange-yellow dawn light
    overlay = np.zeros_like(img_float)
    for i in range(3):
        overlay[:, :, i] = dawn_color[i] * rays
    
    # Enhanced blending - stronger effect in darker areas
    # Convert to grayscale for brightness mask
    gray = cv2.cvtColor(img_float, cv2.COLOR_BGR2GRAY)
    brightness_mask = 1.0 - gray  # Darker areas get more light
    
    # Blend overlay with original image using adaptive blending
    blend_strength = 0.5  # Overall strength
    adaptive_blend = blend_strength * (0.7 + 0.3 * brightness_mask)
    
    result = img_float * (1 - adaptive_blend) + overlay * adaptive_blend
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
