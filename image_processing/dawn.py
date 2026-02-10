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
    
    # Create strong directional rays from top-left
    # Main beam direction
    angle_to_center = np.arctan2(center_y, center_x)
    
    # Distance from top-left corner
    distance_from_corner = np.sqrt(X**2 + Y**2)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    distance_norm = distance_from_corner / max_distance
    
    # Create strong central beam
    # Angle from top-left to each pixel
    angle = np.arctan2(Y, X)
    
    # How well aligned with the center direction
    angle_diff = np.abs(angle - angle_to_center)
    
    # STRONG RAY EFFECT - Multiple layers
    # 1. Main central beam (very strong and focused)
    main_beam = np.exp(-2 * distance_norm) * np.exp(-8 * angle_diff)
    main_beam = np.power(main_beam, 0.5)  # Make it stronger
    
    # 2. Secondary beams (slightly spread out)
    for offset in [-0.1, 0.1, -0.2, 0.2]:
        angle_diff_offset = np.abs(angle - (angle_to_center + offset))
        secondary = np.exp(-3 * distance_norm) * np.exp(-10 * angle_diff_offset)
        main_beam += 0.4 * secondary
    
    # 3. Volumetric rays (atmospheric scattering effect)
    for i in range(3):
        freq = 0.08 + i * 0.02
        volumetric = np.sin(freq * (X * np.cos(angle_to_center) + Y * np.sin(angle_to_center)) + i * 1.0)
        volumetric *= np.exp(-4 * distance_norm)
        main_beam += 0.2 * volumetric
    
    # 4. Strong falloff from corner
    corner_falloff = 1.0 - distance_norm
    corner_falloff = np.power(corner_falloff, 0.8)
    main_beam *= corner_falloff
    
    # 5. Add god rays/caustic patterns
    god_rays = np.zeros_like(main_beam)
    for i in range(5):
        pattern = np.sin(0.03 * X + 0.05 * Y + i * 0.5) * \
                  np.cos(0.04 * X - 0.02 * Y + i * 0.3)
        pattern_mask = np.exp(-5 * angle_diff) * np.exp(-2 * distance_norm)
        god_rays += 0.15 * pattern * pattern_mask
    
    # Combine all ray effects
    rays = main_beam + god_rays
    
    # 6. Add radial blur effect along direction
    blur_strength = 0.3
    for offset in np.linspace(-3, 3, 7):
        x_offset = int(offset * np.cos(angle_to_center))
        y_offset = int(offset * np.sin(angle_to_center))
        x_shifted = np.clip(X + x_offset, 0, cols-1)
        y_shifted = np.clip(Y + y_offset, 0, rows-1)
        rays += 0.1 * rays[y_shifted.astype(int), x_shifted.astype(int)]
    
    # 7. Create hot spot at corner
    corner_hotspot = np.exp(-10 * distance_norm)
    corner_hotspot = np.power(corner_hotspot, 0.3)  # Very bright at corner
    rays += 0.8 * corner_hotspot
    
    # Normalize and enhance contrast
    rays = np.clip(rays, 0, 1)
    rays = np.power(rays, 0.7)  # Boost mid-tones
    
    # Apply global intensity curve
    rays = 1.5 * rays - 0.3 * rays**2  # S-curve for more punch
    
    # Add lens flare effect at corner
    flare_x = X * 0.8
    flare_y = Y * 0.8
    flare_distance = np.sqrt(flare_x**2 + flare_y**2)
    flare = np.exp(-0.01 * flare_distance)
    flare *= np.sin(0.02 * flare_distance) * 0.5 + 0.5
    rays += 0.4 * flare
    
    # Final normalization
    rays = np.clip(rays, 0, 1)
    
    # INTENSE DAWN COLOR - Very warm and saturated
    dawn_color = np.array([0.9, 0.7, 0.4])  # More orange, less yellow
    overlay = np.zeros_like(img_float)
    for i in range(3):
        overlay[:, :, i] = dawn_color[i] * rays
    
    # Enhanced blending with color mixing
    # Convert to grayscale for brightness mask
    gray = cv2.cvtColor(img_float, cv2.COLOR_BGR2GRAY)
    
    # Invert so dark areas get more light
    brightness_mask = 1.0 - gray
    
    # Screen blend mode for stronger effect
    # Instead of simple alpha blending, use screen blend for more vibrant result
    result_screen = 1.0 - (1.0 - img_float) * (1.0 - overlay)
    
    # Strong overlay blend
    blend_strength = 0.7  # Very strong
    adaptive_blend = blend_strength * (0.5 + 0.5 * brightness_mask)
    
    # Add some color to original image too (make it warmer)
    img_warm = img_float.copy()
    img_warm[:, :, 0] *= 0.9  # Reduce blue
    img_warm[:, :, 1] *= 1.1  # Increase green slightly
    img_warm[:, :, 2] *= 1.2  # Increase red significantly
    
    # Blend: weighted average of original, warmed original, and screen blend
    adaptive_blend_3d = adaptive_blend[:, :, np.newaxis]
    
    # Screen blend gives stronger light effect
    result = img_warm * (1 - 0.3 * adaptive_blend_3d) + result_screen * adaptive_blend_3d
    
    # Add some global glow (blurred rays mixed in)
    rays_blurred = cv2.GaussianBlur(rays, (0, 0), sigmaX=15, sigmaY=15)
    glow_overlay = np.zeros_like(img_float)
    for i in range(3):
        glow_overlay[:, :, i] = dawn_color[i] * rays_blurred * 0.3
    
    result = result + glow_overlay
    
    # Final adjustments
    result = np.clip(result, 0, 1)
    
    # Increase overall brightness and contrast
    result = np.power(result, 0.9)  # Gamma correction
    result = result * 1.1 - 0.05  # Brightness/contrast
    
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
