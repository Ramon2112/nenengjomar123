import cv2
import numpy as np
import os

def apply_dawn_effect(image_path, output_folder="output"):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image not found: {image_path}")
        return False

    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.splitext(os.path.basename(image_path))[0]

    rows, cols, _ = img.shape

    # Step 1: Create blank mask for rays
    mask = np.zeros((rows, cols), dtype=np.float32)

    # Step 2: Draw multiple lines from top-left to center with small angle variations
    num_rays = 5  # number of rays
    center_x, center_y = cols // 2, rows // 2

    for i in range(num_rays):
        # Slightly randomize the end point for each ray
        end_x = center_x + np.random.randint(-50, 50)
        end_y = center_y + np.random.randint(-50, 50)
        thickness = np.random.randint(20, 50)  # varying thickness

        cv2.line(mask, (0, 0), (end_x, end_y), 1.0, thickness)

    # Step 3: Apply Gaussian blur to smooth rays
    mask = cv2.GaussianBlur(mask, (101, 101), 0)
    mask = np.clip(mask, 0, 1)

    # Step 4: Convert mask to 3 channels
    mask_3ch = cv2.merge([mask, mask, mask])

    # Step 5: Apply the light rays to the image
    img_float = img.astype(np.float32) / 255.0
    result = img_float + mask_3ch * 0.7  # adjust intensity
    result = np.clip(result, 0, 1)
    result = (result * 255).astype(np.uint8)

    # Step 6: Warm overlay for dawn feel
    warm_overlay = np.full_like(result, (30, 60, 120))  # BGR warm tone
    result = cv2.addWeighted(result, 0.85, warm_overlay, 0.15, 0)

    # Save output
    cv2.imwrite(f"{output_folder}/{filename}_multiple_rays.png", result)
    print(f"Multiple light rays applied: {filename}_multiple_rays.png")
    return True


def process_folder(input_folder):
    for file in os.listdir(input_folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(input_folder, file)
            apply_dawn_effect(path)


if __name__ == "__main__":
    folder = input("Enter folder path containing images: ")
    process_folder(folder)
