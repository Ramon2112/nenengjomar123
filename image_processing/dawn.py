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

    # Step 1: Create a blank mask for the light ray
    mask = np.zeros((rows, cols), dtype=np.float32)

    # Line from top-left to center
    start_point = (0, 0)
    end_point = (cols//2, rows//2)
    thickness = 50  # width of light ray

    # Draw a white line on the mask
    cv2.line(mask, start_point, end_point, 1.0, thickness)

    # Step 2: Apply motion blur to make it soft
    # Create a motion blur kernel along the line direction
    kernel_size = 101
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    cv2.line(kernel, (0, 0), (kernel_size-1, kernel_size-1), 1.0, 1)
    kernel /= kernel.sum()  # normalize

    mask = cv2.filter2D(mask, -1, kernel)
    mask = np.clip(mask, 0, 1)

    # Step 3: Convert mask to 3 channels
    mask_3ch = cv2.merge([mask, mask, mask])

    # Step 4: Apply the light ray mask to the image
    img_float = img.astype(np.float32) / 255.0
    result = img_float + mask_3ch * 0.8  # adjust intensity of ray
    result = np.clip(result, 0, 1)
    result = (result * 255).astype(np.uint8)

    # Step 5: Optional warm overlay
    warm_overlay = np.full_like(result, (30, 60, 120))  # BGR warm tone
    result = cv2.addWeighted(result, 0.85, warm_overlay, 0.15, 0)

    # Save output
    cv2.imwrite(f"{output_folder}/{filename}_light_ray.png", result)
    print(f"Light ray effect applied: {filename}_light_ray.png")
    return True


def process_folder(input_folder):
    for file in os.listdir(input_folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(input_folder, file)
            apply_dawn_effect(path)


if __name__ == "__main__":
    folder = input("Enter folder path containing images: ")
    process_folder(folder)
