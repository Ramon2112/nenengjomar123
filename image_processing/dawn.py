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

    # Create light rays mask
    mask = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = max(0, 1 - ((i + j) / (rows + cols)))  # intensity from top-left

    # Optional subtle rays pattern
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    ray_pattern = np.sin((X + Y * 0.5) * 0.02)
    mask += ray_pattern * 0.3
    mask = np.clip(mask, 0, 1)

    mask_3ch = cv2.merge([mask, mask, mask])
    img_float = img.astype(np.float32) / 255.0
    dawn_img = img_float + mask_3ch * 0.5
    dawn_img = np.clip(dawn_img, 0, 1)
    dawn_img = (dawn_img * 255).astype(np.uint8)

    # Warm overlay for sunrise feel
    warm_overlay = np.full_like(dawn_img, (30, 60, 120))  # BGR warm tone
    dawn_img = cv2.addWeighted(dawn_img, 0.85, warm_overlay, 0.15, 0)

    cv2.imwrite(f"{output_folder}/{filename}_dawn.png", dawn_img)
    print(f"Dawn effect applied: {filename}_dawn.png")
    return True


def process_folder(input_folder):
    for file in os.listdir(input_folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(input_folder, file)
            apply_dawn_effect(path)


if __name__ == "__main__":
    folder = input("Enter folder path containing images: ")
    process_folder(folder)
