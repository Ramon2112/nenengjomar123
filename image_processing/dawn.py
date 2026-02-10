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

    # Step 1: Create radial light mask from top-left
    Y, X = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    center_x, center_y = 0, 0  # top-left corner
    distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_dist = np.sqrt(cols**2 + rows**2)
    mask = 1 - (distance / max_dist)  # intensity falls off with distance
    mask = np.clip(mask, 0, 1)

    # Step 2: Add streaks for God rays
    num_rays = 200  # number of streaks
    for _ in range(num_rays):
        angle = np.random.uniform(-0.2, 0.2)  # narrow angle spread
        length = np.random.randint(int(cols * 0.5), int(cols))
        x = np.arange(cols)
        y = (x * np.tan(angle)).astype(int)
        valid = (y >= 0) & (y < rows)
        mask[y[valid], x[valid]] += 0.05  # increase intensity along streaks

    mask = np.clip(mask, 0, 1)

    # Step 3: Blur the mask to make rays soft
    mask = cv2.GaussianBlur(mask, (101, 101), 0)
    mask_3ch = cv2.merge([mask, mask, mask])

    # Step 4: Apply mask to original image
    img_float = img.astype(np.float32) / 255.0
    dawn_img = img_float + mask_3ch * 0.6  # adjust intensity here
    dawn_img = np.clip(dawn_img, 0, 1)
    dawn_img = (dawn_img * 255).astype(np.uint8)

    # Step 5: Warm overlay for sunrise feel
    warm_overlay = np.full_like(dawn_img, (30, 60, 120))  # BGR warm tone
    dawn_img = cv2.addWeighted(dawn_img, 0.85, warm_overlay, 0.15, 0)

    # Save output
    cv2.imwrite(f"{output_folder}/{filename}_cinematic_dawn.png", dawn_img)
    print(f"Cinematic dawn effect applied: {filename}_cinematic_dawn.png")
    return True


def process_folder(input_folder):
    for file in os.listdir(input_folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(input_folder, file)
            apply_dawn_effect(path)


if __name__ == "__main__":
    folder = input("Enter folder path containing images: ")
    process_folder(folder)
