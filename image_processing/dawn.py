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

    # Step 2: Define multiple target points
    endpoints = [
        (0, rows // 2),        # left-center
        (cols // 2, rows // 2),# center
        (cols // 2, 0)         # top-center
    ]

    # Step 3: Draw triangle-like rays
    rays_per_point = 8  # more rays for realistic sunbeams
    for end_x, end_y in endpoints:
        for i in range(rays_per_point):
            # Random offset for each ray
            offset_x = np.random.randint(-40, 40)
            offset_y = np.random.randint(-40, 40)

            # Define triangle width at the source
            source_width = np.random.randint(30, 60)

            # Define the triangle as a polygon: top-left source + offset for width
            pts = np.array([
                [0 - source_width//2, 0],       # left edge at source
                [0 + source_width//2, 0],       # right edge at source
                [end_x + offset_x, end_y + offset_y]  # tip of the ray
            ], np.int32)

            cv2.fillConvexPoly(mask, pts, 1.0)

    # Step 4: Fade rays: strong at top-left, weaker at end
    Y, X = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    distance = np.sqrt((X - 0)**2 + (Y - 0)**2)
    max_dist = np.sqrt(cols**2 + rows**2)
    fade = np.clip(1 - distance / max_dist, 0, 1)
    mask *= fade

    # Step 5: Blur for smooth cinematic rays
    mask = cv2.GaussianBlur(mask, (101, 101), 0)
    mask = np.clip(mask, 0, 1)

    # Step 6: Convert mask to 3 channels
    mask_3ch = cv2.merge([mask, mask, mask])

    # Step 7: Apply mask to image
    img_float = img.astype(np.float32) / 255.0
    result = img_float + mask_3ch * 0.8  # adjust intensity
    result = np.clip(result, 0, 1)
    result = (result * 255).astype(np.uint8)

    # Step 8: Warm overlay for dawn/sunrise feel
    warm_overlay = np.full_like(result, (30, 60, 120))  # BGR warm tone
    result = cv2.addWeighted(result, 0.85, warm_overlay, 0.15, 0)

    # Save output
    cv2.imwrite(f"{output_folder}/{filename}_sunbeams.png", result)
    print(f"Triangle sunbeams applied: {filename}_sunbeams.png")
    return True


def process_folder(input_folder):
    for file in os.listdir(input_folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(input_folder, file)
            apply_dawn_effect(path)


if __name__ == "__main__":
    folder = input("Enter folder path containing images: ")
    process_folder(folder)
