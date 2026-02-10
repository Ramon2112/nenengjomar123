import cv2
import numpy as np
import os

def apply_dawn_effect(image_path, output_folder="output", num_rays=30, intensity=0.3):
    """
    Applies a dawn light rays effect from the top-left corner to a single image.
    
    :param image_path: Path to input image
    :param output_folder: Folder to save processed images
    :param num_rays: Number of light rays
    :param intensity: Blending intensity of the rays (0 to 1)
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image not found: {image_path}")
        return False

    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.splitext(os.path.basename(image_path))[0]

    height, width = img.shape[:2]
    overlay = np.zeros_like(img, dtype=np.float32)

    # Draw multiple rays from top-left corner
    for i in range(num_rays):
        end_x = np.random.randint(int(width * 0.3), width)
        end_y = np.random.randint(int(height * 0.3), height)
        thickness = np.random.randint(1, 3)
        cv2.line(overlay, (0, 0), (end_x, end_y), (255, 255, 255), thickness)

    overlay = cv2.GaussianBlur(overlay, (25, 25), 0)
    img_float = img.astype(np.float32)
    dawn_img = cv2.addWeighted(img_float, 1.0, overlay, intensity, 0)
    dawn_img = np.clip(dawn_img, 0, 255).astype(np.uint8)

    cv2.imwrite(f"{output_folder}/{filename}_dawn.png", dawn_img)
    print(f"Dawn effect applied and saved to {output_folder}/{filename}_dawn.png")
    return True


def process_folder(input_folder, output_folder="output"):
    """
    Processes all images in a folder and applies the dawn effect.
    """
    os.makedirs(output_folder, exist_ok=True)
    for file in os.listdir(input_folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(input_folder, file)
            apply_dawn_effect(path, output_folder)


if __name__ == "__main__":
    folder = input("Enter folder path containing images: ")
    output = input("Enter output folder path (default 'output'): ") or "output"
    process_folder(folder, output)
