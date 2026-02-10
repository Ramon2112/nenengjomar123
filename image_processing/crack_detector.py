import cv2
import numpy as np
import os


def detect_cracks(image_path, output_folder="output"):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image not found: {image_path}")
        return

    os.makedirs(output_folder, exist_ok=True)

    filename = os.path.splitext(os.path.basename(image_path))[0]

    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    crack_mask = np.zeros_like(gray)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50 < area < 5000:
            cv2.drawContours(crack_mask, [cnt], -1, 255, thickness=1)

    result = original.copy()
    result[crack_mask == 255] = [0, 0, 255]

    cv2.imwrite(f"{output_folder}/{filename}_edges.png", edges)
    return True


def process_folder(input_folder):
    for file in os.listdir(input_folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(input_folder, file)
            detect_cracks(path)


if __name__ == "__main__":
    folder = input("Enter folder path containing wall/building images: ")
    process_folder(folder)
