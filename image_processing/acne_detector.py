import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def detect_acne(image_path, output_folder="output"):
    """
    Detects acne spots in the given image and saves the processed image
    with bounding boxes around detected acne.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image not found: {image_path}")
        return

    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.splitext(os.path.basename(image_path))[0]

    image = cv2.resize(img, (500, 500))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red color range in HSV
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    acne_count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            acne_count += 1
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Determine severity
    if acne_count < 5:
        severity = "Mild Acne"
    elif acne_count < 17:
        severity = "Moderate Acne"
    else:
        severity = "Severe Acne"

    cv2.putText(image, f"Acne Count: {acne_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(image, severity, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Save the processed image
    cv2.imwrite(f"{output_folder}/{filename}_acne.png", image)
    return True

def process_folder(input_folder):
    """
    Processes all images in a folder for acne detection.
    """
    for file in os.listdir(input_folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(input_folder, file)
            detect_acne(path)

if __name__ == "__main__":
    folder = input("Enter folder path containing face images: ")
    process_folder(folder)
