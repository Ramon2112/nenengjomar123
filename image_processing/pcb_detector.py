import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

def align_images(image, template):
    """Align the test image to the reference template using ORB feature matching."""
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_temp = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray_img, None)
    kp2, des2 = orb.detectAndCompute(gray_temp, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    good_matches = matches[:50]  # Take top 50 matches

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    aligned = cv2.warpPerspective(image, matrix, (template.shape[1], template.shape[0]))
    return aligned

def detect_faults(ref_path, test_path, output_folder="output"):
    """Detect PCB faults by comparing a test image to a reference PCB image."""
    ref_img = cv2.imread(ref_path)
    test_img = cv2.imread(test_path)
    if ref_img is None or test_img is None:
        print(f"Error loading images: {ref_path}, {test_path}")
        return

    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.splitext(os.path.basename(test_path))[0]

    # Resize test image to match reference
    test_img = cv2.resize(test_img, (ref_img.shape[1], ref_img.shape[0]))

    # Align test image
    aligned_test = align_images(test_img, ref_img)

    # Preprocess
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(aligned_test, cv2.COLOR_BGR2GRAY)
    ref_blur = cv2.GaussianBlur(ref_gray, (5,5), 0)
    test_blur = cv2.GaussianBlur(test_gray, (5,5), 0)

    # SSIM difference
    score, diff = ssim(ref_blur, test_blur, full=True)
    print(f"{filename} - SSIM Similarity Score: {score:.4f}")
    diff = (diff * 255).astype("uint8")

    # Thresholding
    thresh = cv2.adaptiveThreshold(diff, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological cleanup
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Contour detection
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = aligned_test.copy()
    fault_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 300 < area < 5000:   # Filter noise
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x,y), (x+w,y+h), (0,0,255), 2)
            cv2.putText(output, "PCB Fault", (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
            fault_count += 1

    print(f"{filename} - Detected Fault Regions: {fault_count}")

    cv2.imwrite(f"{output_folder}/{filename}_aligned_test.png", aligned_test)
    cv2.imwrite(f"{output_folder}/{filename}_difference_map.png", diff)
    cv2.imwrite(f"{output_folder}/{filename}_faults.png", thresh)
    cv2.imwrite(f"{output_folder}/{filename}_pcb.png", output)
    return True

def process_folder(ref_path, test_folder, output_folder="output"):
    """Process all test images in a folder against the reference PCB."""
    for file in os.listdir(test_folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            test_path = os.path.join(test_folder, file)
            detect_faults(ref_path, test_path, output_folder)

if __name__ == "__main__":
    reference = input("Enter reference PCB image path: ")
    test_folder = input("Enter folder path containing test PCB images: ")
    process_folder(reference, test_folder)
