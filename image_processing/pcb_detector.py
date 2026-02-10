import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

# ----------------------------
# PCB FAULT DETECTION FUNCTION (WITH ALIGNMENT)
# ----------------------------
def detect_faults(ref_path, test_path, output_folder="output"):
    """Detect PCB faults by aligning test image to reference and comparing."""
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.splitext(os.path.basename(test_path))[0]

    # Load images
    ref_img = cv2.imread(ref_path)
    test_img = cv2.imread(test_path)
    if ref_img is None or test_img is None:
        print(f"Error loading images: {ref_path}, {test_path}")
        return

    # Resize test to match reference
    test_img = cv2.resize(test_img, (ref_img.shape[1], ref_img.shape[0]))

    # ----------------------------
    # IMAGE ALIGNMENT INSIDE THIS FUNCTION
    # ----------------------------
    gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray_test, None)
    kp2, des2 = orb.detectAndCompute(gray_ref, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:50]

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    aligned_test = cv2.warpPerspective(test_img, matrix, (ref_img.shape[1], ref_img.shape[0]))

    # ----------------------------
    # FAULT DETECTION
    # ----------------------------
    # Convert to grayscale and blur
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(aligned_test, cv2.COLOR_BGR2GRAY)
    ref_blur = cv2.GaussianBlur(ref_gray, (5, 5), 0)
    test_blur = cv2.GaussianBlur(test_gray, (5, 5), 0)

    # SSIM comparison
    score, diff = ssim(ref_blur, test_blur, full=True)
    print(f"{filename} - SSIM Similarity Score: {score:.4f}")

    # Highlight differences
    diff = ((1 - diff) * 255).astype("uint8")

    # Threshold differences
    thresh = cv2.adaptiveThreshold(
        diff, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Contour detection
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = aligned_test.copy()
    fault_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 300 < area < 5000:  # Filter noise
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(output, "PCB Fault", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            fault_count += 1

    print(f"{filename} - Detected Fault Regions: {fault_count}")

    # Save results
    cv2.imwrite(os.path.join(output_folder, f"{filename}_aligned.png"), aligned_test)
    cv2.imwrite(os.path.join(output_folder, f"{filename}_diff.png"), diff)
    cv2.imwrite(os.path.join(output_folder, f"{filename}_thresh.png"), thresh)
    cv2.imwrite(os.path.join(output_folder, f"{filename}_faults.png"), output)

    return True

# ----------------------------
# MAIN EXECUTION
# ----------------------------
if __name__ == "__main__":
    reference_path = input("Enter reference PCB image path: ")
    test_image_path = input("Enter test PCB image path: ")
    detect_pcb_faults(reference_path, test_image_path)
