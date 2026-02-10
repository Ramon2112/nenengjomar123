import cv2
import numpy as np
import os
import random

def create_jigsaw(image_path, grid_size=3, output_folder="output"):
    """
    Creates a shuffled jigsaw puzzle from a given image and saves the result.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image not found: {image_path}")
        return

    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_folder, f"{filename}_jigsaw.png")

    height, width, _ = img.shape

    # Resize image so it's divisible by grid size
    new_height = height - (height % grid_size)
    new_width = width - (width % grid_size)
    img = cv2.resize(img, (new_width, new_height))

    piece_h = new_height // grid_size
    piece_w = new_width // grid_size

    pieces = []

    # Split image into pieces
    for row in range(grid_size):
        for col in range(grid_size):
            y1 = row * piece_h
            y2 = y1 + piece_h
            x1 = col * piece_w
            x2 = x1 + piece_w
            piece = img[y1:y2, x1:x2]
            pieces.append(piece)

    # Shuffle pieces
    random.shuffle(pieces)

    # Create puzzle image
    puzzle_img = np.zeros_like(img)
    index = 0
    for row in range(grid_size):
        for col in range(grid_size):
            y1 = row * piece_h
            y2 = y1 + piece_h
            x1 = col * piece_w
            x2 = x1 + piece_w
            puzzle_img[y1:y2, x1:x2] = pieces[index]
            index += 1

    # Save result
    cv2.imwrite(output_path, puzzle_img)
    print(f"Jigsaw puzzle saved: {output_path}")
    return True

def process_folder(input_folder, grid_size=3, output_folder="output"):
    """
    Processes all images in a folder to create jigsaw puzzles.
    """
    for file in os.listdir(input_folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(input_folder, file)
            create_jigsaw(path, grid_size, output_folder)

if __name__ == "__main__":
    folder = input("Enter folder path containing images: ")
    grid_size = int(input("Enter grid size (e.g., 3 for 3x3): "))
    process_folder(folder, grid_size)
