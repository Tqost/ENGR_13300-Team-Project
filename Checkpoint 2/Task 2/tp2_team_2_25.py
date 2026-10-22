"""
Course Number: ENGR 13300
Semester: Fall 2026

Description:
    Write code in python to create two functions, one to build a Gaussian kernel and one to apply it, then use them to blur a user-selected grayscale image
Assignment Information:
    Assignment:     tp2 team 2
    Team ID:        LC5 - 25 
    Author:         Tanmay Deshmukh, deshmu62@purdue.edu
    Date:           10/22/2026

Contributors:
    Baron Chiang, chian163@purdue.edu
    Leo Cohen, cohen180@purdue.edu

    My contributor(s) helped me:
    [ ] understand the assignment expectations without
        telling me how they will approach it.
    [ ] understand different ways to think about a solution
        without helping me plan my solution.
    [ ] think through the meaning of a specific error or
        bug present in my code without looking at my code.
    Note that if you helped somebody else with their code, you
    have to list that person as a contributor here as well.

Academic Integrity Statement:
    I have not used source code obtained from any unauthorized
    source, either modified or unmodified; nor have I provided
    another student access to my code.  The project I am
    submitting is my own original work.
"""

# Import necessary libraries for image processing and visualization
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from PIL import Image, ImageOps
import os

def load_img(file_path):
    # Load with PIL and convert to a float array
    img = np.array(Image.open(file_path)).astype(float)
    # Drop the alpha channel if the image is RGBA
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    # Normalize to [0.0, 1.0] if not already in that range
    if img.max() > 1.0:
        img = img / 255.0
    # Linearize (remove sRGB gamma encoding), Eq. 17.2
    return np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)


def rgb_to_hsv(r, g, b):
    # Equation 7.1
    r_p, g_p, b_p = r / 255.0, g / 255.0, b / 255.0

    # Equation 7.2
    c_max = max(r_p, g_p, b_p)
    c_min = min(r_p, g_p, b_p)
    delta = c_max - c_min

    # Equation 7.3
    if delta == 0:
        h = 0
    elif c_max == r_p:
        h = (60 * ((g_p - b_p) / delta) + 360) % 360
    elif c_max == g_p:
        h = (60 * ((b_p - r_p) / delta) + 120) % 360
    else:
        h = (60 * ((r_p - g_p) / delta) + 240) % 360

    # Equation 7.4
    if c_max == 0:
        s = 0
    else:
        s = delta / c_max

    # Equation 7.5
    v = c_max

    # Merge 7.6 and Return statement into nne line
    return h * 255.0 / 360.0, s * 255.0, v * 255.0


def convert_to_hsv(rgb_image):
    # Create blank array for the hsv image
    hsv_image = np.zeros(rgb_image.shape, dtype=np.uint8)
    # Loop through each row through the size of rgb_image
    for row in range(len(rgb_image)):
        # Loop through each column through the size of rgb_image
        for col in range(len(rgb_image[row])):
            # Get rgb values from the image at specific row and col
            r, g, b = rgb_image[row][col]
            # Insert to hsv_image at row and col
            h, s, v = rgb_to_hsv(r, g, b)
            hsv_image[row][col] = [int(h), int(s), int(v)]
    return hsv_image

def rgb_to_grayscale(img):
    #Converts an RGB image to grayscale.
    if np.issubdtype(img.dtype, np.floating):
        img = img * 255
    if img.ndim == 3 and img.shape[2] >= 3:
        return 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    return img


# Resize a uint8 image array to new_size (width, height)
def resize_image(original_image_array, new_size):
    img = Image.fromarray(original_image_array)
    resized_img = img.resize(new_size, resample=Image.Resampling.BILINEAR)
    return np.array(resized_img)


# Pad a resized image array to target_size (width, height)
def pad_image(resized_img_array, target_size):
    img = Image.fromarray(resized_img_array)
    padded_img = ImageOps.pad(img, target_size)
    return np.array(padded_img)

def create_gaussian_kernel(sigma):
    #Creates a Gaussian kernel with the specified sigma.
    k_size = 2 * int(np.ceil(3 * sigma)) + 1
    pad = k_size // 2

    # Initialize the Gaussian kernel with zeros
    kernel = np.zeros((k_size, k_size), dtype=float)
    for y in range(-pad, pad + 1):
        for x in range(-pad, pad + 1):
            exponent = -(x**2 + y**2) / (2 * (sigma**2))
            kernel[y + pad, x + pad] = (1 / (2 * np.pi * (sigma**2))) * np.exp(exponent)
            
    # Normalize so sum equals 1.0
    kernel /= np.sum(kernel)
    return kernel

def gaussian_filter(image, sigma):
    #Applies a Gaussian filter to the given image using the specified sigma.
    kernel = create_gaussian_kernel(sigma)
    k_size = kernel.shape[0]
    pad_size = k_size // 2
    
    # Zero-pad the image
    padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
    output_image = np.zeros_like(image, dtype=float)
    
    img_h, img_w = image.shape
    # Slide kernel across the image
    for i in range(img_h):
        for j in range(img_w):
            window = padded_image[i:i + k_size, j:j + k_size]
            output_image[i, j] = np.sum(window * kernel)
            
    # Clip the output image to the range [0, 255] and convert to 8-bit unsigned integer before returning.
    return np.clip(output_image, 0, 255).astype(np.uint8)

def sobel_filter(image):
    # Initializes sobel kernels and threshold value.
    sobel_x = np.array([[-1, 0, 1], 
                        [-2, 0, 2], 
                        [-1, 0, 1]], dtype=float)
    sobel_y = np.array([[-1, -2, -1], 
                        [0, 0, 0], 
                        [1, 2, 1]], dtype=float)
    threshold = 50

    # Pads image for sobel filtering and creates blank array.
    image = np.array(image, dtype=float)
    padded_image = np.pad(image, 1, mode="constant", constant_values=0)
    sobel_image = np.zeros_like(image, dtype=float)
    img_h, img_w = image.shape

    # Loops through each pixel and does sobel filtering.
    for i in range(img_h):
        for j in range(img_w):
            window = padded_image[i:i + 3, j:j + 3]
            gradient = np.sqrt(np.sum(window * sobel_x) ** 2 
                               + np.sum(window * sobel_y) ** 2)
            if gradient > 255:
                gradient = 255
            if gradient >= threshold:
                sobel_image[i, j] = 255

    return sobel_image.astype(np.uint8)


def detect_circle(gray_img):
    """Detects if a large circle is present. Returns 1 if found, 0 otherwise."""
    # Hough Circles works best on a grayscale, slightly blurred image

    blurred_img = gaussian_filter(gray_img, 1.5) #Use gaussian_filter (from the previous checkpoint) with sigma=1.5 to blur the input grayscale image
    blurred_img_array = np.array(blurred_img)  # Convert to NumPy array for processing with cv2

    # Detect circles
    circles = cv2.HoughCircles(
        blurred_img,
        cv2.HOUGH_GRADIENT,
        dp=1.2,  # Inverse ratio of accumulator resolution
        minDist=50,  # Minimum distance between centers of detected circles
        param1=100,  # Upper threshold for the internal Canny edge detector
        param2=30,  # Threshold for center detection
        minRadius=20,  # Minimum circle radius to detect
        maxRadius=50  # Maximum circle radius to detect
    )

    return 1 if circles is not None else 0 # return 1 if any circles are detected, otherwise return 0

def detect_triangle(gray_img):
    """Detects if a triangle is present. Returns 1 if found, 0 otherwise."""
    blurred = gaussian_filter(gray_img, sigma=1.0)
    edges = sobel_filter(blurred)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 3 and cv2.contourArea(approx) > 625:
            return 1
    return 0

def count_lines(gray_img):
    """Counts the number of lines in an image using Hough Line Transform."""
    # Edge detection via sobel filtering is a prerequisite for Hough Lines
    edges = sobel_filter(gray_img) #Use sobel_filter (from Task 1) on the input grayscale image

    # Detect lines using the edge map
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)

    return len(lines) if lines is not None else 0 #return the number of lines detected, if no lines are detected return 0

def extract_features(img_array):
    hsv_img = convert_to_hsv(img_array)
    hue_mean = np.mean(hsv_img[:, :, 0])
    hue_std = np.std(hsv_img[:, :, 0])
    saturation_mean = np.mean(hsv_img[:, :, 1])
    saturation_std = np.std(hsv_img[:, :, 1])
    value_mean = np.mean(hsv_img[:, :, 2])
    value_std = np.std(hsv_img[:, :, 2])

    grayscale_img = rgb_to_grayscale(img_array)
    number_lines = count_lines(grayscale_img)
    has_circle = detect_circle(grayscale_img)
    has_triangle = detect_triangle(grayscale_img)

    return [hue_mean, hue_std, saturation_mean, saturation_std, value_mean, value_std, number_lines, has_circle, has_triangle]

def main():
    path = input("Enter the name of the dataset folder:\n")
    met_file = input("Enter the name of the metadata file:\n")
    features_file = input("Enter the name of output csv features file:\n")

    met_df = pd.read_csv(met_file)
    print("\nTraining Metadata DataFrame:")
    print(met_df.head())

    feature_cols = ["hue_mean", "hue_std", "saturation_mean", "saturation_std",
                    "value_mean", "value_std", "number_lines", "has_circle",
                    "has_triangle"]

    rows = []
    for _, row in met_df.iterrows():
        img_path = row["Path"]
        img_array = load_img(os.path.join(path, img_path))
        img_array = (img_array * 255).astype(np.uint8)
        resized_image = resize_image(img_array, (100, 100))
        padded_image = pad_image(resized_image, (100, 100))
        features = extract_features(padded_image)
        rows.append(features + [img_path, row["ClassId"]])

    df = pd.DataFrame(rows, columns=feature_cols + ["Path", "ClassId"])
    # Sample output shows all feature columns as floats (e.g. has_triangle = 0.0)
    df[feature_cols] = df[feature_cols].astype(float)

    df.to_csv(features_file, index=False)
    print(f"\nFeatures have been extracted and saved to {features_file}")

    print("\nFeature Dataset Shape:")
    print(df.shape)
    print("\nFeature Dataset Head:")
    print(df.head())

    img_name = input("\nEnter the Path of the image you want to display the features for:\n")
    print(f"\nFeatures for {img_name}:")
    print(df[df["Path"] == img_name])

if __name__ == "__main__":
    main()

