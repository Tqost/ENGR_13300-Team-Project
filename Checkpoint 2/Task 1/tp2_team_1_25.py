"""
Course Number: ENGR 13300
Semester: Fall 2026

Description:
    Detects edges in a grayscale image using a sobel filter, then saves the sobel filter image.

Assignment Information:
    Assignment:     8.2.1 TP2 Team 1
    Team ID:        LC5 - 25
    Author:         Baron Chiang, chian163@purdue.edu
    Date:           10/22/2026

Contributors:
    Name, login@purdue [repeat for each]

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

import os

import numpy as np
import matplotlib.pyplot as plt


def load_img(file_path):
    # Loads an image from the specified file path.
    return plt.imread(file_path)


def rgb_to_grayscale(img):
    # Converts an RGB image to grayscale.
    if np.issubdtype(img.dtype, np.floating):
        img = img * 255
    if len(img.shape) == 3 and img.shape[2] >= 3:
        return (0.2989 * img[:, :, 0] 
                + 0.5870 * img[:, :, 1] 
                + 0.1140 * img[:, :, 2])
    return img


def create_gaussian_kernel(sigma):
    # Creates a Gaussian kernel with the specified sigma.
    k_size = 2 * int(np.ceil(3 * sigma)) + 1
    pad = k_size // 2

    coefficient = (1 / (2 * np.pi * (sigma**2)))
    
    # Initialize the Gaussian kernel with zeros
    kernel = np.zeros((k_size, k_size), dtype=float)
    for y in range(-pad, pad + 1):
        for x in range(-pad, pad + 1):
            exponent = -(x**2 + y**2) / (2 * (sigma**2))
            kernel[y + pad, x + pad] = coefficient * np.exp(exponent)
            
    # Normalize so sum equals 1.0
    kernel /= np.sum(kernel)
    return kernel


def gaussian_filter(image, sigma):
    # Applies a Gaussian filter to the given image using the specified sigma.
    kernel = create_gaussian_kernel(sigma)
    k_size = kernel.shape[0]
    pad_size = k_size // 2
    
    # Zero-pad the image
    padded_image = np.pad(image, pad_size, mode="constant", constant_values=0)
    output_image = np.zeros_like(image, dtype=float)
    
    img_h, img_w = image.shape
    # Slide kernel across the image
    for i in range(img_h):
        for j in range(img_w):
            window = padded_image[i:i + k_size, j:j + k_size]
            output_image[i, j] = np.sum(window * kernel)
            
    # Clip to [0, 255] and convert to 8-bit unsigned integer.
    return np.clip(output_image, 0, 255).astype(np.uint8)


def sobel_filter(image):
    sobel_x = np.array([[-1, 0, 1], 
                        [-2, 0, 2], 
                        [-1, 0, 1]], dtype=float)
    sobel_y = np.array([[-1, -2, -1], 
                        [0, 0, 0], 
                        [1, 2, 1]], dtype=float)
    threshold = 50
    image = np.array(image, dtype=float)
    padded_image = np.pad(image, 1, mode="constant", constant_values=0)
    sobel_image = np.zeros_like(image, dtype=float)
    img_h, img_w = image.shape

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


def main():
    file_dir = input("Enter the path to the image file: ")
    image = load_img(file_dir)
    grayscale_image = rgb_to_grayscale(image)
    blur_image = gaussian_filter(grayscale_image, 1)
    sobel_image = sobel_filter(blur_image)

    plt.figure()
    plt.imshow(sobel_image, cmap="gray", vmin=0, vmax=255)    
    plt.axis("off")
    
    output_filename = "sobel_" + os.path.basename(file_dir)
    plt.imsave(output_filename, sobel_image, cmap="gray", vmin=0, vmax=255)
    plt.show()


if __name__ == "__main__":
    main()
