"""
Course Number: ENGR 13300
Semester: Fall 2026

Description:
    Write code in python to create two functions, one to build a Gaussian kernel and one to apply it, then use them to blur a user-selected grayscale image
Assignment Information:
    Assignment:     tp1 team 3
    Team ID:        LC5 - 25 
    Author:         Leo Cohen, cohen180@purdue.edu
    Date:           10/06/2026

Contributors:
    Baron Chiang, chian163@purdue.edu
    Tanmay Deshmukh, deshmu62@purdue.edu

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
import os

def load_img(file_path):
    #Loads an image from the specified file path.
    return plt.imread(file_path)

def rgb_to_grayscale(img):
    #Converts an RGB image to grayscale.
    if len(img.shape) == 3 and img.shape[2] >= 3:
        return 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
    return img

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

def main():
    # Prompt the user to enter the image file path
    filename = input("Enter the path to the image file: ")
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the full image path by joining the script directory, "ref_images" folder, and the filename
    image_path = os.path.join(script_dir, "ref_images", filename)

    # Load the image from the constructed path
    img = load_img(image_path)

    # Convert the loaded image to grayscale
    gray_img = rgb_to_grayscale(img)

    # Ensure the grayscale image is in the range [0, 255] as an 8-bit unsigned integer.
    if gray_img.max() > 1.0:
        gray_img = (gray_img * 255).round().astype(np.uint8)    
    
    # Set the standard deviation for the Gaussian filter
    sigma = 1.0
    # Apply the Gaussian filter to the grayscale image
    blurred_img = gaussian_filter(gray_img, sigma)

    # Display the blurred image using matplotlib
    plt.figure()
    plt.imshow(blurred_img, cmap='gray', vmin=0, vmax=255)    
    plt.axis('off')

    # Save and display the blurred image
    output_filename = f"smooth_{filename}"
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == "__main__":
    main()
