"""
Course Number: ENGR 13300
Semester: Fall 2026

Description:
    Kernel Sliding for Team Project 1

Assignment Information:
    Assignment:     tp1 team 2
    Team ID:        LC5 - 25 (e.g. LC1 - 01; for section LC1, team 01)
    Author:         Tanmay Deshmukh, deshmu62@purdue.edu
    Date:           10/15/2026

Contributors:
    Baron Chiang, chian163@purdue [repeat for each]
    Leo Cohen, cohen180@purdue

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
from PIL import Image, ImageOps
import numpy as np

# Create mean kernel ensuring sum of all elements is 1 and size is kernel size * kernel size
def create_mean_kernel(kernel_size):
    data_value = 1 / (kernel_size * kernel_size)
    kernel_matrix = np.full((kernel_size, kernel_size), data_value)
    return kernel_matrix

# Pad a resized image array to target_size (width, height)
def pad_image(resized_img_array, target_size):
    img = Image.fromarray(resized_img_array.astype(np.uint8))
    padded_img = ImageOps.pad(img, target_size, method=Image.BILINEAR)
    return np.array(padded_img)

# Create a padded image and slide kernel over image
def apply_kernel(image, kernel):
    H, W = image.shape
    h, w = kernel.shape
    R = np.zeros((H,W))

    target_size = (W + w - 1, H + h - 1)
    padded = pad_image(image, target_size)


    for j in range(H):
        for i in range(W):
            region = padded[j:j+h, i : i+w]
            R[j, i] = np.sum(region * kernel)

    return R


# Make test image from ENGR website, input kernel size from user, call create_mean_kernel and apply_kernel functions, then display kernel and filtered images.
def main():
    np.set_printoptions(precision=6)
    test_image = np.zeros((5,5), dtype=np.uint8)
    test_image[2,2] = 255
    kernel_size = int(input("Enter kernel size: "))
    print(f"Kernel size: {kernel_size}x{kernel_size}")
    print("Kernel sum: 1.000000")
    kernel = create_mean_kernel(kernel_size)
    filtered = apply_kernel(test_image, kernel)
    print("Kernel:")
    print(kernel)
    print("Filtered image:")
    print(filtered)

if __name__ == "__main__":
    main()
