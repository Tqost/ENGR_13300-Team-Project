"""
Course Number: ENGR 13300
Semester: Fall 2026

Description:
    Replace this line with a description of your program.

Assignment Information:
    Assignment:     7.2.1 TP1 Team 1
    Team ID:        LC5 - 25
    Author:         Baron Chiang, chian163@purdue.edu
    Date:           10/15/2026

Contributors:
    Tanmay Deshmukh, deshmu62@purdue.edu
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


from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import math


def rgb_to_hsv(r, g, b):
    # Equation 7.1
    r_p, g_p, b_p = r / 255.0, g / 255.0, b / 255.0

    # Equation 7.2
    c_max = max([r_p, g_p, b_p])
    c_min = min([r_p, g_p, b_p])
    delta = c_max - c_min

    # Equation 7.3
    if delta == 0:
        h = 0
    elif c_max == r_p:
        h = 60 * ((g_p - b_p) / delta + 360) % 360
    elif c_max == g_p:
        h = 60 * ((b_p - r_p) / delta + 120) % 360
    elif c_max == b_p:
        h = 60 * ((r_p - g_p) / delta + 240) % 360

    # Equation 7.4
    if c_max == 0:
        s = 0
    else:
        s = delta / c_max

    # Equation 7.5
    v = c_max

    # Equation 7.6
    r_h = float(h * (255.0 / 360.0))
        
    r_s = float(s * 255.0)
    r_v = float(v * 255.0)

    return r_h, r_s, r_v


def convert_to_hsv(rgb_image):
    # Create blank array for the hsv image
    hsv_image = np.zeros(rgb_image.shape, dtype=np.uint8)
    # Loop through each row through the size of rgb_image
    for row in range(len(rgb_image)):
        # Loop through each column through the size of rgb_image
        for col in range(len(rgb_image[row])):
            # Get rgb values from the image at specific row and col
            r, g, b = rgb_image[row][col]
            # Get hsv values
            h, s, v = rgb_to_hsv(r, g, b)
            # Insert to hsv_image at row and col
            hsv_image[row][col] = [int(h), int(s), int(v)]

    return hsv_image


# From Python 5 Team Task 3
def load_img(file_dir):
    image = Image.open(file_dir)
    if image.mode != "L":
        image = image.convert("RGB")
    matrix = np.array(image) / 255.0
    linear_array = np.where(matrix <= 0.04045, matrix / 12.92, ((matrix + 0.055) / 1.055) ** 2.4)
    return linear_array

# From Python 5 Team Task 3
def resize_image(original_image_array, new_size):
    img = Image.fromarray(original_image_array)
    resized_img = img.resize(new_size, resample=Image.Resampling.BILINEAR)
    return np.array(resized_img)

# From Python 5 Team Task 3
def pad_image(resized_img_array, target_size):
    img = Image.fromarray(resized_img_array)
    padded_img = ImageOps.pad(img, target_size)
    return np.array(padded_img)


def main():
    img_dir = input("Enter the path of the RGB image you want to convert to hsv: ")
    img = load_img(img_dir)
    img = (img * 255).astype(np.uint8)

    target_size = (100, 100)
    h, w = img.shape[:2]
    if (w / h) > (target_size[0] / target_size[1]):
        new_size = (target_size[0], math.ceil(target_size[0] * h / w))
    else:
        new_size = (math.ceil(target_size[1] * w / h), target_size[1])
    resized = resize_image(img, new_size)
    padded = pad_image(resized, target_size)

    coords = input("Enter the coordinates of the pixel you want inspect: ")
    coords = coords.split(",")
    coords[0], coords[1] = int(coords[0]), int(coords[1])
    r, g, b = padded[coords[0]][coords[1]]

    print(f"RGB values of the ({coords[0]}, {coords[1]}) pixel: R={r}, G={g}, B={b}")
    print(f"Converting {img_dir} to HSV...")
    hsv_image = convert_to_hsv(padded)
    h, s, v = hsv_image[coords[0], coords[1]]
    print(f"HSV values of the ({coords[0]}, {coords[1]}) pixel: H={int(h)}, S={int(s)}, V={int(v)}")
    
    plt.imshow(hsv_image, cmap="hsv")
    plt.show()

if __name__ == "__main__":
    main()
