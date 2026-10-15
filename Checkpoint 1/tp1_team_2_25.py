"""
Course Number: ENGR 13300
Semester: Fall 2026

Description:
    Resize images using the Pillow library while maintaining their aspect ratio to prevent distortion.

Assignment Information:
    Assignment:     tp1 team 2
    Team ID:        LC5 - 25 (e.g. LC1 - 01; for section LC1, team 01)
    Author:         Tanmay Deshmukh, deshmu62@purdue.edu
    Date:           10/06/2026

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

def create_mean_kernel(kernel_size):
    data_value = 1 / (kernel_size * kernel_size)
    kernel_matrix = np.full((kernel_size, kernel_size), data_value)
    return kernel_matrix

def apply_kernel(image, kernel):
    pass


# Get user input, resize while keeping aspect ratio, pad, print shapes, and display
def main():
    pass

if __name__ == "__main__":
    main()
