import tkinter as tk
from tkinter.filedialog import askopenfilename
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Select image using file dialog
tk.Tk().withdraw()
fn = askopenfilename()

if not fn:
    print("No file selected. Exiting.")
    exit()

# Read image using OpenCV
image = cv2.imread(fn)
if image is None:
    print("Failed to load image. Please check the file path.")
    exit()

def convolution(image, kernel, average=False, verbose=False):
    if len(image.shape) == 3:
        print("Found 3 Channels : {}".format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Converted to Gray Channel. Size : {}".format(image.shape))
    else:
        print("Image Shape : {}".format(image.shape))

    print("Kernel Shape : {}".format(kernel.shape))

    if verbose:
        plt.imshow(image, cmap='gray')
        plt.title("Input Image")
        plt.show()

    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = kernel_row // 2
    pad_width = kernel_col // 2

    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    if verbose:
        plt.imshow(padded_image, cmap='gray')
        plt.title("Padded Image")
        plt.show()

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]

    print("Output Image size : {}".format(output.shape))

    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("Output Image using {}x{} Kernel".format(kernel_row, kernel_col))
        plt.show()

    return output

# Edge detection kernel
k = np.array([[-1, -1, -1],
              [-1, 8, -1],
              [-1, -1, -1]])

output_image = convolution(image, k, verbose=True)

# Optional Gaussian blur example
"""
k_blur = np.array([[1, 2, 1],
                   [2, 4, 2],
                   [1, 2, 1]]) / 16

output_image_blur = convolution(image, k_blur, average=True, verbose=True)
"""