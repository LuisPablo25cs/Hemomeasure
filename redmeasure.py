import cv2
import numpy as np
import matplotlib.pyplot as plt

# Imagen original
#img = cv2.imread('C:\\Users\\Edgardo\\Pictures\\matpat.jpg', cv2.IMREAD_COLOR)
#print(img)


import cv2
import numpy as np


image = cv2.imread('C:\\Users\\Edgardo\\Pictures\\gasatest2.jpg')

def findredp(image):
    red_channel = image[:, :, 2]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 0]

    red_value = np.sum(red_channel)
    green_value = np.sum(green_channel)
    blue_value = np.sum(blue_channel)

    totalrgb = red_value + green_value + blue_value
    redpercent = red_value / totalrgb
    print(redpercent)
    greenpercent = green_value / totalrgb
    print(greenpercent)
    bluepercent = blue_value / totalrgb
    print(bluepercent)

    return redpercent

redpercentage = findredp(image)
print(redpercentage*30)
