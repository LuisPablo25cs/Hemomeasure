import cv2
import numpy as np
import matplotlib.pyplot as plt

# Imagen original
#img = cv2.imread('C:\\Users\\Edgardo\\Pictures\\matpat.jpg', cv2.IMREAD_COLOR)
#print(img)


import cv2
import numpy as np

image = cv2.imread('C:\\Users\\Edgardo\\Pictures\\gasatest3.jpg')

def findredp(image):
    red_channel = image[:, :, 2]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 0]
    adjusted_red_channel = red_channel - green_channel - blue_channel
    adjusted_red_channel = np.clip(adjusted_red_channel, 0, 255)
    adjusted_red_value = np.sum(adjusted_red_channel)
    red_value = np.sum(red_channel)
    green_value = np.sum(green_channel)
    blue_value = np.sum(blue_channel)
    
    totalrgb = red_value + green_value + blue_value

    redpercent = adjusted_red_value / totalrgb
    print("Red percent:", redpercent)

    greenpercent = green_value / totalrgb
    print("Green percent:", greenpercent)
    
    bluepercent = blue_value / totalrgb
    print("Blue percent:", bluepercent)

    return redpercent

redpercentage = findredp(image)
print("SATURACION DE GASA: ", redpercentage*30, " ml")

cv2.imshow("Ejemplo", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
