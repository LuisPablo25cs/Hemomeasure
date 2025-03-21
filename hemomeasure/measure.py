import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('C:\\Users\\Edgardo\\Documents\\hemomeasure\\casosdeprueba\\casosdeprueba\\caso1ladoa.jpg')
image2 = cv2.imread('C:\\Users\\Edgardo\\Documents\\hemomeasure\\casosdeprueba\\casosdeprueba\\caso1ladob.jpg')


def adjust_brilliance(image, brilliance_factor=1.5):
    image = image.astype(np.float32)
    mean_brightness = np.mean(image)
    image = (image - mean_brightness) * brilliance_factor + mean_brightness
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)

    return image

def adjust_exposure(image, exposure_factor=1.5):
    image = image.astype(np.float32)

    image = image * exposure_factor

    # OVERFLOW HANDLING
    image = np.clip(image, 0, 255)

    image = image.astype(np.uint8)

    return image


def findredp(image):
    height, width, _ = image.shape
    
    accumulated_red = 0  
    totalred=0
    totalgreen=0
    totalblue=0
    
    for y in range(height):
        for x in range(width):
            red_value = image[y, x, 2]
            totalred = totalred + red_value
            
            green_value = image[y, x, 1]
            totalgreen = totalgreen + green_value
            
            blue_value = image[y, x, 0]
            totalblue = totalblue + blue_value
    
    print("DEBUG: ")
    print("red: ", totalred)
    print("green: ", totalgreen)
    print("blue: ", totalblue)
    
    print("OPERACION: ")
    accumulated_red = totalred - ((totalgreen + totalblue)/2)
    print("TOTAL DE ROJO: ", accumulated_red)
    
    redpercentage = accumulated_red/(255*image.shape[0]*image.shape[1])
    
    print("PORCENTAJE DE ROJO: ", redpercentage)
    
    return redpercentage

def hemomeasure(image, brilliance_factor=5, exposure_factor=3, blur_ksize=(123, 123)):
    image = adjust_brilliance(image, brilliance_factor)
    image = adjust_exposure(image, exposure_factor)
    image = cv2.GaussianBlur(image, blur_ksize, 0)
    redpercentage = findredp(image)
    return redpercentage

#image processing for optimization
if (image.shape[0]*image.shape[1])>100:
    image = cv2.resize(image, (int(image.shape[1] * 0.25), int(image.shape[0] * 0.25)))

if (image2.shape[0]*image2.shape[1])>100:
    image2 = cv2.resize(image2, (int(image2.shape[1] * 0.25), int(image2.shape[0] * 0.25)))


print("LADO A")
print("////////////")
redpercentage = hemomeasure(image, brilliance_factor=5, exposure_factor=3, blur_ksize=(123, 123))

print("LADO B")
print("////////////")
redpercentage2 = hemomeasure(image2, brilliance_factor=5, exposure_factor=3, blur_ksize=(123, 123))

print("")
print("SATURACION ESTIMADA EN GAZA: ", (redpercentage*6)+(redpercentage2*6),"ml")


cv2.imshow("Ejemplo", image)
cv2.imshow("Ejemplo", image2)
cv2.waitKey(0)
cv2.destroyAllWindows()

