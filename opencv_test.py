import numpy as np
import cv2
import matplotlib.pyplot as plt


def conv_helper_color(fragment, kernel):
    f_row, f_col, f_channels = fragment.shape
    k_row, k_col = kernel.shape
    result = np.zeros(f_channels)

    for channel in range(f_channels):
        for row in range(f_row):
            for col in range(f_col):
                result[channel] += fragment[row, col, channel] * kernel[row, col]
    
    return result

def convulution_color(image, kernel):
    image_row, image_col, image_channels = image.shape
    kernel_row, kernel_col = kernel.shape

    output_row = image_row - kernel_row + 1
    output_col = image_col - kernel_col + 1
    output = np.zeros((output_row, output_col, image_channels))

    #Aplica convolución
    for row in range(output_row):
        for col in range(output_col):
            output[row, col] = conv_helper_color(
                image[row:row + kernel_row, col:col + kernel_col], kernel)

    #Normalizar la salida al rango [0, 255]
    output = np.clip(output, 0, 255).astype(np.uint8)


    return output

def alt_colors_spectrum(image, red, green, blue):
    image_output = image.copy()
    image_output[:, :, 0] = np.clip(image_output[:, :, 0] * red, 0, 255)
    image_output[:, :, 1] = np.clip(image_output[:, :, 1] * green, 0, 255)
    image_output[:, :, 2] = np.clip(image_output[:, :, 2] * blue, 0 ,255)

    image_output =image_output.astype(np.uint8)
    return image_output

def padding(image, padding):
    high, witdh, channels = image.shape
    image_output = np.zeros((high+padding*2, witdh+padding*2, channels), dtype= image.dtype)
    image_output[padding: padding + high, padding:padding + witdh, :] = image
    return image_output

#Devuelve los bordes validos de la imagen

def painter(image):
    height, width, channels = image.shape
    output_image = image.copy()
    height, width, channels = output_image.shape
    gaze_px = 0
    px = 0
    #Verde y azul deben ser a lo mucho 160 en RGB
    for row in range(height):
        for col in range(width):
            r, g, b = output_image[row][col]
            if(g>9 and b>9 and r>230):
                output_image[row][col] = 0, 0, 0
                px += 1
            else: 
                gaze_px += 1
    
    print("pixeles de la gaza")
    print(gaze_px)
    return output_image, gaze_px


def depurer(image, gaze_px):
    height, width, channels = image.shape
    output_image = image.copy()
    height, width, channels = output_image.shape
    blood_px = 0
    #Verde y azul deben ser a lo mucho 160 en RGB
    for row in range(height):
        for col in range(width):
            r, g, b = output_image[row][col]
            if(r>170 and g>170 and b>170):
                output_image[row][col] = 0, 0, 0
            elif(r == 0 and g == 0 and b == 0):
                continue
            else: 
                blood_px += 1
    print("El porcentaje de sangre en la gaza es: ", blood_px/gaze_px)
    return output_image

kernelBorders = np.array([
     [1, 1, 1],
     [-1, -4, -1],
     [1, 1, 1]
])


ruta = r"C:\Users\luisp\OneDrive\Documentos\Carrera\Semestre 4\SemanaTec\Hemomeasure\gaza_ejemplo.jpg"
imagen_gaza = cv2.imread(ruta, cv2.IMREAD_COLOR_RGB)
imagen_gaza_grises = cv2.imread(ruta, cv2.IMREAD_REDUCED_GRAYSCALE_2)
#imagen_gaza = alt_colors_spectrum(imagen_gaza, 1, 0, 0)
#borders = convulution_color(imagen_gaza, kernel_sobel_x)
#borders = convulution_color(borders, kernel_sobel_y)

borders2 = convulution_color(imagen_gaza, kernelBorders)
depured, gaze_px = painter(imagen_gaza)
depured = depurer(depured, gaze_px)
#imagen_gaza = convulution_color(imagen_gaza, kernelDef)

cv2.namedWindow("Ejemplo2", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Ejemplo2", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow("Ejemplo2",depured)
cv2.waitKey(0)
cv2.imshow("Ejemplo2", borders2)
cv2.waitKey(0)
