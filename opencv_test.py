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

kernelBorders = np.array([
     [1, 1, 1],
     [-1, -4, -1],
     [1, 1, 1]
])

kernelSat = np.array([
    [0, .25, 0],
    [.25, 0, .25],
    [0, .25, 0]

])

kernelDef = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])



ruta = r"C:\Users\luisp\OneDrive\Documentos\Carrera\Semestre 4\SemanaTec\ProyectoVisionComputacional\semena-tec-tools-vision\Images\tata.jpg"
rubenColor = cv2.imread(ruta, cv2.IMREAD_COLOR_RGB)
rubenColor = alt_colors_spectrum(rubenColor, 0, .5, 1)
#rubenColor = convulution_color(rubenColor, kernelBorders)
rubenColor = convulution_color(rubenColor, kernelDef)
alto, ancho, canales = rubenColor.shape
print(alto)
print(ancho)
rubenColor = padding(rubenColor, 2)
alto2, ancho2, canales2 = rubenColor.shape
print(alto2)
print(ancho2)
plt.imshow(rubenColor)
plt.axis('off')
plt.show()