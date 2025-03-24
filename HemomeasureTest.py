
import tkinter as tk
from tkinter.filedialog import askopenfilename
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
            if(g>9 and b>9 and r>240):
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
            if(r>150 and g>170 and b>170):
                output_image[row][col] = 0, 0, 0
            elif(r == 0 and g == 0 and b == 0):
                continue
            else: 
                blood_px += 1 
    print("El porcentaje de sangre en la gaza es: ", blood_px/gaze_px)
    print("La gaza tiene: ", (blood_px/gaze_px)*12, "ml")
    return output_image

def reader(ruta):
    #Se lee la imagen
    imagen_gaza = cv2.imread(ruta, cv2.IMREAD_COLOR_RGB)
    #Le da color rojo a la sangre
    imagen_gaza = cv2.cvtColor(imagen_gaza, cv2.COLOR_BGR2RGB)

    #Se pinta lo que no es la gaza de negro
    depured, gaze_px = painter(imagen_gaza)

    #Se pinta lo que no es la sangre para establecer una relación
    depured = depurer(depured, gaze_px)
    #Se muestra la sangre detectada en la imagen como una ventana
    cv2.namedWindow("Ejemplo2", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Ejemplo2", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Ejemplo2",depured)
    cv2.waitKey(0)
    return 0

def map_red_value(px):
    v_min = 100
    v_max = 170
    return 100-px / 170-px

#In a 10x10 cm gaze, the maximum amount of blood acumulated is 30ml. 
kernelBorders = np.array([
     [1, 1, 1],
     [-1, -4, -1],
     [1, 1, 1]
])

def llamar():
    # Ruta inicial del directorio, misma donde se encuentra el .py
    ruta = r"C:\Users\arace\Desktop\Araceli Escuela\TEC\SEMESTRE 4\SemanaTEC\Laboratorio\Hemomeasure\hemomeasure"
    # Permite abrir archivos .png o .jpg
    fn = askopenfilename(initialdir=ruta,filetypes =[("Archivo tipo Imagen", "*.png;*.jpg")])
    
    # En caso de no ingresar una imagen
    if not (fn):
        print("No se ha seleccionado una imagen.")
        return

    # OpenCV "lee" las imagenes en las rutas seleccionadas
    reader(fn)

    
print("¡Bienvenid@ a Hemomeasure!")
print("¿Deseas medir el procentaje de sangre?")
try:
    x = int(input("""Teclea el número de la opción que eligas:
    1.Si   2.No \n"""))

    if x==1:
        llamar()
    elif x==2:
        print("Nos vemos luego")
    else:
        print("Por favor seleccione 1 o 2.")
except ValueError:
    print("Por favor ingrese un número")
