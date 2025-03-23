"""Propuesta por
Luis Pablo Cárdenas (A01254955)
Eduardo Cárdenas (A00232432)
Sebastián Blanchet (A00227588)
Araceli Ruiz (A01255302)"""

import tkinter as tk
from tkinter.filedialog import askopenfilename
import numpy as np
import cv2
import matplotlib.pyplot as plt


#Funcion para aumentar brillantez de imagen dada una imagen y el factor de brillantez por el cual se multiplicara.
#Calcula la intensidad promedio de cada pixel, multiplica la imagen por el factor de brillantez ingresado por el usuario y luego le suma el promedio al resultado.
def adjust_brilliance(image, brilliance_factor=1.5):
    image = image.astype(np.float32) 
    mean_brightness = np.mean(image) 
    image = (image - mean_brightness) * brilliance_factor + mean_brightness
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)

    return image

#Ajusta la exposicion de una imagen dada la imagen y un factor de exposicion de parte del usuario.
#Multiplica la imagen por el factor de exposicion.
def adjust_exposure(image, exposure_factor=1.5):
    image = image.astype(np.float32)

    image = image * exposure_factor

    # OVERFLOW HANDLING
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)
    return image


#Busca el porcentaje de rojo de una imagen dada una imagen.
#Suma los valores R, G y B de todos los pixeles de la imagen, los suma y le resta el total de pixeles verdes y azules al total de pixeles rojos, para despues dividir el numero entre el total de pixeles de la imagen.

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
    #Ajusta brillo y exposicion para hacer los blancos mas blancos y los rojos mas rojos.
    image = adjust_brilliance(image, brilliance_factor)
    image = adjust_exposure(image, exposure_factor)
    #Aplica un efecto de difuminado para reducir el nivel de detalles en la imagen que puedan obstruir en la identificacion de colores, como la textura de la tela.
    image = cv2.GaussianBlur(image, blur_ksize, 0)

    #Calcula el porcentaje de rojo de la imagen ya procesada.
    redpercentage = findredp(image)
    return redpercentage


def llamar():
    # Ruta inicial del directorio, misma donde se encuentra el .py
    ruta = r"C:\Users\arace\Desktop\Araceli Escuela\TEC\SEMESTRE 4\SemanaTEC\Laboratorio\Hemomeasure\hemomeasure"
    # Permite abrir archivos .png o .jpg
    fn = askopenfilename(initialdir=ruta,filetypes =[("Archivo tipo Imagen", "*.png;*.jpg")])
    fn2 = askopenfilename(initialdir=ruta,filetypes =[("Archivo tipo Imagen", "*.png;*.jpg")])
    
    # En caso de no ingresar una imagen
    if not (fn or fn2):
        print("No se ha seleccionado una imagen.")
        return

    # OpenCV "lee" las imagenes en las rutas seleccionadas
    image = cv2.imread(fn)
    image2 = cv2.imread(fn2)
    
    #Reduce el tamaño de la imagen en caso de que sea muy grande para facilitar la carga y evitar posibles complicaciones en caso de que la imagen sea muy grande.
    if (image.shape[0]*image.shape[1])>100:
        image = cv2.resize(image, (int(image.shape[1] * 0.25), int(image.shape[0] * 0.25)))

    if (image2.shape[0]*image2.shape[1])>100:
        image2 = cv2.resize(image2, (int(image2.shape[1] * 0.25), int(image2.shape[0] * 0.25)))
    
    #LLama a la funcion con la imagen para calcular sus porcentajes de rojo
    print("LADO A")
    print("////////////")
    redpercentage = hemomeasure(image, brilliance_factor=5, exposure_factor=3, blur_ksize=(123, 123))

    print("LADO B")
    print("////////////")
    redpercentage2 = hemomeasure(image2, brilliance_factor=5, exposure_factor=3, blur_ksize=(123, 123))

    #Multiplica ambos porcentajes de rojo por la mitad de la maxima retencion de sangre de una gaza de 10cm y las suma entre si para aproximar la cantidad de sangre en la gaza.
    
    print("")
    print("SATURACION ESTIMADA EN GAZA: ", (redpercentage*6)+(redpercentage2*6),"ml")


    cv2.imshow("Ejemplo", image)
    cv2.imshow("Ejemplo", image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
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
