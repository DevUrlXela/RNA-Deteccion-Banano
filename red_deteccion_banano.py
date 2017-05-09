# coding=utf-8

from __future__ import division
from datetime import datetime
from Tkinter import *
from tkFileDialog import askopenfilename
from PIL import Image
from matplotlib import pyplot as plt
from math import cos, sin
import neurolab as nl
import cv2
import imutils
import numpy as np
import webcolors as webcolors
import os

green = (0, 255, 0)
np.set_printoptions(suppress=True)

redcolors = (
'lightsalmon',
'salmon',
'darksalmon',
'lightcoral',
'indianred',
'crimson',
'firebrick',
'darkred',
'red'
)

orangecolors = (
'orangered',
'tomato',
'coral',
'darkorange',
'orange'
)


verdeOscuro = (
'darkolivegreen',
'olive',
'olivedrab',
'yellowgreen',
'darkseagreen',
'mediumaquamarine',
'mediumseagreen',
'darkseagreen',
'seagreen',
'forestgreen',
'green',
'darkgreen'
)


verdeClaro = (
'lime',
'lawngreen',
'limegreen',
'chartreuse',
'greenyellow',
'springgreen',
'mediumspringgreen',
'lightgreen',
'palegreen'
)


amarillosC = (
'yellow',
'lightyellow',
'lemonchiffon',
'lightgoldenrodyellow',
'papayawhip',
'moccasin',
'peachpuff',
)

amarillosO = (
'khaki',
'darkkhaki'
'gold',
'palegoldenrod',
'goldenrod',
'chocolate',
'darkgoldenrod'
)


brown = (
'sandybrown',
'rosybrown',
'tan',
'burlywood',
'wheat',
'navajowhite',
'bisque',
'blanchedalmond',
'cornsilk',
'peru',
'saddlebrown',
'sienna',
'brown',
'maroon',
)

black = (
'gainsboro',
'lightgray',
'silver',
'darkgray',
'gray',
'dimgray',
'lightslategray',
'slategray',
'darkslategray',
'black'
)

newnet = nl.load('pesos.net')

class Window(Frame):
    def __init__(self, master = None):
        Frame.__init__(self,master)

        self.master = master
        self.text = Label(self, font="Arial 14 bold", height=200)
        self.init_window()

    def init_window(self):
        self.master.title("Banana Detect")

        self.pack(fill=BOTH, expand=1)
        quitButton = Button(self, text="Quit", command=self.client_exit)
        quitButton.place(x=0,y=25)

        filechooser = Button(self, text="Choose file", command=self.file_chooser)
        filechooser.place(x=0,y=0)

    def client_exit(self):
        exit()

    def file_chooser(self):
        filename = askopenfilename()
        print(filename)

        image = cv2.imread(filename)
        result, cropped = find_banana(image)

        entradas = color_percent(cropped)

        salidas = newnet.sim([entradas])
        salidas = list(salidas[0])

        print "Entradas: {}\nSalida: {}".format(entradas,salidas)

        texto = "Estado del Banano:\n{}".format(verificar_estado(salidas))

        print texto

        self.text.config(text = texto)
        self.text.pack()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        plt.subplot(121),plt.imshow(image),plt.title('Input')
        plt.subplot(122),plt.imshow(result),plt.title('Output')
        plt.show()

        # load = Image.open(filename)
        # img = Image._show(load)

def verificar_estado(salidas):
    maximo = float(max(salidas))
    print float(maximo)
    estado = salidas.index(maximo)
    print estado

    if salidas.index(maximo) == 0:
        return "Recien cortado, inmaduro."
    elif salidas.index(maximo) == 1:
        return "Madurando."
    elif salidas.index(maximo) == 2:
        return "Maduro, en su punto."
    elif salidas.index(maximo) == 3:
        return "Pasado de maduro, a punto de terminar su tiempo de consumo."
    else:
        return "Pasado, en estado de descomposicion."

def color_percent(image):
    pixeles = image
    longitud = len(pixeles[0])

    # print "Longitud de Matriz: {}".format(longitud)

    contRed = 0
    contOrange = 0
    contDarkGreen = 0
    contLightGreen = 0
    contLightYellow = 0
    contDarkYellow = 0
    contBrown = 0
    contBlack = 0
    contOtros = 0

    for pixel in pixeles[0]:
        # print "Pixel: {}".format(pixel)
        actual_name, closest_name = get_colour_name(pixel)

        if closest_name in redcolors:
            contRed += 1
        elif closest_name in orangecolors:
            contOrange += 1
        elif closest_name in verdeOscuro:
            contDarkGreen += 1
        elif closest_name in verdeClaro:
            contLightGreen += 1
        elif closest_name in amarillosC:
            contLightYellow += 1
        elif closest_name in amarillosO:
            contDarkYellow += 1
        elif closest_name in brown:
            contBrown += 1
        elif closest_name in black:
            contBlack +=1
        else:
            contOtros += 1

    contRed = round((contRed * 100) / longitud, 4)
    contOrange = round((contOrange * 100) / longitud, 4)
    contDarkGreen = round((contDarkGreen * 100) / longitud, 4)
    contLightGreen = round((contLightGreen * 100) / longitud, 4)
    contLightYellow = round((contLightYellow * 100) / longitud, 4)
    contDarkYellow = round((contDarkYellow * 100) / longitud, 4)
    contBrown = round((contBrown * 100) / longitud, 4)
    contBlack = round((contBlack * 100) / longitud, 4)
    contOtros = round((contOtros * 100) / longitud, 4)

    print "Rojo: {}%\nNaranja: {}%\nVerdeO: {}%\nVerdeC: {}\nAmarilloO: {}%\nAmarilloC: {}%\nCafe: {}%\nNegro:{}%\nOtros: {}%".format(
                                        contRed,
                                        contOrange,
                                        contDarkGreen,
                                        contLightGreen,
                                        contDarkYellow,
                                        contLightYellow,
                                        contBrown,
                                        contBlack,
                                        contOtros
                                    )

    nueva_linea = [
        contRed,
        contOrange,
        contDarkGreen,
        contLightGreen,
        contDarkYellow,
        contLightYellow,
        contBrown,
        contBlack,
        contOtros
    ]

    return nueva_linea

def displayImage(image):
    displayList=np.array(image).T
    im1 = Image.fromarray(displayList)
    im1.show()

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name

def overlay_mask(mask, image):
	#Mascara en RGB
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    #Se calculan los pesos para los arreglos de la imagen
    #y la entrada es el valor de cada peso.
    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    return img

def find_biggest_contour(image):
    #Copia la imagen
    image = image.copy()
    #Se obtienen todos los contornos (horizontal, vertical y diagonal), y
    #puntos del segmento.
    #Los guiones son para separar informacion innecesaria de retorno de la
    #funcion.
    _, contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Se recorren los contornos hasta aislar el mas grande.
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
    return biggest_contour, mask

def circle_contour(image, contour, ruta):

    # Limitando la elipse
    image_with_ellipse = image.copy()
    image_crop = image.copy()
    shape = image_with_ellipse.shape

    # print "Shape: {}".format(shape[:2])
    #Funcion para el contorno en forma de elipse

    # x, y, w, h = cv2.boundingRect(contour)
    # epsilon = 0.1*cv2.arcLength(contour, True)
    # approx = cv2.approxPolyDP(contour, epsilon, True)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(image_with_ellipse, [box], 0, (255,0,0), 2)
    #=========================================================
    eje_x = np.array(box[:,0:1])
    eje_y = np.array(box[:,1:])

    # print "Rectangulo: {}".format(rect)
    # print "Puntos en X: {}\nPuntos en Y: {}".format(eje_x, eje_y)

    # print box
    rotated = imutils.rotate_bound(image_with_ellipse, 90-rect[2])
    # box_p = imutils.rotate_bound(rect, 360-rect[2])
    dim1 = int(rect[1][0])
    dim2 = int(rect[1][1])
    # print "dim1: {}\ndim2: {}".format(dim1, dim2)
    if ruta == 'resultados':
        box1 = cv2.boxPoints((rect[0], (dim1/3, dim2/3), rect[2]))
    else:
        box1 = cv2.boxPoints((rect[0], (200, 200), rect[2]))

    box1 = np.int0(box1)
    eje_x2 = np.array(box1[:,0:1])
    eje_y2 = np.array(box1[:,1:])
    # print box1

    # plt.subplot(121),plt.plot(eje_x,eje_y, "o")
    # plt.subplot(121),plt.plot(eje_x2,eje_y2)
    #
    # plt.subplot(121),plt.imshow(image_with_ellipse),plt.title('Input')
    # plt.subplot(122),plt.imshow(rotated),plt.title('Output')
    # plt.show()

    min_x = min(eje_x2)[0]
    max_x = max(eje_x2)[0]
    min_y = min(eje_y2)[0]
    max_y = max(eje_y2)[0]

    crop_img = image_crop[min_y:max_y, min_x:max_x]
    cv2.imwrite(ruta+'/cropped.jpg', crop_img)
    #=========================================================
    # center = (int(ellipse[0][0]),int(ellipse[0][1]))
    # radius = 50
    # print ellipse
    #
    # for elemento in 0,1:
    #     ellipse[0][elemento] = ellipse[0][elemento]/3
    #     print elemento
    #

    # rotado, (ancho, altura) = rotar(image_with_ellipse, ellipse[2])

    #Se agrega la elipse a la imagen
    # cv2.rectangle(image_with_ellipse, (x,y), (x+w, y+h), (0, 255, 0), 2)
    # cv2.circle(image_with_ellipse, center, radius, (0,0,255), 2)
    return image_with_ellipse, crop_img

def find_banana(image, ruta = 'resultados'):
    #Se invierte el esquema RGB a BGR, ya que las funciones son más compatibles
    #teniendo el azul con más relevancia
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.imwrite('resultados/bgr.jpg', image)

    #Un tamaño fijo
    #con la dimension mas grande
    max_dimension = max(image.shape)
    #La escala de la imagen de salida no será mayor a 700px
    scale = 700/max_dimension
    #Se redimenciona la imagen para que sea cuadrada.
    image = cv2.resize(image, None, fx=scale, fy=scale)

    #Reducimos el ruido de la imagen usando el filtro Gaussiano, con la escala
    #maxima cuadrada.
    # image_blur = cv2.bilateralFilter(image,9,75,75)
    image_blur = cv2.GaussianBlur(image, (7, 7), 0)
    cv2.imwrite( ruta+'/blur.jpg', image_blur)

    #Tratamos de enfocarnos en el color, y por esta razón nos enfocamos en
    #el esquema HSV, pues resalta el color y maneja solo saturacion y
    #valor
    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)
    cv2.imwrite(ruta+'/hsv.jpg', image_blur_hsv)

    #kernel = np.ones((5,5),np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    erosion = cv2.erode(image_blur_hsv, kernel, iterations = 1)
    cv2.imwrite(ruta+'/erosionado.jpg', erosion)
    dilation = cv2.dilate(image_blur_hsv, kernel, iterations = 1)
    cv2.imwrite(ruta+'/dilatado.jpg', dilation)
    dilation_blur = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(ruta+'/dilatado_blur.jpg', dilation_blur)

    # Filtro por color
    # 20-30 hue
    """Aqui tenemos un problema, pues decidimos colocar un rango de amarillos
    pero no lo reconoce en la imagen y tenemos dificultad al reconocer este
    rango, ya que a veces toma colores que no debe. Según el tono es de
    50 a 70 para amarillos"""
    #hsv(15, 80, 50)
    #hsv(105, 120, 255)
    min_yellow = np.array([15, 100, 80])
    max_yellow = np.array([105, 255, 255])
    # min_yellow = np.array([20, 100, 80])
    # max_yellow = np.array([30, 255, 255])
    #layer
    mask1 = cv2.inRange(dilation_blur, min_yellow, max_yellow)

    #hsv(230, 0, 0)
    #hsv(270, 255, 255)
    black_min = np.array([130, 0, 0])
    black_max = np.array([170, 255, 255])
    black_mask = cv2.inRange(dilation_blur, black_min, black_max)
    cv2.imwrite(ruta+'/mascara_negro.jpg', black_mask)

    #Filtro por brillo
    # 170-180 hue
    #Tratamos de resaltar el brillo para tener un mejor reconocimiento de
    #colores.
    #hsv(170,100,80)
    #hsv(180,255,255)
    min_yellow2 = np.array([170, 100, 80])
    max_yellow2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(dilation_blur, min_yellow2, max_yellow2)
    cv2.imwrite(ruta+'/mascara1.jpg', mask1)
    cv2.imwrite(ruta+'/mascara2.jpg', mask2)

    #Combinamos las mascaras de colores.
    mask = mask1 + mask2 + black_mask
    cv2.imwrite(ruta+'/mask.jpg', mask)
    # opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel)
    # cv2.imwrite('resultados/opening.jpg', opening)

    # Se limpia la imagen y se crea la elipse.

    #Se erosiona la imagen para reducir espacios sin color. Y luego se dilata,
    #Esto dentro de lo que buscamos encerrar.
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_closed = cv2.dilate(mask_closed, kernel, iterations = 3)
    # mask_closed = cv2.dilate(mask_closed, kernel, iterations = 1)
    # mask_closed = cv2.morphologyEx(mask_closed, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(ruta+'/closed.jpg', mask_closed)
    #Se dilata para reducr ruido afuera de lo que identificamos, y luego se erosiona.
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(ruta+'/open.jpg', mask_clean)

    # Se encuentra el mejor patron y se recibe el contorno
    big_banana_contour, mask_bananas = find_biggest_contour(mask_clean)

    # Se resalta la mascara limpia y se aclara en la imagen.
    overlay = overlay_mask(mask_clean, image)
    cv2.imwrite(ruta+'/overlay.jpg', overlay)

    #Se circula el patron con mejor coincidencia.
    circled, cropped = circle_contour(image, big_banana_contour, ruta)

    #Y convertimos al esquema de original de colores.
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

    return circled, cropped

root = Tk()
root.geometry("400x300")
app = Window(root)
root.mainloop()
