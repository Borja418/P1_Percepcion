#! /usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

def calibrar():
    img = cv2.imread(args.imagen)

    # Comprobamos que la imagen se ha podido leer
    if img is None:
        print('Error al cargar la imagen')
        quit()

    # Pasamos la imagen a escala de grises, y despu茅s a float32
    #聽TODO (guardar en img_gray)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = np.float32(img_gray)


    blockSize = 2 
    apertureSize = 3 
    k = 0.04 

    dst = cv2.cornerHarris(img_gray, blockSize, apertureSize, k)

    cv2.imshow(dst)

if __name__ == '__main__':
    calibrar()