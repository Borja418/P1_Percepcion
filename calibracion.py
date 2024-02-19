#! /usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

def calibrar():
    img = cv2.imread("img_calib/pattern.png")

    # Comprobamos que la imagen se ha podido leer
    if img is None:
        print('Error al cargar la imagen')
        quit()

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = np.float32(img_gray)


    blockSize = 2 
    apertureSize = 3 
    k = 0.04 

    dst = cv2.cornerHarris(img_gray, blockSize, apertureSize, k)

    cv2.imshow("Prueba",dst)
    cv2.waitKey()

if __name__ == '__main__':
    calibrar()