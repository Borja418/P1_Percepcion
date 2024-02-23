#! /usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os

def calibrar():

    imgs_name = [f for f in os.listdir("Real_Imgs") if f.endswith('.jpg')]
    
    for img_name in imgs_name:

        img = cv2.imread("Real_Imgs/"+img_name)
        print(img_name)
        # Comprobamos que la imagen se ha podido leer
        if img is None:
            print('Error al cargar la imagen')
            quit()

        img = cv2.resize(img, (1200,675))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        ret, corners = cv2.findChessboardCorners(img_gray, (9,6), None)
        
        if ret:

            img = cv2.drawChessboardCorners(img, (9,6), corners, ret)

            cv2.imshow("Esquinas",img)
            cv2.waitKey(0)

    print(f"Numero de imagenes para la calibracion: {len(imgs_name)}")

if __name__ == '__main__':
    calibrar()