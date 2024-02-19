#! /usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

def calibrar(img_str):
    img = cv2.imread(img_str)

    # Comprobamos que la imagen se ha podido leer
    if img is None:
        print('Error al cargar la imagen')
        quit()

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    ret, corners = cv2.findChessboardCorners(img_gray, (9,6), None)
    
    if ret:

        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)

        cv2.imshow("Esquinas",img)
        cv2.waitKey(0)

if __name__ == '__main__':
    calibrar("img_calib/pattern.png")
    calibrar("img_calib/pattern2.png")
    calibrar("img_calib/pattern3.png")