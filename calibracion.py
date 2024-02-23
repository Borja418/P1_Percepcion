#! /usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os

LONGITUD = 2


def calibrar():

    objpoints = np.zeros((9*6,3),np.float32)
    objpoints[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    objpoints = objpoints*LONGITUD

    objpoints_array = []
    imgpoints_array = []
    imgs_name = [f for f in os.listdir("Real_Imgs") if f.endswith('.jpg')]
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for img_name in imgs_name:

        img = cv2.imread("Real_Imgs/"+img_name)
        # Comprobamos que la imagen se ha podido leer
        if img is None:
            print('Error al cargar la imagen')
            quit()

        img = cv2.resize(img, (1200,675))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        ret, corners = cv2.findChessboardCorners(img_gray, (9,6), None)
        
        if ret:
            objpoints_array.append(objpoints)

            corners2 = cv2.cornerSubPix(img_gray, corners, (11,11), (-1,-1), criteria)
            imgpoints_array.append(corners2)

            img = cv2.drawChessboardCorners(img, (9,6), corners2, ret)

            """ cv2.imshow("Esquinas",img)
            cv2.waitKey(0) """
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints_array, imgpoints_array, img_gray.shape[::-1], None, None)

    print(f"Numero de imagenes para la calibracion: {len(imgs_name)}")

    return mtx, dist, rvecs, tvecs

if __name__ == '__main__':
    mtx, dist, rvecs, tvecs = calibrar()