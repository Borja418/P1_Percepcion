#! /usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os

LONGITUD = 3.6


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

    img = cv2.imread('Real_Imgs/WIN_20240223_11_33_58_Pro.jpg')
    img = cv2.resize(img, (1200,675))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(img_gray, (9,6), None)
        
    if ret:

        objpoints = np.zeros((9*6,3),np.float32)
        objpoints[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
        objpoints = objpoints*LONGITUD

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(img_gray, corners, (11,11), (-1,-1), criteria)

        img = cv2.drawChessboardCorners(img, (9,6), corners2, ret)
        cv2.imshow('Prueba',img)
        cv2.waitKey(0)

        ret, mtx_img, dist, rvecs_img, tvecs_img = cv2.calibrateCamera([objpoints], [corners2], img_gray.shape[::-1], None, None)
        
        
        rotation_matrix, _ = cv2.Rodrigues(rvecs_img[0])
        
        transformation_matrix = np.c_[rotation_matrix, tvecs_img[0]]

        coordinates = np.matmul(mtx_img, np.matmul(transformation_matrix, np.transpose(np.array([LONGITUD,0,0,1]))))
        
        print(mtx)
        print(coordinates/coordinates[2])

        



        
