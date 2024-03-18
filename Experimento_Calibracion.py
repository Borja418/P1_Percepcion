#! /usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os


LONGITUD = 3.6
ESCALA_BORJA = (1280,720)
ESCALA_JUAN = (1200,675)

URI_BORJA = "MovilBorja"
URI_JUAN = "Real_Imgs"

URI = URI_BORJA
ESCALA = ESCALA_BORJA

OBJPOINTS = np.zeros((9*6,3),np.float32)
OBJPOINTS[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
OBJPOINTS = OBJPOINTS*LONGITUD

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def Calibrar(num_img):
    
    
    count = 0
    objpoints_array = []
    imgpoints_array = []

    imgs_name = [f for f in os.listdir(URI) if f.endswith('.jpg')]

    for img_name in imgs_name:
        
        img = cv2.imread(URI+"/"+img_name)
        # Comprobamos que la imagen se ha podido leer
        if img is None:
            print('Error al cargar la imagen')
            quit()

        img = cv2.resize(img, ESCALA)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(img_gray, (9,6), None)
        
        if ret:
            count += 1
            objpoints_array.append(OBJPOINTS)

            corners2 = cv2.cornerSubPix(img_gray, corners, (11,11), (-1,-1), CRITERIA)
            imgpoints_array.append(corners2)
        
        if count == num_img:
            break
            
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints_array, imgpoints_array, img_gray.shape[::-1], None, None)
    
    return mtx, dist

def Comprobar_Error(mtx, dist):
    mean_error = 0
    count_imgs = 0
    imgs_name = [f for f in os.listdir(URI) if f.endswith('.jpg')]
    
    objpoints_array = []
    imgpoints_array = []

    for img_name in imgs_name:
        
        img = cv2.imread(URI+"/"+img_name)
        
        if img is None:
            print('Error al cargar la imagen')
            quit()

        img = cv2.resize(img, ESCALA)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(img_gray, (9,6), None)
        print(img_name, ret)
        if ret:
            
            count_imgs += 1
            corners2 = cv2.cornerSubPix(img_gray, corners, (11,11), (-1,-1), CRITERIA)
            
            ret, rvecs, tvecs = cv2.solvePnP(OBJPOINTS, corners2, mtx, dist, True, cv2.SOLVEPNP_ITERATIVE)
            
            imgpoints2, _ = cv2.projectPoints(OBJPOINTS, rvecs, tvecs, mtx, dist)
            error = cv2.norm(corners2, imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
    

    mean_error = mean_error/count_imgs
    
    return mean_error

def main():
    imgs_name = [f for f in os.listdir(URI) if f.endswith('.jpg')]


    for num_img_calib in range(1,len(imgs_name)+1):
        
        mtx, dist = Calibrar(num_img_calib)
        
        error = Comprobar_Error(mtx, dist)

        print(f"El error de calibracion con {num_img_calib} imagenes, es {error}")


if __name__ == "__main__":
    
    main()
