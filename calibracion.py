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

    cap = cv2.VideoCapture(0)

    while(True):
        ret, img = cap.read()
        
        img = cv2.resize(img, (1200,675))

        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(img_gray, (9,6), None)
            
        if ret:

            objpoints = np.zeros((9*6,3),np.float32)
            objpoints[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
            objpoints = objpoints*LONGITUD

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(img_gray, corners, (11,11), (-1,-1), criteria)

            img = cv2.drawChessboardCorners(img, (9,6), corners2, ret)

            ret, rvecs_img, tvecs_img = cv2.solvePnP(objpoints, corners2, newcameramtx, dist)
            
            rotation_matrix, _ = cv2.Rodrigues(rvecs_img)
            
            transformation_matrix = np.c_[rotation_matrix, tvecs_img]

            coordinates = np.matmul(newcameramtx, np.matmul(transformation_matrix, np.transpose(np.array([0,0,0,1]))))
            coordinates2 = np.matmul(newcameramtx, np.matmul(transformation_matrix, np.transpose(np.array([LONGITUD*4,0,-10,1]))))
            coordinates3 = np.matmul(newcameramtx, np.matmul(transformation_matrix, np.transpose(np.array([LONGITUD*8,0,0,1]))))

            coordinates = coordinates/coordinates[2]
            coordinates2 = coordinates2/coordinates2[2]
            coordinates3 = coordinates3/coordinates3[2]

            coordinates = coordinates.astype(int)
            coordinates2 = coordinates2.astype(int)
            coordinates3 = coordinates3.astype(int)

            img = cv2.line(img, (coordinates[0], coordinates[1]), (coordinates2[0], coordinates2[1]), (0,255,0), 3)
            img = cv2.line(img, (coordinates2[0], coordinates2[1]), (coordinates3[0], coordinates3[1]), (0,255,0), 3)

            cv2.imshow('frame',img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cuando terminemos, paramos la captura
    cap.release()



        



        
