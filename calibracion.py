#! /usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import letras
import copy

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

def DibujarNombres(img, rvecs_img, tvecs_img, newcameramtx, dist):
    

    letra_B = cv2.projectPoints(letras.B, rvecs_img, tvecs_img, newcameramtx, dist)
    Borja_pts = []

    for element in letra_B[0]:
        
        Borja_pts.append(element.ravel().astype(int))

    img = cv2.line(img, Borja_pts[0], Borja_pts[1], (0,255,0), 3)
    img = cv2.line(img, Borja_pts[1], Borja_pts[2], (0,255,0), 3)
    img = cv2.line(img, Borja_pts[2], Borja_pts[3], (0,255,0), 3)
    img = cv2.line(img, Borja_pts[3], Borja_pts[4], (0,255,0), 3)
    img = cv2.line(img, Borja_pts[4], Borja_pts[5], (0,255,0), 3)
    img = cv2.line(img, Borja_pts[4], Borja_pts[6], (0,255,0), 3)
    img = cv2.line(img, Borja_pts[6], Borja_pts[7], (0,255,0), 3)
    img = cv2.line(img, Borja_pts[7], Borja_pts[0], (0,255,0), 3)
    
    letra_O = copy.deepcopy(letras.O)
    for point in letra_O:
        point += [3,0,0]


    letra_O = cv2.projectPoints(letra_O, rvecs_img, tvecs_img, newcameramtx, dist)
    Borja_pts = []

    for element in letra_O[0]:
        
        Borja_pts.append(element.ravel().astype(int))

    img = cv2.line(img, Borja_pts[0], Borja_pts[1], (0,255,0), 3)
    img = cv2.line(img, Borja_pts[1], Borja_pts[2], (0,255,0), 3)
    img = cv2.line(img, Borja_pts[2], Borja_pts[3], (0,255,0), 3)
    img = cv2.line(img, Borja_pts[3], Borja_pts[0], (0,255,0), 3)

    letra_R = copy.deepcopy(letras.R)
    for point in letra_R:
        point += [6,0,0]

    letra_R = cv2.projectPoints(letra_R, rvecs_img, tvecs_img, newcameramtx, dist)
    Borja_pts = []

    for element in letra_R[0]:
        
        Borja_pts.append(element.ravel().astype(int))

    img = cv2.line(img, Borja_pts[0], Borja_pts[1], (0,255,0), 3)
    img = cv2.line(img, Borja_pts[1], Borja_pts[2], (0,255,0), 3)
    img = cv2.line(img, Borja_pts[2], Borja_pts[3], (0,255,0), 3)
    img = cv2.line(img, Borja_pts[3], Borja_pts[4], (0,255,0), 3)
    img = cv2.line(img, Borja_pts[5], Borja_pts[6], (0,255,0), 3)

    letra_J = copy.deepcopy(letras.J)
    for point in letra_J:
        point += [9,0,0]

    letra_J = cv2.projectPoints(letra_J, rvecs_img, tvecs_img, newcameramtx, dist)
    Borja_pts = []

    for element in letra_J[0]:
        
        Borja_pts.append(element.ravel().astype(int))

    img = cv2.line(img, Borja_pts[0], Borja_pts[1], (0,255,0), 3)
    img = cv2.line(img, Borja_pts[2], Borja_pts[3], (0,255,0), 3)
    img = cv2.line(img, Borja_pts[3], Borja_pts[4], (0,255,0), 3)
    img = cv2.line(img, Borja_pts[4], Borja_pts[5], (0,255,0), 3)


    return img



if __name__ == '__main__':

    mtx, dist, rvecs, tvecs = calibrar()

    cap = cv2.VideoCapture('VideoPrueba.mp4')

    while(cap.isOpened()):
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

            img = DibujarNombres(img, rvecs_img, tvecs_img, newcameramtx, dist)

            cv2.imshow('frame',img)
        else:
            cv2.imshow('frame',img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cuando terminemos, paramos la captura
    cap.release()



        



        
