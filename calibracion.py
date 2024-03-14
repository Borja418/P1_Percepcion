#! /usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import letras
import copy
import time
import csv


LONGITUD = 3.6
ESCALA_BORJA = (1280,720)
ESCALA_JUAN = (1200,675)

def calibrar():

    objpoints = np.zeros((9*6,3),np.float32)
    objpoints[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    objpoints = objpoints*LONGITUD

    objpoints_array = []
    imgpoints_array = []
    imgs_name = [f for f in os.listdir("Real_Imgs") if f.endswith('.jpg')]
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    i = 0
    for img_name in imgs_name:
        i+=1
        img = cv2.imread("Real_Imgs/"+img_name)
        # Comprobamos que la imagen se ha podido leer
        if img is None:
            print('Error al cargar la imagen')
            quit()

        img = cv2.resize(img, ESCALA_JUAN)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        ret, corners = cv2.findChessboardCorners(img_gray, (9,6), None)
        
        if ret:
            objpoints_array.append(objpoints)

            corners2 = cv2.cornerSubPix(img_gray, corners, (11,11), (-1,-1), criteria)
            imgpoints_array.append(corners2)

            img = cv2.drawChessboardCorners(img, (9,6), corners2, ret)

            """ cv2.imshow("Esquinas",img)
            cv2.waitKey(0) """
        if i == 24:
            break
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints_array, imgpoints_array, img_gray.shape[::-1], None, None)
    print(ret)
    print(f"Numero de imagenes para la calibracion: {i}")

    return mtx, dist

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



    letra_A = copy.deepcopy(letras.A)
    for point in letra_A:
        point += [12,0,0]

    letra_A = cv2.projectPoints(letra_A, rvecs_img, tvecs_img, newcameramtx, dist)
    Borja_pts = []

    for element in letra_A[0]:
        
        Borja_pts.append(element.ravel().astype(int))

    img = cv2.line(img, Borja_pts[0], Borja_pts[1], (0,255,0), 3)
    img = cv2.line(img, Borja_pts[1], Borja_pts[2], (0,255,0), 3)
    img = cv2.line(img, Borja_pts[2], Borja_pts[3], (0,255,0), 3)
    img = cv2.line(img, Borja_pts[3], Borja_pts[4], (0,255,0), 3)
    img = cv2.line(img, Borja_pts[1], Borja_pts[3], (0,255,0), 3)


    letra_Y = copy.deepcopy(letras.Y)
    for point in letra_Y:
        point += [17,0,0]

    letra_Y = cv2.projectPoints(letra_Y, rvecs_img, tvecs_img, newcameramtx, dist)
    conj_pts = []

    for element in letra_Y[0]:
        
        conj_pts.append(element.ravel().astype(int))

    img = cv2.line(img, conj_pts[0], conj_pts[1], (0,255,0), 3)
    img = cv2.line(img, conj_pts[1], conj_pts[2], (0,255,0), 3)
    img = cv2.line(img, conj_pts[1], conj_pts[3], (0,255,0), 3)


    letra_J = copy.deepcopy(letras.J)
    for point in letra_J:
        point += [22,0,0]

    letra_J = cv2.projectPoints(letra_J, rvecs_img, tvecs_img, newcameramtx, dist)
    Juan_pts = []

    for element in letra_J[0]:
        
        Juan_pts.append(element.ravel().astype(int))

    img = cv2.line(img, Juan_pts[0], Juan_pts[1], (0,255,0), 3)
    img = cv2.line(img, Juan_pts[2], Juan_pts[3], (0,255,0), 3)
    img = cv2.line(img, Juan_pts[3], Juan_pts[4], (0,255,0), 3)
    img = cv2.line(img, Juan_pts[4], Juan_pts[5], (0,255,0), 3)


    letra_U = copy.deepcopy(letras.U)
    for point in letra_U:
        point += [25,0,0]

    letra_U = cv2.projectPoints(letra_U, rvecs_img, tvecs_img, newcameramtx, dist)
    Juan_pts = []

    for element in letra_U[0]:
        
        Juan_pts.append(element.ravel().astype(int))

    img = cv2.line(img, Juan_pts[0], Juan_pts[1], (0,255,0), 3)
    img = cv2.line(img, Juan_pts[1], Juan_pts[2], (0,255,0), 3)
    img = cv2.line(img, Juan_pts[2], Juan_pts[3], (0,255,0), 3)
    img = cv2.line(img, Juan_pts[3], Juan_pts[4], (0,255,0), 3)
    img = cv2.line(img, Juan_pts[4], Juan_pts[5], (0,255,0), 3)
    img = cv2.line(img, Juan_pts[5], Juan_pts[6], (0,255,0), 3)



    letra_A = copy.deepcopy(letras.A)
    for point in letra_A:
        point += [28,0,0]

    letra_A = cv2.projectPoints(letra_A, rvecs_img, tvecs_img, newcameramtx, dist)
    Juan_pts = []

    for element in letra_A[0]:
        
        Juan_pts.append(element.ravel().astype(int))

    img = cv2.line(img, Juan_pts[0], Juan_pts[1], (0,255,0), 3)
    img = cv2.line(img, Juan_pts[1], Juan_pts[2], (0,255,0), 3)
    img = cv2.line(img, Juan_pts[2], Juan_pts[3], (0,255,0), 3)
    img = cv2.line(img, Juan_pts[3], Juan_pts[4], (0,255,0), 3)
    img = cv2.line(img, Juan_pts[1], Juan_pts[3], (0,255,0), 3)


    letra_N = copy.deepcopy(letras.N)
    for point in letra_N:
        point += [31,0,0]

    letra_N = cv2.projectPoints(letra_N, rvecs_img, tvecs_img, newcameramtx, dist)
    Juan_pts = []

    for element in letra_N[0]:
        
        Juan_pts.append(element.ravel().astype(int))

    img = cv2.line(img, Juan_pts[0], Juan_pts[1], (0,255,0), 3)
    img = cv2.line(img, Juan_pts[1], Juan_pts[2], (0,255,0), 3)
    img = cv2.line(img, Juan_pts[2], Juan_pts[3], (0,255,0), 3)



    return img



if __name__ == '__main__':

    mtx, dist = calibrar()
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, ESCALA_JUAN, 0, ESCALA_JUAN)

    objpoints = np.zeros((9*6,3),np.float32)
    objpoints[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    objpoints = objpoints*LONGITUD

    tiempos = []

    cap = cv2.VideoCapture('VideoPrueba.mp4')
    inicio = time.time()
    
    while(cap.isOpened()):
        
        ret, img = cap.read()
        
        if ret:
            img = cv2.resize(img, ESCALA_JUAN)            
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            inicio_chessboard = time.time()
            ret, corners = cv2.findChessboardCorners(img_gray, (9,6), cv2.CALIB_CB_ADAPTIVE_THRESH)
            final_chessboard = time.time()
            if ret:

                inicio_esquinas = time.time()
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(img_gray, corners, (11,11), (-1,-1), criteria)
                
                img = cv2.drawChessboardCorners(img, (9,6), corners2, ret)
                final_esquinas = time.time()

                inicio_mtx = time.time()
                ret, rvecs_img, tvecs_img = cv2.solvePnP(objpoints, corners2, mtx, dist, True, cv2.SOLVEPNP_ITERATIVE)
                final_mtx = time.time()

                inicio_nombres = time.time()
                img = DibujarNombres(img, rvecs_img, tvecs_img, mtx, dist)
                final_nombres = time.time()

                cv2.imshow('frame',img)
                tiempos.append([final_chessboard-inicio_chessboard, final_esquinas-inicio_esquinas, final_mtx-inicio_mtx, final_nombres-inicio_nombres])
            else:
                tiempos.append([final_chessboard-inicio_chessboard, 0, 0, 0])
                cv2.imshow('frame',img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            #print(f"Tiempo Chess: {final_chessboard-inicio_chessboard} Tiempo Esq: {final_esquinas-inicio_esquinas} Tiempo Mtx: {final_mtx-inicio_mtx} Tiempo Nombres: {final_nombres-inicio_nombres}")
        else:
            break
    print(f"Procesamiento: {time.time()-inicio} ")
# Cuando terminemos, paramos la captura
    cap.release()


    with open("VideoPruebaV2.csv", 'w') as f:

        csv.excel.delimiter=";"
        csv_writer = csv.writer(f, dialect=csv.excel)
        csv_writer.writerow(['Encontrar Chessboard', 'Ajustar Esquinas', 'Calcular Matriz', 'Proyectar Puntos'])
        csv_writer.writerows(tiempos)



        



        
