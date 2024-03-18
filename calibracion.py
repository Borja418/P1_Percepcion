#! /usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import letras
import copy
import time
import csv
import math

LONGITUD = 3.6
ESCALA_BORJA = (1280,720)
ESCALA_JUAN = (1200,675)

def Rz(theta):
  return np.matrix([[ math.cos(theta), -math.sin(theta), 0 ],
                   [ math.sin(theta), math.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])

def Transform_Points(points,x,y,z,theta,x_post,y_post,z_post):
    
    
    tmp_points = []
    

    for point in points:
        point += [x,y,z]
        point = np.dot(Rz(theta), point)
        point += [x_post,y_post,z_post]
        tmp_points.append(point.flatten())
    
    tmp_points = np.array(tmp_points)
    
    return tmp_points





def calibrar():

    objpoints = np.zeros((9*6,3),np.float32)
    objpoints[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    objpoints = objpoints*LONGITUD

    objpoints_array = []
    imgpoints_array = []
    imgs_name = [f for f in os.listdir("Real_Imgs") if f.endswith('.jpg')]
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    count = 0
    for img_name in imgs_name:
        count+=1
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
            
        if count == 20:
            break
        
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints_array, imgpoints_array, img_gray.shape[::-1], None, None)
    print(ret)
    print(f"Numero de imagenes para la calibracion: {count}")
                
    mean_error = 0
    for i in range(len(objpoints_array)):
        imgpoints2, _ = cv2.projectPoints(objpoints_array[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints_array[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
        print(error)
 
    print( "total error: {}".format(mean_error/len(objpoints_array)) )

    return mtx, dist

def DibujarNombres(img, rvecs_img, tvecs_img, mtx, dist):
    

    letra_B = copy.deepcopy(letras.B)
    
    letra_B = Transform_Points(letra_B,0,0,0,-math.pi/4,0,3*LONGITUD,0)

    letra_B = cv2.projectPoints(letra_B, rvecs_img, tvecs_img, mtx, dist)
    Pts = []

    for element in letra_B[0]:
        
        Pts.append(element.ravel().astype(int))

    img = cv2.line(img, Pts[0], Pts[1], (0,255,0), 3)
    img = cv2.line(img, Pts[1], Pts[2], (0,255,0), 3)
    img = cv2.line(img, Pts[2], Pts[3], (0,255,0), 3)
    img = cv2.line(img, Pts[3], Pts[4], (0,255,0), 3)
    img = cv2.line(img, Pts[4], Pts[5], (0,255,0), 3)
    img = cv2.line(img, Pts[4], Pts[6], (0,255,0), 3)
    img = cv2.line(img, Pts[6], Pts[7], (0,255,0), 3)
    img = cv2.line(img, Pts[7], Pts[0], (0,255,0), 3)
    

    letra_O = copy.deepcopy(letras.O)
    
    letra_O = Transform_Points(letra_O,3,0,0,-math.pi/4,0,3*LONGITUD,0)
        
    letra_O = cv2.projectPoints(letra_O, rvecs_img, tvecs_img, mtx, dist)
    Pts = []

    for element in letra_O[0]:
        
        Pts.append(element.ravel().astype(int))

    img = cv2.line(img, Pts[0], Pts[1], (0,255,0), 3)
    img = cv2.line(img, Pts[1], Pts[2], (0,255,0), 3)
    img = cv2.line(img, Pts[2], Pts[3], (0,255,0), 3)
    img = cv2.line(img, Pts[3], Pts[0], (0,255,0), 3)

    letra_R = copy.deepcopy(letras.R)
    
    letra_R = Transform_Points(letra_R,6,0,0,-math.pi/4,0,3*LONGITUD,0)

    letra_R = cv2.projectPoints(letra_R, rvecs_img, tvecs_img, mtx, dist)
    Pts = []

    for element in letra_R[0]:
        
        Pts.append(element.ravel().astype(int))

    img = cv2.line(img, Pts[0], Pts[1], (0,255,0), 3)
    img = cv2.line(img, Pts[1], Pts[2], (0,255,0), 3)
    img = cv2.line(img, Pts[2], Pts[3], (0,255,0), 3)
    img = cv2.line(img, Pts[3], Pts[4], (0,255,0), 3)
    img = cv2.line(img, Pts[5], Pts[6], (0,255,0), 3)

    letra_J = copy.deepcopy(letras.J)
    
    letra_J = Transform_Points(letra_J,9,0,0,-math.pi/4,0,3*LONGITUD,0)

    letra_J = cv2.projectPoints(letra_J, rvecs_img, tvecs_img, mtx, dist)
    Pts = []

    for element in letra_J[0]:
        
        Pts.append(element.ravel().astype(int))

    img = cv2.line(img, Pts[0], Pts[1], (0,255,0), 3)
    img = cv2.line(img, Pts[2], Pts[3], (0,255,0), 3)
    img = cv2.line(img, Pts[3], Pts[4], (0,255,0), 3)
    img = cv2.line(img, Pts[4], Pts[5], (0,255,0), 3)



    letra_A = copy.deepcopy(letras.A)
    
    letra_A = Transform_Points(letra_A,12,0,0,-math.pi/4,0,3*LONGITUD,0)

    letra_A = cv2.projectPoints(letra_A, rvecs_img, tvecs_img, mtx, dist)
    Pts = []

    for element in letra_A[0]:
        
        Pts.append(element.ravel().astype(int))

    img = cv2.line(img, Pts[0], Pts[1], (0,255,0), 3)
    img = cv2.line(img, Pts[1], Pts[2], (0,255,0), 3)
    img = cv2.line(img, Pts[2], Pts[3], (0,255,0), 3)
    img = cv2.line(img, Pts[3], Pts[4], (0,255,0), 3)
    img = cv2.line(img, Pts[1], Pts[3], (0,255,0), 3)


    letra_Y = copy.deepcopy(letras.Y)
    
    letra_Y = Transform_Points(letra_Y,4*LONGITUD-1,0,0,0,0,0,0)

    letra_Y = cv2.projectPoints(letra_Y, rvecs_img, tvecs_img, mtx, dist)
    Pts = []

    for element in letra_Y[0]:
        
        Pts.append(element.ravel().astype(int))

    img = cv2.line(img, Pts[0], Pts[1], (0,255,0), 3)
    img = cv2.line(img, Pts[1], Pts[2], (0,255,0), 3)
    img = cv2.line(img, Pts[1], Pts[3], (0,255,0), 3)


    letra_J = copy.deepcopy(letras.J)
    
    letra_J = Transform_Points(letra_J,0,0,0,math.pi/4,6*LONGITUD,0,0)

    letra_J = cv2.projectPoints(letra_J, rvecs_img, tvecs_img, mtx, dist)
    Pts = []

    for element in letra_J[0]:
        
        Pts.append(element.ravel().astype(int))

    img = cv2.line(img, Pts[0], Pts[1], (0,255,0), 3)
    img = cv2.line(img, Pts[2], Pts[3], (0,255,0), 3)
    img = cv2.line(img, Pts[3], Pts[4], (0,255,0), 3)
    img = cv2.line(img, Pts[4], Pts[5], (0,255,0), 3)


    letra_U = copy.deepcopy(letras.U)
    
    letra_U = Transform_Points(letra_U,3,0,0,math.pi/4,6*LONGITUD,0,0)

    letra_U = cv2.projectPoints(letra_U, rvecs_img, tvecs_img, mtx, dist)
    Pts = []

    for element in letra_U[0]:
        
        Pts.append(element.ravel().astype(int))

    img = cv2.line(img, Pts[0], Pts[1], (0,255,0), 3)
    img = cv2.line(img, Pts[1], Pts[2], (0,255,0), 3)
    img = cv2.line(img, Pts[2], Pts[3], (0,255,0), 3)
    img = cv2.line(img, Pts[3], Pts[4], (0,255,0), 3)
    img = cv2.line(img, Pts[4], Pts[5], (0,255,0), 3)
    img = cv2.line(img, Pts[5], Pts[6], (0,255,0), 3)



    letra_A = copy.deepcopy(letras.A)
    
    letra_A = Transform_Points(letra_A,6,0,0,math.pi/4,6*LONGITUD,0,0)

    letra_A = cv2.projectPoints(letra_A, rvecs_img, tvecs_img, mtx, dist)
    Pts = []

    for element in letra_A[0]:
        
        Pts.append(element.ravel().astype(int))

    img = cv2.line(img, Pts[0], Pts[1], (0,255,0), 3)
    img = cv2.line(img, Pts[1], Pts[2], (0,255,0), 3)
    img = cv2.line(img, Pts[2], Pts[3], (0,255,0), 3)
    img = cv2.line(img, Pts[3], Pts[4], (0,255,0), 3)
    img = cv2.line(img, Pts[1], Pts[3], (0,255,0), 3)


    letra_N = copy.deepcopy(letras.N)
    
    letra_N = Transform_Points(letra_N,9,0,0,math.pi/4,6*LONGITUD,0,0)

    letra_N = cv2.projectPoints(letra_N, rvecs_img, tvecs_img, mtx, dist)
    Pts = []

    for element in letra_N[0]:
        
        Pts.append(element.ravel().astype(int))

    img = cv2.line(img, Pts[0], Pts[1], (0,255,0), 3)
    img = cv2.line(img, Pts[1], Pts[2], (0,255,0), 3)
    img = cv2.line(img, Pts[2], Pts[3], (0,255,0), 3)



    return img



if __name__ == '__main__':

    mtx, dist = calibrar()

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


    '''with open("VideoPruebaV2.csv", 'w') as f:

        csv.excel.delimiter=";"
        csv_writer = csv.writer(f, dialect=csv.excel)
        csv_writer.writerow(['Encontrar Chessboard', 'Ajustar Esquinas', 'Calcular Matriz', 'Proyectar Puntos'])
        csv_writer.writerows(tiempos)'''



        



        
