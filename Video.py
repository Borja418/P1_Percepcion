#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Importar librerías

import cv2
import numpy as np
import os
import letras
import copy
import time
import csv
import math

# Variables globales

LONGITUD = 3.6

ESCALA_BORJA = (1280,720)
ESCALA_BORJA_ORIGINAL = (1920,1080)
ESCALA_JUAN = (1200,675)
ESCALA_PEQ = (1920,1080)

URI_BORJA = "Movil"
URI_JUAN = "Ordenador"
URI_PATTERNPEQ = "PatternPeq"

VIDEO_BORJA = "VideoMovil.mp4"
VIDEO_JUAN = "VideoOrdenador.mp4"
VIDEO_PEQ = "VideoPatternPeq.mp4"

GRID_GRANDE = (9,6)
GRID_PEQ = (5,4)

OBJPOINTS_GRANDE = np.zeros((9*6,3),np.float32)
OBJPOINTS_GRANDE[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
OBJPOINTS_GRANDE = OBJPOINTS_GRANDE*LONGITUD

OBJPOINTS_PEQ = np.zeros((5*4,3),np.float32)
OBJPOINTS_PEQ[:,:2] = np.mgrid[0:5,0:4].T.reshape(-1,2)
OBJPOINTS_PEQ = OBJPOINTS_PEQ*LONGITUD

URI = URI_PATTERNPEQ
ESCALA = ESCALA_PEQ
OBJPOINTS = OBJPOINTS_PEQ
GRID = GRID_PEQ
VIDEO = VIDEO_PEQ

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Función para obtener la matriz de rotación en eje z

def Rz(theta):
  return np.matrix([[ math.cos(theta), -math.sin(theta), 0 ],
                   [ math.sin(theta), math.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])

# Función para aplicar transformadas a los puntos

def Transform_Points(points,x,y,z,theta,x_post,y_post,z_post):
    
    tmp_points = [] 

    for point in points:
        point += [x,y,z]
        point = np.dot(Rz(theta), point)
        point += [x_post,y_post,z_post]
        tmp_points.append(point.flatten())
    
    tmp_points = np.array(tmp_points)
    
    return tmp_points

# Función para obtener los parámetros intrínsecos de la cámara

def calibrar():
    # Definición de variables
    objpoints_array = []
    imgpoints_array = []
    count = 0
    
    imgs_name = [f for f in os.listdir(URI) if f.endswith('.jpg')]  # Obtener nombres de archivos dentro de la carpeta
    
    for img_name in imgs_name:
        
        
        img = cv2.imread(URI+"/"+img_name)  # Cargar imagen
        
        if img is None:     # Comprobar que la imagen se ha podido leer
            print('Error al cargar la imagen')
            quit()

        img = cv2.resize(img, ESCALA)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(img_gray, GRID, None) # Encontrar esquinas del patrón de calibración
        
        if ret:     # Si se ha podido encontrar
            
            count+=1
            objpoints_array.append(OBJPOINTS)   # Añadir información neceria del punto para la calibración

            corners2 = cv2.cornerSubPix(img_gray, corners, (11,11), (-1,-1), CRITERIA)
            imgpoints_array.append(corners2)


    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints_array, imgpoints_array, img_gray.shape[::-1], None, None)  # Calibrar cámara
    
    print(f"Numero de imagenes para la calibracion: {count}")

    return mtx, dist

    # Función que proyecta los puntos de los nombres y dibuja lineas

def DibujarNombres(img, rvecs_img, tvecs_img, mtx, dist):   
    
    letra_B = copy.deepcopy(letras.B)   # Copiar puntos 3D de la letra
    
    letra_B = Transform_Points(letra_B,0,0,0,-math.pi/4,0,3*LONGITUD,0)     # Transformar las letras 

    letra_B = cv2.projectPoints(letra_B, rvecs_img, tvecs_img, mtx, dist)   # Obtener puntos 2D
    Pts = []

    for element in letra_B[0]:  # Pasarlos a tipo int
        
        Pts.append(element.ravel().astype(int))
    
    # Dibujar lineas

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

    mtx, dist = calibrar()  # Calibrar cámara para obtener parámetros intrínsecos 

    tiempos = []

    cap = cv2.VideoCapture(VIDEO)   # Obtener video

    inicio = time.time()        # Obtener tiempo inicial
    
    while(cap.isOpened()):      # Si se ha abierto correctamente el video
        
        ret, img = cap.read()   # Obtener frame
        print(img.shape)

        if ret:     
            
            img = cv2.resize(img, ESCALA)            
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            inicio_chessboard = time.time()
            ret, corners = cv2.findChessboardCorners(img_gray, GRID, None) # Encontrar esquinas del patrón de calibración
            final_chessboard = time.time()
            
            if ret:

                inicio_esquinas = time.time()
                corners2 = cv2.cornerSubPix(img_gray, corners, (11,11), (-1,-1), CRITERIA)  # Refinar esquinas del patrón de calibración
                
                img = cv2.drawChessboardCorners(img, GRID, corners2, ret)      # Dibujar esquinas refinadas
                final_esquinas = time.time()

                inicio_mtx = time.time()
                ret, rvecs_img, tvecs_img = cv2.solvePnP(OBJPOINTS, corners2, mtx, dist, True, cv2.SOLVEPNP_ITERATIVE)  # Obtener parámetros extrinsecos
                final_mtx = time.time()

                inicio_nombres = time.time()
                img = DibujarNombres(img, rvecs_img, tvecs_img, mtx, dist)      # Proyectar puntos de las letras
                final_nombres = time.time()

                cv2.imshow('frame',img)     # Mostrar imagen
                
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



        



        
