#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Importar librerías

import cv2
import numpy as np
import os

# Variables globales

LONGITUD = 3.6
ESCALA_BORJA = (1280,720)
ESCALA_JUAN = (1200,675)

URI_BORJA = "Movil"
URI_JUAN = "Ordenador"
URI_PATTERNPEQ = "PatternPeq"

GRID_GRANDE = (9,6)
GRID_PEQ = (5,4)

OBJPOINTS_GRANDE = np.zeros((9*6,3),np.float32)
OBJPOINTS_GRANDE[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
OBJPOINTS_GRANDE = OBJPOINTS_GRANDE*LONGITUD

OBJPOINTS_PEQ = np.zeros((5*4,3),np.float32)
OBJPOINTS_PEQ[:,:2] = np.mgrid[0:5,0:4].T.reshape(-1,2)
OBJPOINTS_PEQ = OBJPOINTS_PEQ*LONGITUD



URI = URI_PATTERNPEQ
ESCALA = ESCALA_JUAN
OBJPOINTS = OBJPOINTS_PEQ
GRID = GRID_PEQ

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Función de calibración de cámara empleando n imágenes

def Calibrar(num_img):
    
    # Definición de variables
    count = 0
    objpoints_array = []
    imgpoints_array = []

    imgs_name = [f for f in os.listdir(URI) if f.endswith('.jpg')]  # Obtener nombres de archivos dentro de la carpeta

    for img_name in imgs_name:
        
        img = cv2.imread(URI+"/"+img_name)  # Cargar imagen
        
        if img is None: # Comprobar que la imagen se ha podido leer
            print('Error al cargar la imagen')
            quit()
        
        print(img.shape)
        img = cv2.resize(img, ESCALA)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(img_gray, GRID, None)     # Encontrar esquinas del patrón de calibración
        
        if ret:     # Si se ha podido encontrar
                        
            count += 1          # Añadir información neceria del punto para la calibración
            objpoints_array.append(OBJPOINTS)

            corners2 = cv2.cornerSubPix(img_gray, corners, (11,11), (-1,-1), CRITERIA)
            imgpoints_array.append(corners2)
        
        if count == num_img:    # Si se alcanzan el número de imágenes establecidos para la calibración
            break
            
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints_array, imgpoints_array, img_gray.shape[::-1], None, None)  # Calibrar la cámara
    print(ret)
    return mtx, dist    # Devolver parámetros intrínsecos


    
    # Calcular RMS del error de reproyección

def Comprobar_Error(mtx, dist):
    
    # Definir variables necesarias

    mean_error = 0
    count_imgs = 0
    imgs_name = [f for f in os.listdir(URI) if f.endswith('.jpg')]
    
    objpoints_array = []
    imgpoints_array = []


    for img_name in imgs_name:
        
        img = cv2.imread(URI+"/"+img_name)  # Obtener imagen
        
        if img is None:     # Comprobar que se ha cargado correctamente
            print('Error al cargar la imagen')
            quit()

        img = cv2.resize(img, ESCALA)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(img_gray, GRID, None) # Encontrar esquinas del patrón de calibración
        
        if ret:     # Si se ha encontrado el patrón
            
            count_imgs += 1
            corners2 = cv2.cornerSubPix(img_gray, corners, (11,11), (-1,-1), CRITERIA)
            
            ret, rvecs, tvecs = cv2.solvePnP(OBJPOINTS, corners2, mtx, dist, True, cv2.SOLVEPNP_ITERATIVE)  # Obtener parámetros extrínsecos
            
            imgpoints2, _ = cv2.projectPoints(OBJPOINTS, rvecs, tvecs, mtx, dist)   # Proyectar puntos conociendo todos los parámetros de la cámara
            
            error = cv2.norm(corners2, imgpoints2, cv2.NORM_L2)/len(imgpoints2)     # Obtener error entre las proyecciones y esquinas del patrón encontrados.
            mean_error += error
            
    mean_error = mean_error/count_imgs  # Calcular error medio
    
    return mean_error

def main():
    
    imgs_name = [f for f in os.listdir(URI) if f.endswith('.jpg')]


    for num_img_calib in range(1,len(imgs_name)+1): # Cada iteración se añade una imagen más a la calibración
        
        mtx, dist = Calibrar(num_img_calib)
        
        error = Comprobar_Error(mtx, dist)

        print(f"El error de calibracion con {num_img_calib} imagenes, es {error}")


if __name__ == "__main__":
    
    main()
