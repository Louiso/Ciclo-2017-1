import numpy as np
import cv2
import cv2.cv as cv
#Creamos un objeto que maneje la camara
cap = cv2.VideoCapture(0)
cap.open(0)
#Repetimos hasta que se cierre el programa
while(True):
   #Capturamos la imagen actual
   ret, img = cap.read()
   #Si la captura es correcta entonces la mostramos
   if ret:
       cv2.imshow('img',img)
   #Eperamos que  el usuario presione alguna tecla, en este caso la  letra "q"
   # que nos servira para cerrar el programa.
   key=cv2.waitKey(1)
   if  key == ord('q'):
       break
#Al salir del bucle infinito cerramos la ventana y dejamos de utilizar la camara.
cap.release()
cv.DestroyWindow("img")
