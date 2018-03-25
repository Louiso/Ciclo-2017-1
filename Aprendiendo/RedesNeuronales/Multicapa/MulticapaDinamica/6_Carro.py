import numpy as np
from numpy import *;
import pygame
from pygame.locals import *
from Funciones import *;
import pickle
from OpenGL.GL import *
from OpenGL.GLU import *

r = [2,10,10,1];

M = len(r);
(WB,f,df) = ini_Pesos_Funciones_Derivadas(r,M);
fr = open("carro.pkl","rb");
WB = pickle.load(fr);
fr.close();

vel = 0.1;
L = 2.0;
Z_Obj = array([
    [0],
    [pi/2]
]);

theta = -pi/2;
x = -25;
y = -5;
#print z

clock = pygame.time.Clock()

def drawCar(pos,theta):
    w = 1.0;
    l = 2.0;
    color = array([1.0,1.0,1.0]);
    amarillo = array([203,203,73],dtype = float)/255;
    gris = array([203,203,203],dtype = float)/255;
    glBegin(GL_QUADS);
    temPos = pos + w*array([-sin(theta),cos(theta),0])/2;
    glColor3fv(gris);
    glVertex3fv(temPos);
    temPos = pos + w*array([-np.sin(theta),np.cos(theta),0])/2 + l*array([cos(theta),sin(theta),0]);        
    glColor3fv(amarillo);
    glVertex3fv(temPos);
    temPos = pos - w*array([-np.sin(theta),np.cos(theta),0])/2 + l*array([cos(theta),sin(theta),0]);        
    glColor3fv(amarillo);    
    glVertex3fv(temPos); 
    glColor3fv(gris);
    temPos = pos - w*array([-np.sin(theta),np.cos(theta),0])/2;    
    glVertex3fv(temPos);
    glEnd();    
#iniciamos pygame
pygame.init()
#Seteamos el tamano de la pantalla
display = (800,800)
pygame.display.set_mode(display,DOUBLEBUF | OPENGL)

#La camara la trasladamos 5 hacia atras para poder ver el cubo
gluPerspective(45,1,0.1,80.0);
glTranslatef(0.0,0.0,-70.0)
#No rotamos la camara
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
    z = array([
        [x],
        [theta]
    ]);
    a= z - Z_Obj;
    for m in range(1,M):
        a = np.vstack((a,np.array([[1]])));
        n = WB[m].dot(a);
        a = f[m](n);
    u = a
    x = x + vel*np.cos(theta);
    y = y + vel*np.sin(theta);
    theta = theta - vel/L*u[0,0];
    pos = [x,y,0];
    #Limpia la pantalla GL_COLOR_BUFFER_BIT
    #Activa el bit de profundidad
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    print 'Pos : ',pos
    print 'Theta: ',theta
    #Dibujamos respecto a los nuevos datos cargados
    drawCar(pos,theta);
    pygame.display.flip();
    #Cada 1000 es un segundo
    pygame.time.wait(10)