#Este algoritmo minimiza el error cuadratico mas no garantiza la clasificacion correcta
import numpy as np
import matplotlib.pyplot as plt

def signo(m):
    x = np.empty_like(m)
    x[m<0] = -1
    x[m>=0] = 1
    return x
#Para calcular los pesos sinapticos optimos debemos ver
#como minimizar la sgt funcion
# E (e^2) ; e = ( t - a )
# = > E (e^2) = E (( t - a )^2) ; a = W.P + b = x'.z ; x' = [w1 w2 ... wr b] && z = [p1 p2 ... pr 1]
# Luego reemplazando :
# E ((t-x'z)^2) = E (( t*t - 2tx'z + (x'z)(x'z))
# = > E (t*t) - 2*E(t*z)x + x'E(z'*z)x
# sea : E(t*t) = c ; constante
# sea : E(t*z) = h ;
# sea : E(z'z) = R ; este valor no cambia en caso que haya mas neuronas
# Entonces tenemos :
# E (e^2) = c - 2hx + x'Rx
# entonces calculando :
# dE(e^2)/dx = -2h + 2Rx => x = R^-1h

r = 2 #el numero de entradas
s = 2 #el numero de neuronas el cual en un monocapa es tambn el tamano de la salida
#Recordar que segun sea el tamano de salida podemos obtener 2^s clases posibles, en nuestro caso 4
k = 8 #el numero de pruebas

P = np.array([
    [0.7 ,  3],#Prueba 1
    [1.5 ,  5],#Prueba 2
    [2.0 ,  9],#Prueba 3
    [0.9 , 11],#Prueba 4
    [4.2 ,  0],#Prueba 5
    [2.2 ,  1],#Prueba 6
    [3.6 ,  7],#Prueba 7
    [4.5 ,  6] #Prueba 8
])
P = P.T
xtra = np.ones([1,k])
Pm = np.vstack((P,xtra))
# print(Pm)

#Calculando R = E(Pm*Pm.T)
R = Pm.dot(Pm.T)/k
# print(R)

T = np.array([
    [-1,-1],#Resultado de la primera prueba
    [-1,-1],#Resultado de la segunda prueba
    [-1, 1],
    [-1, 1],
    [ 1,-1],
    [ 1,-1],
    [ 1, 1],
    [ 1, 1]#Resultado de la ultima prueba
])
T = T.T
# print(T)

#Calculando H = E(Pm.T.T)
H = Pm.dot(T.T)/k
# print(H)

#Calculando W optimo

Woptimo = np.linalg.pinv(R).dot(H)
# print(Woptimo)

W = Woptimo[:r,:]
print(W)

b = Woptimo[r,:]
print(b)

def redAdaline(P):
    return signo(W.T.dot(P) + b)

for i in range(k):
    print(redAdaline(P[:,i]))
