#Encuentra una solucion al problema , pero no la mejor solucion
import numpy as np
import matplotlib.pyplot as plt
#Una neurona se puede interpretar como una funcion
#que operera con las entradas y da una salida
def H(m):
    x = np.empty_like(m)
    x[:] = m[:]
    x[m<0]=0
    x[m>=0]=1
    return x

#Reconocimiento de numeros en un display
# numero de entradas 7 : a,b,c,d,e,f,g,h
#        a
#      -----
#    g|     |b
#     |  h  |
#      -----
#    f|     |c
#     |     |
#      -----
#        d
# Interpretacion :
# 0 : [1,1,1,1,1,1,0]
# 1 : [0,1,1,0,0,0,0]
# 2 : [1,1,0,1,1,0,1]
# 3 : [1,1,1,1,0,0,1]
# 4 : [0,1,1,0,0,1,1]
# 5 : [1,0,1,1,0,1,0]
# 6 : [1,0,1,1,1,1,1]
# 7 : [1,1,1,0,0,0,0]
# 8 : [1,1,1,1,1,1,1]
# 9 : [1,1,1,1,0,1,1]
#Entonces defino mi patron de prueba
P = np.array([
    [1,1,1,1,1,1,0],
    [0,1,1,0,0,0,0],
    [1,1,0,1,1,0,1],
    [1,1,1,1,0,0,1],
    [0,1,1,0,0,1,1],
    [1,0,1,1,0,1,0],
    [1,0,1,1,1,1,1],
    [1,1,1,0,0,0,0],
    [1,1,1,1,1,1,1],
    [1,1,1,1,0,1,1]
])
#Sabemos que habra una multiplicacion por la matriz W de dimension salidasxentradas
#Sabemos que P es de dimension 10xentradas, por lo cual debemos tomar a p como la transpuesta de esta misma
P = P.T
#Clasificando si es un numero par o no
# entonces las respuestas deberias ser :  [0,1,0,1,0,1,0,1,0,1]
esPar = np.array([0,1,0,1,0,1,0,1,0,1])
#Clasificando si un numero es mayor a 5 :
# entonces las respuestas  deberian ser : [0,0,0,0,0,0,1,1,1,1]
esMayor5 = np.array([0,0,0,0,0,0,1,1,1,1])
#Clasificando si un numero es primo :
# entonce las respuestas deberian ser :   [0,0,1,1,0,1,0,1,0,0]
esPrimo = np.array([0,0,1,1,0,1,0,1,0,0])

Prueba = esPrimo
#Inicializando los pesos sinapticos, tendre 7 entradas y quiero generar una salida
#entonces debe ser de 1x7
W = np.random.random([1,7])*2-1
#Polarizacion, es mas o menos cuanto se desplaza mi frontera, en este caso tendre una salida
# b debe ser de 1
b = np.random.random(1)*2-1
e = np.random.random(10)

print("W ... antes del entrenamiento : ")
print(W)

#Fase de entrenamiento:

#Como el perceptron solo consiste en neuronas de entrada y salida
#Su estructura es muy simple :
# y = f(W.x+b)
# P = es una matriz que guarda las entradas
# Prueba = es un vector que guarda los resultados esperados
for Epocas in range(2):
    for numero in range(10):
        #Como W no tiene los pesos bn calibrados , entonces la salida calculada
        #tendra error entonces pasamos a pasar
        e[numero] = Prueba[numero] - H(W.dot(P[:,numero])+b)
        print("Error : ")
        print(e[numero])
        print("Correccion : ")
        print(e[numero]*P[:,numero])
        W = W + e[numero]*P[:,numero]
        b = b + e[numero]
        print("W corregido : ")
        print(W)
        print(b)
        print("---------------")

print(e)
print(W)
print(b)

def perceptron(x):
    M = W.dot(x) + b
    N = H(M)
    return N

x = [1,1,1,1,0,1,1]

y = perceptron(x)

if y == 1:
    print('es Primo')
else :
    print('no es Primo')














#
