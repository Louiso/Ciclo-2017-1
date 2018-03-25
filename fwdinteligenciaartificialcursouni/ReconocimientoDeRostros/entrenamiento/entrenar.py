import numpy as np
import matplotlib.pyplot as plt
import pickle

matrizVectorImg = pickle.load(open('matrizVectorImg','rb'))
fotos = pickle.load(open('fotos','rb'))

def sigmoidea(n):
    return 1/(1+np.exp(-n))

def diag(x,f = None):
    n = len(x)
    temp = np.zeros([n,n])
    for i in range(n):
        if f!=None:
            temp[i,i] = f(x[i])
        else :
            temp[i,i] = x[i]
    return temp

################################################################################
#Tamano de patron entrada:
r = 1200
#Tamano de patron salida:
s0 = len(fotos)
#Sesgo , es la mitad del rango de una distribucion uniforme
sesgo = 0.001

s  = {}
W  = {}
b  = {}
f  = {}
df = {}

a  = {}
S  = {}
n  = {}

#La cantidad de CAPAS
M = 2

s[-1] = r #el tamano del patron de entrada
#Definiendo primera capa
s[0] = 2*len(fotos) #la cantidad de neuronas de las segunda capa
W[0] = np.random.random([s[0],s[-1]])*2*sesgo-sesgo #Ya esta transpuesto
b[0] = np.random.random([s[0],1])*2*sesgo-sesgo
f[0] = lambda n: 1/(1+np.exp(-n))
df[0] = lambda n: (1-n)*n

#Definiendo la segunda capa
s[1] = s0 #La cantidad de neuronas en la segunda capa, pero como solo tenemos dos capaz entonces tambn define la salida
W[1] = np.random.random([s[1],s[0]])*2*sesgo-sesgo
b[1] = np.random.random([s[1],1])*2*sesgo-sesgo
f[1] = lambda n: n
df[1] = lambda n : 1

# print(W[0])
# print(b[0])
# print(W[1])
# print(b[1])
#Generar 5 patrones de prueba
Q = len(fotos)*4
print Q
print len(matrizVectorImg)
P = np.zeros([r,Q])
#Cargar las caras en P
temp = matrizVectorImg
#
for q in range(Q) :
    P[:,q] = temp[q]
#Definiendo los valores de salida esperada
t = np.array([
    #0 1 2 3 4 5 6 7 8
    [1,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0],
    [0,0,0,0,1,0,0,0,0],
    [0,0,0,0,1,0,0,0,0],
    [0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,1],
])
t = t.T
# print t
#Inicio de entrenamiento de red Neuronal
Epocas = 1000
sigma = 0.1
converge = False
J = 0
epoca = 0
while epoca < Epocas and converge != True:
    s = 0
    print 'Epoca '+str(epoca)+' : '
    for q in range(Q):
        temp = np.zeros([r,1])
        temp[:,0] = P[:,q]
        #Calculando a, propagacion
        a[-1] = temp
        for m in range(M):
            n[m] = W[m].dot(a[m-1])+b[m]
            # print n[m]
            a[m] = f[m](n[m])
            # print a[m]
        temp = np.zeros([s0,1])
        temp[:,0] = t[:,q]
        err = temp - a[M-1]
        s += err.T.dot(err)
        #Calculando S, retropropagacion
        S[M-1] = -2*diag(n[M-1],df[M-1]).dot(err)
        for m in range(M-2,-1,-1):
            #normalmente seria :
            #S[m] = diag(n[m],df[m]).dot(W[m+1].T).dot(S[m+1])
            #pero cuando como f[m] es sigmoidea la diagonal trabaja con a[m]
            S[m] = diag(a[m],df[m]).dot(W[m+1].T).dot(S[m+1])
        for m in range(M):
            W[m] = W[m] - sigma*S[m].dot(a[m-1].T)
            b[m] = b[m] - sigma*S[m]
    J = s/2
    print(J)
    # if J < 0.01:
    #     convege = True
    #     break
    epoca += 1

pickle.dump(W,open('W','wb'))
pickle.dump(b,open('b','wb'))





# print 'Despues de converger ... nos toca verificar'
# temp = np.zeros([r,1])
# temp[:,0] = P[:,4]
# a = temp
# for m in range(M):
#     n = W[m].dot(a)+b[m]
#     a = f[m](n)
# print(a)
# cantFotos = len(fotos)
# max = 0
# for i in range(1,cantFotos):
#     if a[max] < a[i] :
#         max = i
# print('Eres '+fotos[max])
#Ejecutando redMulticapa
#sea p0 = [1]
# for q in range(Q):
#     temp = np.zeros([r,1])
#     temp[:,0] = P[:,q]
#     #Calculando a, propagacion
#     a = temp
#     for m in range(M):
#         n = W[m].dot(a)+b[m]
#         a = f[m](n)
#     print a
