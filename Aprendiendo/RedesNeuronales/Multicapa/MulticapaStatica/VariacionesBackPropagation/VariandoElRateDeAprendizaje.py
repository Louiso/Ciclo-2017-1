import numpy as np
import matplotlib.pyplot as plt
from time import time


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

x = np.linspace(-10,10,100)
# print(x)
y = sigmoidea(x)
#
# plt.plot(x,y)
# plt.show()

################################################################################
#Tamano de patron entrada:
r = 1
#Tamano de patron salida:
s0 = 1

s  = {}
W  = {}
b  = {}
f  = {}
df = {}

#Variables cambiantes por cada iteracion
a  = {}
S  = {}
n  = {}

#La cantidad de CAPAS
M = 2

s[-1] = r
#Definiendo primera capa
s[0] = 30
W[0] = np.random.random([s[0],s[-1]])*(2)-1 #Ya esta transpuesto
b[0] = np.random.random([s[0],1])*(2)-1
# W[0] = np.array([
#     [-0.27],
#     [-0.41]
# ])
# b[0] = np.array([
#     [-0.48],
#     [-0.13]
# ])
f[0] = lambda n: 1/(1+np.exp(-n))
df[0] = lambda n: (1-n)*n
# f[0] = lambda n: n
# df[0] = lambda n : 1

#Definiendo la segunda capa
s[1] = 1
W[1] = np.random.random([s[1],s[0]])*2-1
b[1] = np.random.random([s[1],1])*2-1
# W[1] = np.array([
#     [0.09,-0.17]
# ])
# b[1] = np.array([[0.48]])
f[1] = lambda n: n
df[1] = lambda n : 1

# print(W[0])
# print(b[0])
# print(W[1])
# print(b[1])

#Problema : Crear una red neuronal capaz de aproximar la sgt funcion
#g(p) = 1 + sin(np.pi/4*p) ; -2<=p<=2
def g(p):
    return 1+ np.sin(np.pi/4*p)

#Generar P con 21 puntos
Q = 21
P = np.zeros([r,Q])
temp = np.linspace(-10,10,Q)
plt.figure()
plt.plot(temp,g(temp),"o")
for q in range(Q) :
    P[:,q] = temp[q]
# print(P)
# print(g(P))
#Inicio de entrenamiento de red Neuronal
Epocas = 10000
Emin = 0.001
PFiltro = 0.04
sigma = 0.1
gammadef = 0.2
gamma = gammadef
sigmaLess = 0.8
sigmaPlus = 1.2
converge = False
DJ = 0
epoca = 0
# epoca < Epocas and
iter = 1
DJA = []
tiempo_inicial = time()
while converge != True:
    q = 0
    Eq = 0
    SDW = {}
    DW = {}
    DSb = {}
    Db = {}
    for m in range(M):
        SDW[m] = 0
        DW[m] = 0
        Db[m] = 0
        DSb[m] = 0
    for q in range(Q):
        Pq = np.zeros([r,1])
        Pq[:,0] = P[:,q]
        #########################################################
        #Feed For Ward!!!
        a[-1] = Pq# Pq es una matriz, por lo tanto a[-1], tambn lo es
        for m in range(M):
            n[m] = W[m].dot(a[m-1])+b[m]
            a[m] = f[m](n[m])
        ##########################################################
        err = g(Pq) - a[M-1]
        #Solo se usa el error cuadratico
        Eq += err.T.dot(err)
        ##########################################################
        #Retropropation
        S[M-1] = -diag(n[M-1],df[M-1]).dot(err)
        for m in range(M-2,-1,-1):
            S[m] = diag(a[m],df[m]).dot(W[m+1].T).dot(S[m+1])
        ##########################################################
        #Se van acumulando los incrementos por cada prueba
        for m in range(M):
            DW[m] = gamma*DW[m] - (1-gamma)*sigma*S[m].dot(a[m-1].T)
            Db[m] = gamma*Db[m] - (1-gamma)*sigma*S[m]
            SDW[m] += DW[m]
            DSb[m] += Db[m]
    ##############################################################
    #Verificando si la red neuronal  necesita correccion
    DJ = Eq/Q
    print str(DJ[0,0])+"  "+str(sigma)
    if iter!=1:
        DJanterior = DJA[len(DJA)-1]
        DDJ = (DJ[0,0]-DJanterior)/DJanterior
        # print "*"*10
        # print "DDJ",DDJ
        # print "sigma",sigma
        if DDJ < -PFiltro: #Se redujo el error mas que el filtro
            sigma *= sigmaPlus
            gamma = gammadef
            # print "Error reducido mas que el pedido"
        elif DDJ < 0: #Se redujo el error considerablemente
            gamma = gammadef
            # print "Error reducido menos que el pedido"
        # elif DDJ < PFiltro: #Se incremento el error considerablemente
            # gamma = gammadef
            # print "Error incrementado menos que el pedido"
        elif DDJ > PFiltro: #if DDJ > PFiltro #El error se incremento demasiado
            sigma *= sigmaLess
            # print DDJ
            gamma = 0
            # print "Error incrementado mas que el pedido"
    DJA.append(DJ[0,0])
    if DJ < Emin:
        converge = True
        break
    #
    # Es necesario modificar el codigo segun los indices
    #
    #Actualizando con batching
    for m in range(M):
        W[m] = W[m] + SDW[m]/Q
        b[m] = b[m] + DSb[m]/Q
    epoca += 1
    iter += 1
tiempo_final = time()

print 'Despues de converger ... nos toca verificar'

print 'Capa 1 , neuronas 10'
print W[0]
print b[0]
print 'Capa 2 , neuronas 1'
print W[1]
print b[1]
a = np.array([
    [1]
])
print 'Despues de la capa 1'
n = W[0].dot(a)+b[0]
a = f[0](n)
print a

print 'Despues de la capa 2'
n = W[1].dot(a)+b[1]
a = f[1](n)
print a

Q = 100
P = np.zeros([r,Q])
temp = np.linspace(-10,10,Q)
Salida = []
# print temp
for q in range(Q) :
    P[:,q] = temp[q]
    a = np.zeros([1,1])
    a[:,0] = P[:,q]
    # print a
    # print g(a)
    for m in range(M):
        n = W[m].dot(a)+b[m]
        # print n
        a = f[m](n)
    Salida.append(a[0,0])
print temp
tiempo_ejecucion = tiempo_final-tiempo_inicial
print "El tiempo de ejecucion fue de : ", tiempo_ejecucion
plt.plot(temp,Salida)

plt.plot(temp,g(temp))

plt.figure()
x = np.arange(iter)
# print x
# print DJA
plt.plot(x,DJA)
plt.axis([0,iter,0,1])
plt.show()
