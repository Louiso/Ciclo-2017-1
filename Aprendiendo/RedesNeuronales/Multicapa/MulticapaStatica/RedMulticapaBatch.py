import numpy as np
import matplotlib.pyplot as plt

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
Q = 100
P = np.zeros([r,Q])
temp = np.linspace(-10,10,Q)
for q in range(Q) :
    P[:,q] = temp[q]
# print(P)
# print(g(P))


#Funcionamiento de una red neuronal statica

# PatronEntrada_(k) => |sistema| => Salida_(k+1)

#Funcionamiento de una red neuronal dinamica

# LoQueQuieroHacerAhora_(k)         => |sistema| =>   estadoNuevo_(k+1)
# estadoActual_(k)                                         |
#   <-------------------------------------------------------

#Cosas a tomar en cuenta :
# P0 = np.vstack((u,x0))

#Al momento de entrenar tener en cuenta :
# W[m]ij = W[m]ij - sigma*dTJ/dTW[m]i,j

# ST[0]="Hola"

#Inicio de entrenamiento de red Neuronal
Epocas = 2000
sigma = 0.01
converge = False
J = 0
epoca = 0
while epoca < Epocas and converge != True:
    q = 0
    Eq = 0
    DWEq = {}
    DbEq = {}
    for m in range(M):
        DWEq[m] = 0
        DbEq[m] = 0
    for q in range(Q):
        temp = np.zeros([r,1])
        temp[:,0] = P[:,q]
        #Calculando a, propagacion
        a[-1] = temp
        for m in range(M):
            n[m] = W[m].dot(a[m-1])+b[m]
            a[m] = f[m](n[m])
        err = g(temp) - a[M-1]
        Eq += err.T.dot(err)
        #Calculando S, retropropagacion
        S[M-1] = -2*diag(n[M-1],df[M-1]).dot(err)
        for m in range(M-2,-1,-1):
            S[m] = diag(a[m],df[m]).dot(W[m+1].T).dot(S[m+1])
        for m in range(M):
            DWEq[m] += S[m].dot(a[m-1].T)
            DbEq[m] += S[m]
    for m in range(M):
        W[m] = W[m] - sigma*(DWEq[m]/Q)
        b[m] = b[m] - sigma*(DbEq[m]/Q)
    J = Eq/Q
    print(J)
    # if J < 0.1:
    #     convege = True
    #     break
    epoca += 1

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
plt.plot(temp,Salida)
plt.show()

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
