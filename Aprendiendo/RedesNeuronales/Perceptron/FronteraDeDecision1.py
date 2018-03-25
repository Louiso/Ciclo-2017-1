import numpy as np
import matplotlib.pyplot as plt

def escalon(m):
    x = np.empty_like(m)
    x[:] = m[:]
    x[m<0]=0
    x[m>=0]=1
    return x

#Dibujando los ejes
def drawEjes(I):
    #Eje X
    x = I
    y = I*0
    plt.plot(x,y,'g')
    #Eje Y
    x = I*0
    y = I
    plt.plot(x,y,'g')


def drawFrontera(W,b):
    x = np.linspace(-5,5,100)
    y = (-x*W[0,0] - b[0,0])/W[0,1]
    plt.plot(x,y)

# W = np.array([
#     [ 1.0 , -0.8 ]
# ])
#
# b = 0

def percentron(P):
    (rw,cw) = W.shape
    a = np.zeros([rw,1])
    a = escalon(W.dot(P)+b)
    return a

def drawSalida(P,t):
    x = P[0]
    y = P[1]
    salida = percentron(P)
    if t == 0:
        logo = 'ro'
    else :
        logo = 'bo'
    plt.plot(x,y,logo)
    # print salida

I = np.linspace(-10,10,100)
drawEjes(I)

# W = np.array([
#     [ 1.0 , -0.8 ]
# ])

#Como busco crear una frontera de decision , entonces solo necesitare 1 neurona
#Como el problema se hara en el plano entonces r = 2
W = np.random.random([1,2])

# b = np.array([
#     [ 0.0 ]
# ])

b = np.random.random([1,1])

plt.title('Frontera sin entrenar')
drawFrontera(W,b)
plt.show()
Q = 8
P = np.array([
    [ 0, 2],
    [ 2, 2],
    [ 2, 0],
    [ 2,-2],
    [ 0,-2],
    [-2,-2],
    [-2, 0],
    [-2, 2]
])
P = P.T

T = np.array([
    [1,0,0,0,0,1,1,1]
])
#Dibujar puntos
def dibujarPuntos():
    for q in range(Q):
        drawSalida(P[:,q],T[:,q])

#entrenamiento
Epocas = 3

time = {}
time[0] = []
time[1] = []
print 'entrenamiento'
for epoca in range(Epocas):
    s = 0
    for q in range(Q):
        #P lo estoy pasando como si fuese matriz
        a = percentron(P[:,q:q+1])
        error = T[:,q:q+1]-a
        W += error*P[:,q:q+1].T
        b += error
        s += error
        drawEjes(I)
        dibujarPuntos()
        drawFrontera(W,b)
        plt.show()
    time[0].append(epoca)
    time[1].append(s[0,0])

drawEjes(I)
plt.plot(time[0],time[1])
plt.show()
