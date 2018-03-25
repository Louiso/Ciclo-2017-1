import numpy as np
import matplotlib.pyplot as plt
#Una neurona se puede interpretar como una funcion
#que operera con las entradas y da una salida

def sigmoidea(m):
    return 1/(1+np.exp(-m))

def identidad(m):
    return m

def signo(m):
    x = np.empty_like(m)
    x[m<0] = -1
    x[m>=0] = 1
    return x
def H(m):
    x = np.empty_like(m)
    x[:] = m[:]
    x[m<0]=0
    x[m>=0]=1
    return x

def linealTramos(m):
    x = np.empty_like(m)
    x[m<-1] = -1
    x[(-1<=m)*(m<1)] = m[(-1<=m)*(m<1)]
    x[1<=m] = 1
    return x

def gaussiana(m):
    return np.exp(-m*m)

def sinusoidal(m):
    return np.sin(m)

def perceptron(x):
    xTam = 2
    yTam = 1
    if len(x) !=xTam:
        return None
    nNeuronas = 1
    # W = np.random.random([nNeuronas,xTam])
    W = np.array([
        [-1,-1]
    ])
    b = 0.5
    M = W.dot(x) + b
    N = H(M)
    # V = np.random.random([yTam,nNeuronas])
    # y = V.dot(N)
    return N

# x = np.random.random(2)
x = np.array([
    [1.5 ,-0.3],
    [0.9 , 0.05],
    [2.1 ,  0.2],
    [0.24,-0.87],
    [0.45,-0.60],
    [0.15,-0.43]
])

frutas = {0:'Pina',1:'Manzana'}
# print(frutas[1])

for i in range(6):
    y = perceptron(x[i])
    print(frutas[int(y)])

# plt.plot(x,perceptron(x))
# plt.show()


####################################
#Hacer una red neuronal que decida cuando un fruto es pina y cuando es manzana:
#Pina :  [1.5 ,-0.3] [0.9 , 0.05] [2.1 ,  0.2]
#Manza : [0.24,0.87] [0.45,-0.60] [0.15,-0.43]
#El primer dado es el peso del fruto, el segundo es el color del fruto
