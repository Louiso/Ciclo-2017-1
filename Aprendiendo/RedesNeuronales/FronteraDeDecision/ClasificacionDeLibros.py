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
def monoCapa(x):
    xTam = 2
    if len(x) !=xTam:
        return None
    # W = np.random.random([nNeuronas,xTam])
    W = np.array([
        [5,-1],
        [2, 1]
    ])
    b = np.array([-5,-10])
    M = W.dot(x) + b
    N = H(M)
    # V = np.random.random([yTam,nNeuronas])
    # y = V.dot(N)
    return N
#Casos posibles:
# 00 1 ligeros y usados
# 01 2 ligeros y no usados
# 10 3 pesados y usados
# 11 4 pesados y no usados
# x = np.random.random(2)
# x = np.array([
#     [0.7 ,    3],#Ligeros y poco usados
#     [1.5 ,    5],
#     [2.0 ,    9],#Ligeros y muy usados
#     [0.9 ,   11],
#     [4.2 ,    0],#Pesados y poco usados
#     [2.2 ,    1],
#     [3.6 ,    7],#Pesados y muy usados
#     [4.5 ,    6]
# ])
x = np.random.random([10,2])*(10)
# print(x)
for i in range(10):
    y = monoCapa(x[i])
    # print(y)
    out = ''
    linestyle = ''
    if y[0] == 0 :
        out += 'Ligero y'
        linestyle += 'c'
    else :
        out += 'Pesado y'
        linestyle += 'b'
    if y[1] == 0:
        out += ' usados'
        linestyle += 'd'
    else :
        out += ' no usados'
        linestyle += 'v'
    plt.plot(x[i,0],x[i,1],linestyle)
    print(out)

#Graficar las fronteras de decision
# plt.plot(x[:,0],x[:,1],'*')

plt.show()


####################################
#Hacer una red neuronal que decida cuando un fruto es pina y cuando es manzana:
#Pina :  [1.5 ,-0.3] [0.9 , 0.05] [2.1 ,  0.2]
#Manza : [0.24,0.87] [0.45,-0.60] [0.15,-0.43]
#El primer dado es el peso del fruto, el segundo es el color del fruto
