import numpy as np
import matplotlib.pyplot as plt
import Image
import pickle

def transformar(matriz,newH,newW):
    (height,width) = matriz.shape
    print height , width
    escalaX = float(width)/newW
    escalaY = float(height)/newH
    print escalaX , escalaY
    temp = np.zeros([newH,newW],dtype = float)

    for i in  range(newH):
        for j in range(newW):
            ii = escalaY*i
            jj = escalaX*j
            mean = np.mean(matriz[ii:ii+escalaY,jj:jj+escalaX])
            if mean<132:
                mean = 0
            else :
                mean = 1
            temp[i,j] = mean
    return temp

fotos = pickle.load(open('../entrenamiento/fotos','rb'))
W = pickle.load(open('../entrenamiento/W','rb'))
b = pickle.load(open('../entrenamiento/b','rb'))

M = len(W)

f  = {}
df = {}
f[0] = lambda n: 1/(1+np.exp(-n))
df[0] = lambda n: (1-n)*n
f[1] = lambda n: n
df[1] = lambda n : 1

imgAlguien = Image.open('./pudge.jpg').convert('L')
Alguien = np.asarray(imgAlguien,dtype = float)

Alguien = transformar(Alguien,40,30)

#transformar temp en un vector de 1200
matriz = Alguien
vector = []
for ii in range(40):
    if ii%2==0:
        for jj in range(30):
            vector.append(matriz[ii,jj])
    else:
        for jj in range(29,-1,-1):
            vector.append(matriz[ii,jj])

Alguien = vector

print 'Despues de converger ... nos toca verificar'
temp = np.zeros([1200,1])
#Cambiar el P[:,4]
temp[:,0] = Alguien

a = temp
for m in range(M):
    n = W[m].dot(a)+b[m]
    a = f[m](n)
print(a)
cantFotos = len(fotos)
max = 0
for i in range(1,cantFotos):
    if a[max] < a[i] :
        max = i
print('Eres '+fotos[max])
