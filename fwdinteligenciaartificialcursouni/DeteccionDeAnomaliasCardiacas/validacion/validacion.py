import numpy as np
import matplotlib.pyplot as plt
import Image
import pickle

tipos = ['I','II','III','IV']

anomalias = pickle.load(open('../entrenamiento/anomalias','rb'))
W = pickle.load(open('../entrenamiento/W','rb'))
b = pickle.load(open('../entrenamiento/b','rb'))

M = len(W)

f  = {}
df = {}
f[0] = lambda n: 1/(1+np.exp(-n))
df[0] = lambda n: (1-n)*n
f[1] = lambda n: n
df[1] = lambda n : 1

temp = np.zeros([686,1])
#Cambiar el P[:,4]
temp[:,0] = anomalias[5]

a = temp
for m in range(M):
    n = W[m].dot(a)+b[m]
    a = f[m](n)
print(a)

out = 'Pulso Cardiaco de tipo ' + tipos[int(np.round(a[0]-1))] + ' '

Estados = {0:'Normal',1:'Anomalia 1',2:'Anomalia 2',3:'Anomalia 3'}

max = 1
for i in range(1,4):
    if a[i] > a[max]:
        max = i
temp = np.round(a[max])
if temp == 0 :
    max = 0
out += Estados[max]

print out
