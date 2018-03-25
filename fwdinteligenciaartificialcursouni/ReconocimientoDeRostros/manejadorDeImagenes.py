import Image                      # funciones para cargar y manipular imagenes
import numpy as np                # funciones numericas (arrays, matrices, etc.)
import matplotlib.pyplot as plt   # funciones para representacion grafica
import pickle
img = []
a = []
for i in range(5):
    nombre = "./cara"+str(i+1)+".jpg"
    img.append(Image.open(nombre).convert('L'))
    a.append(np.asarray(img[i],dtype=float))

def transformar(matriz,newH,newW):
    (height,width) = matriz.shape
    escalaX = width/newW
    escalaY = height/newH

    temp = np.zeros([newH,newW],dtype = float)

    for i in  range(newH):
        for j in range(newW):
            ii = escalaY*i
            jj = escalaX*j
            mean = np.mean(matriz[ii:ii+escalaY,jj:jj+escalaX])
            if mean<145:
                mean = 0
            else :
                mean = 1
            temp[i,j] = mean
    return temp


def limLeft(temp):
    (h,w) = temp.shape
    for jk in range(w):
        for ik in range(h/2):
            if temp[ik,jk]<255:
                return jk
    return None
def limRigth(temp):
    (h,w) = temp.shape
    for jk in range(w-1,-1,-1):
        for ik in range(h/2):
            if temp[ik,jk]<255:
                return jk
    return None
def limBottom(temp):
    (h,w) = temp.shape
    for ik in range(h-1,-1,-1):
        if temp[ik,0]==255:
            return ik
    return None
#Aqui encuadrare las caras
plt.figure()
for i in range(5):
    plt.subplot(2,3,i+1)
    temp = a[i]
    (h,w)=temp.shape
    ymin = 0
    ymax = h
    xmin = 0
    xmax = w
    while np.mean(temp[ymin,:])==255:
        ymin = ymin + 1

    xmin = limLeft(temp)
    xmax = limRigth(temp)
    ymax = limBottom(temp[:,xmin:xmax])

    a[i] = temp[ymin:ymax,xmin:xmax]
    plt.imshow(a[i],cmap = 'gray',interpolation='nearest')
    plt.title("Cara"+str(i))
# plt.show()

def mostrarFiguras():
    plt.figure()
    for i in range(5):
        temp = transformar(a[i],40,30)
        # print(temp)
        plt.subplot(2,3,i+1)
        plt.imshow(temp,cmap = 'gray',interpolation='nearest')
        plt.title("Cara"+str(i))
    plt.show()
# mostrarFiguras()
data = []
for i in range(5):
    temp = transformar(a[i],40,30)
    #transformar temp en un vector de 1200
    temp2 = []
    for i in range(40):
        if i%2==0:
            for j in range(30):
                temp2.append(temp[i,j])
        else:
            for j in range(29,-1,-1):
                temp2.append(temp[i,j])
    data.append(temp2)

pickle.dump(data,open('caras','wb'))
