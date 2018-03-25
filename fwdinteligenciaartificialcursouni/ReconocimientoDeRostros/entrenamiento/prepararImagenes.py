import numpy as np
import matplotlib.pyplot as plt
import Image
import pickle

imagenes = []

matrizImagenes = []

fotos = ['Alfredo','Bardo','China','Claudia','Don','Gerson','Melissa','Sid','Sofia']

def prepararFotos():
    cantFotos = len(fotos)
    for i in range(cantFotos):
        for j in range(4):
            indice = 4*i + j
            nombre = './imagenes/' + fotos[i] + str(j+1) + '.jpg'
            imagenes.append(Image.open(nombre).convert('L'))
            matrizImagenes.append(np.asarray(imagenes[indice],dtype=float))


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

def showFotos():
    cantFotos = len(fotos)
    for i in range(cantFotos):
        plt.figure()
        for j in range(4):
            indice = 4*i + j
            plt.subplot(2,2,j+1)
            plt.imshow(matrizImagenes[indice],cmap='gray',interpolation='nearest')
            plt.title(fotos[i] + str(j+1))
        plt.show()

def transformarFotos():
    cantFotos = len(fotos)
    for i in range(cantFotos):
        for j in range(4):
            indice = 4*i + j
            matrizImagenes[indice] = transformar(matrizImagenes[indice],40,30)

data = []
def toVector():
    cantFotos = len(fotos)
    for i in range(cantFotos):
        for j in range(4):
            indice = 4*i + j
            #transformar temp en un vector de 1200
            matriz = matrizImagenes[indice]
            vector = []
            for ii in range(40):
                if ii%2==0:
                    for jj in range(30):
                        vector.append(matriz[ii,jj])
                else:
                    for jj in range(29,-1,-1):
                        vector.append(matriz[ii,jj])
            data.append(vector)


prepararFotos()
transformarFotos()
toVector()
showFotos()
#Convirtiendo las matrizImagenes en vector

print len(data)

#Las imagenes ya estan listas para ser procesadas entonces ahora tengo que guardarlas
pickle.dump(matrizImagenes,open('matrizImagenes','wb'))
pickle.dump(data,open('matrizVectorImg','wb'))
pickle.dump(fotos,open('fotos','wb'))

# showFotos()
