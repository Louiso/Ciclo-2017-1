import numpy as np                # funciones numericas (arrays, matrices, etc.)
import matplotlib.pyplot as plt   # funciones para representacion grafica
import pickle

matrizImagenes = pickle.load(open('matrizImagenes','rb'))
fotos = pickle.load(open('fotos','rb'))

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

showFotos()
