import Image                      # funciones para cargar y manipular imagenes
import numpy as np                # funciones numericas (arrays, matrices, etc.)
import matplotlib.pyplot as plt   # funciones para representacion grafica

img = Image.open("./img.png")
# img.show()

print img.size, img.mode, img.format
# img.size = (1920,1200) RGBA PNG

imgGrisScale = img.convert('L')
# imgGrisScale.show()

#guardar una imagen nueva en disco:
imgGrisScale.save('./imgGris.png')

#convirtiendo la imagen en una matriz
a = np.asarray(imgGrisScale,dtype=float)

#convirtiendo de matriz a imagen y guardado
#primero se convierte la matriz a tipo uint8 y luego recien a imagen
Image.fromarray(a[100:200,100:200].astype(np.uint8)).save("prueba.png")

mainImg = a[:1000,100:1000+100]

width = 1000
height = 1000
prueba = a[:width,:height]

print(prueba)
#matriz de 10x10

#Objetivo ... reducir en una escala de 1/2
#Aprender como calcular el promedio de una matriz

escala = 15

prueba2 = np.zeros([width/escala,height/escala])

print(prueba2.shape)

for i in  range(width/escala):
    for j in range(width/escala):
        ii = escala*i
        jj = escala*j
        mean = np.mean(prueba[ii:ii+escala,jj:jj+escala])
        if mean<255/2:
            mean = 0
        else :
            mean = 255
        prueba2[i,j] = mean
# print(prueba2)

plt.figure()
# plt.subplot(121)
# plt.imshow(a,cmap='gray',interpolation='nearest')
# plt.title('Rem')
# plt.subplot(122)
# plt.imshow(mainImg,cmap='gray',interpolation='nearest')
# plt.title('Lo que sea')
plt.subplot(121)
plt.imshow(prueba,cmap='gray',interpolation='nearest')
plt.title('Prueba')
plt.subplot(122)
plt.imshow(prueba2,cmap='gray',interpolation='nearest')
plt.title('Prueba2')
plt.show()





#
