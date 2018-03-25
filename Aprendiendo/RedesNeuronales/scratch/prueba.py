import numpy as np

matriz = np.random.random([4,4])
print matriz

#imprime un vector igual a la primera columna
print matriz[:,0]

#imprime la columna 0
print matriz[:,:1]

#imprime toda la matriz
print matriz[:,:]

#imprime un vector igual a la ultima columna
print matriz[:,-1]
#imprime un vector igual a la penultima columna
print matriz[:,-2]

print '-'
#imprime toda la matriz , excepto la ultima
print matriz[:,:-1]

#imprime la matriz desde el ultimo para adelante
print matriz[:,-1:]

#imprime toda la matriz , excepto las dos ultima columnas
print matriz[:,:-2]

#imprime la matriz desde el penultimo elemento
print matriz[:,-2:]
