import pickle

a = [1,2]
f = open('./dato','wb')

#Serializando la variable en el archivo f
pickle.dump(a,f)

f = open('./dato','rb')
a = pickle.load(f)

print a

#Mas sobre el manejo de archivo en python:
#Archivo de solo lectura:
archivo = open("dato", "r")
#Leer todo el contenido
contenido = archivo.read()
print 'contenido'
print contenido
#Leer una linea
archivo = open("dato", "r")
linea = archivo.readline()
print 'primera linea'
print linea
#Leer todas las lineas de un archivo
archivo = open("dato", "r")
lineas = archivo.readlines()
print 'lineas'
for linea in lineas:
    print linea
#Mover el puntero al final del archivo:
# archivo.seek(0)

#Para poder leer y escribir en el archivo se usa
archivo = open("dato", "r+")
print archivo.tell()
contenido = archivo.read()
print archivo.tell()
final_de_archivo = archivo.tell()

archivo.write('Nueva linea')
print archivo.tell()
archivo.seek(final_de_archivo)
archivo.write('te la creiste wex')
archivo.seek(0)
nuevo_contenido = archivo.read()

print nuevo_contenido

#Llenado datos
archivo = open("./remeras", "r+")
contenido = archivo.read()
final_de_archivo = archivo.tell()
lista = ['Linea 1\n', 'Linea 2']

archivo.writelines(lista)
archivo.seek(final_de_archivo)

print archivo.readline()
print archivo.readline()

archivo = open("remeras", "r")
contenido = archivo.read()
archivo.close()
print contenido
