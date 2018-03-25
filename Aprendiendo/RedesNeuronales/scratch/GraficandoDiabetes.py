import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# Cargando diabetes
diabetes = datasets.load_diabetes()

# Usando una caracteristica de esta colleccion
# diabetes es una biblioteca con campos:
# data , target
# Data es una matriz de  442x10
# Target es un vector de 442 elementos

#diabetes_x es una matriz de 442x1 , el cual toma la columna con indice 2
diabetes_X = diabetes.data[:, np.newaxis , 2]
# print diabetes.data[:,2:3]
# print diabetes_X
# Split the data into training/testing sets

#Toma todos los datos , excepto los ultimos 20
diabetes_X_train = diabetes_X[:-20]

#Toma todos lo datos , a partir de los ultimos 20 datos
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

#Con esta funcion entrenamos al objeto regr
# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

#Calcular el coeficiente de la regresion lineal, calcula por minimos cuadrados la pendiente de los datos
#por minimos cuadrados
# The coefficients
print('Coefficients: \n', regr.coef_)

#Calcular el promedio del error al cuadrado, es decir me indica por cuanto falla la recta respecto a los datos en general
# The mean squared error
print("Mean squared error: %.2f" % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

# Plot outputs
#Distribuye los datos de prueba y lo coloca en color negro
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')

#Plotea la recta calcula por la funcion regr
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',linewidth=3)

#Hace q los ejes , no se muestren
# plt.xticks(())
# plt.yticks(())

plt.show()
