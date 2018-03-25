import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# X is the 10x10 Hilbert matrix
_1_10 = np.arange(1,11)
_0_9  = np.arange(0,10)[:,np.newaxis]
X = 1. / ( _1_10 + _0_9)
#Imprime una matriz de 10x10
#print _1_10 + _0_9

#y es un vector de 10 unos
y = np.ones(10)

n_alphas = 200

#alphas es un vector de : 10^-10 hasta 10^-2
alphas = np.logspace(-10, -2, n_alphas)

clf = linear_model.Ridge(fit_intercept=False)
#Es un objeto que permite el calculo de regresion de cresta , es decir no lineales

coefs = []
for a in alphas:
    clf.set_params(alpha=a)
    #Se entra al objeto con los datos X, y
    clf.fit(X, y)
    coefs.append(clf.coef_)

ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()
