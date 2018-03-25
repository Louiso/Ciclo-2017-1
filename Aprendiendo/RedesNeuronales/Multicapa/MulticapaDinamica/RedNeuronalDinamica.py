import numpy as np
import matplotlib.pyplot as plt
import pickle

from Funciones import *

'''Generando datos
Del array u solo tamare los valores de 1 hasta n-1'''
'''Simula el movimiento MRU de un objeto, desde 0 a 1 segundos con intervalos de 0.2'''
dt = 0.05;
t = np.linspace(0,2,40);
V = np.random.random([len(t)]);
X = np.zeros([len(t)]);
for i in range(1,len(t)):
    X[i] = X[i-1] + V[i]*dt;

bias = True

r = [1,30,1]

'''Inicializando'''
print r
# El numero  de capas, donde:
# la primera capa es 0
# la ultima capa M-1
M = len(r)

'''Inicializamos :
    -   los pesos sinapticos
    -   las funciones de activacion
    -   las derivadas de las funciones de activacion'''
# Con los tamanos ya especificados en cada capa
WB = {}; f = {}; df = {};
# Recordar que las capas van de 1 a M-1, es decir la ultima capa tiene indice M-1
WB[1] = 0.01*np.random.random([r[1],r[0]+r[M-1]+1]);
f[1] = sigmoidea;
df[1] = lambda a: (1-a)*a;
for m in range(2,M):
    WB[m] = 0.01*np.random.random([r[m],r[m-1]+1]);
    f[m] = sigmoidea if m < M-1 else lambda n: n;
    df[m] = lambda n: (1-n)*n if m < M-1 else lambda n:1;

fr = open("data.pkl","rb");
WB = pickle.load(fr);
fr.close();
'''Variables auxiliares'''
a = {};S = {}; n = {};

# Como los valores de entrada V, son de 1, y las salidas son de 1
# Cantidad de valores de entrenamiento
Q = len(t);

# Variables con las que se entrenara
Patron_X = np.zeros([r[0],Q]);
Patron_X[:,:] = V[:]/max(V);
Patron_Y = np.zeros([r[M-1],Q]);
Patron_Y[:,0] = X[0]/max(X);

# Valores con los que se evaluara
T = np.zeros([r[M-1],Q]);
T[:,:] = X[:]/max(X);
#print Patron_X
#print Patron_Y
#print T
'''Probando el feedforward'''
## Probamos como deberia correr el feedforward de la red neuronal dinamica
# Lo cual no funciona como en la estatica, para tener la respuesta de todos,
# cada valor que sale debe ser usado por el que sigue
for q in range(1,Q):
    x = Patron_X[:,q:q+1];
    y = Patron_Y[:,q-1:q];
    a[0] = np.vstack((x,y));
    for m in range(1,M):
        a[m-1] = np.vstack((a[m-1],np.array([[1]])));
        n[m] = WB[m].dot(a[m-1]);
        a[m] = f[m](n[m]);
    Patron_Y[:,q:q+1] = a[M-1]
#print Patron_Y
###################################################################################

J       = 0;
eta     = 0.0001;

error_max = 10;
error_max = float(error_max/100);
Epocas  = 1000;
Jold = 10000;
epoca   = 0;
dJ_rel = 1000;

# Si el numero de epocas supera el limite , sale o si dJ_rel es menor que el esperado
# sale
while epoca <= Epocas:
#while 1:
    J = 0;
    '''Inicializando los dJdWBT'''
    dJdWBT = {};
    dJdWBT[1] = np.zeros([r[1],r[0]+r[M-1]+1]);
    for m in range(2,M):
        dJdWBT[m] = np.zeros([r[m],r[m-1]+1]);
    
    '''Inicializacion de los dydWBT[i][m]'''
    dydWBT = {};
    for i in range(r[M-1]):
        dydWBT[i] = {};
        dydWBT[i][1] = np.zeros([r[1],r[0]+r[M-1]+1]);
        for m in range(2,M):
            dydWBT[i][m] = np.zeros([r[m],r[m-1]+1]);
    for q in range(1,Q):
        '''Dentro de una iteracion k'''
        x = Patron_X[:,q:q+1];
        y = Patron_Y[:,q-1:q];
        a[0] = np.vstack((x,y));
        for m in range(1,M):
            a[m-1] = np.vstack((a[m-1],np.array([[1]])));
            n[m] = WB[m].dot(a[m-1]);
            a[m] = f[m](n[m]);
        Patron_Y[:,q:q+1] = a[M-1]
        '''Calcular el J_q'''
        err = T[:,q] - Patron_Y[:,q];
        J_q = err.T.dot(err); 
        J = J + J_q;
        '''Calcular el Jacobiano'''
        Jacobiano = WB[M-1][:,:-1];
        # Si el numero de capaz de la red es menor o igual a 2 no entra a este for
        for m in range(M-2,1,-1): 
            F = diag(a[m][:-1],df[m]);
            Jacobiano = Jacobiano.dot(F).dot(WB[m][:,:-1]);
        F = diag(a[1][:-1],df[1]);
        Jacobiano = Jacobiano.dot(F).dot(WB[1][:,r[0]:r[0]+r[M-1]]);
        '''Calcular las sensibilidades'''
        S = {}
        for i in range(r[M-1]):
            S[i] = {};            
            S[i][M-1] = np.zeros([r[M-1],1]);
            S[i][M-1][i,0] = 1;
            for m in range(M-2,0,-1):
                F = diag(a[m][:-1],df[m]);
                S[i][m] = F.dot(WB[m+1][:,:-1].T).dot(S[i][m+1]);
        
        '''Calculando los dydWBT de la iteracion actual'''
        temp = {};
        for i in range(r[M-1]):
            temp[i] = {};
        for m in range(1,M):
            for i in range(r[M-1]):
                dydWB_i_m = S[i][m].dot(a[m-1].T);
                temp[i][m] = dydWB_i_m[:,:]
                for j in range(r[M-1]):
                    temp[i][m] = temp[i][m] + Jacobiano[i,j]*(dydWBT[j][m]);
        
        for m in range(1,M):
            for i in range(r[M-1]):
                dydWBT[i][m] = temp[i][m];
        
        '''Inicializando los dJqdWBT, estos son temporales por iteracion'''
        dJqdWBT = {};
        dJqdWBT[1] = np.zeros([r[1],r[0]+r[M-1]+1]);
        for m in range(2,M):
            dJqdWBT[m] = np.zeros([r[m],r[m-1]+1]);
        '''Calculando los dJqdWBT'''
        for m in range(1,M):
            for i in range(r[M-1]):
                dJqdWBT[m] = dJqdWBT[m] -2*err[i]*dydWBT[i][m];
            dJdWBT[m] = dJdWBT[m] + dJqdWBT[m];
    #end bucle para los q

    dJ_rel = np.sqrt(abs(J-Jold))/Jold;
    #print dJ_rel;
    print J
    #if dJ_rel <= error_max:
    #    break;
    for m in range(1,M):
        dJdWBT[m] = dJdWBT[m]/(Q-1);
        WB[m] = WB[m] - eta*dJdWBT[m];
    Jold = J;
    epoca = epoca + 1;

# De esta forma representamos a 'V' print WB[1][index_output:index_output+n_output,:]
'''Validando'''
for q in range(1,Q):
    x = Patron_X[:,q:q+1];
    y = Patron_Y[:,q-1:q];
    a[0] = np.vstack((x,y));
    for m in range(1,M):
        a[m-1] = np.vstack((a[m-1],np.array([[1]])));
        n[m] = WB[m].dot(a[m-1]);
        a[m] = f[m](n[m]);
    Patron_Y[:,q:q+1] = a[M-1]


plt.plot(t,X,'o');
plt.plot(t,Patron_Y[0,:]*max(X));
plt.show();

fo = open("data.pkl","wb");
pickle.dump(WB,fo);
fo.close();
#ftest = open("prueba.pkl","rb");
#test = pickle.load(ftest);
#print test;
#ftest.close();
#ftest = open("prueba.pkl","wb");
#test = "Hola";
#print test
#pickle.dump(test,ftest);
#ftest.close();
