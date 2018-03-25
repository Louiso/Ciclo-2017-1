import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import pickle

from Funciones import *

bias = True;
r = [2,10,10,1];
M = len(r);
Q =  1000; #Para el problema del carro, el numero de iteraciones es libre;

'''Inicializamos :
    -   los pesos sinapticos
    -   las funciones de activacion
    -   las derivadas de las funciones de activacion'''
(WB,f,df) = ini_Pesos_Funciones_Derivadas(r,M);
fr = open("carro.pkl","rb");
WB = pickle.load(fr);
fr.close();
# Recordar que las capas van de 1 a M-1, es decir la ultima capa tiene indice M-1
'''establecemos como lo entrenaremos la red neuronal dinamica'''
Z_Obj = np.array([
    [0],
    [pi/2]
]);

radio = 11;
Z_Train = np.array([
    [-radio  ,-radio   ,radio   ,radio    ],
    [ pi/2   ,-pi/2    ,pi/2    ,-pi/2    ]
]);
vel = 0.1;
L = 2.0;
Patron_Z = np.zeros([r[0],Q]);
Patron_Z[:,0] = Z_Train[:,0];

#############################################################

eta     = 0.01;

error_max = 10;
error_max = float(error_max/100);
Epocas  = 100;
Jold = 10000;
epoca   = 0;
dJ_rel = 1000;

PLOT = {};

for epoca in range(Epocas):
    J = 0;
    dJdWBT = ini_dJdWBT(r,M);
        
    dudWBT = ini_dudWBT(r,M);
    dzdWBT = ini_dzdWBT(r,M);
    for t in range(4):
        Patron_Z[:,0] = Z_Train[:,t];
        for q in range(1,Q):
            a = {};n = {};
            '''Dentro de una iteracion k'''
            z = Patron_Z[:,q-1:q];
            #print z
            a[0] = z - Z_Obj;
            for m in range(1,M):
                a[m-1] = np.vstack((a[m-1],np.array([[1]])));
                n[m] = WB[m].dot(a[m-1]);
                a[m] = f[m](n[m]);
            u = a[M-1]
            Patron_Z[0,q] = Patron_Z[0,q-1] + vel*np.cos(Patron_Z[1,q-1]);
            Patron_Z[1,q] = Patron_Z[1,q-1] - vel/L*u;
            x = Patron_Z[0,q];
            theta = Patron_Z[1,q];
            '''Calcular el J_q'''
            err = Z_Obj - Patron_Z[:,q:q+1];
            J_q = err.T.dot(err); 
            J = J + J_q;
            Jacobiano = ini_Jacobiano(WB,df,a,r,M);
            S = ini_Sensibilidades(WB,df,a,r,M);
            '''Dentro de la red neuronal'''
            update_dudWBT(dudWBT,dzdWBT,S,Jacobiano,a,r,M);
            '''Dentro del controlador'''
            update_dJdWBT(dJdWBT,dzdWBT,dudWBT,err,r,M,vel,L,Patron_Z,q);
        PLOT[t] = Patron_Z.copy();
    dJ_rel = np.sqrt(abs(J-Jold))/Jold;
    #print dJ_rel;
    print dJ_rel*100, J
    #if dJ_rel <= error_max:
    #    break;
    for m in range(1,M):
        dJdWBT[m] = dJdWBT[m]/(4*(Q-1));
        WB[m] = WB[m] - eta*dJdWBT[m];
    Jold = J;

fo = open("carro.pkl","wb");
pickle.dump(WB,fo);
fo.close();

limit = 150;

for t in range(4):
    plt.figure(t+1);
    plt.plot(PLOT[t][1,:limit]);
    plt.plot(PLOT[t][0,:limit]);
    plt.savefig('./plot'+str(t));
plt.show();