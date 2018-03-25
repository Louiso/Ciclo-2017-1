import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pickle
mat = scipy.io.loadmat('./ECGVilla.mat')
cant = len(mat)

x = mat['time'][:,0]

indices = [ #
            'yy3r2','yy3r3','yy3r1','yy5r1','yy5r2', 'yy4' , 'yy2r3',
            'yy2r2','yy2r1','yy4r1','yy4r3','yy4r2',
            #new Version
            'yy5' , 'yy1r1',
           'yy1r2','yy1r3', 'yy1' , 'yy3' , 'yy2' ,
           #cabecera y global
           'yy5r3']
indices = [ #   Normal  Anomalia1   Anomalia2   Anomalia3
                'yy1',  'yy1r1',    'yy1r2',    'yy1r3',
                'yy2',  'yy2r1',    'yy2r2',    'yy2r3',
                'yy3',  'yy3r1',    'yy3r2',    'yy3r3',
                'yy4',  'yy4r1',    'yy4r2',    'yy4r3',
                'yy5',  'yy5r1',    'yy5r2',    'yy5r3'
           ]

#Cantidad de entradas
r = 686 # por periodo
offset = 415
x = np.array([4808,1])
x = mat['time'][:,0]
x = x[offset:offset+686]
#686 dura su periodos
tiposDeOndas = 5
min = 0
data = []
for i in range(tiposDeOndas):
    # plt.figure()
    for j in range(4):
        index = 4*i + j
        y = mat[indices[index]][:,0]
        y = y[offset:offset+686]
        data.append(y)
        # plt.subplot(2,2,j+1)
        # plt.plot(x,y)
        # plt.title(indices[index])
    # plt.show()
pickle.dump(data,open('anomalias','wb'))
