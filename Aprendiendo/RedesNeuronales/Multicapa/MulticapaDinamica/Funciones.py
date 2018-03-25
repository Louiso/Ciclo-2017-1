import numpy as np
def sigmoidea(n):
    return 1/(1+np.exp(-n))

def sigmoidea2(n):
    return 1 - 2/(1+np.exp(2*n))

def diag(x,f = None):
    n = len(x)
    temp = np.zeros([n,n])
    for i in range(n):
        if f!=None:
            temp[i,i] = f(x[i])
        else :
            temp[i,i] = x[i]
    return temp

def ini_Pesos_Funciones_Derivadas(r,M):
    WB = {};f = {}; df = {};
    radio =  0.01;
    for m in range(1,M-1):
        WB[m] = 2*radio*np.random.random([r[m],r[m-1]+1]) - radio;
        f[m] = sigmoidea2;
        df[m] = lambda n: 1-n**2;
    WB[M-1] = 2*radio*np.random.random([r[M-1],r[M-2]+1])- radio;
    f[M-1] = lambda n:n;
    df[M-1] = lambda n: 1;
    return (WB,f,df);
def ini_dJdWBT(r,M):
    dJdWBT = {};
    dJdWBT[1] = np.zeros([r[1],r[0]+1]);
    for m in range(2,M):
        dJdWBT[m] = np.zeros([r[m],r[m-1]+1]);
    return dJdWBT;

def ini_dzdWBT(r,M):
    dzdWBT = {};
    for i in range(r[0]):
        dzdWBT[i] = {};
        for m in range(1,M):
            dzdWBT[i][m] = np.zeros([r[m],r[m-1]+1]);
    return dzdWBT;

def ini_dudWBT(r,M):
    dudWBT = {};
    for i in range(r[M-1]):
        dudWBT[i] = {};
        dudWBT[i][1] = np.zeros([r[1],r[0]+1]);
        for m in range(2,M):
            dudWBT[i][m] = np.zeros([r[m],r[m-1]+1]);
    return dudWBT;
def ini_Jacobiano(WB,df,a,r,M):
    Jacobiano = WB[M-1][:,:-1];
    for m in range(M-2,1,-1): 
        F = diag(a[m][:-1],df[m]);
        Jacobiano = Jacobiano.dot(F).dot(WB[m][:,:-1]);
    F = diag(a[1][:-1],df[1]);
    Jacobiano = Jacobiano.dot(F).dot(WB[1][:,:-1]);
    return Jacobiano;
def ini_Sensibilidades(WB,df,a,r,M):
    S = {}
    for i in range(r[M-1]):
        S[i] = {};            
        S[i][M-1] = np.zeros([r[M-1],1]);
        S[i][M-1][i,0] = 1;
        for m in range(M-2,0,-1):
            F = diag(a[m][:-1],df[m]);
            S[i][m] = F.dot(WB[m+1][:,:-1].T).dot(S[i][m+1]);
    return S;
def update_dudWBT(dudWBT,dzdWBT,S,Jacobiano,a,r,M):
    temp = {};
    for i in range(r[M-1]):
        temp[i] = {};
    for m in range(1,M):
        for i in range(r[M-1]):
            dudWB_i_m = S[i][m].dot(a[m-1].T);
            dudWBT[i][m] = dudWB_i_m[:,:]
            for j in range(r[0]):
                dudWBT[i][m] = dudWBT[i][m] + Jacobiano[i,j]*(dzdWBT[j][m]);

def update_dJdWBT(dJdWBT,dzdWBT,dudWBT,err,r,M,vel,L,Patron_Z,q):
    dJqdWBT = {};
    for m in range(1,M):
        dJqdWBT[m] = np.zeros([r[m],r[m-1]+1]);
    
    for m in range(1,M):
        dzdWBT[0][m] = dzdWBT[0][m] - vel*np.sin(Patron_Z[1,q-1])*dzdWBT[1][m];
        dzdWBT[1][m] = dzdWBT[1][m] - vel/L*dudWBT[0][m];
        dJqdWBT[m] = -2*err[0]*dzdWBT[0][m] - 2*err[1]*dzdWBT[1][m];
        dJdWBT[m] = dJdWBT[m] + dJqdWBT[m];