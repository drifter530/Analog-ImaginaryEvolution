import numpy as np
import matplotlib.pyplot as plt
# import torch
import scipy 
#from scipy.special import legendre
import time
import math
from scipy.linalg import expm

X=np.array([[0,1],[1,0]])
Y=np.array([[0,-1j],[1j,0]])
Z=np.array([[1,0],[0,-1]])
I=np.array([[1,0],[0,1]])

def multikron(*args):
    ret = np.array([[1]])
    for op in args:
        ret = np.kron(ret, op)
    return ret

graph_path='log\\figure_2\\{}:square_box_{}.png'
data_path='log\\data_2\\{}:square_box_{}.npy'
date=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

lr=0.2
epoches=200
n_qubit=2
n_layer=2
n_sample=10
T_layer=[[np.pi],[np.pi]]
psi0=np.random.rand(2**n_qubit)
psi0=psi0/np.linalg.norm(psi0)
Hsys=0
Hs=[multikron(X,Z),multikron(Z,X),multikron(X,X)]
n_pulse=len(Hs)
coeff=np.zeros((n_layer,n_pulse))

a0=-1.0524
a1=-0.0113
a2=0.1809
a3=-0.3979
a4=0.3979
Hc=a0*np.kron(I,I)+a1*np.kron(Z,Z)+a2*np.kron(X,X)+a3*np.kron(Z,I)+a4*np.kron(I,Z)

def sigmoid(x):
    if x>0:
        return (1-math.exp(-x)) / (1 + math.exp(-x))
    else:
        return (math.exp(x)-1) / (math.exp(x) + 1)
    
def trotter_const(t_start,t_end,psi):
    t=t_end-t_start
    H_t=Hsys
    for i in range(n_pulse):
        H_t=H_t+Hs[i]*sigmoid(coeff[i])
    U=expm(-1j*H_t*t)             
    psi=U@psi   
    return psi

def compute_natural_gradient():
    grad=np.zeros((n_layer,n_pulse))
    hessian=np.zeros((n_layer,n_pulse,n_layer,n_pulse))
    grad_psi=np.zeros((n_layer,n_pulse,2**n_qubit),dtype=np.complex128)

    for i in range(n_layer):
        for j in range(n_pulse):
            psi_ij=np.zeros((n_sample,2**n_qubit),dtype=np.complex128)
            for t in range(n_sample):
                
                for k in range(n_layer):
                    if k==i:
                        psi=trotter_const(0,T_layer[i][k],psi0)
                        psi=Hs[j]@psi
                        psi=trotter_const(T_layer[i][k],T_layer[i][-1],psi)
                    else:
                        psi=trotter_const(0,T_layer[i][k],psi)


