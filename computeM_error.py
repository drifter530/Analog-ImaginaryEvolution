import qutip as qp
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy 
from scipy.special import legendre
import time
import math
from scipy.linalg import expm


n_basis=3
basis='legendre'

T=3
point_M=25
point_C=6

a0=-1.0524
a1=-0.0113
a2=0.1809
a3=-0.3979
a4=0.3979

e1=1
e2=1
J12=0.7

X=np.array([[0,1],[1,0]])
Y=np.array([[0,-1j],[1j,0]])
Z=np.array([[1,0],[0,-1]])
I=np.array([[1,0],[0,1]])

spectral_coeff=np.array([1,-1,1,0.5,1,-1])

Hsys=e1/2*(np.kron(I,I)-np.kron(Z,I))+e2/2*(np.kron(I,I)-np.kron(I,Z))+J12*(np.kron(Y,Y)+np.kron(X,X))/4

psi0=np.array([1/np.sqrt(2),1/np.sqrt(2),0,0])

Hs=[np.kron(X,I),np.kron(I,X)]

Hc=a0*np.kron(I,I)+a1*np.kron(Z,Z)+a2*np.kron(X,X)+a3*np.kron(Z,I)+a4*np.kron(I,Z)

def sigmoid(x):
    return (1-math.exp(-x)) / (1 + math.exp(-x))

def u(i,t):
    u=0
    n=n_basis
    for j in range(n):
        if basis == 'legendre':
            u+=spectral_coeff[i*n+j]*legendre(j)(2*t/T-1)
    u=sigmoid(u)
    return u

def trotter_U(t_start,t_end):
    n_step=15
    dt=(t_end-t_start)/n_step
    U=np.kron(I,I)
    for t in range(n_step):
        H_t=Hsys
        for i in range(len(Hs)):
            H_t=H_t+Hs[i]*u(i,t_start+t*dt)
        dU=expm(-1j*H_t*dt)
        U=dU@U
    return U

def computeM1():
    M=np.zeros((2*n_basis,2*n_basis))
    
    for i1 in range(2):
        for j1 in range(n_basis):
            for i2 in range(2):
                for j2 in range(n_basis):
                    t=np.linspace(T/20,T*19/20,10)
                    z=np.zeros(100)
                    U_0tot=np.zeros((10,4,4),dtype=complex)
                    U_ttoT=np.zeros((10,4,4),dtype=complex)
                    Utt=np.zeros((9,4,4),dtype=complex)
                    for i in range(9):
                        Utt[i]=trotter_U(T*(i+0.5)/10,T*(i+1.5)/10)
                    Ut=trotter_U(0,T/20)
                    U_0tot[0]=Ut
                    for i in range(9):
                        Ut=Utt[i]@Ut
                        U_ttoT[i+1]=Ut
                    Ut=trotter_U(T*19/20,T)
                    U_ttoT[9]=Ut
                    for i in range(9):
                        Ut=Ut@Utt[8-i]
                        U_0tot[8-i]=Ut
                    for i in range(10):
                        for j in range(10):
                            psi1=U_0tot[i]@psi0
                            psi1=Hs[i1]@psi1
                            psi1=U_ttoT[i]@psi1
                            psi2=U_0tot[j]@psi0
                            psi2=Hs[i2]@psi2
                            psi2=U_ttoT[j]@psi2
                            du1=0.5*(1+u(i1,t[i]))*(1-u(i1,t[i]))*legendre(j1)(2*t[i]/T-1)
                            du2=0.5*(1+u(i2,t[j]))*(1-u(i2,t[j]))*legendre(j2)(2*t[j]/T-1)
                            z[i*10+j]=(du1*du2*np.conjugate(psi2)@psi1).real
                    M[i1*n_basis+j1][i2*n_basis+j2]=np.mean(z)*np.mean(z)*(T**2)
    return M 


t=time.localtime()
path='C:\\Users\\insta\\Documents\\coding\\analog_quantum\\analog_virtual_time_evolution\\log\\Merr_'+str(t.tm_mday)+str(t.tm_hour)+str(t.tm_min)+'.txt'
file=open(path, 'w')

file.write('spectral_coeff: '+str(spectral_coeff)+'\n')

for i in range(1):
    M=computeM1()
    file.write(str(M)+'\n')
    print(i)

file.close()

