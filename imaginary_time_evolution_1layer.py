# import qutip as qp
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

I4=multikron(I,I,I,I)

n_qubit=4
n_basis=5

basis='legendre'
path='log\\figure\\NGD(lr=0.3,n=(2,2,3),H2).png'

Ts=[2*np.pi]
lr=0.02
epoches=400

psi0=np.random.rand(2**n_qubit)
psi0=psi0/np.linalg.norm(psi0)
Hsys=0
Hs=[multikron(X,I,I,I),multikron(I,X,I,I),multikron(I,I,X,I),multikron(I,I,I,X),multikron(Z,Z,I,I),multikron(I,Z,Z,I),multikron(I,I,Z,Z),multikron(Z,I,I,Z)]
n_pulse=len(Hs)

a0=-1.0524
a1=-0.0113
a2=0.1809
a3=-0.3979
a4=0.3979
Hc=multikron(I,I,I,I)+a0*multikron(I,I,I,I)+a1*multikron(Z,Z,I,I)+a2*multikron(X,X,I,I)+a3*multikron(Z,I,I,I)+a4*multikron(I,Z,I,I)

def sigmoid(x):
    if x>0:
        return (1-math.exp(-x)) / (1 + math.exp(-x))
    else:
        return (math.exp(x)-1) / (math.exp(x) + 1)
    
def legendre(n,x):
    if n==0:
        return 1
    elif n==1:
        return x
    elif n==2:
        return (3*x**2-1)/2
    elif n==3:
        return (5*x**3-3*x)/2
    elif n==4:
        return (35*x**4-30*x**2+3)/8
    elif n==5:
        return (63*x**5-70*x**3+15*x)/8

def u(i,t):
    u=0
    n=n_basis
    for j in range(n):
        if basis == 'legendre':
            u+=spectral_coeff[i*n+j]*legendre(j,2*t/T-1)
    u=sigmoid(u)
    return u

def trotter(t_start,t_end,psi):
        """
        Trotterization of the quantum state under Hamiltonian H(t)
        """
        dt=0.1
        n_step=int(abs(t_end-t_start)/dt)+1
               
        dt=(t_end-t_start)/n_step
        for t in range(n_step):
            H_t=Hsys
            for i in range(n_pulse):
                H_t=H_t+Hs[i]*u(i,t_start+t*dt)
            dU=expm(-1j*H_t*dt) 
                      
            psi=dU@psi   
        return psi

def computeMC(n_sample=10):
    
    psi_it=np.zeros((n_pulse,n_sample,2**n_qubit),dtype=complex)

    ti=np.linspace(T/(2*n_sample),T*(2*n_sample-1)/(2*n_sample),n_sample)
    for i in range(n_pulse):
        for t in range(n_sample):
            psi=trotter(0,ti[t],psi0)
            psi=Hs[i]@psi
            psi=trotter(ti[t],T,psi)
            psi_it[i][t]=psi

    #print(psi_it)
    
    M=np.zeros((n_pulse*n_basis,n_pulse*n_basis))
    for i1 in range(n_pulse):
        for j1 in range(n_basis):
            for i2 in range(n_pulse):
                for j2 in range(n_basis):
                    z=np.zeros(n_sample**2)
                    for t1 in range(n_sample):
                        for t2 in range(n_sample):
                            du1=0.5*(1+u(i1,ti[t1]))*(1-u(i1,ti[t1]))*legendre(j1,(2*t1-n_sample+1)/n_sample)
                            du2=0.5*(1+u(i2,ti[t2]))*(1-u(i2,ti[t2]))*legendre(j2,(2*t2-n_sample+1)/n_sample)
                            z[t1*n_sample+t2]=(du1*du2*np.conjugate(psi_it[i1][t1])@psi_it[i2][t2]).real
                    M[i1*n_basis+j1][i2*n_basis+j2]=np.mean(z)*(T**2)

    psi_f=trotter(0,T,psi0)
    C=np.zeros(n_pulse*n_basis)
    for i in range(n_pulse):
        for j in range(n_basis):
            z=np.zeros(n_sample)
            for t in range(n_sample):
                du=0.5*(1+u(i,ti[t]))*(1-u(i,ti[t]))*legendre(j,(2*t-n_sample+1)/n_sample)
                z[t]=(du*np.conjugate(psi_f)@Hc@psi_it[i][t]).imag
            C[i*n_basis+j]=-np.mean(z)*T


    return M,C

t=time.localtime()


dE=np.zeros(epoches)
x=np.linspace(0,epoches,epoches)
E,p=np.linalg.eig(Hc)
E0=np.min(E)
coeff0=np.random.rand(n_basis*n_pulse)/10

for T in Ts:
    spectral_coeff=coeff0
    for i in range(epoches):
            
        M,C=computeMC() 
        spectral_coeff=spectral_coeff+lr*np.linalg.pinv(M)@C
        psi_p=trotter(0,T,psi0)
        dE[i]=np.log(abs(np.conjugate(psi_p)@Hc@psi_p-E0))/np.log(10)
        print('T='+str(T)+"   epoch="+str(i)+"   lgloss="+str(dE[i]))

    plt.plot(x,dE,label='T='+str(T))

plt.legend()
plt.savefig(path)


#file.close()
    

