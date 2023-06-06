import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
import time

from Simplain import Analog

sim=Analog(n_basis=3,basis='legendre',n_qubit=2,dt=0.01)

X=np.array([[0,1],[1,0]])
Y=np.array([[0,-1j],[1j,0]])
Z=np.array([[1,0],[0,-1]])
I=np.array([[1,0],[0,1]])

sim.spectral_coeff=np.array([0,0,0,0,0,0],dtype=np.float64)

e1=0.3
e2=0.3
J12=0.5
sim.Hsys=e1/2*(np.kron(I,I)-np.kron(Z,I))+e2/2*(np.kron(I,I)-np.kron(I,Z))+J12*(np.kron(Y,Y)+np.kron(X,X))/4
sim.psi0=np.array([1/np.sqrt(2),1/np.sqrt(2),0,0])
sim.Hs=[np.kron(X,I),np.kron(I,X)]

a0=-1.0524
a1=-0.0113
a2=0.1809
a3=-0.3979
a4=0.3979
sim.Hc=a0*np.kron(I,I)+a1*np.kron(Z,Z)+a2*np.kron(X,X)+a3*np.kron(Z,I)+a4*np.kron(I,Z)

E,V0=np.linalg.eigh(sim.Hc)
E0=min(E)

dt0=0.05

t=time.localtime()

path='log\\text\\vte_'+str(t.tm_mday)+str(t.tm_hour)+str(t.tm_min)+'.txt'
file=open(path, 'w')


file.write("time:"+":"+str(t.tm_wday)+":"+str(t.tm_mday)+":"+str(t.tm_hour)+":"+str(t.tm_min)+":"+str(t.tm_sec)+'\n'+'\n')


for t in [2,4,6,8,10,12,14]:
    sim.T=t
    dE=np.zeros(400)

    for i in range(400):
        print('t='+str(t)+"    "+str(i))
        M,C=sim.computeMC()
        sim.spectral_coeff=sim.spectral_coeff+dt0*np.linalg.pinv(M)@C
        psif=sim.trotter(0,t,sim.psi0)
        dE[i]=abs(np.conjugate(psif)@sim.Hsys@psif-E0)

    x=np.linspace(0,400,400)
    plt.plot(x,dE)

    path2='log\\figure\\NGD_t='+str(t)+'.png'
    plt.savefig(path2,dpi=400)