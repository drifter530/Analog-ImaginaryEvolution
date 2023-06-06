import qutip as qp
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy 
from scipy.special import legendre
from scipy.linalg import expm
import time
import math

class Analog(object):



    def __init__(self,n_basis=4,basis='legendre',n_qubit=2,dt=0.01):
        self.n_basis = n_basis
        self.basis = basis
        self.dt = dt
        self.T=1
        self.H0 = 0
        self.Hs = []
        self.Hsys = 0
        self.psi0 = 0
        self.Hc = 0
        self.spectral_coeff = []
        self.n_qubit = n_qubit
        self.omega = []

    def setcoeff(self):
        self.spectral_coeff = np.random.rand(self.n_basis*self.n_qubit)
        self.omega = np.random.rand(self.n_basis*self.n_qubit)

    def sigmoid(self,x):
        if x>0:
            return (1-math.exp(-x)) / (1 + math.exp(-x))
        else:
            return (math.exp(x)-1) / (math.exp(x) + 1)
        
    def Legendre(self,n,x):
        if n==0:
            return 1
        elif n==1:
            return x
        elif n==2:
            return (3*x**2-1)/2
        
    def u(self,i,t):
        u=0
        n=self.n_basis
        for j in range(n):
            if self.basis == 'legendre':
                u+=self.spectral_coeff[i*n+j]*self.Legendre(j,2*t/self.T-1)
        u=self.sigmoid(u)
        return u
    
    def trotter(self,t_start,t_end,psi):
        """
        Trotterization of the quantum state under Hamiltonian H(t)
        """
        dt=self.dt
        n_step=int(abs(t_end-t_start)/dt)+1
               
        dt=(t_end-t_start)/n_step
        for t in range(n_step):
            H_t=self.Hsys
            for i in range(len(self.Hs)):
                H_t=H_t+self.Hs[i]*self.u(i,t_start+t*dt)
            dU=expm(-1j*H_t*dt) 
            psi=dU@psi   
        return psi
    
    def computeMC(self,n=10):
        m_pauli=2
        psi_it=np.zeros((m_pauli,n,4),dtype=complex)
        ti=np.linspace(self.T/(2*n),self.T*(2*n-1)/(2*n),n)
        for i in range(m_pauli):
            for t in range(n):
                psi=self.trotter(0,ti[t],self.psi0)
                psi=self.Hs[i]@psi
                psi=self.trotter(ti[t],self.T,psi)
                psi_it[i][t]=psi


        M=np.zeros((m_pauli*self.n_basis,m_pauli*self.n_basis))
        for i1 in range(m_pauli):
            for j1 in range(self.n_basis):
                for i2 in range(m_pauli):
                    for j2 in range(self.n_basis):
                        z=np.zeros(n*n)
                        for t1 in range(n):
                            for t2 in range(n):
                                du1=0.5*(1-self.u(i1,ti[t1])**2)*self.Legendre(j1,(2*t1-n+1)/n)
                                du2=0.5*(1-self.u(i1,ti[t1])**2)*self.Legendre(j2,(2*t2-n+1)/n)
                                z[t1*n+t2]=(du1*du2*np.conjugate(psi_it[i1][t1])@psi_it[i2][t2]).real
                        M[i1*self.n_basis+j1][i2*self.n_basis+j2]=np.mean(z)*(self.T**2)

        psi_f=self.trotter(0,self.T,self.psi0)
        C=np.zeros(m_pauli*self.n_basis)
        for i in range(m_pauli):
            for j in range(self.n_basis):
                z=np.zeros(n)
                for t in range(n):
                    du=0.5*(1+self.u(i,ti[t]))*(1-self.u(i,ti[t]))*self.Legendre(j,(2*t-9)/10)
                    z[t]=(du*np.conjugate(psi_f)@self.Hc@psi_it[i][t]).imag
                C[i*self.n_basis+j]=-np.mean(z)*self.T


        return M,C


    



