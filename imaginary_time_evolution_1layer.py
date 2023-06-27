# import qutip as qp
import numpy as np
import matplotlib.pyplot as plt
# import torch
import scipy 
from scipy.special import legendre
import time
import math
from scipy.linalg import expm


X=np.array([[0,1],[1,0]])
Y=np.array([[0,-1j],[1j,0]])
Z=np.array([[1,0],[0,-1]])
I=np.array([[1,0],[0,1]])

#多个矩阵的Kronecker积
def multikron(*args):
    ret = np.array([[1]])
    for op in args:
        ret = np.kron(ret, op)
    return ret

#保存图片的标签
label=''

#qubit数，基底数，采样点数,基底类型
n_qubit=2
n_basis=7
n_sample=5
basis='legendre' 

#pulse总时长，Learning rate，epoch数
Ts=[4*np.pi,6*np.pi,8*np.pi,10*np.pi]
lr=0.3
epoches=100

#随机初始量子态
psi0=np.random.rand(2**n_qubit)
psi0=psi0/np.linalg.norm(psi0)

#控制Hamiltonian
J12=0.5
Hsys=0.3/2*(np.kron(I,I)-np.kron(Z,I))+0.5/2*(np.kron(I,I)-np.kron(I,Z))+J12*(np.kron(Y,Y)+np.kron(X,X))/4
#Hs=[multikron(X,I,I,I),multikron(I,X,I,I),multikron(I,I,X,I),multikron(I,I,I,X),multikron(Z,Z,I,I),multikron(I,Z,Z,I),multikron(I,I,Z,Z),multikron(Z,I,I,Z)]
Hs=[multikron(X,I),multikron(I,X)]
n_pulse=len(Hs)

#目标Hamiltonian
a0=-1.0524
a1=-0.0113
a2=0.1809
a3=-0.3979
a4=0.3979
Hc=a0*np.kron(I,I)+a1*np.kron(Z,Z)+a2*np.kron(X,X)+a3*np.kron(Z,I)+a4*np.kron(I,Z)

legendre_ps=[legendre(j) for j in range(n_basis)]

#定义sigmoid函数（用于将控制函数映射到[-1,1]）
def sigmoid(x):
    if x>0:
        return (1-math.exp(-x)) / (1 + math.exp(-x))
    else:
        return (math.exp(x)-1) / (math.exp(x) + 1)

def u(i,t):
    u=0
    n=n_basis
    for j in range(n):
        if basis == 'legendre':
            u+=spectral_coeff[i*n+j]*legendre_ps[j](2*t/T-1)
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

def natural_gradient():

    #计算自然梯度： M Delta_L =C
    
    #采样时间（等间隔）
    ti=np.linspace(T/(2*n_sample),T*(2*n_sample-1)/(2*n_sample),n_sample)

    #对于不同的控制函数和采样时间，计算|psi_it>。注意到对于不同基函数并不需要重复计算。
    psi_it=np.zeros((n_pulse,n_sample,2**n_qubit),dtype=complex)
    for i in range(n_pulse):
        for t in range(n_sample):
            psi=trotter(0,ti[t],psi0)
            psi=Hs[i]@psi
            psi=trotter(ti[t],T,psi)
            psi_it[i][t]=psi

    M=np.zeros((n_pulse*n_basis,n_pulse*n_basis))
    for i1 in range(n_pulse):
        for j1 in range(n_basis):
            for i2 in range(n_pulse):
                for j2 in range(n_basis):

                    #计算M矩阵矩阵元
                    #矩形积分
                    #M_ij=sum(t1,t2) du_i(t1)du_j(t2) Re[<psi_it(t1)|psi_it(t2)>]

                    z=0
                    for t1 in range(n_sample):
                        for t2 in range(n_sample):
                            du1=0.5*(1+u(i1,ti[t1]))*(1-u(i1,ti[t1]))*legendre_ps[j1]((2*t1-n_sample+1)/n_sample)
                            du2=0.5*(1+u(i2,ti[t2]))*(1-u(i2,ti[t2]))*legendre_ps[j2]((2*t2-n_sample+1)/n_sample)
                            z+=(du1*du2*np.conjugate(psi_it[i1][t1])@psi_it[i2][t2]).real

                    M[i1*n_basis+j1][i2*n_basis+j2]=z*(T**2)/(n_sample**2)
                    


    psi_f=trotter(0,T,psi0)
    C=np.zeros(n_pulse*n_basis)
    for i in range(n_pulse):
        for j in range(n_basis):

            #计算C向量
            #C_i=-sum(t) du_i(t) Im[<psi_f|Hc|psi_it(t)>]

            z=0
            for t in range(n_sample):
                du=0.5*(1+u(i,ti[t]))*(1-u(i,ti[t]))*legendre_ps[j]((2*t-n_sample+1)/n_sample)
                z+=(du*np.conjugate(psi_f)@Hc@psi_it[i][t]).imag
            C[i*n_basis+j]=-np.mean(z)*T/n_sample

    #取M的广义逆矩阵

    return np.linalg.pinv(M,rcond=0.001)@C

def natural_gradient_random():
    #计算自然梯度： M Delta_L =C
    
    #采样时间（等间隔）
    ti=np.linspace(T/(2*n_sample),T*(2*n_sample-1)/(2*n_sample),n_sample)

    #对于不同的控制函数和采样时间，计算|psi_it>。注意到对于不同基函数并不需要重复计算。
    psi_it=np.zeros((n_pulse,n_sample,2**n_qubit),dtype=complex)
    for i in range(n_pulse):
        for t in range(n_sample):
            psi=trotter(0,ti[t],psi0)
            psi=Hs[i]@psi
            psi=trotter(ti[t],T,psi)
            psi_it[i][t]=psi

    M=np.zeros((n_pulse*n_basis,n_pulse*n_basis))
    for i1 in range(n_pulse):
        for j1 in range(n_basis):
            for i2 in range(n_pulse):
                for j2 in range(n_basis):

                    #计算M矩阵矩阵元
                    #矩形积分
                    #M_ij=sum(t1,t2) du_i(t1)du_j(t2) Re[<psi_it(t1)|psi_it(t2)>]

                    z=0
                    for t1 in range(n_sample):
                        for t2 in range(n_sample):
                            du1=0.5*(1+u(i1,ti[t1]))*(1-u(i1,ti[t1]))*legendre_ps[j1]((2*t1-n_sample+1)/n_sample)
                            du2=0.5*(1+u(i2,ti[t2]))*(1-u(i2,ti[t2]))*legendre_ps[j2]((2*t2-n_sample+1)/n_sample)
                            z+=(du1*du2*np.conjugate(psi_it[i1][t1])@psi_it[i2][t2]).real

                    M[i1*n_basis+j1][i2*n_basis+j2]=z*(T**2)/(n_sample**2)


    psi_f=trotter(0,T,psi0)
    C=np.zeros(n_pulse*n_basis)
    for i in range(n_pulse):
        for j in range(n_basis):

            #计算C向量
            #C_i=-sum(t) du_i(t) Im[<psi_f|Hc|psi_it(t)>]

            z=0
            for t in range(n_sample):
                du=0.5*(1+u(i,ti[t]))*(1-u(i,ti[t]))*legendre_ps[j]((2*t-n_sample+1)/n_sample)
                z+=(du*np.conjugate(psi_f)@Hc@psi_it[i][t]).imag
            C[i*n_basis+j]=-np.mean(z)*T/n_sample

    #取M的广义逆矩阵
    return np.linalg.pinv(M)@C

lgloss=np.zeros(epoches)
x=np.linspace(0,epoches,epoches)
E,p=np.linalg.eig(Hc)
E0=np.min(E)

#随机初始参数
coeff0=np.random.rand(n_basis*n_pulse)/10

for n_sample in [3,5,7,10]:
    for T in Ts:
        spectral_coeff=coeff0
        for i in range(epoches):

            #更新参数
            spectral_coeff=spectral_coeff+lr*natural_gradient()
            #print(spectral_coeff)

            #计算loss function（对数形式）
            psi_p=trotter(0,T,psi0)
            lgloss[i]=np.log(abs(np.conjugate(psi_p)@Hc@psi_p-E0))/np.log(10)
            print("T={}    epoch={}    lgloss={}".format(T,i,lgloss[i]))

        plt.plot(x,lgloss,label='T='+str(T))

    #保存图片至log文件夹
    path="log\\figure_0627\\NGD(lr={},(q,b,p,s)=({},{},{},{}),{}).png".format(lr,n_qubit,n_basis,n_pulse,n_sample,str(Ts))
    plt.legend()
    plt.savefig(path)
    plt.cla()



    

