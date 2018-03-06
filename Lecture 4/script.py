import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
import math

T=5
N=4

def forward(alpha,A,b,sequence):
    a=0
    pos=0
    for t in range(1,T):
        for i in range(0,alpha.shape[0]):
                alpha[i][t]=((alpha[0][t-1]*A[0][i])+(alpha[1][t-1]*A[1][i])+(alpha[2][t-1]*A[2][i])+(alpha[3][t-1]*A[3][i]))*b[i][sequence[pos]]     
        pos=pos+1
    return alpha

def decoding(A, b, alpha):
    hidden_sequence_probable=np.array([0, 0, 0, 0, 0])
    tmp=0 #variable that takes the maximum of each columns
    for t in range(0,alpha.shape[1]):
        for i in range(0,alpha.shape[0]):
            if alpha[i][t]>tmp:
                tmp=alpha[i][t]
                hidden_sequence_probable[t]=i #stores the index of the maximum
        tmp=0
    return hidden_sequence_probable

def backward(beta,A,b,sequence):
    a=0
    pos=3
    beta[0][T-1]=1
    for t in range(T-2,-1,-1):
        for i in range(0,alpha.shape[0]):
                beta[i][t]=((beta[0][t+1])*A[i][0]*b[0][sequence[t]])+((beta[1][t+1])*A[i][1]*b[1][sequence[t]])+((beta[2][t+1])*A[i][2]*b[2][sequence[t]])+((beta[3][t+1])*A[i][3]*b[3][sequence[t]])
                #beta[i][t]=(beta[i][t+1])*((A[0][i]*b[i][sequence[pos-1]])+(A[1][i]*b[i][sequence[pos-1]])+(A[2][i]*b[i][sequence[pos-1]])+(A[3][i]*b[i][sequence[pos-1]]))
                #beta[i][t]=((beta[0][t+1])*A[i][0]*b[i][sequence[pos-1]] )+((beta[1][t+1])*A[i][1]*b[i][sequence[pos-1]] )+((beta[2][t+1])*A[i][2]*b[i][sequence[pos-1]] )+((beta[3][t+1])*A[i][3]*b[i][sequence[pos-1]])
        pos=pos-1
    return beta

def get_gamma(gamma,alpha,A,b,beta,sequence):
    for t in range(1,T):
        for i in range(N):
            for j in range(N):
                #print(t)
                gamma[i][j][t]=alpha[i][t-1]*A[i][j]*b[j][sequence[t-1]]*beta[j][t]

    return gamma
        
def update_a(A,gamma):
    for i in range(N):
        for j in range(N): 
            denominateur=0
            for t in range(1,T):
                A[i][j]+=gamma[i][j][t]
                denominateur += np.sum(gamma[i][:][t-1])
            if denominateur != 0:
                A[i][j] /= denominateur
    A[0][0]=1
    return A

def update_b(b,gamma,sequence):
    for j in range(N):
        for k in range(N):
            denominateur=0
            for t in range(1, T):
                denominateur += np.sum(gamma[j][:][t-1])
                kxt = sequence[t-1]
                if kxt == k:
                    for l in range(N):
                        b[j][k+1]+= gamma[j][l][t]
            if denominateur != 0:
                    b[j][k+1] /= denominateur
    b[0,0] = 1
    return b
        

def Baum_Welch(alpha,beta,gamma,A,b,sequence):
    for i in range(1,5):
        alpha=forward(alpha,A,b,sequence)
        beta=backward(beta,A,b,sequence)
        gamma=get_gamma(gamma,alpha,A,b,beta,sequence)
        A=update_a(A,gamma)
        b=update_b(b,gamma,sequence)
    return A,b

                
A = np.array([[1.0, 0.0, 0.0, 0.0], [0.2, 0.3, 0.1, 0.4], [0.2, 0.5, 0.2, 0.1], [0.7, 0.1, 0.1, 0.1]])
b= np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0, 0.3, 0.4, 0.1, 0.2], [0, 0.1, 0.1, 0.7, 0.1], [0, 0.5, 0.2, 0.1, 0.2]])
sequence=np.array([1, 3, 2, 0])
alpha= np.random.randn(b.shape[0], b.shape[1])
alpha2= np.random.randn(b.shape[0], b.shape[1])
beta= np.random.randn(N,T)
gamma = np.zeros((N, N, T))
for t in range(b.shape[0]):
    for j in range(b.shape[1]):
        alpha[t][j]=0
        alpha2[t][j]=0
        beta[t][j]=0
alpha[1][0]=1
alpha2[1][0]=1

alpha=forward(alpha,A,b,sequence)
print("HMM forward diagram- Evolution:")
print(alpha)
sequence_prob=decoding(A,b,alpha)
print("")
print("")
print("The most probable hidden states of our model:")
print("Z%d Z%d Z%d Z%d Z%d" %(sequence_prob[0],sequence_prob[1],sequence_prob[2],sequence_prob[3],sequence_prob[4]))
print("")
beta=backward(beta,A,b,sequence)
print("")
print("HMM beta diagram - backward")
print(beta)

print("")

A,b=Baum_Welch(alpha,beta,gamma,A,b,sequence)

print("new A")
print(np.around(A, 3))
print("new B")
print(np.around(b, 3))
print("----------")

alpha2=forward(alpha2,A,b,sequence)


