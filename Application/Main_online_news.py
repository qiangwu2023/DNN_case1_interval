import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from I_spline import I_S
from Least_FD import LFD
from iteration_deep import Est_deep
from iteration_linear import Est_linear
from iteration_additive import Est_additive

def set_seed(seed):
    np.random.seed(seed)  
    torch.manual_seed(seed) 
set_seed(6)
p = 3 
n_layer = 3 
n_node = 50 
n_epoch = 200 
n_lr = 5e-4 
m = 15 
df = pd.read_csv('New_complete_center.csv')
U = np.array(df['U'], dtype='float32')
U = np.log(U)
U = np.clip(U, 0.01, np.max(U))
T = np.array(df['T'], dtype='float32')
T = np.log(T)

De = np.array(df['Delta'], dtype='float32') # censoring rate

Z = np.array(df[["Z1","Z2"]], dtype='float32') # binary covariate

X = np.array(df[["X1","X2","X3","X4"]], dtype='float32')

A = np.arange(len(U))
np.random.shuffle(A)
U_R0 = U[A]
De_R0 = De[A]
T_R0 = T[A]
Z_R0 = Z[A]
X_R0 = X[A]

nodevec = np.array(np.linspace(0, np.max(U_R0), m+2), dtype="float32") 
c0 = np.array(0.1*np.ones(m+p), dtype="float32") 
Beta0 = np.array([1, 1], dtype='float32')
m0 = 15
nodevec0 = np.array(np.linspace(-3, 43, m0+2), dtype="float32") 

U_R = U_R0[np.arange(35680)]
De_R = De_R0[np.arange(35680)]
Z_R = Z_R0[np.arange(35680)]
X_R = X_R0[np.arange(35680)]
c_n = int(len(U_R)/5)

U_R_test = np.delete(U_R0, np.arange(35680), axis=0)
De_R_test = np.delete(De_R0, np.arange(35680), axis=0)
Z_R_test = np.delete(Z_R0, np.arange(35680), axis=0)
X_R_test = np.delete(X_R0, np.arange(35680), axis=0)
T_R_test = np.delete(T_R0, np.arange(35680), axis=0)

U_test1 = U_R_test[De_R_test==1]
Z_test1 = Z_R_test[De_R_test==1]
X_test1 = X_R_test[De_R_test==1]

U_sort1 = sorted(U_test1)
n_U1 = len(U_sort1)
U_sort1 = np.array(U_sort1)
U1_025 = U_sort1[round(n_U1*0.25)]
U1_050 = U_sort1[round(n_U1*0.5)]
U1_075 = U_sort1[round(n_U1*0.75)]
U1 = [U1_025, U1_050, U1_075]

U1_value = np.array(np.linspace(0, 10, 20), dtype="float32")


U_test0 = U_R_test[De_R_test==0]
Z_test0 = Z_R_test[De_R_test==0]
X_test0 = X_R_test[De_R_test==0]

U_sort0 = sorted(U_test0)
n_U0 = len(U_sort0)
U_sort0 = np.array(U_sort0)
U0_025 = U_sort0[round(n_U0*0.25)]
U0_050 = U_sort0[round(n_U0*0.5)]
U0_075 = U_sort0[round(n_U0*0.75)]
U0 = [U0_025, U0_050, U0_075]

U0_value = np.array(np.linspace(0, 10, 20), dtype="float32")

fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
ax1.set_title(r'$(a)\quad {\hat\beta}_1$', fontsize=10) 
ax1.set_xlabel("Fold",fontsize=8)    
ax1.set_ylabel("Estimates of effect",fontsize=8) 
ax1.tick_params(axis='both',labelsize=6) 
ax1.grid(True)

fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
ax2.set_title(r'$(b)\quad {\hat\beta}_2$', fontsize=10) 
ax2.set_xlabel("Fold",fontsize=8) 
ax2.set_ylabel("Estimates of effect",fontsize=8) 
ax2.tick_params(axis='both',labelsize=6) 
ax2.grid(True)

B_g_deep1 = np.zeros((5,3)) 
B_g_deep0 = np.zeros((5,3)) 
C_deep = np.zeros((5, m+3))
B_g_L1 = np.zeros((5,3)) 
B_g_L0 = np.zeros((5,3)) 
C_L = np.zeros((5, m+3))
B_g_A1 = np.zeros((5,3)) 
B_g_A0 = np.zeros((5,3)) 
C_A = np.zeros((5, m+3))

B_g_all_deep = np.zeros((5,len(U_R_test)))
B_g_all_L = np.zeros((5,len(U_R_test)))
B_g_all_A = np.zeros((5,len(U_R_test)))

for i in range(5):
    print('i =', i)
    Z_train = np.delete(Z_R, np.arange(i*c_n, (i+1)*c_n), axis=0)
    X_train = np.delete(X_R, np.arange(i*c_n, (i+1)*c_n), axis=0)
    U_train = np.delete(U_R, np.arange(i*c_n, (i+1)*c_n), axis=0)
    De_train = np.delete(De_R, np.arange(i*c_n, (i+1)*c_n), axis=0)
    n = len(Z_train)
   
    Est_hat = Est_deep(X_R_test,Z_train,X_train,U_train,De_train,Beta0,n_layer,n_node,n_lr,n_epoch,nodevec,m,c0)
    
    Beta_deep = Est_hat['Beta']
   
    h_v = I_S(m,Est_hat['c'],U_train,nodevec) * np.exp(np.dot(Z_train,Beta_deep) + Est_hat['g_train'])
    Q_y = h_v * (De_train * np.exp(-h_v)/(1-np.exp(-h_v)+1e-8) - (1-De_train))
    a_b1 = LFD(Z_train[:,0],Z_train,X_train,U_train,De_train,I_S(m,Est_hat['c'],U_train,nodevec),Est_hat['g_train'],Beta_deep,n_layer,n_node=50,n_lr=5e-4,n_epoch=200)
    a_b2 = LFD(Z_train[:,1],Z_train,X_train,U_train,De_train,I_S(m,Est_hat['c'],U_train,nodevec),Est_hat['g_train'],Beta_deep,n_layer,n_node=50,n_lr=5e-4,n_epoch=200)
    
    Info = np.zeros((2,2))
    Info[0,0] = np.mean(Q_y**2 * (Z_train[:,0]-a_b1)**2)
    Info[1,1] = np.mean(Q_y**2 * (Z_train[:,1]-a_b2)**2)
    Info[0,1] = np.mean(Q_y**2 * (Z_train[:,0]-a_b1)*(Z_train[:,1]-a_b2))
    Info[1,0] = Info[0,1]
    Sigma = np.linalg.inv(Info)/n
    sd1 = np.sqrt(Sigma[0,0])
    sd2 = np.sqrt(Sigma[1,1])
   
    y_min1 = Beta_deep[0] - 1.96*sd1
    y_max1 = Beta_deep[0] + 1.96*sd1
    
    ax1.plot(i+1-0.2, Beta_deep[0], marker='s', markersize=4, ls='-', color='blue', label='DPLCM')
    if (i == 0):
        ax1.legend(loc='best', fontsize=6)
    ax1.plot((i+1-0.2)*np.ones(2), np.array([y_min1, y_max1]), color='blue', marker='_', ls='-')
    
    y_min2 = Beta_deep[1] - 1.96*sd2
    y_max2 = Beta_deep[1] + 1.96*sd2

    ax2.plot(i+1-0.2, Beta_deep[1], marker='s', markersize=4, ls='-', color='blue', label='DPLCM')
    if (i == 0):
        ax2.legend(loc='best', fontsize=6)
    ax2.plot((i+1-0.2)*np.ones(2), np.array([y_min2, y_max2]), color='blue', marker='_', ls='-')
    
   
    B_g_deep = Beta_deep[0]*Z_R_test[:,0] + Beta_deep[1]*Z_R_test[:,1] + Est_hat['g_test']
    B_g_all_deep[i] = B_g_deep
    for s in range(3):
        B_g_deep1[i,s] = B_g_deep[De_R_test==1][U_test1==U1[s]]
        B_g_deep0[i,s] = B_g_deep[De_R_test==0][U_test0==U0[s]]
    C_deep[i] = Est_hat['c']
    
    
    Est_L = Est_linear(X_R_test,Z_train,X_train,U_train,De_train,Beta0,nodevec,m,c0)
    Beta_L = Est_L['Beta']
    h_v_L = I_S(m,Est_L['c'],U_train,nodevec) * np.exp(np.dot(Z_train,Beta_L)+ Est_L['g_train'])
    Q_y_L = h_v_L * (De_train * np.exp(-h_v_L)/(1-np.exp(-h_v_L)) - (1-De_train))
    a_b_L1 = LFD(Z_train[:,0],Z_train,X_train,U_train,De_train,I_S(m,Est_L['c'],U_train,nodevec),Est_L['g_train'],Beta_L,n_layer,n_node=50,n_lr=2e-3,n_epoch=200)
    a_b_L2 = LFD(Z_train[:,1],Z_train,X_train,U_train,De_train,I_S(m,Est_L['c'],U_train,nodevec),Est_L['g_train'],Beta_L,n_layer,n_node=50,n_lr=2e-3,n_epoch=200)
    Info_L = np.zeros((2,2))
    Info_L[0,0] = np.mean(Q_y**2 * (Z_train[:,0]-a_b_L1)**2)
    Info_L[1,1] = np.mean(Q_y**2 * (Z_train[:,1]-a_b_L2)**2)
    Info_L[0,1] = np.mean(Q_y**2 * (Z_train[:,0]-a_b_L1)*(Z_train[:,1]-a_b_L2))
    Info_L[1,0] = Info_L[0,1]
    Sigma_L = np.linalg.inv(Info_L)/n
    sd_L1 = np.sqrt(Sigma_L[0,0])
    sd_L2 = np.sqrt(Sigma_L[1,1])
  
    y_min_L1 = Beta_L[0] - 1.96*sd_L1
    y_max_L1 = Beta_L[0] + 1.96*sd_L1
    ax1.plot(i+1, Beta_L[0], marker='o', markersize=4, ls='-', color='orange', label='CPH')
    if (i == 0):
        ax1.legend(loc='best', fontsize=6)
    ax1.plot((i+1)*np.ones(2), np.array([y_min_L1, y_max_L1]), marker='_', ls='-', color='orange')
    
    y_min_L2 = Beta_L[1] - 1.96*sd_L2
    y_max_L2 = Beta_L[1] + 1.96*sd_L2
    ax2.plot(i+1, Beta_L[1], marker='o', markersize=4, ls='-', color='orange', label='CPH')
    if (i == 0):
        ax2.legend(loc='best', fontsize=6)
    ax2.plot((i+1)*np.ones(2), np.array([y_min_L2, y_max_L2]), marker='_', ls='-', color='orange')
    
    B_g_L = Beta_L[0]*Z_R_test[:,0] + Beta_L[1]*Z_R_test[:,1] + Est_L['g_test']
    B_g_all_L[i] = B_g_L
    for ss in range(3):
        B_g_L1[i,ss] = B_g_L[De_R_test==1][U_test1==U1[ss]]
        B_g_L0[i,ss] = B_g_L[De_R_test==0][U_test0==U0[ss]]
    C_L[i] = Est_L['c']

    Est_A = Est_additive(X_R_test,Z_train,X_train,U_train,De_train,Beta0,nodevec,m,c0,m0,nodevec0)
    Beta_A = Est_A['Beta']
    h_v_A = I_S(m,Est_A['c'],U_train,nodevec) * np.exp(np.dot(Z_train,Beta_A)+ Est_A['g_train'])
    Q_y_A = h_v_A * (De_train * np.exp(-h_v_A)/(1-np.exp(-h_v_A)) - (1-De_train))
    a_b_A1 = LFD(Z_train[:,0],Z_train,X_train,U_train,De_train,I_S(m,Est_A['c'],U_train,nodevec),Est_A['g_train'],Beta_A,n_layer,n_node=50,n_lr=2e-3,n_epoch=200)
    a_b_A2 = LFD(Z_train[:,1],Z_train,X_train,U_train,De_train,I_S(m,Est_A['c'],U_train,nodevec),Est_A['g_train'],Beta_A,n_layer,n_node=50,n_lr=2e-3,n_epoch=200)
    Info_A = np.zeros((2,2))
    Info_A[0,0] = np.mean(Q_y**2 * (Z_train[:,0]-a_b_A1)**2)
    Info_A[1,1] = np.mean(Q_y**2 * (Z_train[:,1]-a_b_A2)**2)
    Info_A[0,1] = np.mean(Q_y**2 * (Z_train[:,0]-a_b_A1)*(Z_train[:,1]-a_b_A2))
    Info_A[1,0] = Info_A[0,1]
    Sigma_A = np.linalg.inv(Info_A)/n
    sd_A1 = np.sqrt(Sigma_A[0,0])
    sd_A2 = np.sqrt(Sigma_A[1,1])
    
    y_min_A1 = Beta_A[0] - 1.96*sd_A1
    y_max_A1 = Beta_A[0] + 1.96*sd_A1
    ax1.plot(i+1+0.2, Beta_A[0], marker='^', markersize=4, ls='-', color='green', label='PLACM')
    if (i == 0):
        ax1.legend(loc='best', fontsize=6)
    ax1.plot((i+1+0.2)*np.ones(2), np.array([y_min_A1, y_max_A1]), marker='_', ls='-', color='green')
   
    y_min_A2 = Beta_A[1] - 1.96*sd_A2
    y_max_A2 = Beta_A[1] + 1.96*sd_A2
    ax2.plot(i+1+0.2, Beta_A[1], marker='^', markersize=4, ls='-', color='green', label='PLACM')
    if (i == 0):
        ax2.legend(loc='best', fontsize=6)
    ax2.plot((i+1+0.2)*np.ones(2), np.array([y_min_A2, y_max_A2]), marker='_', ls='-', color='green')
    
    B_g_A = Beta_A[0]*Z_R_test[:,0] + Beta_A[1]*Z_R_test[:,1] + Est_A['g_test']
    B_g_all_A[i] = B_g_A
    for ss in range(3):
        B_g_A1[i,ss] = B_g_A[De_R_test==1][U_test1==U1[ss]]
        B_g_A0[i,ss] = B_g_A[De_R_test==0][U_test0==U0[ss]]
    C_A[i] = Est_A['c']

B_g_deep1 = np.mean(B_g_deep1,axis=0) # delta=1
B_g_deep0 = np.mean(B_g_deep0,axis=0) # delta=0
C_deep = np.mean(C_deep,axis=0)
B_g_L1 = np.mean(B_g_L1,axis=0) # delta=1
B_g_L0 = np.mean(B_g_L0,axis=0) # delta=0
C_L = np.mean(C_L,axis=0)
B_g_A1 = np.mean(B_g_A1,axis=0) # delta=1
B_g_A0 = np.mean(B_g_A0,axis=0) # delta=0
C_A = np.mean(C_A,axis=0)

B_g_all_deep = np.mean(B_g_all_deep,axis=0)
B_g_all_L = np.mean(B_g_all_L,axis=0)
B_g_all_A = np.mean(B_g_all_A,axis=0)
S_t_D = np.exp(-I_S(m,C_deep,U_R_test,nodevec) * np.exp(B_g_all_deep))
S_t_L = np.exp(-I_S(m,C_L,U_R_test,nodevec) * np.exp(B_g_all_L))
S_t_A = np.exp(-I_S(m,C_A,U_R_test,nodevec) * np.exp(B_g_all_A))

for k in range(3):
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(1, 1, 1)
    ax3.set_xlabel("t",fontsize=8)   
    ax3.set_ylabel(r'$\hat{S}(t)$',fontsize=8)
    ax3.tick_params(axis='both',labelsize=6)
    ax3.xaxis.set_ticks_position('bottom')
    ax3.spines['bottom'].set_position(('data',0))
    ax3.yaxis.set_ticks_position('left')
    ax3.spines['left'].set_position(('data',0))
    ax3.grid(True)
   
    St_deep1 = np.exp(-I_S(m,C_deep,U1_value,nodevec) * np.exp(B_g_deep1[k]))
    St_deep1_L = np.exp(-I_S(m,C_L,U1_value,nodevec) * np.exp(B_g_L1[k]))
    St_deep1_A = np.exp(-I_S(m,C_A,U1_value,nodevec) * np.exp(B_g_A1[k]))
   
    ax3.plot(U1_value, St_deep1, color='blue', linestyle='-')
    ax3.plot(U1_value, St_deep1_L, color='orange', linestyle='--')
    ax3.plot(U1_value, St_deep1_A, color='green', linestyle=':')
    
    if (k==0):
        ax3.plot(U1_025, np.exp(-I_S(m,C_deep,np.array([U1_025]),nodevec) * np.exp(B_g_deep1[k])), label='DPLCM', marker='s', markersize=4, ls='-', color='blue')
        ax3.plot(U1_025, np.exp(-I_S(m,C_L,np.array([U1_025]),nodevec) * np.exp(B_g_L1[k])), label='CPH', marker='o', markersize=4, ls='--', color='orange')
        ax3.plot(U1_025, np.exp(-I_S(m,C_A,np.array([U1_025]),nodevec) * np.exp(B_g_A1[k])), label='PLACM', marker='^', markersize=4, ls=':', color='green')
        ax3.plot(np.array([U1_025,U1_025]), np.array([0,np.max([np.exp(-I_S(m,C_deep,np.array([U1_025]),nodevec) * np.exp(B_g_deep1[k])), np.exp(-I_S(m,C_L,np.array([U1_025]),nodevec) * np.exp(B_g_L1[k])), np.exp(-I_S(m,C_A,np.array([U1_025]),nodevec) * np.exp(B_g_A1[k]))])], dtype='float32'), color='k', linestyle='--')
        ax3.legend(loc='best', fontsize=6)
        ax3.set_title(r'$\Delta=1, 25^{\rm{th}}$', fontsize=10) # Set title and size
        fig3.savefig('fig1_25.jpeg', dpi=400, bbox_inches='tight')
    elif (k==1):
        ax3.plot(U1_050, np.exp(-I_S(m,C_deep,np.array([U1_050]),nodevec) * np.exp(B_g_deep1[k])), label='DPLCM', marker='s', markersize=4, ls='-', color='blue')
        ax3.plot(U1_050, np.exp(-I_S(m,C_L,np.array([U1_050]),nodevec) * np.exp(B_g_L1[k])), label='CPH', marker='o', markersize=4, ls='--', color='orange')
        ax3.plot(U1_050, np.exp(-I_S(m,C_A,np.array([U1_050]),nodevec) * np.exp(B_g_A1[k])), label='PLACM', marker='^', markersize=4, ls=':', color='green')
        ax3.plot(np.array([U1_050,U1_050]), np.array([0,np.max([np.exp(-I_S(m,C_deep,np.array([U1_050]),nodevec) * np.exp(B_g_deep1[k])), np.exp(-I_S(m,C_L,np.array([U1_050]),nodevec) * np.exp(B_g_L1[k])), np.exp(-I_S(m,C_A,np.array([U1_050]),nodevec) * np.exp(B_g_A1[k]))])], dtype='float32'), color='k', linestyle='--')
        ax3.legend(loc='best', fontsize=6)
        ax3.set_title(r'$\Delta=1, 50^{\rm{th}}$', fontsize=10) # Set title and size
        fig3.savefig('fig1_50.jpeg', dpi=400, bbox_inches='tight')
    else:
        ax3.plot(U1_075, np.exp(-I_S(m,C_deep,np.array([U1_075]),nodevec) * np.exp(B_g_deep1[k])), label='DPLCM', marker='s', markersize=4, ls='-', color='blue')
        ax3.plot(U1_075, np.exp(-I_S(m,C_L,np.array([U1_075]),nodevec) * np.exp(B_g_L1[k])), label='CPH', marker='o', markersize=4, ls='--', color='orange')
        ax3.plot(U1_075, np.exp(-I_S(m,C_A,np.array([U1_075]),nodevec) * np.exp(B_g_A1[k])), label='PLACM', marker='^', markersize=4, ls=':', color='green')
        ax3.plot(np.array([U1_075,U1_075]), np.array([0,np.max([np.exp(-I_S(m,C_deep,np.array([U1_075]),nodevec) * np.exp(B_g_deep1[k])), np.exp(-I_S(m,C_L,np.array([U1_075]),nodevec) * np.exp(B_g_L1[k])), np.exp(-I_S(m,C_A,np.array([U1_075]),nodevec) * np.exp(B_g_A1[k]))])], dtype='float32'), color='k', linestyle='--')
        ax3.legend(loc='best', fontsize=6)
        ax3.set_title(r'$\Delta=1, 75^{\rm{th}}$', fontsize=10) # Set title and size
        fig3.savefig('fig1_75.jpeg', dpi=400, bbox_inches='tight')

for j in range(3):
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(1, 1, 1)
    ax4.set_xlabel("t",fontsize=8)  
    ax4.set_ylabel(r'$\hat{S}(t)$',fontsize=8) 
    ax4.tick_params(axis='both',labelsize=6)
    ax4.xaxis.set_ticks_position('bottom')
    ax4.spines['bottom'].set_position(('data',0))
    ax4.yaxis.set_ticks_position('left')
    ax4.spines['left'].set_position(('data',0))
    ax4.grid(True)
   
    St_deep0 = np.exp(-I_S(m,C_deep,U0_value,nodevec) * np.exp(B_g_deep0[j]))
    St_deep0_L = np.exp(-I_S(m,C_L,U0_value,nodevec) * np.exp(B_g_L0[j]))
    St_deep0_A = np.exp(-I_S(m,C_A,U0_value,nodevec) * np.exp(B_g_A0[j]))
   
    ax4.plot(U0_value, St_deep0, color='blue', linestyle='-')
    ax4.plot(U0_value, St_deep0_L, color='orange', linestyle='--')
    ax4.plot(U0_value, St_deep0_A, color='green', linestyle=':')
    if (j==0):
        ax4.plot(U0_025, np.exp(-I_S(m,C_deep,np.array([U0_025]),nodevec) * np.exp(B_g_deep0[j])), label='DPLCM', marker='s', markersize=4, ls='-', color='blue')
        ax4.plot(U0_025, np.exp(-I_S(m,C_L,np.array([U0_025]),nodevec) * np.exp(B_g_L0[j])),label='CPH', marker='o', markersize=4, ls= '--', color='orange')
        ax4.plot(U0_025, np.exp(-I_S(m,C_A,np.array([U0_025]),nodevec) * np.exp(B_g_A0[j])), label='PLACM', marker='^', markersize=4, ls=':', color='green')
        ax4.plot(np.array([U0_025,U0_025]), np.array([0,np.max([np.exp(-I_S(m,C_deep,np.array([U0_025]),nodevec) * np.exp(B_g_deep0[j])), np.exp(-I_S(m,C_L,np.array([U0_025]),nodevec) * np.exp(B_g_L0[j])), np.exp(-I_S(m,C_A,np.array([U0_025]),nodevec) * np.exp(B_g_A0[j]))])], dtype='float32'), color='k', linestyle='--')
        ax4.legend(loc='best', fontsize=6)
        ax4.set_title(r'$\Delta=0, 25^{\rm{th}}$', fontsize=10) # Set title and size
        fig4.savefig('fig0_25.jpeg', dpi=400, bbox_inches='tight')
    elif (j==1):
        ax4.plot(U0_050, np.exp(-I_S(m,C_deep,np.array([U0_050]),nodevec) * np.exp(B_g_deep0[j])), label='DPLCM', marker='s', markersize=4, ls='-', color='blue')
        ax4.plot(U0_050, np.exp(-I_S(m,C_L,np.array([U0_050]),nodevec) * np.exp(B_g_L0[j])), label='CPH', marker='o', markersize=4, ls='--', color='orange')
        ax4.plot(U0_050, np.exp(-I_S(m,C_A,np.array([U0_050]),nodevec) * np.exp(B_g_A0[j])), label='PLACM', marker='^', markersize=4, ls=':', color='green')
        ax4.plot(np.array([U0_050,U0_050]), np.array([0,np.max([np.exp(-I_S(m,C_deep,np.array([U0_050]),nodevec) * np.exp(B_g_deep0[j])), np.exp(-I_S(m,C_L,np.array([U0_050]),nodevec) * np.exp(B_g_L0[j])), np.exp(-I_S(m,C_A,np.array([U0_050]),nodevec) * np.exp(B_g_A0[j]))])], dtype='float32'), color='k', linestyle='--')
        ax4.legend(loc='best', fontsize=6)
        ax4.set_title(r'$\Delta=0, 50^{\rm{th}}$', fontsize=10) # Set title and size
        fig4.savefig('fig0_50.jpeg', dpi=400, bbox_inches='tight')
    else:
        ax4.plot(U0_075, np.exp(-I_S(m,C_deep,np.array([U0_075]),nodevec) * np.exp(B_g_deep0[j])), label='DPLCM', marker='s', markersize=4, ls='-', color='blue')
        ax4.plot(U0_075, np.exp(-I_S(m,C_L,np.array([U0_075]),nodevec) * np.exp(B_g_L0[j])), label='CPH', marker='o', markersize=4, ls='--', color='orange')
        ax4.plot(U0_075, np.exp(-I_S(m,C_A,np.array([U0_075]),nodevec) * np.exp(B_g_A0[j])), label='PLACM', marker='^', markersize=4, ls=':', color='green')
        ax4.plot(np.array([U0_075,U0_075]), np.array([0,np.max([np.exp(-I_S(m,C_deep,np.array([U0_075]),nodevec) * np.exp(B_g_deep0[j])), np.exp(-I_S(m,C_L,np.array([U0_075]),nodevec) * np.exp(B_g_L0[j])), np.exp(-I_S(m,C_A,np.array([U0_075]),nodevec) * np.exp(B_g_A0[j]))])], dtype='float32'), color='k', linestyle='--')
        ax4.legend(loc='best', fontsize=6)
        ax4.set_title(r'$\Delta=0, 75^{\rm{th}}$', fontsize=10) # Set title and size
        fig4.savefig('fig0_75.jpeg', dpi=400, bbox_inches='tight')

fig1.savefig('fig1.jpeg', dpi=400, bbox_inches='tight')
fig2.savefig('fig2.jpeg', dpi=400, bbox_inches='tight')
Results = [1-np.mean(De), 1-np.mean(De_R_test), 1-np.mean(S_t_D < 0.5), 1-np.mean(S_t_L < 0.5), 1-np.mean(S_t_A < 0.5), 1-np.mean(np.abs((S_t_D < 0.5)*1-De_R_test)), 1-np.mean(np.abs((S_t_L < 0.5)*1-De_R_test)), 1-np.mean(np.abs((S_t_A < 0.5)*1-De_R_test)), np.sum(De_R_test==0), np.sum(De_R_test==1), np.sum(S_t_D[De_R_test==0]>0.5), np.sum(S_t_D[De_R_test==1]<0.5), np.sum(S_t_L[De_R_test==0]>0.5), np.sum(S_t_L[De_R_test==1]<0.5), np.sum(S_t_A[De_R_test==0]>0.5), np.sum(S_t_A[De_R_test==1]<0.5)]

dic = {"tot_right_rate": Results[0], "test_right_rate": Results[1],"deep_right_r": Results[2], "L_right_r": Results[3], "A_right_r": Results[4], "deep_true": Results[5], "L_true": Results[6], "A_true": Results[7]}
Result = pd.DataFrame(dic,index=[0])
Result.to_csv('Result.csv')
dic1 = {"num_right_test": Results[8], "num_left_test": Results[9], "deep_right_num": Results[10], "deep_left_num": Results[11], "L_right_num": Results[12], "L_left_num": Results[13], "A_right_num": Results[14], "A_left_num": Results[15]}
Result1 = pd.DataFrame(dic1,index=[0])
Result1.to_csv('Result1.csv')
