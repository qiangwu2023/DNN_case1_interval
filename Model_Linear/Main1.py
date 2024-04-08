
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from data_generator import generate_case_1
from iteration_deep import Est_deep
from I_spline import I_S
from Least_FD import LFD
from iteration_linear import Est_linear
from iteration_additive import Est_additive


def set_seed(seed):
    np.random.seed(seed) 
    torch.manual_seed(seed) 

set_seed(1)
tau = 10
p = 3 
Set_n = np.array([500, 1000, 2000])
corr = 0.5
n_layer = 3
n_node = 50 
n_epoch = 200
Set_lr = np.array([2.6e-4, 3e-4, 4e-4]) 
Beta = 1 

node_D = np.array([45, 30, 33]) 
lr_D = np.array([5e-4, 3e-4, 4e-4])

node_L = np.array([35, 30, 33])
lr_L = np.array([5e-4, 3e-4, 4e-4])

node_A = np.array([45, 40, 40]) 
lr_A = np.array([5e-4, 4e-4, 4e-4])

B = 200

test_data = generate_case_1(200, corr, Beta)
X_test = test_data['X']
g_true = test_data['g_X']
dim_x = X_test.shape[0]
u_value = np.array(np.linspace(0, tau, 50), dtype="float32") 
Lambda_true = np.sqrt(u_value)/5 
m = 10 
nodevec = np.array(np.linspace(0, tau, m+2), dtype="float32")

m0 = 4
nodevec0 = np.array(np.linspace(0, 2, m0+2), dtype="float32")

Markers = np.array(['s','o','^'])
lines = np.array([':','--','-.'])

fig1 = plt.figure()
ax1_1 = fig1.add_subplot(1, 3, 1)
plt.ylim(-2,2)
ax1_1.set_title("Case 1, n=500",fontsize=10)
ax1_1.set_xlabel("Predictor",fontsize=8) 
ax1_1.set_ylabel("Error",fontsize=8)
ax1_1.tick_params(axis='both',labelsize=6)

ax1_2 = fig1.add_subplot(1, 3, 2)
plt.ylim(-2,2)
ax1_2.set_title("Case 1, n=1000",fontsize=10) 
ax1_2.set_xlabel("Predictor",fontsize=8) 
# ax1_2.set_ylabel("Error",fontsize=8) 
ax1_2.tick_params(axis='both',labelsize=6)
plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.15)

ax1_3 = fig1.add_subplot(1, 3, 3)
plt.ylim(-2,2) 
ax1_3.set_title("Case 1, n=2000",fontsize=10) 
ax1_3.set_xlabel("Predictor",fontsize=8)
# ax1_3.set_ylabel("Error",fontsize=8) 
ax1_3.tick_params(axis='both',labelsize=6) 
plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.15)

fig2 = plt.figure()
ax2_1 = fig2.add_subplot(1, 3, 1)
plt.ylim(0,1)
ax2_1.set_title("Case 1, n=500", fontsize=10) 
ax2_1.set_xlabel("Time",fontsize=8) 
ax2_1.set_ylabel("Cumulative hazard function",fontsize=8) 
ax2_1.tick_params(axis='both',labelsize=6) 
ax2_1.plot(u_value, Lambda_true, color='k', label='True')
ax2_1.legend(loc='upper left', fontsize=6) 

ax2_2 = fig2.add_subplot(1, 3, 2)
plt.ylim(0,1) 
ax2_2.set_title("Case 1, n=1000", fontsize=10) 
ax2_2.set_xlabel("Time",fontsize=8) 
# ax2_2.set_ylabel("Cumulative hazard function",fontsize=8)
ax2_2.tick_params(axis='both',labelsize=6) 
ax2_2.plot(u_value, Lambda_true, color='k', label='True')
ax2_2.legend(loc='upper left', fontsize=6) 
plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.15)

ax2_3 = fig2.add_subplot(1, 3, 3)
plt.ylim(0,1) 
ax2_3.set_title("Case 1, n=2000", fontsize=10) 
ax2_3.set_xlabel("Time",fontsize=8) 
# ax2_3.set_ylabel("Cumulative hazard function",fontsize=8) 
ax2_3.tick_params(axis='both',labelsize=6) 
ax2_3.plot(u_value, Lambda_true, color='k', label='True')
ax2_3.legend(loc='upper left', fontsize=6) 
plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.15)

Bias_deep = []; Sse_deep = []; Ese_deep = []; Cp_deep = []; Re_deep = []; G_deep_sd = []
Bias_L = []; Sse_L = []; Ese_L = []; Cp_L = []; Re_L = []; G_L_sd = []
Bias_A = []; Sse_A = []; Ese_A = []; Cp_A = []; Re_A = []; G_A_sd = []
for i in range(len(Set_n)):
    n = Set_n[i]
    n_lr = Set_lr[i]
    G_test_deep = []; C_deep=[]; beta_deep = []; Info_deep = []; re_deep = []
    G_test_L = []; C_L = []; beta_L = []; Info_L = []; re_L = []
    G_test_A = []; C_A = []; beta_A = []; Info_A = []; re_A = []
    for b in range(B):
        print('n=', n, 'b=', b)
        set_seed(12 + b)
        c0 = np.array(0.1*np.ones(m+p), dtype="float32") 
        Beta0 = np.array(0, dtype='float32')
        train_data = generate_case_1(n, corr, Beta)
        Z_train = train_data['Z']
        U_train = train_data['U']
        De_train = train_data['De']
        g_train = train_data['g_X']

        Est_hat = Est_deep(train_data=train_data,X_test=X_test,Beta=Beta,Beta0=Beta0,n_layer=n_layer,n_node=n_node,n_lr=n_lr,n_epoch=n_epoch,nodevec=nodevec,m=m,c0=c0)
        G_test_deep.append(Est_hat['g_test']) 
        re_deep.append(np.sqrt(np.mean((Est_hat['g_test']-np.mean(Est_hat['g_test'])-g_true)**2)/np.mean(g_true**2)))
        C_deep.append(Est_hat['c']) 
        a_b_deep = LFD(train_data,I_S(m,Est_hat['c'],U_train,nodevec),Est_hat['g_train'],Est_hat['Beta'],n_layer,n_node=node_D[i],n_lr=lr_D[i],n_epoch=200)
        # Calculate information matrix I(beta_0)
        h_v_deep = I_S(m,Est_hat['c'],U_train,nodevec) * np.exp(Z_train*Est_hat['Beta'] + Est_hat['g_train'])
        Q_y_deep = h_v_deep * (De_train * np.exp(-h_v_deep)/(1-np.exp(-h_v_deep)) - (1-De_train))
        beta_deep.append(Est_hat['Beta'])
        Info_deep.append(np.mean(Q_y_deep**2 * (Z_train-a_b_deep)**2))

        Est_L = Est_linear(train_data,X_test,Beta0,nodevec,m,c0)
        G_test_L.append(Est_L['g_test'])
        re_L.append(np.sqrt(np.mean((Est_L['g_test']-np.mean(Est_L['g_test'])-g_true)**2)/np.mean(g_true**2)))
        C_L.append(Est_L['c']) # Calculate the spline function value at u_value according to the estimated parameters
        a_b_L = LFD(train_data,I_S(m,Est_L['c'],U_train,nodevec),Est_L['g_train'],Est_L['Beta'],n_layer,n_node=node_L[i],n_lr=lr_L[i],n_epoch=200)
        h_v_L = I_S(m,Est_L['c'],U_train,nodevec) * np.exp(Z_train*Est_L['Beta'] + Est_L['g_train'])
        Q_y_L = h_v_L * (De_train * np.exp(-h_v_L)/(1-np.exp(-h_v_L)) - (1-De_train)) 
        beta_L.append(Est_L['Beta'])
        Info_L.append(np.mean(Q_y_L**2 * (Z_train-a_b_L)**2))
        Est_A = Est_additive(train_data,X_test,Beta0,nodevec,m,c0,m0,nodevec0)
        G_test_A.append(Est_A['g_test'])
        re_A.append(np.sqrt(np.mean((Est_A['g_test']-np.mean(Est_A['g_test'])-g_true)**2)/np.mean(g_true**2)))
        C_A.append(Est_A['c'])  # Calculate the spline function value at u_value according to the estimated parameters
        a_b_A = LFD(train_data,I_S(m,Est_A['c'],U_train,nodevec),Est_A['g_train'],Est_A['Beta'],n_layer,n_node=node_A[i],n_lr=lr_A[i],n_epoch=200)
        h_v_A = I_S(m,Est_A['c'],U_train,nodevec) * np.exp(Z_train*Est_A['Beta'] + Est_A['g_train'])
        Q_y_A = h_v_A * (De_train * np.exp(-h_v_A)/(1-np.exp(-h_v_A)) - (1-De_train)) 
        beta_A.append(Est_A['Beta'])
        Info_A.append(np.mean(Q_y_A**2 * (Z_train-a_b_A)**2))
        
    Error_deep = np.mean(np.array(G_test_deep), axis=0) - g_true
    if (i == 0):
        ax1_1.scatter(np.arange(dim_x), Error_deep, s=4, marker='o', label='DPLCM')
        ax1_1.legend(loc='upper left', fontsize=4)
        ax2_1.plot(u_value, I_S(m,np.mean(np.array(C_deep), axis=0),u_value,nodevec), label='DPLCM', linestyle='--')
        ax2_1.legend(loc='upper left', fontsize=6)
    elif(i == 1):
        ax1_2.scatter(np.arange(dim_x), Error_deep, s=4, marker='o', label='DPLCM')
        ax1_2.legend(loc='upper left', fontsize=4)
        ax2_2.plot(u_value, I_S(m,np.mean(np.array(C_deep), axis=0),u_value,nodevec), label='DPLCM', linestyle='--')
        ax2_2.legend(loc='upper left', fontsize=6)
    else:
        ax1_3.scatter(np.arange(dim_x), Error_deep, s=4, marker='o', label='DPLCM')
        ax1_3.legend(loc='upper left', fontsize=4)
        ax2_3.plot(u_value, I_S(m,np.mean(np.array(C_deep), axis=0),u_value,nodevec), label='DPLCM', linestyle='--')
        ax2_3.legend(loc='upper left', fontsize=6)
    Bias_deep.append(np.mean(np.array(beta_deep))-Beta)
    Sse_deep.append(np.sqrt(np.mean((np.array(beta_deep)-np.mean(np.array(beta_deep)))**2)))
    Ese_deep.append(1/np.sqrt(n*np.mean(np.array(Info_deep))))
    Cp_deep.append(np.mean((np.array(beta_deep)-1.96/np.sqrt(n*np.mean(np.array(Info_deep)))<=Beta)*(Beta<=np.array(beta_deep)+1.96/np.sqrt(n*np.mean(np.array(Info_deep))))))
    Re_deep.append(np.mean(re_deep))
    G_deep_sd.append(np.sqrt(np.mean((re_deep-np.mean(re_deep))**2)))
    
    Error_L = np.mean(np.array(G_test_L), axis=0) - g_true
    if (i == 0):
        ax1_1.scatter(np.arange(dim_x), Error_L, s=4, marker='s', label='CPH')
        ax1_1.legend(loc='upper left', fontsize=4)
        ax2_1.plot(u_value, I_S(m,np.mean(np.array(C_L), axis=0),u_value,nodevec), label='CPH', linestyle=':')
        ax2_1.legend(loc='upper left', fontsize=6)
    elif (i == 1):
        ax1_2.scatter(np.arange(dim_x), Error_L, s=4, marker='s', label='CPH')
        ax1_2.legend(loc='upper left', fontsize=4)
        ax2_2.plot(u_value, I_S(m,np.mean(np.array(C_L), axis=0),u_value,nodevec), label='CPH', linestyle=':')
        ax2_2.legend(loc='upper left', fontsize=6)
    else:
        ax1_3.scatter(np.arange(dim_x), Error_L, s=4, marker='s', label='CPH')
        ax1_3.legend(loc='upper left', fontsize=4)
        ax2_3.plot(u_value, I_S(m,np.mean(np.array(C_L), axis=0),u_value,nodevec), label='CPH', linestyle=':')
        ax2_3.legend(loc='upper left', fontsize=6)

    Bias_L.append(np.mean(np.array(beta_L))-Beta)
    Sse_L.append(np.sqrt(np.mean((np.array(beta_L)-np.mean(np.array(beta_L)))**2)))
    Ese_L.append(1/np.sqrt(n*np.mean(np.array(Info_L))))
    Cp_L.append(np.mean((np.array(beta_L)-1.96/np.sqrt(n*np.mean(np.array(Info_L)))<=Beta)*(Beta<=np.array(beta_L)+1.96/np.sqrt(n*np.mean(np.array(Info_L))))))
    Re_L.append(np.mean(re_L))
    G_L_sd.append(np.sqrt(np.mean((re_L-np.mean(re_L))**2)))
    

    Error_A = np.mean(np.array(G_test_A), axis=0) - g_true
    if (i == 0):
        ax1_1.scatter(np.arange(dim_x), Error_A, s=4, marker='^', label='PLACM')
        ax1_1.legend(loc='upper left', fontsize=4)
        ax2_1.plot(u_value, I_S(m,np.mean(np.array(C_A), axis=0),u_value,nodevec), label='PLACM', linestyle='-.')
        ax2_1.legend(loc='upper left', fontsize=6)
    elif (i == 1):
        ax1_2.scatter(np.arange(dim_x), Error_A, s=4, marker='^', label='PLACM')
        ax1_2.legend(loc='upper left', fontsize=4)
        ax2_2.plot(u_value, I_S(m,np.mean(np.array(C_A), axis=0),u_value,nodevec), label='PLACM', linestyle='-.')
        ax2_2.legend(loc='upper left', fontsize=6)
    else:
        ax1_3.scatter(np.arange(dim_x), Error_A, s=4, marker='^', label='PLACM')
        ax1_3.legend(loc='upper left', fontsize=4)
        ax2_3.plot(u_value, I_S(m,np.mean(np.array(C_A), axis=0),u_value,nodevec), label='PLACM', linestyle='-.')
        ax2_3.legend(loc='upper left', fontsize=6)
    Bias_A.append(np.mean(np.array(beta_A))-Beta)
    Sse_A.append(np.sqrt(np.mean((np.array(beta_A)-np.mean(np.array(beta_A)))**2)))
    Ese_A.append(1/np.sqrt(n*np.mean(np.array(Info_A))))
    Cp_A.append(np.mean((np.array(beta_A)-1.96/np.sqrt(n*np.mean(np.array(Info_A)))<=Beta)*(Beta<=np.array(beta_A)+1.96/np.sqrt(n*np.mean(np.array(Info_A))))))
    Re_A.append(np.mean(re_A))
    G_A_sd.append(np.sqrt(np.mean((re_A-np.mean(re_A))**2)))

fig1.savefig('fig_g_1_t.jpeg', dpi=400, bbox_inches='tight')
fig2.savefig('fig_Lambda_1_t.jpeg', dpi=400, bbox_inches='tight')

dic_error = {"n": Set_n, "Bias_deep": np.array(Bias_deep), "SSE_deep": np.array(Sse_deep), "ESE_deep": np.array(Ese_deep), "CP_deep": np.array(Cp_deep), "Bias_L": np.array(Bias_L),  "SSE_L": np.array(Sse_L), "ESE_L": np.array(Ese_L), "CP_L": np.array(Cp_L), "Bias_A": np.array(Bias_A), "SSE_A": np.array(Sse_A), "ESE_A": np.array(Ese_A), "CP_A": np.array(Cp_A)}
result_error = pd.DataFrame(dic_error)
result_error.to_csv('result_error_linear_t.csv')

dic_re = {"n": Set_n, "Re_deep": np.array(Re_deep), "G_deep_sd": np.array(G_deep_sd), "Re_L": np.array(Re_L), "G_L_sd": np.array(G_L_sd), "Re_A": np.array(Re_A), "G_A_sd": np.array(G_A_sd)}
result_re = pd.DataFrame(dic_re)
result_re.to_csv('result_re_linear_t.csv')

