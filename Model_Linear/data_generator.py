"""
Created on Wednesday December 20 2023
@author: qiangwu
"""
import numpy as np
import numpy.random as ndm

def uniform_data(n, u1, u2):
    a = ndm.rand(n)
    b = (u2 - u1) * a + u1
    return b

def generate_case_1(n, corr, Beta):
    Z = ndm.binomial(1, 0.5, n)
    mean = np.zeros(5)
    cov = np.identity(5)*(1-corr) + np.ones((5, 5))*corr
    def multivariatet(mu,Sigma,N,M):
        d = len(Sigma)
        g = np.tile(np.random.gamma(N/2,1/2,M),(d,1)).T
        Z = np.random.multivariate_normal(np.zeros(d),Sigma,M)
        return mu + Z/np.sqrt(g/N)
    
    X = multivariatet(mean,cov,5,n)
    X = np.clip(X, 0, 2)
    g_X = X[:,0]/2 + X[:,1]/3 + X[:,2]/4 + X[:,3]/5 + X[:,4]/6 - 0.94
    Y = ndm.rand(n)
    T = (-5 * np.log(Y) * np.exp(-Z * Beta - g_X)) ** 2
    U = uniform_data(n, 0, 10)
    De = (T <= U)
    return {
        'Z': np.array(Z, dtype='float32'),
        'X': np.array(X, dtype='float32'),
        'T': np.array(T, dtype='float32'),
        'U': np.array(U, dtype='float32'),
        'De': np.array(De, dtype='float32'),
        'g_X': np.array(g_X, dtype='float32')
    }
