import numpy as np
from scipy import stats


def E_step(x, theta):

    Theta_temp = [1-theta["phi_1"]-theta["phi_2"],theta["phi_1"], theta["phi_2"]]

    P_1 = stats.multivariate_normal(theta["mu_0"], theta["st_0"]).pdf(x)
    
    P_2 = stats.multivariate_normal(theta["mu_1"], theta["st_1"]).pdf(x)
    
    P_3 = stats.multivariate_normal.pdf(x,theta["mu_2"], theta["st_2"])
    
    num = np.multiply(np.array(Theta_temp),np.array([P_1,P_2,P_3]).T)
    den = np.sum(num,axis=1)

    return np.log(den),np.array([num[i]/den[i] for i in range(len(x))])


def M_step(x, theta):
    total_samp = x.shape[0]

    l,PI = E_step(x, theta)
    PI = PI.T


    mu_k = np.array(PI@x)

    sum_PI = np.sum(PI,axis=1)
    
    mu_k = np.array([mu_k[i]/sum_PI[i] for i in range(3)])
    
    diff = [x - mu_k[0],x - mu_k[1],x-mu_k[2]]
    
    
    phi = sum_PI/total_samp

    
    sigma0 = ((PI[0])*(diff[0].T)).dot(diff[0]) / sum_PI[0]
    
    sigma1 = ((PI[1])*(diff[1].T)).dot(diff[1]) / sum_PI[1]
    
    sigma2 = ((PI[2])*(diff[2].T)).dot(diff[2]) / sum_PI[2]
    
    
    theta = {'phi_1': phi[1],'phi_2':phi[2], 'mu_0': mu_k[0], 'mu_1': mu_k[1],'mu_2':mu_k[2], 'st_0': sigma0, 'st_1': sigma1,'st_2':sigma2}
    
    return theta
    
    
    