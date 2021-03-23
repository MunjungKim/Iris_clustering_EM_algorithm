import numpy as np
import sklearn
import scipy
from scipy import stats
from sklearn import datasets


def get_random_theta(n):
    x = np.random.normal(0,1,size=(n,n))
    y = np.random.normal(0,1,size=(n,n))
    z = np.random.normal(0,1,size=(n,n))
    
    theta = {'phi_1': np.random.uniform(0, 1),
             'phi_2': np.random.uniform(0, 1),
             
             'mu_0': np.random.normal(0, 1, size=(2,)),
             'mu_1': np.random.normal(0, 1, size=(2,)),
             'mu_2': np.random.normal(0, 1, size=(2,)),
             
             'st_0': np.dot(x,x.transpose()),
             'st_1': np.dot(y,y.transpose()),
             'st_2': np.dot(z,z.transpose())}
                
    return theta