# Examples

from LAMM_2S import LAMM_2S
import numpy as np
import pandas as pd

np.random.seed(0)

n = 100
p = 200
s = 10
beta0 = np.concatenate((np.ones((s)), np.zeros((p - s))))

##################################
# Cox Proportional Hazards Model
##################################

# Generate survival data
def generate_survival_data(beta, n, p):
    """
    Input:
    beta: numpy.array
    n: int, number of samples
    p: int, number of variables
    
    Output:
    X: numpy.array, shape = (n, p), covariates
    y: numpy.array, shape = (n, 2), response
       y[:, 0] is the observed follow-up time = min(Failure Time, Censoring Time)
       y[:, 1] is the censoring indicator I(Failure Time < Censoring Time)
    """
    def censordata(x,beta):
        """ 
        Censoring record for one single sample.
        Assumption:
        The failure time (lifetime) follows an exponential distribution with mean 1 / Hazards.
        The censoring time follows an exponential distribution with mean U * Hazards,
        where U is a uniformly distributed variable in [2, 3].
        """
        h = np.exp(np.dot(x,beta)) # Hazards 
        lifetime = np.random.exponential(1.0 / h, 1) # Lifetime ~ exp(h) with mean 1 / h
        U = np.random.uniform(2, 3, 1) # Uniform random number U[2, 3]
        ctime = np.random.exponential(U * h, 1) # Censoring time ~ exp with mean U * h
        if ctime >= lifetime:
            return min(lifetime, ctime), 1
        else:
            return min(lifetime, ctime), 0
    
    X = np.random.normal(0, 1, n * p).reshape(n, p)
    
    obs = np.zeros((n, 2))
    for i in range(n):
        obs[i, :] = censordata(X[i, :], beta)
    
    # Sort the samples according to Z = min(Censoring time, Failure time)
    samples = pd.DataFrame(np.concatenate((obs, X), axis=1))
    samples = samples.rename(index=str, columns={0: "time", 1: "T<=C"})
    samples = samples.sort_values(by=['time'])
    samples = np.array(samples, dtype='float')
    
    X = samples[:, 2:]
    y = samples[:, :2]
    return X, y

X_cox, y_cox = generate_survival_data(beta0, n, p)
penalty = 'SCAD'
lam = np.sqrt(np.log(p) / n) * 0.5

betahat_cox = LAMM_2S(X_cox, y_cox, penalty, lam, 'survival')


#########
# OLS
#########

X_ols = np.random.normal(0, 1, n * p).reshape(n, p)
y_ols = np.dot(X_ols, beta0) + np.random.normal(scale=0.05, size=n)
penalty = 'SCAD'
lam = np.sqrt(np.log(p) / n) * 0.5

betahat_ols = LAMM_2S(X_ols, y_ols, penalty, lam, 'ols')


##############
# Logistic
##############
X_logistic = np.random.normal(0, 1, n * p).reshape(n, p)
y_logistic = np.random.binomial(n=1, p=1/(1+np.exp(-np.dot(X_logistic, beta0))))
penalty = 'SCAD'
lam = np.sqrt(np.log(p) / n) * 0.5

betahat_logistic = LAMM_2S(X_logistic, y_logistic, penalty, lam, 'logistic')

