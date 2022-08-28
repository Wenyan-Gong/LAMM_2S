import numpy as np

ru = 2

def LAMM_2S(X, y, penalty, lam, model='survival', phi0=0.1, epsilon_c=0.05, epsilon_t=0.05):
    n,p = X.shape
    lam = np.ones((p)) * lam
    beta = np.random.normal(0, 1, p)/100
    #stage 1
    newbeta = LAMM_Stage1(X, y, lam, beta, phi0, epsilon_c, model)
    #stage 2
    newbeta = LAMM_Stage2(X, y, lam, newbeta, phi0, epsilon_t, penalty, model)
    return newbeta

def _loss_survival(X, y, beta):
    """
    Input:
    X: numpy.array, shape = (n, p)
    y: numpy.array, shape = (n, 2)
       y[:, 0] is the observed follow-up time = min(Failure Time, Censoring Time)
       y[:, 1] is the censoring indicator I(Failure Time < Censoring Time)
    beta: numpy.array, shape = (p,)
    """
    # Sort events by time
    idx = np.argsort(y[:, 0])
    X = X[idx, :]
    y = y[idx, :]
    
    n = X.shape[0]
    llh = 0.0
    xbeta = np.dot(X,beta)
    for i in range(n - 1):
        if y[i, 1] == 0: #failure time not observed, skip
            continue
        llh += xbeta[i]
        llh -= np.log(np.sum(np.exp(xbeta[i:])))
    return - llh / n

def _gradient_survival(X, y, beta):
    # Sort events by time
    idx = np.argsort(y[:, 0])
    X = X[idx, :]
    y = y[idx, :]
    
    n, p = X.shape
    g = np.zeros((p)) 
    theta = np.exp(np.dot(X, beta))
    for i in range(n - 1):
        if y[i, 1] == 0: #failure not observed, skip
            continue        
        riskset = X[i:,:]     
        g += X[i,:]
        q1 = np.sum(np.multiply(np.dot(theta[i:].reshape(-1, 1),np.ones((1, p))), riskset), axis=0)   
        q2 = np.sum(theta[i:])
        g -= q1 / q2    
    return - g / n

def _loss_ols(X, y, beta):
    return np.linalg.norm(y - np.dot(X, beta)) ** 2 / y.shape[0]

def _gradient_ols(X, y, beta):
    return 2 * (np.dot(np.dot(X.transpose(), X), beta) - np.dot(X.transpose(), y)) / y.shape[0]

def _loss_logistic(X, y, beta):
    n = y.shape[0]
    l = np.dot(y, np.dot(X, beta))
    for i in range(n):
        l -= np.log(1 + np.exp(np.dot(X[i, :], beta)))
    return - l / n

def _gradient_logistic(X, y, beta):
    n = y.shape[0]
    g = np.dot(y, X)
    for i in range(n):
        g -= X[i, :] / (1 + np.exp(-np.dot(X[i, :], beta)))
    return - g / n

def loss(X, y, beta, model='survival'):
    """
    Input:
    
    X: numpy.array, shape = (n, p)
    y: numpy.array, shape = (n, *)
    beta: numpy.array, shape = (p,)
    model: {'ols', 'logistic', 'survival'}
    
    When model = 'survival', y.shape = (n, 2); otherwise, y.shape = (n, 1).
    When model = 'survival':
        y[:, 0] is the observed follow-up time = min(Failure Time, Censoring Time)
        y[:, 1] is the censoring indicator I(Failure Time < Censoring Time)
    
    Output: float, the loss (negative log-likelihood)
    """
    if model == 'survival':
        return _loss_survival(X, y, beta)
    elif model == 'ols':
        return _loss_ols(X, y, beta)
    elif model == 'logistic':
        return _loss_logistic(X, y, beta)
    else:
        raise ValueError('Specified model is not implemented.')

def gradient(X, y, beta, model='survival'):
    """
    Input:
    
    X: numpy.array, shape = (n, p)
    y: numpy.array, shape = (n, *)
    beta: numpy.array, shape = (p,)
    model: {'ols', 'logistic', 'survival'}
    
    When model = 'survival', y.shape = (n, 2); otherwise, y.shape = (n, 1).
    When model = 'survival':
        y[:, 0] is the observed follow-up time = min(Failure Time, Censoring Time)
        y[:, 1] is the censoring indicator I(Failure Time < Censoring Time)
    
    Output: numpy.array, shape = (p,)
    """
    if model == 'survival':
        return _gradient_survival(X, y, beta)
    elif model == 'ols':
        return _gradient_ols(X, y, beta)
    elif model == 'logistic':
        return _gradient_logistic(X, y, beta)
    else:
        raise ValueError('Specified model is not implemented.')    

def Phi_1(X, y, phi, beta1, beta2, model):
    """
    Majorization function in the first stage of LAMM_2S.
    """
    num = loss(X, y, beta2, model)
    num += np.dot(gradient(X, y, beta2, model), beta1 - beta2) 
    num += float(phi) / 2 * (np.linalg.norm(beta1 - beta2, 2) ** 2)
    return num

def w(X, y, lam, beta, model):
    grad = gradient(X, y, beta, model)
    mmax = 0
    for i in range(beta.shape[0]):
        if beta[i] == 0:
            if abs(grad[i]) - lam[i] > mmax:
                mmax = abs(grad[i]) - lam[i]
        else:
            if abs(grad[i] + np.sign(beta[i]) * lam[i]) > mmax:
                mmax = abs(grad[i] + np.sign(beta[i]) * lam[i])
    return mmax

def SoftThres(x, lam):
    for i in range(x.shape[0]):
        if x[i] != 0:
            x[i] = np.sign(x[i]) * max(abs(x[i]) - lam[i], 0)
    return x

def T(X, y, lam, phi, beta, model):
    return SoftThres(beta - gradient(X, y, beta, model) / phi, lam / phi)

def LAMM_1(X, y, lam, beta, phi0, phi, model):
    """
    One single LAMM iteration in the first stage of LAMM_2S.
    """
    phi = max(phi0, phi / ru)
    while True:
        newbeta = T(X, y, lam, phi, beta, model)
        if loss(X, y, newbeta, model) > Phi_1(X, y, phi, newbeta, beta, model):
            phi = phi * ru
        else:
            break
    return newbeta, phi

def LAMM_Stage1(X, y, lam, beta, phi0, epsilon, model):
    """
    First stage of LAMM_2S.
    """
    phi = phi0 / ru
    while w(X, y, lam, beta, model) > epsilon:
        beta, phi=LAMM_1(X, y, lam, beta, phi0, phi, model)
    return beta

def scadpen(beta, lam, a=3.7):
    """
    SCAD penalty
    
    Input:
    beta: float, co-efficient to be penalized
    lam: float, tuning parameter
    
    Output: float, penalty function value
    """
    pen = 0
    if np.abs(beta) <= lam:
        pen = np.abs(beta) * lam
    elif np.abs(beta) <= lam * a:
        pen=a / (a - 1) * lam * np.abs(beta) - beta ** 2 / 2.0 / (a - 1) - lam ** 2 / 2.0 / (a - 1)
    else:
        pen=(a + 1) / 2.0 * lam ** 2 
    return pen

def scadpend(beta, lam, a=3.7):
    """
    Subgradient of SCAD penalty
    
    Input:
    beta: float, co-efficient to be penalized
    lam: float, tuning parameter
    
    Output: float
    """
    subgradient = 0
    if np.abs(beta) <= lam:
        subgradient = lam * np.sign(beta)
    elif np.abs(beta) <= lam * a:
        subgradient = (a * lam / (a - 1) - np.abs(beta) / (a - 1)) * np.sign(beta)
    return subgradient

def mcppen(beta, lam, gamma=3.0):
    """
    MCP penalty
    
    Input:
    beta: float, co-efficient to be penalized
    lam: float, tuning parameter
    
    Output: float, penalty function value
    """
    if np.abs(beta) <= gamma * lam:
        pen = lam * np.abs(beta) - beta ** 2 / gamma / 2.0
    else:
        pen = gamma / 2.0 * lam ** 2
    return pen

def mcppend(beta, lam, gamma=3.0):
    """
    Subgradient of MCP
    
    Input:
    beta: float, co-efficient to be penalized
    lam: float, tuning parameter
    
    Output: float
    """    
    subgradient = (np.maximum(0, lam - np.abs(beta) / gamma)) * np.sign(beta)
    return subgradient

def loss_tilt(X, y, beta, lam, penalty, model):
    """
    Shifted loss.
    """
    llh_tilt = loss(X, y, beta, model)
    if penalty == 'LASSO':
        llh_tilt = llh_tilt
    elif penalty == 'SCAD':
        llh_tilt += np.sum([scadpen(beta[i], lam[i]) for i in range(beta.shape[0])]) - np.dot(lam, np.abs(beta))
    elif penalty == 'MCP':
        llh_tilt += np.sum([mcppen(beta[i], lam[i]) for i in range(beta.shape[0])]) - np.dot(lam, np.abs(beta))
    else:
        raise ValueError('Specified penalty function is not implemented.')
    return llh_tilt

def gradient_tilt(X, y, beta, lam, penalty, model):
    """
    Gradient of shifted loss.
    """
    g_tilt = gradient(X, y, beta, model)
    if penalty == 'LASSO':
        g_tilt = g_tilt
    elif penalty == 'SCAD':
        g_tilt += np.array([scadpend(beta[i], lam[i]) for i in range(beta.shape[0])]) - np.multiply(lam, np.sign(beta))
    elif penalty == 'MCP':
        g_tilt += np.array([mcppend(beta[i], lam[i]) for i in range(beta.shape[0])]) - np.multiply(lam, np.sign(beta))
    else:
        raise ValueError('Specified penalty function is not implemented.')
    return g_tilt

def T_2(X, y, lam, phi, beta, penalty, model):
    g = gradient_tilt(X, y, beta, lam, penalty, model)
    x = beta - g / phi
    return SoftThres(x, lam / phi)

def Phi_2(X, y, phi, beta1, beta2, lam, penalty, model):
    """
    Majorization function in the second stage of LAMM_2S.
    """
    num = loss_tilt(X, y, beta2, lam, penalty, model)
    num += np.dot(gradient_tilt(X, y, beta2, lam, penalty, model), beta1 - beta2)
    num += float(phi) / 2 * (np.linalg.norm(beta1 - beta2, 2) ** 2)
    return num

def LAMM_2(X, y, lam, beta, phi0, phi, penalty, model):
    """
    One single LAMM iteration in the second stage of LAMM_2S.
    """
    phi = max(phi0, float(phi) / ru)
    while True:
        newbeta = T_2(X, y, lam, phi, beta, penalty, model)
        if loss_tilt(X, y, newbeta, lam, penalty, model) > Phi_2(X, y, phi, newbeta, beta, lam, penalty, model):
            phi = phi * ru
        else:
            break
    return newbeta, phi

def LAMM_Stage2(X, y, lam, beta, phi0, epsilon, penalty, model):
    """
    The second stage of LAMM_2S.
    """
    phi = float(phi0) / ru
    while True:
        newbeta,phi=LAMM_2(X, y, lam, beta, phi0, phi, penalty, model)
        if np.linalg.norm(beta - newbeta, 2) < epsilon:
            break
        beta = newbeta
    return beta