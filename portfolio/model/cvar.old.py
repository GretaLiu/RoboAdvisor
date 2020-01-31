import pandas as pd
import pandas as pd
import numpy as np
import cvxpy as cvx
from numpy.linalg import inv
from scipy.stats.mstats import gmean
from scipy.linalg import cholesky, block_diag
from statsmodels.stats.moment_helpers import cov2corr
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pylab as pylab
def ReturnFactors():
    prices_org = pd.read_csv("adjClose.csv")
    factor_org = pd.read_csv("factors.csv")

    # make sure fama-french factors and prices have the same length of datas
    prices_org['Date'] = prices_org['date'].apply(lambda x: x[:7])
    prices = pd.merge(prices_org, factor_org['Date'], on='Date', how='inner')

    # save dates
    dates = pd.to_datetime(prices['date'])
    dates = dates[1:]

    # remove dates in data
    prices = prices.drop(['date','Date'], axis=1)
    factors = factor_org.drop('Date', axis=1)

    test_yr = 12
    prices = prices[:-test_yr]
    factors = factors[:-test_yr]
    test_prices = prices[-test_yr:]

    # organize data into riskfree, factors, and returns
    factors = factors[1:]
    factors = factors.apply(lambda x: x/100)
    riskfree = factors.loc[:,'RF']
    riskfree = riskfree.reset_index(drop=True)
    factors = factors.drop('RF', axis=1)
    returns = prices[:-1] / prices[1:].values - 1
    returns = returns.apply(lambda x: x-riskfree)
    return returns, factors
def generate(
    mean_returns: pd.DataFrame,
    variance_covariance: pd.DataFrame,
    risk_aversion: float,
    shortable: bool,
    previous_portfolio_weights: pd.Series,
    transaction_cost,
    holding_cost,
):
    new_weights = pd.Series()
    this_transaction_cost = 0
    num_asset = len(mu)
    rho = cov2corr(Q)
    nPaths = 400
    L = cholesky(rho, lower=True)
    T = 12
    N = 1
    dt = T/N
    confidence_level = 0.95
    S = np.array([[[0.0 for k in range(nPaths)] for j in range(N+1)] for i in range(num_asset)])#Matrix of simulated price path
    S[:, 0, :] = 100
    currentPrices=[]
    for row in prices.tail(1).values:
        for v in row:
            currentPrices.append(v)

    for i in range(num_asset):
        for j in range(nPaths):
            S[i,0,j] = currentPrices[i]
            
    for i in range(nPaths):
        for j in range(N):
            xi = np.dot(L,np.random.randn(num_asset, 1))
            for k in range(num_asset):
                S[k, j+1, i] = S[k, j, i] * np.exp( ( mu[k] - 0.5 * Q[k, k] ) * dt \
                                + np.sqrt(Q[k, k]) * np.sqrt(dt) * xi[k] )

    # returns_sample n_asset * nPeriod * nPaths
    returns_sample = np.array([[0.0 for k in range(nPaths)] for j in range(num_asset)])

    for i in range(nPaths):
        for j in range(num_asset):
            returns_sample[j, i] = S[j,-1,i] / S[j, 0, i] - 1
    #construct f
    f=np.zeros(num_asset + nPaths + 1)
    f[:nPaths]=1 / ((1 - confidence_level) * nPaths)
    f[nPaths : num_asset + nPaths] = 0
    f[-1] = 1
    #construct A
    A= np.array([[0.0 for k in range(nPaths + num_asset + 1)] for j in range(2*nPaths)])
    A[:nPaths, :nPaths] = -1 * np.eye(nPaths)
    A[nPaths:(2 * nPaths), :nPaths]= -1 * np.eye(nPaths)
    A[nPaths:(2 * nPaths), -1] = -1
    for i in range((nPaths), (2 * nPaths)):
            A[i,nPaths:(nPaths + num_asset)] = -(returns_sample[:, i - nPaths])
    temp= -1 * np.eye(nPaths + num_asset + 1)
    A=np.concatenate((A, temp), axis=0)
    #construct Aeq
    Aeq = np.array([[0.0 for k in range(nPaths + num_asset + 1)] for j in range(1)])
    Aeq[0 , nPaths:(nPaths + num_asset)] = 1

    #construct beq
    beq = np.array([1])

    #construct b
    b = np.array([0.0 for k in range(3 * nPaths + num_asset +1)] )
    b[(2*nPaths):(3*nPaths)] = 1000000000000
    b[(3*nPaths) : num_asset + 3*nPaths] = 0
    b[-1] = 1000000000000

    #temp = ([[0.0 for k in range(nPaths + num_asset + 1)] for j in range(nPaths + num_asset + 1)])
    temp= -1 * np.eye(nPaths + num_asset + 1)
    A=np.concatenate((A, temp), axis=0)

    res = linprog(c=f, A_ub=A, b_ub=b,A_eq=Aeq,b_eq=beq)
    print(res.x[nPaths:nPaths+num_asset])
    #x = cvx.Variable(num_asset)
    #prob = cvx.Problem(cvx.Minimize(f.T@x),[A@x<=b,Aeq@x==beq])
    #prob.solve()
    return new_weights, this_transaction_cost
def fama_french(returns, factors):
    
    # Slove for a and beta
    B = np.ones(len(factors))
    B = np.c_[B, factors.to_numpy()]
    
    loadings = np.dot(np.dot(inv(np.dot(B.T,B)), B.T), returns.to_numpy())
        
    #generate n_stock x 1 vector of alphas
    A = loadings[0,:]
        
    #generate n_factor x n_stock matrix of betas
    V = loadings[1:,:]
    
    #factor expected returns, f_bar, n_factor x 1 
    f_bar = gmean(B[:,1:]+1)-1
    f_bar = f_bar.T
        
    #factor covariance matrix, F, n_factor x n_factor
    F = np.cov(factors.T)
    
    #Regression residuals, epsilon
    epsilon = returns.to_numpy() - np.dot(B, loadings)
        
    #Diagonal n_stock x n_stock matrix of residual variance
    D = np.diag(np.diag(np.cov(epsilon.T)))
        
    #1 x n_stock vector of asset exp. returns
    mu = np.dot(V.T, f_bar) + A
    mu = mu.T
    
    #n_stock x n_stock asset covariance matrix
    Q = np.dot(np.dot(V.T,F),V) + D
    
    return [mu, Q]

[mu, Q] = fama_french(returns, factors)

def test():
    # make sure generate is working with a tiny example
    pass


if __name__ == "__main__":
    test()
