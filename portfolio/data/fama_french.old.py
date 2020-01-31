import pandas as pd
import numpy as np

from numpy.linalg import inv
from scipy.stats.mstats import gmean
from scipy.linalg import cholesky, block_diag
from statsmodels.stats.moment_helpers import cov2corr

import cvxpy as cvx

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pylab as pylab
params = {'legend.fontsize': 10,
          'figure.figsize': (15, 10),
         'axes.labelsize': 10,
         'axes.titlesize': 10,
         'xtick.labelsize': 5,
         'ytick.labelsize': 5}
pylab.rcParams.update(params)

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


## ********************************************
## Fama-French 4 Factor Model
## ********************************************

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


## ********************************************
## stochastic MVO
## ********************************************
    
num_asset = len(mu)
rho = cov2corr(Q)

nPaths = 400
L = cholesky(rho, lower=True)
T = 12
N = 3
dt = T/N

# Because it is in a minimization problem, so minus means reward
reward_per_dollor_surplus = -2
# Because it is in a minimization problem, so positive means punishment
punishment_per_dollor_shortfall = 1

#risk adversion coefficient
risk_weight_coefficient = 1000

#target return
targetRet = np.mean(mu) + 0.01
#targetRet = 0

#short sell allow switch (short sell does not work with stocastic programming :( )
shortSell = 0

#transaction cost
tcost = 0.05
currentPrices = np.array(prices[-1:])

# ######################
# monte carlo simulaton 
# ######################

# Matrix of simulated price paths num_asset * nPeriods+1 * nPaths, set initial price to be $100
S = np.array([[[0.0 for k in range(nPaths)] for j in range(N+1)] for i in range(num_asset)])
S[:, 0, :] = 100

# Generate paths
for i in range(nPaths):
    for j in range(N):
        xi = np.dot(L,np.random.randn(num_asset, 1))
        for k in range(num_asset):
            S[k, j+1, i] = S[k, j, i] * np.exp( ( mu[k] - 0.5 * Q[k, k] ) * dt \
                            + np.sqrt(Q[k, k]) * np.sqrt(dt) * xi[k] )

# returns_sample n_asset * nPeriod * nPaths
returns_sample = np.array([[[0.0 for k in range(nPaths)] for j in range(N)] for i in range(num_asset)])

for i in range(num_asset):
    returns_sample[i] = S[i,:-1,:] / S[i, 1:,:] - 1

'''
Have nPath scenarios and each scenario has a surplus and shortfall
Also has num_asset * N (n_period) weight variables
In total, have 2 * nPath * num_asset * N + num_asset * N (n_period) variables.
First nPath * N variables are surplus variables.
Second nPath * num_asset * N  are shortfall variables.
N-1 transaction cost variables.
The last num_asset * N are weight variables.
'''

# objective function

# linear portion: surplus and shortfall
q = np.zeros(2 * nPaths * N + N + num_asset * N)
q[:nPaths * N] = (1 / nPaths) * reward_per_dollor_surplus
q[nPaths * N : 2 * nPaths * N] = (1 / nPaths) * punishment_per_dollor_shortfall
q[2 * nPaths * N + 1 : 2 * nPaths * N + N] = 1

# quadratic portion: minimize risk
P = np.zeros((2 * nPaths * N + N + num_asset * N, 2 * nPaths * N + N + num_asset * N))
blocks = risk_weight_coefficient * Q
for i in range(N-1):
    blocks = block_diag(blocks, risk_weight_coefficient * Q)

P[2 * nPaths * N + N:, 2 * nPaths * N + N:] = blocks

# Linear equal constrain
Aeq = np.zeros((nPaths * N + 2 * N, 2 * nPaths * N + N + num_asset * N))
blocks = np.eye(nPaths)
for i in range(N-1):
    blocks = block_diag(blocks, np.eye(nPaths))
Aeq[:nPaths * N, :nPaths * N] = -1 * blocks
Aeq[:nPaths * N, nPaths * N : 2 * nPaths * N] = blocks
Aeq[Aeq==0.] = 0.

for i in range(N):
    for j in range(nPaths):
        Aeq[i*nPaths + j, 2 * nPaths * N + num_asset * i : 2 * nPaths * N + num_asset * (i+1)] = returns_sample[:,i,j]

blocks = np.ones(num_asset)
for i in range(N-1):
    blocks = block_diag(blocks, np.ones(num_asset))
Aeq[nPaths * N :, 2 * nPaths * N : 2 * nPaths * N + num_asset * N] = blocks

beq = np.ones(nPaths * N + N)
beq[: nPaths * N] = targetRet

# upper and lower bound
lb = np.zeros(2 * nPaths * N + num_asset * N)
ub = np.ones(2 * nPaths * N + num_asset * N)
b = np.hstack((ub,lb))
A = np.vstack((np.eye(2 * nPaths * N + num_asset * N),-1 * np.eye(2 * nPaths * N + num_asset * N)))

x = cvx.Variable(2 * nPaths * N + num_asset * N)
prob = cvx.Problem(cvx.Minimize((1/2)*cvx.quad_form(x, P) + q.T@x), \
                 [A@x <= b, \
                  Aeq@x == beq])
prob.solve()

weight = x.value
weight = weight[-num_asset * N:].reshape(N, num_asset)
weight = weight.T


######################
# MVO
######################

P = Q
q = -1 * mu
q = np.zeros(len(mu))

# upper and lower bound
b = np.hstack((np.zeros(num_asset),targetRet))
A = np.vstack((np.eye(num_asset), mu))

# sum of weights = 1
Aeq = np.ones(num_asset)
beq = 1

x = cvx.Variable(num_asset)
prob = cvx.Problem(cvx.Minimize((1/2)*cvx.quad_form(x, P) + q.T@x), \
                 [A@x >= b, \
                  Aeq@x == beq])
prob.solve()

mvo_weight = x.value


######################
# Portfolio Analytics
######################

# Normalized Weight
initialVal = 100
currentPrices = np.array(prices[-1:])
def normalize_weight(initialVal, currentPrices, weight):
    
    if weight.ndim == 1:
        NoShares = (weight * initialVal / currentPrices).T
    else:
        NoShares = np.zeros(weight.shape)
        currentVal = np.zeros(weight.shape[1]+1)
        currentVal[0] = initialVal
        
        for i in range(weight.shape[1]):
            NoShares[:,i] = weight[:,i] * currentVal[i] / currentPrices
            currentVal[i+1] = np.dot(NoShares[:,i].T, currentPrices.T)
    return NoShares

NoShares = normalize_weight(initialVal, currentPrices, weight)
equal_weight = np.ones(NoShares[:,0].shape)*(1/len(NoShares[:,0]))
NoShares_equal = normalize_weight(initialVal, currentPrices, equal_weight)
NoShares_mvo = normalize_weight(initialVal, currentPrices, mvo_weight)

portfolio_value = np.dot(prices,NoShares[:,0])
equal_weight = np.dot(prices,NoShares_equal)
mvo_port = np.dot(prices,NoShares_mvo)
fig, ax = plt.subplots()
ax.plot(portfolio_value, label='Stochastic MVO')
ax.plot(equal_weight, label='Equal Weight')
ax.plot(mvo_port, label='Benchmark MVO')
legend = ax.legend()
plt.show()

portfolio_value = np.dot(test_prices,NoShares[:,0])
equal_weight = np.dot(test_prices,NoShares_equal)
mvo_port = np.dot(test_prices,NoShares_mvo)
fig, ax = plt.subplots()
ax.plot(portfolio_value, label='Stochastic MVO')
ax.plot(equal_weight, label='Equal Weight')
ax.plot(mvo_port, label='Benchmark MVO')
legend = ax.legend()
plt.show()

sns.heatmap(weight, annot=True, fmt='.2f')
plt.show()

sns.heatmap(mvo_weight.reshape(len(mvo_weight),1), annot=True, fmt='.2f')
plt.show()

sns.heatmap(NoShares, annot=True, fmt='.2f')
plt.show()









