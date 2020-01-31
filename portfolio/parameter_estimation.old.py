import pandas as pd
import numpy as np

from numpy.linalg import inv
from scipy.stats.mstats import gmean
from scipy.linalg import cholesky, block_diag
from statsmodels.stats.moment_helpers import cov2corr

# just using raw prices to estimate parameters
def raw(prices: pd.DataFrame):
    # write code here

    mean_returns = pd.DataFrame()
    variance_covariance = pd.DataFrame()
    ###
    return mean_returns, variance_covariance


def test_raw():
    # make sure raw is working properly with a tiny example
    pass


def carhart(prices: pd.DataFrame, factors: pd.DataFrame):
    # write code here
    # make sure fama-french factors and prices have the same length of datas
    prices_org = prices
    factor_org = factors

    prices_org['Date'] = prices_org['date'].apply(lambda x: x[:7])
    prices = pd.merge(prices_org, factor_org['Date'], on='Date', how='inner')

    # remove dates in data
    prices = prices.drop(['date','Date'], axis=1)
    factors = factor_org.drop('Date', axis=1)

    # organize data into riskfree, factors, and returns
    factors = factors[1:]
    factors = factors.apply(lambda x: x/100)
    riskfree = factors.loc[:,'RF']
    riskfree = riskfree.reset_index(drop=True)
    factors = factors.drop('RF', axis=1)
    returns = prices[:-1] / prices[1:].values - 1
    returns = returns.apply(lambda x: x-riskfree)

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
    mean_returns = mu.T
    
    #n_stock x n_stock asset covariance matrix
    variance_covariance = np.dot(np.dot(V.T,F),V) + D
    
    ###### No Used, mean_returns and variance_covariance return as numpy array
    # mean_returns = pd.DataFrame()
    # variance_covariance = pd.DataFrame()
    ###
    return mean_returns, variance_covariance

def fama_french(adjclose_df: pd.DataFrame, factor_returns_df: pd.DataFrame):
    # make sure fama-french factors and prices have the same length of datas
    adjclose_df= pd.merge(adjclose_df, factor_returns_df['Date'], on='Date', how='inner')
    factor_returns_df = pd.merge(factor_returns_df, adjclose_df["Date"], on = "Date", how = "inner")
    factors = factor_returns_df[1:].drop(["Date"],axis = 1)
    adjclose_df = adjclose_df.drop(["Date"], axis = 1)
    # extract market risk free rate
    RiskFree = factor_returns_df.loc[:,"RF"]
    RiskFree = RiskFree.reset_index(drop=True)
    # extract five factors - Market Risk Premium, SMB, HML, RMW, CMA
    FactorReturns = factor_returns_df.loc[:,("Mkt-RF", "SMB", "HML", "RMW", "CMA")].apply(lambda x: x/100)
    # calculate arithmetic returns for adjusted closing prices
    adjCloseReturns = (adjclose_df[:-1]/adjclose_df.values[1:] - 1).to_numpy()
    # prepare Factor Returns for OLS Regression
    FactorReturns = np.c_[np.ones(len(FactorReturns)),FactorReturns.to_numpy()]
    # perform OLS Regression and extract alpha and betas
    x = FactorReturns[1:]
    y = adjCloseReturns
    OLS_Parameters = np.dot(np.dot(inv(np.dot(x.T,x)),x.T),y)
    alpha = OLS_Parameters[0,:]
    beta = OLS_Parameters[1:,:]
    # calculate geometric mean return for each factor
    total_return = FactorReturns[:,1:] + 1
    f_bar = gmean(total_return) - 1
    # calculate mu and variance-covariance matrix
    mu = (alpha + np.dot(beta.T, f_bar)).T
    epilson = alpha+np.dot(FactorReturns[1:,1:], beta)
    D = np.dot(np.eye(len(epilson[0])),np.cov(epilson.T))
    F = np.cov(x[:,1:].T)
    VarCov = np.dot(np.dot(beta.T, F),beta)+D
    return mu, VarCov

def carhart_model(prices: pd.DataFrame, factors: pd.DataFrame):
    # write code here
    # make sure fama-french factors and prices have the same length of datas
    prices_org = prices
    factor_org = factors

    prices_org['Date'] = prices_org['date'].apply(lambda x: x[:7])
    prices = pd.merge(prices_org, factor_org['Date'], on='Date', how='inner')

    # remove dates in data
    prices = prices.drop(['date','Date'], axis=1)
    factors = factor_org.drop('Date', axis=1)

    # organize data into riskfree, factors, and returns
    factors = factors[1:]
    factors = factors.apply(lambda x: x/100)
    riskfree = factors.loc[:,'RF']
    riskfree = riskfree.reset_index(drop=True)
    factors = factors.drop('RF', axis=1)
    returns = prices[:-1] / prices[1:].values - 1
    returns = returns.apply(lambda x: x-riskfree)

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
    mean_returns = mu.T
    
    #n_stock x n_stock asset covariance matrix
    variance_covariance = np.dot(np.dot(V.T,F),V) + D
    
    ###### No Used, mean_returns and variance_covariance return as numpy array
    # mean_returns = pd.DataFrame()
    # variance_covariance = pd.DataFrame()
    ###
    return mean_returns, variance_covariance

def test_carhart():
    sample_prices = pd.DataFrame(data = [['2018-08-31 00:00:00+00:00', 1, 1, 2], \
                                        ['2018-09-28 00:00:00+00:00', 3, 1, 1], \
                                        ['2018-10-30 00:00:00+00:00', 1, 2, 1],\
                                        ['2018-11-30 00:00:00+00:00', 2, 1, 1]], \
                                columns = ['date', 'AAPL', 'GOOGL', 'MSFT'])
    sample_factors = pd.DataFrame(data = [['2018-09', 2, 2, 2, 0, 2], \
                                         ['2018-10', 1, 2, 2, 0, 1],
                                         ['2018-11', 0, 2, 1, 0, 2]],
                                columns = ['Date','Mkt-RF', 'SMB', 'HML','RF', 'Mom'])

    mu,Q = carhart(sample_prices, sample_factors)
    assert sum(np.around(mu,3) - np.around([-2.8076577,  2.7340383,  0.0], 3)) == 0
    assert sum(sum(np.around(Q,3) - np.around([[31.80810016, -0.73433088, 0.0], \
                                     [-0.73433088, 1.76283936,  0.0], \
                                     [ 0.0 ,  0.0,  0.0]], 3))) == 0
    
    pass

def garch_params(adjclose_df: pd.DataFrame):
  # get a list of stock tickers
  tickers = adjclose_df.columns[1:]
  # prepare empty to contain results from maximumm likelihood
  adjCloseReturns = []
  w = []
  alpha = []
  beta = []
  for items in tickers: 
    prices_df = adjclose_df[items]
    returns = (prices_df[:-1]/prices_df.values[1:] - 1).to_numpy()
    adjCloseReturns.append(returns)
    print(returns)
  for i in range(len(adjCloseReturns)):
    res = arch_model(adjCloseReturns[i]*100).fit(iter = 100)
    w.append(res.params["omega"])
    alpha.append(res.params["alpha[1]"])
    beta.append(res.params["beta[1]"])
  return adjCloseReturns, w, alpha, beta

def garch11_process(adjCloseReturns, w, alpha, beta, Q, time_periods):
  LastPeriodReturns = []
  for i in range(len(adjCloseReturns)):
    LastPeriodReturns.append(adjCloseReturns[i][-1])
  Corr = np.zeros([len(Q),len(Q)])
  for i in range(len(Q)):
    for j in range(len(Q)):
      Corr[i, j] = Q[i,j]/((Q[i,i]**0.5)*Q[j,j]**0.5)
  for i in range(5):
    for j in range(len(Q)):
     for k in range(len(Q)):
      if j == k: 
        Q[j, k] = w[j]+alpha[j]*(LastPeriodReturns[j]**2)+beta[j]*Q[j,k]
      else:
        LR_weight1 = 1-alpha[j]-beta[j]
        LR_weight2 = 1-alpha[k]-beta[k]
        if (LR_weight1 == 0) or (LR_weight2 == 0):
          alpha_avg = (alpha[j]+alpha[k])/2
          beta_avg = (beta[j]+beta[k])/2
          Q[j,k] = (alpha_avg*(LastPeriodReturns[j]*LastPeriodReturns[k])+
                    beta_avg*Q[j,k])
        else:
          LRVar1 = w[j]/((1-alpha[j]-beta[j]))
          LRVar2 = w[k]/((1-alpha[k]-beta[k]))
          alpha_avg = (alpha[j]+alpha[k])/2
          beta_avg = (beta[j]+beta[k])/2
          w_avg = 1-alpha_avg-beta_avg
          Q[j,k] = (Corr[j,k]*(LRVar1**0.5)*(LRVar2**0.5)*(1-alpha_avg-beta_avg)+
                    alpha_avg*(LastPeriodReturns[j]*LastPeriodReturns[k])+
                    beta_avg*Q[j,k])
  return Q

def test_garch():
    pass


if __name__ == "__main__":
    test_raw()
    test_carhart()
    test_garch()
