import pandas as pd
import carhart_harry
import raw_harry


def estimate(carhart_mu: pd.DataFrame, carhart_Q: pd.DataFrame):
    # Modify code here
    Q = None
    return Q

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
  for i in range(len(adjCloseReturns)):
    res = arch_model(adjCloseReturns[i]).fit()
    w.append(res.params["omega"])
    alpha.append(res.params["alpha[1]"])
    beta.append(res.params["beta[1]"])
  return adjCloseReturns, w, alpha, beta

def estimate(adjCloseReturns, w, alpha, beta, Q, time_periods:int):
  # Pre-allocate a list to hold the Garch Series
  Cov_Series = []
  # Pre-allocate a 2d correlation matrix
  Corr = np.zeros([len(Q),len(Q)])
  # Derive correlation estimate from Q
  for i in range(len(Q)):
    for j in range(len(Q)):
      Corr[i, j] = Q[i,j]/((Q[i,i]**0.5)*Q[j,j]**0.5)
  # Perform GARCH 1,1 estimate    
  for i in range(time_periods):
    Cov = np.zeros([len(Q),len(Q)])
    for j in range(len(Q)):
     for k in range(len(Q)):
     # for diagonal variances
      if j == k: 
        Cov[j, k] = w[j]+alpha[j]*(adjCloseReturns[j][i]**2)+beta[j]*Q[j,k]
     # for non-diagonal covariances
      else:
        LR_weight1 = (1-alpha[j]-beta[j])
        LR_weight2 = (1-alpha[k]-beta[k])
        if (round(LR_weight1, 4) == 0.0) or (round(LR_weight2, 4) == 0.0):
          alpha_avg = (alpha[j]+alpha[k])/2
          beta_avg = (beta[j]+beta[k])/2
          Cov[j,k] = (alpha_avg*(adjCloseReturns[j][-1]*adjCloseReturns[k][-1])+
                    beta_avg*Q[j,k])
        else:
          LRVar1 = w[j]/((1-alpha[j]-beta[j]))
          LRVar2 = w[k]/((1-alpha[k]-beta[k]))
          alpha_avg = (alpha[j]+alpha[k])/2
          beta_avg = (beta[j]+beta[k])/2
          w_avg = 1-alpha_avg-beta_avg
          Cov[j,k] = (Corr[j,k]*(LRVar1**0.5)*(LRVar2**0.5)*w_avg+
                    alpha_avg*(adjCloseReturns[j][-1]*adjCloseReturns[k][-1])+
                    beta_avg*Q[j,k])
    Cov_Series.append(Cov)
  return np.array(Cov_Series)

def test():
    """run this function to test"""
    prices = pd.read_csv("../data/adjClose.csv", index_col=["date"])
    factors = pd.read_csv("../data/factors.csv", index_col=["Date"])
    carhart_mu, carhart_Q = carhart_harry.estimate(prices, factors)
    garch_Q = estimate(carhart_mu, carhart_Q)
    # check one example, will raise Error if not true
    assert garch_Q == [[]]
    # write additional code to compare Q with raw
    raw_mu, raw_Q = raw_harry.estimate(prices, factors)
    # plot some differences


if __name__ == "__main__":
    test()
