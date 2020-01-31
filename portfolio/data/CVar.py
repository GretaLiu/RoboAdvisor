def CVar(mu, Q, prices )  : 
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
    print(S)
    # returns_sample n_asset * nPeriod * nPaths
    returns_sample = np.array([[0.0 for k in range(nPaths)] for j in range(num_asset)])
    print(len(returns_sample))
    for i in range(nPaths):
        for j in range(num_asset):
            returns_sample[j, i] = S[j,-1,i] / S[j, 0, i] - 1
            
    f=np.zeros(num_asset + nPaths + 1)
    f[:nPaths]=1 / ((1 - confidence_level) * nPaths)
    f[nPaths : num_asset + nPaths] = 0
    f[-1] = 1

    A= np.array([[0.0 for k in range(nPaths + num_asset + 1)] for j in range(2 * nPaths)])

    A[:nPaths, :nPaths] = -1 * np.eye(nPaths)
    A[nPaths:(2 * nPaths), :nPaths]= -1 * np.eye(nPaths)
    A[nPaths:(2 * nPaths), -1] = -1
    #A= np.array([[0.0 for k in range(nPaths + num_asset + 1)] for j in range(3 * nPaths)])

    #A[:nPaths, :nPaths] = -1 * np.eye(nPaths)
    #A[nPaths:(2 * nPaths), :nPaths]= -1 * np.eye(nPaths)
    #A[nPaths:(2 * nPaths), -1] = -1
    for i in range((nPaths), (2 * nPaths)):
            A[i,nPaths:(nPaths + num_asset)] = - (returns_sample[:, i - nPaths].transpose())

    Aeq = np.array([[0.0 for k in range(nPaths + num_asset + 1)] for j in range(1)])
    Aeq[0 , nPaths:(nPaths + num_asset)] = 1

    beq = np.array([1])
    #b = np.array([0.0 for k in range(2 * nPaths)] )
    #lowerbound = np.zeros(num_asset + nPaths + 1)
    #lowerbound[:nPaths] = -1000000
    #lowerbound[nPaths : num_asset + nPaths] = 0
    #lowerbound[-1] = -1000000
    b = np.array([0.0 for k in range(3 * nPaths + num_asset +1)] )
    b[(2*nPaths):(3*nPaths)] = 1000000000000
    b[(3*nPaths) : num_asset + 3*nPaths] = 0
    b[-1] = 1000000000000
    #temp = ([[0.0 for k in range(nPaths + num_asset + 1)] for j in range(nPaths + num_asset + 1)])
    temp= -1 * np.eye(nPaths + num_asset + 1)
    A=np.concatenate((A, temp), axis=0)

    res = linprog(c=f, A_ub=A, b_ub=b,A_eq=Aeq,b_eq=beq)
    print(res.x[nPaths:nPaths+num_asset])
    return (res.x[nPaths:nPaths+num_asset])
    #x = cvx.Variable(num_asset)
    #prob = cvx.Problem(cvx.Minimize(f.T@x),[A@x<=b,Aeq@x==beq])
    #prob.solve()