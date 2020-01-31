def cvar_transaction_cost(oldweights:pd.Series, newweights:pd.Series,prices:pd.Dataframe):
    currentPrices=[]
        for row in prices.tail(1).values:
            for v in row:
                currentPrices.append(v)

    return abs(oldweights-newweights)*currentPrices*0.001

    