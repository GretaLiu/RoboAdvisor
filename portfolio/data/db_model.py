from typing import List
from pandas import DataFrame, concat, read_sql_table, read_sql
from pandas import Series
from pandas_datareader import famafrench
import os
from datetime import date
from tiingo import TiingoClient
import sqlalchemy as db 
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from backend.investment_profile import RISKTOLERANCE



dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

client = TiingoClient(
    {"session": True, "api_key": "9ca5dfd053eb30abcb2610069a54cacd2afb4763"}
)

def get_historical_price(
    tickers: List[str], startDate: date, endDate: date, frequency: str = "monthly"
):
    assert tickers  # make sure there are at least one ticker
    #print("Getting prices for", " ".join(tickers))
    df = client.get_dataframe(
        tickers, metric_name="adjClose", startDate=startDate, frequency=frequency
    )
    return df
    

def get_factors(startDate: date, endDate: date, frequency: str= "monthly"):
    factors = famafrench.FamaFrenchReader("F-F_Research_Data_Factors", start=startDate)
    ff3factors: DataFrame = factors.read()[0]
    momentum = famafrench.FamaFrenchReader("F-F_Momentum_Factor", start=startDate)
    momfactors: DataFrame = momentum.read()[0]
    together = concat([ff3factors, momfactors], axis=1)
    return together 


def get_riskfree(startDate: date, endDate: date, frequency: str):
    # write code here
    return DataFrame()


def save_historical_price(
    tickers: List[str], startDate: date, endDate: date, frequency: str = "monthly"
):
    engine = db.create_engine('postgresql+psycopg2://nzehvkdfogjlxh:4dd1f72a746963e155f5331796005be87d6750164d1698820cc561987beac08d@ec2-54-225-115-177.compute-1.amazonaws.com:5432/de82kq7ni8ngs9')
    conn = engine.connect()
    update_query = ""
    # calculate historical price 
    new_data = get_historical_price(tickers, startDate, endDate)
    new_data.columns = map(str.upper, new_data)
    # pull old from database
    try:
        old_data = pd.read_sql_query('''select * from public.stock_price''', conn)
    except:
        new_data.to_sql("stock_price", conn, if_exists="replace")
        return
    # if database is empty => dump
    if (old_data.empty == True):
        new_data.to_sql("stock_price", conn, if_exists="replace")
    else:
        new_data.to_sql("temp_stock_price", conn, if_exists="replace")
        # if ticker doesn't exist in the database as a column name => add column
        old_ticker_df = pd.read_sql_query('''SELECT column_name FROM information_schema.columns 
                                                WHERE table_schema = 'public'
   									            AND table_name   = 'stock_price' ''',conn)
        new_data = pd.read_sql_query("select * from public.temp_stock_price", conn)
        new_data.columns = map(str.upper, new_data.columns)
        new_ticker_array = (new_data.columns[~new_data.columns.isin(old_ticker_df.column_name)]).tolist()
        # add new column to old table
        for i in new_ticker_array:
            if (i != "DATE" and i != "INDEX"):
                conn.execute("ALTER TABLE public.stock_price ADD COLUMN \""+ i + "\" double precision")
        # if timestamp not exist => insert, then update the whole table 
        conn.execute("insert into stock_price (\"date\") select \"date\" from temp_stock_price where \"date\" not in (select \"date\" from stock_price)")
        trans = conn.begin()
        update_query = "UPDATE stock_price SET "
        for index in range(1, len((new_data.columns).tolist())-1 ): #exclude "date" and the last one
            update_query += "\""+(new_data.columns).tolist()[index] +"\" = temp_stock_price.\""+ (new_data.columns).tolist()[index]+"\", "
        update_query += "\""+(new_data.columns).tolist()[len((new_data.columns).tolist())-1] +"\" = temp_stock_price.\""+ (new_data.columns).tolist()[len((new_data.columns).tolist())-1]+"\""
        update_query += " from temp_stock_price where stock_price.\"date\" = temp_stock_price.\"date\""
        try:
            conn.execute(update_query)
            trans.commit()
        except:
            trans.rollback()
            raise
    
    return 

def save_factors(startDate: date, endDate: date, frequency: str= "monthly"):
    engine = db.create_engine('postgresql+psycopg2://nzehvkdfogjlxh:4dd1f72a746963e155f5331796005be87d6750164d1698820cc561987beac08d@ec2-54-225-115-177.compute-1.amazonaws.com:5432/de82kq7ni8ngs9')
    conn = engine.connect()
    update_query = ""
    new_factors = get_factors(startDate, endDate)
    new_factors.index.name = (new_factors.index.name).lower()
    new_factors.columns = map(str.lower, new_factors)
    new_factors.index = pd.to_datetime(new_factors.index.to_timestamp(), errors='raise')
    new_factors.to_sql("temp_FF_factors", conn, if_exists="replace")
    # pull old from database 
    try:
        old_factors = pd.read_sql_query('''select * from \"FF_factors\"''', conn)
    except: # create new table if old table doesn't exist
        new_factors.to_sql("FF_factors", conn, if_exists="replace")
        return
    # if database is empty => dump
    if (old_factors.empty == True):
        new_factors.to_sql("FF_factors", conn, if_exists="append")
    else:
    # if timestamp not exist => insert, then update
        conn.execute("insert into \"FF_factors\" (\"date\") select \"date\" from \"temp_FF_factors\" where \"date\" not in (select \"date\" from \"FF_factors\")")
        update_query = "UPDATE \"FF_factors\" SET \"mkt-rf\" = \"temp_FF_factors\".\"mkt-rf\", " 
        update_query += "\"smb\" = \"temp_FF_factors\".\"smb\", " 
        update_query += "\"hml\" = \"temp_FF_factors\".\"hml\", " 
        update_query += "\"rf\" = \"temp_FF_factors\".\"rf\", " 
        update_query += "\"mom   \" = \"temp_FF_factors\".\"mom   \" "
        update_query += "from \"temp_FF_factors\" where \"FF_factors\".\"date\" = \"temp_FF_factors\".\"date\" "
        conn.execute(update_query)
    return 


def provide_investment_universe(risk_attitude: RISKTOLERANCE):
    # provide 10 tickers for running model according to investment character 
    engine = db.create_engine('postgresql+psycopg2://nzehvkdfogjlxh:4dd1f72a746963e155f5331796005be87d6750164d1698820cc561987beac08d@ec2-54-225-115-177.compute-1.amazonaws.com:5432/de82kq7ni8ngs9')
    conn = engine.connect()
    selected_ten_assets=[]
    if (risk_attitude == RISKTOLERANCE.risky): #select highest return 
        stock_prices = pd.read_sql_table("stock_price", conn,index_col='date')
        historical_returns = (stock_prices.iloc[-12:,:]/stock_prices.iloc[-13:-1,:].values).mean()
        top_10_returns = historical_returns.sort_values()[-10:]
        selected_ten_assets = top_10_returns.index
    elif (risk_attitude == RISKTOLERANCE.moderate): #select highest capital 
        query_largemc = "select \"Symbol\" from (select *, rank() over (partition by \"Sector\""
        query_largemc += "order by \"Market Cap\" desc) from sp500_sector_mc)b where RANK<2 "
        df_largemc = pd.read_sql_query(query_largemc, conn)
        selected_ten_assets = (df_largemc.iloc[:,0].values)[:10]
    else: #select ten ETFs
        selected_ten_assets = ['SPY', 'VTV', 'VOE', 'VBR', 'VWO', 'AGG', 'HYLB', 'NEAR', 'EMB', 'VOE','VEA']
    
    return selected_ten_assets


def get_historical_return (tickers: List[str], startDate: date, endDate: date):
    query_price = "select \"date\", "
    if len(tickers)>0:
        for i in range(0,len(tickers)-1):
            query_price += "\""+tickers[i]+"\", "
        query_price += "\""+ tickers[len(tickers)-1]+"\" "
        query_price += "from stock_price where \"date\" < '"+ endDate.strftime("%Y-%m-%d") +"' and \"date\" >= '"+ startDate.strftime("%Y-%m-%d") +"' "
        price = pd.read_sql_query(query_price, conn)
    #calculate price return 
    price = price.set_index('date')
    rate_of_return =price.pct_change()
    return rate_of_return

def carhart_data_source (tickers: List[str], startDate: date, endDate: date):
    engine = db.create_engine('postgresql+psycopg2://nzehvkdfogjlxh:4dd1f72a746963e155f5331796005be87d6750164d1698820cc561987beac08d@ec2-54-225-115-177.compute-1.amazonaws.com:5432/de82kq7ni8ngs9')
    conn = engine.connect()
    #get factor data 
    query_factor = "select * from \"FF_factors\" where \"date\" < '"+ endDate.strftime("%Y-%m-%d") +"' and \"date\" >= '"+ startDate.strftime ("%Y-%m-%d") +"' "
    factors = pd.read_sql_query(query_factor, conn)
    factors = factors.set_index('date')
    #get histotical price data
    rate_fo_return = get_historical_return(tickers, startDate, endDate)
    #combine rate_of_return and factors 
    combined = (rate_of_return.assign(year=rate_of_return.index.year,month=rate_of_return.index.month)
        .merge(factors.assign(year=factors.index.year, month=factors.index.month),on=['year','month'],right_index=True)
        .drop(['year','month'],axis=1))
    return  combined

if __name__ == "__main__":
    engine = db.create_engine('postgresql+psycopg2://nzehvkdfogjlxh:4dd1f72a746963e155f5331796005be87d6750164d1698820cc561987beac08d@ec2-54-225-115-177.compute-1.amazonaws.com:5432/de82kq7ni8ngs9')
    conn = engine.connect()
    
    query_top_tickers = "select \"tickers\" from " 
    query_top_tickers += "(select *, rank() over (partition by \"Sector\" order by \"sum_volume\" desc) from sp500_sector_vol)b "
    query_top_tickers += "where RANK<6"
    #top_tickers = pd.read_sql_query(query_top_tickers, conn)
    #print(top_tickers["tickers"].values.tolist())
    #save_historical_price (ticker_list, datetime.now() - timedelta(days=13 * 365),datetime.now())
    #print(get_historical_price(ticker_list, datetime.now() - timedelta(days=  365),datetime.now()))
    
    #save_factors(datetime.now() - timedelta(days= 13*365),datetime.now)
    
    #print(carhart_data_source (["AAPL", "GOOGL", "MSFT", "SPY", "EEM", "XLF"], datetime.now() - timedelta(days= 365), datetime.now()))
    
    #print(carhart_data_source(['BABA'],datetime.now() - timedelta(days= 365*6), datetime.now() - timedelta(days= 365*3) ))

    #load csv to database 
    #data = pd.read_csv("/Users/gliu/Desktop/python/MIE479-Capstone/backend/portfolio/data/sp500_financial_status.csv")
    #data.to_sql("sp500_ticker_MC", conn, if_exists="replace")

    #['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB', 'BRK-'B, 'BABA', 'JPM', 'V', 'TCEHY', 'JNJ', 'WMT', 'NSRGY', 'PG', 'BAC']
    #['XOM', 'T', 'MA', 'TSM', 'DIS', 'HD', 'RHHVF', 'IDCBF', 'UNH', 'INTC', 'VZ', 'CVX', 'WFC', 'KO', 'LVMUY', 'MRK'] 
    #['PIAIF', 'BA', 'RHHBY', 'PFE', 'NVS', 'CMCSA', 'TM']

    
    print(provide_investment_universe(RISKTOLERANCE.risky))

   