import sqlalchemy as db


class Database():
    engine = db.create_engine('postgresql+psycopg2://postgres:446Crigi@localhost/robo_advisor_data')
    def __init__(self):
        self.connection = self.engine.connect()
        print("DB Instance created")

    def fetchByQuery(self, query):
        fetchQuery = self.connection.execute(f"SELECT * FROM {query}")
        for data in fetchQuery.fetchall():
            print(data)

    def saveHistoricalPrice(self, data):
        self.connection.execute(
            f"""INSERT INTO monthly_stock_price( ticker, date_time, price, last_update_time) VALUES('{data.ticker}','{data.time}','{data.price}','{data.update_time}')""")

class HistoryPriceRecord():
    def __init__(self, ticker, time, price, update_time):
        self.ticker = ticker
        self.time = time
        self.price = price
        self.update_time = update_time
