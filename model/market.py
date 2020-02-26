#this is the environment of the rl system
#it represents the stock market and the trader's portfolio within
import os 
from os import listdir
from os.path import isfile, join
import pandas as pd

day_global = 0 #day should correspond to the index of the current day
stock_data = {} #contains stock data of the market


class Portfolio:
    stocks = {}
    cash = 1000000
    total_value = cash
    '''
    def __init__(self):
        #todo
    '''
    def update_stock(self, stock, volume, cash_change):
        self.stocks[stock] += volume
        self.cash += cash_change

    def update_value(self):
        temp_total = self.cash
        for stock in self.stocks:
            temp_total += (stocks[stock] * stock_data[stock][day_global]['close'])

    def get_total_value(self):
        return self.total_value

portfolio = Portfolio()#contains the trader's portfolio info: stocks, cash, total value,  

class Market_Environment:

    #read in the clean stock and store them
    def __init__(self, stocks_folder):
        dir = os.getcwd() + "/Dissertation_Project/data"
        folder = join(dir, stocks_folder)
        
        files = [f for f in listdir(folder) if isfile(join(folder, f))]
        for f in files:
            df = pd.read_csv(join(folder, f))
            key = df['Name'][0]

            #drop the name from the data frame (less data handled)
            df.drop(columns='Name', inplace=True)
            
            #add data to raw_stock_data dictonary
            stock_data[key] = df

    #provide new observations or initialise the first observation
    #day is (0, #days]
    def observations(self, day):
        df = pd.DataFrame
        for key in stock_data:
            df[key] = stock_data[key][day]
        
        return df

    #provide portfolio data, (from this possible actions can be computed)
    
    def compute_rewards(self, prev_value, portfolio):
        portfolio.update_value()
        new_value = portfolio.get_total_value()
        reward = new_value - prev_value
        return reward

    def get_next_state(self):
        #think about returning a dataframe
        next_state = {}
        #should return largest length of all stocks
        total_days = max(lambda stock: len(stock_data[stock]))
        if day_global + 1 < total_days:
            for stock in stock_data:
                next_state[stock] = stock_data[stock][day_global + 1]


        return next_state


    #represents a day of trading: 
    #actions is a dictionary of stocks and the amount of trades made for them
    #actions are subjected to conditions such that it cannot buy/sell more stocks that is possible
    def trade(self, actions, day):
        total_value = portfolio.get_total_value()
        day_global = day
        for stock in actions:           
            volume = actions[stock] #volume bought or sold: + for bought - for sold
            cash_change = stock_data[stock][day]['close'] #the cash recieved or taken from buying or selling a stock 
            cash_change = cash_change * (- volume)
            portfolio.update_stock(stock, volume, cash_change)

        reward = self.compute_rewards(total_value, portfolio)
        next_state = self.get_next_state()

        total_days = max(lambda stock: len(stock_data[stock]))
        done = (day_global + 1 == total_days)
        
        return reward, next_state, done

    def get_num_states():
        

          



