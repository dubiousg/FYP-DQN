#this is the environment of the rl system
#it represents the stock market and the trader's portfolio within
import os 
from os import listdir
from os.path import isfile, join
import pandas as pd
import sys

day_global = 0 #day should correspond to the index of the current day
stock_data = {} #contains stock data of the market

class Portfolio:
    stocks = {} #stock name : #stocks
    cash = 1000000
    total_value = cash
    
    def __init__(self):
        print("initialising portfolio")
        #todo
    
    def update_stock(self, stock, volume, cash_change):
        self.stocks[stock] += volume
        self.cash += cash_change

    def update_value(self):
        global day_global
        temp_total = self.cash
        for stock in self.stocks:
            temp_total += (self.stocks[stock] * stock_data[stock][day_global]['close'])

        self.total_value = temp_total

    def get_total_value(self):
        return self.total_value

    def reset(self):
        self.stocks = {} #stock name : #stocks
        self.cash = 1000000
        self.total_value = self.cash

portfolio = Portfolio()#contains the trader's portfolio info: stocks, cash, total value,  

class Market_Environment:

    #read in the clean stock and store them
    def __init__(self, stocks_folder):
        print("initialising market")
        dir = os.getcwd() + "/Dissertation_Project/data"
        folder = join(dir, stocks_folder)
        self.stock_names = []

        files = [f for f in listdir(folder) if isfile(join(folder, f))]
        for f in files:
            df = pd.read_csv(join(folder, f))
            key = df['Name'][0]

            #drop the name from the data frame (less data handled)
            df.drop(columns='Name', inplace=True)
            
            #add data to raw_stock_data dictonary
            stock_data[key] = df
            self.stock_names.append(key)

        self.total_days = len(max(stock_data.values(), key = lambda x: len(x)))
    #provide portfolio data, (from this possible actions can be computed)
    
    def compute_rewards(self, portfolio):
        prev_value = portfolio.get_total_value()
        portfolio.update_value()
        new_value = portfolio.get_total_value()
        reward = new_value - prev_value
        return reward

    def get_observations(self):
        #think about returning a dataframe
        global day_global
        next_state = {}
        #should return largest length of all stocks
        if day_global + 1 < self.total_days:
            for stock in stock_data:
                next_state[stock] = stock_data[stock].iloc[day_global + 1]
                next_state[stock].drop(columns=['Name', 'date'], axis=1)
        return next_state

    #represents a day of trading: 
    #actions is transformed into a  dictionary of stocks and the amount of trades made for them
    #actions are subjected to conditions such that it cannot buy/sell more stocks that is possible
    def trade(self, actions):        
        #convert actions
        global day_global
        actions_dict = {}
        minimum = sys.float_info.max
        
        for i in range(len(actions)):
            actions_dict[self.stock_names[i]] = actions[i] 
            if minimum < actions[i]:
                minimum = actions[i]

        float_int_scaler = 1 / minimum

        for stock, action in actions_dict.items():            
            actions_dict[stock] =  round(action * float_int_scaler) #round to make sure it is an integer

        for stock in actions_dict:           
            volume = actions_dict[stock] #volume bought or sold: + for bought - for sold
            cash_change = stock_data[stock].iloc[day_global]['close'] #the cash recieved or taken from buying or selling a stock 
            cash_change = cash_change * (- volume)
            portfolio.update_stock(stock, volume, cash_change)    
        
        day_global += 1
        
        observations = self.get_observations()
        
        reward = self.compute_rewards(portfolio)

        done = (day_global + 1 == self.total_days)

        return observations, reward, done

    def reset(self):
        global day_global
        day_global = 0
        portfolio.reset()

        return self.get_observations()

#Ten states, one for each feature (continous values)
    def get_num_states(self):
        return 10 

          



