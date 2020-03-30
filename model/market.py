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
    
    def __init__(self, stock_names):
        print("initialising portfolio")
        self.stocks = {} #stock name : #stocks
        self.cash = 1000000
        self.total_value = self.cash

        for stock in stock_names:
            self.stocks[stock] = 0

        print(self.stocks)
        #todo
    
    def update_stock(self, stock, volume, cash_change):
        #print("current stock:" + stock)
        #print(self.stocks)
        #print(stock in self.stocks.keys())
        self.stocks[stock] += volume
        self.cash += cash_change

    def update_value(self):
        global day_global
        temp_total = self.cash
        for stock in self.stocks:
            temp_total += (self.stocks[stock] * stock_data[stock].iloc[day_global]['close'])

        self.total_value = temp_total

    def get_total_value(self):
        return self.total_value

    def reset(self):
        for stock in self.stocks.keys():
            self.stocks[stock] = 0
        #self.stocks = {} #stock name : #stocks
        self.cash = 1000000
        self.total_value = self.cash

#contains the trader's portfolio info: stocks, cash, total value,  

class Market_Environment:
    portfolio = None
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
        self.portfolio = Portfolio(self.stock_names)
    #provide portfolio data, (from this possible actions can be computed)
    
    def compute_rewards(self):
        prev_value = self.portfolio.get_total_value()
        self.portfolio.update_value()
        new_value = self.portfolio.get_total_value()
        reward = new_value - prev_value
        return reward

    def get_observations(self):
        #returns a list of dataframes
        global day_global
        next_state = []
        #should return largest length of all stocks
        if day_global + 1 < self.total_days:
            i = 0
            for data in stock_data.values():
                next_state.append(data.iloc[day_global + 1])
                next_state[i].drop(columns=['Name', 'date'], axis=1)
                i += 1
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

        for stock, action in actions_dict.items():           
            volume = action #volume bought or sold: + for bought - for sold
            cash_change = stock_data[stock].iloc[day_global]['close'] #the cash recieved or taken from buying or selling a stock 
            cash_change = cash_change * (- volume)
            self.portfolio.update_stock(stock, volume, cash_change)    

        day_global += 1
        
        observations = self.get_observations()
        
        reward = self.compute_rewards()

        done = (day_global + 1 == self.total_days)

        return observations, reward, done

    def reset(self):
        global day_global
        day_global = 0
        self.portfolio.reset()

        return self.get_observations()

#Ten states, one for each feature (continous values)
    def get_num_states(self):
        return 10 

          



