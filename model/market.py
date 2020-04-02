#this is the environment of the rl system
#it represents the stock market and the trader's portfolio within
import os 
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import sys
from benchmarking.timer import Timer

timer = Timer()
day_global = 0 #day should correspond to the index of the current day
stock_data = np.ones(1)


class Portfolio:
    
    def __init__(self, stock_names):
        print("initialising portfolio")
        self.stocks = {} #stock name : #stocks
        self.cash = 1000000
        self.total_value = self.cash

        for stock in stock_names:
            self.stocks[stock] = 0

        #print(self.stocks)
        #todo
    
    def update_stock(self, stock, volume, cash_change):
        #print("current stock:" + stock)
        #print(self.stocks)
        #print(stock in self.stocks.keys())
        self.stocks[stock] += volume
        self.cash += cash_change

    def update_value(self):
        global day_global, stock_data
        temp_total = self.cash
        for stock in self.stocks:
            key = stock_data['stock'] == stock
            temp_total += (self.stocks[stock] * stock_data[key].iloc[day_global]['close'])

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

#contains stock data of the market

class Market_Environment:
    portfolio = None
    #read in the clean stock and store them
    def __init__(self, stocks_folder):
        global stock_data
        print("initialising market")
        dir = os.getcwd() + "/Dissertation_Project/data"
        folder = join(dir, stocks_folder)
        self.stock_names = []

        stock_data_temp = {}
        files = [f for f in listdir(folder) if isfile(join(folder, f))]
        for f in files:
            df = pd.read_csv(join(folder, f), index_col=0)
            key = df['Name'][0]

            #drop the name from the data frame (less data handled)
            df.drop(columns='Name', inplace=True)
            
            #add data to raw_stock_data dictonary, converted to numpy
            stock_data_temp[key] = df.to_numpy()
            self.stock_names.append(key)

        d_type = np.dtype(stock_data_temp['A'].dtype)

        stock_data = np.zeros(len(self.stock_names), dtype={'names':['stock', 'data'],
                          'formats':['U10', d_type]})

        stock_data['stock'] = self.stock_names
        stock_data['data'] = stock_data_temp

        #print(stock_data[stock_data['stock'] == 'A'])
        self.total_days = len(max(stock_data_temp.values(), key = lambda x: len(x)))
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
        global day_global, stock_data
        #print(stock_data['data'][0]['A'].dtype)
        d_type = stock_data['data'][0]['A'].dtype
        next_state = np.ndarray(shape=(len(stock_data['data']),), dtype=d_type)
        #should return largest length of all stocks
        if day_global + 1 < self.total_days:
            i = 0
            for data in stock_data['data'][0]:
                #print(data.drop(columns=['date'], axis=1))
                needed_data = data.drop(columns=['date'], axis=1).iloc[day_global + 1]

                next_state[i] = needed_data
                #next_state.append(needed_data)

                i += 1
                
        return next_state

    #represents a day of trading: 
    #actions is transformed into a  dictionary of stocks and the amount of trades made for them
    #actions are subjected to conditions such that it cannot buy/sell more stocks that is possible
    def trade(self, actions):        
        #convert actions
        global day_global, stock_data
        actions_dict = {}
        minimum = sys.float_info.max
        
        timer.start_timer()

        for i in range(len(actions)):
            actions_dict[self.stock_names[i]] = actions[i] 
            if minimum < actions[i]:
                minimum = actions[i]

        float_int_scaler = 1 / minimum

        print("loading action dict: " + str(timer.get_time()))

        timer.start_timer()

        for stock, action in actions_dict.items():            
            actions_dict[stock] =  np.round(action * float_int_scaler) #round to make sure it is an integer

        print("manipulate dict: " + str(timer.get_time()))

        timer.start_timer()

        for stock, action in actions_dict.items():           
            volume = action #volume bought or sold: + for bought - for sold
            key = stock_data['stock'] == stock
            cash_change = stock_data[key].iloc[day_global]['close'] #the cash recieved or taken from buying or selling a stock 
            cash_change = cash_change * (- volume)
            self.portfolio.update_stock(stock, volume, cash_change)    

        print("updating stocks: " + str(timer.get_time()))

        day_global += 1
        
        timer.start_timer()
        observations = self.get_observations()
        print("get_obsverations: " + str(timer.get_time()))

        timer.start_timer()
        reward = self.compute_rewards()
        print("compute_rewards: " + str(timer.get_time()))
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

          



