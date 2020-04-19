#this is the environment of the rl system
#it represents the stock market and the trader's portfolio within
import os 
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import sys
from benchmarking.timer import Timer
from benchmarking.unittesting import Tester 
import math

timer = Timer()
tester = Tester()
day_global = 0 #day should correspond to the index of the current day
stock_data = np.ones(1)

#The Portfolio class:
#   *contains the assets of the trader
#   *updates the trader's assets
class Portfolio:
    
    #desc: a constructor, initializes assets and a stocks dictionary
    #input: stock_names - list of strings containing stock names
    def __init__(self, stock_names):
        print("initialising portfolio")
        self.stocks = {} #stock name : stocks, holds portfolio stocks
        self.cash = 1000000
        self.total_value = self.cash

        for stock in stock_names:
            self.stocks[stock] = 0

        #print(self.stocks)
        #todo
    
    #desc: updates stocks in the portfolio during a trading session
    #input: actions_dict - 
    def update(self, actions_dict):
        #sell all stocks before resetting weights
        for stock, volume in self.stocks.items():

            key = stock_data['stock'] == stock
            data_array = stock_data[key][0][1]
            if day_global < len(data_array):
                cash_change = data_array[day_global][4] #the cash recieved or taken from buying or selling a stock 
                cash_change = cash_change * volume

                tester.test_equal(volume % 1, 0)
                self.update_stock(stock, -volume, cash_change)


        for stock, action in actions_dict.items():           
            volume = np.floor(action * self.total_value)

            key = stock_data['stock'] == stock
            data_array = stock_data[key][0][1]
            cash_change = data_array[day_global][4] #the cash recieved or taken from buying or selling a stock 
            cash_change = cash_change * (- volume)

            tester.test_equal(volume % 1, 0)
            self.update_stock(stock, volume, cash_change)

    #desc:
    #input:
    #output:
    def update_stock(self, stock, volume, cash_change):
        self.stocks[stock] += volume
        #if self.stocks[stock] % 1 != 0:
        #    gggg = 0
        tester.test_equal(self.stocks[stock] % 1, 0)

        self.cash += cash_change

    #desc:
    #input:
    #output:
    def update_value(self):
        global day_global, stock_data
        temp_total = self.cash

        #for stock in self.stocks:
        for i in range(0, len(stock_data['data'])):
            stock = stock_data['stock'][i]

            #check if stock is still in circulation
            if day_global < len(stock_data['data'][i]):
                #tedee = stock_data['data'][i][day_global][4]
                temp_total += (self.stocks[stock] * stock_data['data'][i][day_global][4])
                #if math.isnan(temp_total):
                #    abdc = 0
                tester.test_not_nan(temp_total)
            elif day_global == len(stock_data['data'][i]): #sell all stock with last previous value if no longer in circulation
                temp_total -= self.cash
                cash_change = self.stocks[stock] * stock_data['data'][i][day_global - 1][4]
                tester.test_equal(self.stocks[stock] % 1, 0)
                self.update_stock(stock, -self.stocks[stock], cash_change)
                tester.test_not_nan(temp_total)
                temp_total += self.cash
                #temp_total += (self.stocks[stock] * stock_data['data'][i][day_global - 1][4])
        self.total_value = temp_total

    #desc:
    #input:
    #output:
    def get_total_value(self):
        return self.total_value

    #desc:
    #input:
    #output:
    def reset(self):
        for stock in self.stocks.keys():
            self.stocks[stock] = 0
        #self.stocks = {} #stock name : #stocks
        self.cash = 1000000
        self.total_value = self.cash

    def get_stocks(self):
        return self.stocks

#contains the trader's portfolio info: stocks, cash, total value,  

#contains stock data of the market

class Market_Environment:
    portfolio = None
    #read in the clean stock and store them

    #desc:
    #input:
    #output:
    def __init__(self, stocks_folder):
        global stock_data
        print("initialising market")
        dir = os.getcwd() + "/Dissertation_Project/data"
        folder = join(dir, stocks_folder)
        self.stock_names = []

        #stock_data_temp = {}
        stock_data_temp = []
        files = [f for f in listdir(folder) if isfile(join(folder, f))]
        for f in files:
            df = pd.read_csv(join(folder, f), index_col=0)
            key = df['Name'][0]

            #drop the name from the data frame (less data handled)
            df.drop(columns='Name', inplace=True)
            
            #add data to raw_stock_data dictonary, converted to numpy
            stock_data_temp.append(df.to_numpy())
            self.stock_names.append(key)

        d_type = np.dtype(stock_data_temp[0].dtype)

        stock_data = np.zeros(len(self.stock_names), dtype={'names':['stock', 'data'],
                          'formats':['U10', d_type]})

        stock_data['stock'] = self.stock_names
        #print(stock_data_temp.values())
        stock_data['data'] = stock_data_temp

        self.total_days = len(max(stock_data_temp, key = lambda x: len(x)))
        self.portfolio = Portfolio(self.stock_names)
    #provide portfolio data, (from this possible actions can be computed)
    
    #desc:
    #input:
    #output:
    def compute_rewards(self):
        prev_value = self.portfolio.get_total_value()
        self.portfolio.update_value()
        new_value = self.portfolio.get_total_value()
        reward = new_value - prev_value
        return reward

    #desc:
    #input:
    #output:
    def get_observations(self):
        #returns a list of dataframes
        global day_global, stock_data
        #print(stock_data['data'][0]['A'].dtype)
        d_type = stock_data['data'][0].dtype
        next_state = np.ndarray(shape=(len(stock_data['data']),), dtype=d_type)
        #should return largest length of all stocks
        if day_global + 1 < self.total_days:
            i = 0
            for data in stock_data['data']:

                if day_global + 1 < len(data):
                    needed_data = data[day_global + 1][1:]
                else:
                    needed_data = np.zeros(10)

                next_state[i] = needed_data

                i += 1
                
        return next_state

    #represents a day of trading: 
    #actions is transformed into a  dictionary of stocks and the amount of trades made for them
    #actions are subjected to conditions such that it cannot buy/sell more stocks that is possible
    #desc:
    #input:
    #output:
    def trade(self, actions):        
        #convert actions
        global day_global, stock_data
        actions_dict = {}

        total = 0
        for i in range(actions.size):

            #check if stock is available
            sd = stock_data[stock_data['stock'] == self.stock_names[i]]
            sd = sd[0]['data']
            if day_global < len(sd):
                if math.isnan(actions[i]) or actions[i] < 0:
                    actions[i] = 0

                tester.test_not_nan(actions[i])

                actions_dict[self.stock_names[i]] = actions[i] 
                total += actions[i]
                #if actions[i] < minimum:
                #    minimum = actions[i]

        if total == 0:
            total = 1

        tester.test_not_nan(total)


        for stock, action in actions_dict.items():       
            actions_dict[stock] = action / total #make all actions sum to 1 


        #timer.start_timer()
        self.portfolio.update(actions_dict)
        #all of this should be handled by the portfolio class

        day_global += 1
        

        observations = self.get_observations()

        reward = self.compute_rewards()

        tester.test_not_nan(reward)
        done = (day_global + 1 == self.total_days)

        if done:
            print(self.portfolio.get_total_value())

        return observations, reward, done

    def reset(self):
        global day_global
        day_global = 0
        self.portfolio.reset()
    
        return self.get_observations()

#Ten states, one for each feature (continous values)
    #desc:
    #input:
    #output:
    def get_num_states(self):
        return 10 

    #desc:
    #input:
    #output:
    def get_allocation(self):
        return self.portfolio.get_stocks()

    #desc:
    #input:
    #output:
    def get_portfolio_value(self):
        return self.portfolio.get_total_value() 

        #desc:
    #input:
    #output:
    def test_mode(self):
        global day_global
        day_global = round(self.total_days * 0.7)

    #desc:
    #input:
    #output:
    def get_prices(self):
        global stock_data
        prices = {}

        for i in range(len(stock_data)):
            if day_global < len(stock_data["data"][i]):
                prices[stock_data["stock"][i]] = stock_data["data"][i][day_global][4]

        return prices


          



