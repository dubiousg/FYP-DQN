from sklearn import preprocessing
import numpy as np
import pandas as pd
import os 
from os import listdir
from os.path import isfile, join

def relative_strength_index(close):
    days = 14
    avg_gain = 0
    avg_loss = 0

    for i in range(0, days - 1):
        if close[i + 1] > close[i]:
            avg_gain += 100 * ((close[i + 1] - close[i])/close[i])
        elif close[i + 1] < close[i]:
            avg_loss -= 100 * ((close[i + 1] - close[i])/close[i])

    avg_gain /= days
    avg_loss /= days

    rsi = np.zeros(shape=(close.size))
    rsi[days] = 100 - (100 / (1 + (avg_gain / avg_loss)))

    for i in range(days + 1, close.size):
        prev_gain = avg_gain
        prev_loss  = avg_loss
        avg_gain = 0
        avg_loss = 0

        for j in range(i - days, i):
            if close[j + 1] > close[j]:
                avg_gain += 100 * ((close[j + 1] - close[j])/close[j])
            elif close[j + 1] < close[j]:
                avg_loss -= 100 * ((close[j + 1] - close[j])/close[j])

        avg_gain /= days
        avg_loss /= days

        gain = (prev_gain * 13) + avg_gain
        loss = (prev_loss * 13) + avg_loss
        if loss == 0:
            rs = 0
        else:
            rs = gain / loss

        rsi[i] = 100 - (100 / (1 + rs))

    return rsi

def on_balance_volume(close, volume):
    obv = np.empty(shape=(close.size))
    obv[0] = volume[0] 
    for i in range(0, close.size - 1):
        if close[i + 1] > close[i]:
            obv[i + 1] = obv[i] + volume[i]
        elif close[i + 1] < close[i]:
            obv[i + 1] = obv[i] - volume[i]
        else: 
            obv[i + 1] = obv[i]

    return obv

def write_data(data):
    dir = os.getcwd() + "/Dissertation_Project/data/processed stock data"
    #dir = 'C:/Users/Cooper/VSCodeWorkSpace/Dissertation_Project/data/processed stock data'
    for stock in data.keys():
        file_name = "clean_" + stock + ".csv"
        file = join(dir, file_name)
        data[stock]['Name'] = np.full(shape=(data[stock]['close'].size), fill_value=stock)
        data[stock].to_csv(file)



class data_cleaner:
    raw_stock_data =  {} 

    #def __init__(self, file):

    #add stocks from a folder to the raw_stock_data dictionary
    def add_files_from_folder(self, folder_dir):
        dir = os.getcwd() + "/Dissertation_Project/data"
        folder = join(dir, folder_dir)
        #print(isfile(listdir(folder)[1]))
        files = [f for f in listdir(folder) if isfile(join(folder, f))]
        for f in files:
            df = pd.read_csv(join(folder, f))
            key = df['Name'][0]

            #clean data
            df.drop(columns='Name', inplace=True)

            to_normalise = ['high', 'low', 'close', 'open', 'volume']

            for feature in to_normalise: 
                df[feature] = df[feature]/df[feature][0]
            
            #50 calculate day average and 200 day average:
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['sma_100'] = df['close'].rolling(window=100).mean()

            #calculate the relative strength index
            df['rsi'] = relative_strength_index(df['close'])

            #calculate on balance volume
            df['obv'] = on_balance_volume(df['close'], df['volume'])
            
            #add data to raw_stock_data dictonary
            self.raw_stock_data[key] = df

        write_data(self.raw_stock_data)

    def add_file(self, file):
        file = join('C:/Users/Cooper/VSCodeWorkSpace/Dissertation/data/', file)
        df = pd.read_csv(file)
        self.raw_stock_data[df['Name'][0]] = df #add to dictionary, stock name as the key

    def run(self): #for now this is a test function
        print(self.raw_stock_data['AAL'].head())
        print(self.raw_stock_data['XOM'].head())
        print(self.raw_stock_data['AAL'].iloc[25])
        