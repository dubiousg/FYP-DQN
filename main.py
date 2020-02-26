#import all project files and libraries
from model.clean_data import data_cleaner
from model.market import Market_Environment
import model.trader as trader
import pandas as pd
import numpy as np
from os.path import isfile, join
#cleaner = data_cleaner()
#cleaner.add_files_from_folder("original stock data/")

market_env = Market_Environment("processed stock data")

trader.run(market_env)

'''
index=[0,1,2,3]
columns=['one', 'two']
data = np.array([np.arange(2, 10, 2)]*2).T

df = pd.DataFrame(data, index=index, columns=columns)
#df['one'] = 
df['one'] = df['one'].div(df['one'][0])
print(df)
'''