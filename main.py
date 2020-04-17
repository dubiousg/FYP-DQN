#import all project files and libraries
from model.clean_data import data_cleaner
from model.market import Market_Environment
from GUI.app import GUI
import model.trader as trader
import pandas as pd
import numpy as np
from os.path import isfile, join
#cleaner = data_cleaner()
#cleaner.add_files_from_folder("original stock data/")

#market_env = Market_Environment("processed stock data")

#trader.run(market_env)
#trader.run_test(market_env)
gui = GUI()
gui.run_gui()
