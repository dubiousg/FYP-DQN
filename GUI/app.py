import os
import matplotlib.pyplot as plt
import tkinter as tk
from tkcalendar import Calendar, DateEntry
from model.market import Market_Environment
import model.trader as trader
from tkinter import *
import datetime
import re

class GUI:

    def __init__(self):
        self.day_one = "5/8/16" #index 881 of processed data
        self.date_one = datetime.datetime(day=5, month=8, year=2016)
        self.states = { "testing": False, "tested": False } #state of the gui
        self.root = tk.Tk()
        self.height = 500
        self.width = 1000

        self.canvas = tk.Canvas(self.root, height=self.height, width=self.width, bg="#263d42")
        self.canvas.pack()



        self.frame = tk.Frame(self.root, bg="#3e646c")

        self.frame.place(width=self.width * 0.6, height=self.height, y=0, x=0)

        self.btns = {}
        self.labels = {}

        self.test_list = []

        self.btns["run"] = Button(self.frame, text="Play Trading Period", fg="#263d42")
        self.btns["run"].grid(row=0)
        self.btns["run"].bind("<Button-1>", self.run_test)

        self.btns["graph"] = Button(self.frame, text="Create Graph", fg="#263d42")
        self.btns["graph"].grid(row=0, column=1)
        self.btns["graph"].bind("<Button-1>", self.update_chart)

        self.btns["alloc"] = Button(self.frame, text="Show Allocations", fg="#263d42")
        self.btns["alloc"].grid(row=0, column=2)
        self.btns["alloc"].bind("<Button-1>", self.show_allocations)

        self.btns["roi"] = Button(self.frame, text="Show ROI", fg="#263d42")
        self.btns["roi"].grid(row=0, column=3)
        self.btns["roi"].bind("<Button-1>", self.show_roi)

        self.labels["run"] = Label(self.frame, text="Not Running")

        #top = tk.Toplevel(self.root)

        self.cal = Calendar(self.canvas,
        font="Arial 14", selectmode='day',
        cursor="hand1", year=2018, month=2, day=5)

        self.cal.pack(fill="both", expand=True)
        self.canvas.create_window((self.width * 0.6, 0), window=self.cal, anchor='nw')

    def run_test(self, event):
        print("running test")
        self.states["testing"] = True
        self.market = Market_Environment("processed stock data")
        self.test_list = trader.run_test(self.market)
        self.states["testing"] = False
        self.states["tested"] = True

    def update_chart(self, event):
        data = []
        for item in self.test_list:
            data.append(item["value"])

        plt.plot(data)
        plt.show()

    def show_allocations(self, event):
        #https://matplotlib.org/3.1.1/gallery/pie_and_polar_charts/pie_features.html
        #print(self.cal.get_date())

        if self.states["tested"]:
            day = self.calc_day(self.cal.get_date()) #days is an index

            if -1 < day and day < 378: 
                allocations = self.test_list[day]["allocation"]

                total_value = 0
                for stock, amt in allocations.items():
                    if stock in self.test_list[day]["prices"]:
                        total_value += amt * self.test_list[day]["prices"][stock]


                stocks = []
                amounts = []
                for stock, amt in allocations.items():
                    if 0 < amt:
                        amounts.append(amt / total_value)
                        stocks.append(stock)

                

                labels = stocks

                #for amt in allocations.vals()
                #    total_num_stocks += amt
                #explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

                fig = plt.figure()
                ax = fig.add_axes([0,0,1,1])
                ax.axis('equal')
                ax.pie(amounts, labels = labels, autopct='%1.1f%%')
                plt.show()

    def show_roi(self, event):
        if self.states["tested"]:
            roi = (self.test_list[364]["value"] -  self.test_list[0]["value"]) / self.test_list[0]["value"]
            roi *= 100
            self.labels["roi"] = Label(self.frame, text="Yearly roi: " + str(roi))
            self.labels["roi"].grid(row=1, column=3)

    def run_gui(self):
        self.root.mainloop()

    #desc: gives index based on difference between first day and choosen
    def calc_day(self, date):
        value_p = re.compile(r'\d+')
        (month, day, year) = value_p.findall(date)
        (day, month, year) = (int(day), int(month), int(year) + 2000)
        date = datetime.datetime(day=day, month=month, year=year)
        index = (date - self.date_one).days
        return index


#todo list:
#1 run a trading period with a finished model






#2 create analytics for
#2.1 yearly roi



#2.2 daily portolfio value

#2.3 view asset allocations of the stocks at the current day





