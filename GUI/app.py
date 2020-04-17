import os
import matplotlib.pyplot as plt
import tkinter as tk
from model.market import Market_Environment
import model.trader as trader
from tkinter import *

class GUI:

    def __init__(self):
        self.states = { "testing": False, "tested": False } #state of the gui
        self.root = tk.Tk()
        self.height = 800
        self.width = 1200

        self.canvas = tk.Canvas(self.root, height=self.height, width=self.width, bg="#263d42")
        self.canvas.pack()

        self.frame = tk.Frame(self.root, bg="#3e646c")

        self.frame.place(relwidth=0.85, relheight=0.85, rely=0.1, relx=0.1)

        self.btns = {}
        self.labels = {}

        self.test_list = []

        self.btns["run"] = Button(self.frame, text="Play Trading Period", fg="#263d42")
        self.btns["run"].grid(row=0)
        self.btns["run"].bind("<Button-1>", self.run_test)

        self.btns["graph"] = Button(self.frame, text="Create Graph", fg="#263d42")
        self.btns["graph"].grid(row=0, column=1)
        self.btns["graph"].bind("<Button-1>", self.update_chart)

        self.labels["run"] = Label(self.frame, text="Not Running")

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

    def pie_chart(self, event):
        #https://matplotlib.org/3.1.1/gallery/pie_and_polar_charts/pie_features.html
        data = []
        for item in self.test_list:
            data.append(item["allocation"])

        labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
        sizes = [15, 30, 45, 10]
        explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


    def run_gui(self):
        self.root.mainloop()

##todo list:
#1 run a trading period with a finished model






#2 create analytics for
#2.1 yearly roi



#2.2 daily portolfio value

#2.3 view asset allocations of the stocks at the current day





