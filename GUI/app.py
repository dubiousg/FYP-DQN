import tkinter as tk
from tkinter import *
import os

root = tk.Tk()

height = 800
width = 1200
canvas = tk.Canvas(root, height=height, width=width, bg="#263d42")
canvas.pack()

frame = tk.Frame(root, bg="#3e646c")

frame.place(relwidth=0.85, relheight=0.85, rely=0.1, relx=0.1)

btns = {}
labels = {}
#btn = Button(frame, text="Click", fg="#263d42")
#btn.place(relwidth=0.15, relheight=0.15, rely=0.1, relx=0.1)

#btn.grid(row=0)

##todo list:
#1 run a trading period with a finished model

btns["run"] = Button(frame, text="Play Trading Period", fg="#263d42")
btns["run"].grid(row=0)

labels["run"] = Label(frame, text="")
#labels["run"].grid(row=1, column=0)


#2 create analytics for
#2.1 yearly roi



#2.2 daily portolfio value

#2.3 view asset allocations of the stocks at the current day



root.mainloop()

