import datetime
#print(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
stock_data = {}

li = []
for i in range(10):
    li.append(i)
    stock_data[str(i)] = li

print(stock_data)
total_days = len(max(stock_data.values(), key = lambda x: len(x)))

print(total_days)