import datetime
import numpy as np
import tensorflow as tf
import math
from benchmarking.timer import Timer

arr = np.array([1, 2, 3, 0, -1, float("nan")])
total = 0
for i in range(arr.size):
    if math.isnan(arr[i]) or arr[i] < 0:
        arr[i] = 0

    total += arr[i]


abc = 0
#print(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

