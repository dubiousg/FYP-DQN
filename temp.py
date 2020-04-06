import datetime
import numpy as np
import tensorflow as tf
from benchmarking.timer import Timer
arr = np.atleast_2d(np.array([float('nan'), 123, 12.3], dtype='float32'))
a = tf.convert_to_tensor(arr)
print(a[0])
#print(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

