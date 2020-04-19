import time

#Timer class
#   *timer class for time testing (benchmarking)
class Timer:

    #desc: constructor, sets start and end timestamps to 0
    def __init__(self):
        self.start = 0
        self.end = 0
    
    #desc: collects timestamp when called
    def start_timer(self):
        self.start = time.time()

    #desc: gets a timestamp to compare to start time
    #output: the difference between end and start timestamp in milliseconds
    def get_time(self):
        self.end = time.time()
        return (self.end - self.start) * 1000 

