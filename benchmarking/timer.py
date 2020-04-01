import time

class Timer:

    def __init__(self):
        self.start = 0
        self.end = 0
    
    def start_timer(self):
        self.start = time.time()

    def get_time(self):
        self.end = time.time()
        return (self.end - self.start) * 1000 

    #to finish later if needed
    def get_time_str(self):
        self.end = time.time()
        return self.end - self.start 

