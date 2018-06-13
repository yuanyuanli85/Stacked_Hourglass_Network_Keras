import time


class Timer(object):
    def __init__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(time.time() - self.start_time)

    def __enter__(self):
        return self
