import time

from utils import GlobalLogger as mylogger

def log_run_time_ms(func_name):
    def cost_time(func):
        def real_func(*args, **kwargs):
            start = time.time()
            ret = func(*args, **kwargs)
            end = time.time()
            mylogger.debug(func_name,
                           f"cost time: %.5f (ms)" % ((end-start) * 1000))
            return ret
        return real_func
    return cost_time