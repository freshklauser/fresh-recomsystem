# -*- coding: utf-8 -*-
# @Author: F1684324
# @Date:   2019-08-28 10:56:34
# @Last Modified by:   F1684324
# @Last Modified time: 2019-08-28 11:08:14
# ------------------------------------------------------------------------------
# Description: decorator function
# ------------------------------------------------------------------------------
import time


def timmer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        stop_time = time.time()
        print('Func %s, run time: %s' % (func.__name__, stop_time - start_time))
        return res
    return wrapper
