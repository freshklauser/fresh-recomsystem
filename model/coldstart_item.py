# -*- coding: utf-8 -*-
# @Author: Klaus
# @Date:   2019-09-05 23:46:50
# @Last Modified by:   sniky-lyu
# @Last Modified time: 2019-09-06 23:57:42
# ------------------------------------------------------------------------------
# Description:
# >> 建立 word-item 倒排表
# >> 计算item间的相似度
# >> 按相似度降序排列
# Dataset: ml-1m
# ------------------------------------------------------------------------------
import sys
sys.path.append("..")

import random
import math
import numpy as np
from utility.decora import timmer

# fp='../data/ml-1m/ratings.dat', ip='../data/ml-1m/movies.dat'
class Dataset():
    def __init__(self, fp, ip):
        ''' fp: data file path , ip: movie info file'''
        self.data, self.content = self.loadData(fp, ip)

    @timmer
    def loadData(self, fp, ip):
        data = []                           # [(1, 1193)] --> [(user, item/movie)]
        for line in open(fp, 'r'):
            data.append(tuple(map(int, line.strip().split('::')[:2])))
            # print(line)                                 # 1::1193::5::978300760
            # print(data)                                 # [(1, 1193)] --> [(user, item/movie)]
            # break
        contents = {}                       # {1: ['Animation', "Children's", 'Comedy\\n']} --> {movie_id, [labels_seq]}
        for line in open(ip, 'rb'):         # 'open中用mode='r'会有编码问题，illegal multibyte sequence，因此换成'rb', 字节模式
            line = str(line.strip())[2:-1]                  # 去掉换行符后的字节转str的 b"和"
            # print(line)                                   # 1::Toy Story (1995)::Animation|Children's|Comedy
            temp = line.strip().split('::')
            # print(temp, type(temp))                       # ['1', 'Toy Story (1995)', "Animation|Children's|Comedy"] <class 'list'>
            contents[int(temp[0])] = temp[-1].split('|')
            # print(contents)                               # {1: ['Animation', "Children's", 'Comedy\\n']}
            # break
        return data, contents


if __name__ == '__main__':
    fp='../data/ml-1m/ratings.dat'
    ip='../data/ml-1m/movies.dat'
    dataset = Dataset(fp, ip)
