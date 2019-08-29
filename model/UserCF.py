# -*- coding: utf-8 -*-
# @Author: F1684324
# @Date:   2019-07-31 09:12:09
# @Last Modified by:   kkkk
# @Last Modified time: 2019-08-29 14:59:49

import sys
sys.path.append("..")

from utility.decora import timmer
from utility.metrics import Metric
from utility.dataset import Dataset
import random


def Random(train, K, N):
    ''' docstring for Random 随机推荐
    :params: train, 训练数据集
    :params: K, 可忽略
    :params: N, 超参数，设置取TopN推荐物品数目
    :return: GetRecommendation, 推荐接口函数
    '''
    items = {}                   # keys: item, value: item出现的次数
    for user in train.keys():
        for item in train[user]:
            if item not in items.keys():
                items[item] = 0  # 若果是新item, 先设置item出现的次数为0，再+1计数
            items[item] += 1     # user中出现一次item 则+1计数一次

    def GetRecommendation(user):
        '''根据items字典中的item出现次数为依据，随机选取topN个未见过的item作为推荐内容'''
        user_items = set(
            train[user])   # 目标用户user的列表中的item集合（推荐的item不应与该集合中的item相同）
        recom_items = {}                # 定义推荐列表，字典（item, #item）
        for item in items.keys():       # items 前文定义的，包含所有 item 的字典(item, #item)
            if item not in user_items:
                # 未见过的item及出现的次数添加为recom_items的元素
                recom_items[item] = items[item]
        # 从recom_items中随机挑选 N个 ： [(item1, #item1), (item2, #item2), ...]
        recom_items = list(recom_items.items())
        random.shuffle(recom_items)
        return recom_items[: N]


class Experiment():

    def __init__(self, M, K, N, fp='../dataset/ml-1m/ratings.dat', rt='UserCF'):
        '''
        :params: M, 进行多少次实验
        :params: K, TopK相似用户的个数
        :params: N, TopN推荐物品的个数
        :params: fp, 数据文件路径
        :params: rt, 推荐算法类型
        '''
        self.M = M
        self.K = K
        self.N = N
        self.fp = fp
        self.rt = rt
        self.alg = {'Random': Random, 'MostPopular': MostPopular,
                    'UserCF': UserCF, 'UserIIF': UserIIF}

    # 定义单次实验
    @timmer
    def worker(self, train, test):
        '''
        :params: train, 训练数据集
        :params: test, 测试数据集
        :return: 各指标的值
        '''
        getRecommendation = self.alg[self.rt](train, self.K, self.N)
        metric = Metric(train, test, getRecommendation)
        return metric.eval()

    # 多次实验取平均
    @timmer
    def run(self):
        metrics = {'Precision': 0, 'Recall': 0,
                   'Coverage': 0, 'Popularity': 0}
        dataset = Dataset(self.fp)
        for ii in range(self.M):
            train, test, _ = dataset.splitData(self.M, ii)
            print('Experiment {}:'.format(ii))
            metric = self.worker(train, test)
            metrics = {k: metrics[k] + metric[k] for k in metrics}
        metrics = {k: metrics[k] / self.M for k in metrics}
        print('Average Result (M={}, K={}, N={}): {}'.format(
            self.M, self.K, self.N, metrics))


if __name__ == "__main__":
    M, K, N = 8, 30, 10
    exp = Experiment(M, K, N)
    exp.run()
