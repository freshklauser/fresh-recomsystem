# -*- coding: utf-8 -*-
# @Author: F1684324
# @Date:   2019-08-28 16:40:23
# @Last Modified by:   Klaus
# @Last Modified time: 2019-08-30 22:21:21
# ------------------------------------------------------------------------------
# Description:
# MostPopular
# GenderMostPopular
# AgeMostPopular
# CountryMostPopular
# DemographicMostPopular
# Notes:
# 1) adit = {"a": 1, "b": 2}; temp = adit; temp['c'] = 3, 则 adit = {"a": 1, "b": 2, "c": 3}
# 2) 若 temp = adit.copy(); temp['d'] = 12, 则 adit不变，仍是 adit = {"a": 1, "b": 2, "c": 3}
# ------------------------------------------------------------------------------

import sys
sys.path.append('..')

from utility.dataset import Dataset
from utility.dataset import convert_path
from utility.metrics import Metric
from utility.decora import timmer


def MostPopular(train, profile, N):
    ''' MostPopular算法
    :params: train, 训练数据
    :params: profile, 用户的注册信息
    :params: N, 推荐TopN物品的个数
    :return: GetRecommendation, 获取推荐结果的接口
    '''
    items = {}
    for user in train:
        for item in train[user]:
            if item not in items:
                items[item] = 0
            items[item] += 1
    items = sorted(items.items(), key=lambda x: x[1], reverse=True)
    # 获取接口函数

    def GetRecommendation(user):
        seen_items = set(train[user]) if user in train else set()
        recs = [x for x in items if x[0] not in seen_items][:N]
        return recs
    return GetRecommendation


def GenderMostPopular(train, profile, N):
    ''' GenderMostPopular
    :params: train, 训练数据
    :params: profile, 用户的注册信息
    :params: N, 推荐TopN物品的个数
    :return: GetRecommendation, 获取推荐结果的接口
    '''
    mitems, fitems = {}, {}
    for user in train.keys():
        if profile[user]['gender'] == 'm':
            temp = mitems                                           # 赋值（注意赋值、浅拷贝和深拷贝的区别）
        elif profile[user]['gender'] == 'f':
            temp = fitems
        for item in train[user]:
            if item not in temp.keys():
                temp[item] = 0
            temp[item] += 1
    mitems = sorted(mitems.items(), key=lambda x: x[1], reverse=True)
    fitems = sorted(fitems.items(), key=lambda x: x[1], reverse=True)
    # [('b', 32), ('d', 20), ('c', 3), ('a', 1)] 格式

    mostPopular = MostPopular(train, profile, N)

    # 获取接口函数
    def GetRecommendation(user):
        seen_items = set(train[user]) if user in train else set()
        if profile[user]['gender'] == 'm':
            recs = [x for x in mitems if x[0] not in seen_items][:N]
        elif profile[user]['gender'] == 'f':
            recs = [x for x in fitems if x[0] not in seen_items][:N]
        else:
            recs = mostPopular(user)                                # 没有提供性别信息的，按照MostPopular推荐
        return recs

    return GetRecommendation


def AgeMostPopular(train, profile, N):
    ''' GenderMostPopular
    :params: train, 训练数据
    :params: profile, 用户的注册信息
    :params: N, 推荐TopN物品的个数
    :return: GetRecommendation, 获取推荐结果的接口
    '''
    # 年龄分段
    ages = set()
    for user in profile.keys():
        if profile[user]['age'] >= 0:
            ages.add(profile[user]['age'])
    maxAge, minAge = max(ages), min(ages)
    items = [{} for _ in range(int(maxAge) // 10 + 1)]              # 分成 int(maxAge) // 10 +1 个年龄段，再汇总每个年龄段的popular物品

    # 汇总统计每个年龄段的物品列表（按流行度降序）


def CountryMostPopular(train, profile, N):
    pass


def DemographicMostPopular(train, profile, N):
    pass


class Experiment():
    def __init__(self, M, N, method='MostPopular',
                 fp='../data/lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv',
                 up='../data/lastfm-dataset-360K/usersha1-profile.tsv'):
        '''
        :params: M, 进行多少次实验
        :params: N, TopN推荐物品的个数
        :params: fp, 数据文件路径
        :params: up, 用户注册信息文件路径
        '''
        self.M = M
        self.N = N
        self.fp = fp
        self.up = up
        self.method = method
        self.alg = {'MostPopular': MostPopular, 'GenderMostPopular': GenderMostPopular,
                    'AgeMostPopular': AgeMostPopular, 'CountryMostPopular': CountryMostPopular,
                    'DemographicMostPopular': DemographicMostPopular}

    # @timmer
    def worker(self, train, test, profile):
        '''
        :params: train, 训练数据集
        :params: test, 测试数据集
        :params: profile, 用户注册信息
        :return: 各指标的值
        '''
        getRecommendation = self.alg[self.method](train, profile, self.N)
        metric = Metric(train, test, getRecommendation)
        return metric.eval()

    # @timmer
    def run(self):
        metrics = {'Precision': 0, 'Recall': 0, 'Coverage': 0, 'Popularity': 0}
        dataset = Dataset(self.fp, self.up)
        for i in range(self.M):
            train, test, profile = dataset.splitData(self.M, i)
            print('Experiment {}: '.format(i))
            metric = self.worker(train, test, profile)
            metrics = {k: metrics[k] + metric[k] for k in metrics.keys()}
        metrics = {k: metrics[k] / self.M for k in metrics.keys()}
        print('Average Result (M={}, N={}): {}'.format(self.M, self.N, metrics))


if __name__ == '__main__':
    # filepath = convert_path(r'..\data\lastfm-dataset-360K\usersha1-artmbid-artname-plays.tsv')
    # ext1filepath = convert_path(r'..\data\lastfm-dataset-360K\usersha1-profile.tsv')
    # dataset = Dataset(filepath, ext1filepath)
    # train, test, profile = dataset.splitData(5, 2)
    M = 5
    N = 10
    method = 'GenderMostPopular'
    exp = Experiment(M, N, method=method)
    exp.run()
