# -*- coding: utf-8 -*-
# @Author: F1684324
# @Date:   2019-08-28 16:40:23
# @Last Modified by:   kkkk
# @Last Modified time: 2019-08-29 15:07:15
# ------------------------------------------------------------------------------
# Description:
# MostPopular
# GenderMostPopular
# AgeMostPopular
# CountryMostPopular
# DemographicMostPopular
# ------------------------------------------------------------------------------

import sys
sys.path.append('..')

from utility.dataset import Dataset, convert_path
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


class Experiment():
    def __init__(self, M, N, at='MostPopular',
                 fp='../dataset/lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv',
                 up='../dataset/lastfm-dataset-360K/usersha1-profile.tsv'):
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
        self.at = at
        self.alg = {'MostPopular': MostPopular, 'GenderMostPopular': GenderMostPopular,
                    'AgeMostPopular': AgeMostPopular, 'CountryMostPopular': CountryMostPopular,
                    'DemographicMostPopular': DemographicMostPopular}

        定义单次实验
        @timmer
        def worker(self, train, test, profile):
            '''
            :params: train, 训练数据集
            :params: test, 测试数据集
            :params: profile, 用户注册信息
            :return: 各指标的值
            '''
            getRecommendation = self.alg[]


if __name__ == '__main__':
    filepath = convert_path(r'..\data\lastfm-dataset-360K\usersha1-artmbid-artname-plays.tsv')
    ext1filepath = convert_path(r'..\data\lastfm-dataset-360K\usersha1-profile.tsv')
    dataset = Dataset(filepath, ext1filepath)
    train, test, profile = dataset.splitData(5,2)
