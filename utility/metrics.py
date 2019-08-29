# -*- coding: utf-8 -*-
# @Author: F1684324
# @Date:   2019-08-28 09:58:00
# @Last Modified by:   F1684324
# @Last Modified time: 2019-08-28 16:39:49

import math


class Metric():
    '''  推荐系统的评价指标 '''

    def __init__(self, train, test, GetRecommendation):
        '''
        :params: train, 训练数据
        :params: test, 测试数据
        :params: GetRecommendation, 为某个用户获取推荐物品的接口函数
        '''
        self.train = train
        self.test = test
        self.GetRecommendation = GetRecommendation
        self.recs = self.getRec()

    def getRec(self):
        '''为 test 中的每个用户进行推荐'''
        recs = {}
        count = 0
        for user in self.test.keys():
            rank = self.GetRecommendation(user)  # rank为列表
            recs[user] = rank
            count += 1
        print('Count of test data: {}'.format(count))
        return recs

    def precision(self):
        '''准确率：最终的推荐列表中有多少比例是 **发生过的用户-物品行为记录** '''
        All, hit = 0, 0
        for user in self.test.keys():
            test_items = set(self.test[user])                           # 用户在test中喜欢的item 集合 T(u)
            rank = self.recs[user]                                      # 对用户推荐的N个item 列表 （rank为列表）
            for item, score in rank:
                if item in test_items:
                    hit += 1                                            # 分子： sum(T(u) & R(u)))
            All += len(rank)                                            # 分母： sum(R(u))
        return round(hit / All * 100, 2)                                # precision = (T(u) & R(u)) / R(u)

    def recall(self):
        '''召回率：有多少比例的 **用户-物品行为记录** 包含在最终的推荐列表中'''
        All, hit = 0, 0
        for user in self.test.keys():
            test_items = set(self.test[user])
            rank = self.recs[user]
            for item, score in rank:
                if item in test_items:
                    hit += 1                                            # 分子： sum(T(u) & R(u)))
            All += len(test_items)                                      # 分母： sum(T(u))
        return round(hit / All * 100, 2)                                # recall = (T(u) & R(u)) / T(u)

    def coverage(self):
        '''覆盖率：最终的推荐列表中包含多大比例的 **物品**'''
        all_item, recom_item = set(), set()
        for user in self.test.keys():                                   # test 中的 user
            for item in self.train[user]:                               # test中user在train中包含的所有item
                if user not in self.train.keys():
                    continue
                all_item.add(item)                                      # 所有物品集合 I
            rank = self.recs[user]
            for item, score in rank:
                recom_item.add(item)                                    # 推荐的物品集合 R(u)
        return round(len(recom_item) / len(all_item) * 100, 2)          # coverage = #R(u) / #I

    def popularity(self):
        '''新颖度：推荐的是目标用户喜欢的但未发生过的用户-行为
           用推荐列表中物品的平均流行度来度量新颖度(新颖度越高，流行度越低)'''
        item_popularity = {}                                            # 计算item 的流行度 （train set）
        for user, items in self.train.items():
            for item in items:
                if item not in item_popularity:
                    item_popularity[item] = 0
                item_popularity[item] += 1                              # 若item在train的user中发生过记录，则该item的流行度+1

        popular = 0
        num = 0
        for user in self.test.keys():
            rank = self.recs[user]                                      # 向test中的 user 推荐topN 物品
            for item, score in rank:
                # 对每个物品的流行度取对数运算, 防止长尾问题带来的北流行物品主导的推荐（避免热门推荐）
                popular += math.log(1 + item_popularity[item])
                num += 1                                                # 汇总所有user的总推荐物品个数
        return round(popular / num, 6)                                  # 计算平均流行度 = popular / n

    def eval(self):
        ''' 汇总 metric 指标 '''
        metric = {'Precision': self.precision(),
                  'Recall': self.recall(),
                  'Coverage': self.coverage(),
                  'Popularity': self.popularity()}
        print('Metric: {}'.format(metric))
        return metric
