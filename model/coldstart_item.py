# -*- coding: utf-8 -*-
# @Author: Klaus
# @Date:   2019-09-05 23:46:50
# @Last Modified by:   sniky-lyu
# @Last Modified time: 2019-10-08 20:32:31
# ------------------------------------------------------------------------------
# Description: 基于内容的KNN算法实现物品冷启动
# >> 建立 word-item 倒排表
# >> 计算item间的相似度
# >> 按相似度降序排列
# Dataset: ml-1m
# train / test: {user1:[item1, item2, ...], user2:[...], ...}
# content: {1: ['Animation', "Children's", 'Comedy\\n']} --> {item/movie_id, [labels_seq]}
# ------------------------------------------------------------------------------
import sys
sys.path.append("..")

import random
import math
import numpy as np
from utility.decora import timmer
# from utility.metrics import Metric

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
            # print(line)                                   # 1::1193::5::978300760
            # print(data)                                   # [(1, 1193)] --> [(user, item/movie)]
            # break
        contents = {}                                       # {1: ['Animation', "Children's", 'Comedy\\n']} --> {movie_id, [labels_seq]}
        for line in open(ip, 'rb'):                         # 'open中用mode='r'会有编码问题，illegal multibyte sequence，因此换成'rb', 字节模式
            line = str(line.strip())[2:-1]                  # 去掉换行符后的字节转str的 b"和"
            # print(line)                                   # 1::Toy Story (1995)::Animation|Children's|Comedy
            temp = line.strip().split('::')
            # print(temp, type(temp))                       # ['1', 'Toy Story (1995)', "Animation|Children's|Comedy"] <class 'list'>
            contents[int(temp[0])] = temp[-1].split('|')
            # print(contents)                               # {1: ['Animation', "Children's", 'Comedy\\n']}
            # break
        return data, contents

    def splitData(self, M, k, seed=1):
        '''
        :params: data, 加载的所有(user, item)数据条目
        :params: M, 划分的数目，最后需要取M折的平均
        :params: k, 本次是第几次划分，k~[0, M)
        :params: seed, random的种子数，对于不同的k应设置成一样的
        :return: train, test
        '''
        train, test = [], []
        random.seed(seed)
        for user, item in self.data:
            if random.randint(0, M - 1) == k:
                test.append((user, item))
            else:
                train.append((user, item))

        # 处理成字典形式 {user:[item1, item2, ...], [...], ...}
        def convert_dict(data):
            data_dict = {}
            for user, item in data:
                if user not in data_dict.keys():
                    data_dict[user] = set()
                data_dict[user].add(item)               # tuple的方法，tuple.add(ele)
            data_dict = {k: list(data_dict[k]) for k in data_dict.keys()}
            return data_dict

        return convert_dict(train), convert_dict(test), self.content


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
            if user not in self.train.keys():
                print('user-{} in test is not in train!!')
                continue
            for item in self.train[user]:                               # test中user在train中包含的所有item
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
                if item in item_popularity:
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


def ContentItemKNN(train, content, K, N):
    '''
    :params: train, 训练数据
    :params: content, 物品内容信息
    :params: K, 取相似Top-K相似物品
    :params: N, 推荐TopN物品的个数
    :return: GetRecommendation, 获取推荐结果的接口
    '''
    # 建立 word-item 倒排表 (出现则值为1)
    word_item = {}
    for item in content.keys():
        for word in content[item]:
            if word not in word_item.keys():
                word_item[word] = {}
            word_item[word][item] = 1

    for word in word_item.keys():
        for item in word_item[word]:
            word_item[word][item] /= math.log(1 + len(word_item[word]))

    # 计算物品之间的余弦相似度
    item_sim = {}
    mo = {}
    for word in word_item.keys():
        for u in word_item[word]:
            if u not in item_sim:
                item_sim[u] = {}
                mo[u] = 0
            mo[u] += word_item[word][u] ** 2    # 包含关键词word的item, 余弦相似度分母之一
            for v in word_item[word]:
                if v == u:
                    continue
                if v not in item_sim[u]:        # {user1: {u2:v2, u3:v3, ...}}
                    item_sim[u][v] = 0
                # 余弦相似度分子部分：同时对包含关键词word的item之间的相关性
                item_sim[u][v] += word_item[word][u] * word_item[word][v]
    for u in item_sim.keys():
        for v in item_sim[u].keys():
            # 计算余弦相似度：除以分母
            item_sim[u][v] /= math.sqrt(mo[u] * mo[v])

    # 对item_sim={u1:{u2:v2, u3:v3,...}}每一行u对应的u_other按value降序排序
    sorted_item_sim = {}
    for u, v_dict in item_sim.items():               # v是u对应的u的集合
        sorted_item_sim[u] = sorted(v_dict.items(), key=lambda x: x[1], reverse=True)
    # print(sorted_item_sim, '-----------')     # {1: [(1064, 1.0), (2141, 1.0), ...], 2:[...], ...}

    # 获取接口函数
    def GetRecommendation(user):
        '''
        遍历user的movie列表，对每一个movie_row, 取与movie_row相似的topK个movie添加到推荐列表中(以相似度作为value)
        获得一个包含每一个movie_row的字典{topk_movie, topk_movie_sim},添加到rank字典，rank以topk_movie作为key，topk_movie_sim为value
        对rank按照value排序，取topN作为推荐列表
        '''
        items_rank = {}
        seen_items = set(train[user]) if user in train.keys() else set()
        for item in train[user]:
            for u, _ in sorted_item_sim[item][:K]:        # sorted_item_sim[item] --> list [(1064, 1.0), (2141, 1.0), ...]
                if u not in seen_items:
                    if u not in items_rank:
                        items_rank[u] = 0
                    items_rank[u] += item_sim[item][u]
        recs = sorted(items_rank.items(), key=lambda x: x[1], reverse=True)
        return recs

    return GetRecommendation


class Experiment():
    def __init__(self, M, N, K, fp='../data/ml-1m/ratings.dat', ip='../data/ml-1m/movies.dat'):
        '''
        :params: M, 进行多少次实验
        :params: N, TopN推荐物品的个数
        :params: K, 取Top-K相似物品数目
        :params: fp, 数据文件路径
        :params: ip, 物品内容路径
        '''
        self.M = M
        self.K = K
        self.N = N
        self.fp = fp
        self.ip = ip
        self.alg = ContentItemKNN

    # 定义单次实验
    @timmer
    def worker(self, train, test, content):
        '''
        :params: train, 训练数据集
        :params: test, 测试数据集
        :return: 各指标的值
        '''
        getRecommendation = self.alg(train, content, self.K, self.N)
        metric = Metric(train, test, getRecommendation)
        return metric.eval()

    # 多次实验取平均
    @timmer
    def run(self):
        metrics = {'Precision': 0, 'Recall': 0, 'Coverage': 0, 'Popularity': 0, 'Diversity':0}
        dataset = Dataset(self.fp, self.ip)
        for ii in range(self.M):
            train, test, content = dataset.splitData(self.M, ii)
            print('Experiment {}:'.format(ii))
            metric = self.worker(train, test, content)
            metrics = {k: metrics[k] + metric[k] for k in metrics}
        metrics = {k: metrics[k] / self.M for k in metrics}
        print('Average Result (M={}, N={}, K={}): {}'.format(self.M, self.N, self.K, metrics))


if __name__ == '__main__':
    # fp='../data/ml-1m/ratings.dat'
    # ip='../data/ml-1m/movies.dat'
    # dataset = Dataset(fp, ip)
    M, N, K = 8, 10, 15
    exp = Experiment(M, N, K)
    exp.run()
