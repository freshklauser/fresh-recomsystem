# -*- coding: utf-8 -*-
# @Author: KlausLyu
# @Date:   2019-09-17 10:05:19
# @Last Modified by:   KlausLyu
# @Last Modified time: 2019-09-17 15:28:15
# ------------------------------------------------------------------------------
# Description: 給用户推荐标签
# 算法实现
# 1. Popular
# 2. UserPopular
# 3. ItemPopular
# 4. HybridPopular
# ------------------------------------------------------------------------------
import sys
sys.path.append("..")

import random
# from utility.decora import timmer
from operator import itemgetter


class Dataset():
    def __init__(self, fp):
        # fp: data file path
        self.data = self.loadData(fp)

    # @timmer
    def loadData(self, fp):
        data = []
        for line in open(fp).readlines()[1:]:
            data.append(line.strip().split('\t')[:3])
        return data

    # @timmer
    def splitData(self, M, k, seed=1):
        '''
        :params: data, 加载的所有(user, item)数据条目
        :params: M, 划分的数目，最后需要取M折的平均
        :params: k, 本次是第几次划分，k~[0, M)
        :params: seed, random的种子数，对于不同的k应设置成一样的
        :return: train, test
        '''
        # 按照(user, item)作为key进行划分
        train, test = [], []
        random.seed(seed)
        for user, item, tag in self.data:
            # 这里与书中的不一致，本人认为取M-1较为合理，因randint是左右都覆盖的
            if random.randint(0, M - 1) == k:
                test.append((user, item, tag))
            else:
                train.append((user, item, tag))

        # 处理成字典的形式，user->set(items)
        def convert_dict(data):
            data_dict = {}
            for u, i, t in data:
                if u not in data_dict.keys():
                    data_dict[u] = {}
                if i not in data_dict[u].keys():
                    data_dict[u][i] = set()                 # set() 去重
                data_dict[u][i].add(t)
            for u in data_dict.keys():
                for i in data_dict[u].keys():
                    data_dict[u][i] = list(data_dict[u][i])  # 将tuple转化为list
            return data_dict

        return convert_dict(train), convert_dict(test)


class Metric():

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

    # 为test中的每个用户进行推荐
    def getRec(self):
        recs = {}
        for user in self.test:
            recs[user] = {}
            for item in self.test[user]:
                rank = self.GetRecommendation(user, item)
                recs[user][item] = rank
        return recs

    # 定义精确率指标计算方式
    def precision(self):
        All, hit = 0, 0
        for user in self.test.keys():
            for item in self.test[user].keys():
                test_tags = set(self.test[user][item])
                rank = self.recs[user][item]
                for tag, score in rank:
                    if tag in test_tags:
                        hit += 1
                All += len(rank)
        return round(hit / All * 100, 2)

    # 定义召回率指标计算方式
    def recall(self):
        All, hit = 0, 0
        for user in self.test.keys():
            for item in self.test[user].keys():
                test_tags = set(self.test[user][item])
                rank = self.recs[user][item]
                for tag, score in rank:
                    if tag in test_tags:
                        hit += 1
                All += len(test_tags)
        return round(hit / All * 100, 2)

    def eval(self):
        metric = {'Precision': self.precision(),
                  'Recall': self.recall()}
        # print('Metric:', metric)
        return metric


# 1. 推荐热门标签
# @timmer
def Popular(train, N):
    '''
    :params: train, 训练数据集
    :params: N, 超参数，设置取TopN推荐物品数目
    :return: GetRecommendation，推荐接口函数
    '''
    tags = {}
    for u in train.keys():
        for i in train[u].keys():
            for t in train[u][i]:
                if t not in tags:
                    tags[t] = 0
                tags[t] += 1
    tags = sorted(tags.items(), key=itemgetter(1), reverse=True)[:N]

    def GetRecommendation(user, item):
        return tags

    return GetRecommendation


# 2. 推荐用户最热门的标签
# @timmer
def UserPopular(train, N):
    '''
    :params: train, 训练数据集
    :params: N, 超参数，设置取TopN推荐物品数目
    :return: GetRecommendation，推荐接口函数
    '''
    user_tags = {}
    for u in train.keys():
        if u not in user_tags.keys():
            user_tags[u] = {}
        for i in train[u].keys():
            for t in train[u][i]:
                if t not in user_tags[u]:
                    user_tags[u][t] = 0
                user_tags[u][t] += 1
    user_tags = {k: sorted(v.items(), key=itemgetter(1), reverse=True) for k, v in user_tags.items()}

    def GetRecommendation(user, item):
        return user_tags[user][:N] if user in user_tags.keys() else []

    return GetRecommendation


# 3. 推荐物品最热门的标签
# @timmer
def ItemPopular(train, N):
    '''
    :params: train, 训练数据集
    :params: N, 超参数，设置取TopN推荐物品数目
    :return: GetRecommendation，推荐接口函数
    '''
    item_tags = {}
    for u in train.keys():
        for i in train[u].keys():
            if i not in item_tags.keys():
                item_tags[i] = {}
            for t in train[u][i]:
                if t not in item_tags[i].keys():
                    item_tags[i][t] = 0
                item_tags[i][t] += 1
    item_tags = {k: sorted(v.items(), key=itemgetter(1), reverse=True) for k, v in item_tags.items()}

    def GetRecommendation(user, item):
        return item_tags[item][:N] if item in item_tags.keys() else []

    return GetRecommendation


# 4. 联合用户和物品进行推荐
# @timmer
def HybridPopular(train, N, alpha):
    '''
    :params: train, 训练数据集
    :params: N, 超参数，设置取TopN推荐物品数目
    :params: alpha，超参数，设置用户和物品的融合比例
    :return: GetRecommendation，推荐接口函数
    '''
    # 统计user_tags
    user_tags = {}
    for u in train.keys():
        if u not in user_tags.keys():
            user_tags[u] = {}
        for i in train[u].keys():
            for t in train[u][i]:
                if t not in user_tags[u]:
                    user_tags[u][t] = 0
                user_tags[u][t] += 1

    # 统计item_tags
    item_tags = {}
    for user in train:
        for item in train[user]:
            if item not in item_tags:
                item_tags[item] = {}
            for tag in train[user][item]:
                if tag not in item_tags[item]:
                    item_tags[item][tag] = 0
                item_tags[item][tag] += 1

    def GetRecommendation(user, item):
        rank = {}
        if user in user_tags.keys():
            max_user_tag_w = max(user_tags[user].values())      # 对用户user, tag字典所有值中的最大值，归一化用
            for tag in user_tags[user].keys():
                if tag not in rank.keys():
                    rank[tag] = 0
                rank[tag] += (1 - alpha) * user_tags[user][tag] / max_user_tag_w
        if item in item_tags.keys():
            max_item_tag_w = max(item_tags[item].values())
            for tag in item_tags[item].keys():
                if tag not in rank.keys():
                    rank[tag] = 0
                rank[tag] += alpha * item_tags[item][tag] / max_item_tag_w
        rank = sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]
        return rank

    return GetRecommendation


class Experiment():
    def __init__(self, M, N, fp='../data/hetrec2011-delicious-2k/user_taggedbookmarks.dat', rt='Popular'):
        '''
        :params: M, 进行多少次实验
        :params: N, TopN推荐物品的个数
        :params: fp, 数据文件路径
        :params: rt, 推荐算法类型
        '''
        self.M = M
        self.N = N
        self.fp = fp
        self.rt = rt
        self.alg = {'Popular': Popular, 'UserPopular': UserPopular,
                    'ItemPopular': ItemPopular, 'HybridPopular': HybridPopular}

    # 定义单次实验
    # @timmer
    def worker(self, train, test, **kwargs):
        '''
        :params: train, 训练数据集
        :params: test, 测试数据集
        :return: 各指标的值
        **kwargs 的使用技巧
        '''
        getRecommendation = self.alg[self.rt](train, self.N, **kwargs)
        metric = Metric(train, test, getRecommendation)
        return metric.eval()

    # 多次实验取平均
    # @timmer
    def run(self, **kwargs):
        metrics = {'Precision': 0, 'Recall': 0}
        dataset = Dataset(self.fp)
        for ii in range(self.M):
            train, test = dataset.splitData(self.M, ii)
            # print('Experiment {}: >>>>>>>>>>>>>>'.format(ii))
            metric = self.worker(train, test, **kwargs)
            metrics = {k: metrics[k] + metric[k] for k in metrics}
        metrics = {k: metrics[k] / self.M for k in metrics}
        print('Average Result (M={}, N={}): {}'.format(self.M, self.N, metrics))


if __name__ == '__main__':
    # 1. SimpleTagBased实验
    print('>>>>>>>>>>>>>>>>>>>>>>>>> Popular <<<<<<<<<<<<<<<<<<<<<<<<<<')
    M, N = 10, 10
    exp = Experiment(M, N, rt='Popular')
    exp.run()

    # 2. TagBasedTFIDF实验
    print('>>>>>>>>>>>>>>>>>>>>>>>>> UserPopular <<<<<<<<<<<<<<<<<<<<<<<<<<<')
    M, N = 10, 10
    exp = Experiment(M, N, rt='UserPopular')
    exp.run()

    # 3. TagBasedTFIDF_imp 实验
    print('>>>>>>>>>>>>>>>>>>>>>>>>> ItemPopular <<<<<<<<<<<<<<<<<<<<<<<')
    M, N = 10, 10
    exp = Experiment(M, N, rt='ItemPopular')
    exp.run()

    print('>>>>>>>>>>>>>>>>>>>>>>>>> HybridPopular <<<<<<<<<<<<<<<<<<<<<<<<<<')
    # 4. ExpandTagBased 实验
    M, N = 10, 10
    for alpha in range(0, 11):
        alpha /= 10
        print('alpha =', alpha)
        exp = Experiment(M, N, rt='HybridPopular')
        exp.run(alpha=alpha)
