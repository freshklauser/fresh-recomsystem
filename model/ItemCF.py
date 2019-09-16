# -*- coding: utf-8 -*-
# @Author: Klaus
# @Date:   2019-08-04 23:04:28
# @Last Modified by:   KlausLyu
# @Last Modified time: 2019-09-16 15:46:38
# ------------------------------------------------------------------------------
# 代码模块化在不同的py文件中，用itemCF来做比赛的Baseline
# 将比赛数据转化为 {user1:{item1, item2, item3, ...},
#                   user2:{item1, item2, item3, ...},
#                   ....}
# 后续可以考虑以 数组的形式来保存数据额和完成代码
# --->
# TO DO LIST:
# LFM隐语义模型
# ------------------------------------------------------------------------------
import sys
sys.path.append("..")

import math
from utility.metrics import Metric
from utility.dataset import Dataset
from utility.decora import timmer


def ItemCF(train, K, N):
    '''
    :params: train, 训练数据集
    :params: K, 超参数，设置取TopK相似物品数目
    :params: N, 超参数，设置取TopN推荐物品数目
    :return: GetRecommendation, 推荐接口函数
    '''
    # 计算 items--items 的稀疏矩阵
    # 两两item同时出现的次数 #(item1, item2). {item1:{item1,item2,item3,..}, item2:{item1,item2,item3,..},...}
    cmat = {}
    num = {}         # 单一item出现的次数 #item1  (相当于行索引的item出现的次数)
    for user in train.keys():
        items = train[user]
        for item1 in items:
            if item1 not in num.keys():  # 单一item出现的次数
                num[item1] = 0
            num[item1] += 1
            if item1 not in cmat.keys():
                cmat[item1] = {}
            for item2 in items:
                if item2 == item1:
                    continue
                if item2 not in cmat[item1]:
                    cmat[item1][item2] = 0
                cmat[item1][item2] += 1

    # 计算余弦相似度
    sim = {}          # 初始化 相似度矩阵
    for i in cmat.keys():
        sim[i] = {}   # 初始化 sim[i]，确保sim[i]也是dict
        for j, cij in cmat[i].items():
            sim[i][j] = cij / math.sqrt(num[i] * num[j])

    # 按照相似度的值对矩阵的每一行进行降序排序
    sim_item_sorted = {}
    for key, values in sim.items():
        sim_item_sorted[key] = sorted(values.items(), key=lambda x: x[1], reverse=True)[:K]
        # sorted函数返回的是列表 list

    # 为待推荐的用户获取推荐接口函数
    def GetRecommendation(user):
        rank = {}                                   # 待推荐列表  {item1:rank1, item2:rank2,...}
        # 用户见过的item列表 [item1, item2, item3, ...]
        interacted_items = set(train[user])
        # print(interacted_items, '---------------')
        # 根据相似度高的用户的列表对user进行推荐（去掉user见过的item）
        for item in train[user]:                    # 遍历user的物品列表
            for i, _ in sim_item_sorted[item]:            # 与排序后的topK个相似物品进行相似度计算
                if i not in interacted_items:       # topK中的item不在user的已有列表中
                    if i not in rank.keys():        # topK中的item不在待推荐列表中
                        rank[i] = 0              # 不在rank中则添加进去并赋初值为0
                    # topK与用户已有items的两两共现余弦相似度矩阵
                    rank[i] += sim[item][i]
        # 对rank字典排序，获得 topN 对应的 item
        rank_sorted = sorted(rank.items(), key=lambda x: x[1], reverse=True)[:N]
        '''若只保存 item，不需要#item: [item1, item2, item3, ...]'''
#         rank_sorted = list(map(lambda x: x[0], rank_sorted))
        # 返回值是列表
        return rank_sorted

    return GetRecommendation


def ItemIUF(train, K, N):
    '''
    :params: train, 训练数据集
    :params: K, 超参数，设置取TopK相似物品数目
    :params: N, 超参数，设置取TopN推荐物品数目
    :return: GetRecommendation, 推荐接口函数
    '''
    # 计算 items--items 的稀疏矩阵
    # 两两item同时出现的次数 #(item1, item2). {item1:{item1,item2,item3,..}, item2:{item1,item2,item3,..},...}
    cmat = {}
    num = {}         # 单一item出现的次数 #item1  (相当于行索引的item出现的次数)
    for user in train.keys():
        items = train[user]
        for item1 in items:
            if item1 not in num.keys():  # 单一item出现的次数
                num[item1] = 0
            num[item1] += 1
            if item1 not in cmat.keys():
                cmat[item1] = {}
            for item2 in items:
                if item2 == item1:
                    continue
                if item2 not in cmat[item1]:
                    cmat[item1][item2] = 0
                cmat[item1][item2] += 1 / math.log(1 + len(items))

    # 计算余弦相似度
    sim = {}          # 初始化 相似度矩阵
    for i in cmat.keys():
        sim[i] = {}   # 初始化 sim[i]，确保sim[i]也是dict
        for j, cij in cmat[i].items():
            sim[i][j] = cij / math.sqrt(num[i] * num[j])

    # 按照相似度的值对矩阵的每一行进行降序排序
    sim_item_sorted = {}
    for key, values in sim.items():
        sim_item_sorted[key] = sorted(values.items(), key=lambda x: x[1], reverse=True)[:K]
        # sorted函数返回的是列表 list

    # 为待推荐的用户获取推荐接口函数
    def GetRecommendation(user):
        rank = {}                                   # 待推荐列表  {item1:rank1, item2:rank2,...}
        # 用户见过的item列表 [item1, item2, item3, ...]
        interacted_items = set(train[user])
        # 根据相似度高的用户的列表对user进行推荐（去掉user见过的item）
        for item in train[user]:                    # 遍历user的物品列表
            for i, _ in sim_item_sorted[item]:            # 与排序后的topK个相似物品进行相似度计算
                if i not in interacted_items:       # topK中的item不在user的已有列表中
                    if i not in rank.keys():        # topK中的item不在待推荐列表中
                        rank[i] = 0              # 不在rank中则添加进去并赋初值为0
                    # topK与用户已有items的两两共现余弦相似度矩阵
                    rank[i] += sim[item][i]
        # 对rank字典排序，获得 topN 对应的 item
        rank_sorted = sorted(
            rank.items(), key=lambda x: x[1], reverse=True)[:N]
        '''若只保存 item，不需要#item: [item1, item2, item3, ...]'''
#         rank_sorted = list(map(lambda x: x[0], rank_sorted))
        # 返回值是列表
        return rank_sorted

    return GetRecommendation


def ItemCF_Norm(train, K, N):
    '''
    :params: train, 训练数据集
    :params: K, 超参数，设置取TopK相似物品数目
    :params: N, 超参数，设置取TopN推荐物品数目
    :return: GetRecommendation, 推荐接口函数
    '''
    # 计算 items--items 的稀疏矩阵
    # 两两item同时出现的次数 #(item1, item2). {item1:{item1,item2,item3,..}, item2:{item1,item2,item3,..},...}
    cmat = {}
    num = {}         # 单一item出现的次数 #item1  (相当于行索引的item出现的次数)
    for user in train.keys():
        items = train[user]
        for item1 in items:
            if item1 not in num.keys():  # 单一item出现的次数
                num[item1] = 0
            num[item1] += 1
            if item1 not in cmat.keys():
                cmat[item1] = {}
            for item2 in items:
                if item2 == item1:
                    continue
                if item2 not in cmat[item1]:
                    cmat[item1][item2] = 0
                cmat[item1][item2] += 1 / math.log(1 + len(items))

    # 计算余弦相似度
    sim = {}          # 初始化 相似度矩阵
    for i in cmat.keys():
        sim[i] = {}   # 初始化 sim[i]，确保sim[i]也是dict
        for j, cij in cmat[i].items():
            sim[i][j] = cij / math.sqrt(num[i] * num[j])

    '''相似度矩阵归一化'''
    for i in sim.keys():
        s_max = 0                # 每遍历一行之前先初始化s_max为0
        for j in sim[i].keys():
            if sim[i][j] >= s_max:
                s_max = sim[i][j]
            sim[i][j] /= s_max

    # 按照相似度的值对矩阵的每一行进行降序排序
    sim_item_sorted = {}
    for key, values in sim.items():
        sim_item_sorted[key] = sorted(values.items(), key=lambda x: x[1], reverse=True)[:K]
        # sorted函数返回的是列表 list

    # 为待推荐的用户获取推荐接口函数
    def GetRecommendation(user):
        rank = {}                                   # 待推荐列表  {item1:rank1, item2:rank2,...}
        # 用户见过的item列表 [item1, item2, item3, ...]
        interacted_items = set(train[user])
        # 根据相似度高的用户的列表对user进行推荐（去掉user见过的item）
        for item in train[user]:                    # 遍历user的物品列表
            # 与排序后的topK个相似物品进行相似度计算
            for i, _ in sim_item_sorted[item]:
                if i not in interacted_items:       # topK中的item不在user的已有列表中
                    if i not in rank.keys():        # topK中的item不在待推荐列表中
                        rank[i] = 0              # 不在rank中则添加进去并赋初值为0
                    # topK与用户已有items的两两共现余弦相似度矩阵
                    rank[i] += sim[item][i]
        # 对rank字典排序，获得 topN 对应的 item
        rank_sorted = sorted(rank.items(), key=lambda x: x[1], reverse=True)[:N]
        '''若只保存 item，不需要#item: [item1, item2, item3, ...]'''
#         rank_sorted = list(map(lambda x: x[0], rank_sorted))
        # 返回值是列表
        return rank_sorted

    return GetRecommendation


class Experiment():
    def __init__(self, M, K, N, fp=r'..\data\ml-1m\ratings.dat', method='ItemCF'):
        '''
        :params: M, 交叉验证实验次数
        :params: K, TopK相似物品的个数
        :params: N, TopN推荐物品的个数
        :params: fp, 数据文件路径
        :params: method, 推荐算法
        '''
        self.M = M
        self.K = K
        self.N = N
        self.fp = fp
        self.method = method
        self.alg = {"ItemCF": ItemCF, "ItemIUF": ItemIUF, "ItemCF_Norm": ItemCF_Norm}

    @timmer
    def worker(self, train, test):
        '''
        :params: train, 训练数据集
        :params: test, 测试数据集
        :return: 各指标的值
        '''
        getRecommendation = self.alg[self.method](train, self.K, self.N)
        metric = Metric(train, test, getRecommendation)
        return metric.eval()

    @timmer
    def run(self):
        metrics = {'Precision': 0, 'Recall': 0, 'Coverage': 0, 'Popularity': 0, 'Diversity':0}
        dataset = Dataset(self.fp)
        for ii in range(self.M):
            train, test, _ = dataset.splitData(self.M, ii)
            print('Experiment {}:'.format(ii))
            metric = self.worker(train, test)
            metrics = {k: metrics[k] + metric[k] for k in metrics}
        metrics = {k: metrics[k] / self.M for k in metrics}
        print('Average Result (M={}, N={}, ratio={}): {}'.format(self.M, self.N, self.ratio, metrics))


if __name__ == '__main__':
    print('---------------------- ItemCF ----------------------')
    # 1. ItemCF
    M = 5
    N = 10
    for K in [5, 10, 20, 40, 80, 160]:
        cf_exp = Experiment(M, K, N, method='ItemCF')
        cf_exp.run()
    print('---------------------- ItemIUF ----------------------')
    M = 5
    K = 10
    iuf_exp = Experiment(M, K, N, method='ItemIUF')
    iuf_exp.run()
    print('\n')
    print('---------------------- ItemCF_Norm ----------------------')
    cfnorm_exp = Experiment(M, K, N, method='ItemCF_Norm')
    cfnorm_exp.run()

# ---------------------- ItemCF ----------------------
# Func loadData, run time: 1.3587447
# Func splitData, run time: 1.5897664999999999
# count: 6035
# Metric: {'Precision': 29.69, 'Recall': 8.95, 'Coverage': 22.45, 'Popularity': 7.101221}
# Func worker, run time: 96.853724
# Done!!

# Func run, run time: 99.9004774
# Func loadData, run time: 1.3753652999999986
# Func splitData, run time: 1.8933593999999943
# count: 6035
# Metric: {'Precision': 30.62, 'Recall': 9.22, 'Coverage': 19.55, 'Popularity': 7.182597}
# Func worker, run time: 93.3659322
# Done!!

# Func run, run time: 96.75672019999999
# Func loadData, run time: 1.354055800000026
# Func splitData, run time: 1.6839875000000006
# count: 6035
# Metric: {'Precision': 30.22, 'Recall': 9.11, 'Coverage': 17.24, 'Popularity': 7.277239}
# Func worker, run time: 95.8356875
# Done!!

# Func run, run time: 98.9685244
# Func loadData, run time: 1.2680831999999782
# Func splitData, run time: 1.6084959000000367
# count: 6035
# Metric: {'Precision': 29.46, 'Recall': 8.88, 'Coverage': 15.37, 'Popularity': 7.331447}
# Func worker, run time: 96.5398199
# Done!!

# Func run, run time: 99.51386910000002
# Func loadData, run time: 1.2554929000000357
# Func splitData, run time: 1.542421499999989
# count: 6035
# Metric: {'Precision': 28.42, 'Recall': 8.56, 'Coverage': 13.49, 'Popularity': 7.349231}
# Func worker, run time: 108.44748620000001
# Done!!

# Func run, run time: 111.3402949
# Func loadData, run time: 1.2332144000000085
# Func splitData, run time: 1.5512279000000149
# count: 6035
# Metric: {'Precision': 27.18, 'Recall': 8.19, 'Coverage': 12.06, 'Popularity': 7.32431}
# Func worker, run time: 142.33041429999997
# Done!!

# Func run, run time: 145.2348804
# [Finished in 652.1s]
