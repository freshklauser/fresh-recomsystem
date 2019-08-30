# -*- coding: utf-8 -*-
# @Author: Klaus
# @Date:   2019-08-22 20:20:02
# @Last Modified by:   KlausLyu
# @Last Modified time: 2019-08-30 15:06:57
# ------------------------------------------------------------------------------
# 隐语义模型
# ------------------------------------------------------------------------------
import sys
sys.path.append("..")

from utility.decora import timmer
from utility.metrics import Metric
from utility.dataset import Dataset
import numpy as np


def LFM(train, ratio, F, N, alpha, lamb, step):
    '''
    :params: train, 训练数据
    :params: ratio, 正负采样的比例 一般取1，保持数量相近
    :params: F, 隐语义个数
    :params: alpha, 初始学习率
    :params: step, 迭代次数
    :params: lamb, 正则化系数
    :params: N, 推荐TopN物品的个数
    :return: GetRecommendation, 获取推荐结果的接口
    '''
    # 提取总的item集合，即所有用户的不重复item
    all_items = {}
    for user in train.keys():
        for item in train[user]:
            if item not in all_items:
                all_items[item] = 0
            all_items[item] += 1
    # [('a', {'aa': 4}), ('b', {'bb': 14}), ('c', {'cc': 24})]
    all_items = list(all_items.items())
    # 所有物品的列表list
    items = [x[0] for x in all_items]
    # 列表物品对应的流行度
    pops = [x[1] for x in all_items]
    # 即样本集只有正样本，正样本的流行度即为样品被不同用户喜欢的次数
    print('items and their popularity: done')

    # 负采样 ---> 保证正负样本数量比率均衡
    @timmer
    def RandomSelectNegSample(data, ratio):
        new_data = {}       # 包含正负样本的数据集, 正样本兴趣rui为1，负样本兴趣rui为0
        # 正样本  r_u1_i1 = data[user_1][item_1]
        for user in data:
            if user not in new_data.keys():
                new_data[user] = {}
            for item in data[user]:
                new_data[user][item] = 1    # 用户user消费过的物品item，标记为1（表示正样本）
        # 负样本
        print("Negtiva sampling start.")
        for user in new_data.keys():
            seen = set(new_data[user])
            num_pos = len(seen)
            # 从所有物品items中根据流行度pops的值作为概率依据选择pum_pos*ratio*3个样本，依次将这些样本
            # 与有过行为的样品seen做对比，只选取不在seen中的物品，取len(pos)*ratio个样本作为负样本
            item = np.random.choice(items, int(
                num_pos * ratio * 3), pops)      # 用法见tips
            item_neg = []
            count = 0
            for element in item:
                if element not in seen:
                    item_neg.append(element)
                    count += 1
                if count > int(num_pos * ratio):   # 从负样本集合中取正负样本成比率的样本作为负样本
                    break
            new_data[user].update({x: 0 for x in item_neg})  # 负样本标记为0
        print("Negtiva sampling end.")
        return new_data

    # 训练 随机梯度下降 --> 先初始化
    P, Q = {}, {}       # Puk, Qki
    # P：用户u的兴趣和第f个隐类的关系，Q:第f个隐类和物品i的关系
    for user in train.keys():
        P[user] = np.random.random(F)   # P: (u,f)   P[u]:(f,)
    for item in items:
        Q[item] = np.random.random(F)   # Q: (i,f)   Q[i]:(f,)   P[u]*Q[i]:(f,)
    # 迭代
    print('GSD start.')
    for s in range(step):
        data = RandomSelectNegSample(train, ratio)
        for user in data.keys():
            for item in data[user]:
                eui = data[user][item] - \
                    (P[user] * Q[item]).sum()  # 常量如0.76 float
                P[user] += alpha * (eui * Q[item] - lamb * P[user])
                Q[item] += alpha * (eui * P[user] - lamb * Q[item])
                # (f, ) += const * (const * (f, ) - const * (f, ))
        # SGD中需要设置每一次迭代之后调整学习率(衰减)
        alpha *= 0.9
    print('GSD end.')

    # 获取接口函数
    @timmer
    def Recommendation(user):
        seen_items = set(train[user])
        recs = {}
        for item in items:                  # 遍历所有物品
            if item not in seen_items:      # 未消费过，则计算用户u对物品i的兴趣
                # 兴趣rui公式： rui = (Puf * Qif).sum()
                recs[item] = (P[user] * Q[item]).sum()
        # 根据兴趣值recs[item]对recs中的item进行排序 降序，取topN
        print('Get the top N items recommended for {}!'.format(user))
        recs_top = sorted(recs.items(), key=lambda x: x[1], reverse=True)[:N]
        return recs_top

    return Recommendation


# 实验测试
class Experiment():
    def __init__(self, M, N, ratio=1, F=100, alpha=0.02, step=1, lamb=0.01, fp='../data/ml-1m/ratings.dat'):
        '''
        params: M, 进行多少次实验
        params: N, TopN推荐物品的个数
        params: ratio, 正负样本比例
        params: F, 隐语义个数
        params: alpha, 学习率
        params: step, 训练步数
        params: lamb, 正则化系数
        params: fp, 数据文件路径
        '''
        self.M = M
        self.F = F
        self.N = N
        self.ratio = ratio
        self.alpha = alpha
        self.step = step
        self.lamb = lamb
        self.fp = fp
        self.alg = LFM

    @timmer
    def worker(self, train, test):
        '''
        :params: train, 训练数据集
        :params: test, 测试数据集
        :return: 各指标的值
        '''
        recommendation = self.alg(
            train, self.ratio, self.F, self.N, self.alpha, self.lamb, self.step)
        metric = Metric(train, test, recommendation)
        return metric.eval()

    @timmer
    def run(self):
        metrics = {'Precision': 0, 'Recall': 0, 'Coverage': 0, 'Popularity': 0}
        dataset = Dataset(self.fp)
        for i in range(self.M):
            train, test, _ = dataset.splitData(self.M, i)
            print("Experiment {}:".format(i))
            metric = self.worker(train, test)
            metrics = {k: metrics[k] + metric[k] for k in metrics.keys()}
        metrics = {k: metrics[k] / self.M for k in metrics.keys()}
        print('Average Result (M={}, N={}, ratio={}): {}'.format(self.M, self.N, self.ratio, metrics))


if __name__ == '__main__':
    M, N, ratio = 5, 10, 1
    exp = Experiment(M, N, ratio=ratio)
    exp.run()
