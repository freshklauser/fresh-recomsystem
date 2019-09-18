# -*- coding: utf-8 -*-
# @Author: F1684324
# @Date:   2019-09-16 09:29:34
# @Last Modified by:   KlausLyu
# @Last Modified time: 2019-09-17 11:17:55
# ------------------------------------------------------------------------------
# Description:
# dataset1: user_taggedbookmarks.dat:(userID / bookmarkID / tagID / day month / year / hour / minute / second)
# 算法实现
# SimpleTagBased:
# 1)统计每个用户最常用的标签, user_tags
# 2)对于每个标签 统计被打过这个标签最多的物品, tag_items
# 3)对于每个用户，首先找到他最常用的的标签，然后找到具有这些标签的最热门的物品推荐给该用户
# 4)计算用户u对物品i的兴趣： p(u,i) = sum( n(u,b) * n(b,i) )
# TagBasedTFIDF
# TagBasedTFIDF++
# TagExtend
# ------------------------------------------------------------------------------
import sys
sys.path.append("..")

import random
import math
# import numpy as np
from utility.decora import timmer
from utility.metrics import Metric


class Dataset():
    def __init__(self, fp):
        # fp: data file path
        self.data = self.loadData(fp)

    @timmer
    def loadData(self, fp):
        data = []           # [[userID, bookmarkID, tagID], [...], ] --> [..., [...], ['147', '272', '193'], [...], ...]
        for line in open(fp).readlines()[1:]:
            data.append(line.strip().split('\t')[:3])
        new_data = {}       # {'8': {'1': {'1'}, '2': {'1'}, '7': {'6', '7', '1'}, '8': {'1'}}, {...}}
        for u, i, t in data:
            if u not in new_data.keys():
                new_data[u] = {}
            if i not in new_data[u].keys():
                new_data[u][i] = set()      # 去除重复的tag
            new_data[u][i].add(t)
            # print(new_data)   # { user: {book:{tags}}, user2:{...}, ... }
        # 用records存儲标签数据三元组 records[i] = [(user, item, tags_list), ...]
        records = []                # [..., ('8', '7', ['1', '7', '6']), ('8', '8', ['1', '9', '8']), ...]
        for user in new_data:
            for item in new_data[user]:
                records.append((user, item, list(new_data[user][item])))
        return records

    @timmer
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
        for user, item, tags in self.data:
            if random.randint(0, M - 1) == k:
                test.append((user, item, tags))
            else:
                train.append((user, item, tags))

        # 处理成字典形式
        def convert_dict(data):
            data_dict = {}
            for user, item, tags in data:
                if user not in data_dict.keys():
                    data_dict[user] = {}
                data_dict[user][item] = tags
            return data_dict

        return convert_dict(train), convert_dict(test)


# 1. 基于热门标签的推荐
@timmer
def SimpleTagBased(train, N):
    '''
    :params: train, 训练数据集
    :params: N, 超参数，设置取TopN推荐物品数目
    :return: GetRecommendation，推荐接口函数
    '''
    # 统计 user_tags 和 tag_items
    user_tags = {}                  # 存储 n(u,b), 其中 user_tags[u][b] = n(u,b) 用户u打过标签b的次数
    tag_items = {}                  # 存储 n(b,i), 其中 user_tags[b][i] = n(b,i) 物品i被打过标签b的次数
    for user in train.keys():
        if user not in user_tags.keys():
            user_tags[user] = {}                    # 存储所有user打过的tag
        for item in train[user].keys():
            for tag in train[user][item]:
                # 统计每一个user打过的所有tag及每一个tag的次数
                if tag not in user_tags[user].keys():
                    user_tags[user][tag] = 0
                user_tags[user][tag] += 1
                # 统计每一个item被打过的标签及次数
                if tag not in tag_items.keys():
                    tag_items[tag] = {}
                if item not in tag_items[tag].keys():
                    tag_items[tag][item] = 0
                tag_items[tag][item] += 1

    def GetRecommendation(user):
        ''' 对用户user, 找到常用的tag, 然后找到具有这些tag的最热门的物品作为推荐列表 '''
        if user not in user_tags.keys():
            print('User-{} is a newfresher that is not in the list of user_tags')
            return []
        seen_items = set(train[user])
        rank = {}                                   # {item1:pop, item2:pop, ...}
        for tag in user_tags[user]:                 # 遍历给定user的tags
            for item in tag_items[tag]:             # 便利给定user的每一个tag中的item
                if item in seen_items:
                    continue
                if item not in rank:                # 如果item不在rank中, 设为0
                    rank[item] = 0
                # 计算用户u对物品i的兴趣： p(u,i) = sum( n(u,b) * n(b,i) )
                rank[item] += user_tags[user][tag] * tag_items[tag][item]
        rank = sorted(rank.items(), key=lambda x: x[1], reverse=True)
        # print(rank[:N], '-----')
        return rank[:N]

    return GetRecommendation


# 2. 改进一： 热门标签加入惩罚項
@timmer
def TagBasedTFIDF(train, N):
    '''
    :params: train, 训练数据集
    :params: N, 超参数，设置取TopN推荐物品数目
    :return: GetRecommendation，推荐接口函数
    '''
    # 统计 user_tags 和 tag_items
    user_tags = {}                  # 存储 n(u,b), 其中 user_tags[u][b] = n(u,b) 用户u打过标签b的次数
    tag_items = {}                  # 存储 n(b,i), 其中 user_tags[b][i] = n(b,i) 物品i被打过标签b的次数
    tag_pop = {}                    # 统计标签的热门程度，即打过此标签的不同用户数
    for user in train.keys():
        user_tags[user] = {}                    # 存储所有user打过的tag
        for item in train[user].keys():
            for tag in train[user][item]:
                # 统计每一个user打过的所有tag及每一个tag的次数
                if tag not in user_tags[user].keys():
                    user_tags[user][tag] = 0
                user_tags[user][tag] += 1
                # 统计每一个item被打过的标签及次数
                if tag not in tag_items.keys():
                    tag_items[tag] = {}
                if item not in tag_items[tag].keys():
                    tag_items[tag][item] = 0
                tag_items[tag][item] += 1
                # 统计标签的热门程度，即打过此标签的不同用户数
                if tag not in tag_pop.keys():
                    tag_pop[tag] = set()
                tag_pop[tag].add(user)
    tag_pop = {k: len(v) for k, v in tag_pop.items()}

    def GetRecommendation(user):
        ''' 对用户user, 找到常用的tag, 然后找到具有这些tag的最热门的物品作为推荐列表 '''
        if user not in user_tags.keys():
            print('User-{} is a newfresher that is not in the list of user_tags')
            return []
        seen_items = set(train[user])
        rank = {}                                   # {item1:pop, item2:pop, ...}
        for tag in user_tags[user]:                 # 遍历给定user的tags
            for item in tag_items[tag]:             # 便利给定user的每一个tag中的item
                if item in seen_items:
                    continue
                if item not in rank:                # 如果item不在rank中, 设为0
                    rank[item] = 0
                # 计算用户u对物品i的兴趣： p(u,i) = sum( n(u,b) * n(b,i) / log(1 + n(b)))
                rank[item] += int(user_tags[user][tag] * tag_items[tag][item] / math.log(1 + tag_pop[tag]))
        rank = sorted(rank.items(), key=lambda x: x[1], reverse=True)
        return rank[:N]

    return GetRecommendation


# 改进3：同时也为热门商品加入惩罚项
@timmer
def TagBasedTFIDF_imp(train, N):
    '''
    :params: train, 训练数据集
    :params: N, 超参数，设置取TopN推荐物品数目
    :return: GetRecommendation，推荐接口函数
    '''
    # 统计 user_tags 和 tag_items
    user_tags = {}                  # 存储 n(u,b), 其中 user_tags[u][b] = n(u,b) 用户u打过标签b的次数
    tag_items = {}                  # 存储 n(b,i), 其中 user_tags[b][i] = n(b,i) 物品i被打过标签b的次数
    tag_pop = {}                    # 统计标签的热门程度，即打过此标签的不同用户数
    item_pop = {}                   # 统计物品的热门程度，即物品i被多少不同用户数打过标签
    for user in train.keys():
        user_tags[user] = {}                    # 存储所有user打过的tag
        for item in train[user].keys():
            # 统计热门商品
            if item not in item_pop.keys():
                item_pop[item] = 0
            item_pop[item] += 1
            for tag in train[user][item]:
                # 统计每一个user打过的所有tag及每一个tag的次数
                if tag not in user_tags[user].keys():
                    user_tags[user][tag] = 0
                user_tags[user][tag] += 1
                # 统计每一个item被打过的标签及次数
                if tag not in tag_items.keys():
                    tag_items[tag] = {}
                if item not in tag_items[tag].keys():
                    tag_items[tag][item] = 0
                tag_items[tag][item] += 1
                # 统计标签的热门程度，即打过此标签的不同用户数    # 也可以用统计热门商品的表述
                if tag not in tag_pop.keys():
                    tag_pop[tag] = set()
                tag_pop[tag].add(user)
    tag_pop = {k: len(v) for k, v in tag_pop.items()}

    def GetRecommendation(user):
        ''' 对用户user, 找到常用的tag, 然后找到具有这些tag的最热门的物品作为推荐列表 '''
        if user not in user_tags.keys():
            print('User-{} is a newfresher that is not in the list of user_tags')
            return []
        seen_items = set(train[user])
        rank = {}                                   # {item1:pop, item2:pop, ...}
        for tag in user_tags[user]:                 # 遍历给定user的tags
            for item in tag_items[tag]:             # 便利给定user的每一个tag中的item
                if item in seen_items:
                    continue
                if item not in rank:                # 如果item不在rank中, 设为0
                    rank[item] = 0
                # 计算用户u对物品i的兴趣： p(u,i) = sum( n(u,b) * n(b,i) / log(1 + n(b)) / log(1 + n(i)))
                rank[item] += int(user_tags[user][tag] * tag_items[tag][item] / math.log(1 + tag_pop[tag]) / math.log(1 + item_pop[item]))
        rank = sorted(rank.items(), key=lambda x: x[1], reverse=True)
        return rank[:N]

    return GetRecommendation


# 4. 基于标签改进的推荐
@timmer
def ExpandTagBased(train, N, M=20):
    '''
    :params: train, 训练数据集
    :params: N, 超参数，设置取TopN推荐物品数目
    :params: M，超参数，设置取TopM的标签填补不满M个标签的用户
    :return: GetRecommendation，推荐接口函数
    '''
    # 1) 计算标签两两之间的相似度:
    # 当两个标签同时出现在很多物品的标签集合中，可以认为这两个标签具有较大相似度
    # ---> 遍历物品对应的标签集合 item_tags {item1: set(t1,t2,...), item2:set(t2,t4,...), ...}
    item_tags = {}                               # item --> set(tags)
    for user in train.keys():
        for item in train[user].keys():
            if item not in item_tags.keys():
                item_tags[item] = set()
            for tag in train[user][item]:
                item_tags[item].add(tag)
    tag_sim = {}
    tag_count = {}
    for item in item_tags.keys():
        for t in item_tags[item]:
            if t not in tag_count.keys():
                tag_count[t] = 0
            tag_count[t] += 1                   # tag_count[t] += 1 ** 2 (平方后再累加)
            if t not in tag_sim.keys():
                tag_sim[t] = {}
            for b in item_tags[item]:
                if b == t:
                    continue
                if b not in tag_sim[t].keys():
                    tag_sim[t][b] = 0
                tag_sim[t][b] += 1              # 两个标签出现在同一个物品上，则计数+1衡量两者间相似度
    # 根据tag t和b 同时出现的次数，计算t和b相似度
    for t in tag_sim.keys():
        for b in tag_sim[t].keys():
            tag_sim[t][b] /= math.sqrt(tag_count[t] * tag_count[b])

    # 2) 为每个用户扩展标签： user_tags --> expand_tags
    user_tags = {}
    tag_items = {}
    for user in train.keys():
        if user not in user_tags.keys():
            user_tags[user] = {}
        for item in train[user].keys():
            for tag in train[user][item]:
                # 统计每一个user打过的所有tag及每一个tag的次数
                if tag not in user_tags[user]:
                    user_tags[user][tag] = 0
                user_tags[user][tag] += 1
                # 统计每一个item被打过的标签及次数
                if tag not in tag_items.keys():
                    tag_items[tag] = {}
                if item not in tag_items[tag]:
                    tag_items[tag][item] = 0
                tag_items[tag][item] += 1
    # 扩展标签
    expand_tags = {}
    for u in user_tags.keys():
        if len(user_tags[u].keys()) >= M:       # 满M个tag的取前M个tag，不足M个的扩展后取前M个
            expand_tags[u] = user_tags[u]
            continue
        # 不满M个标签的进行用户的标签扩展: 按照相似度扩充后降序取M个
        expand_tags[u] = {}
        seen_tags = set(user_tags[u])
        for tag in user_tags[u].keys():         # 遍历user_tags中用户u的tag集合
            for t in tag_sim[tag]:              # 遍历与指定tag相似的tag集合
                if t in seen_tags:
                    continue
                if t not in expand_tags[u]:
                    expand_tags[u][t] = 0
                # 扩展依据 --> 用户u打过标签tag的次数 * 权重:标签tag和标签t的相似度
                expand_tags[u][t] += user_tags[u][tag] * tag_sim[tag][t]
        # 合并用户u原有的tag集合与扩展的根据相似度获取的tag集合
        expand_tags[u].update(user_tags[u])
        # 对用户u而言，合并后的tag集合取前M个标签, sorted返回的是列表，需再转化为字典
        expand_tags[u] = dict(sorted(expand_tags[u].items(), key=lambda x: x[1], reverse=True)[:M])

    # 3) 按照SimpleTagBased算法的推荐接口求推荐列表
    def GetRecommendation(user):
        ''' 按照统计的 expand_tags 和 tag_items 进行个性化推荐 '''
        if user not in user_tags.keys():
            return []
        seen_items = set(train[user])
        rank = {}
        for tag in expand_tags[user].keys():
            for item in tag_items[tag].keys():
                if item in seen_items:
                    continue
                if item not in rank.keys():
                    rank[item] = 0
                rank[item] += expand_tags[user][tag] * tag_items[tag][item]
        rank = sorted(rank.items(), key=lambda x: x[1], reverse=True)
        return rank[:N]

    return GetRecommendation


class Experiment():
    def __init__(self, M, N, fp='../data/hetrec2011-delicious-2k/user_taggedbookmarks.dat', rt='SimpleTagBased'):
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
        self.alg = {'SimpleTagBased': SimpleTagBased, 'TagBasedTFIDF': TagBasedTFIDF,
                    'TagBasedTFIDF_imp': TagBasedTFIDF_imp, 'ExpandTagBased': ExpandTagBased}

    # 定义单次实验
    @timmer
    def worker(self, train, test):
        '''
        :params: train, 训练数据集
        :params: test, 测试数据集
        :return: 各指标的值
        '''
        getRecommendation = self.alg[self.rt](train, self.N)
        metric = Metric(train, test, getRecommendation)
        return metric.eval()

    # 多次实验取平均
    @timmer
    def run(self):
        metrics = {'Precision': 0, 'Recall': 0,
                   'Coverage': 0, 'Popularity': 0, 'Diversity': 0}
        dataset = Dataset(self.fp)
        for ii in range(self.M):
            train, test = dataset.splitData(self.M, ii)
            print('Experiment {}:'.format(ii))
            metric = self.worker(train, test)
            metrics = {k: metrics[k] + metric[k] for k in metrics}
        metrics = {k: metrics[k] / self.M for k in metrics}
        print('Average Result (M={}, N={}): {}'.format(self.M, self.N, metrics))


if __name__ == '__main__':
    # 1. SimpleTagBased实验
    print('>>>>>>>>>>>>>>>>>>>>>>>>> SimpleTagBased <<<<<<<<<<<<<<<<<<<<<<<<<<')
    M, N = 10, 10
    exp = Experiment(M, N, rt='SimpleTagBased')
    exp.run()

    # 2. TagBasedTFIDF实验
    print('>>>>>>>>>>>>>>>>>>>>>>>>> TagBasedTFIDF <<<<<<<<<<<<<<<<<<<<<<<<<<<')
    M, N = 10, 10
    exp = Experiment(M, N, rt='TagBasedTFIDF')
    exp.run()

    # 3. TagBasedTFIDF_imp 实验
    print('>>>>>>>>>>>>>>>>>>>>>>>>> TagBasedTFIDF_imp <<<<<<<<<<<<<<<<<<<<<<<')
    M, N = 10, 10
    exp = Experiment(M, N, rt='TagBasedTFIDF_imp')
    exp.run()

    print('>>>>>>>>>>>>>>>>>>>>>>>>> ExpandTagBased <<<<<<<<<<<<<<<<<<<<<<<<<<')
    # 4. ExpandTagBased 实验
    M, N = 10, 10
    exp = Experiment(M, N, rt='ExpandTagBased')
    exp.run()

    # Average Result (M=10, N=10): {'Precision': 0.337, 'Recall': 0.554, 'Coverage': 3.363, 'Popularity': 2.340, 'Diversity': 0.791}
    # Average Result (M=10, N=10): {'Precision': 0.367, 'Recall': 0.602, 'Coverage': 5.295, 'Popularity': 2.223, 'Diversity': 0.800}
    # Average Result (M=10, N=10): {'Precision': 0.272, 'Recall': 0.446, 'Coverage': 12.047, 'Popularity': 1.281, 'Diversity': 0.746}
    # Average Result (M=10, N=10): {'Precision': 0.344, 'Recall': 0.567, 'Coverage': 3.418, 'Popularity': 2.336, 'Diversity': 0.790}

    # fp = '../data/hetrec2011-delicious-2k/user_taggedbookmarks.dat'
    # train, test = Dataset(fp).splitData(5, 1)
    # f = SimpleTagBased(train, 10)
    # print(f('16153'))
    # f1 = TagBasedTFIDF(train, 10)
    # print(f1('16153'))
    # f2 = TagBasedTFIDF_imp(train, 10)
    # print(f2('16153'))
    # f3 = ExpandTagBased(train, 10)
    # print(f3('16153'))
