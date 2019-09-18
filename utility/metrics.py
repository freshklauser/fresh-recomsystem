

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
        # 覆盖率：最终的推荐列表中包含多大比例的 **物品**
        # 最简单定义：推荐系统能够推荐出来的物品占总物品集合的比例
        # 总物品集合I: 有采用所有train中的user的总物品集合，也有才有test中用户在train集合中的物品集合
        all_item, recom_item = set(), set()
        for user in self.test.keys():                                   # test 中的 user
            if user not in self.train.keys():
                print('user-{} in test is not in train!!')
                continue
            for item in self.train[user]:                               # test中user在train中包含的所有item
                all_item.add(item)                                      # 所有物品集合 I(train中有而test中没有的user的物品不在I中)
            rank = self.recs[user]
            for item, score in rank:
                recom_item.add(item)                                    # 推荐的物品集合 R(u)
        return round(len(recom_item) / len(all_item) * 100, 2)          # coverage = #R(u) / #I

    # 定义覆盖率指标计算方式
    def coverage_all(self):
        all_item, recom_item = set(), set()
        # 所用train中user的总物品集合作为I (train中的所有user的物品都在I中 -- test中的user在train中)
        for user in self.train:
            for item in self.train[user]:
                all_item.add(item)
        # test中的所有user的推荐列表
        for user in self.test:
            rank = self.recs[user]
            for item, score in rank:
                recom_item.add(item)
        return round(len(recom_item) / len(all_item) * 100, 2)

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
                if item in item_popularity:                 # (冷启动一定要做这个判断)判断item是否在热门列表中
                    # 对每个物品的流行度取对数运算, 防止长尾问题带来的北流行物品主导的推荐（避免热门推荐）
                    popular += math.log(1 + item_popularity[item])
                    num += 1                                               # 汇总所有user的总推荐物品个数
        return round(popular / num, 6)                                  # 计算平均流行度 = popular / n

    def diversity(self):
        '''定义多样性指标计算方式'''
        # 计算item_vec，每个tag的个数
        item_tags = {}
        for user in self.train:
            for item in self.train[user]:
                if item not in item_tags:
                    item_tags[item] = {}
                for tag in self.train[user][item]:
                    if tag not in item_tags[item]:
                        item_tags[item][tag] = 0
                    item_tags[item][tag] += 1

        # 计算两个item (i, j)的相似度
        def CosineSim(i, j):
            ret = 0
            ni, nj = 0, 0
            for tag in item_tags[i].keys():
                ni += item_tags[i][tag] ** 2
                if tag in item_tags[j].keys():
                    # i 和 j 都打标签 tag
                    ret += item_tags[i][tag] * item_tags[j][tag]
            for tag in item_tags[j].keys():
                nj += item_tags[j][tag] ** 2
            return ret / math.sqrt(ni * nj)

        # 计算diversity
        div = []
        for user in self.test:
            rank = self.recs[user]
            sim, cnt = 0, 0
            for i, _ in rank:
                for j, _ in rank:
                    if i == j:
                        continue
                    sim += CosineSim(i, j)
                    cnt += 1
            sim = sim / cnt if sim != 0 else 0      # 用户user的推荐列表的相似度
            # 多样性
            div.append(1 - sim)                     # 用户user的推荐列表的多样性
        return sum(div) / len(div)                  # 所有用户的推荐列表多样性的平均值

    def eval(self):
        ''' 汇总 metric 指标 '''
        metric = {'Precision': self.precision(),
                  'Recall': self.recall(),
                  'Coverage': self.coverage_all(),
                  'Popularity': self.popularity(),
                  'Diversity': self.diversity()}
        print('Metric: {}'.format(metric))
        return metric


class Metric_tagrec():
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

        # 为test中的每个用户推荐