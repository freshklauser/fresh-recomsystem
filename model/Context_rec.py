# -*- coding: utf-8 -*-
# @Author: KlausLyu
# @Date:   2019-09-18 14:20:43
# @Last Modified by:   klaus
# @Last Modified time: 2019-09-18 23:31:02
# ------------------------------------------------------------------------------
# Description:
# 实现算法：
# RecentPopular
#   TItemCF: ItemCF with fushion time information
#   TUserCF: UserCF with fushion time information
#   ItemCF: just ItemCF
#   UserCF: just UserCF
# Tips: codecs是Python中标准库的内容，而codecs.open和内置函数open（）的不同在于，
# open函数无法打开一份由不同编码组成的同一份文件，而codecs.open如文档所说，始终
# 以二进制模式打开，故打开就是Unicode格式，所以，codecs.open能打开由不同编码格式组成的文件。
# ------------------------------------------------------------------------------
import sys
sys.path.append('..')

import random
import math
import chardet
import time
import codecs
from operator import itemgetter
from utility.decora import timmer


class Dataset():
    def __init__(self, site=None):
        # site: which site to load
        self.bookmark_path = '../data/hetrec2011-delicious-2k/bookmarks.dat'
        self.user_bookmark_path = '../data/hetrec2011-delicious-2k/user_taggedbookmarks-timestamps.dat'
        self.site = site
        self.data = self.loadData()

    def loadData(self):
        site_ids = {}                   # key: 网址, value: 物品的id即bookmarkid
        for line in open(self.bookmark_path, 'rb').readlines()[1:]:
            # 混合编码，经测试，latin_1:iso-8859-1, iso8859-1, 8859, cp819, latin, latin1, L1: West Europe 编码可行，line也需要encode
            # 或者，直接采用字节模式 rb 读取，再转化为str后进行字符串处理
            line = str(line.strip())[2:-1].split(r'\t')                         # 字符串組成的列表
            # print(i, '--->', line)
            # 12 ---> ['19', '629286805378ffb56dc048d109af82b5', 'IXL Math', 'http://www.ixl.com/', 'b92ccf2f89d1ef4aaacbade44649278a', 'www.ixl.com']
            if line[-1] not in site_ids:
                site_ids[line[-1]] = set()
            site_ids[line[-1]].add(line[0])
            # {..., 'www.media-awareness.ca': {'79', '72', '76', '85', '78', '75'}, 'www.library20.org': {'73'}, ...}

        data = {}                       # key: userid, value:(bookmarkid, int(timestamp))
        for line in open(self.user_bookmark_path, 'r', encoding='iso-8859-1').readlines()[1:]:
            line = line.strip().split('\t')             # ['8', '7', '6', '1289238901000']
            if self.site is None or (self.site in site_ids.kyes() and line[1] in site_ids[self.site]):
                if line[0] not in data.keys():
                    data[line[0]] = set()
                data[line[0]].add((line[1], int(line[-1][:-3])))
                # data: {'8': {('1', 1289255362), ('7', 1289238901), ('2', 1289255159), ...}, '9':{...}, ...}
        # data的value转化为列表,且按照时间戳 升序 排列：用户对物品的行为从早到晚排序
        for k in data.keys():
            data[k] = sorted(data[k], key=itemgetter(1), reverse=False)
        return data

    def splitData(self):
        '''
        :params: data, 加载的所有{user, [(item, timestamp), ...], ...}数据条目
        :return: train, test
        data是时间戳从早到晚排序即升序，最后一个产生行为的物品作为测试集，之前的作为训练集
        '''
        train, test = {}, {}
        for user in self.data:
            if user not in train.keys():
                train[user] = []
                test[user] = []
            train[user].extend(self.data[user][:-1])
            test[user].append(self.data[user][-1])
        return train, test


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
            rank = self.GetRecommendation(user)
            recs[user] = rank
        return recs

    # 定义精确率指标计算方式
    def precision(self):
        All, hit = 0, 0
        for user in self.test:
            test_items = set([x[0] for x in self.test[user]])
            rank = self.recs[user]
            for item, score in rank:
                if item in test_items:
                    hit += 1
            All += len(rank)            # precision与recall不同的地方
        return round(hit / All * 100, 2) if All > 0 else 0.0

    # 定义召回率指标计算方式
    def recall(self):
        All, hit = 0, 0
        for user in self.test:
            test_items = set([x[0] for x in self.test[user]])
            rank = self.recs[user]
            for item, score in rank:
                if item in test_items:
                    hit += 1
            All += len(test_items)      # precision与recall不同的地方
        return round(hit / All * 100, 2) if All > 0 else 0.0

    def eval(self):
        metric = {'Precision': self.precision(),
                  'Recall': self.recall()}
        return metric


# 1. 给用户推荐近期最热门的物品
def RecentPopular(train, K, N, alpha=1.0, t0=int(time.time())):
    '''
    :params: train, 训练数据集
    :params: K, 可忽略
    :params: N, 超参数，设置取TopN推荐物品数目
    :params: alpha, 时间衰减因子
    :params: t0, 当前的时间戳
    :return: GetRecommendation，推荐接口函数
    '''
    rank = {}                   # key:item, value:score
    for u in train.keys():
        for i, t in train[u]:
            if i not in rank.keys():
                rank[i] = 0
            rank[i] += 1 / (1.0 + alpha * (t0 - t))
    rank = sorted(rank.items(), key=itemgetter(1), reverse=True)

    def GetRecommendation(user):
        # 推荐N个最近最热门的
        seen_items = set(train[user])
        rec_items = []
        for x in rank:
            if x[0] not in seen_items:
                rec_items.append(x)
        print(rec_items[:N])
        return rec_items[:N]

    return GetRecommendation


if __name__ == '__main__':
    train, test = Dataset().splitData()
    RecentPopular(train, 10, 10)('8')

    # bookmark_path = '../data/hetrec2011-delicious-2k/bookmarks.dat'
    # bookmarks = [f.strip() for f in open(bookmark_path, 'rb').readlines()][1:]
    # for i, v in enumerate(bookmarks):
    #     # line = v.encode('utf-8')
    #     line = v
    #     print(i, '--->', line)

    #     if i >= 150:
    #         break
