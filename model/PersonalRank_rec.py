# -*- coding: utf-8 -*-
# @Author: Administrator
# @Date:   2019-08-26 10:37:54
# @Last Modified by:   KlausLyu
# @Last Modified time: 2019-09-17 09:59:28
# -------------------------------------------------------------------------------
# PersonalRank算法对通过连接的边为每个节点打分，具体来讲，在PersonalRank算法中，不区分用户和商品，
# 因此计算用户A对所有的商品的感兴趣的程度就变成了对用户A计算各个节点B，C，a，b，c，d的重要程度
# Tips:
#     1) csc_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
#           where `data`, `row_ind` and `col_ind` satisfy `a[row_ind[k], col_ind[k]] = data[k]`.
#        稀疏矩阵
#     2) index = np.argsort(r)[::-1][:N]                 # 降序排列后的top-N个值对应的索引
# -------------------------------------------------------------------------------
import sys
sys.path.append("..")   # (必须放在代码首)引用父目录， 修改了环境变量的搜索路径，脚本运行时生效，结束后失效

import time
import numpy as np
from scipy.sparse import eye            # 稀疏对角矩阵 (eye(m, n=None, k=0), m:rows, n:cols)
from scipy.sparse.linalg import inv     # 逆矩阵
from scipy.sparse import csc_matrix     # 稀疏矩阵
from scipy.sparse import isspmatrix_csc
from utility.metrics import Metric
from utility.dataset import Dataset
from utility.decora import timmer


@timmer
def PersonalRank(train, alpha, N):
    '''matrix  format to calc r （note the format of train dataset）
    :params: train, 训练数据  ({user1:{item1, item2, ...}, user2:{item1, item2,...}})
    :params: alpha, 继续随机游走的概率
    :params: N, 推荐TopN物品的个数
    :return: GetRecommendation, 获取推荐结果的接口
    '''
    # 看看训练数据集的格式
    # print('train --> ', train)

    # 构建索引,方便利用row和col对应value构建矩阵,且矩阵行列前len(users_index)行或列对应用户节点，之后的为物品节点
    items = []
    for user in train.keys():
        items.extend(train[user])
    id2item = list(set(items))                                              # 物品清单 （不重复清单）
    users_index = {u: i for i, u in enumerate(train.keys())}                # {u1: index_u1, u2: index_u2, u3: index_u3, ...}
    items_index = {v: i + len(users_index) for i, v in enumerate(id2item)}  # {it1:}
    # print('users --> ', users_index)
    # print('items --> ', items_index)
    dim_length = int(len(users_index) + len(items_index))                   # 构建的稀疏矩阵的维度
    print('sparse matrix dimension: ', dim_length)

    "!! 计算转移矩阵（按照出度进行归一化, 即转化为u--v的概率p) !!"
    item_user = {}                      # 用户物品倒排表
    for user in train.keys():
        for item in train[user]:
            if item not in item_user:
                item_user[item] = []
            item_user[item].append(user)
    # print('item_user', '-->', item_user)
    # 矩阵data行中user-->item 部分
    data, row, col = [], [], []
    for u in train.keys():
        for v in train[u]:
            # print(users_index[u], '->', items_index[v], ':', 1 / len(train[u]))
            data.append(1 / len(train[u]))
            row.append(users_index[u])
            col.append(items_index[v])
    # 矩阵data行中 item-->user 部分
    for v in item_user.keys():
        for u in item_user[v]:
            # print(items_index[v], '->', users_index[u], ':', 1 / len(item_user[v]))
            data.append(1 / len(item_user[v]))
            row.append(items_index[v])
            col.append(users_index[u])
    # user->item 和 item->user 合并起来 构成 转移矩阵data（节点不区分用户和商品）
    M = csc_matrix((data, (row, col)), shape=(dim_length, dim_length))
    # print(M)
    print('sparse matrix shape', M.shape)

    def getRecommendation(user):
        #        seen_items = set(train[user])
        # 解矩阵方程   r = (1-a)r0 + a(M.T)r --> r = (1-a)*(1-a(M.T))**(-1)*r0
        # 初始化为0，再当前节点作为根节点root赋初始值为1, 并转化为洗漱矩阵
        r0 = [0] * dim_length
        r0[users_index[user]] = 1
        r0 = csc_matrix(r0)                         # （0，user_index） 1
        # 方程结果 r = (1-a)*(1-a(M.T))**(-1)*r0
        # 矩阵减法 需要 matix 有相同的 dims, 故1-a(M.T)中1需要转化为对角矩阵(diag=1, other=0)
        # print(inv(eye(len(data)) - alpha * M.T).shape)
        if not isspmatrix_csc(csc_matrix(eye(dim_length))):
            print('---******************---')
        t1 = time.perf_counter()
        r = (1 - alpha) * inv(eye(dim_length) - alpha * M) * r0.T      # (7,7)*(7,1)
        print('time consuming by calculate r: {}s'.format(time.perf_counter() - t1))
        if not isspmatrix_csc(r):
            print('******************')
        # print('matrix solution -- r: ', r.shape, '\n', r)
        r = r.toarray()[len(users_index):, 0]     # user索引为0:[len(users_index)], item索引为[len(users_idnex):]
        print('r shape: ', r.shape)
        # print(r)
        index = np.argsort(r)[::-1][:N]                 # 降序排列后的top-N个值对应item的索引
        # print(index)
        recs = [(id2item[i], r[i]) for i in index]
        print('recommendation list(top{}) for {} -- recs: {}'.format(N, user, recs))
        return recs

    return getRecommendation


class Experiment():
    def __init__(self, M, N, alpha, fp='../data/ml-1m/ratings.dat'):
        '''
        :params: M, 进行多少次实验
        :params: N, TopN推荐物品的个数
        :params: alpha, 继续随机游走的概率
        :params: fp, 数据文件路径
        '''
        self.M = M
        self.N = N
        self.alpha = alpha
        self.fp = fp
        self.alg = PersonalRank

    # 定义单次实验
    @timmer
    def worker(self, train, test):
        '''
        :params: train, 训练数据集
        :params: test, 测试数据集
        :return: 各指标的值
        '''
        getRecommendation = self.alg(train, self.alpha, self.N)
        metric = Metric(train, test, getRecommendation)
        return metric.eval()

    # 多次实验取平均
    @timmer
    def run(self):
        metrics = {'Precision': 0, 'Recall': 0,
                   'Coverage': 0, 'Popularity': 0, 'Diversity':0}
        dataset = Dataset(self.fp)
        for ii in range(self.M):
            train, test, _ = dataset.splitData(self.M, ii)
            print('Experiment {}:'.format(ii))
            metric = self.worker(train, test)
            metrics = {k: metrics[k] + metric[k] for k in metrics}
        metrics = {k: metrics[k] / self.M for k in metrics}
        print('Average Result (M={}, N={}, ratio={}): {}'.format(self.M, self.N, self.ratio, metrics))


if __name__ == '__main__':
    # G = {'A': {'a', 'c'}, 'B': {'a', 'b', 'c', 'd'}, 'C': {'c', 'd'}}
    # f = PersonalRank(G, 0.8, 4)('B')
    # test
    M = 8
    N = 3
    alpha = 0.8
    exp = Experiment(M, N, alpha)
    exp.run()
#  [('b', 0.3117744610281923), ('a', 0.22222222222222218),
# ('d', 0.22222222222222218), ('c', 0.19237147595356544)]
