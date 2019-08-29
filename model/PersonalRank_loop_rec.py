# -*- coding: utf-8 -*-
# @Author: Klaus
# @Date:   2019-08-26 22:33:44
# @Last Modified by:   F1684324
# @Last Modified time: 2019-08-28 10:20:38


def PersonalRank(G, alpha, root, max_step):
    '''
    G:{'A' : {'a' : 1, 'c' : 1}, 'B' : {'c' : 1, 'd' : 1},...}
    '''
    # 定义推荐列表及访问概率(概率之和为1)
    rank = dict()
    rank = {k: 0 for k in G.keys()}
    rank[root] = 1
    print('Initial rank: ', rank)

    for step in range(max_step):
        print('\n', '>>> Step {}:'.format(step))
        # 初始化推荐列表
        temp = dict()
        temp = {x: 0 for x in G.keys()}
        # 取节点i和它的出边尾节点集合ri
        for i, ri in G.items():
            # 取节点i的出边的尾节点j以及边E(i,j)的权重wij,边的权重都为1，归一化后就是1/len(ri)
            for j, wij in ri.items():
                if j not in temp.keys():
                    temp[j] = 0
                temp[j] += alpha * rank[i] / (1.0 * len(ri))
                # if j == root:
                #     temp[j] += (1 - alpha)        # root=B, rank[B]=1.559
        temp[root] += 1 - alpha
        rank = temp
        print(rank)
        # get the recommendation list
        recs = sorted(rank.items(), key=lambda x: x[1], reverse=True)
        print('recommendation list: ', recs)

    return recs


if __name__ == '__main__':
    alpha = 0.8
    G = {'A': {'a': 1, 'c': 1},
         'B': {'a': 1, 'b': 1, 'c': 1, 'd': 1},
         'C': {'c': 1, 'd': 1},
         'a': {'A': 1, 'B': 1},
         'b': {'B': 1},
         'c': {'A': 1, 'B': 1, 'C': 1},
         'd': {'B': 1, 'C': 1}}

    recs = PersonalRank(G, alpha, 'B', 50)
    for ele in recs:
        print("%s:%.3f, \t" % (ele[0], ele[1]))
# B:0.390,
# c:0.144,
# a:0.111,
# d:0.111,
# A:0.083,
# C:0.083,
# b:0.078,
