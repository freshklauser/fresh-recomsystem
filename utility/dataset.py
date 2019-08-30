# -*- coding: utf-8 -*-
# @Author: F1684324
# @Date:   2019-08-28 09:56:29
# @Last Modified by:   sniky-lyu
# @Last Modified time: 2019-08-30 22:59:06
# ------------------------------------------------------------------------------
# Description:
# Notes:
#   1) os.sep: 当前系统的分隔符
# ------------------------------------------------------------------------------
import sys
sys.path.append("..")

import os
import random
from utility.decora import timmer


class Dataset():
    '''
    docstring for Dataset
    for different datasets, func "loadData" will deal with them accordingly.
    dataset: ['bookcross', 'hetrec2011-delicious-2k', 'lastfm-dataset-360K',
              'ml-1m', 'soc-Epinions1', 'soc-Slashdot0902']
    open方法或者pandas.read_csv均可读取数据
    '''

    def __init__(self, filepath, ext1filepath=None):
        # filepath: data file path( element in list of /data/)
        # ext1filepath: user profile path
        # self.filepath = filepath
        # self.ext1filepath = ext1filepath
        self.data_list = datasetList()
        if ext1filepath is None:
            self.data = self.loadData(filepath)
            self.profile = None
            # print('profile:', self.profile)
        else:
            self.data, self.profile = self.loadData(filepath, ext1filepath)

    @timmer
    def loadData(self, filepath, ext1filepath=None):
        '''
        根据filepath的data, 选择data_list中不同的数据集:
        eg.  filepath='../data/ml-1m/ratings.dat'＇
        '''
        filepath = convert_path(filepath)                                       # 标准化路径格式
        "-------------------- dataset: ml-1m --------------------"
        if os.path.dirname(filepath).split(os.sep)[-1] == 'ml-1m':
            print('Current dataset: {}'.format('ml-1m'))
            data = []
            for l in open(filepath, 'r'):
                # data form filepath of "ml-1m/ratings.dat"
                data.append(tuple(map(int, l.strip().split('::')[:2])))
            return data
        "-------------------- dataset: lastfm --------------------"
        if os.path.dirname(filepath).split(os.sep)[-1] == 'lastfm-dataset-360K':
            print('Current dataset: {}'.format('lastfm-dataset-360K'))
            data = []                           # [user, item(music)]
            count = 0
            for line in open(filepath, 'rt', encoding='utf-8'):
                try:
                    data.append(line.strip().split('\t')[:2])                   # 不使用try..except(pass)跳过UnicodeEncodeError的话会报错
                    count += 1
                except UnicodeEncodeError:                                      # UnicodeEncodeError: 'cp950' codec can't encode character '\xc4'
                    pass
            print('Count of usersha1-artmbid-artname-plays: ', count)           # 17559530条记录
            profile = {}                                                        # dict{u1:{g:g1,a:a1,c:c1}, u2:{g:g2,a:a2,c:c2}, ...}
            count = 0
            for line in open(ext1filepath, 'rt', encoding='utf-8'):             # 有缺失值
                try:
                    user, gender, age, country = line.strip().split('\t')[:-1]
                    if age == '':                                               # gender也有缺失值，这里没处理
                        age = -1
                    profile[user] = {'gender': gender, 'age': int(age), 'country': country}
                    count += 1
                except UnicodeEncodeError:
                    pass
            print('Count of usersha1-profile(unique users): ', count)           # 359347条记录:unique users
            # 按照用户进行采样  （取user中的前5000个作为user对data和profile进行采样）
            users = list(profile.keys())
            random.shuffle(users)
            users = set(users[:5000])                                           # 取5000个unique users
            data = [x for x in data if x[0] in users]
            profile = {k: profile[k] for k in users}
            return data, profile

    @timmer
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
        for user, item in self.data:                                            # self.data 划分为train和test
            if random.randint(0, M - 1) == k:                                   # randint是左右都覆盖的
                test.append((user, item))
            else:
                train.append((user, item))

        # 处理成字典的形式，user->set(items)
        def convert_dict(data):
            data_dict = {}
            for user, item in data:
                if user not in data_dict:
                    data_dict[user] = set()
                data_dict[user].add(item)
            data_dict = {k: list(data_dict[k]) for k in data_dict}
            return data_dict

        return convert_dict(train), convert_dict(test), self.profile


def datasetList():
    '''根据当前python脚本绝对路径, 获取上一级目录列表，即data文件夹下的list'''
    # 根据当前python脚本绝对路径获取项目名字和项目的root路径
    program_name = os.path.dirname(os.getcwd()).split(os.sep)[-1]
    root = os.path.dirname(os.getcwd())
    print('Program name: ', program_name, type(program_name))
    # print('Root abspath: ', root, type(root))
    dataset_abspath = os.path.join(root, 'data')
    dataset_list = os.listdir(dataset_abspath)
    # print('Dataset list: ', dataset_list, type(dataset_list))
    for i, element in enumerate(dataset_list):
            # check if element in dataset_list is a directory
        # os.path.isfile(abspath / relpath)
        if os.path.isfile(os.path.join(dataset_abspath, element)):
                # delete file element in dataset_list
            del dataset_list[i]
    print('Dataset list: ', dataset_list, type(dataset_list))
    return dataset_list


def convert_path(path):
    ''' 自动将路径转化为当前系统下的路径 '''
    return path.replace(r'\/'.replace(os.sep, ''), os.sep)


if __name__ == '__main__':
    # cla = Dataset('../data/ml-1m/ratings.dat')
    filepath = convert_path(r'..\data\lastfm-dataset-360K\usersha1-artmbid-artname-plays.tsv')
    ext1filepath = convert_path(r'..\data\lastfm-dataset-360K\usersha1-profile.tsv')
    cla = Dataset(filepath, ext1filepath)
    # datasetList()
    # test ----------------------------------------
    # import pandas as pd
    # pd.set_option('display.width', 1000)
    # reader = pd.read_csv(filepath, delimiter='\t', nrows=10, encoding='utf-8')
    # print(type(reader))

    # for line in open(filepath, 'rt', encoding='utf-8'):
    #     try:
    #         # 不使用try..except(pass)跳过UnicodeEncodeError的话会报错
    #         print(line.strip().split('\t'))
    #         count += 1
    #         if count >= 10:
    #             break
    #     except UnicodeEncodeError:      # UnicodeEncodeError: 'cp950' codec can't encode character '\xc4'
    #         pass
