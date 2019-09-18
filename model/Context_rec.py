# -*- coding: utf-8 -*-
# @Author: KlausLyu
# @Date:   2019-09-18 14:20:43
# @Last Modified by:   KlausLyu
# @Last Modified time: 2019-09-18 17:22:04
# ------------------------------------------------------------------------------
# Description:

# Tips: codecs是Python中标准库的内容，而codecs.open和内置函数open（）的不同在于，
# open函数无法打开一份由不同编码组成的同一份文件，而codecs.open如文档所说，始终
# 以二进制模式打开，故打开就是Unicode格式，所以，codecs.open能打开由不同编码格式组成的文件。
# ------------------------------------------------------------------------------
import sys
sys.path.append('..')

import random
import math
import chardet
import codecs
from utility.decora import timmer


class Dataset():
    def __init__(self, site=None):
        # site: which site to load
        self.bookmark_path = '../data/hetrec2011-delicious-2k/bookmarks.dat'
        self.user_bookmark_path = '../data/hetrec2011-delicious-2k/user_taggedbookmarks-timestamps.dat'
        self.site = site
        self.loadData()

    def loadData(self):
        site_ids = {}                   # key: website, value: idslist-visit-website 访问该website的id记录
        i = 0
        for line in open(self.bookmark_path, 'rb').readlines()[1:]:
            # line = line.strip().split('\t')
            line = str(line.strip())[2:-1].split(r'\t')                         # 字符串組成的列表
            print(i, '--->', line)
            if line[-1] not in site_ids:
                site_ids[line[-1]] = set()
            site_ids[line[-1]].add(line[0])
            # {..., 'www.media-awareness.ca': {'79', '72', '76', '85', '78', '75'}, 'www.library20.org': {'73'}, ...}
            i += 1
            if i >= 15:
                break
        # print(site_ids)

        data = {}                       # key: userid, value:(item, int(timestamp))
        for line in open(self.user_bookmark_path, 'r', encoding='iso-8859-1').readlines()[1:]:
            line = line.strip().split('\t')             # ['8', '7', '6', '1289238901000']
            if self.site is None or (self.site in site_ids.kyes() and line[1] in site_ids[self.site]):
                if line[0] not in data.keys():
                    data[line[0]] = set()
                data[line[0]].add((line[1], int(line[-1][:-3])))
                # data: {'8': {('1', 1289255362), ('7', 1289238901), ('2', 1289255159), ...}, '9':{...}, ...}
        return site_ids, data


if __name__ == '__main__':
    a, b = Dataset().loadData()
    # bookmark_path = '../data/hetrec2011-delicious-2k/bookmarks.dat'
    # bookmarks = [f.strip() for f in open(bookmark_path, 'rb').readlines()][1:]
    # for i, v in enumerate(bookmarks):
    #     # line = v.encode('utf-8')
    #     line = v
    #     print(i, '--->', line)

    #     if i >= 150:
    #         break
