#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import pandas as pd
import numpy as np
import os
import random
from matplotlib import pyplot as plt
import time
from datetime import datetime
import seaborn as sns
from collections import Counter
import copy
import transbigdata as tbd
import geopandas as gpd
import math
from rtree import index
from bidict import bidict
from scipy.optimize import curve_fit
from scipy.special import factorial
from sklearn.metrics import r2_score
import autopep8
import logging
def get_logger(logger_name,log_file,level=logging.INFO):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('[%(asctime)s] %(name)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(fileHandler)
    return logger


# # 函数
# ## 选车函数

# In[2]:


# TODO1:用户行为选车
# 随机选车函数
def select1(pool):
    rank = -1
    bikeid = -1
    list1 = []
    for bb in pool:
        if bb[-1] == 0:
            list1.append(bb)
    if len(list1) > 0:
        i = random.randint(0, len(list1)-1)
        bikeid = list1[i][0]
        rank = i + 1
    return rank, bikeid


# 用户行为选车函数
def select2(pool, PROBABILITY_MAP):
    
    """
    1. 计算每辆车的概率值
    2. 使用轮盘赌的方法选出被骑的车辆
    3. 
    """
    # bike_Id, 使用次数，等待时间（订单时间-上次使用时间），车况值, 上次使用时间（结束时刻），车辆状态0/1
    global sort_t
    global cal_t
    bikeid = -1
    rank = -1
    list1 = []
    for bb in pool:
        if bb[-1] == 0:
            # 车辆号、车况值
            list1.append([bb[0], bb[3]])
    if len(list1)==0:
        return -1,-1
    # 按车况值排名
    t1 = time.time()
    list1.sort(key=lambda x: x[1], reverse=True)
    t2 = time.time()
#     print("按车况排序时间："+str(t2-t1))
    sort_t += (t2-t1)
    probability = []
    p_sum = 0
    t1 =time.time()
    for i, item in enumerate(list1):
        list1[i].append(i+1)
        if i+1 not in PROBABILITY_MAP:
            PROBABILITY_MAP[i+1] = func7(i+1, *r_popt)
            print(PROBABILITY_MAP[i+1])
        pp = PROBABILITY_MAP[i+1]
        probability.append(pp)
        p_sum += pp
    t2 = time.time()
#     print("计算选择概率时间："+str(t2-t1))
    cal_t+=(t2-t1)
    norm_probability = [p / p_sum for p in probability]
    selected = random.choices(list1, norm_probability)
#     print(list1)
#     print(selected)
    rank = selected[0][2]
    bikeid = selected[0][0]
#     p_random = random.uniform(0, p_sum)
    
#     p_cum = 0
#     for i, item in enumerate(list1):
#         p_cum += list1[i][2]
#         if p_cum > p_random:
#             rank = i+1
#             bikeid = list1[i][0]
#             break
    return rank, bikeid


def CalculateDistance(lng1, lat1, lng2, lat2):
    RADIUS = 6378137
    PI = math.pi
    return 2 * RADIUS * math.asin(math.sqrt(pow(math.sin(PI * (lat1 - lat2) / 360), 2) + math.cos(PI * lat1 / 180) *
                                            math.cos(PI * lat2 / 180) * pow(math.sin(PI * (lng1 - lng2) / 360), 2)))


def log_pdf(x, y, left, right, number):
    bins = np.logspace(left, right, number)
    bins2 = list(bins)
    bins_all = {}
    for i in range(len(bins) - 1):
        bins_all[bins[i]] = []
    widths = (bins[1:] - bins[:-1])
    for i in range(len(x)):
        for j in range(len(bins)):
            if (x[i] < bins[j]):
                bins_all[bins[j - 1]].append(y[i])
                break
            if (x[i] == bins[j]):
                bins_all[bins[j]].append(y[i])
                break
    x_new = []
    y_new = []
    for key, value in bins_all.items():
        if (len(value) > 0):
            index_this = bins2.index(key)
            y_new.append(np.sum(value) / widths[index_this])
            #            y_new.append(np.mean(value))
            x_new.append(key)
    return x_new, y_new
# 拟合函数
# def func2(x, a, b1, b2):
#     return a * pow(x, -b1) * np.exp((-b2) * x)


# ## 拟合函数

# In[3]:


# 直线
def func(x, a, b):
    return a * x + b


# 
def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
# 泊松
def func1(x, a):
    return (a**x/factorial(x)) * np.exp(-a)

# normal


def func2(x, mu, sigma):
    return np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))

# 指数


def func3(x, a, b):
    return a * np.exp(-b * x)

# log_normal


def func4(x, a, mu, sigma):
    return a / x * 1. / (sigma * np.sqrt(2. * np.pi)) * np.exp(-(np.log(x) - mu)**2 / (2. * sigma**2))

# 幂律


def func5(x, a, b):
    return a*pow(x, b)

# 幂律和指数


def func6(x, a, b1, b2, c):
    return a*pow(x+c, -b1)*np.exp((-b2)*x)


def func7(x, a, b1, b2):
    return a*pow(x, -b1)*np.exp((-b2)*x)


def r2(y_true, y_no):
    y_true = np.array(y_true)
    y_no = np.array(y_no)
    ybar = np.mean(y_true)
    ssreg = np.sum((y_no - y_true)**2)
    sstot = np.sum((y_true - ybar)**2)
    return 1-ssreg/sstot


# In[4]:


def gyration(dict1):
    gyration_list = []
    bbkeys = dict1.keys()
    for bb in bbkeys:
        x = []
        y = []
        # (bikeid, self.start_time, self.start_time, sid, sid,lng1,lat1,lng2,lat2)
        for item in dict1[bb]:
            x.append(item[-2])
            y.append(item[-1])
#         print(x)
#         print(y)
        x_mid = np.mean(x)
        y_mid = np.mean(y)
        dis = 0
        for _x, _y in zip(x, y):
            dis += (CalculateDistance(_x, _y, x_mid, y_mid) / 1000)
        if dis == 0:
            continue
        gyration_list.append(dis / len(x))
    return gyration_list


# ## Bike类

# In[5]:


class Bike:
    def __init__(self):
        # 车辆原始订单数据
        self.order_data = []
        # 按小时分组的trip(start_pos,end_pos,start_id,end_id)
        self.trip_dict = {}
        # 按车辆分组的订单列表字典，()
        self.bike_dict = {}
        # 栅格字典(记录每个栅格的车辆信息)
        self.move_list = []
        self.user_dict = {}
        self.grid_dict = {}
        # 控制冗余率
        self.grid_dict_raw = {}
        self.time_format = ''
        self.start_time = None
        self.sday = None
        self.eday = None
        self.vechile_condition = {}
        self.vcs = {}
        self.max_demand = 0

        self.history_nums = {}
        self.trip_nums = {}

        # 真实数据的rank_list
        self.rank_list = []
        self.time_list = []
        self.trips_list = []
        self.r_popt = []
        self.r_pcov = []

        # 车辆、用户的平均骑行距离
        self.distance_list = []
        self.bike_distance_list = []
        self.user_distance_list = []

        # 车辆、用户的总骑行次数
        self.bike_trips_list = []
        self.user_trips_list = []

        # 车辆/用户的独特地点数，每个小时有一个list
        self.bike_unique_list = {}
        self.user_unique_list = {}

        # i_th trip
        self.bike_ith_list = []
        self.user_ith_list = []

        # wait interval
        self.bike_wait_interval_list = []
        self.user_wait_interval_list = []

    def set_start_time(self, st):
        self.start_time = datetime.strptime(st, self.time_format)
        self.start_timestamp = time.mktime(self.start_time.timetuple())

    def set_time_format(self, tf):
        self.time_format = tf

    def set_day(self, s, e):
        self.sday = s
        self.eday = e

    def read_data(self, path):
        """
        读取原始订单数据
        """
        data = []
        with open(path) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                data.append(line)
            self.order_data = data

    def get_rank_list(self):
        data = self.order_data
        bike_index = index.Index()
        dict1 = {}
        dict2 = {}
        rank_list = []
        err = 0
        succ = 0
        ii = 0
        start_time = self.start_time
        for trip in data[1:]:
            bikeid = int(trip[2])
            stime = trip[4]
            etime = trip[5]
            sdt = datetime.strptime(stime, self.time_format)
            edt = datetime.strptime(etime, self.time_format)
            sx = float(trip[6])
            sy = float(trip[7])
            ex = float(trip[8])
            ey = float(trip[9])
            if bikeid not in dict1:
                # 使用次数，等待时间（订单时间-上次使用时间），车况值, 上次使用时间（结束时刻），车辆状态0/1
                dict1[bikeid] = [1, 0, 1.0, start_time, 0]
                # 初始化插入
                bike_index.insert(bikeid, [sx, sy, sx, sy])
                # 车辆的位置
                dict2[bikeid] = [sx, sy]
            else:
                continue
        err = 0
        for trip in data[1:]:
            bikeid = int(trip[2])
            stime = trip[4]
            etime = trip[5]
            sdt = datetime.strptime(stime, self.time_format)
            edt = datetime.strptime(etime, self.time_format)
            sx = float(trip[6])
            sy = float(trip[7])
            ex = float(trip[8])
            ey = float(trip[9])
    #         if bikeid not in dict2:
    #             dict2[bike] = []
    #         dict2[bike] = [ex, ey]
            bound = 0.0025
            intersec = list(bike_index.intersection(
                [sx-bound, sy-bound, sx+bound, sy+bound]))
            # 可用的车辆列表
            ok = []
            isMove = True
            for bid in intersec:
                if bid == bikeid:
                    # 如果找到了这辆车说明没有被移动过
                    ismove = False
                if dict1[bid][4] == 1 and dict1[bid][3] < sdt:
                    dict1[bid][4] = 0
                if dict1[bid][4] == 0:
                    wtime = time.mktime(sdt.timetuple()) -                         time.mktime(dict1[bid][3].timetuple())
                    dict1[bid][1] = wtime
    #                 if dict1[bid][0] == 0 or dict1[bid][1] == 0:
    #                     print(dict1[bid][0], dict1[bid][1])
    #                     print(trip)
                    cond = 1 / dict1[bid][0] * 3600 / (3600 + dict1[bid][1])
                    dict1[bid][2] = cond
                    ok.append([bid, cond])
            if isMove:
                # 真实位置
                tx, ty = dict2[bikeid][0], dict2[bikeid][1]
                # 从原位置移出
                bike_index.delete(bikeid, [tx, ty, tx, ty])
                # 移到新位置
                bike_index.insert(bikeid, [sx, sy, sx, sy])
                wtime = time.mktime(sdt.timetuple()) -                     time.mktime(dict1[bikeid][3].timetuple())
                dict1[bikeid][1] = wtime
                cond = 1 / dict1[bikeid][0] * 3600 / (3600 + dict1[bikeid][1])
                dict1[bikeid][2] = cond
                ok.append([bikeid, cond])
            ok.sort(key=lambda x: x[1], reverse=True)
            ii = 0
            while ii < len(ok):
                if ok[ii][0] == bikeid:
                    break
                ii += 1
            if ii == len(ok):
                err += 1
                print("error")
            else:
                succ += 1
                bike_index.delete(bikeid, [sx, sy, sx, sy])
                bike_index.insert(bikeid, [ex, ey, ex, ey])
                dict2[bikeid] = [ex, ey]
                dict1[bikeid][0] += 1
                dict1[bikeid][3] = edt
                dict1[bikeid][4] = 1
                rank_list.append(ii+1)
        self.rank_list = rank_list

    def init_position(self):
        """
        1.确定0时刻车辆的位置
        2.计算出行距离、订单、等待时间、
        """
        data = self.order_data
        for item in data[1:]:
            # 遍历订单数据
            userid = item[1]
            bikeid = item[2]
            start_time = item[4]
            end_time = item[5]
            sdt = datetime.strptime(start_time, self.time_format)
            edt = datetime.strptime(end_time, self.time_format)
            sid = item[-2]
            eid = item[-1]
            stt = (sdt.day - self.sday) * 24 + sdt.hour
            ett = (edt.day - self.sday) * 24 + edt.hour
            lng1, lat1 = float(item[6]), float(item[7])
            lng2, lat2 = float(item[8]), float(item[9])

            distance = int(CalculateDistance(lng1, lat1, lng2, lat2))
            self.distance_list.append(distance)
            self.time_list.append(sdt.hour)
            if stt not in self.trip_nums:
                self.trip_nums[stt] = 0
            self.trip_nums[stt] += 1

            if ett not in self.trip_nums:
                self.trip_nums[ett] = 0
            self.trip_nums[ett] += 1

            if sid not in self.history_nums:
                self.history_nums[sid] = {}
            # 订单数+1
            shh = (sdt.day-self.sday)*24 + sdt.hour
            if shh not in self.history_nums[sid]:
                self.history_nums[sid][shh] = 0
            self.history_nums[sid][shh] += 1

            if eid not in self.history_nums:
                self.history_nums[eid] = {}
            ehh = (edt.day-self.sday)*24 + edt.hour
            if ehh not in self.history_nums[eid]:
                self.history_nums[eid][ehh] = 0
            # 订单数+1
            self.history_nums[eid][ehh] += 1

            if sid not in self.grid_dict_raw:
                self.grid_dict_raw[sid] = {}
                self.grid_dict_raw[sid][0] = []
            if eid not in self.grid_dict_raw:
                self.grid_dict_raw[eid] = {}
                self.grid_dict_raw[eid][0] = []

            if stt not in self.trip_dict:
                # 更新trip
                self.trip_dict[stt] = []
            sdtimestamp = time.mktime(sdt.timetuple())
            edtimestamp = time.mktime(edt.timetuple())
            self.trip_dict[stt].append((sid, eid, sdtimestamp, edtimestamp))

            dis = CalculateDistance(lng1, lat1, lng2, lat2)
            b = (bikeid, sdt, edt, sid, eid, lng1, lat1, lng2, lat2, dis)
            if bikeid not in self.bike_dict:
                # 车辆第一次出现更新P(0), P(t)
                # bike_Id, 使用次数，等待时间（订单时间-上次使用时间），车况值, 上次使用时间（结束时刻），车辆状态0/1
                # 每次在选车前计算一下等待时间就行了
                self.grid_dict_raw[sid][0].append(
                    [bikeid, 1, 0, 1.0, self.start_timestamp, 0])
                self.bike_dict[bikeid] = []
            if len(self.bike_dict[bikeid]) != 0:
                last_trip = self.bike_dict[bikeid][-1]
                t1 = time.mktime(last_trip[2].timetuple())
                t2 = time.mktime(b[1].timetuple())
                self.bike_wait_interval_list.append((t2 - t1) // 3600)
            self.bike_dict[bikeid].append(b)

            if userid not in self.user_dict:
                self.user_dict[userid] = []

            if len(self.user_dict[userid]) != 0:
                last_trip = self.user_dict[userid][-1]
                t1 = time.mktime(last_trip[2].timetuple())
                t2 = time.mktime(b[1].timetuple())
                self.user_wait_interval_list.append((t2-t1) // 3600)
                # 是否发生了移动？这次的订单起点与上次的订单终点进行比较如果不同就发生了移动，记录移动次数
                if last_trip[4] != b[3]:
                    self.move_list.append((last_trip[4],b[3]))
            self.user_dict[userid].append(b)
        

    def reduce(self, r):
        grid_keys = self.grid_dict_raw.keys()
        dict1 = copy.deepcopy(self.grid_dict_raw)
        for g in grid_keys:
            temp = len(dict1[g][0])
            num = int(temp * r)
#             print(temp,num)
            seq = 1
            while temp < num:
                bikeid = str(g) + 'add'+ str(seq)
                newbike = [bikeid, 1, 0, 1.0, self.start_timestamp, 0]
                dict1[g][0].append(newbike)
                temp = len(dict1[g][0])
                seq += 1
            while temp > num:
                dict1[g][0].pop()
                temp = len(dict1[g][0])
        self.grid_dict = copy.deepcopy(dict1)

    def get_distance_list(self):
        """
        bike_distance_list:
        user_distance_list:
        """
        blist = []
        ulist = []
        b_ulist = []
        u_ulist = []
        b_ithlist = []
        u_ithlist = []
        bbkeys = self.bike_dict.keys()
        uukeys = self.user_dict.keys()
        for b in bbkeys:
            temp = set()
            distance = 0
            trip_num = len(self.bike_dict[b])
            max_dis = -1
            max_i = 0
            ii = 0
            for trip in self.bike_dict[b]:
                temp.add(trip[4])
                lng1, lat1, lng2, lat2 = trip[-5], trip[-4], trip[-3], trip[-2]
                dis = trip[-1]
                distance += dis
                if dis >= max_dis:
                    max_dis = dis
                    max_i = ii
                ii += 1

            distance /= 1000
            if (distance == 0):
                continue
            blist.append(distance / trip_num)
#             if len(temp) == 1:
#                 print(self.bike_dict[b])
            b_ulist.append(len(temp))
            b_ithlist.append(max_i+1)

        for u in uukeys:
            temp = set()
            distance = 0
            trip_num = len(self.user_dict[u])
            if trip_num > 200:
                print(u, trip_num)
            max_dis = -1
            max_i = 0
            ii = 0
            for trip in self.user_dict[u]:
                #                 print(trip)
                temp.add(trip[4])
                lng1, lat1, lng2, lat2 = trip[-5], trip[-4], trip[-3], trip[-2]
                dis = trip[-1]
                distance += dis
                if dis >= max_dis:
                    max_dis = dis
                    max_i = ii
                ii += 1
            distance /= 1000
            if distance == 0:
                continue
            ulist.append(distance / trip_num)
#             print(temp)
            u_ulist.append(len(temp))
            u_ithlist.append(max_i+1)

        self.bike_distance_list = blist
        self.user_distance_list = ulist
        self.bike_unique_list = b_ulist
        self.user_unique_list = u_ulist
        self.bike_ith_list = b_ithlist
        self.user_ith_list = u_ithlist
    def simulation(self, t, rank_list, bike_list,bike_trip,count_list,PROBABILITY_MAP,err_list):
        """
        rank_list:出行的选车排名分布
        bike_list:车辆的使用次数分布
        time_list:出行时间分布
        distance_list:出行距离分布
        """
#         current_t = dict()
        grid_keys = self.grid_dict.keys()
        for gg in grid_keys:
#             current_t[gg] = copy.deepcopy(self.grid_dict[gg][t])
            self.grid_dict[gg][t+1] = copy.deepcopy(self.grid_dict[gg][t])
        count = 0
        flag = 0
        suc = 0
        err = 0
        t10=0
        t20=0
        t30=0
        mkt = 0
        global sort_t,cal_t
        sort_t,cal_t = 0,0
        for trip in self.trip_dict[t]:
#             print('ttttt')
            """
            2023年3月31日16:33:12
            关心的是每次选车的rank情况，所以只需要有选车时的车况就可以解决。
            1.每次读到订单时更新一下出发地栅格的车辆状况。
            2.更新之后再选择车辆。
            """
#             print(trip)
            count += 1
            sid = trip[0]
            eid = trip[1]
            stime = trip[2]
            etime = trip[3]
            pool1 = self.grid_dict[sid][t+1]
#             print(pool1)
            ii = 0
            st= time.time()
            for ii in range(len(self.grid_dict[sid][t+1])):
                # 结束时间
                item = pool1[ii]
                mkt1 = time.time()
#                 t1 = time.mktime(item[4].timetuple())
                # 订单时间
#                 t2 = time.mktime(stime.timetuple())
                t1 = item[4]
#                 print(t1)
                t2 = stime
#                 print(t2)
                mkt2 = time.time()
                mkt += (mkt2-mkt1)
                if t2 >= t1:
                    # 计算到当前订单的等待时间
                    wtime = t2 - t1
                    vc = 1 / item[1] * 3600 / (3600 + wtime)
                    # 车辆为可用状态
                    pool1[ii] = [item[0], item[1], wtime, vc, item[4], 0]
            et= time.time()
            t10+=(et-st)
            l1 = len(pool1)
            count_list.append(l1)
#             print(f"订单开始:{len(pool1)}")
            if l1 == 0:
                err += 1
#                 print(sid, pool1)
                continue
            t1= time.time()
        
            rk, bikeid = select2(pool1, PROBABILITY_MAP)
            t2= time.time()
            t20+=(t2-t1)
            if bikeid == -1:
#                 print(sid, pool1)
                err += 1
                continue
            suc += 1
            rank_list.append(rk)
            flag = True
            
            t1= time.time()
            for ii in range(len(pool1)):
                if pool1[ii][0] == bikeid:
                    bike_sel = pool1.pop(ii)
                    flag = False
                    break
            if flag:
                print("没找到这辆车！！！！！")
            # 使用次数+1，等待时间置0，上次使用时间更新为订单结束时间，使用状态为1
            if bike_sel[0] not in bike_list:
                bike_list[bike_sel[0]] = 0
            bike_list[bike_sel[0]] += 1
            if bike_sel[0] not in bike_trip:
                bike_trip[bike_sel[0]] = []
            bike_trip[bike_sel[0]].append(sid)
            new_b = [bike_sel[0], bike_sel[1] + 1, 0, bike_sel[3], etime, 1]
            self.grid_dict[eid][t+1].append(new_b)
            t2= time.time()
            t30+=(t2-t1)
        err_list.append(err)
        
        print("计算等待时间："+str(mkt))
        print("状态更新："+str(t10))
        print("车辆排序时间："+str(sort_t))
        print("计算概率事件："+str(cal_t))
        print("选择车辆："+str(t20))
        print("移动车辆："+str(t30))

        print(suc, err)


    def select2(self, pool):
        """
        1. 计算每辆车的概率值
        2. 使用轮盘赌的方法选出被骑的车辆
        3. 
        """
        # bike_Id, 使用次数，等待时间（订单时间-上次使用时间），车况值, 上次使用时间（结束时刻），车辆状态0/1
        bikeid = -1
        rank = -1
        list1 = []
        for bb in pool:
            if bb[-1] == 0:
                # 车辆号、车况值
                list1.append([bb[0], bb[3]])
        # 按车况值排名
        list1.sort(key=lambda x: x[1], reverse=True)
        p_sum = 0
        for i, item in enumerate(list1):
            pp = func7(i + 1, *self.r_popt)
            list1[i].append(pp)
            p_sum += pp
        p_random = random.uniform(0, p_sum)
        p_cum = 0
    #     print(list1)
    #     print(p_sum,p_random)
        for i, item in enumerate(list1):
            p_cum += list1[i][2]
            if p_cum > p_random:
                rank = i+1
                bikeid = list1[i][0]
                break
        return rank, bikeid
    def move1(self, t, move_list,th,CENTRE_MAP):
        """
        调度策略：按照使用次数
        2023年4月3日10:28:46
        建立索引，根据t时刻最开始的车辆分布，移动车辆，得到移动后的车辆分布。
        1. 如果是0时刻，不用移动
        2. 如果是t时刻，需要移动，移动是在simulation前完成的
        3. 结合[t, t + 1）时间内的订单数量，得到移入栅格、移出车辆。（这部分以文件形式输出提供优化调度使用）
        4. 得到移入移出
        """
        gg = self.grid_dict
        grid_key = self.grid_dict.keys()
#         t1 = time.time()
        in_grids, out_grids = self.find_bike_grid5(t, thre=th)
#         t2 = time.time()
#         print("找车的时间："+str(t2-t1))
        in_keys = in_grids.keys()
        out_keys = out_grids.keys()
        in_nums, out_nums = 0, 0
        for ii in in_keys:
            in_nums += in_grids[ii][1]
        for oo in out_keys:
            out_nums += out_grids[oo][1]
#         id_map = bidict()
        out_bike_index = index.Index()
        # 通过车辆索引找到所属栅格，车辆编号
        omap = {}
        # 车辆索引
        bidx = 0
        # 将所有的移出车辆添加到R-tree中
#         t1 = time.time()
        for oo in out_keys:
            for bb in out_grids[oo][0]:
                cor = list(map(int, oo.split(',')))
                cond = float(bb[3])
                old_rank = int(bb[-1])
                # 应该是这一步的时间很长,可以优化，2023年6月2日15:35:08
                lon, lat = CENTRE_MAP[oo]
                out_bike_index.insert(bidx,[lon, lat, lon, lat])
                omap[bidx] = [oo, bb, lon, lat]
                bidx += 1
#         t2 = time.time()
#         print("R1的时间："+str(t2-t1))
        success = 0
        error = 0
        t1 = time.time()
        for ii in in_keys:
            d= 0.0025
            cor = list(map(int,ii.split(',')))
            lon, lat = CENTRE_MAP[ii]
#             self.grid_dict[ii][t].sort(
#                                 key=lambda x: x[1], reverse=True)
            p=0
            while in_grids[ii][1] > 0 and p <= 20:
                box = [lon-d*p, lat-d*p, lon+d*p, lat+d*p]
                list1 = list(out_bike_index.intersection(box))
                for item in list1:
                    if in_grids[ii][1] == 0:
                        break
                    oo = omap[item][0]
                    bb = omap[item][1]
                    cond = float(bb[3])
                    old_rank = int(bb[-1])
                    jj = 0
                    # TODO:这里用二分查找优化一下
                    if True:
                        # 说明排名提升了
                        success+=1
                        # 移动到栅格里
                        self.grid_dict[ii][t].append(bb[:6])
#                         self.grid_dict[ii][t].sort(
#                                 key=lambda x: x[1], reverse=True)
                        k = 0
                        # 从原栅格移出
                        flag = True
                        
                        while k < len(self.grid_dict[oo][t]):
                            if self.grid_dict[oo][t][k][0] == bb[0]:
                                self.grid_dict[oo][t].pop(k)
                                flag= False
                                break
                            k+=1
                        if flag:
                            print("没找到这辆车！！！！！")
                            print(bb)
#                             print(type(bb[0]))
                            print(self.grid_dict[oo][t])
                            return 
                        if bb[0] not in move_list:
                            move_list[bb[0]] = 0
                        move_list[bb[0]] += 1
                        _lon, _lat = omap[item][2], omap[item][3]
                        out_bike_index.delete(item, [_lon, _lat, _lon, _lat])
                        in_grids[ii][1] -= 1
                p += 1
        t2=time.time()
        print("R2："+str(t2-t1))
        for ii in in_keys:
            error+=in_grids[ii][1]
        print(success, error)
        
    def move2(self, t,move_list,thre1,thre2):
        """
        按照排名
        2023年4月3日10:28:46
        建立索引，根据t时刻最开始的车辆分布，移动车辆，得到移动后的车辆分布。
        1. 如果是0时刻，不用移动
        2. 如果是t时刻，需要移动，移动是在simulation前完成的
        3. 结合[t, t + 1]时间内的订单数量，得到移入栅格、移出车辆。（这部分以文件形式输出提供优化调度使用）
        4. 得到移入移出
        """
        gg = self.grid_dict
        grid_key = self.grid_dict.keys()
        in_grids, out_grids = self.find_bike_grid2(t,thre1,thre2)
        
        in_keys = in_grids.keys()
        out_keys = out_grids.keys()
        in_nums, out_nums = 0, 0
        for ii in in_keys:
            in_nums += in_grids[ii][1]
        for oo in out_keys:
            out_nums += out_grids[oo][1]
#         id_map = bidict()
        out_bike_index = index.Index()
        # 通过车辆索引找到所属栅格，车辆编号
        omap = {}
        # 车辆索引
        bidx = 0
        # 将所有的移出车辆添加到R-tree中
        for oo in out_keys:
            for bb in out_grids[oo][0]:
                cor = list(map(int, oo.split(',')))
                cond = float(bb[3])
                old_rank = int(bb[-1])
                lon, lat = tbd.grid_to_centre(cor, params_rec)
                out_bike_index.insert(bidx,[lon, lat, lon, lat])
                omap[bidx] = [oo, bb, lon, lat]
                bidx += 1
        success = 0
        error = 0
        for ii in in_keys:
            d= 0.0025
            cor = list(map(int,ii.split(',')))
            lon, lat = tbd.grid_to_centre(cor, params_rec)
            self.grid_dict[ii][t].sort(
                                key=lambda x: x[1], reverse=True)
            p=0
            while in_grids[ii][1] > 0 and p <= 20:
                # 查找范围，以lon,lat为中心
                box = [lon-d*p, lat-d*p, lon+d*p, lat+d*p]
                list1 = list(out_bike_index.intersection(box))
#                 print(list1)
                for item in list1:
                    if in_grids[ii][1] == 0:
                        break
                    oo = omap[item][0]
                    bb = omap[item][1]
                    cond = float(bb[3])
                    old_rank = int(bb[-1])
                    jj = 0
                    # TODO:这里用二分查找优化一下
                    while jj < len(self.grid_dict[ii][t]):
                        if cond >= self.grid_dict[ii][t][jj][3]:
                            break
                        jj += 1
                    if old_rank > jj + 1:
                        # 说明排名提升了
#                         print(f"排名提升了{old_rank-jj-1}")
                        success+=1
                        # 移动到栅格里
                        self.grid_dict[ii][t].append(bb[:6])
                        self.grid_dict[ii][t].sort(
                                key=lambda x: x[3], reverse=True)
                        k = 0
                        # 从原栅格移出
                        flag=True
                        while k < len(self.grid_dict[oo][t]):
                            if self.grid_dict[oo][t][k][0] == bb[0]:
                                self.grid_dict[oo][t].pop(k)
                                flag=False
                                break
                            k+=1
                        if flag:
                            print("没找到车，移出失败")
                        _lon, _lat = omap[item][2], omap[item][3]
                        out_bike_index.delete(item, [_lon, _lat, _lon, _lat])
                        in_grids[ii][1] -= 1
                p += 1
        for ii in in_keys:
            error+=in_grids[ii][1]
        print(success, error)
            
        
        
        
#         in_grid_index = index.Index()
#         # 1.向Rtree中插入所有的接收栅格
#         for ii, ig in enumerate(in_keys):
#             id_map[ig] = ii
#             cor = list(map(int, ig.split(',')))
#             lon, lat = tbd.grid_to_centre(cor, params_rec)
#             in_grid_index.insert(id_map[ig], [lon, lat, lon, lat])
#         # 2.移动车辆
#         success = 0
#         error = 0
#         od = {}
#         for oo in out_keys:
#             # 记录被移除的车辆
#             bb_remove = []
#             for bb in out_grids[oo][0]:
#                 # ['7910418184', 2, 16047.0, 0.09161704076958314, datetime.datetime(2020, 9, 14, 5, 9, 1), 0, 130, '59,78']
#                 #             print(bb)
#                 cor = list(map(int, oo.split(',')))
#                 cond = float(bb[3])
#                 old_rank = int(bb[-2])
#                 lon, lat = tbd.grid_to_centre(cor, params_rec)
#                 nearest_grid = list(in_grid_index.nearest(
#                     [lon, lat, lon, lat], objects=True))
#                 flag = False
#                 for gb in nearest_grid:
#                     if flag:
#                         break
#                 # TODO：这里要增加一下选车的函数2023年3月3日14:07:05
#                     g = gb.id
#                     raw_id = id_map.inverse[g]
#             #         print(f"栅格{g}的当前可接受量：{in_grids[id_map.inverse[g]]}")
#                     target = in_grids[raw_id]
#     #                 print(target)
#                     if target[1] > 0:
#                         # 如果当前栅格还可以接受车辆的话，判断排名
#                         k = 0
#                         while k < len(self.grid_dict[raw_id][t]):
#                             #                         print(bike.grid_dict[raw_id][t][k])
#                             if cond >= self.grid_dict[raw_id][t][k][3]:
#                                 break
#                             k += 1
#                         if k + 1 >= old_rank:
#                             continue
#                         else:
#                             self.grid_dict[raw_id][t].append(bb[:6])
#                             bb_remove.append(bb[0])
#                             jj = 0
#                             while jj < len(self.grid_dict[oo][t]):
#                                 if self.grid_dict[oo][t][jj][0] == bb[0]:
#                                     # 从原栅格中移除车辆
#                                     self.grid_dict[oo][t].pop(jj)
#                                     break
#                                 jj += 1
#                             self.grid_dict[raw_id][t].sort(
#                                 key=lambda x: x[3], reverse=True)
#                             success += 1
#                             flag = True
#                             if (oo, raw_id) not in od:
#                                 od[oo, raw_id] = 0
#                             od[oo, raw_id] += 1
#                             target[1] -= 1
#                     if target[1] == 0:
#                         # 移除该栅格
#                         in_grid_index.delete(gb.id, gb.bbox)
#                 if flag == False:
#                     error += 1
    #             for item in out_grids[oo][0]:
    #                 if item[0] in bb_remove:
    #                     print(item)
    #     print(out_grids)
#         print(out_nums, success, error)

    def group_by_order(self):
        # orderid	userid	bikeid	count_bike	start_time	end_time	start_location_x	start_location_y	end_location_x	end_location_y	d	sid	eid
        data = self.order_data
        for item in data[1:]:
            # 遍历订单数据
            userid = item[1]
            bikeid = item[2]
            start_time = item[4]
            end_time = item[5]
#             print(start_time,end_time)
            sdt = datetime.strptime(start_time, self.time_format)
            edt = datetime.strptime(end_time, self.time_format)
            sid = item[-2]
            eid = item[-1]

            b = (bikeid, sdt, edt, sid, eid)

            if bikeid not in self.bike_dict:
                fb = (bikeid, self.start_time, self.start_time, sid, sid)
                self.bike_dict[bikeid] = [fb]
            self.bike_dict[bikeid].append(b)

            if userid not in self.user_dict:
                self.user_dict[userid] = []
            self.user_dict[userid].append(b)

            # 统计栅格的历史订单数量,根据历史订单数量确定栅格车辆的投放量2023年2月27日17:23:47
            if sid not in self.history_nums:
                self.history_nums[sid] = 0
            # 订单数+1
            self.history_nums[sid] += 1

            if sid not in self.grid_dict:
                self.grid_dict[sid] = {}

            if eid not in self.history_nums:
                self.history_nums[eid] = 0
            # 订单数+1
            self.history_nums[eid] += 1

            if eid not in self.grid_dict:
                self.grid_dict[eid] = {}

            # 初始化车辆位置
            self.init_position()
#                 self.history_nums[sid] = [[0 for k in range(24)]for j in range(self.eday - self.sday + 1)]
#             self.history_nums[sid][sdt.day-self.sday][sdt.hour]+=1
#         print(self.bike_dict)

    def cal_vehicle_condition(self):
        # 计算每个小时的车况Vc(t,k)
        # Vc(0,1)
        bike_list = self.bike_dict.keys()
        for b_i in bike_list:
            b = bike.bike_dict[b_i]  # 每辆车的订单信息
            s_t = self.start_time
            for day in range(self.sday, self.eday + 1):
                if day not in self.vcs:
                    self.vcs[day] = [[] for t in range(24)]
                for t in range(0, 24):
                    _t = datetime(year=s_t.year, month=s_t.month,
                                  day=day, hour=t, minute=0, second=0)  # 当前时间点
                    times = 0
                    det_t = 0
                    idx = 0
                    flag = 0
                    for idx in range(len(b)):
                        item = b[idx]
                        end_time = item[2]
        #                 print(end_time)
                        if _t >= end_time:
                            # 如果当前时间>=订单的结束时间的话，使用次数加一
                            times += 1
                        else:
                            # 如果当前时间<订单的结束时间的话，delta_t = 当前时间-上一次订单的结束时间
                            flag = 1
                            break
                    if flag:
                        # 所有订单都小于当前时间，最后完成的订单为最末尾的订单
                        pre_order = b[idx-1]
                    else:
                        # 最后完成的订单为当前订单的上一个订单
                        pre_order = b[-1]

                    # 这里要转换成timestamp进行秒数运算
                    u1 = time.mktime(pre_order[2].timetuple())
                    u2 = time.mktime(_t.timetuple())
                    det_t = u2 - u1
                    _vc = 1 / times * 3600 / (det_t+3600)

#                     # 这里有问题，每当0点的时候vc都为1
#                     if day == 11 and t == 0 and _vc ==1:
# #                         print(f"_t:{_t}")
#                         print(f"order:{b}")
#                         print(f"preorder:{pre_order}")
#                         print(f"_t:{_t}")
#                         print(f"det_t:{det_t}")
#                         print("-"*50)

                    """
                        将车辆添加到栅格里面
                    """
                    p_bikeid = pre_order[0]
                    # 最后出现的位置
                    p_eid = pre_order[-1]
                    if p_eid not in self.grid_dict:
                        self.grid_dict[p_eid] = [
                            [[]for k in range(24)]for j in range(self.eday - self.sday + 1)]
                    self.grid_dict[p_eid][day -
                                          self.sday][t].append((p_eid, p_bikeid, _vc))
                    self.vcs[day][t].append(_vc)

    def distribution_plot(self, _kde=False):
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        for day in range(self.sday, self.eday + 1):
            plt.figure(figsize=(15, 10))
            for t in range(1, 7):
                plt.subplot(2, 3, t)
                plt.title("day={},t={}".format(day, t-1))
                sns.set_style('darkgrid')
                sns.distplot(self.vcs[day][t-1],
                             kde=_kde,
                             kde_kws={'linestyle': '--', 'linewidth': '1', 'color': '#c72e29',  # 设置外框线属性
                                      },
                             color='#098154',
                             axlabel='Vehicle Condition',  # 设置x轴标题
                             )
            plt.show()
            plt.figure(figsize=(15, 10))
            for t in range(7, 13):
                plt.subplot(2, 3, t-6)
                plt.title("day={},t={}".format(day, t-1))
                sns.set_style('darkgrid')
                sns.distplot(self.vcs[day][t-1],
                             kde=_kde,
                             kde_kws={'linestyle': '--', 'linewidth': '1', 'color': '#c72e29',  # 设置外框线属性
                                      },
                             color='#098154',
                             axlabel='Vehicle Condition',  # 设置x轴标题
                             )
            plt.show()
            plt.figure(figsize=(15, 10))
            for t in range(13, 19):
                plt.subplot(2, 3, t-12)
                plt.title("day={},t={}".format(day, t-1))
                sns.set_style('darkgrid')
                sns.distplot(self.vcs[day][t-1],
                             kde=_kde,
                             kde_kws={'linestyle': '--', 'linewidth': '1', 'color': '#c72e29',  # 设置外框线属性
                                      },
                             color='#098154',
                             axlabel='Vehicle Condition',  # 设置x轴标题
                             )
            plt.show()
            plt.figure(figsize=(15, 10))
            for t in range(19, 25):
                plt.subplot(2, 3, t-18)
                plt.title("day={},t={}".format(day, t-1))
                sns.set_style('darkgrid')
                sns.distplot(self.vcs[day][t-1],
                             kde=_kde,
                             kde_kws={'linestyle': '--', 'linewidth': '1', 'color': '#c72e29',  # 设置外框线属性
                                      },
                             color='#098154',
                             axlabel='Vehicle Condition',  # 设置x轴标题
                             )
            plt.show()

        # 画某个小时的分布

    def plot_t(self, t):
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.figure(figsize=(15, 10))
        plt.title("t={}".format(t))
        hist, bin_edges = np.histogram(self.vcs[t])
        sns.set_style('darkgrid')
        sns.distplot(self.vcs[t])

    def plot_t_grid(self, day, hour, thre, _kde=False):
        day = int(day)
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        grid_list = self.grid_dict.keys()  # 栅格编号列表
        gb_list = []
        threshold = thre  # 调度阈值
        bike_sum = 0
        for g in grid_list:
            #                 print(g)
            #                 print(self.grid_dict[g])
            #                 print(self.grid_dict.get(g))
            gb = self.grid_dict.get(g)[day-self.sday][hour]
            if len(gb) >= threshold:  # 如果大于阈值则加到list里面
                gb_list.append((g, gb))
                bike_sum += len(gb)
        bike_mean = bike_sum // len(gb_list)
        for g, gb in gb_list:
            bikeCount = len(gb)
            # 需求订单数
            vc_list = []
            for item in gb:
                vc_list.append(item[2])
#                 print(vc_list)
            plt.title("Grid:[{}], bikeCount:{}, Q:{}".format(
                g, bikeCount, bikeCount - bike_mean))
            sns.distplot(vc_list,
                         kde=_kde,
                         kde_kws={'linestyle': '--', 'linewidth': '1', 'color': '#c72e29',  # 设置外框线属性
                                  },
                         color='#098154',
                         axlabel='Vehicle Condition',  # 设置x轴标题
                         )
            plt.show()

    def group_by_grid(self):
        # 订单按照栅格分类
        pass

    def plot_pdf(self, log=True):
        # 计算车辆的使用次数分布
        times_list = ll
#         entires = 0
#         for b in self.bike_dict.keys():
#             times = len(self.bike_dict[b])
#             times_list.append(times)
#             entires+=times
        result = Counter(times_list)
        x = list(result.keys())
        y = [result[_x]/entires for _x in x]
        plt.scatter(x, y)
        if log:
            plt.xscale('log')
            plt.yscale('log')

    def plot_bike_pdf(self, log=True):
        # 计算车辆的使用次数分布
        times_list = []
        entires = 0
        for b in self.bike_dict.keys():
            times = len(self.bike_dict[b])
            times_list.append(times)
            entires += times
        result = Counter(times_list)
        x = list(result.keys())
        y = [result[_x]/entires for _x in x]
        plt.scatter(x, y)
        if log:
            plt.xscale('log')
            plt.yscale('log')

    # 改成分布图

    def plot_user_pdf(self, log=True):
        # 计算用户的骑行次数分布
        times_list = []
        entires = 0
        for u in self.user_dict.keys():
            times = len(self.user_dict[u])
            times_list.append(times)
            entires += times
        result = Counter(times_list)
        x = list(result.keys())
        y = [result[_x]/entires for _x in x]
        plt.scatter(x, y)
        if log:
            plt.xscale('log')
            plt.yscale('log')

    def find_bike_grid3(self, t, thre=0):
        """
        t = (day - sday) * 24 + hour
        第一步，计算t时刻的移动量，确定in/out栅格
        这里需要修改一下，2023年4月17日10:57:20
        调度时间为6-晚10点
        """
        out_bikes = {}
        in_grids = {}
        grid_list = self.grid_dict.keys()
        gb_list = []
        bike_sum = 0
        for g in grid_list:
            gb = self.grid_dict[g][t]
            if len(gb) >= thre:  # 如果大于阈值则加到list里面
                gb_list.append((g, gb))
                bike_sum += len(gb)
#         bike_mean = bike_sum // len(gb_list)
        for g, gb in gb_list:
            # 每个栅格合理的投放量
            if t not in self.history_nums[g]:
                self.history_nums[g][t] = 0
            # TODO4:每个小时调度的量不能太大，不能只考虑订单占总订单的比例，2023年4月17日11:02:19
            # TODO5:冗余率：计算整天栅格的总订单数，某个时刻的车辆数，冗余率 = 车辆数 / 总订单数。计算总体冗余率
            temp = int(bike_sum * self.history_nums[g][t] / self.trip_nums[t])
#             if temp == 0:
#                 # 保证每个栅格至少有一辆车
#                 temp += 1
#                 print(self.history_nums[g], len(gb))
            if len(gb) > temp:
                # out_bikes
                on = len(gb) - temp
                gb.sort(key=lambda x: x[3], reverse=True)
                obs = []
                for idx in range(temp, len(gb)):
                    #                     print(gb[idx])
                    rank = idx + 1
                    if gb[idx][5] == 1:
                        continue
        # id, times, wait, cond, 车辆状况，rank=idx+1, idx=rank-1(排序过后的，移出时要进行一次排序操作)
                    obs.append([gb[idx][0], gb[idx][1], gb[idx][2],
                               gb[idx][3], gb[idx][4], gb[idx][5], rank, g])
                    # 将车从栅格移出,要现在移出吗？还是等等？这里暂时不能移出
                if g not in out_bikes:
                    # 只要把这些车移进去就行
                    # {g:[obs, len]}
                    out_bikes[g] = [obs, len(obs)]
            if len(gb) < temp:
                # in_grids,移入栅格
                in_grids[g] = [t, temp - len(gb)]
        return in_grids, out_bikes

    def find_bike_grid2(self, t, thre1,thre2):
        """
        计算冗余率
        """
        out_grids = {}
        in_grids = {}
        grid_keys = self.grid_dict.keys()
        gb_list = []
        for g in grid_keys:
            if t not in self.history_nums[g]:
                self.history_nums[g][t] = 0
            gb = self.grid_dict[g][t]
            gb_1 = []
            for b in gb:
                if b[4] == 0:
                    gb_1.append(b)
            if self.history_nums[g][t] == 0:
                continue
            rr = len(gb) / self.history_nums[g][t] 
            if rr < thre1:
                # 需要调入
                in_grids[g] = [t, int(thre1 * self.history_nums[g][t]) - len(gb)]
            elif rr > thre2:
                # 可以调出
                obs = []
                out_grids[g] = []
                # 车况只从大到小排
                gb.sort(key=lambda x: x[3], reverse=True)
                kk = int(thre2* self.history_nums[g][t])
                # 保留kk辆
                # 这里保证找出的车是有序的
                while kk < len(gb):
                    if gb[kk][5]==1:
                        kk+=1
                        continue
                    obs.append([gb[kk][0],gb[kk][1],gb[kk][2],gb[kk][3],gb[kk][4],gb[kk][5],kk + 1])
                    kk+=1
                out_grids[g]=[obs, len(obs)]
        return in_grids, out_grids
                
#         for g, gb in gb_list:
#             # 每个栅格合理的投放量
#             if t not in self.history_nums[g]:
#                 self.history_nums[g][t] = 0
#             # TODO4:每个小时调度的量不能太大，不能只考虑订单占总订单的比例，2023年4月17日11:02:19
#             # TODO5:冗余率：计算整天栅格的总订单数，某个时刻的车辆数，冗余率 = 车辆数 / 总订单数。计算总体冗余率
#             if self.history_nums[g][t] > 0:
#                 rr = len(gb) / self.history_nums[g][t]
#                 if rr < 2:
#                     print(len(gb), self.history_nums[g][t], len(
#                         gb) / self.history_nums[g][t])
#         return in_grids, out_bikes
    def find_bike_grid4(self, t, thre):
        """
        只考虑使用次数
        """
        out_grids = {}
        in_grids = {}
        grid_keys = self.grid_dict.keys()
        gb_list = []
        for g in grid_keys:
            if t not in self.history_nums[g]:
                self.history_nums[g][t] = 0
            gb = self.grid_dict[g][t]
            gb_1 = []
            for b in gb:
                if b[4] == 0:
                    gb_1.append(b)
            if self.history_nums[g][t] == 0:
                continue
            rr = len(gb) / self.history_nums[g][t] 
            if rr < thre:
                # 需要调入
                in_grids[g] = [t, int(thre * self.history_nums[g][t]) - len(gb)]

            elif rr > thre:
                # 可以调出
                obs = []
                out_grids[g] = []
                # 车况只从大到小排
#                 gb.sort(key=lambda x: x[3], reverse=True)
                # 从小到大排
                gb.sort(key=lambda x: x[1],reverse=True)
#                 print(gb)
                jj = int(thre * self.history_nums[g][t])
                # 保留kk辆
                # 这里保证找出的车是有序的
                curnum = len(gb)
                kk = 0
                while kk < len(gb):
                    if curnum <= jj:
                        break
                    if gb[kk][5]==1:
                        kk+=1
                        continue
                    obs.append([gb[kk][0], gb[kk][1], gb[kk][2], gb[kk][3], gb[kk][4], gb[kk][5], kk + 1])
                    curnum-=1
                    kk+=1
                    
                out_grids[g]=[obs, len(obs)]
        return in_grids, out_grids
    def find_bike_grid5(self, t, thre):
        """
        随机选择车
        """
        out_grids = {}
        in_grids = {}
        grid_keys = self.grid_dict.keys()
        gb_list = []
        for g in grid_keys:
            if t not in self.history_nums[g]:
                self.history_nums[g][t] = 0
            gb = copy.deepcopy(self.grid_dict[g][t])
            gb_1 = []
            for b in gb:
                if b[4] == 0:
                    gb_1.append(b)
            if self.history_nums[g][t] == 0:
                continue
            rr = len(gb) / self.history_nums[g][t] 
            if rr < thre:
                # 需要调入
                in_grids[g] = [t, int(thre * self.history_nums[g][t]) - len(gb)]
            elif rr > thre:
                # 可以调出
                obs = []
                out_grids[g] = []
                # 车况只从大到小排
#                 gb.sort(key=lambda x: x[3], reverse=True)
                # 从小到大排
                gb.sort(key=lambda x: x[1],reverse=True)
#                 print(gb)
                jj = int(thre * self.history_nums[g][t])
                # 保留kk辆
                # 这里保证找出的车是有序的
                curnum = len(gb)
                kk = 0
                while True:
#                     print('*')
#                     print(curnum,len(gb))
                    if curnum <= jj or len(gb) == 0:
                        break
                    kk = random.randint(0, len(gb) - 1)
#                     print(kk)
                    obs.append([gb[kk][0], gb[kk][1], gb[kk][2], gb[kk][3], gb[kk][4], gb[kk][5], kk + 1])
                    gb.pop(kk)
                    curnum = len(gb)
#                 while kk < len(gb):
#                     if curnum <= jj:
#                         break
#                     if gb[kk][5]==1:
#                         kk+=1
#                         continue
#                     obs.append([gb[kk][0], gb[kk][1], gb[kk][2], gb[kk][3], gb[kk][4], gb[kk][5], kk + 1])
#                     curnum-=1
#                     kk+=1
                    
                out_grids[g]=[obs, len(obs)]
        return in_grids, out_grids
    def cal_redundancy(self, _t):
        redundancy = [0 for t in range(_t)]
#         r_sum = [0 for t in range(24)]
        grid_list = self.grid_dict.keys()
        for t in range(_t):
            redundancy[t] = len(self.bike_dict.keys()) / len(self.trip_dict[t])
#             for g in grid_list:
#                 if t not in self.history_nums[g]:
#                     self.history_nums[g][t] = 0
#                 if self.history_nums[g][t] > 0:
#                     redun = len(self.grid_dict[g][t]) / self.history_nums[g][t]
#                     redundancy[t] = redundancy[t] + redun
#                     r_sum[t % 24] += 1
#         for t in range(24):
#             redundancy[t] = redundancy[t] / r_sum[t]
        return redundancy

    def find_bike_by_grid(grid_list):
        pass


# # Simulation

# ## bike0-真实出行
# 计算rank分布作为选车的依据

# In[6]:


bike0 = Bike()
# 设置时间格式
path  = './bj-1000m-10-16.csv'
start_time = '2017-05-10 00:00:00'
bike0.set_time_format("%Y-%m-%d %H:%M:%S")
ACCURACY = 1000
bike0.set_start_time(start_time)
bike0.set_day(10, 16)
DAY = 7
bike0.read_data(path)
bike0.init_position()
logger = get_logger("beijing", "./log/beijing.log")


# In[10]:


print(len(bike0.move_list))
# print(bike0.move_list)


# ### 计算栅格的中心点坐标

# In[ ]:


district = gpd.read_file(r'./beijing.geojson')
fig = plt.figure(1, (7, 5), dpi=150)
ax1 = plt.subplot(111)
district.plot(ax=ax1)
grid_rec, params_rec = tbd.area_to_grid(district, accuracy=ACCURACY)
CENTRE_MAP = dict()
grid_keys = bike0.grid_dict_raw.keys()
# print(grid_keys)
for key in grid_keys:
    if key not in CENTRE_MAP:
        cor = list(map(int, key.split(',')))
        CENTRE_MAP[key] = tbd.grid_to_centre(cor, params_rec)


# ### 真实出行的rank分布

# In[ ]:


PROBABILITY_MAP = dict()
bike0.get_rank_list()
result = Counter(bike0.rank_list)
rank_x = list(result.keys())
rank_x.sort()
rank_y = [result[_x] / len(bike0.rank_list) for _x in rank_x]
r_popt, r_pcov = curve_fit(func7, rank_x, rank_y, maxfev=10000)
rank_y_fit = []
for _x in rank_x:
    if _x not in PROBABILITY_MAP:
        prob = func7(_x, *r_popt) 
        PROBABILITY_MAP[_x] = prob
        rank_y_fit.append(prob)
plt.plot(rank_x, rank_y_fit,c='red')
plt.scatter(rank_x, rank_y)
plt.xscale('log')
plt.yscale('log')


# In[ ]:


# y_fit = []
# for _x in x:
#     if _x not in PROBABILITY_MAP:
#         prob = func7(_x, *r_popt) 
#         PROBABILITY_MAP[_x] = prob
#         y_fit.append(prob)
PROBABILITY_MAP = dict()
rank_x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1115, 1116, 1117, 1118, 1119, 1121, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1134, 1135, 1137, 1138, 1139, 1140, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1158, 1160, 1161, 1162, 1163, 1164, 1166, 1167, 1168, 1169, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1179, 1180, 1182, 1183, 1184, 1186, 1187, 1188, 1189, 1190, 1193, 1195, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 1235, 1236, 1237, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1247, 1248, 1249, 1250, 1251, 1253, 1255, 1256, 1257, 1260, 1262, 1264, 1265, 1266, 1267, 1268, 1270, 1271, 1274, 1276, 1277, 1279, 1280, 1282, 1284, 1285, 1286, 1287, 1288, 1290, 1297, 1298, 1300, 1301, 1302, 1304, 1305, 1306, 1307, 1308, 1310, 1311, 1312, 1315, 1318, 1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1338, 1340, 1341, 1342, 1345, 1346, 1347, 1349, 1350, 1352, 1353, 1354, 1358, 1359, 1360, 1361, 1362, 1369, 1370, 1371, 1374, 1377, 1381, 1382, 1383, 1385, 1388, 1389, 1391, 1392, 1393, 1395, 1397, 1400, 1402, 1403, 1405, 1406, 1408, 1410, 1415, 1416, 1417, 1420, 1422, 1423, 1428, 1429, 1434, 1436, 1440, 1441, 1445, 1446, 1447, 1448, 1451, 1453, 1455, 1456, 1457, 1458, 1459, 1464, 1468, 1469, 1474, 1475, 1476, 1480, 1481, 1484, 1487, 1488, 1489, 1493, 1496, 1498, 1499, 1502, 1520, 1527]
rank_y=[0.04078674832024518, 0.030941167885670022, 0.026470802858249463, 0.0233296473216541, 0.021430696431678744, 0.019728208093134476, 0.01828841825193235, 0.017452610425986944, 0.016507344892815645, 0.01568837671413576, 0.014900842543684852, 0.0146314081874365, 0.014064473396163929, 0.013550864154565509, 0.012968212359178449, 0.01275883941151046, 0.012423169109351056, 0.011974673170512654, 0.011646300048834978, 0.011230922082952103, 0.01097664340924272, 0.010842487552694064, 0.010649392930716078, 0.010249731968947691, 0.01005775999012074, 0.0099370758513845, 0.009607580086555786, 0.009458829869043676, 0.009124282210035308, 0.008965428204163883, 0.008834078955492812, 0.008667927769139662, 0.00855847006191377, 0.008396248126589241, 0.008287351740938866, 0.008107167515197782, 0.007865799237725301, 0.0077670066404342385, 0.007411690083131726, 0.00735443528242895, 0.007292689909122037, 0.007188284096075801, 0.007101840573446121, 0.006852052472340879, 0.006767292914437752, 0.006775151416494996, 0.006367070631093847, 0.006350792305403843, 0.006263787461198646, 0.006246947813933124, 0.006124579710470332, 0.0059825653518644295, 0.005825395310719558, 0.005744003682269535, 0.005625564829835364, 0.005487479722258084, 0.005552593025018103, 0.005412262631138753, 0.005255653911569399, 0.005189417965658346, 0.005173700961543859, 0.004983412947443461, 0.004898653389540333, 0.0048251202631475545, 0.004763374889840641, 0.004624728460687844, 0.004542214189086786, 0.004583190664099556, 0.004378869610611223, 0.004321053488332931, 0.004263798687630156, 0.004227312785221525, 0.004112803183815976, 0.004081369175587002, 0.004015133229675949, 0.003971911468361109, 0.0038905198399110867, 0.0037507507676072544, 
 0.0037440149087010458, 0.003676656319638958, 0.0034947881291713207, 0.003621646805238253, 0.0034785098034813165, 0.003546429714118922, 0.0033858917435209456, 0.003256226459576427, 0.003235457561282283, 0.0031585565054363996, 0.0030693063749291333, 0.003073235625957755, 0.003052466727663611, 0.0029974572132629063, 0.0029160655848128833, 0.0029205561574170225, 0.002788645587170434, 0.002722409641259381, 0.002702763386116272, 0.002726338892288003, 0.0026045321104007273, 0.002564678278538992, 0.002482164006937935, 0.002498442332627939, 0.0024782347559093127, 0.0024731828617296564, 0.0024355743161699906, 0.002376635550740664, 0.0022744750239964973, 0.0022576353767309754, 0.002218904188020275, 0.00219196075239544, 0.0022200268311713097, 0.002043210534883329, 0.0020926068335288604, 0.002030300138646429, 0.002033106746524016, 0.0020027953814460765, 0.001961257584857789, 0.0019230877177226061, 0.001813068688921196, 0.0018433800539991356, 0.0017900545043249827, 0.0018433800539991356, 0.001816997939949818, 0.001750761994038765, 0.001677790189221503, 0.0016884552991563338, 0.0016446722162659767, 0.0016048183844042413, 0.001578997591930441, 0.001531285258011462, 0.0015453182973993972, 0.0014790823514883441, 0.0014414738059286786, 0.0014291247312672958, 0.001442035127504196, 0.001408355832973152, 0.0013920775072831474, 0.001411162440850739, 0.0013320160987027858, 0.0013033886983513986, 0.001270270725395872, 0.001278690549028633, 0.0012708320469713894, 0.0011978602421541277, 0.0011866338106437796, 0.0011905630616724015, 0.0011821432380396406, 0.0011709168065292925, 0.0010738081739647828, 0.001094015750683409, 0.0011181525784306572, 0.001079982711295474, 0.001088402534928235, 0.0010322703773764951, 0.0010064495849026948, 0.0010266571616213212, 0.0009772608629757902, 0.0009834354003064815, 0.0009738929335226858, 0.0009497561057754377, 0.0009200060622730155, 0.0009256192780281896, 0.0009099022739137025, 0.0008964305561012848, 0.0008941852697992152, 0.0008852041245909369, 0.0008431050064271321, 0.0008846428030154195, 0.0008425436848516146, 0.0008257040375860927, 0.0008094257118960881, 0.0007869728488753922, 0.0007965153156591879, 0.000755538840646418, 0.0007841662409978052, 0.0007577841269484875, 0.0007617133779771093, 0.0007819209546957356, 0.0007431897659850352, 0.0007140010440581304, 0.0006921095026129519, 0.0007005293262457129, 0.0007095104714539913, 0.0007128784009070956, 0.0006483264197225949, 0.0006415905608163861, 0.0006713406043188082, 0.0006314867724570729, 0.0006359773450612122, 0.000628680164579486, 0.0006376613097877643, 0.0006314867724570729, 0.000625873556701899, 0.0005888263327177507, 0.0006101565525874119, 0.0005669347912725722, 0.0005753546149053331, 0.0005764772580563679, 0.0005450432498273936, 0.0005613215755173982, 0.000531571532014976, 0.0005444819282518762, 0.0005248356731087673, 0.0005197837789291107, 0.0005534630734601545, 0.0005152932063249715, 0.0005332554967415283, 0.0004810525902184102, 0.0005108026337208323, 0.000499014880634967, 0.0004513025467159881, 0.0004658969076794405, 0.0004698261587080623, 0.0004619676566508187, 0.0004698261587080623, 0.00044737329568736634, 0.00043165629157287916, 0.0004614063350753013, 0.0004541091545935751, 0.00044737329568736634, 0.0004282883621197748, 0.000429972326846327, 0.0003985383186173527, 0.0003985383186173527, 0.00043165629157287916, 0.0004339015778749488, 0.0004305336484218444, 0.0003614910946332044, 0.00039573171073976573, 0.00038394395765390033, 0.0003873118871070047, 0.0003772080987476916, 0.0003614910946332044, 0.000366542988812861, 0.00036934959669044796, 0.00035531655730251305, 0.00034970334154733906, 0.0003468967336697521, 0.0003620524162087218, 0.0003418448394900955, 0.0003519486278494086, 0.0003395995531880259, 0.00031658536859181254, 0.00032500519222457356, 0.00031265611756319076, 0.0003205146196204344, 0.00032163726277146913, 0.000314340082289743, 0.00031995329804491697, 0.000314340082289743, 0.00030760422338353417, 0.00028851928981594267, 0.0002941325055711166, 0.0002986230781752558, 0.00025764660316248577, 0.0002952551487221514, 0.0002756088935790425, 0.0002778541798811121, 0.0002677503915217989, 0.0002986230781752558, 0.0002800994661831817, 0.0002800994661831817, 0.0002800994661831817, 0.0002638211404931771, 0.00025483999528489874, 0.00026606642679524674, 0.0002778541798811121, 0.00025708528158696837, 0.0002402456343214464, 0.00022340598705592447, 0.00025034942268075956, 0.0002441748853500682, 0.0002486654579542074, 0.00025540131686041615, 0.00024585885007662037, 0.00023631638329282462, 0.0002408069558969638, 0.0002436135637745508, 0.00022845788123558106, 0.0002093729476679895, 0.00022845788123558106, 0.00022340598705592447, 0.00020375973191281554, 0.00022396730863144187, 0.00022733523808454625, 0.00022733523808454625, 0.00020881162609247213, 0.00022733523808454625, 0.00018916537094936317, 0.00019926915930867635, 0.00019421726512901976, 0.00021330219869661131, 0.0002194767360273027, 0.00021330219869661131, 0.00019141065725143276, 0.00019870783773315895, 0.00019197197882695017, 0.00020207576718626335, 0.00017120308053280645, 0.0001813068688921196, 0.00020488237506385032, 0.00017400968841039342, 0.00020263708876178073, 0.00019646255143108935, 0.00019477858670453716, 0.0001829908336186718, 0.0001857974414962588, 0.0001577313627203889, 0.00017008043738177164, 0.00017120308053280645, 0.00016166061374901067, 0.00014425964490797133, 0.00016895779423073686, 0.00017008043738177164, 0.0001498728606631453, 0.0001841134767697066, 0.00016615118635314986, 0.00017120308053280645, 0.0001829908336186718, 0.00016446722162659767, 0.0001554860764183193, 0.00015660871956935408, 0.0001509955038141801, 0.0001532407901162497, 0.00015604739799383668, 0.00016334457847556286, 0.00015660871956935408, 0.00016222193532452808, 0.00014145303703038434, 0.00014313700175693652, 0.00014538228805900612, 0.00014369832333245393, 0.0001515568253896975, 0.00013471717812417556, 0.00013583982127521037, 0.00013583982127521037, 0.00013696246442624515, 0.00014425964490797133, 0.00014145303703038434, 0.000124052068189345, 0.00013640114285072774, 0.00011282563667899703, 0.00013976907230383215, 0.00014594360963452352, 0.00014033039387934955, 0.00013134924867107118, 0.0001234907466138276, 0.00013134924867107118, 0.00013920775072831474, 0.00013022660552003637, 0.00012854264079348418, 0.0001223681034627928, 0.0001223681034627928, 0.00013640114285072774, 0.00013078792709555378, 0.00011226431510347963, 0.00012798131921796678, 0.00011450960140554922, 0.00011956149558520581, 0.00010945770722589264, 0.00013134924867107118, 0.00011338695825451442, 0.00011731620928313622, 0.00011787753085865361, 0.00010889638565037525, 0.00010665109934830566, 0.00011114167195244483, 0.0001223681034627928, 0.00011563224455658402, 0.0001190001740096884, 0.00011114167195244483, 0.00010608977777278825, 0.00010496713462175346, 0.00010496713462175346, 9.26180599603707e-05, 0.00010159920516864907, 9.823127571554468e-05, 0.00011507092298106662, 9.654731098899249e-05, 0.00010777374249934045, 0.00011731620928313622, 9.09340952338185e-05, 0.00010721242092382305, 0.00011394827983003183, 0.00010272184831968386, 9.486334626244029e-05, 0.00010103788359313167, 9.43020246869229e-05, 0.00011058035037692744, 0.00010496713462175346, 9.766995414002728e-05, 0.00010945770722589264, 9.43020246869229e-05, 8.92501305072663e-05, 0.00010159920516864907, 0.00011450960140554922, 8.868880893174891e-05, 8.532087947864452e-05, 9.26180599603707e-05, 9.26180599603707e-05, 7.970766372347054e-05, 8.92501305072663e-05, 9.710863256450988e-05, 7.858502057243574e-05, 7.914634214795314e-05, 9.14954168093359e-05, 8.700484420519671e-05, 8.363691475209232e-05, 9.879259729106208e-05, 7.633973427036615e-05, 8.812748735623151e-05, 8.139162845002273e-05, 8.700484420519671e-05, 8.195295002554013e-05, 8.251427160105752e-05, 7.970766372347054e-05, 7.858502057243574e-05, 9.20567383848533e-05, 0.00010103788359313167, 8.139162845002273e-05, 7.184916166622696e-05, 8.251427160105752e-05, 8.756616578071412e-05, 7.690105584588354e-05, 6.791991063760518e-05, 6.399065960898339e-05, 8.139162845002273e-05, 6.511330276001819e-05, 7.521709111933135e-05, 6.904255378863998e-05, 7.128784009070957e-05, 7.633973427036615e-05, 7.914634214795314e-05, 7.241048324174437e-05, 7.016519693967478e-05, 8.700484420519671e-05, 7.802369899691834e-05, 7.072651851519217e-05, 7.297180481726176e-05, 7.072651851519217e-05, 5.8938765429326804e-05, 7.521709111933135e-05, 6.679726748657039e-05, 6.679726748657039e-05, 6.960387536415737e-05, 6.17453733069138e-05, 6.00614085803616e-05, 6.567462433553559e-05, 6.399065960898339e-05, 6.735858906208778e-05, 5.95000870048442e-05, 6.511330276001819e-05, 6.00614085803616e-05, 4.658969076794405e-05, 5.4448192825187625e-05, 4.995762022104844e-05, 6.23066948824312e-05, 6.735858906208778e-05, 7.297180481726176e-05, 7.633973427036615e-05, 7.072651851519217e-05, 5.669347912725721e-05, 5.4448192825187625e-05, 5.725480070277461e-05, 5.5570835976222415e-05, 7.072651851519217e-05, 5.8377443853809406e-05, 5.725480070277461e-05, 5.220290652311803e-05, 6.00614085803616e-05, 6.511330276001819e-05, 6.0622730155879e-05, 6.45519811845008e-05, 6.511330276001819e-05, 4.4905726041391854e-05, 4.995762022104844e-05, 5.613215755173981e-05, 5.1641584947600634e-05, 5.500951440070502e-05, 4.4905726041391854e-05, 5.4448192825187625e-05, 4.715101234346145e-05, 5.725480070277461e-05, 5.500951440070502e-05, 6.45519811845008e-05, 4.939629864553104e-05, 4.4905726041391854e-05, 3.648590240863088e-05, 4.378308289035706e-05, 5.4448192825187625e-05, 4.4905726041391854e-05, 4.209911816380486e-05, 5.5570835976222415e-05, 4.715101234346145e-05, 5.1641584947600634e-05, 3.985383186173527e-05, 4.883497707001364e-05, 5.1641584947600634e-05, 4.322176131483966e-05, 4.4905726041391854e-05, 4.7712333918978846e-05, 5.388687124967023e-05, 4.378308289035706e-05, 5.220290652311803e-05, 4.322176131483966e-05, 4.546704761690925e-05, 4.209911816380486e-05, 3.648590240863088e-05, 4.097647501277007e-05, 4.8273655494496244e-05, 4.8273655494496244e-05, 4.8273655494496244e-05, 3.592458083311348e-05, 4.715101234346145e-05, 4.602836919242665e-05, 4.883497707001364e-05, 4.322176131483966e-05, 4.995762022104844e-05, 3.648590240863088e-05, 4.378308289035706e-05, 3.424061610656129e-05, 4.8273655494496244e-05, 3.648590240863088e-05, 4.378308289035706e-05, 4.209911816380486e-05, 4.4905726041391854e-05, 3.704722398414828e-05, 3.704722398414828e-05, 3.255665138000909e-05, 4.322176131483966e-05, 3.5363259257596084e-05, 4.322176131483966e-05, 4.4905726041391854e-05, 3.704722398414828e-05, 2.8066078775869907e-05, 3.929251028621787e-05, 3.704722398414828e-05, 4.4905726041391854e-05, 3.5363259257596084e-05, 4.097647501277007e-05, 4.378308289035706e-05, 4.266043973932226e-05, 3.760854555966568e-05, 4.097647501277007e-05, 3.1995329804491694e-05, 3.424061610656129e-05, 3.08726866534569e-05, 3.648590240863088e-05, 2.6382114049317715e-05, 3.648590240863088e-05, 3.367929453104389e-05, 3.424061610656129e-05, 3.5363259257596084e-05, 2.750475720035251e-05, 3.4801937682078686e-05, 2.6943435624835113e-05, 3.1995329804491694e-05, 3.985383186173527e-05, 3.311797295552649e-05, 3.1434008228974296e-05, 3.03113650779395e-05, 4.1537796588287465e-05, 4.041515343725267e-05, 3.08726866534569e-05, 2.97500435024221e-05, 2.8627400351387305e-05, 3.592458083311348e-05, 2.97500435024221e-05, 2.9188721926904703e-05, 2.6943435624835113e-05, 2.189154144517853e-05, 3.311797295552649e-05, 2.2452863020695927e-05, 3.424061610656129e-05, 2.97500435024221e-05, 2.97500435024221e-05, 3.255665138000909e-05, 2.6382114049317715e-05, 3.03113650779395e-05, 3.311797295552649e-05, 3.311797295552649e-05, 2.8627400351387305e-05, 3.4801937682078686e-05, 2.6382114049317715e-05, 2.97500435024221e-05, 2.97500435024221e-05, 3.255665138000909e-05, 2.6382114049317715e-05, 2.6382114049317715e-05, 2.2452863020695927e-05, 3.03113650779395e-05, 2.133021986966113e-05, 3.03113650779395e-05, 2.97500435024221e-05, 2.4136827747248122e-05, 2.3575506171730724e-05, 2.133021986966113e-05, 2.469814932276552e-05, 2.750475720035251e-05, 2.3014184596213326e-05, 2.3014184596213326e-05, 2.525947089828292e-05, 2.469814932276552e-05, 2.4136827747248122e-05, 2.3014184596213326e-05, 2.9188721926904703e-05, 2.525947089828292e-05, 2.4136827747248122e-05, 2.2452863020695927e-05, 2.189154144517853e-05, 2.189154144517853e-05, 2.469814932276552e-05, 2.6382114049317715e-05, 2.3575506171730724e-05, 2.189154144517853e-05, 2.469814932276552e-05, 2.4136827747248122e-05, 1.7400968841039343e-05, 2.133021986966113e-05, 1.6839647265521945e-05, 2.3575506171730724e-05, 2.4136827747248122e-05, 1.6839647265521945e-05, 1.796229041655674e-05, 2.133021986966113e-05, 2.5820792473800317e-05, 2.469814932276552e-05, 2.3575506171730724e-05, 2.4136827747248122e-05, 1.9646255143108936e-05, 1.9084933567591538e-05, 1.6278325690004546e-05, 2.133021986966113e-05, 1.9646255143108936e-05, 2.189154144517853e-05, 2.0768898294143732e-05, 2.5820792473800317e-05, 1.7400968841039343e-05, 1.6839647265521945e-05, 1.3471717812417557e-05, 1.852361199207414e-05, 1.6839647265521945e-05, 1.6278325690004546e-05, 1.9084933567591538e-05, 2.0768898294143732e-05, 1.9084933567591538e-05, 1.7400968841039343e-05, 1.852361199207414e-05, 1.515568253896975e-05, 2.2452863020695927e-05, 1.9646255143108936e-05, 1.6839647265521945e-05, 1.7400968841039343e-05, 2.4136827747248122e-05, 1.515568253896975e-05, 2.469814932276552e-05, 2.0768898294143732e-05, 1.852361199207414e-05, 2.0768898294143732e-05, 2.3014184596213326e-05, 2.525947089828292e-05, 1.6839647265521945e-05, 1.3471717812417557e-05, 1.9646255143108936e-05, 1.6839647265521945e-05, 1.7400968841039343e-05, 1.3471717812417557e-05, 2.0207576718626334e-05, 1.9646255143108936e-05, 1.796229041655674e-05, 1.3471717812417557e-05, 2.133021986966113e-05, 1.852361199207414e-05, 1.3471717812417557e-05, 1.9084933567591538e-05, 1.796229041655674e-05, 1.3471717812417557e-05, 2.0207576718626334e-05, 1.6278325690004546e-05, 1.9646255143108936e-05, 1.4033039387934953e-05, 1.9084933567591538e-05, 8.419823632760972e-06, 2.0768898294143732e-05, 1.4594360963452352e-05, 9.542466783795769e-06, 8.98114520827837e-06, 1.4594360963452352e-05, 1.4594360963452352e-05, 1.0665109934830565e-05, 1.852361199207414e-05, 1.1226431510347964e-05, 1.5717004114487148e-05, 1.4033039387934953e-05, 1.4033039387934953e-05, 1.5717004114487148e-05, 1.1787753085865362e-05, 1.2910396236900158e-05, 1.234907466138276e-05, 1.4594360963452352e-05, 1.515568253896975e-05, 1.796229041655674e-05, 1.515568253896975e-05, 1.852361199207414e-05, 1.515568253896975e-05, 1.3471717812417557e-05, 1.0103788359313167e-05, 1.4033039387934953e-05, 1.852361199207414e-05, 1.6278325690004546e-05, 1.6278325690004546e-05, 1.4594360963452352e-05, 1.0665109934830565e-05, 1.2910396236900158e-05, 9.542466783795769e-06, 1.3471717812417557e-05, 1.1226431510347964e-05, 1.3471717812417557e-05, 1.4594360963452352e-05, 1.0665109934830565e-05, 1.3471717812417557e-05, 1.4594360963452352e-05, 1.515568253896975e-05, 1.4594360963452352e-05, 1.9084933567591538e-05, 1.4033039387934953e-05, 8.98114520827837e-06, 7.858502057243574e-06, 1.4033039387934953e-05, 1.1226431510347964e-05, 1.5717004114487148e-05, 1.0665109934830565e-05, 8.98114520827837e-06, 8.98114520827837e-06, 1.1787753085865362e-05, 1.0665109934830565e-05, 1.1226431510347964e-05, 1.4033039387934953e-05, 8.419823632760972e-06, 9.542466783795769e-06, 7.858502057243574e-06, 1.515568253896975e-05, 9.542466783795769e-06, 8.98114520827837e-06, 1.1787753085865362e-05, 1.0665109934830565e-05, 1.2910396236900158e-05, 8.419823632760972e-06, 1.1226431510347964e-05, 9.542466783795769e-06, 8.98114520827837e-06, 1.2910396236900158e-05, 7.297180481726176e-06, 1.1787753085865362e-05, 1.1787753085865362e-05, 1.0665109934830565e-05, 7.858502057243574e-06, 1.0103788359313167e-05, 1.234907466138276e-05, 1.2910396236900158e-05, 7.858502057243574e-06, 9.542466783795769e-06, 8.419823632760972e-06, 1.4594360963452352e-05, 1.234907466138276e-05, 1.1787753085865362e-05, 1.0665109934830565e-05, 6.735858906208778e-06, 1.234907466138276e-05, 9.542466783795769e-06, 1.3471717812417557e-05, 1.1226431510347964e-05, 1.0103788359313167e-05, 7.297180481726176e-06, 8.98114520827837e-06, 5.613215755173982e-06, 1.0665109934830565e-05, 5.0518941796565835e-06, 5.0518941796565835e-06, 1.4033039387934953e-05, 8.98114520827837e-06, 1.0103788359313167e-05, 1.0103788359313167e-05, 7.858502057243574e-06, 1.0103788359313167e-05, 1.0103788359313167e-05, 6.735858906208778e-06, 1.234907466138276e-05, 8.98114520827837e-06, 1.1226431510347964e-05, 1.2910396236900158e-05, 5.613215755173982e-06, 6.17453733069138e-06, 7.297180481726176e-06, 1.3471717812417557e-05, 7.858502057243574e-06, 9.542466783795769e-06, 9.542466783795769e-06, 7.297180481726176e-06, 7.858502057243574e-06, 6.735858906208778e-06, 6.17453733069138e-06, 7.858502057243574e-06, 4.490572604139185e-06, 8.98114520827837e-06, 5.613215755173982e-06, 7.858502057243574e-06, 6.17453733069138e-06, 6.735858906208778e-06, 6.735858906208778e-06, 4.490572604139185e-06, 7.297180481726176e-06, 5.613215755173982e-06, 5.613215755173982e-06, 8.419823632760972e-06, 6.735858906208778e-06, 6.17453733069138e-06, 7.297180481726176e-06, 9.542466783795769e-06, 6.735858906208778e-06, 6.17453733069138e-06, 6.17453733069138e-06, 8.419823632760972e-06, 4.490572604139185e-06, 5.613215755173982e-06, 3.929251028621787e-06, 8.419823632760972e-06, 9.542466783795769e-06, 4.490572604139185e-06, 5.613215755173982e-06, 1.0665109934830565e-05, 3.367929453104389e-06, 5.613215755173982e-06, 8.98114520827837e-06, 9.542466783795769e-06, 5.0518941796565835e-06, 7.858502057243574e-06, 7.858502057243574e-06, 6.735858906208778e-06, 9.542466783795769e-06, 5.613215755173982e-06, 8.419823632760972e-06, 3.367929453104389e-06, 5.613215755173982e-06, 7.858502057243574e-06, 5.0518941796565835e-06, 4.490572604139185e-06, 3.367929453104389e-06, 9.542466783795769e-06, 5.613215755173982e-06, 5.0518941796565835e-06, 5.0518941796565835e-06, 3.929251028621787e-06, 3.929251028621787e-06, 6.17453733069138e-06, 7.297180481726176e-06, 3.367929453104389e-06, 7.297180481726176e-06, 5.613215755173982e-06, 6.17453733069138e-06, 7.297180481726176e-06, 3.367929453104389e-06, 3.929251028621787e-06, 4.490572604139185e-06, 6.17453733069138e-06, 7.297180481726176e-06, 3.929251028621787e-06, 1.6839647265521946e-06, 8.419823632760972e-06, 6.17453733069138e-06, 3.367929453104389e-06, 2.806607877586991e-06, 6.17453733069138e-06, 1.6839647265521946e-06, 5.0518941796565835e-06, 3.929251028621787e-06, 5.613215755173982e-06, 3.929251028621787e-06, 3.367929453104389e-06, 4.490572604139185e-06, 5.0518941796565835e-06, 4.490572604139185e-06, 3.929251028621787e-06, 5.613215755173982e-06, 3.929251028621787e-06, 3.367929453104389e-06, 3.367929453104389e-06, 4.490572604139185e-06, 5.613215755173982e-06, 6.17453733069138e-06, 3.367929453104389e-06, 5.0518941796565835e-06, 3.929251028621787e-06, 3.929251028621787e-06, 5.0518941796565835e-06, 5.613215755173982e-06, 3.929251028621787e-06, 2.806607877586991e-06, 3.929251028621787e-06, 2.806607877586991e-06, 5.0518941796565835e-06, 3.367929453104389e-06, 3.367929453104389e-06, 5.0518941796565835e-06, 7.297180481726176e-06, 4.490572604139185e-06, 2.806607877586991e-06, 2.806607877586991e-06, 2.2452863020695926e-06, 4.490572604139185e-06, 4.490572604139185e-06, 6.735858906208778e-06, 5.613215755173982e-06, 3.367929453104389e-06, 4.490572604139185e-06, 4.490572604139185e-06, 4.490572604139185e-06, 3.367929453104389e-06, 1.6839647265521946e-06, 8.419823632760972e-06, 6.17453733069138e-06, 2.806607877586991e-06, 4.490572604139185e-06, 5.613215755173982e-06, 3.929251028621787e-06, 4.490572604139185e-06, 6.17453733069138e-06, 5.0518941796565835e-06, 6.17453733069138e-06, 6.17453733069138e-06, 6.17453733069138e-06, 1.6839647265521946e-06, 3.929251028621787e-06, 2.806607877586991e-06, 4.490572604139185e-06, 1.6839647265521946e-06, 3.929251028621787e-06, 4.490572604139185e-06, 2.2452863020695926e-06, 4.490572604139185e-06, 6.17453733069138e-06, 2.806607877586991e-06, 4.490572604139185e-06, 2.806607877586991e-06, 1.6839647265521946e-06, 2.2452863020695926e-06, 4.490572604139185e-06, 5.0518941796565835e-06, 5.0518941796565835e-06, 1.6839647265521946e-06, 3.367929453104389e-06, 3.929251028621787e-06, 2.806607877586991e-06, 4.490572604139185e-06, 7.297180481726176e-06, 1.1226431510347963e-06, 3.929251028621787e-06, 2.806607877586991e-06, 3.929251028621787e-06, 1.1226431510347963e-06, 2.2452863020695926e-06, 3.929251028621787e-06, 3.367929453104389e-06, 3.929251028621787e-06, 7.297180481726176e-06, 2.806607877586991e-06, 3.367929453104389e-06, 1.6839647265521946e-06, 2.2452863020695926e-06, 2.2452863020695926e-06, 3.929251028621787e-06, 3.367929453104389e-06, 4.490572604139185e-06, 1.6839647265521946e-06, 2.806607877586991e-06, 3.367929453104389e-06, 2.2452863020695926e-06, 5.0518941796565835e-06, 1.6839647265521946e-06, 3.367929453104389e-06, 2.2452863020695926e-06, 1.1226431510347963e-06, 1.6839647265521946e-06, 2.806607877586991e-06, 2.806607877586991e-06, 1.6839647265521946e-06, 3.367929453104389e-06, 2.806607877586991e-06, 1.6839647265521946e-06, 3.367929453104389e-06, 5.0518941796565835e-06, 3.367929453104389e-06, 1.6839647265521946e-06, 2.2452863020695926e-06, 3.929251028621787e-06, 2.806607877586991e-06, 2.806607877586991e-06, 3.929251028621787e-06, 4.490572604139185e-06, 3.367929453104389e-06, 1.6839647265521946e-06, 3.929251028621787e-06, 1.6839647265521946e-06, 5.613215755173982e-07, 1.6839647265521946e-06, 2.806607877586991e-06, 2.2452863020695926e-06, 2.806607877586991e-06, 2.806607877586991e-06, 1.1226431510347963e-06, 3.929251028621787e-06, 2.2452863020695926e-06, 5.0518941796565835e-06, 1.6839647265521946e-06, 5.613215755173982e-06, 2.806607877586991e-06, 2.806607877586991e-06, 2.2452863020695926e-06, 3.929251028621787e-06, 1.6839647265521946e-06, 5.613215755173982e-07, 2.2452863020695926e-06, 5.613215755173982e-07, 2.806607877586991e-06, 5.0518941796565835e-06, 1.6839647265521946e-06, 2.806607877586991e-06, 3.367929453104389e-06, 1.6839647265521946e-06, 5.613215755173982e-07, 1.6839647265521946e-06, 3.367929453104389e-06, 2.806607877586991e-06, 3.367929453104389e-06, 1.6839647265521946e-06, 2.2452863020695926e-06, 1.6839647265521946e-06, 2.806607877586991e-06, 2.2452863020695926e-06, 3.929251028621787e-06, 3.929251028621787e-06, 3.367929453104389e-06, 1.6839647265521946e-06, 2.806607877586991e-06, 2.806607877586991e-06, 2.2452863020695926e-06, 5.613215755173982e-07, 2.2452863020695926e-06, 3.367929453104389e-06, 2.2452863020695926e-06, 2.2452863020695926e-06, 2.806607877586991e-06, 3.367929453104389e-06, 5.613215755173982e-07, 1.6839647265521946e-06, 1.6839647265521946e-06, 5.613215755173982e-07, 2.806607877586991e-06, 1.6839647265521946e-06, 1.6839647265521946e-06, 2.2452863020695926e-06, 2.2452863020695926e-06, 2.806607877586991e-06, 1.1226431510347963e-06, 1.6839647265521946e-06, 3.929251028621787e-06, 1.6839647265521946e-06, 2.2452863020695926e-06, 2.806607877586991e-06, 1.6839647265521946e-06, 5.613215755173982e-07, 1.1226431510347963e-06, 1.6839647265521946e-06, 1.6839647265521946e-06, 2.2452863020695926e-06, 1.6839647265521946e-06, 1.1226431510347963e-06, 5.613215755173982e-07, 1.6839647265521946e-06, 1.6839647265521946e-06, 1.6839647265521946e-06, 2.2452863020695926e-06, 1.6839647265521946e-06, 2.806607877586991e-06, 2.2452863020695926e-06, 2.2452863020695926e-06, 1.1226431510347963e-06, 5.613215755173982e-07, 5.613215755173982e-07, 1.1226431510347963e-06, 5.613215755173982e-07, 2.2452863020695926e-06, 2.2452863020695926e-06, 5.613215755173982e-07, 2.2452863020695926e-06, 2.806607877586991e-06, 1.1226431510347963e-06, 1.1226431510347963e-06, 2.806607877586991e-06, 1.6839647265521946e-06, 1.1226431510347963e-06, 1.1226431510347963e-06, 2.2452863020695926e-06, 1.1226431510347963e-06, 1.6839647265521946e-06, 3.367929453104389e-06, 2.2452863020695926e-06, 2.2452863020695926e-06, 1.6839647265521946e-06, 2.2452863020695926e-06, 1.6839647265521946e-06, 1.1226431510347963e-06, 2.2452863020695926e-06, 1.6839647265521946e-06, 1.6839647265521946e-06, 1.6839647265521946e-06, 1.6839647265521946e-06, 1.1226431510347963e-06, 1.6839647265521946e-06, 2.2452863020695926e-06, 1.6839647265521946e-06, 5.613215755173982e-07, 1.1226431510347963e-06, 1.6839647265521946e-06, 1.1226431510347963e-06, 2.2452863020695926e-06, 1.6839647265521946e-06, 1.6839647265521946e-06, 2.806607877586991e-06, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 2.806607877586991e-06, 1.6839647265521946e-06, 1.1226431510347963e-06, 1.1226431510347963e-06, 5.613215755173982e-07, 1.1226431510347963e-06, 3.367929453104389e-06, 1.6839647265521946e-06, 3.367929453104389e-06, 5.613215755173982e-07, 1.1226431510347963e-06, 1.1226431510347963e-06, 5.613215755173982e-07, 1.1226431510347963e-06, 2.2452863020695926e-06, 1.6839647265521946e-06, 1.1226431510347963e-06, 5.613215755173982e-07, 1.6839647265521946e-06, 1.1226431510347963e-06, 3.367929453104389e-06, 1.1226431510347963e-06, 5.613215755173982e-07, 1.6839647265521946e-06, 1.6839647265521946e-06, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 2.2452863020695926e-06, 2.2452863020695926e-06, 3.929251028621787e-06, 5.613215755173982e-07, 1.1226431510347963e-06, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 2.806607877586991e-06, 2.2452863020695926e-06, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 1.1226431510347963e-06, 1.1226431510347963e-06, 5.613215755173982e-07, 1.1226431510347963e-06, 1.6839647265521946e-06, 5.613215755173982e-07, 1.1226431510347963e-06, 1.1226431510347963e-06, 1.1226431510347963e-06, 1.6839647265521946e-06, 1.6839647265521946e-06, 2.2452863020695926e-06, 5.613215755173982e-07, 5.613215755173982e-07, 1.1226431510347963e-06, 1.1226431510347963e-06, 1.1226431510347963e-06, 5.613215755173982e-07, 1.1226431510347963e-06, 1.6839647265521946e-06, 1.1226431510347963e-06, 1.1226431510347963e-06, 5.613215755173982e-07, 1.1226431510347963e-06, 1.1226431510347963e-06, 1.1226431510347963e-06, 1.1226431510347963e-06, 1.1226431510347963e-06, 1.1226431510347963e-06, 1.6839647265521946e-06, 1.1226431510347963e-06, 1.1226431510347963e-06, 1.1226431510347963e-06, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 1.1226431510347963e-06, 1.1226431510347963e-06, 1.1226431510347963e-06, 5.613215755173982e-07, 1.1226431510347963e-06, 2.2452863020695926e-06, 5.613215755173982e-07, 1.6839647265521946e-06, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 1.6839647265521946e-06, 1.1226431510347963e-06, 2.806607877586991e-06, 2.2452863020695926e-06, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 1.1226431510347963e-06, 1.6839647265521946e-06, 5.613215755173982e-07, 5.613215755173982e-07, 1.1226431510347963e-06, 1.6839647265521946e-06, 5.613215755173982e-07, 5.613215755173982e-07, 1.1226431510347963e-06, 1.1226431510347963e-06, 1.1226431510347963e-06, 1.1226431510347963e-06, 5.613215755173982e-07, 1.1226431510347963e-06, 5.613215755173982e-07, 5.613215755173982e-07, 1.1226431510347963e-06, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 1.6839647265521946e-06, 5.613215755173982e-07, 1.6839647265521946e-06, 5.613215755173982e-07, 1.1226431510347963e-06, 5.613215755173982e-07, 1.1226431510347963e-06, 1.6839647265521946e-06, 5.613215755173982e-07, 1.6839647265521946e-06, 5.613215755173982e-07, 5.613215755173982e-07, 1.6839647265521946e-06, 5.613215755173982e-07, 5.613215755173982e-07, 1.6839647265521946e-06, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 1.1226431510347963e-06, 1.1226431510347963e-06, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 1.1226431510347963e-06, 2.2452863020695926e-06, 1.1226431510347963e-06, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 1.6839647265521946e-06, 5.613215755173982e-07, 1.1226431510347963e-06, 5.613215755173982e-07, 5.613215755173982e-07, 1.1226431510347963e-06, 1.6839647265521946e-06, 5.613215755173982e-07, 1.1226431510347963e-06, 1.1226431510347963e-06, 5.613215755173982e-07, 1.1226431510347963e-06, 5.613215755173982e-07, 5.613215755173982e-07, 1.1226431510347963e-06, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 1.6839647265521946e-06, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 1.1226431510347963e-06, 5.613215755173982e-07, 5.613215755173982e-07, 1.1226431510347963e-06, 5.613215755173982e-07, 5.613215755173982e-07, 1.1226431510347963e-06, 5.613215755173982e-07, 1.1226431510347963e-06, 1.1226431510347963e-06, 1.1226431510347963e-06, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 1.1226431510347963e-06, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 1.1226431510347963e-06, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 1.1226431510347963e-06, 5.613215755173982e-07, 1.1226431510347963e-06, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07, 5.613215755173982e-07]

r_popt, r_pcov = curve_fit(func7, rank_x, rank_y, maxfev=10000)
rank_y_fit = []
for _x in rank_x:
    if _x not in PROBABILITY_MAP:
        prob = func7(_x, *r_popt) 
        PROBABILITY_MAP[_x] = prob
        rank_y_fit.append(prob)
# print(PROBABILITY_MAP)
plt.plot(rank_x, rank_y_fit,c='red')
plt.scatter(rank_x, rank_y)
plt.xscale('log')
plt.yscale('log')
# logger.info("rank_list"+'*'*50)
# logger.info(rank_x)
# logger.info(rank_y)
# logger.info("*"*50)


# In[ ]:


plt.plot(rank_x, rank_y_fit,c='red')
plt.scatter(rank_x, rank_y)
plt.xscale('log')
plt.yscale('log')


# In[ ]:


# PROBABILITY_MAP = dict()
# result = Counter(bike0.rank_list)
# x = list(result.keys())
# print(bike0.rank_list)


# In[ ]:


# print(PROBABILITY_MAPBABILITY_MAP)


# ### 真实冗余率

# In[ ]:


redundancy = bike0.cal_redundancy(24*DAY)
t = [_t for _t in range(DAY*24)]
plt.plot(t, redundancy)


# In[ ]:


# demand = [len(bike0.trip_dict[t]) for t in range(DAY * 24)]
# t = [_t for _t in range(DAY * 24)]
# max_demand = max(demand)
# plt.plot(t, demand)
# r0 = float(len(bike0.bike_dict.keys()) / max_demand)
# r1 = 5
# r2 = 4
# r3 = 3
# r4 = 2
# r5 = 1
# ratio1 = 1/(len(bike0.bike_dict.keys()) / (r1 * max_demand))
# ratio2 = 1/(len(bike0.bike_dict.keys()) / (r2 * max_demand))
# ratio3 = 1/(len(bike0.bike_dict.keys()) / (r3 * max_demand))
# ratio4 = 1/(len(bike0.bike_dict.keys()) / (r4 * max_demand))
# ratio5 = 1/(len(bike0.bike_dict.keys()) / (r5 * max_demand))
# print("真实车辆数："+len(bike0.bike_dict.keys))
# print(ratio1, ratio2, ratio3,ratio4,ratio5)
# print(len(bike0.bike_dict.keys()) / max_demand)
demand = [len(bike0.trip_dict[t]) for t in range(DAY * 24)]
t = [_t for _t in range(DAY * 24)]
max_demand = max(demand)
plt.plot(t, demand)
r0 = float(len(bike0.bike_dict.keys()) / max_demand)
r1 = 1
r15 = 1.5
r2 = 2
r25 = 2.5
r3 = 3
r35 = 3.5
r4 = 4
r5 = 5
r6 = 6
r7 = 7
r8= 8
# ratio1 = 1/(len(bike0.bike_dict.keys()) / (r1 * max_demand))
# ratio2 = 1/(len(bike0.bike_dict.keys()) / (r2 * max_demand))
# ratio3 = 1/(len(bike0.bike_dict.keys()) / (r3 * max_demand))
# ratio4 = 1/(len(bike0.bike_dict.keys()) / (r4 * max_demand))
ratio1 = (r1 * max_demand)/ len(bike0.bike_dict.keys())
ratio15 = 1.5 *max_demand / len(bike0.bike_dict.keys())
ratio2 = (r2 * max_demand)/ len(bike0.bike_dict.keys())
ratio25 = 2.5 *max_demand / len(bike0.bike_dict.keys())
ratio3 = (r3 * max_demand)/ len(bike0.bike_dict.keys())
ratio35 = 3.5 *max_demand / len(bike0.bike_dict.keys())
ratio4 = (r4 * max_demand)/ len(bike0.bike_dict.keys())
ratio5 = (r5 * max_demand)/ len(bike0.bike_dict.keys())
ratio6 = (r6 * max_demand) / len(bike0.bike_dict.keys())
ratio7 = (r7 * max_demand) / len(bike0.bike_dict.keys())
ratio8 = (r8 * max_demand) / len(bike0.bike_dict.keys())
print("r6:"+str(r6*max_demand))
ratio8 = (r8 * max_demand)/len(bike0.bike_dict.keys())
print("真实车辆数："+str(len(bike0.bike_dict.keys())))

print(ratio1,ratio15, ratio2,ratio25, ratio3,ratio35,ratio4,ratio5,ratio6,ratio8)
print(len(bike0.bike_dict.keys()) / max_demand)


# In[ ]:


demand = [len(bike0.trip_dict[t]) for t in range(DAY * 24)]
t = [_t for _t in range(DAY * 24)]
max_demand = max(demand)
plt.plot(t, demand)
r0 = float(len(bike0.bike_dict.keys()) / max_demand)
r1 = 1
r2 = 2
r3 = 3
r4 = 4
r5 = 5
r6 = 6
r8= 8
# ratio1 = 1/(len(bike0.bike_dict.keys()) / (r1 * max_demand))
# ratio2 = 1/(len(bike0.bike_dict.keys()) / (r2 * max_demand))
# ratio3 = 1/(len(bike0.bike_dict.keys()) / (r3 * max_demand))
# ratio4 = 1/(len(bike0.bike_dict.keys()) / (r4 * max_demand))
ratio1 = (r1 * max_demand)/ len(bike0.bike_dict.keys())
ratio2 = (r2 * max_demand)/ len(bike0.bike_dict.keys())
ratio3 = (r3 * max_demand)/ len(bike0.bike_dict.keys())
ratio4 = (r4 * max_demand)/ len(bike0.bike_dict.keys())
ratio5 = (r5 * max_demand)/ len(bike0.bike_dict.keys())
ratio6 = (r6 * max_demand) / len(bike0.bike_dict.keys())
ratio8 = (r8 * max_demand) / len(bike0.bike_dict.keys())
print("r6:"+str(r6*max_demand))
ratio8 = (r8 * max_demand)/len(bike0.bike_dict.keys())
print("真实车辆数："+str(len(bike0.bike_dict.keys())))

print(ratio1, ratio2, ratio3,ratio4,ratio5,ratio6,ratio8)
print(len(bike0.bike_dict.keys()) / max_demand)


# In[ ]:


bike_list00 = {}
for key in bike0.bike_dict.keys():
#     if len(bike0.bike_dict[key]) <=5:
        
#         print(key,bike0.bike_dict[key])
    times = len(bike0.bike_dict[key])
    if times not in bike_list00:
        bike_list00[times] = 0
    bike_list00[times] +=1
# print(bike_list00)
# result00 = Counter(list(bike_list00.values()))
# result32 = Counter(list(bike_list22.values()))
# result33 = Counter(list(bike_list23.values()))
x00 = list(bike_list00.keys())
# x32 = list(result32.keys())
# x33 = list(result33.keys())
x00.sort()
y_sum = sum(bike_list00.values())
y00 = [bike_list00[_x] /y_sum  for _x in x00]
print(x00)
print(y00)


# In[ ]:


plt.scatter(x00,y00)
plt.scatter(x11,y11)


# ## bike1-无调度

# In[ ]:


bike1 = Bike()
# 设置时间格式
start_time = '2020-09-14 00:00:00.0'
bike1.set_time_format("%Y-%m-%d %H:%M:%S.%f")
bike1.set_start_time(start_time)
bike1.set_day(14, 16)
# 测试读数据
bike1.read_data('./jinhua-v1_14-20.csv')
bike1.init_position()


# ### 车辆数不变

# In[ ]:


# bike1.reduce(1) 
rank_list11 = []
bike_list11 = {}
time_list11 = []
err_list11 = []
grid_keys = bike2.grid_dict.keys()
for t in range(DAY*24):
    print(t)
#     bn = 0
#     for g in grid_keys:
#         bn+=len(bike1.grid_dict[g][t])
#     print("调度前的车辆数："+str(bn))
    bike1.simulation(t, rank_list11, bike_list11, err_list11)
#     bn = 0
#     for g in grid_keys:
#         bn+=len(bike1.grid_dict[g][t])
#     print("调度后的车辆数："+str(bn))


# In[ ]:


times_values = bike_list11.values()
# print(times_values)
times_count = Counter(times_values)
# print(times_count)
x11 = list(times_count.keys())
# print(x11)
x11.sort()
y11 = []
for _x in x11:
    y11.append(times_count[_x] / len(times_values))
# print(bike_list11)
# print(y11)
plt.scatter(x11,y11)
# plt.scatter(x00,y00)


# ### 冗余率为5

# In[ ]:


bike1.reduce(ratio5)
rank_list12 = []
bike_list12 = {}
time_list12 = []
err_list12 = []
for t in range(DAY*24):
    print(t)
    bike1.simulation(t, rank_list12, bike_list12, err_list12)


# ### 冗余率为4

# In[ ]:


bike1.reduce(ratio4)
rank_list13 = []
bike_list13 = {}
time_list13 = []
err_list13 = []
for t in range(DAY*24):
    print(t)
    bike1.simulation(t, rank_list13, bike_list13, err_list13)


# ### 冗余率为3(真实）

# In[ ]:


bike1.reduce(ratio3)
rank_list14 = []
bike_list14 = {}
time_list14 = []
err_list14 = []
for t in range(DAY*24):
    print(t)
    bike1.simulation(t, rank_list14, bike_list14, err_list14)


# ### 冗余率为2

# In[ ]:


bike1.reduce(ratio2)
rank_list15 = []
bike_list15 = {}
time_list15 = []
err_list15 = []
for t in range(DAY*24):
    print(t)
    bike1.simulation(t, rank_list15, bike_list15, err_list15)


# ### 冗余率为1

# In[ ]:


bike1.reduce(ratio1)
rank_list16 = []
bike_list16 = {}
time_list16 = []
err_list16 = []
for t in range(DAY*24):
    print(t)
    bike1.simulation(t, rank_list16, bike_list16, err_list16)


# ### 比较

# In[ ]:


fig11, axes11 = plt.subplots(1, 3, figsize=(20, 6))
# print(err_list11,err_list12,err_list13)
t = [_t for _t in range(DAY*24)]
axes11[0].plot(t, err_list11, label=f'redundancy={r0}(r0)')
axes11[0].plot(t, err_list12, label='redundancy=5')
axes11[0].plot(t, err_list13, label='redundancy=4')
axes11[0].set_xlabel('Time(h)')
axes11[0].set_ylabel('ErrorCount')
result21 = Counter(rank_list11)
result22 = Counter(rank_list12)
result23 = Counter(rank_list13)
x21 = list(result21.keys())
x22 = list(result22.keys())
x23 = list(result23.keys())
x21.sort()
y21 = [result21[_x] / len(rank_list11) for _x in x21]
x22.sort()
y22 = [result22[_x] / len(rank_list12) for _x in x22]
x23.sort()
y23 = [result23[_x] / len(rank_list13) for _x in x23]
axes11[1].scatter(x21, y21, label=f'redundancy={r0}(r0)')
axes11[1].scatter(x22, y22, label='redundancy=5')
axes11[1].scatter(x23, y23, label='redundancy=4')
axes11[1].set_xscale('log')
axes11[1].set_yscale('log')
axes11[1].set_xlabel('log' + r'$_{10}$($rank$)')
axes[1].set_ylabel('log' + r'$_{10}P$($rank$)')
result31 = Counter(list(bike_list11.values()))
result32 = Counter(list(bike_list12.values()))
result33 = Counter(list(bike_list13.values()))
x31 = list(result31.keys())
x32 = list(result32.keys())
x33 = list(result33.keys())
x31.sort()
y31 = [result31[_x] / len(bike_list11.values()) for _x in x31]
x32.sort()
y32 = [result32[_x] / len(bike_list12.values()) for _x in x32]
x33.sort()
y33 = [result33[_x] / len(bike_list13.values()) for _x in x33]
axes11[2].scatter(x31, y31, label=f'redundancy={r0}(r0)')
axes11[2].scatter(x32, y32, label='redundancy=5')
axes11[2].scatter(x33, y33, label='redundancy=4')
# axes[2].set_xscale('log')
# axes[2].set_yscale('log')
axes11[2].set_xlabel('log' + r'$_{10}$($times$)')
axes11[2].set_ylabel('log' + r'$_{10}P$($times$)')
# print(x31)
# result31 = Counter()
# axes[1].plot(t, rank_list11)x31 = list(bike)
# axes[1].plot(t,rank_list12)
# axes[1].plot(t,rank_list13)
# print(rank_list11)
# print(rank_list12)
# print(bike_list11)


fig12, axes12 = plt.subplots(1, 3, figsize=(20, 6))
t = [_t for _t in range(DAY*24)]
axes12[0].plot(t, err_list14, label=f'redundancy=3')
axes12[0].plot(t, err_list15, label='redundancy=2')
axes12[0].plot(t, err_list16, label='redundancy=1')
axes12[0].set_xlabel('Time(h)')
axes12[0].set_ylabel('ErrorCount')

result24 = Counter(rank_list14)
result25 = Counter(rank_list15)
result26 = Counter(rank_list16)
x24 = list(result24.keys())
x25 = list(result25.keys())
x26 = list(result26.keys())
x24.sort()
y24 = [result24[_x] / len(rank_list14) for _x in x24]
x25.sort()
y25 = [result25[_x] / len(rank_list15) for _x in x25]
x26.sort()
y26 = [result26[_x] / len(rank_list16) for _x in x26]
axes12[1].scatter(x24, y24, label=f'redundancy=3')
axes12[1].scatter(x25, y25, label='redundancy=2')
axes12[1].scatter(x26, y26, label='redundancy=1')
axes12[1].set_xscale('log')
axes12[1].set_yscale('log')
axes12[1].set_xlabel('log' + r'$_{10}$($rank$)')
axes12[1].set_ylabel('log' + r'$_{10}P$($rank$)')
result34 = Counter(list(bike_list14.values()))
result35 = Counter(list(bike_list15.values()))
result36= Counter(list(bike_list16.values()))
x34 = list(result34.keys())
x35 = list(result35.keys())
x36 = list(result36.keys())
x34.sort()
y34 = [result34[_x] / len(bike_list14.values()) for _x in x34]
x35.sort()
y35 = [result35[_x] / len(bike_list15.values()) for _x in x35]
x36.sort()
y36 = [result36[_x] / len(bike_list16.values()) for _x in x36]
axes11[2].scatter(x34, y34, label=f'redundancy=3')
axes12[2].scatter(x33, y33, label='redundancy=4')
axes12[2].scatter(x34, y34, label=f'redundancy=3')
axes12[2].scatter(x35, y35, label='redundancy=2')
axes12[2].scatter(x36, y36, label='redundancy=1')
# axes[2].set_xscale('log')
# axes[2].set_yscale('log')
axes12[2].set_xlabel('log' + r'$_{10}$($times$)')
axes12[2].set_ylabel('log' + r'$_{10}P$($times$)')
axes11[0].legend()
axes11[1].legend()
axes11[2].legend()
axes12[0].legend()
axes12[1].legend()
axes12[2].legend()


# In[ ]:


# TODO:跑一个随机选择，一周的数据。2023年5月4日12:17:03


# ## bike2-全时刻调度

# In[ ]:


# bike2不加调度
bike2 = Bike()
# 设置时间格式
bike2.set_time_format("%Y-%m-%d %H:%M:%S")
bike2.set_start_time(start_time)
bike2.set_day(10, 16)
# 测试读数据
bike2.read_data(path)
bike2.init_position()


# ### 车辆数不变

# In[ ]:


# bike2.get_rank_list()
#     bike.move(t)

rank_list21 = []
bike_list21 = {}
time_list21 = []
move_list21 = {}
count_list21 = []
bike_trip={}
count_list = {}
err_list21 = []
bike2.reduce(1)
bike_num = len(bike0.bike_dict.keys())
print(bike_num)
bn = 0
for g in bike2.grid_dict.keys():
    bn+= len(bike2.grid_dict[g][0])
print(bn)
for t in range(DAY*24):
    print(t)
#     bn = 0
#     for g in grid_keys:
#         bn+=len(bike2.grid_dict[g][t])
#     print("调度前的车辆数："+str(bn))
    if t!=0:
        bike2.move2(t,move_list21,2,4)
    bike2.simulation(t, rank_list21, bike_list21,bike_trip,count_list,err_list21)
#     bn = 0
#     grid_keys = bike2.grid_dict.keys()
#     for g in grid_keys:
#         bn+=len(bike2.grid_dict[g][t])
#     print("调度后的车辆数："+str(bn))


# In[ ]:


result31 = Counter(list(bike_list21.values()))
# result32 = Counter(list(bike_list22.values()))
# result33 = Counter(list(bike_list23.values()))
x31 = list(result31.keys())
# x32 = list(result32.keys())
# x33 = list(result33.keys())
x31.sort()
y31 = [result31[_x] / len(bike_list21.values()) for _x in x31]
# plt.xlim([0, 16])
print(x31)
print(y31)
plt.scatter(x31, y31, label='move')
plt.scatter(x11,y11,label='simulation')
plt.legend()


# In[ ]:


_x31 = []
_y31 = []
y_sum = 0
# print(_x31)
# print(len(bike_list215.values()))
for _x in x31:
    _y = (result31[_x]/len(bike_list21.values()))
    y_sum+=_y
#     print(_y)
    if y_sum >=0.95:
        print(_y)
        break
#     y_sum+=_y
    _x31.append(_x)
    _y31.append(_y)
print(_x31)
print(_y31)
print(y_sum)
plt.scatter(_x31, _y31, label=f'redundancy={r0}(r0)')


# ### 冗余率=5

# In[ ]:


# time1 =time.time()
# # print(time1)
# ratios = [ratio35,ratio4,ratio5,ratio6,ratio7,ratio8]
# rs = [r35,r4,r5,r6,r7,r8]
# result_x = []
# result_y = []
# # print(ratios)
# for i in range(1):
#     # print
# #     bn = 0
# #     for g in grid_keys:
# #         bn+=len(bike2.grid_dict[g][0])
# #     print(bn)
# #     print((rs[i] * max_demand)) 
#     for thre in range(1,2):
#         rank_list22 = []
#         bike_list22 = {}
#         move_list22 = {}
#         time_list22 = []
#         count_list22 = []
#         bike_trip22 ={}
#         err_list22 = []
#         bike2.reduce(ratios[i])
#         print(ratios[i],rs[i])
#         tempx = []
#         tempy = []
#         for t in range(DAY*24):
#             print(t)
#             if t != 0:
#                 mt1=time.time()
#                 bike2.move2(t,move_list22,3,3)
#                 mt2=time.time()
#                 print(f"排序时间:{mt2-mt1}")
#             st1=time.time()
#             bike2.simulation(t, rank_list22, bike_list22,bike_trip22,count_list22,err_list22)
#             st2=time.time()
#             print(f"仿真时间:{st2-st1}")
#         result32 = Counter(list(bike_list22.values()))
#         # result32 = Counter(list(bike_list22.values()))
#         # result33 = Counter(list(bike_list23.values()))
#         x32 = list(result32.keys())
#         # x32 = list(result32.keys()) 
#         # x33 = list(result33.keys())
#         x32.sort()
#         y32 = [result32[_x] / len(bike_list22.values()) for _x in x32]
# #         temp.append
#         # plt.xlim([0, 25])
#         # plt.scatter(x32, y32, label=f'redundancy={r0}(r0)')
#         print(x32)
#         print(y32)
#         tempx.append(x32)
#         tempy.append(y32)
#         result_x.append(tempx)
#         result_y.append(tempy)
# time2 = time.time()
# print(time2-time1)
# time1 =time.time()
# print(time1)
ratios = [ratio7,ratio8]
rs = [r7,r8]
result_x = []
result_y = []
# print(ratios)
for i in range(len(rs)):
    # print
#     bn = 0
#     for g in grid_keys:
#         bn+=len(bike2.grid_dict[g][0])
#     print(bn)
#     print((rs[i] * max_demand)) 
    for thre in range(1, 6):
        rank_list22 = []
        bike_list22 = {}
        move_list22 = {}
        time_list22 = []
        count_list22 = []
        bike_trip22 ={}
        err_list22 = []
        bike2.grid_dict = None
#         print(bike2.grid_dict)
        bike2.reduce(ratios[i])
#         print(len(bike2.grid_dict))
        print(ratios[i],rs[i])
        tempx = []
        tempy = []
#         print(len(bike2.grid))
        logger.info(f"r:{rs[i]},threshold:{thre}")
        for t in range(DAY*24):
            stime = time.time()
            sort_t = 0
            cal_t = 0
            print(t)
            if t != 0:
                time1 = time.time()
                bike2.move1(t,move_list22,thre,CENTRE_MAP)
                time2 = time.time()
#                 print("排序时间："+str(time2-time1))
            time1 = time.time()
            bike2.simulation(t, rank_list22, bike_list22,bike_trip22,count_list22,PROBABILITY_MAP,err_list22)
            time2 = time.time()
#             print("仿真时间："+str(time2-time1))
            etime= time.time()
#             print("总时间："+str(etime-stime))
        result32 = Counter(list(bike_list22.values()))
        # result32 = Counter(list(bike_list22.values()))
        # result33 = Counter(list(bike_list23.values()))
        x32 = list(result32.keys())
        # x32 = list(result32.keys()) 
        # x33 = list(result33.keys())
        x32.sort()
        y32 = [result32[_x] / len(bike_list22.values()) for _x in x32]
#         temp.append
        # plt.xlim([0, 25])
        # plt.scatter(x32, y32, label=f'redundancy={r0}(r0)')
        logger.info(x32)
        logger.info(y32)
        logger.info(err_list22)
    logger.info('-'*50)
#         print(x32)
#         print(y32)
#         tempx.append(x32)
#         tempy.append(y32)
#         result_x.append(tempx)
#         result_y.append(tempy)
# time2 = time.time()
# print(time2-time1)


# In[ ]:


print(len(bike0.order_data[1:]))
print(sum(err_list22))
print(sum(err_list22)/len(bike0.order_data[1:]))


# In[ ]:


## test
# print(bike2.grid_dict_raw.keys())
print(len(bike2.grid_dict_raw['75,118'][0]))
print(len(bike2.grid_dict['75,118'][1]))
print(ratios[i])
bike2.reduce(ratios[i])
print(len(bike2.grid_dict_raw['75,118'][0]))
print(len(bike2.grid_dict['75,118'][1]))


# In[ ]:


print(bike2.trip_dict[0])


# In[ ]:


print(result_x)
print(result_y)


# In[ ]:


print(len(result_x))


# In[ ]:


i = 0
for i in range(6):
    if i%5==0:
        print('-'*100)
    print(result_x[i][0])
    print(result_y[i][0])
    


# In[ ]:


result32 = Counter(list(bike_list22.values()))
# result32 = Counter(list(bike_list22.values()))
# result33 = Counter(list(bike_list23.values()))
x32 = list(result32.keys())
# x32 = list(result32.keys()) 
# x33 = list(result33.keys())
x32.sort()
y32 = [result32[_x] / len(bike_list22.values()) for _x in x32]
# plt.xlim([0, 25])
plt.scatter(x32, y32, label=f'redundancy={r0}(r0)')
print(x32)
print(y32)


# ### 冗余率=4

# In[ ]:


rank_list23 = []
bike_list23 = {}
time_list23 = []
err_list23 = []
bike2.reduce(ratio2)
for t in range(DAY*24):
    print(t)
    if t!=0:
        bike2.move(t,3)
    bike2.simulation(t, rank_list23, bike_list23, err_list23)


# ### 冗余率=3

# In[ ]:


rank_list24 = []
bike_list24 = {}
time_list24 = []
err_list24 = []
bike2.reduce(ratio3)
for t in range(DAY*24):
    print(t)
    if t!=0:
        bike2.move(t,3)
    bike2.simulation(t, rank_list24, bike_list24, err_list24)


# ### 冗余率=2

# In[ ]:


rank_list25 = []
bike_list25 = {}
time_list25 = []
err_list25 = []
bike2.reduce(ratio4)
for t in range(DAY*24):
    print(t)
    if t!=0:
        bike2.move(t)
    bike2.simulation(t, rank_list25, bike_list25, err_list25)


# ### 冗余率=1

# In[ ]:


rank_list26 = []
bike_list26 = {}
time_list26 = []
err_list26 = []
bike2.reduce(ratio5)
for t in range(DAY*24):
    print(t)
    if t!=0:
        bike2.move(t)
    bike2.simulation(t, rank_list26, bike_list26, err_list26)


# In[ ]:


err_list26


# ### 比较

# In[ ]:


fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6))
# print(err_list11,err_list12,err_list13)
t = [_t for _t in range(DAY*24)]
axes2[0].plot(t, err_list21, label=f'redundancy={r0}(r0)')
axes2[0].plot(t, err_list22, label='redundancy=5')
axes2[0].plot(t, err_list23, label='redundancy=4')
axes2[0].set_xlabel('Time(h)')
axes2[0].set_ylabel('ErrorCount')
result21 = Counter(rank_list21)
result22 = Counter(rank_list22)
result23 = Counter(rank_list23)
x21 = list(result21.keys())
x22 = list(result22.keys())
x23 = list(result23.keys())
x21.sort()
y21 = [result21[_x] / len(rank_list21) for _x in x21]
x22.sort()
y22 = [result22[_x] / len(rank_list22) for _x in x22]
x23.sort()
y23 = [result23[_x] / len(rank_list23) for _x in x23]
axes2[1].scatter(x21, y21, label=f'redundancy={r0}(r0)')
axes2[1].scatter(x22, y22, label='redundancy=5')
axes2[1].scatter(x23, y23, label='redundancy=4')
axes2[1].set_xscale('log')
axes2[1].set_yscale('log')
axes2[1].set_xlabel('log' + r'$_{10}$($rank$)')
axes2[1].set_ylabel('log' + r'$_{10}P$($rank$)')
result31 = Counter(list(bike_list21.values()))
result32 = Counter(list(bike_list22.values()))
result33 = Counter(list(bike_list23.values()))
x31 = list(result31.keys())
x32 = list(result32.keys())
x33 = list(result33.keys())
x31.sort()
y31 = [result31[_x] / len(bike_list21.values()) for _x in x31]
x32.sort()
y32 = [result32[_x] / len(bike_list22.values()) for _x in x32]
x33.sort()
y33 = [result33[_x] / len(bike_list23.values()) for _x in x33]
axes2[2].scatter(x31, y31, label=f'redundancy={r0}(r0)')
axes2[2].scatter(x32, y32, label='redundancy=5')
axes2[2].scatter(x33, y33, label='redundancy=4')

# axes2[2].set_xscale('log')
# axes2[2].set_yscale('log')
# axes2[2].set_xlim([0,20])
axes2[2].set_xlabel('log' + r'$_{10}$($times$)')
axes2[2].set_ylabel('log' + r'$_{10}P$($times$)')
# print(x31)
# result31 = Counter()
# axes[1].plot(t, rank_list11)x31 = list(bike)
# axes[1].plot(t,rank_list12)
# axes[1].plot(t,rank_list13)
# print(rank_list11)
# print(rank_list12)
# print(bike_list11)

fig22, axes22 = plt.subplots(1, 3, figsize=(20, 6))
# print(err_list11,err_list12,err_list13)
t = [_t for _t in range(DAY*24)]
axes22[0].plot(t, err_list24, label=f'redundancy=3')
axes22[0].plot(t, err_list25, label='redundancy=2')
axes22[0].plot(t, err_list26, label='redundancy=1')
axes22[0].set_xlabel('Time(h)')
axes22[0].set_ylabel('ErrorCount')
result24 = Counter(rank_list24)
result25 = Counter(rank_list25)
result26 = Counter(rank_list26)
x24 = list(result24.keys())
x25 = list(result25.keys())
x26 = list(result26.keys())
x24.sort()
y24 = [result24[_x] / len(rank_list24) for _x in x24]
x25.sort()
y25 = [result25[_x] / len(rank_list25) for _x in x25]
x26.sort()
y26 = [result26[_x] / len(rank_list26) for _x in x26]
axes22[1].scatter(x24, y24, label=f'redundancy=3')
axes22[1].scatter(x25, y25, label='redundancy=2')
axes22[1].scatter(x26, y26, label='redundancy=1')
axes22[1].set_xscale('log')
axes22[1].set_yscale('log')
axes22[1].set_xlabel('log' + r'$_{10}$($rank$)')
axes22[1].set_ylabel('log' + r'$_{10}P$($rank$)')
result34 = Counter(list(bike_list24.values()))
result35 = Counter(list(bike_list25.values()))
result36 = Counter(list(bike_list26.values()))
x34 = list(result34.keys())
x35 = list(result35.keys())
x36 = list(result36.keys())
x34.sort()
y34 = [result34[_x] / len(bike_list24.values()) for _x in x34]
x35.sort()
y35 = [result35[_x] / len(bike_list25.values()) for _x in x35]
x36.sort()
y36 = [result36[_x] / len(bike_list26.values()) for _x in x36]
axes2[2].scatter(x34, y34, label=f'redundancy=3')
axes22[2].scatter(x33, y33, label='redundancy=4')
axes22[2].scatter(x34, y34, label=f'redundancy=3')
axes22[2].scatter(x35, y35, label='redundancy=2')
axes22[2].scatter(x36, y36, label='redundancy=1')
# axes22[2].set_xscale('log')
# axes22[2].set_yscale('log')
# axes22[2].set_xlim([0,50])
axes22[2].set_xlabel('log' + r'$_{10}$($times$)')
axes22[2].set_ylabel('log' + r'$_{10}P$($times$)')
# print(x31)
# result31 = Counter()
# axes[1].plot(t, rank_list11)x31 = list(bike)
# axes[1].plot(t,rank_list12)
# axes[1].plot(t,rank_list13)
# print(rank_list11)
# print(rank_list12)
# print(bike_list11)
axes2[0].legend()
axes2[1].legend()
axes2[2].legend()
axes22[0].legend()
axes22[1].legend()
axes22[2].legend()


# In[ ]:


fig22, axes22 = plt.subplots(1, 3, figsize=(20, 6))
# print(err_list11,err_list12,err_list13)
t = [_t for _t in range(DAY*24)]
axes22[0].plot(t, err_list24, label=f'redundancy=3')
axes22[0].plot(t, err_list25, label='redundancy=2')
axes22[0].plot(t, err_list26, label='redundancy=1')
axes22[0].set_xlabel('Time(h)')
axes22[0].set_ylabel('ErrorCount')
result24 = Counter(rank_list24)
result25 = Counter(rank_list25)
result26 = Counter(rank_list26)
x24 = list(result24.keys())
x25 = list(result25.keys())
x26 = list(result26.keys())
x24.sort()
y24 = [result24[_x] / len(rank_list24) for _x in x24]
x25.sort()
y25 = [result25[_x] / len(rank_list25) for _x in x25]
x26.sort()
y26 = [result26[_x] / len(rank_list26) for _x in x26]
axes22[1].scatter(x24, y24, label=f'redundancy=3')
axes22[1].scatter(x25, y25, label='redundancy=2')
axes22[1].scatter(x26, y26, label='redundancy=1')
axes22[1].set_xscale('log')
axes22[1].set_yscale('log')
axes22[1].set_xlabel('log' + r'$_{10}$($rank$)')
axes22[1].set_ylabel('log' + r'$_{10}P$($rank$)')
result34 = Counter(list(bike_list24.values()))
result35 = Counter(list(bike_list25.values()))
result36 = Counter(list(bike_list26.values()))
x34 = list(result34.keys())
x35 = list(result35.keys())
x36 = list(result36.keys())
x34.sort()
y34 = [result34[_x] / len(bike_list24.values()) for _x in x34]
x35.sort()
y35 = [result35[_x] / len(bike_list25.values()) for _x in x35]
x36.sort()
y36 = [result36[_x] / len(bike_list26.values()) for _x in x36]
# axes2[2].scatter(x34, y34, label=f'redundancy=3')
# axes22[2].scatter(x33, y33, label='redundancy=4')
x31.sort()
y31 = [result31[_x] / len(bike_list21.values()) for _x in x31]
axes22[2].scatter(x31, y31, label=f'redundancy={r0}(r0)')
axes22[2].scatter(x34, y34, label=f'redundancy=3')
axes22[2].scatter(x35, y35, label='redundancy=2')
axes22[2].scatter(x36, y36, label='redundancy=1')
# axes22[2].set_xscale('log')
# axes22[2].set_yscale('log')
# axes22[2].set_xlim([0,50])
axes22[2].set_xlabel('log' + r'$_{10}$($times$)')
axes22[2].set_ylabel('log' + r'$_{10}P$($times$)')


# In[ ]:


fig23 = plt.figure()
plt.scatter(x31, y31, label=f'redundancy={int(r0)}')
plt.scatter(x32, y32, label=f'redundancy={5}')
plt.scatter(x33, y33, label=f'redundancy={4}')
plt.scatter(x34, y34, label=f'redundancy={3}')
plt.scatter(x35, y35, label=f'redundancy={2}')
plt.scatter(x36, y36, label=f'redundancy={1}')
# plt.xscale('log')
# plt.yscale('log')
plt.legend()


# In[ ]:


print(len(bike2.bike_dict))
print(len(bike_list21))
print(len(bike_list22))
print(len(bike_list23))
print(len(bike_list24))
print(len(bike_list25))
print(len(bike_list26))


# ## bike3-部分时刻调度

# In[ ]:


bike3 = Bike()
# 设置时间格式
start_time = '2020-09-14 00:00:00.0'
bike3.set_time_format("%Y-%m-%d %H:%M:%S.%f")
bike3.set_start_time(start_time)
bike3.set_day(14, 16)


# 测试读数据
bike3.read_data('./lishui-v2_14-16.csv')
bike3.init_position()


# ### 车辆数不变

# In[ ]:


rank_list31 = []
bike_list31 = {}
time_list31 = []
err_list31 = []
for t in range(3*24):
    h = t % 24
    bike3.simulation(t, rank_list31, bike_list31, err_list31)
    if h >=6 and h <=22:
        bike3.move(t)


# In[ ]:


rank_list32 = []
bike_list32 = {}
time_list32 = []
err_list31 = []
for t in range(3*24):
    h = t % 24
    bike3.simulation(t, rank_list31, bike_list31, err_list31)
    if h >=6 and h <=22:
        bike3.move(t)


# # 结果

# ## 出行距离分布

# In[ ]:


# 图1.distance distribution
result = Counter(bike.distance_list)
#         print(result)
x = list(result.keys())
x.sort()
x1 = []
for _x in x:
    if _x >= 100:
        x1.append(_x)
x = x1
# print(x)
y = [result[_x] / len(times_list) for _x in x]
log = True
x_log, y_log = log_pdf(x, y, 0, 4, 40)
if log:
    plt.xscale('log')
    plt.yscale('log')
# x_new ,y_new = log_pdf(x,y)
# plt.xlim([100,10000])
popt, pcov = curve_fit(func4, x, y, maxfev=10000)  # 100_150
y2 = [func4(_x, *popt) for _x in x]

r2 = r2_score(y, y2)

# print(r2)
plt.scatter(x_log, y_log, label=r'$\mu$=%d, $\sigma$=%d, R$^2$=%.2f' %
            (popt[1]*100, popt[2]*100, r2))
plt.plot(x, y2, color='red', linewidth=2, linestyle='-')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'log$_{10}d$(m)')
plt.ylabel('log' + r'$_{10}P$($d$)')
plt.legend()


# ## 出行时间分布

# In[ ]:


# 时间分布图
result = Counter(bike2.time_list)
x = list(result.keys())
y = [result[_x] / len(bike2.time_list) for _x in x]
# print(x)
plt.plot(x, y)
plt.scatter(x, y)
log = False
if log:
    plt.xscale('log')
    plt.yscale('log')
plt.xlabel('Time(h)')
plt.ylabel('P')


# ## 等待时间分布

# In[ ]:


# 图1.e等待时间
bresult = Counter(bike.bike_wait_interval_list)
xb = list(bresult.keys())
xb.sort()
yb = [bresult[_x] / len(bike.bike_wait_interval_list) for _x in xb]
plt.plot(xb, yb)


uresult = Counter(bike.user_wait_interval_list)
xu = list(uresult.keys())
xu.sort()
yu = [uresult[_x] / len(bike.user_wait_interval_list) for _x in xu]
plt.plot(xu, yu)
plt.xscale('log')
plt.yscale('log')


# ## 平均出行距离分布

# In[ ]:


# 图3.平均出行距离-用户-车辆
blist = bike.bike_distance_list
ulist = bike.user_distance_list

bresult = Counter(blist)
result1 = dict()
dd = 0.5
tt1 = math.ceil(max(blist) / dd)
for t in range(1, tt1 + 1):
    result1[t * dd] = 0
for _x in blist:
    t = math.ceil(_x / dd)
    result1[t * dd] += 1
x1 = list(result1.keys())
x1.sort()
y1 = [result1[_x] / len(blist) for _x in x1]

uresult = Counter(ulist)
result2 = dict()
tt2 = math.ceil(max(ulist) / dd)
for t in range(1, tt2 + 1):
    result2[t * dd] = 0
for _x in ulist:
    t = math.ceil(_x / dd)
    result2[t * dd] += 1
x2 = list(result2.keys())
x2.sort()
y2 = [result2[_x] / len(ulist) for _x in x2]

plt.scatter(x1, y1, label='bike')
plt.scatter(x2, y2, label='user')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'log$_{10}d$(m)')
plt.ylabel('log' + r'$_{10}P$($d$)')
plt.ylim([0.00001, 1])
plt.legend()


# ## 独特地点分布

# In[ ]:


# uniqueLocation
b_ulist = bike.bike_unique_list
u_ulist = bike.user_unique_list

resultb = Counter(b_ulist)
xb = list(resultb.keys())
xb.sort()
yb = [result[_x] / len(b_ulist) for _x in xb]
popt1, pcov1 = curve_fit(func2, xb, yb, maxfev=10000)  # 100_150
yb1 = [func2(_x, *popt1) for _x in xb]
plt.plot(xb, yb1)
plt.scatter(xb, yb, label='bike')

resultu = Counter(u_ulist)
xu = list(resultu.keys())
xu.sort()
yu = [result[_x] / len(u_ulist) for _x in xu]
popt2, pcov2 = curve_fit(func2, xu, yu, maxfev=10000)  # 100_150
yu1 = [func2(_x, *popt2) for _x in xu]
plt.plot(xu, yu1)
plt.scatter(xu, yu, label='user')

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'log$_{10}L$')
plt.ylabel('log' + r'$_{10}P$($L$)')
plt.legend()


# ## 第i次最长出行分布

# In[ ]:


# 图2.c-ith
b_ithlist = bike.bike_ith_list
u_ithlist = bike.user_ith_list
print(max(b_ithlist))
print(max(u_ithlist))
resultb = Counter(b_ithlist)
xb = list(resultb.keys())
xb.sort()
yb = [result[_x] / len(b_ithlist) for _x in xb]
plt.scatter(xb, yb, label='bike')

resultu = Counter(u_ithlist)
xu = list(resultu.keys())
xu.sort()
yu = [result[_x] / len(u_ithlist) for _x in xu]
plt.scatter(xu, yu, label='user')

# plt.xscale('log')
# plt.yscale('log')
plt.xlabel(r'$i^th$ trip')
plt.ylabel('P(i)')
plt.legend()


# ## gyration分布

# In[ ]:


# gy分布
gy_list1 = gyration(bike.bike_dict)
gy_list2 = gyration(bike.user_dict)
print(len(bike.bike_dict))
print(max(gy_list))
# result = Counter(gy_list1)
result1 = dict()
result2 = dict()
tt1 = math.ceil(max(gy_list1) / 0.5)
for t in range(1, tt1 + 1):
    result1[t*0.5] = 0

for item in gy_list1:
    t = math.ceil(item / 0.5) * 0.5
    result1[t] += 1

tt2 = math.ceil(max(gy_list2) / 0.5)
for t in range(1, tt2 + 1):
    result2[t*0.5] = 0

for item in gy_list2:
    t = math.ceil(item / 0.5) * 0.5
    result2[t] += 1

x1 = list(result1.keys())
x2 = list(result2.keys())
x1.sort()
x2.sort()

y1 = [result1[_x] / len(gy_list1) for _x in x1]
y2 = [result2[_x] / len(gy_list2) for _x in x2]

popt1, pcov1 = curve_fit(func7, x1, y1, maxfev=10000)  # 100_150
y11 = [func7(_x, *popt1) for _x in x1]

popt2, pcov2 = curve_fit(func4, x2, y2, maxfev=10000)  # 100_150
y22 = [func4(_x, *popt2) for _x in x2]

r21 = r2_score(y1, y11)
r22 = r2_score(y2, y22)
# print(r21)
# print(r22)

plt.plot(x1, y11)
plt.scatter(x1, y1, label=r'bike')

plt.plot(x2, y22)
plt.scatter(x2, y2, label='user')
# plt.scatter(x2, y2, label=r'user, $\mu$=%d, $\sigma$=%d, R$^2$=%.2f' % (popt2[1] * 100, popt2[2] * 100, r22))
# plt.scatter(x1, y1, label=r'bike, $\mu$=%d, $\sigma$=%d, R$^2$=%.2f'%(popt1[1] * 100, popt1[2] * 100, r21))
# plt.scatter(x2, y2, label=r'user, $\mu$=%d, $\sigma$=%d, R$^2$=%.2f'%(popt[1] * 100, popt[2] * 100, r2))

plt.xscale('log')
plt.xlabel('log'+r'$_{10}r_g$'+'(km)')
plt.yscale('log')
plt.ylabel('log'+r'$_{10}P(r_g)$')
plt.legend()


# ## 冗余率

# In[ ]:


print(redundancy)
y = redundancy
x = [t for t in range(24)]
plt.plot(x, y)
plt.scatter(x, y)
plt.xlabel('Time')
plt.ylabel('Redundancy')
print(np.mean(y))


# In[ ]:


print("车辆总数："+str(len(bike.bike_dict.keys())))
print("日均订单数："+str((len(bike.order_data) - 1) // 14))
print("用户："+str(len(bike.user_dict.keys())))

# TODO:车辆/用户数量


# In[ ]:


redundancy = bike2.cal_redundancy(3*24)


# In[ ]:


rk_list = bike2.rank_list
result = Counter(rk_list)
#         print(result)
x = list(result.keys())
y = [result[_x] / len(bike2.rank_list) for _x in x]
r_popt, r_pcov = curve_fit(func7, x, y, maxfev=10000)


# In[ ]:


plt.rcParams['font.family'] = ['WenQuanYi Micro Hei']  # 指定默认字体
x = [t for t in range(24*3)]
print(err_list1)
# print(err_list2)
print(err_list3)

# 全天调度
plt.plot(x,err_list1,label='全天调度')
# 不调度
# plt.plot(x,err_list2,label='不调度')
# 6-22点调度
plt.plot(x,err_list3,label='6-22点调度')
plt.legend()


# In[ ]:


## 测试simulation   
def test_simulation(t,rank_list,bike_list):
        # copy_grid[t]计算t+1时刻的车辆的位置
        copy_grid = copy.deepcopy(bike.grid_dict)
        count = 0
#         rank_distribution = []
        flag=0
        suc = 0
        err = 0
        for trip in bike.trip_dict[t]:
            """
            2023年3月31日16:33:12
            关心的是每次选车的rank情况，所以只需要有选车时的车况就可以解决。
            1.每次读到订单时更新一下出发地栅格的车辆状况。
            2.更新之后再选择车辆。
            """
#             if count >=10:
#                 break
#             print(trip)
            count+=1
            sid = trip[0]
            eid = trip[1]
            stime = trip[2]
            etime = trip[3]
            pool1 = copy_grid[sid][t]
            ii = 0
            for ii in range(len(pool1)):
                # 结束时间
                item = pool1[ii]
                t1 = time.mktime(item[4].timetuple())
                # 订单时间
                t2 = time.mktime(stime.timetuple())
                if t2 >= t1:
                    # 计算到当前订单的等待时间
                    wtime = t2 - t1
                    vc = 1 / item[1] * 3600 / ( 3600 + wtime)
                    # 车辆为可用状态
                    pool1[ii] = [item[0], item[1], wtime, vc, item[4], 0]
            l1 = len(pool1)
            # 随机选择一辆可用的车
            if l1 == 0:
                err += 1
                continue
#             rs = random.randint(0, len(pool1) - 1)
            # 用户行为选车
#             print(pool1)
#             break
            rk, bikeid = select2(pool1)
            if bikeid == -1:
                print(pool1)
                continue
            print(rk, bikeid)
            # 要确保车不是使用状态
#             ok = 0
#             for ii in range(len(pool1)):
#                 if pool1[ii][-1] == 0:
#                     ok = 1
#                     break
#             if not ok:
#                 # 找不到可用的车
#                 err+=1
#                 continue
#             while ok and pool1[rs][-1] == 1:
#                 rs = random.randint(0, len(pool1) - 1)
#             print(f"rs:{rs}***************{l1}") 
            suc+=1
            rank_list.append(rk)
            for ii in range(len(pool1)):
                if pool1[ii][0] == bikeid:
                    bike_sel = pool1.pop(ii)
                    break
#             bike_sel = pool1.pop(rs)
            # 使用次数+1，等待时间置0，上次使用时间更新为订单结束时间，使用状态为1
            if bike_sel[0] not in bike_list:
                bike_list[bike_sel[0]] = 0
            bike_list[bike_sel[0]] += 1
            new_b = [bike_sel[0], bike_sel[1] + 1, 0, bike_sel[3],etime,1]
            l2 = len(pool1)
            pool2 = copy_grid[eid][t]
            l3 = len(pool2)
            pool2.append(new_b)
            l4 = len(pool2)
#             print(f"从{sid}移动第{rs}辆车, {bike_sel}去到{eid}, {l1, l2, l3, l4}")
        grid_keys = copy_grid.keys()
#         print(grid_keys)
        for key in grid_keys:
            bike.grid_dict[key][t + 1] = copy_grid[key][t]
#         print(len(bike.grid_dict['59,77']))
        print(suc, err)
#         return pool1
        return copy_grid
        # Copy_grid[t] = Position(t) + Move(t) + Trip(t) => Position(t + 1)
# pool1 = test_simulation(0, rank_list1, bike_list1)


# In[ ]:


# orderid	userid	bikeid	count_bike	start_time	end_time	start_location_x	start_location_y	end_location_x	end_location_y	d	sid	eid
bike1 = Bike()
# 设置时间格式
start_time = '2020-09-14 00:00:00.0'
bike1.set_time_format("%Y-%m-%d %H:%M:%S.%f")
bike1.set_start_time(start_time)
bike1.set_day(14, 16)

# 测试读数据
bike1.read_data('./lishui-v2_14-16.csv')
bike1.init_position()
def get_rank(bike):
    data = bike.order_data
    bike_index = index.Index()
    dict1 = {}
    dict2 = {}
    rank_list = []
    err = 0
    succ = 0
    ii = 0
    start_time = bike.start_time
    for trip in data[1:]:
        bikeid = int(trip[2])
        stime = trip[4]
        etime = trip[5]
        sdt = datetime.strptime(stime, bike.time_format)
        edt = datetime.strptime(etime, bike.time_format)
        sx = float(trip[6])
        sy = float(trip[7])
        ex = float(trip[8])
        ey = float(trip[9])
        if bikeid not in dict1:
            # 使用次数，等待时间（订单时间-上次使用时间），车况值, 上次使用时间（结束时刻），车辆状态0/1
            dict1[bikeid] = [1, 0, 1.0, start_time, 0]
            # 初始化插入
            bike_index.insert(bikeid,[sx, sy, sx, sy])
            # 车辆的位置
            dict2[bikeid] = [sx,sy]
        else:
            continue
    err = 0
    for trip in data[1:]:
        bikeid = int(trip[2])
        stime = trip[4]
        etime = trip[5]
        sdt = datetime.strptime(stime, bike.time_format)
        edt = datetime.strptime(etime, bike.time_format)
        sx = float(trip[6])
        sy = float(trip[7])
        ex = float(trip[8])
        ey = float(trip[9])
#         if bikeid not in dict2:
#             dict2[bike] = []
#         dict2[bike] = [ex, ey]
        bound = 0.0025
        intersec = list(bike_index.intersection([sx-bound, sy-bound, sx+bound, sy+bound]))
        # 可用的车辆列表
        ok = []
        isMove = True
        for bid in intersec:
            if bid == bikeid:
                # 如果找到了这辆车说明没有被移动过
                ismove = False
            if dict1[bid][4]== 1 and dict1[bid][3] < sdt:
                dict1[bid][4] = 0
            if dict1[bid][4] == 0:
                wtime = time.mktime(sdt.timetuple()) - time.mktime(dict1[bid][3].timetuple())
                dict1[bid][1] = wtime
#                 if dict1[bid][0] == 0 or dict1[bid][1] == 0:
#                     print(dict1[bid][0], dict1[bid][1])
#                     print(trip)
                cond = 1 / dict1[bid][0] * 3600 / (3600 + dict1[bid][1])
                dict1[bid][2] = cond
                ok.append([bid, cond])
        if isMove:
            # 真实位置
            tx, ty = dict2[bikeid][0], dict2[bikeid][1]
            # 从原位置移出
            bike_index.delete(bikeid, [tx, ty, tx, ty])
            # 移到新位置
            bike_index.insert(bikeid, [sx, sy, sx, sy])
            wtime = time.mktime(sdt.timetuple()) - time.mktime(dict1[bikeid][3].timetuple())
            dict1[bikeid][1] = wtime
            cond = 1 / dict1[bikeid][0] * 3600 / (3600 + dict1[bikeid][1])
            dict1[bikeid][2] = cond
            ok.append([bikeid, cond])
        ok.sort(key=lambda x:x[1], reverse=True)
        ii = 0
        while ii < len(ok):
            if ok[ii][0] == bikeid:
                break
            ii += 1
        if ii == len(ok):
            err += 1
            print("error")
        else:
            succ+=1
            bike_index.delete(bikeid,[sx, sy, sx, sy])
            bike_index.insert(bikeid,[ex, ey, ex, ey])
            dict2[bikeid] = [ex, ey]
            dict1[bikeid][0] += 1
            dict1[bikeid][3] = edt
            dict1[bikeid][4] = 1
            rank_list.append(ii)
    print(err,succ)
    return rank_list
rk_list = get_rank(bike1)


# In[ ]:


times_list = [_x + 1 for _x in rk_list]
result = Counter(times_list)
#         print(result)
x = list(result.keys())
y = [result[_x] / len(times_list) for _x in x]
plt.scatter(x, y)
log = True
if log:
    plt.xscale('log')
    plt.yscale('log')


# In[ ]:


result = Counter(times_list)
#         print(result)
x = list(result.keys())
y = [result[_x] / len(times_list) for _x in x]
popt, pcov = curve_fit(func2, x, y, maxfev=10000)  # 100_150
y888 = [func2(i, *popt) for i in x]
# print(y888)
plt.scatter(x, y)
plt.plot(d2.keys(), d2.values(), color='red', linewidth=2, linestyle='-')
plt.xscale('log')
plt.yscale('log')


# In[ ]:


## 测试move
def test_move(t):
    """
    2023年4月3日10:28:46
    建立索引，根据t时刻最开始的车辆分布，移动车辆，得到移动后的车辆分布。
    1. 如果是0时刻，不用移动
    2. 如果是t时刻，需要移动，移动是在simulation前完成的
    3. 结合[t, t + 1]时间内的订单数量，得到移入栅格、移出车辆。（这部分以文件形式输出提供优化调度使用）
    4. 得到移入移出
    """
    gg = bike.grid_dict
    grid_key = bike.grid_dict.keys()
    in_grids, out_grids = bike.find_bike_grid(t)
#     print(out_grids)
    in_keys = in_grids.keys()
    out_keys = out_grids.keys()
    in_nums, out_nums = 0, 0
    for ii in in_keys:
        in_nums += in_grids[ii][1]
    
    for oo in out_keys:
        out_nums += out_grids[oo][1]
    
    id_map = bidict()
    in_grid_index = index.Index()
    # 1.向Rtree中插入所有的接收栅格
    for ii,ig in enumerate(in_keys):
        id_map[ig] = ii
        cor = list(map(int,ig.split(',')))
#         Q = int(in_grids[id_map.inverse[ii]][1])
        lon, lat = tbd.grid_to_centre(cor, params_rec)
        in_grid_index.insert(id_map[ig],[lon, lat, lon, lat])
    # 2.移动车辆
    success=0
    error = 0
    od ={}
    for oo in out_keys:
        # 记录被移除的车辆
        bb_remove = []
        for bb in out_grids[oo][0]:
            # ['7910418184', 2, 16047.0, 0.09161704076958314, datetime.datetime(2020, 9, 14, 5, 9, 1), 0, 130, '59,78']
#             print(bb)
            cor = list(map(int, oo.split(',')))
            cond = float(bb[3])
            old_rank = int(bb[-2])
            lon, lat = tbd.grid_to_centre(cor, params_rec)
            nearest_grid = list(in_grid_index.nearest([lon, lat, lon, lat], objects=True))
            flag = False
            for gb in nearest_grid:
                if flag:
                    break
                g=gb.id
                raw_id = id_map.inverse[g]
        #         print(f"栅格{g}的当前可接受量：{in_grids[id_map.inverse[g]]}")
                target = in_grids[raw_id]
#                 print(target)
                if target[1] > 0:
                    # 如果当前栅格还可以接受车辆的话，判断排名
                    k = 0
                    while k < len(bike.grid_dict[raw_id][t]):
#                         print(bike.grid_dict[raw_id][t][k])
                        if cond >= bike.grid_dict[raw_id][t][k][3]:
                            break
                        k += 1
                    if k + 1 >= old_rank:
                        # 不移动
#                         print("移动失败")
                        continue
                    else:
#                         print(f"{len(bike.grid_dict[raw_id][t])}")
#                         print(bike.grid_dict[raw_id][t])
                        bike.grid_dict[raw_id][t].append(bb[:6])
                        bb_remove.append(bb[0])
                        jj = 0
                        while jj <len(bike.grid_dict[oo][t]):
                            if bike.grid_dict[oo][t][jj][0] == bb[0]:
                                # 从原栅格中移除车辆
                                bike.grid_dict[oo][t].pop(jj)
                                break
                            jj += 1
                        bike.grid_dict[raw_id][t].sort(key=lambda x:x[3], reverse=True)
#                         target[1] = sorted(target[1] , reverse=True)
#                         print(f"{len(bike.grid_dict[raw_id][t])}")
#                         print(f"移动成功，排名提升了{old_rank - k - 1}")
                        success+=1
                        flag = True
                        if (oo, raw_id) not in od:
                            od[oo, raw_id] = 0
                        od[oo ,raw_id] += 1
        #                 print(f'before{in_grids[id_map.inverse[g]][0]}')
                        target[1] -= 1
        #                 print(f'next{in_grids[id_map.inverse[g]][0]}')
                if target[1] == 0:
                    # 移除该栅格
                    in_grid_index.delete(gb.id, gb.bbox)
            if flag == False:
                error += 1
#             for item in out_grids[oo][0]:
#                 if item[0] in bb_remove:
#                     print(item)
#     print(out_grids)
    print(out_nums, success, error)

        
test_move(10)


# In[ ]:


bike = Bike()
# 设置时间格式
start_time = '2020-09-14 00:00:00.0'
bike.set_time_format("%Y-%m-%d %H:%M:%S.%f")
bike.set_start_time(start_time)
bike.set_day(14, 16)

# 测试读数据
bike.read_data('./lishui-v2.csv')
bike.init_position()
# bike.simulation()
## 不进行调度的
bike_list1 = {}
rank_list1 = []
for t in range(0, 14*24):
    print(t)
    test_simulation(t, rank_list1, bike_list1)

print(rank_list1)
times_list = rank_list1
#         entires = 0
#         for b in self.bike_dict.keys():
#             times = len(self.bike_dict[b])
#             times_list.append(times)
#             entires+=times
result = Counter(times_list)
#         print(result)
x = list(result.keys())
y = [result[_x]/len(times_list) for _x in x]
plt.scatter(x,y)
log = True
if log:
    plt.xscale('log')
    plt.yscale('log')
times_list2 = list(bike_list1.values())
result = Counter(times_list2)
#         print(result)
x = list(result.keys())
y = [result[_x]/len(times_list2) for _x in x]
plt.scatter(x,y)
log = True
if log:
    plt.xscale('log')
    plt.yscale('log')
# TODO:原始所有天/这一天真实/模拟这一天的累积分布，计算KS距离/L1距离。2023年4月6日11:32:30


# In[ ]:


plt.rcParams['font.family'] = ['Simhei']
# times_list = rank_list1
# #         entires = 0
# #         for b in self.bike_dict.keys():
# #             times = len(self.bike_dict[b])
# #             times_list.append(times)
# #             entires+=times
# result = Counter(times_list)
# #         print(result)
# x = list(result.keys())
# y = [result[_x]/len(times_list) for _x in x]
# plt.scatter(x,y)
# log = False
# if log:
#     plt.xscale('log')
#     plt.yscale('log')
times_list2 = list(bike_list1.values())
result = Counter(times_list2)
#         print(result)
x = list(result.keys())
y = [result[_x]/len(times_list2) for _x in x]
plt.scatter(x,y)
log = True
if log:
    plt.xscale('log')
    plt.yscale('log')
times_list22 = list(bike_list2.values())
result = Counter(times_list22)
#         print(result)
x = list(result.keys())
y = [result[_x]/len(times_list22) for _x in x]
plt.scatter(x,y)
log = True
if log:
    plt.xscale('log')
    plt.yscale('log')
# plt.xlim([0,60])
# TODO:选车的时候加上用户的选择，2023年4月11日12:10:57。


# In[ ]:


bike = Bike()
# 设置时间格式
start_time = '2020-09-14 00:00:00.0'
bike.set_time_format("%Y-%m-%d %H:%M:%S.%f")
bike.set_start_time(start_time)
bike.set_day(14, 27)


# 测试读数据
bike.read_data('./lishui-v2.csv')
bike.init_position()

## 进行调度的
bike_list2 = {}
rank_list2 = []
for t in range(0, 24*14):
    print(t)
    test_simulation(t, rank_list2, bike_list2)
    mod = t % 24
    if mod >= 6 and mod <= 22:
        # 如果t大于10的话，调度
        # TODO3:修改调度的时间段，早6-晚10，每两小时调度一次。2023年4月17日10:34:46
        test_move(t)

print(rank_list2)
times_list = rank_list2
#         entires = 0
#         for b in self.bike_dict.keys():
#             times = len(self.bike_dict[b])
#             times_list.append(times)
#             entires+=times
result = Counter(times_list)
#         print(result)
x = list(result.keys())
y = [result[_x]/len(times_list) for _x in x]
plt.scatter(x,y)
log = True
if log:
    plt.xscale('log')
    plt.yscale('log')
times_list2 = list(bike_list2.values())
result = Counter(times_list2)
#         print(result)
x = list(result.keys())
y = [result[_x]/len(times_list2) for _x in x]
plt.scatter(x,y)
log = True
if log:
    plt.xscale('log')
    plt.yscale('log')


# In[ ]:


times_list = rank_list2
#         entires = 0
#         for b in self.bike_dict.keys():
#             times = len(self.bike_dict[b])
#             times_list.append(times)
#             entires+=times
result = Counter(times_list)
#         print(result)
x = list(result.keys())
y = [result[_x]/len(times_list) for _x in x]
plt.scatter(x,y)
log = True
if log:
    plt.xscale('log')
    plt.yscale('log')
times_list2 = list(bike_list2.values())
result = Counter(times_list2)
#         print(result)
x = list(result.keys())
y = [result[_x]/len(times_list2) for _x in x]
plt.scatter(x,y)
log = True
if log:
    plt.xscale('log')
    plt.yscale('log')


# In[ ]:


"""
使用次数分布图，真实/模拟/模拟（有调度）
"""
plt.rcParams['font.family'] = ['WenQuanYi Micro Hei']  # 指定默认字体
list11 = list(bike_list1.values())
result11 = Counter(list11)
list22 = list(bike_list2.values())
result22 = Counter(list22)
x1 = list(result11.keys())
y1 = [result11[_x] / len(list11) for _x in  x1]
x1_new,y1_new = log_pdf(x1, y1)
# plt.scatter(x_new,y_new)
plt.scatter(x1_new,y1_new,label="模拟")
x2 = list(result22.keys())
y2 = [result22[_x] / len(list22) for _x in  x2]
x2_new,y2_new = log_pdf(x2, y2)

plt.scatter(x2_new, y2_new, label='模拟（有调度）')
bike_keys = bike.bike_dict.keys()
list33 = []
for bb in bike_keys:
    list33.append(len(bike.bike_dict[bb]) - 1)
# times_list3 = list(bike_list2.values())
result = Counter(list33)
#         print(result)
x3 = list(result.keys())
y3 = [result[_x] / len(list33) for _x in x3]
x3_new, y3_new = log_pdf(x3, y3)

plt.scatter(x3_new, y3_new, label='真实')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('#trips')
plt.ylabel('P')
plt.legend()
# TODO2: 度量variance，看有无调度的情况下哪个更小。


# In[ ]:


"""
CDF分布图
"""
def plot_cdf(_list, label=None):
    result = Counter(_list)
    x = list(result.keys())
    x.sort()
    xmax = x[-1]
#     print(xmax)
    # 归一化
    x1 = [_x/xmax for _x in x]
#     print(x)
    cum = 0
    y = []
    for _x in x:
        cum += result[_x]
        y.append(cum/len(_list))
    plt.scatter(x,y,label=label)
plot_cdf(list33,label="真实值")
plot_cdf(list11,label='模拟')
plot_cdf(list22,label='模拟（调度）')
plt.xlabel('#trip')
plt.ylabel('P')
plt.legend()


# In[ ]:


times_list = rank_list2
#         entires = 0
#         for b in self.bike_dict.keys():
#             times = len(self.bike_dict[b])
#             times_list.append(times)
#             entires+=times
result = Counter(times_list)
#         print(result)
x = list(result.keys())
print(len(x),max(x))
y = [result[_x]/len(times_list) for _x in x]
plt.scatter(x,y)
log = True
if log:
    plt.xscale('log')
    plt.yscale('log')
times_list2 = list(bike_list2.values())
result = Counter(times_list2)
#         print(result)
x = list(result.keys())
y = [result[_x]/len(times_list2) for _x in x]
plt.scatter(x,y)
log = True
if log:
    plt.xscale('log')
    plt.yscale('log')
# plt.xlim([0,60])
# x.sort()
# print(x)


# In[ ]:


plt.rcParams['font.family'] = ['WenQuanYi Micro Hei']  # 指定默认字体
# plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# from matplotlib.font_manager import FontManager
# fm = FontManager()
# mat_fonts = set(f.name for f in fm.ttflist)
# print('我的电脑上能显示中文的字体有：',mat_fonts)
# CDF
def plot_cdf(_list, label=None):
    result = Counter(_list)
    x = list(result.keys())
    x.sort()
    xmax = x[-1]
#     print(xmax)
    # 归一化
    x1 = [_x/xmax for _x in x]
#     print(x)
    cum = 0
    y = []
    for _x in x:
        cum += result[_x]
        y.append(cum/len(_list))
    plt.scatter(x,y,label=label)
plot_cdf(bike_list3,label="真实值")
plot_cdf(list(bike_list1.values()),label='模拟')
plot_cdf(times_list2,label='模拟（调度）')
plt.xlabel('#trip')
plt.ylabel('P')
plt.legend()


# In[ ]:


# 归一化后的CDF
def plot_cdf(_list,label=None):
    result = Counter(_list)
    x = list(result.keys())
    x.sort()
    xmax = x[-1]
    # 归一化
    x1 = [_x/xmax for _x in x]
#     print(x)
    cum = 0
    y = []
    for _x in x:
        cum += result[_x]
        y.append(cum/len(_list))
    plt.scatter(x1,y,label=label)
plot_cdf(bike_list3,label="真实值")
plot_cdf(list(bike_list1.values()),label='模拟')
plot_cdf(times_list2,label='模拟（调度）')
plt.xlabel('#trip')
plt.ylabel('P')
plt.legend()


# In[ ]:


bike = Bike()
# 设置时间格式
start_time = '2020-09-14 00:00:00.0'
bike.set_time_format("%Y-%m-%d %H:%M:%S.%f")
bike.set_start_time(start_time)
bike.set_day(14, 27)


# 测试读数据
bike.read_data('./lishui-v2.csv')
bike.init_position()
bike_keys = bike.bike_dict.keys()
bike_list3 = []
for bb in bike_keys:
    bike_list3.append(len(bike.bike_dict[bb]) - 1)
print(bike_list3)
# times_list3 = list(bike_list2.values())
times_list3 = bike_list3
result = Counter(times_list3)
#         print(result)
x = list(result.keys())
y = [result[_x] / len(times_list3) for _x in x]
plt.scatter(x,y)
log = True
if log:
    plt.xscale('log')
    plt.yscale('log')


# In[ ]:


# TODO:真实出行的rank_list，画分布
rank_list3 = []
raw_data = bike.order_data
copy_grid = copy.deepcopy(bike.grid_dict)
# print(copy_grid)
# for item in raw_data[1:]:
#     print(item)


# In[ ]:


random.seed(1)
# a = [random.randint(1,10) for k in range(10)]
# print(a)
# b = a.pop(0)
# print(b)
# print(a)
a  = {0:[random.randint(1,10) for k in range(10)], 1:[random.randint(1,10) for k in range(10)]}
print(a)
b = copy.deepcopy(a)
print(b)
c = b[0]
c.pop(0)
print(b)
print(a)
a[0].pop(0)
print(a)

