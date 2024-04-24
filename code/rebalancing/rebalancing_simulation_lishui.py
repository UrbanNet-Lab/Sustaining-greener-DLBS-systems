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


# In[2]:


def save_data(filename, x, y):
    data = pd.DataFrame(columns=['x', 'y'])
    data['x'] = x
    data['y'] = y
    data.to_csv(f'{filename}.csv', index=False)


# In[32]:


class Bike:
    def __init__(self):
        # 车辆原始订单数据
        self.order_data = []
        # 按小时分组的trip(start_pos,end_pos,start_id,end_id)
        self.trip_dict = {}
        # 按车辆分组的订单列表字典，()
        self.bike_dict = {}
        # 栅格字典(记录每个栅格的车辆信息)
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
            #  orderid	userid	bikeid	biketype	starttime	endtime	start_x	start_y	end_x	end_y	start_lng_col	start_lat_col	end_lng_col	end_lat_col	sid	eid
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
    def simulation(self, t, rank_list, bike_list,bike_trip,count_list,PROBABILITY_MAP,err_list,bike_ids):
        """
        rank_list:出行的选车排名分布
        bike_list:车辆的使用次数分布
        time_list:出行时间分布
        distance_list:出行距离分布
        """
#         current_t = dict()
        grid_keys = self.grid_dict.keys()
        current_t = dict()
        for gg in grid_keys:
#             current_t[gg] = copy.deepcopy(self.grid_dict[gg][t])
            current_t[gg] = copy.deepcopy(self.grid_dict[gg][t])
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
        
        if t not in self.trip_dict:
            self.trip_dict[t] = []
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
#             pool1 = self.grid_dict[sid][t+1]
            pool1 = current_t[sid]
#             print(pool1)
            ii = 0
            st= time.time()
            for ii in range(len(pool1)):
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
                bike_ids.append(np.nan)
                print(sid, pool1)
                continue
            t1= time.time()
        
            rk, bikeid = select2(pool1, PROBABILITY_MAP)
            t2= time.time()
            t20+=(t2-t1)
            
            if bikeid == -1:
                print(sid, pool1)
                err += 1
                bike_ids.append(np.nan)
                continue
            bike_ids.append(bikeid)
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
#             self.grid_dict[eid][t+1].append(new_b)
            current_t[eid].append(new_b)
            t2= time.time()
            t30+=(t2-t1)
        err_list.append(err)
        grid_keys = current_t.keys()
        for key in grid_keys:
            self.grid_dict[key][t + 1] = copy.deepcopy(current_t[key])
#         print("计算等待时间："+str(mkt))
#         print("状态更新："+str(t10))
#         print("车辆排序时间："+str(sort_t))
#         print("计算概率事件："+str(cal_t))
#         print("选择车辆："+str(t20))
#         print("移动车辆："+str(t30))

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
    def moveBytimes(self, t, move_list,th,CENTRE_MAP,RADIUS,RebalancingList):
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
        in_grids, out_grids = self.find_bike_grid4(t, thre=th)
#         t2 = time.time()
#         print("找车的时间："+str(t2-t1))
        temp=dict()
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
            # 0.01：1000m
            d= RADIUS
            cor = list(map(int, ii.split(',')))
            lon, lat = CENTRE_MAP[ii]
#             self.grid_dict[ii][t].sort(
#                                 key=lambda x: x[1], reverse=True)
            p=0
            while in_grids[ii][1] > 0 and p <= 5:
                # 查找范围，以lon,lat为中心
                box = [lon - d*p, lat - d*p, lon + d*p, lat + d*p]
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
#                     if old_rank > jj + 1:
#                     jj = 
                    if True:
                        # 说明排名提升了
                        success+=1
                        # 移动到栅格里
                        self.grid_dict[ii][t].append(bb[:6])
                        self.grid_dict[ii][t].sort(
                                key=lambda x: x[3], reverse=True)
                        k = 0
                        # 从原栅格移出
                        flag = True
                        while k < len(self.grid_dict[oo][t]):
                            if self.grid_dict[oo][t][k][0] == bb[0]:
                                self.grid_dict[oo][t].pop(k)
                                flag= False
                                break
                            k += 1
                        if flag:
                            print("没找到这辆车！！！！！")
                            print(bb)
#                             print(type(bb[0]))
                            print(self.grid_dict[oo][t])
                            return 
                        if bb[0] not in move_list:
                            move_list[bb[0]] = 0
                        if (oo,ii) not in temp:
                            temp[(oo,ii)] = 0
                        temp[(oo,ii)] += 1
                        move_list[bb[0]] += 1
                        _lon, _lat = omap[item][2], omap[item][3]
                        out_bike_index.delete(item, [_lon, _lat, _lon, _lat])
                        in_grids[ii][1] -= 1
                p += 1
        unsatisfy = 0
        for (o,i) in temp:
            row = [o, CENTRE_MAP[o][0],CENTRE_MAP[o][1], i, CENTRE_MAP[i][0],CENTRE_MAP[i][1], temp[(o, i)], t]
            RebalancingList.append(row)
        for ii in in_keys:
            if in_grids[ii][1]>0:
                unsatisfy+=1
            error+=in_grids[ii][1]
        print(success, error, unsatisfy)
        
    def moveByrank(self, t,move_list,thre1,thre2):
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

    def find_bike_grid2(self, t, thre1=1,thre2=1):
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
        grid_list = self.grid_dict.keys()
        for t in range(_t):
            redundancy[t] = len(self.bike_dict.keys()) / len(self.trip_dict[t])
        return redundancy
    def find_bike_by_grid(grid_list):
        pass


# In[4]:


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
    sort_t +=(t2-t1)
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
    norm_probability = [p/p_sum for p in probability]
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
def save_data(filename, x, y):
    data = pd.DataFrame(columns=['x', 'y'])
    data['x'] = x
    data['y'] = y
    data.to_csv(f'{filename}.csv', index=False)


# In[7]:


# 设置时间格式
# 北京 './bj-1000m-10-16.csv'
# 丽水 './lishui-500m_14-20.csv' '2020-09-14 00:00:00.0' '%Y-%m-%d %H:%M:%S.%f'
# 上海 
# 金华 jinhua-v1_14-20.csv
SH_PATH = './sh-1000m.csv'
SH_PATH1 = './sh-1000m_1week.csv'
SH_PATH2 = './sh-1000m_2week.csv'
SH_PATH3 = './sh-1000m_3week.csv'
SH_PATH4 = './sh-1000m_4week.csv'
SH_ACC=1000
SH_DIST = 'shanghai.geojson'
SH_TF = '%Y-%m-%d  %H:%M:%S'
SH_ST = '2016-8-1  00:00:00'


BJ_PATH = '../bj_1000m_2.csv'
BJ_PATH1 = '../bj-1000m_10-16.csv'
BJ_ST = '2017-05-10 00:00:00'
BJ_TF = '%Y-%m-%d %H:%M:%S'
BJ_DIST = '../beijing.geojson'
BJ_ACC = 1000


LS_PATH = '../lishui-v2.csv'
LS_PATH1 = '../lishui-500m_14-20.csv'
LS_PATH2 = '../lishui-500m_21-27.csv'
LS_ST='2020-09-14 00:00:00.0'
LS_TF = '%Y-%m-%d %H:%M:%S.%f'
LS_DIST='../lishui_liandu.geojson'
LS_ACC = 500


NB_PATH = './ningbo_2-1000m.csv'
NB_PATH1 = './ningbo_2-1000m_14-20.csv'
# nb_path2 = './ningbo-1000m_21-27.csv'
NB_ST='2020-09-14 00:00:00.0'
NB_TF=LS_TF
NB_ACC=1000
NB_DIST = './ningbo_2.geojson'


JH_PATH = 'jinhua-v1.csv'
JH_PATH1 = './jinhua-v1_14-20.csv'
JH_PATH2 = './jinhua-v1_14-20.csv'
JH_TF=LS_TF
JH_ACC = 1000
path  = LS_PATH
city = 'LISHUI'
dist = LS_DIST
ACCURACY = LS_ACC
start_time = LS_ST
time_format = LS_TF


# In[6]:


data = pd.read_csv('../bj_1000m_2.csv')


# In[8]:


bike0 = Bike()
bike0.set_time_format(time_format)
bike0.set_start_time(start_time)
bike0.set_day(14, 27)
DAY = 14
bike0.read_data(path)
bike0.init_position()
# logger = get_logger(f"{city}", f"../log/{city}-v2-调度后车辆数小于需求数.log")
bike_list00 = {}
for key in bike0.bike_dict.keys():
    times = len(bike0.bike_dict[key])
    if times not in bike_list00:
        bike_list00[times] = 0
    bike_list00[times] += 1
x00 = list(bike_list00.keys())
x00.sort()
y_sum = sum(bike_list00.values())
y00 = [bike_list00[_x] /y_sum  for _x in x00]


# In[8]:


save_data(f'../raw/{city}_raw_usage_1',x00,y00)


# In[14]:


rank_raw = pd.read_csv(f'../raw/{city}_raw_rank.csv')
rank_x = rank_raw['x'].values
rank_y = rank_raw['y'].values
r_popt, r_pcov = curve_fit(func7, rank_x, rank_y, maxfev=10000)
PROBABILITY_MAP = dict()
rank_y_fit = []
for _x in rank_x:
    if _x not in PROBABILITY_MAP:
        prob = func7(_x, *r_popt) 
        PROBABILITY_MAP[_x] = prob
        rank_y_fit.append(prob)
plt.plot(rank_x, rank_y_fit,c='red')
plt.xscale('log')
plt.yscale('log')
plt.scatter(rank_x, rank_y)

# logger.info('raw_rank'+'*'*50)
# logger.info(rank_x)
# logger.info(rank_y)
# logger.info('*'*50)


# In[11]:


bike_list00 = {}
for key in bike0.bike_dict.keys():
    times = len(bike0.bike_dict[key])
    if times not in bike_list00:
        bike_list00[times] = 0
    bike_list00[times] += 1
x00 = list(bike_list00.keys())
x00.sort()
y_sum = sum(bike_list00.values())
y00 = [bike_list00[_x] /y_sum  for _x in x00]
save_data(f'../raw/{city}_raw_usage_2',x00,y00)


# In[ ]:


print(bike0.order_data[0:-2])


# In[20]:


# 计算地图数据
district = gpd.read_file(dist)
fig = plt.figure(1, (7, 5), dpi=150)
ax1 = plt.subplot(111)
district.plot(ax=ax1)
grid_rec, params_rec = tbd.area_to_grid(district, accuracy=ACCURACY)
CENTRE_MAP = dict()
grid_keys = bike0.grid_dict_raw.keys()
print(grid_keys)
for key in grid_keys:
    if key not in CENTRE_MAP:
        cor = list(map(int, key.split(',')))
#         print(cor)
        CENTRE_MAP[key] = tbd.grid_to_centre(cor, params_rec)


# In[9]:


# 北京冗余率14.9
for t in range(DAY * 24):
    if t not in bike0.trip_dict:
        print(t//24,t%24)
demand = [len(bike0.trip_dict[t]) if t in bike0.trip_dict else 0 for t in range(DAY * 24)]
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
print("每小时最大需求："+str(max_demand))

print(ratio1,ratio15, ratio2,ratio25, ratio3,ratio35,ratio4,ratio5,ratio6,ratio8)
print(len(bike0.bike_dict.keys()) / max_demand)


# In[28]:


bike1 = Bike()
# 设置时间格式
bike1.set_time_format(time_format)
bike1.set_start_time(start_time)
bike1.set_day(14, 27)
# 测试读数据
bike1.read_data(path)
bike1.init_position()


# In[16]:


rank_list22 = []
bike_list22 = {}
move_list22 = {}
time_list22 = []
count_list22 = []
bike_trip22 ={}
err_list22 = []
unsatisfy = []
bike1.grid_dict = None
bike1.reduce(ratio25)
for t in range(DAY*24):
    print(t)
    bike_ids=[]
    if t % 168==0 and t > 0:
        times = t /(7*24) 
        result = Counter(bike_list22.values())
        x = list(result.keys())
        x.sort()
        y = [result[_x] / len(bike_list22.values()) for _x in x]
        save_data(f'../raw/{city}_round_{times}',x,y)
    bike1.simulation(t, rank_list22, bike_list22,bike_trip22
                             ,count_list22,PROBABILITY_MAP,err_list22,bike_ids)
result = Counter(bike_list22.values())
x = list(result.keys())
x.sort()
y = [result[_x] / len(bike_list22.values()) for _x in x]
save_data(f'../raw/{city}_round_{times+1}',x,y)


# In[33]:


# 进行调度
bike2 = Bike()
# 设置时间格式
# start_time = '2020-09-14 00:00:00.0'
bike2.set_time_format(time_format)
bike2.set_start_time(start_time)
bike2.set_day(14, 27)
# 测试读数据
bike2.read_data(path)
bike2.init_position()
# print(bike2.trip_dict[0])


# In[34]:


# data = pd.read_csv(path)
ratios = [ratio1,ratio2,ratio3,ratio15,ratio25,ratio35,ratio4,ratio5,ratio6,ratio7,ratio8]
rs = [r1,r2,r3,r15,r25,r35,r4,r5,r6,r7,r8]
ratios = [ratio25]
rs = [r25]
result_x = []
result_y = []
RebalancingList = []
# print(ratios)

for i in range(len(rs)):
    # 记录每次骑了哪些车
    bike_ids=[]
    # print
#     bn = 0
#     for g in grid_keys:
#         bn+=len(bike2.grid_dict[g][0])
#     print(bn)
#     print((rs[i] * max_demand)) 
    for thre in range(1, 2):
        rank_list22 = []
        bike_list22 = {}
        move_list22 = {}
        time_list22 = []
        count_list22 = []
        bike_trip22 ={}
        err_list22 = []
        unsatisfy = []
        bike2.grid_dict = None
        bike2.reduce(ratios[i])
        print(ratios[i],rs[i])
        tempx = []
        tempy = []
        for t in range(DAY*24):
            if t % 168==0 and t > 0:
                times = t /(7*24) 
                result = Counter(bike_list22.values())
                x = list(result.keys())
                x.sort()
                y = [result[_x] / len(bike_list22.values()) for _x in x]
                save_data(f'../raw/{city}_round_{times}_rand',x,y)
            temp =0
            sort_t = 0
            cal_t = 0
            print(t)
            if True:
                bike2.moveBytimes(t,move_list22,1.5,CENTRE_MAP,0.01*ACCURACY/1000,RebalancingList)
                for g in bike2.grid_dict.keys():
                    if len(bike2.grid_dict[g][t])< bike2.history_nums[g][t]:
                        temp+=1
                unsatisfy.append(temp)
            bike2.simulation(t, rank_list22, bike_list22,bike_trip22
                             ,count_list22,PROBABILITY_MAP,err_list22,bike_ids)
        result = Counter(bike_list22.values())
        x = list(result.keys())
        x.sort()
        y = [result[_x] / len(bike_list22.values()) for _x in x]
        save_data(f'../raw/{city}_round_{times+1}_rand',x,y)


# In[ ]:


x = list(result.keys())
x.sort()
y = [result[_x] / len(bike_list22.values()) for _x in x]


# In[61]:


## TODO:画location，hour/Demand（平均），hour/CV(Demand) 
# 1, 25%, 50%, 75%, 100%------需求
# 按order的开始地点分组，统计每个地点的小时需求放到一个列表里，然后求一个平均需求，再求一个CV值。
# 
columns = bike2.order_data[0]
df = pd.DataFrame(bike2.order_data[1:],columns=columns)
df['t'] = pd.to_datetime(df['StartTime'])


# In[66]:


df['hour']= df['t'].tm_mhour


# In[46]:


groups = df.groupby('sid')


# In[79]:


### 每个小时都有哪些栅格有需求
_data = bike2.order_data
locations = {}
for item in _data[1:]:
    userid = item[1]
    bikeid = item[2]
    start_time = item[4]
    end_time = item[5]
    sdt = datetime.strptime(start_time, bike2.time_format)
    edt = datetime.strptime(end_time, bike2.time_format)
    sid = item[-2]
    eid = item[-1]
    if sid not in locations:
        locations[sid] = [[0]*24 for d in range(bike2.eday-bike2.sday+1)]
    locations[sid][sdt.day-bike2.sday][sdt.hour] +=1
#     print(locations[sid][sdt.hour])


# In[122]:


lkeys = locations.keys()
print(len((lkeys)))
dict1 = dict()
for key in lkeys:
    dict1[key] = np.sum(locations[key])
sorted_data = sorted(dict1.items(), key=lambda x: x[1], reverse=True)
total_keys = len(sorted_data)
keys_at_percentiles = {
    0:sorted_data[0][0],
    25: sorted_data[min(total_keys * 25 // 100, total_keys - 1)][0],
    50: sorted_data[min(total_keys * 50 // 100, total_keys - 1)][0],
    75: sorted_data[min(total_keys * 75 // 100, total_keys - 1)][0],
    100: sorted_data[min(total_keys * 100 // 100, total_keys - 1)][0]
}


# In[149]:


loc =keys_at_percentiles[0]
# print(len(locations[keys_at_percentiles[0]]))
mat = np.array(locations[loc])
print((np.sum(mat)))
# 平均需求
fig,axes = plt.subplots(1,2,figsize=(12,4))
mean_demand=np.mean(mat,axis=0)
cv=np.std(mat,axis=0)/np.mean(mat,axis=0)
print(cv)
axes[0].plot(mean_demand)
axes[0].set_xlabel('t', fontsize=14, family='Arial')
axes[0].set_ylabel('demand', fontsize=14, family='Arial')
axes[1].plot(cv)
axes[1].set_xlabel('t', fontsize=14, family='Arial')
axes[1].set_ylabel('CV', fontsize=14, family='Arial')
axes[1].set_ylim(0,1.7)


# In[145]:


loc =sorted_data[1][0]
# print(len(locations[keys_at_percentiles[0]]))
mat = np.array(locations[loc])
print((np.sum(mat)))
# 平均需求
fig,axes = plt.subplots(1,2,figsize=(12,4))
mean_demand=np.mean(mat,axis=0)
cv=np.std(mat,axis=0)/np.mean(mat,axis=0)
print(cv)
axes[0].plot(mean_demand)
axes[0].set_xlabel('t', fontsize=14, family='Arial')
axes[0].set_ylabel('demand', fontsize=14, family='Arial')
axes[1].plot(cv)
axes[1].set_xlabel('t', fontsize=14, family='Arial')
axes[1].set_ylabel('CV', fontsize=14, family='Arial')
axes[1].set_ylim(0,1.7)


# In[144]:


loc =sorted_data[2][0]
# print(len(locations[keys_at_percentiles[0]]))
mat = np.array(locations[loc])
print((np.sum(mat)))
# 平均需求
fig,axes = plt.subplots(1,2,figsize=(12,4))
mean_demand=np.mean(mat,axis=0)
cv=np.std(mat,axis=0)/np.mean(mat,axis=0)
print(cv)
axes[0].plot(mean_demand)
axes[0].set_xlabel('t', fontsize=14, family='Arial')
axes[0].set_ylabel('demand', fontsize=14, family='Arial')
axes[1].plot(cv)
axes[1].set_xlabel('t', fontsize=14, family='Arial')
axes[1].set_ylabel('CV', fontsize=14, family='Arial')
axes[1].set_ylim(0,1.7)


# In[148]:


loc =sorted_data[3][0]
# print(len(locations[keys_at_percentiles[0]]))
mat = np.array(locations[loc])
print((np.sum(mat)))
# 平均需求
fig,axes = plt.subplots(1,2,figsize=(12,4))
mean_demand=np.mean(mat,axis=0)
cv=np.std(mat,axis=0)/np.mean(mat,axis=0)
print(cv)
axes[0].plot(mean_demand)
axes[0].set_xlabel('t', fontsize=14, family='Arial')
axes[0].set_ylabel('demand', fontsize=14, family='Arial')
axes[1].plot(cv)
axes[1].set_xlabel('t', fontsize=14, family='Arial')
axes[1].set_ylabel('CV', fontsize=14, family='Arial')
axes[1].set_ylim(0,1.7)


# In[143]:


loc =sorted_data[5][0]
# print(len(locations[keys_at_percentiles[0]]))
mat = np.array(locations[loc])
print((np.sum(mat)))
# 平均需求
fig,axes = plt.subplots(1,2,figsize=(12,4))
mean_demand=np.mean(mat,axis=0)
cv=np.std(mat,axis=0)/np.mean(mat,axis=0)
print(cv)
axes[0].plot(mean_demand)
axes[0].set_xlabel('t', fontsize=14, family='Arial')
axes[0].set_ylabel('demand', fontsize=14, family='Arial')
axes[1].plot(cv)
axes[1].set_xlabel('t', fontsize=14, family='Arial')
axes[1].set_ylabel('CV', fontsize=14, family='Arial')


# In[153]:


df_sim = pd.DataFrame(bike2.order_data[1:])
df_sim.columns = bike2.order_data[0]
df_sim['sim_bike'] = bike_ids
df_sim.dropna()
selected_columns = ['BikeID','sim_bike', 'StartTime','EndTime']
select_df = df_sim[selected_columns]
select_df.to_csv('ls_sim.csv',index=False)

