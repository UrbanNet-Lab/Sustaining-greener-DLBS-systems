# 仿真，利用次数（前10名）,车真动
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy.optimize import curve_fit
from rtree import index

#2
lspath = r'../DATA/lishui-statistic.csv'
jhpath = r'../DATA/jinhua-statistic.csv'
nbpath = r'../DATA/ningbo-statistic.csv'
#
path = lspath
data = pd.read_csv(path)
rows = data.index
LSDR = 0.0015
JHDR = 0.003
NBDR = 0.0025
DR = 0.001
bike_all = []
# 初始位置
dict_init = {}
for i, j, k in zip(data['BikeID'], data['StartLng'], data['StartLat']):
    if i not in dict_init.keys():
        dict_init[i] = [j, k]
        bike_all.append(i)
    else:
        continue

# 初始次数
dict_init2 = {}
for i in bike_all:
    if i not in dict_init2.keys():
        dict_init2[i] = 1

deci = 4
for j in range(10):
    print(j)
    idx = index.Index()
    for key, value in dict_init.items():
        idx.insert(key, [value[0], value[1], value[0], value[1]])
    dict_change = dict_init.copy()
    dict_change2 = dict_init2.copy()
    bikeid_near = []
    #    xx = list(idx.intersection( [120.8,30.6,122.1,31.6] ))
    #    print(len(xx))
    for row in rows:
        limit_x = data.loc[row, 'StartLng']
        limit_y = data.loc[row, 'StartLat']
        intersecs = list(idx.intersection([round(limit_x - DR, deci), round(limit_y - DR, deci),
                                           round(limit_x + DR, deci), round(limit_y + DR, deci)]))
        if (len(intersecs) > 0):
            intersecs_count = []
            intersecs_good = []
            for intersecs_this in intersecs:
                intersecs_count.append(dict_change2[intersecs_this])
            intersecs_count, intersecs = (list(t) for t in zip(*sorted(zip(intersecs_count, intersecs))))
            intersecs_good = intersecs[:10]

            bike_this = random.choice(intersecs_good)
            bikeid_near.append(bike_this)
            old_x = dict_change[bike_this][0]
            old_y = dict_change[bike_this][1]
            new_x = data.loc[row, 'EndLng']
            new_y = data.loc[row, 'EndLat']
            idx.delete(bike_this, [old_x, old_y, old_x, old_y])
            idx.insert(bike_this, [new_x, new_y, new_x, new_y])
            dict_change[bike_this] = [new_x, new_y]
            dict_change2[bike_this] = dict_change2[bike_this] + 1
        else:
            bikeid_near.append(np.nan)
    name = 'bikeid_new_t' + str(j + 1)
    data[name] = bikeid_near
# 3
data.to_csv('lishui/simulation3.csv', index=False)

# def pdf1(data,step):
#    all_num=len(data)    
#    x_range=np.arange(0,130,step)
##    y=np.zeros((len(x_range)-1), dtype=np.int)
#    y=np.zeros(len(x_range)-1)
#    x=x_range[:-1]+step/2
#        
#    for data1 in data:
#        a=int(data1//step)        
#        y[a]=y[a]+1
#    y=y/all_num    
#    x1=[]
#    y1=[]
#    for i in range(len(x)):
#        if(y[i]!=0):
#            x1.append(x[i])
#            y1.append(y[i])
#        
#    return x1,y1
#
# data=pd.read_csv('F:/python/MobikeData/nm2.csv')
# fig	=	plt.figure(figsize=(8, 6))
# ax1	=	fig.add_subplot(1,	1,	1)
# plt.yticks(fontproperties = 'Times New Roman')
# plt.xticks(fontproperties = 'Times New Roman')
# plt.tick_params(labelsize=18)
#
# fig2	=	plt.figure(figsize=(8, 6))
# ax2	=	fig2.add_subplot(1,	1,	1)
# plt.yticks(fontproperties = 'Times New Roman')
# plt.xticks(fontproperties = 'Times New Roman')
# plt.tick_params(labelsize=18)
#
# for kk in range(1,3,1):
##    name='bikeid'
##    name='bikeid_com_t'+str(kk)
#    name='bikeid_new_t'+str(kk)
##    name='bikeid_near_t'+str(kk)
##    name='bikeid_random'+str(kk)
#    bikeid_random1=data[name]
#    dis_all=data['d']
#    dict1={}
#    for i,j in zip(bikeid_random1,dis_all):
#        if(np.isnan(i)==False):
#            if i not in dict1.keys():
#                dict1[i]=[j]
#            else:
#                dict1[i].append(j)
#    bike_count1=[]
#    bike_dis1=[]
#    for key,value in dict1.items():
#        bike_count1.append(len(value))
#        bike_dis1.append(round(np.mean(value),3))
#    print(np.sum(bike_count1))
#    count_full1=pd.Series(bike_count1).value_counts(normalize=True)
#    dis_x1,dis_y1=pdf1(bike_dis1,1)
#    ax1.scatter(count_full1.index,count_full1.values,label=kk)
#    ax2.scatter(dis_x1,dis_y1,label=kk)
#
# ax1.set_yscale('log')
# ax1.set_xscale('log')
# ax1.set_ylim([0.0000001,1])
# ax1.legend(loc=3,handletextpad=0.1,prop={'size':10,'family':'Times New Roman'})
# ax1.set_xlabel('log'+r'$_{10}$'+'#trips',size=18,family='Times New Roman')
# ax1.set_ylabel('log'+r'$_{10}P$(#)',size=18,family='Times New Roman')
#
# ax2.set_yscale('log')
# ax2.set_xscale('log')
# ax2.set_ylim([0.0000001,1])
# ax2.legend(loc=1,handletextpad=0.1,prop={'size':10,'family':'Times New Roman'})
# ax2.set_xlabel('log'+r'$_{10}\langle d \rangle$'+'(km)',size=18,family='Times New Roman')
# ax2.set_ylabel('log'+r'$_{10}P(\langle d \rangle)$',size=18,family='Times New Roman')
