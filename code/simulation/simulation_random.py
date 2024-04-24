# 仿真，随机选车
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy.optimize import curve_fit


def pdf1(data, step):
    all_num = len(data)
    x_range = np.arange(0, 130, step)
    #    y=np.zeros((len(x_range)-1), dtype=np.int)
    y = np.zeros(len(x_range) - 1)
    x = x_range[:-1] + step / 2

    for data1 in data:
        a = int(data1 // step)
        y[a] = y[a] + 1
    y = y / all_num
    x1 = []
    y1 = []
    for i in range(len(x)):
        if (y[i] != 0):
            x1.append(x[i])
            y1.append(y[i])

    return x1, y1


def func1(x, mu, sigma):
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))


def func2(x, a, b):
    return a * pow(x, b)


data = pd.read_csv('../DATA/jinhua-statistic.csv')
bike_all = list(set(data['BikeID']))
print(len(bike_all))
for j in range(10):
    bikeid_random = []
    for i in range(len(data)):
        bike_this = random.choice(bike_all)
        bikeid_random.append(bike_this)
    name = 'bikeid_random' + str(j + 1)
    data[name] = bikeid_random
data.to_csv('jinhua/simulation1.csv', index=False)

# data=pd.read_csv('F:/python/MobikeData/lak222.csv')
# bikeid_random1=data['bikeid_random1']
# bikeid_random2=data['bikeid_random2']
# bikeid_random3=data['bikeid_random3']
# bikeid_random4=data['bikeid_random4']
# bikeid_random5=data['bikeid_random5']
# bikeid_random6=data['bikeid_random6']
# bikeid_random7=data['bikeid_random7']
# bikeid_random8=data['bikeid_random8']
# bikeid_random9=data['bikeid_random9']
# bikeid_random10=data['bikeid_random10']
# dis_all=data['d']
#
# dict1={}
# for i,j in zip(bikeid_random1,dis_all):
#    if i not in dict1.keys():
#        dict1[i]=[j]
#    else:
#        dict1[i].append(j)
# dict2={}
# for i,j in zip(bikeid_random2,dis_all):
#    if i not in dict2.keys():
#        dict2[i]=[j]
#    else:
#        dict2[i].append(j)
# dict3={}
# for i,j in zip(bikeid_random3,dis_all):
#    if i not in dict3.keys():
#        dict3[i]=[j]
#    else:
#        dict3[i].append(j)
# dict4={}
# for i,j in zip(bikeid_random4,dis_all):
#    if i not in dict4.keys():
#        dict4[i]=[j]
#    else:
#        dict4[i].append(j)
# dict5={}
# for i,j in zip(bikeid_random5,dis_all):
#    if i not in dict5.keys():
#        dict5[i]=[j]
#    else:
#        dict5[i].append(j)
# dict6={}
# for i,j in zip(bikeid_random6,dis_all):
#    if i not in dict6.keys():
#        dict6[i]=[j]
#    else:
#        dict6[i].append(j)
# dict7={}
# for i,j in zip(bikeid_random7,dis_all):
#    if i not in dict7.keys():
#        dict7[i]=[j]
#    else:
#        dict7[i].append(j)
# dict8={}
# for i,j in zip(bikeid_random8,dis_all):
#    if i not in dict8.keys():
#        dict8[i]=[j]
#    else:
#        dict8[i].append(j)
# dict9={}
# for i,j in zip(bikeid_random9,dis_all):
#    if i not in dict9.keys():
#        dict9[i]=[j]
#    else:
#        dict9[i].append(j)
# dict10={}
# for i,j in zip(bikeid_random10,dis_all):
#    if i not in dict10.keys():
#        dict10[i]=[j]
#    else:
#        dict10[i].append(j)
#
#        
# bike_count1=[]
# bike_dis1=[]
# for key,value in dict1.items():
#    bike_count1.append(len(value))
#    bike_dis1.append(round(np.mean(value),3))
# bike_count2=[]
# bike_dis2=[]
# for key,value in dict2.items():
#    bike_count2.append(len(value))
#    bike_dis2.append(round(np.mean(value),3))
# bike_count3=[]
# bike_dis3=[]
# for key,value in dict3.items():
#    bike_count3.append(len(value))
#    bike_dis3.append(round(np.mean(value),3))
# bike_count4=[]
# bike_dis4=[]
# for key,value in dict4.items():
#    bike_count4.append(len(value))
#    bike_dis4.append(round(np.mean(value),3))
# bike_count5=[]
# bike_dis5=[]
# for key,value in dict5.items():
#    bike_count5.append(len(value))
#    bike_dis5.append(round(np.mean(value),3))
# bike_count6=[]
# bike_dis6=[]
# for key,value in dict6.items():
#    bike_count6.append(len(value))
#    bike_dis6.append(round(np.mean(value),3))
# bike_count7=[]
# bike_dis7=[]
# for key,value in dict7.items():
#    bike_count7.append(len(value))
#    bike_dis7.append(round(np.mean(value),3))
# bike_count8=[]
# bike_dis8=[]
# for key,value in dict8.items():
#    bike_count8.append(len(value))
#    bike_dis8.append(round(np.mean(value),3))
# bike_count9=[]
# bike_dis9=[]
# for key,value in dict9.items():
#    bike_count9.append(len(value))
#    bike_dis9.append(round(np.mean(value),3))
# bike_count10=[]
# bike_dis10=[]
# for key,value in dict10.items():
#    bike_count10.append(len(value))
#    bike_dis10.append(round(np.mean(value),3))
#
#
# count_full1=pd.Series(bike_count1).value_counts(normalize=True)
# count_full2=pd.Series(bike_count2).value_counts(normalize=True)
# count_full3=pd.Series(bike_count3).value_counts(normalize=True)
# count_full4=pd.Series(bike_count4).value_counts(normalize=True)
# count_full5=pd.Series(bike_count5).value_counts(normalize=True)
# count_full6=pd.Series(bike_count6).value_counts(normalize=True)
# count_full7=pd.Series(bike_count7).value_counts(normalize=True)
# count_full8=pd.Series(bike_count8).value_counts(normalize=True)
# count_full9=pd.Series(bike_count9).value_counts(normalize=True)
# count_full10=pd.Series(bike_count10).value_counts(normalize=True)
# dis_x1,dis_y1=pdf1(bike_dis1,1)
# dis_x2,dis_y2=pdf1(bike_dis2,1)
# dis_x3,dis_y3=pdf1(bike_dis3,1)
# dis_x4,dis_y4=pdf1(bike_dis4,1)
# dis_x5,dis_y5=pdf1(bike_dis5,1)
# dis_x6,dis_y6=pdf1(bike_dis6,1)
# dis_x7,dis_y7=pdf1(bike_dis7,1)
# dis_x8,dis_y8=pdf1(bike_dis8,1)
# dis_x9,dis_y9=pdf1(bike_dis9,1)
# dis_x10,dis_y10=pdf1(bike_dis10,1)
#
#
# dict_all1={}
# for i,j in zip(count_full1.index,count_full1.values):
#    if i not in dict_all1.keys():
#        dict_all1[i]=[j]
#    else:
#        dict_all1[i].append(j)
# for i,j in zip(count_full2.index,count_full2.values):
#    if i not in dict_all1.keys():
#        dict_all1[i]=[j]
#    else:
#        dict_all1[i].append(j)
# for i,j in zip(count_full3.index,count_full3.values):
#    if i not in dict_all1.keys():
#        dict_all1[i]=[j]
#    else:
#        dict_all1[i].append(j)
# for i,j in zip(count_full4.index,count_full4.values):
#    if i not in dict_all1.keys():
#        dict_all1[i]=[j]
#    else:
#        dict_all1[i].append(j)
# for i,j in zip(count_full5.index,count_full5.values):
#    if i not in dict_all1.keys():
#        dict_all1[i]=[j]
#    else:
#        dict_all1[i].append(j)
# for i,j in zip(count_full6.index,count_full6.values):
#    if i not in dict_all1.keys():
#        dict_all1[i]=[j]
#    else:
#        dict_all1[i].append(j)        
# for i,j in zip(count_full7.index,count_full7.values):
#    if i not in dict_all1.keys():
#        dict_all1[i]=[j]
#    else:
#        dict_all1[i].append(j)
# for i,j in zip(count_full8.index,count_full8.values):
#    if i not in dict_all1.keys():
#        dict_all1[i]=[j]
#    else:
#        dict_all1[i].append(j)
# for i,j in zip(count_full9.index,count_full9.values):
#    if i not in dict_all1.keys():
#        dict_all1[i]=[j]
#    else:
#        dict_all1[i].append(j)
# for i,j in zip(count_full10.index,count_full10.values):
#    if i not in dict_all1.keys():
#        dict_all1[i]=[j]
#    else:
#        dict_all1[i].append(j)   
#
# dict_all2={}
# for i,j in zip(dis_x1,dis_y1):
#    if i not in dict_all2.keys():
#        dict_all2[i]=[j]
#    else:
#        dict_all2[i].append(j)
# for i,j in zip(dis_x2,dis_y2):
#    if i not in dict_all2.keys():
#        dict_all2[i]=[j]
#    else:
#        dict_all2[i].append(j)        
# for i,j in zip(dis_x3,dis_y3):
#    if i not in dict_all2.keys():
#        dict_all2[i]=[j]
#    else:
#        dict_all2[i].append(j)
# for i,j in zip(dis_x4,dis_y4):
#    if i not in dict_all2.keys():
#        dict_all2[i]=[j]
#    else:
#        dict_all2[i].append(j)         
# for i,j in zip(dis_x5,dis_y5):
#    if i not in dict_all2.keys():
#        dict_all2[i]=[j]
#    else:
#        dict_all2[i].append(j)
# for i,j in zip(dis_x6,dis_y6):
#    if i not in dict_all2.keys():
#        dict_all2[i]=[j]
#    else:
#        dict_all2[i].append(j) 
# for i,j in zip(dis_x7,dis_y7):
#    if i not in dict_all2.keys():
#        dict_all2[i]=[j]
#    else:
#        dict_all2[i].append(j)
# for i,j in zip(dis_x8,dis_y8):
#    if i not in dict_all2.keys():
#        dict_all2[i]=[j]
#    else:
#        dict_all2[i].append(j)
# for i,j in zip(dis_x9,dis_y9):
#    if i not in dict_all2.keys():
#        dict_all2[i]=[j]
#    else:
#        dict_all2[i].append(j)
# for i,j in zip(dis_x10,dis_y10):
#    if i not in dict_all2.keys():
#        dict_all2[i]=[j]
#    else:
#        dict_all2[i].append(j) 
#
#
# all1=[]
# all1_x=[]
# for key,value in dict_all1.items():
#    all1.append(np.sum(value)/10)
#    all1_x.append(key)
# all2=[]
# all2_x=[]
# for key,value in dict_all2.items():
#    all2.append(np.sum(value)/10)
#    all2_x.append(key)  
# all1_x,all1 = (list(t) for t in zip(*sorted(zip(all1_x,all1))))
# all2_x,all2 = (list(t) for t in zip(*sorted(zip(all2_x,all2))))
#
#      
# no=6
# fig	=	plt.figure(figsize=(8, 6))
# ax1	=	fig.add_subplot(1,	1,	1)
# ax1.scatter(all1_x,all1)
# print(all1_x)
# print(all1)
# popt, pcov = curve_fit(func1, all1_x[no:], all1[no:],maxfev = 10000)
# y777 = [func1(i, *popt) for i in all1_x]
# ax1.plot(all1_x,y777,color='blue',linestyle='-',label=r'$\mu$=%.2f, $\sigma$=%.2f'%(popt[0],popt[1]))
# ax1.set_yscale('log')
# ax1.set_xscale('log')
# ax1.set_ylim([0.0000001,1])
# ax1.legend(loc=3,handletextpad=0.1,prop={'size':20,'family':'Times New Roman'})
# ax1.set_xlabel('log'+r'$_{10}$'+'#trips',size=18,family='Times New Roman')
# ax1.set_ylabel('log'+r'$_{10}P$(#)',size=18,family='Times New Roman')
# plt.yticks(fontproperties = 'Times New Roman')
# plt.xticks(fontproperties = 'Times New Roman')
# plt.tick_params(labelsize=18)
#
#
# fig	=	plt.figure(figsize=(8, 6))
# ax2	=	fig.add_subplot(1,	1,	1)
# ax2.scatter(all2_x,all2)
# print(all2_x)
# print(all2)
# popt, pcov = curve_fit(func2, all2_x[10:-20], all2[10:-20],maxfev = 10000)
# y888 = [func2(i, *popt) for i in all2_x]
# ax2.plot(all2_x,y888,color='blue',linestyle='-',label=r'$\alpha$=%.2f'%(popt[1]))
# ax2.set_yscale('log')
# ax2.set_xscale('log')
# ax2.set_ylim([0.0000001,1])
# ax2.legend(loc=1,handletextpad=0.1,prop={'size':18,'family':'Times New Roman'})
# ax2.set_xlabel('log'+r'$_{10}\langle d \rangle$'+'(km)',size=18,family='Times New Roman')
# ax2.set_ylabel('log'+r'$_{10}P(\langle d \rangle)$',size=18,family='Times New Roman')
# plt.yticks(fontproperties = 'Times New Roman')
# plt.xticks(fontproperties = 'Times New Roman')
# plt.tick_params(labelsize=18)
