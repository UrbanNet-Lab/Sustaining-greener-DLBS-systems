#仿真，利用间隔时间和次数（前10名）,车真动
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy.optimize import curve_fit
from rtree import index



lspath = r'../DATA/lishui-statistic.csv'
jhpath = r'../DATA/jinhua-statistic.csv'
nbpath = r'../DATA/ningbo-statistic.csv'
path = nbpath
data = pd.read_csv(path)
rows = data.index
LSDR = 0.0015
JHDR = 0.003
NBDR = 0.0025
# 通过DR调搜索范围
DR = JHDR


bike_all=[]
#初始位置
dict_init={}
for i,j,k in zip(data['BikeID'],data['StartLng'],data['StartLat']):
    if i not in dict_init.keys():
        dict_init[i]=[j,k] 
        bike_all.append(i)
    else:
        continue

#初始次数，时间        
dict_init2={}
dict_init3={}
time_init=pd.to_datetime('2020-09-13 23:30:00')
for i in bike_all:  
    if i not in dict_init2.keys():
        dict_init2[i]=1
    if i not in dict_init3.keys():
        dict_init3[i]=time_init
        
data['time111'] = pd.to_datetime(data['StartTime'])
data['time222'] = pd.to_datetime(data['EndTime'])
        
deci=4                         
for j in range(10):
    print(j)
    idx = index.Index()
    for key,value in dict_init.items():
        idx.insert(key,[value[0],value[1],value[0],value[1]])
    dict_change=dict_init.copy()
    dict_change2=dict_init2.copy()
    dict_change3=dict_init3.copy()
    bikeid_near=[]
#    xx = list(idx.intersection( [120.8,30.6,122.1,31.6] ))
#    print(len(xx))
    for row in rows:
        limit_x=data.loc[row,'StartLng']
        limit_y=data.loc[row,'StartLat']
        time_start=data.loc[row,'time111']
        time_end=data.loc[row,'time222']        
        intersecs = list(idx.intersection( [round(limit_x-DR,deci),round(limit_y-DR,deci),
                                            round(limit_x+DR,deci),round(limit_y+DR,deci)] ))
        if(len(intersecs)>0):
            intersecs_count=[]
            intersecs_good=[] 
            intersecs_time=[]
            choose_time=[]
            choose=[]
            for intersecs_this in intersecs:
                intersecs_count.append(dict_change2[intersecs_this]) 
                intersecs_time.append(dict_change3[intersecs_this])
            for time_this in intersecs_time:
#                choose_time.append( (time_start.day-time_this.day)*24*60+(time_start.hour-time_this.hour)*60+(time_start.minute-time_this.minute) )
                choose_time.append( (time_start.month-time_this.month)*30*24*60+(time_start.day-time_this.day)*24*60+(time_start.hour-time_this.hour)*60+(time_start.minute-time_this.minute) )
            for m,n in zip(intersecs_count,choose_time):
                choose.append(m*n)
            choose,intersecs = (list(t) for t in zip(*sorted(zip(choose,intersecs))))
            intersecs_good=intersecs[:10]
                    
            bike_this=random.choice(intersecs_good)
            bikeid_near.append(bike_this)        
            old_x=dict_change[bike_this][0]
            old_y=dict_change[bike_this][1]
            new_x=data.loc[row,'EndLng']
            new_y=data.loc[row,'EndLat']
            idx.delete(bike_this,[old_x,old_y,old_x,old_y])
            idx.insert(bike_this,[new_x,new_y,new_x,new_y])
            dict_change[bike_this]=[new_x,new_y] 
            dict_change2[bike_this]=dict_change2[bike_this]+1
            dict_change3[bike_this]=time_end
        else:
            bikeid_near.append(np.nan)
    name='bikeid_com_t'+str(j+1)
    data[name]=bikeid_near

    
data_haha=data[['BikeID','StartTime','EndTime',
               'StartLng','StartLat','EndLng','EndLat','d(km)','duration',
               'bikeid_com_t1','bikeid_com_t2',
               'bikeid_com_t3','bikeid_com_t4','bikeid_com_t5','bikeid_com_t6',
               'bikeid_com_t7','bikeid_com_t8','bikeid_com_t9','bikeid_com_t10']]
#data_haha.to_csv('C:/python/Nanjing2/nj888.csv',index=False)
data_haha.to_csv('jinhua/simulation4.csv',index=False)




#def pdf1(data,step):
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
#data=pd.read_csv('F:/python/MobikeData/nmsl550.csv')
#fig	=	plt.figure(figsize=(8, 6))
#ax1	=	fig.add_subplot(1,	1,	1)
#plt.yticks(fontproperties = 'Times New Roman')
#plt.xticks(fontproperties = 'Times New Roman') 
#plt.tick_params(labelsize=18)
#
##fig2	=	plt.figure(figsize=(8, 6))
##ax2	=	fig2.add_subplot(1,	1,	1)
##plt.yticks(fontproperties = 'Times New Roman')
##plt.xticks(fontproperties = 'Times New Roman') 
##plt.tick_params(labelsize=18)
#
#true_trip_x=[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
#            18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
#            36, 35, 37, 39, 38, 42, 44, 40, 46, 41, 49, 43, 45, 47, 61, 54, 60,
#            56, 84, 52, 51, 50, 48, 98]
#true_trip=[3.86051606e-01, 2.00262070e-01 ,1.20402363e-01 ,7.75062910e-02,
# 5.34864467e-02, 3.74819093e-02 ,2.73413562e-02, 2.05972854e-02,
# 1.57665880e-02, 1.22299438e-02, 8.98992138e-03, 7.40902512e-03,
# 5.85094593e-03 ,4.80788036e-03 ,4.03210034e-03 ,3.06726469e-03,
# 2.53269359e-03, 2.18717812e-03 ,1.81884559e-03 ,1.39184062e-03,
# 1.17018919e-03, 9.32239853e-04 ,7.75780017e-04 ,6.77992620e-04,
# 5.44349844e-04, 4.33524127e-04 ,4.07447488e-04, 3.94409169e-04,
# 2.96621771e-04, 1.92315214e-04 ,1.72757735e-04, 1.40161936e-04,
# 8.80086574e-05 ,8.80086574e-05 ,6.84511780e-05, 6.19320182e-05,
# 5.54128584e-05 ,5.21532785e-05 ,3.58553790e-05, 2.60766392e-05,
# 2.60766392e-05 ,2.28170593e-05 ,1.95574794e-05, 1.95574794e-05,
# 1.62978995e-05 ,1.30383196e-05 ,1.30383196e-05, 9.77873972e-06,
# 3.25957991e-06 ,3.25957991e-06 ,3.25957991e-06, 3.25957991e-06,
# 3.25957991e-06 ,3.25957991e-06, 3.25957991e-06 ,3.25957991e-06,
# 3.25957991e-06 ,3.25957991e-06]
#
#for kk in range(1,2,1):
#    name='bikeid_com_t'+str(kk)
##    name='bikeid_new_t'+str(kk)
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
##    ax2.scatter(dis_x1,dis_y1,label=kk)
#
#ax1.scatter(true_trip_x,true_trip,label='true')
#ax1.set_yscale('log')
#ax1.set_xscale('log')
#ax1.set_ylim([0.0000001,1])
#ax1.legend(loc=3,handletextpad=0.1,prop={'size':10,'family':'Times New Roman'})
#ax1.set_xlabel('log'+r'$_{10}$'+'#trips',size=18,family='Times New Roman')  
#ax1.set_ylabel('log'+r'$_{10}P$(#)',size=18,family='Times New Roman')
#
##ax2.set_yscale('log')
##ax2.set_xscale('log')
##ax2.set_ylim([0.0000001,1])
##ax2.legend(loc=1,handletextpad=0.1,prop={'size':10,'family':'Times New Roman'})
##ax2.set_xlabel('log'+r'$_{10}\langle d \rangle$'+'(km)',size=18,family='Times New Roman') 
##ax2.set_ylabel('log'+r'$_{10}P(\langle d \rangle)$',size=18,family='Times New Roman')

        