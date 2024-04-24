#仿真，随机从trip起点处选车,车真动
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy.optimize import curve_fit
from rtree import index
from math import radians, cos, sin, asin, sqrt

def geodistance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) 
    #lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2 
    dis=2*asin(sqrt(a))*6371*1000 # 地球平均半径，6371km
    dis=round(dis/1000,3)
    return dis

def pdf1(data,step):
    all_num=len(data)    
    x_range=np.arange(0,130,step)
#    y=np.zeros((len(x_range)-1), dtype=np.int)
    y=np.zeros(len(x_range)-1)
    x=x_range[:-1]+step/2
        
    for data1 in data:
        a=int(data1//step)        
        y[a]=y[a]+1
    y=y/all_num    
    x1=[]
    y1=[]
    for i in range(len(x)):
        if(y[i]!=0):
            x1.append(x[i])
            y1.append(y[i])
        
    return x1,y1

def func1(x,mu,sigma):
    return np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))

def func2(x, a, b):
    return a*pow(x,b)

def func3(x, a,b1,b2,c):
    return a*pow(x+c,-b1)*np.exp((-b2)*x)


lspath=r'../DATA/lishui-statistic.csv'
jhpath=r'../DATA/jinhua-statistic.csv'
nbpath=r'../DATA/ningbo-statistic.csv'
path=lspath
data=pd.read_csv(path)
rows=data.index
LSDR=0.0015
JHDR=0.003
NBDR=0.0025
DR=0.001
bike_all=[]
#初始位置
dict_init={}
for i,j,k in zip(data['BikeID'],data['StartLng'],data['StartLat']):
    if i not in dict_init.keys():
        dict_init[i]=[j,k] 
        bike_all.append(i)
    else:
        continue  
#idx_init = index.Index()  
#for key,value in dict_init.items():
#    idx_init.insert(key,[value[0],value[1],value[0],value[1]])   
deci=4                          
for j in range(10):
    print(j)
    idx = index.Index()
    for key,value in dict_init.items():
        idx.insert(key,[value[0],value[1],value[0],value[1]])
    dict_change=dict_init.copy()
    bikeid_near=[]
#    xx = list(idx.intersection( [120.8,30.6,122.1,31.6] ))
#    print(len(xx))
    for row in rows:
        limit_x=data.loc[row,'StartLng']
        limit_y=data.loc[row,'StartLat']
        intersecs = list(idx.intersection( [round(limit_x-DR,deci),round(limit_y-DR,deci),
                                            round(limit_x+DR,deci),round(limit_y+DR,deci)] ))
        if(len(intersecs)>0):
            bike_this=random.choice(intersecs)
            bikeid_near.append(bike_this)        
            old_x=dict_change[bike_this][0]
            old_y=dict_change[bike_this][1]
            new_x=data.loc[row,'EndLng']
            new_y=data.loc[row,'EndLat']
            idx.delete(bike_this,[old_x,old_y,old_x,old_y])
            idx.insert(bike_this,[new_x,new_y,new_x,new_y])
            dict_change[bike_this]=[new_x,new_y] 
        else:
            bikeid_near.append(np.nan)
    name='bikeid_near_t'+str(j+1)
    data[name]=bikeid_near
    
#    xx = list(idx.intersection( [120.8,30.6,122.1,31.6] ))
#    print(len(xx))
data.to_csv('lishui/simulation2.csv',index=False)



#data=pd.read_csv('F:/python/MobikeData/lak333.csv')
#fig	=	plt.figure(figsize=(8, 6))
#ax1	=	fig.add_subplot(1,	1,	1)
#plt.yticks(fontproperties = 'Times New Roman')
#plt.xticks(fontproperties = 'Times New Roman') 
#plt.tick_params(labelsize=18)
#
#fig2	=	plt.figure(figsize=(8, 6))
#ax2	=	fig2.add_subplot(1,	1,	1)
#plt.yticks(fontproperties = 'Times New Roman')
#plt.xticks(fontproperties = 'Times New Roman') 
#plt.tick_params(labelsize=18)
#
#fig3	=	plt.figure(figsize=(8, 6))
#ax3	=	fig3.add_subplot(1,	1,	1)
#plt.yticks(fontproperties = 'Times New Roman')
#plt.xticks(fontproperties = 'Times New Roman') 
#plt.tick_params(labelsize=18)
#
#dict_all1={}
#dict_all2={}
#dict_all3={}
#for kk in range(1,11,1):
#    name='bikeid_near_t'+str(kk)
##    name='bikeid_random'+str(kk)
#    bikeid_random1=data[name]
##    dis_all=data['d']
##    dict1={}
##    for i,j in zip(bikeid_random1,dis_all):
##        if(np.isnan(i)==False):
##            if i not in dict1.keys():
##                dict1[i]=[j]
##            else:
##                dict1[i].append(j)
##    bike_count1=[]
##    bike_dis1=[]
##    for key,value in dict1.items():
##        bike_count1.append(len(value))
##        bike_dis1.append(round(np.mean(value),3))
##    print(np.sum(bike_count1))
##    count_full1=pd.Series(bike_count1).value_counts(normalize=True)
##    dis_x1,dis_y1=pdf1(bike_dis1,1)
###    ax1.scatter(count_full1.index,count_full1.values,label=kk)
###    ax2.scatter(dis_x1,dis_y1,label=kk)
##    for i,j in zip(count_full1.index,count_full1.values):
##        if i not in dict_all1.keys():
##            dict_all1[i]=[j]
##        else:
##            dict_all1[i].append(j) 
##    for i,j in zip(dis_x1,dis_y1):
##        if i not in dict_all2.keys():
##            dict_all2[i]=[j]
##        else:
##            dict_all2[i].append(j)
#    endx=data['end_location_x']
#    endy=data['end_location_y']
#    dict_gy={}
#    for i,j,k in zip(bikeid_random1,endx,endy):
#        if i not in dict_gy.keys():
#            dict_gy[i]=[[j,k]]
#        else:
#            dict_gy[i].append([j,k])
#    bike_end=[]  
#    for key,value in dict_gy.items():
#        bike_point1_array=np.array(value)
#        endx_mid=np.mean(bike_point1_array[:,0])
#        endy_mid=np.mean(bike_point1_array[:,1])
#        dis1=0
#        for point1 in value:
#            dis1=dis1+geodistance(point1[0],point1[1],endx_mid,endy_mid)    
#        bike_end.append(dis1/len(value))
#    bike_end=[no for no in bike_end if no<100 ]
#    gy_x,gy_y=pdf1(bike_end,1)
#    for i,j in zip(gy_x,gy_y):
#        if i not in dict_all3.keys():
#            dict_all3[i]=[j]
#        else:
#            dict_all3[i].append(j)
#
#all1=[]
#all1_x=[]
#for key,value in dict_all1.items():
#    all1.append(np.sum(value)/10)
#    all1_x.append(key)
#all2=[]
#all2_x=[]
#for key,value in dict_all2.items():
#    all2.append(np.sum(value)/10)
#    all2_x.append(key)  
#all3=[]
#all3_x=[]
#for key,value in dict_all3.items():
#    all3.append(np.sum(value)/10)
#    all3_x.append(key)
##all1_x,all1 = (list(t) for t in zip(*sorted(zip(all1_x,all1)))) 
##all2_x,all2 = (list(t) for t in zip(*sorted(zip(all2_x,all2))))
#all3_x,all3 = (list(t) for t in zip(*sorted(zip(all3_x,all3))))
#
##no=7
##ax1.scatter(all1_x,all1)
###print(all1_x)
###print(all1)
##popt, pcov = curve_fit(func1, all1_x[no:], all1[no:],maxfev = 10000)
##y777 = [func1(i, *popt) for i in all1_x]
##ax1.plot(all1_x,y777,color='blue',linestyle='-',label=r'$\mu$=%.2f, $\sigma$=%.2f'%(popt[0],popt[1]))
##ax1.set_yscale('log')
##ax1.set_xscale('log')
##ax1.set_ylim([0.0000001,1])
##ax1.legend(loc=3,handletextpad=0.1,prop={'size':20,'family':'Times New Roman'})
##ax1.set_xlabel('log'+r'$_{10}$'+'#trips',size=18,family='Times New Roman')  
##ax1.set_ylabel('log'+r'$_{10}P$(#)',size=18,family='Times New Roman')
##
##ax2.scatter(all2_x,all2)
##popt, pcov = curve_fit(func2, all2_x[10:-30], all2[10:-30],maxfev = 10000)
##y888 = [func2(i, *popt) for i in all2_x]
###popt, pcov = curve_fit(func3, all2_x[3:-20], all2[3:-20],maxfev = 1000000)
###y888 = [func3(i, *popt) for i in all2_x]
##ax2.plot(all2_x,y888,color='blue',linestyle='-',label=r'$\alpha$=%.2f'%(popt[1]))
##ax2.set_yscale('log')
##ax2.set_xscale('log')
##ax2.set_ylim([0.0000001,1])
##ax2.legend(loc=1,handletextpad=0.1,prop={'size':18,'family':'Times New Roman'})
##ax2.set_xlabel('log'+r'$_{10}\langle d \rangle$'+'(km)',size=18,family='Times New Roman') 
##ax2.set_ylabel('log'+r'$_{10}P(\langle d \rangle)$',size=18,family='Times New Roman')
#
#ax3.scatter(all3_x,all3)
##print(all3_x)
##print(all3)
#ax3.set_yscale('log')
#ax3.set_xscale('log')
#ax3.set_ylim([0.0000001,1])
#ax3.legend(loc=1,handletextpad=0.1,prop={'size':18,'family':'Times New Roman'})
#ax3.set_xlabel('log'+r'$_{10}r_g$'+'(km)',size=18,family='Times New Roman') 
#ax3.set_ylabel('log'+r'$_{10}P(r_g)$',size=18,family='Times New Roman')
        



