#比较数据的指标
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy.optimize import curve_fit
from rtree import index
from math import radians, cos, sin, asin, sqrt
import time

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
    x_range=np.arange(0,100,step)
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

#骑行次数和平均距离
def trips_d(bikeid_random1,dis_all):
    dict1={}
    for i,j in zip(bikeid_random1,dis_all):
        if(np.isnan(i)==False):
            if i not in dict1.keys():
                dict1[i]=[j]
            else:
                dict1[i].append(j)
    bike_count1=[]
    bike_dis1=[]
    for key,value in dict1.items():
        bike_count1.append(len(value))
        bike_dis1.append(round(np.mean(value),3))
#    print(np.sum(bike_count1))
    count_full1=pd.Series(bike_count1).value_counts(normalize=True)
    dis_x1,dis_y1=pdf1(bike_dis1,1)
    trips_x1,trips_y1 = (list(t) for t in zip(*sorted(zip(count_full1.index,count_full1.values))))
    return trips_x1,trips_y1,dis_x1,dis_y1

#gy分布
def gyration(bikeid_random1,endx,endy):
    dict_gy={}
    for i,j,k in zip(bikeid_random1,endx,endy):
        if(np.isnan(i)==False):
            if i not in dict_gy.keys():
                dict_gy[i]=[[j,k]]
            else:
                dict_gy[i].append([j,k])
    bike_end=[]  
    for key,value in dict_gy.items():
        bike_point1_array=np.array(value)
        endx_mid=np.mean(bike_point1_array[:,0])
        endy_mid=np.mean(bike_point1_array[:,1])
        dis1=0
        for point1 in value:
            dis1=dis1+geodistance(point1[0],point1[1],endx_mid,endy_mid)    
        bike_end.append(dis1/len(value))
    bike_end=[no for no in bike_end if no<100 ]
    gy_x,gy_y=pdf1(bike_end,1)
    return gy_x,gy_y

#活跃天数
def act_days(bikeid,time1):
    dict2={}
    for i,j in zip(bikeid,time1):
        if(np.isnan(i)==False):
            if i not in dict2.keys():
                dict2[i]=[j.day]
            else:
                dict2[i].append(j.day)
    bike_time=[]   
    for key,value in dict2.items():
        bike_time.append(len(np.unique(value)))
    bbb=pd.Series(bike_time).value_counts(normalize=True)
    return bbb.index,bbb.values

#独特地点数
def uni_trips(bikeid,endx,endy,step):
    cut_true=26
    cut_random=20
    cut_com10_200=45
    cut_com10_law=32
    haha=[32253,76787,99079,117422]
    middle=[116.41667,39.91667]
    dict2={}
    for i,j,k in zip(bikeid,endx,endy):
        if(np.isnan(i)==False):
            if i not in dict2.keys():
                dict2[i]=[[j,k]]
            else:
                dict2[i].append([j,k])
    for key,value in dict2.items():
        value_new=[]
        for point1 in value:
            disx=geodistance(point1[0],point1[1],middle[0],point1[1]) 
            disy=geodistance(point1[0],point1[1],point1[0],middle[1])
            a=int(disx//step)
            b=int(disy//step)
            if(point1[0]>=middle[0]):
                a=a
            else:
                a=-a
            if(point1[1]>=middle[1]):
                b=b
            else:
                b=-b
            value_new.append((a,b))
        dict2[key]=value_new
    dict_all2={}    
    for key,value in dict2.items():
        trip=[]
        for m in range(len(value)):
            value_this=value[:m+1]
            trip.append(len(list(set(value_this))))
        if(len(trip)>100):
            trip=trip[:100]  
#            print(key)
        for i,j in zip(range(1,len(trip)+1),trip):
            if i not in dict_all2.keys():
                dict_all2[i]=[j]
            else:
                dict_all2[i].append(j)  
    all2=[]
    b_x=[]
#    bar2=[]
    for key,value in dict_all2.items():
        all2.append(np.mean(value))
#        bar2.append(np.std(value))
        b_x.append(key)
    return b_x,all2


#独特地点数和时间间隔
def uni_time(bikeid,endx,endy,step,time1):
    middle=[116.41667,39.91667]
    middle_nj=[118.78333,32.04695]
    middle_cd=[104.06667,30.66667]
    middle_xa=[108.95000,34.26667]
    middle_xm=[118.11022,24.490474]
    dict2={}
    dict_t={}
    for i,j,k,z in zip(bikeid,endx,endy,time1):
        if(np.isnan(i)==False):
            if i not in dict2.keys():
                dict2[i]=[[j,k]]
            else:
                dict2[i].append([j,k])
            if i not in dict_t.keys():
                dict_t[i]=[z]
            else:
                dict_t[i].append(z)
    for key,value in dict2.items():
        value_new=[]
        for point1 in value:
            disx=geodistance(point1[0],point1[1],middle[0],point1[1]) 
            disy=geodistance(point1[0],point1[1],point1[0],middle[1])
            a=int(disx//step)
            b=int(disy//step)
            if(point1[0]>=middle[0]):
                a=a
            else:
                a=-a
            if(point1[1]>=middle[1]):
                b=b
            else:
                b=-b
            value_new.append((a,b))
        dict2[key]=value_new
    u_all2=0  
    for key,value in dict_t.items():
        if(len(value)>1):
            u_all2=u_all2+1
            user_this=[]
            for m in range(1,len(value)):
                user_this.append(((time.mktime(value[m].timetuple())-time.mktime(value[0].timetuple()))/60)//60)
            dict_t[key]=user_this
    dict_all2={}    
    for key,value in dict2.items():
        if(len(value)>1):
            trip=[]
            for m in range(len(value)):
                value_this=value[:m+1]
                trip.append(len(list(set(value_this))))
            if(len(trip)>200):
                trip=trip[:200]  
#               print(key)
            for i,j in zip(dict_t[key],trip[1:]):
                if i not in dict_all2.keys():
                    dict_all2[i]=[j]
                else:
                    dict_all2[i].append(j)  
    all2=[]
    b_x=[]
#    bar2=[]
    for key,value in dict_all2.items():
        all2.append(np.mean(value))
#        all2.append(np.sum(value)/u_all2)
#        bar2.append(np.std(value))
        b_x.append(key)
    return b_x,all2


#rank
def rank(bikeid,endx,endy,step):
    middle=[116.41667,39.91667]
    dict2={}
    for i,j,k in zip(bikeid,endx,endy):
        if(np.isnan(i)==False):
            if i not in dict2.keys():
                dict2[i]=[[j,k]]
            else:
                dict2[i].append([j,k])
    for key,value in dict2.items():
        value_new=[]
        for point1 in value:
            disx=geodistance(point1[0],point1[1],middle[0],point1[1]) 
            disy=geodistance(point1[0],point1[1],point1[0],middle[1])
            a=int(disx//step)
            b=int(disy//step)
            if(point1[0]>=middle[0]):
                a=a
            else:
                a=-a
            if(point1[1]>=middle[1]):
                b=b
            else:
                b=-b
            value_new.append((a,b))
        dict2[key]=value_new
    dict_all2={}    
    for key,value in dict2.items():
        value2=pd.Series(value)
        bike_full=value2.value_counts(normalize=True)
        bike_count=np.array(bike_full.values)
    #    bike_count=bike_count/bike_count[0]
        for i,j in zip(range(1,len(bike_count)+1),bike_count):
            if i not in dict_all2.keys():
                dict_all2[i]=[j]
            else:
                dict_all2[i].append(j)  
    all2=[]
    b_x=[]
#    bar2=[]
    for key,value in dict_all2.items():
        all2.append(np.mean(value))
#        bar2.append(np.std(value))
        b_x.append(key)
    return b_x,all2    

#统计最长出行是第几次的分布  
def i_trip(bikeid,dis_all):
    dict1={}
    for i,j in zip(bikeid,dis_all):
        if(np.isnan(i)==False):
            if i not in dict1.keys():
                dict1[i]=[j]
            else:
                dict1[i].append(j)
    ith=[]
    for key,value in dict1.items():
        ith.append(value.index(np.max(value))+1)
    bike_full=pd.Series(ith).value_counts(normalize=True)
    x_bike,y_bike = (list(t) for t in zip(*sorted(zip(bike_full.index,bike_full.values))))
    return x_bike,y_bike


def log_pdf(x,y):
#    bins=np.logspace(0, 4, 20,base=2)
    bins=np.logspace(0, 3, 30)
    bins2=list(bins)
    bins_all={}
    for i in range(len(bins)-1):
        bins_all[bins[i]]=[]
    widths = (bins[1:] - bins[:-1])
    for i in range(len(x)):
        if(x[i]>=1):
            for j in range(len(bins)):
                if(x[i]<bins[j]):
                    bins_all[bins[j-1]].append(y[i])
                    break
                if(x[i]==bins[j]):
                    bins_all[bins[j]].append(y[i])
                    break
    x_new=[]
    y_new=[]
    for key,value in bins_all.items():
        if(len(value)>0):
#            index_this=bins2.index(key)
#            y_new.append(np.sum(value)/widths[index_this])
            y_new.append(np.mean(value))
            x_new.append(key)
        
    return x_new,y_new


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
#fig4	=	plt.figure(figsize=(8, 6))
#ax4	=	fig4.add_subplot(1,	1,	1)
#plt.yticks(fontproperties = 'Times New Roman')
#plt.xticks(fontproperties = 'Times New Roman') 
#plt.tick_params(labelsize=18)
###
#fig5	=	plt.figure(figsize=(8, 6))
#ax5	=	fig5.add_subplot(1,	1,	1)
#plt.yticks(fontproperties = 'Times New Roman')
#plt.xticks(fontproperties = 'Times New Roman') 
#plt.tick_params(labelsize=18)

#fig6	=	plt.figure(figsize=(8, 6))
#ax6	=	fig6.add_subplot(1,	1,	1)
#plt.yticks(fontproperties = 'Times New Roman')
#plt.xticks(fontproperties = 'Times New Roman') 
#plt.tick_params(labelsize=18)

#fig7	=	plt.figure(figsize=(8, 6))
#ax7	=	fig7.add_subplot(1,	1,	1)
#plt.yticks(fontproperties = 'Times New Roman')
#plt.xticks(fontproperties = 'Times New Roman') 
#plt.tick_params(labelsize=18)

data=pd.read_csv('C:/python/MOBIKE_CUP_2017/bj888_law200.csv')
time1 = pd.to_datetime(data['starttime'])
endx=data['end_x']
endy=data['end_y']
data2=pd.read_csv('C:/python/MOBIKE_CUP_2017/Mobike_bj2.csv')
dis_all=data2['d(m)']/1000
dict_all1={}
dict_all2={}
dict_all3={}
dict_all4={}
dict_all5={}
dict_all6={}
dict_all7={}
for kk in range(1,11,1):
#    name='bikeid_time_t'+str(kk)
    name='bikeid_com_t'+str(kk)
#    name='bikeid_new_t'+str(kk)
#    name='bikeid_near_t'+str(kk)
#    name='bikeid_random'+str(kk)
    bikeid_random1=data[name]
    trips_x,trips_y,dis_x1,dis_y1=trips_d(bikeid_random1,dis_all)    
    gy_x,gy_y=gyration(bikeid_random1,endx,endy)
#    days_x,days_y=act_days(bikeid_random1,time1)
#    unique_x,unique_y=uni_trips(bikeid_random1,endx,endy,1)
#    rank_x,rank_y=rank(bikeid_random1,endx,endy,1)
    ith_x,ith_y=i_trip(bikeid_random1,dis_all)
    for i,j in zip(trips_x,trips_y):
        if i not in dict_all1.keys():
            dict_all1[i]=[j]
        else:
            dict_all1[i].append(j) 
    for i,j in zip(dis_x1,dis_y1):
        if i not in dict_all2.keys():
            dict_all2[i]=[j]
        else:
            dict_all2[i].append(j)
    for i,j in zip(gy_x,gy_y):
        if i not in dict_all3.keys():
            dict_all3[i]=[j]
        else:
            dict_all3[i].append(j)
#    for i,j in zip(days_x,days_y):
#        if i not in dict_all4.keys():
#            dict_all4[i]=[j]
#        else:
#            dict_all4[i].append(j)
#    for i,j in zip(unique_x,unique_y):
#        if i not in dict_all5.keys():
#            dict_all5[i]=[j]
#        else:
#            dict_all5[i].append(j)
#    for i,j in zip(rank_x,rank_y):
#        if i not in dict_all6.keys():
#            dict_all6[i]=[j]
#        else:
#            dict_all6[i].append(j)
    for i,j in zip(ith_x,ith_y):
        if i not in dict_all7.keys():
            dict_all7[i]=[j]
        else:
            dict_all7[i].append(j)

all1=[]
all1_x=[]
for key,value in dict_all1.items():
    all1.append(np.sum(value)/10)
    all1_x.append(key)
all2=[]
all2_x=[]
for key,value in dict_all2.items():
    all2.append(np.sum(value)/10)
    all2_x.append(key)  
all3=[]
all3_x=[]
for key,value in dict_all3.items():
    all3.append(np.sum(value)/10)
    all3_x.append(key)
all4=[]
all4_x=[]
for key,value in dict_all4.items():
    all4.append(np.sum(value)/10)
    all4_x.append(key)
all5=[]
all5_x=[]
for key,value in dict_all5.items():
    all5.append(np.sum(value)/10)
    all5_x.append(key)
all6=[]
all6_x=[]
for key,value in dict_all6.items():
    all6.append(np.sum(value)/10)
    all6_x.append(key)
all7=[]
all7_x=[]
for key,value in dict_all7.items():
    all7.append(np.sum(value)/10)
    all7_x.append(key)
all1_x,all1 = (list(t) for t in zip(*sorted(zip(all1_x,all1)))) 
all2_x,all2 = (list(t) for t in zip(*sorted(zip(all2_x,all2))))
all3_x,all3 = (list(t) for t in zip(*sorted(zip(all3_x,all3))))
#all4_x,all4 = (list(t) for t in zip(*sorted(zip(all4_x,all4))))
#all5_x,all5 = (list(t) for t in zip(*sorted(zip(all5_x,all5))))
#all6_x,all6 = (list(t) for t in zip(*sorted(zip(all6_x,all6))))
all7_x,all7 = (list(t) for t in zip(*sorted(zip(all7_x,all7))))


print(all1_x)
print(all1)
print(all2_x)
print(all2)
print(all3_x)
print(all3)
print(all7_x)
print(all7)


#no=7
#ax1.scatter(all1_x,all1)
#print(all1_x)
#print(all1)
#ax1.set_yscale('log')
#ax1.set_xscale('log')
#ax1.set_ylim([0.0000001,1])
#ax1.legend(loc=3,handletextpad=0.1,prop={'size':20,'family':'Times New Roman'})
#ax1.set_xlabel('log'+r'$_{10}$'+'#trips',size=18,family='Times New Roman')  
#ax1.set_ylabel('log'+r'$_{10}P$(#)',size=18,family='Times New Roman')
#
#ax2.scatter(all2_x,all2)
#print(all2_x)
#print(all2)
#ax2.set_yscale('log')
#ax2.set_xscale('log')
#ax2.set_ylim([0.0000001,1])
#ax2.legend(loc=1,handletextpad=0.1,prop={'size':18,'family':'Times New Roman'})
#ax2.set_xlabel('log'+r'$_{10}\langle d \rangle$'+'(km)',size=18,family='Times New Roman') 
#ax2.set_ylabel('log'+r'$_{10}P(\langle d \rangle)$',size=18,family='Times New Roman')
#
#ax3.scatter(all3_x,all3)
#print(all3_x)
#print(all3)
#ax3.set_yscale('log')
#ax3.set_xscale('log')
#ax3.set_ylim([0.0000001,1])
#ax3.legend(loc=1,handletextpad=0.1,prop={'size':18,'family':'Times New Roman'})
#ax3.set_xlabel('log'+r'$_{10}r_g$'+'(km)',size=18,family='Times New Roman') 
#ax3.set_ylabel('log'+r'$_{10}P(r_g)$',size=18,family='Times New Roman')
#
#ax4.scatter(all4_x,all4)
#print(all4_x)
#print(all4)
#ax4.set_yscale('log')
#ax4.set_xscale('log')
#ax4.set_ylim([0.0000001,1])
#ax4.legend(loc=3,handletextpad=0.1,prop={'size':18,'family':'Times New Roman'})
#ax4.set_xlabel('#Active Days',size=18,family='Times New Roman') 
#ax4.set_ylabel(r'$P$(#)',size=18,family='Times New Roman')
###        
#ax5.scatter(all5_x,all5)
#print(all5_x)
#print(all5)
#ax5.legend(loc=2,handletextpad=0.1,prop={'size':18,'family':'Times New Roman'})
#ax5.set_xlabel('#trips',size=18,family='Times New Roman')  
#ax5.set_ylabel('#Unique Locations',size=18,family='Times New Roman')
               
#ax6.scatter(all6_x,all6)
#print(all6_x)
#print(all6)
#ax6.set_yscale('log')
#ax6.set_xscale('log')
#ax6.set_ylim([0.0001,1])
#ax6.legend(loc=3,handletextpad=0.1,prop={'size':18,'family':'Times New Roman'})
#ax6.set_xlabel('log'+r'$_{10}L$',size=18,family='Times New Roman') 
#ax6.set_ylabel('log'+r'$_{10}P(L)$',size=18,family='Times New Roman')  

#ax7.scatter(all7_x,all7)
#print(all7_x)
#print(all7)
#ax7.set_yscale('log')
#ax7.set_xscale('log')
#ax7.set_ylim([0.0000001,1])
##ax7.legend(loc=3,handletextpad=0.1,prop={'size':20,'family':'Times New Roman'})
#ax7.set_xlabel(r'i$^{th}$ trip',size=18,family='Times New Roman')  
#ax7.set_ylabel(r'$P(i)$',size=18,family='Times New Roman')





             