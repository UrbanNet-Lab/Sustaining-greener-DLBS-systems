import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from math import radians, cos, sin, asin, sqrt
import time
import math
import h5py



#data=pd.read_csv('C:/python/MOBIKE_CUP_2017/Mobike_bj2.csv')
#user_all=set(data['userid'])
#print(len(user_all))
#keyy=34000
#for i in range(1,6):
#    num=random.sample(user_all, keyy*i)
#    data_new=data[ data['userid'].isin(num) ]
#    file='C:/python/MOBIKE_CUP_2017/sample/sample_bj_ra'+str(i*10)+'.csv'
#    data_new.to_csv(file,index=False)

#data=pd.read_csv('C:/python/MOBIKE_CUP_2017/Mobike_bj2.csv')
#user_all=set(data['userid'])
#print(len(user_all))
#num=random.sample(user_all, 200000)
#data_new=data[ data['userid'].isin(num) ]
#file='C:/python/MOBIKE_CUP_2017/sample/sample_bj999.csv'
#data_new.to_csv(file,index=False)




#a="[(1,2),(2,2)]"
#print(eval(a))
#z2 = user1.intersection(user2,user3) 







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
        if i not in dict2.keys():
            dict2[i]=[j.day]
        else:
            dict2[i].append(j.day)
    bike_time=[]   
    for key,value in dict2.items():
        bike_time.append(len(np.unique(value)))
    bbb=pd.Series(bike_time).value_counts(normalize=True)
    return bbb.index,bbb.values


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


#独特地点数和时间间隔
def uni_time(bikeid,endx,endy,step,time1):
    middle=[116.41667,39.91667]
#    middle_nj=[118.78333,32.04695]
#    middle=[121.492,31.225]
#    middle=[104.06667,30.66667]
#    middle=[108.95000,34.26667]
    dict2={}
    dict_t={}
    for i,j,k,z in zip(bikeid,endx,endy,time1):
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
            new_x,new_y=log_pdf(dict_t[key],trip[1:])
#            new_x,new_y=dict_t[key],trip[1:]
            for i,j in zip(new_x,new_y):
                if i not in dict_all2.keys():
                    dict_all2[i]=[j]
                else:
                    dict_all2[i].append(j)  
    all2=[]
    b_x=[]
    bar2=[]
    for key,value in dict_all2.items():
        all2.append(np.mean(value))
#        all2.append(np.sum(value)/u_all2)
        bar2.append(np.std(value))
        b_x.append(key)
    b_x,all2,bar2 = (list(t) for t in zip(*sorted(zip(b_x,all2,bar2)))) 
    return b_x,all2,bar2


#rank
def rank(bikeid,endx,endy,step):
    middle=[116.41667,39.91667]
    dict2={}
    for i,j,k in zip(bikeid,endx,endy):
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
    bar2=[]
    for key,value in dict_all2.items():
        all2.append(np.mean(value))
        bar2.append(np.std(value))
        b_x.append(key)
    return b_x,all2,bar2   

#统计最长出行是第几次的分布  
def i_trip(bikeid,dis_all):
    dict1={}
    for i,j in zip(bikeid,dis_all):
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





#file='C:/python/MOBIKE_CUP_2017/sample/sample_bj.csv'
##file='C:/python/MOBIKE_CUP_2017/sample/sample_bj_ra50.csv'
##file='C:/python/MobikeData/lak3.csv'
#data=pd.read_csv(file)
##data_all['start_time2']=pd.to_datetime(data_all['start_time'])
#
##deadline3=pd.datetime(2016, 8, 15, 0, 0)
##data=data_all[data_all['start_time2']>=deadline3]
##time1 =data['start_time2']
#time1 = pd.to_datetime(data['starttime'])
#endx=data['end_x']
#endy=data['end_y']
#dis_all=data['d(m)']/1000
##dis_all=data['d']
#bikeid_random1=data['userid']
#bikeid_random2=data['bikeid']
##trips_x1,trips_y1,dis_x1,dis_y1=trips_d(bikeid_random1,dis_all)    
##gy_x1,gy_y1=gyration(bikeid_random1,endx,endy)
#unique_t_x1,unique_t_y1,unique_t_std1=uni_time(bikeid_random1,endx,endy,1,time1)
##unique_t_x1,unique_t_y1=log_pdf(unique_t_x1,unique_t_y1)
##rank_x1,rank_y1,rank_std1=rank(bikeid_random1,endx,endy,1)
##ith_x1,ith_y1=i_trip(bikeid_random1,dis_all)
#
#trips_x2,trips_y2,dis_x2,dis_y2=trips_d(bikeid_random2,dis_all)    
#gy_x2,gy_y2=gyration(bikeid_random2,endx,endy)
#unique_t_x2,unique_t_y2,unique_t_std2=uni_time(bikeid_random2,endx,endy,1,time1)
#unique_t_x2,unique_t_y2=log_pdf(unique_t_x2,unique_t_y2)
#rank_x2,rank_y2,rank_std2=rank(bikeid_random2,endx,endy,1)
#ith_x2,ith_y2=i_trip(bikeid_random2,dis_all)
#   
#
#print(unique_t_x1) 
#print(unique_t_y1) 
#print(rank_x1) 
#print(rank_y1) 
#print(ith_x1) 
#print(ith_y1) 
#print(trips_x1) 
#print(trips_y1) 
#print(gy_x1) 
#print(gy_y1) 
#print(dis_x1) 
#print(dis_y1) 
#
#
#f_family='Arial'
#size_x=8
#size_y=6
#alpha_xy=1
#fig	=	plt.figure(figsize=(size_x, size_y))
#ax1	=	fig.add_subplot(1,	1,	1)
#plt.yticks(fontproperties = f_family)
#plt.xticks(fontproperties = f_family) 
#plt.tick_params(labelsize=18)
#ax1.scatter(unique_t_x1,unique_t_y1,label=r'uesr',alpha=alpha_xy)
##ax1.scatter(unique_t_x2,unique_t_y2,label=r'bike',alpha=alpha_xy)
##ax1.errorbar(unique_t_x1,unique_t_y1,fmt="o",yerr = unique_t_std1,label='user',alpha=alpha_xy)
##ax1.errorbar(unique_t_x2,unique_t_y2,fmt="o",yerr = unique_t_std2,label='bike',alpha=alpha_xy)
##print(unique_t_x1)
##print(unique_t_y1)
##print(unique_t_std1)
##print(unique_t_x2)
##print(unique_t_y2)
##print(unique_t_std2)
#ax1.plot([24,24],[1,100],color='k',linestyle='--')
#ax1.plot([168,168],[1,100],color='k',linestyle='--')
#ax1.plot([720,720],[1,100],color='k',linestyle='--')
#ax1.text(19, 70,r'Day', size = 18,family=f_family)
#ax1.text(120, 70,r'Week', size = 18,family=f_family)
#ax1.text(400, 70,r'Month', size = 18,family=f_family)
#ax1.set_yscale('log')
#ax1.set_xscale('log')
#ax1.set_xlim([1,1000])
#ax1.set_ylim([1.6,100])
#ax1.legend(loc=2,handletextpad=0.1,prop={'size':18,'family':f_family})
#ax1.set_xlabel('t(h)',size=18,family=f_family) 
#ax1.set_ylabel(r'S(t)',size=18,family=f_family)
##plt.savefig('C:/python/摩拜单车/draw2/sf_4_nj1.pdf',bbox_inches='tight') 
#
#
#fig2	=	plt.figure(figsize=(size_x, size_y))
#ax2	=	fig2.add_subplot(1,	1,	1)
#plt.yticks(fontproperties = f_family)
#plt.xticks(fontproperties = f_family) 
#plt.tick_params(labelsize=18)
#ax2.scatter(rank_x1,rank_y1,label=r'uesr',alpha=alpha_xy)
##ax2.scatter(rank_x2,rank_y2,label=r'bike',alpha=alpha_xy)
##ax2.errorbar(rank_x1,rank_y1,fmt="o",yerr = rank_std1,label=r'uesr')
##ax2.errorbar(rank_x2,rank_y2,fmt="o",yerr = rank_std2,label=r'bike')
#ax2.set_yscale('log')
#ax2.set_xscale('log')
#ax2.set_ylim([0.001,1])
#ax2.legend(loc=3,handletextpad=0.1,prop={'size':18,'family':f_family})
#ax2.set_xlabel('log'+r'$_{10}L$',size=18,family=f_family) 
#ax2.set_ylabel('log'+r'$_{10}P(L)$',size=18,family=f_family) 
##plt.savefig('C:/python/摩拜单车/draw2/sf_4_nj2.pdf',bbox_inches='tight') 
#
#
#fig3	=	plt.figure(figsize=(size_x, size_y))
#ax3	=	fig3.add_subplot(1,	1,	1)
#plt.yticks(fontproperties = f_family)
#plt.xticks(fontproperties = f_family) 
#plt.tick_params(labelsize=18)
#ax3.scatter(ith_x1,ith_y1,label=r'user')
##ax3.scatter(ith_x2,ith_y2,label=r'bike')
##ax3.set_yscale('log')
##ax3.set_xscale('log')
##ax3.set_ylim([0.000001,1])
##ax3.set_xlim([0.5,150])
##ax3.legend(loc=3,handletextpad=0.1,prop={'size':18,'family':f_family})
#ax3.set_xlabel(r'i$^{th}$ trip',size=18,family=f_family)  
#ax3.set_ylabel(r'$P(i)$',size=18,family=f_family)
##plt.savefig('C:/python/摩拜单车/draw2/sf_4_nj3.pdf',bbox_inches='tight') 
#
#
#fig4	=	plt.figure(figsize=(size_x, size_y))
#ax4	=	fig4.add_subplot(1,	1,	1)
#plt.yticks(fontproperties = f_family)
#plt.xticks(fontproperties = f_family) 
#plt.tick_params(labelsize=18)
#ax4.scatter(trips_x1,trips_y1,label=r'user',alpha=alpha_xy)
##ax4.scatter(trips_x2,trips_y2,label=r'bike',alpha=alpha_xy)
#ax4.set_yscale('log')
#ax4.set_xscale('log')
#ax4.set_ylim([0.0000001,1])
#ax4.set_xlim([0.8,160])
#ax4.legend(loc=3,handletextpad=0.1,prop={'size':20,'family':f_family})
#ax4.set_xlabel('log'+r'$_{10}$'+'#trips',size=18,family=f_family)  
#ax4.set_ylabel('log'+r'$_{10}P$(#)',size=18,family=f_family)
##plt.savefig('C:/python/摩拜单车/draw2/sf_4_nj4.pdf',bbox_inches='tight') 
#
#
#fig5	=	plt.figure(figsize=(size_x, size_y))
#ax5	=	fig5.add_subplot(1,	1,	1)
#plt.yticks(fontproperties = f_family)
#plt.xticks(fontproperties = f_family) 
#plt.tick_params(labelsize=18)
#ax5.scatter(gy_x1,gy_y1,label=r'user',alpha=alpha_xy)
##ax5.scatter(gy_x2,gy_y2,label=r'bike')
#ax5.set_yscale('log')
#ax5.set_xscale('log')
#ax5.set_ylim([0.0000001,1])
#ax5.legend(loc=3,handletextpad=0.1,prop={'size':18,'family':f_family})
#ax5.set_xlabel('log'+r'$_{10}r_g$'+'(km)',size=18,family=f_family) 
#ax5.set_ylabel('log'+r'$_{10}P(r_g)$',size=18,family=f_family)
##plt.savefig('C:/python/摩拜单车/draw2/sf_4_nj5.pdf',bbox_inches='tight') 
#
#
#fig6	=	plt.figure(figsize=(size_x, size_y))
#ax6	=	fig6.add_subplot(1,	1,	1)
#plt.yticks(fontproperties = f_family)
#plt.xticks(fontproperties = f_family) 
#plt.tick_params(labelsize=18)
#ax6.scatter(dis_x1,dis_y1,label=r'user',alpha=alpha_xy)
##ax6.scatter(dis_x2,dis_y2,label=r'bike',alpha=alpha_xy)
#ax6.set_yscale('log')
#ax6.set_xscale('log')
#ax6.set_ylim([0.0000001,1])
#ax6.set_xlim([0.1,50])
#ax6.legend(loc=1,handletextpad=0.1,prop={'size':18,'family':f_family})
#ax6.set_xlabel('log'+r'$_{10}\langle d \rangle$'+'(km)',size=18,family=f_family) 
#ax6.set_ylabel('log'+r'$_{10}P(\langle d \rangle)$',size=18,family=f_family)
##plt.savefig('C:/python/摩拜单车/draw2/sf_4_nj6.pdf',bbox_inches='tight') 





