#仿真，利用间隔时间和次数（前10名）,车真动
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy.optimize import curve_fit
from rtree import index
from scipy import interpolate


def geodistance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) 
    #lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2 
    dis=2*asin(sqrt(a))*6371*1000 # 地球平均半径，6371km
    dis=round(dis/1000,3)
    return dis


#data=pd.read_csv('F:/python/MobikeData/lak3.csv')
#rows=data.index
#
##DR=0.002  
#DRS=np.arange(0.002,0.101,0.001)                                             
#data2=data[data['count_bike']==0]
##初始位置
#dict_init={}
#for i,j,k in zip(data2['bikeid'],data2['start_location_x'],data2['start_location_y']):  
#    if i not in dict_init.keys():
#        dict_init[i]=[j,k]  
#
##初始次数，时间        
#dict_init2={}
#dict_init3={}
#time_init=pd.to_datetime('2016/07/31 23:30')
#for i in data2['bikeid']:  
#    if i not in dict_init2.keys():
#        dict_init2[i]=1
#    if i not in dict_init3.keys():
#        dict_init3[i]=time_init
#        
#data['time111'] = pd.to_datetime(data['start_time'])
#data['time222'] = pd.to_datetime(data['end_time'])
#        
#                          
#for j in range(10):
#    print(j)
#    idx = index.Index()
#    for key,value in dict_init.items():
#        idx.insert(key,[value[0],value[1],value[0],value[1]])
#    dict_change=dict_init.copy()
#    dict_change2=dict_init2.copy()
#    dict_change3=dict_init3.copy()
#    bikeid_near=[]
#    bike_dis=[]
#    bike_dmin=[]
##    xx = list(idx.intersection( [120.8,30.6,122.1,31.6] ))
##    print(len(xx))
#    for row in rows:
#        limit_x=data.loc[row,'start_location_x']
#        limit_y=data.loc[row,'start_location_y']
#        time_start=data.loc[row,'time111']
#        time_end=data.loc[row,'time222']  
#        for DR in DRS:
#            intersecs = list(idx.intersection( [limit_x-DR,limit_y-DR,limit_x+DR,limit_y+DR] ))
#            if(len(intersecs)>0):
#                if(DR==0.002):
#                    intersecs_count=[]
#                    intersecs_good=[] 
#                    intersecs_time=[]
#                    choose_time=[]
#                    choose=[]
#                    intersecs_loc=[]
#                    dis_all=[]
#                    for intersecs_this in intersecs:
#                        intersecs_count.append(dict_change2[intersecs_this]) 
#                        intersecs_time.append(dict_change3[intersecs_this])
#                        intersecs_loc.append(dict_change[intersecs_this])
#                    for time_this in intersecs_time:
#                        choose_time.append( (time_start.month-time_this.month)*31*24+(time_start.day-time_this.day)*24+(time_start.hour-time_this.hour) )
#                    for m,n in zip(intersecs_count,choose_time):
#                        choose.append(m*n)
#                    choose,intersecs = (list(t) for t in zip(*sorted(zip(choose,intersecs))))
#                    intersecs_good=intersecs[:10]
#                            
#                    bike_this=random.choice(intersecs_good)
#                    bikeid_near.append(bike_this)        
#                    old_x=dict_change[bike_this][0]
#                    old_y=dict_change[bike_this][1]
#                    new_x=data.loc[row,'end_location_x']
#                    new_y=data.loc[row,'end_location_y']
#                    idx.delete(bike_this,[old_x,old_y,old_x,old_y])
#                    idx.insert(bike_this,[new_x,new_y,new_x,new_y])
#                    dict_change[bike_this]=[new_x,new_y] 
#                    dict_change2[bike_this]=dict_change2[bike_this]+1
#                    dict_change3[bike_this]=time_end
#                    dis=geodistance(limit_x,limit_y,old_x,old_y)
#                    bike_dis.append(dis)
#                    for loc_this in intersecs_loc:
#                        dis_all.append(geodistance(limit_x,limit_y,loc_this[0],loc_this[1]))
#                    bike_dmin.append(np.min(dis_all))
#                else:
#                    intersecs_loc=[]
#                    dis_all=[]
#                    for intersecs_this in intersecs:
#                        intersecs_loc.append(dict_change[intersecs_this])
#                    for loc_this in intersecs_loc:
#                        dis_all.append(geodistance(limit_x,limit_y,loc_this[0],loc_this[1]))
#                    dis_all,intersecs = (list(t) for t in zip(*sorted(zip(dis_all,intersecs))))
#                    
#                    bike_this=intersecs[0]
#                    bikeid_near.append(bike_this)        
#                    old_x=dict_change[bike_this][0]
#                    old_y=dict_change[bike_this][1]
#                    new_x=data.loc[row,'end_location_x']
#                    new_y=data.loc[row,'end_location_y']
#                    idx.delete(bike_this,[old_x,old_y,old_x,old_y])
#                    idx.insert(bike_this,[new_x,new_y,new_x,new_y])
#                    dict_change[bike_this]=[new_x,new_y] 
#                    dict_change2[bike_this]=dict_change2[bike_this]+1
#                    dict_change3[bike_this]=time_end
#                    bike_dis.append(dis_all[0])
#                    bike_dmin.append(dis_all[0])
#                break
#            else:
#                continue
#    name='bikeid_com_t'+str(j+1)
#    name2='dis'+str(j+1)
#    name3='dmin'+str(j+1)
#    data[name]=bikeid_near
#    data[name2]=bike_dis
#    data[name3]=bike_dmin
#
#    
#data_haha=data[['orderid','userid',	'bikeid','count_bike','start_time',
#               'end_time','start_location_x','start_location_y','end_location_x',
#               'end_location_y','d','bikeid_com_t1','bikeid_com_t2',
#               'bikeid_com_t3','bikeid_com_t4','bikeid_com_t5','bikeid_com_t6',
#               'bikeid_com_t7','bikeid_com_t8','bikeid_com_t9','bikeid_com_t10',
#               'dis1','dis2','dis3','dis4','dis5','dis6','dis7','dis8','dis9','dis10',
#               'dmin1','dmin2','dmin3','dmin4','dmin5','dmin6','dmin7','dmin8','dmin9','dmin10']]
#data_haha.to_csv('F:/python/MobikeData/lak888_dis.csv',index=False)






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

def pdf2(data,step):
    all_num=len(data)    
    x_range=np.arange(0,6000,step)
#    y=np.zeros((len(x_range)-1), dtype=np.int)
    y=np.zeros(len(x_range)-1)
    x=x_range[:-1]+step/2
        
    for data1 in data:
        a=int(data1//step)        
        y[a]=y[a]+1
#    y=y/all_num    
    x1=[]
    y1=[]
    for i in range(len(x)):
        if(y[i]!=0):
            x1.append(x[i])
            y1.append(y[i])
        
    return x1,y1


#data=pd.read_csv('C:/python/MobikeData/lak888_dis.csv')
#dict_all1={}
#dict_all2={}
#for kk in range(1,11,1):
#    name='dis'+str(kk)
#    name2='dmin'+str(kk)
##    name='bikeid_new_t'+str(kk)
##    name='bikeid_near_t'+str(kk)
##    name='bikeid_random'+str(kk)
#    bike_dis=[]
#    for this in data[name]:
#        if(np.isnan(this)==False):
#            bike_dis.append(this*1000)
#    bike_dmin=[]
#    for this in data[name2]:
#        if(np.isnan(this)==False):
#            bike_dmin.append(this*1000)
##    all_dis=pd.Series(bike_dis).value_counts()
##    ax3.scatter(all_dis.index,all_dis.values,label=kk)
##    dis_new,dis_p = (list(t) for t in zip(*sorted(zip(all_dis.index,all_dis.values))))
##    cdf_all=[0]
##    for i in range(len(dis_new)):
##        cdf_all.append(cdf_all[i]+dis_p[i])
##    cdf=cdf_all/cdf_all[-1]
##    for i,j in zip(dis_new,cdf[:-1]):
##        if i not in dict_all1.keys():
##             dict_all1[i]=[j]
##        else:
##            dict_all1[i].append(j)
#    dis_x1,dis_y1=pdf2(bike_dis,20)
#    cdf_all=[0]
#    for i in range(len(dis_x1)):
#        cdf_all.append(cdf_all[i]+dis_y1[i])
#    cdf=cdf_all/cdf_all[-1]
#    for i,j in zip(dis_x1,cdf[:-1]):
#        if i not in dict_all1.keys():
#             dict_all1[i]=[j]
#        else:
#            dict_all1[i].append(j)
##    ax3.scatter(dis_x1,cdf[:-1],label=kk)
#    dis_x2,dis_y2=pdf2(bike_dmin,20)
#    cdf_all2=[0]
#    for i in range(len(dis_x2)):
#        cdf_all2.append(cdf_all2[i]+dis_y2[i])
#    cdf2=cdf_all2/cdf_all2[-1]
#    for i,j in zip(dis_x2,cdf2[:-1]):
#        if i not in dict_all2.keys():
#             dict_all2[i]=[j]
#        else:
#            dict_all2[i].append(j)
##    ax3.scatter(dis_x2,cdf2[:-1],label=kk)
#all1=[]
#all1_x=[]
#x1=[]
#y1=[]
#for key,value in dict_all1.items():
#    all1.append(np.sum(value)/10)
#    all1_x.append(key)
#all1_x,all1 = (list(t) for t in zip(*sorted(zip(all1_x,all1))))
#for i in range(len(all1_x)):
#    if(all1_x[i]<600):
#        x1.append(all1_x[i])
#        y1.append(all1[i])
#all2=[]
#all2_x=[]
#x2=[]
#y2=[]
#for key,value in dict_all2.items():
#    all2.append(np.sum(value)/10)
#    all2_x.append(key)
#all2_x,all2 = (list(t) for t in zip(*sorted(zip(all2_x,all2))))
#for i in range(len(all2_x)):
#    if(all2_x[i]<600):
#        x2.append(all2_x[i])
#        y2.append(all2[i])


x1=[10.0, 90.0, 110.0, 150.0, 190.0, 230.0, 250.0, 290.0, 310.0, 330.0, 350.0, 370.0, 
    390.0, 430.0, 450.0, 470.0, 490.0, 510.0, 530.0, 550.0, 570.0, 590.0]
y1=[0.0, 0.21267720972826373, 0.3510835598291351, 0.4707034773709659, 0.6480509620445869, 
    0.7046844028044388, 0.8511309135950315, 0.9438579034805017, 0.9995517046377339, 
    0.9996324212841479, 0.99965697871233, 0.9996922005216742, 0.9997311401886716, 
    0.9997724279762918, 0.9997924848399462, 0.999823793114919, 0.9998262390739013, 
    0.9998428715949805, 0.9998521662391129, 0.999870755527378, 0.9998747669001091, 
    0.9998792674646364]

x2=[10.0, 90.0, 110.0, 150.0, 190.0, 230.0, 250.0, 290.0, 310.0, 330.0, 350.0, 370.0, 
    390.0, 430.0, 450.0, 470.0, 490.0, 510.0, 530.0, 550.0, 570.0, 590.0]
y2=[0.0, 0.9841198559036645, 0.9951496633382056, 0.9975955244820927, 0.9985304678434664, 
    0.9987372981350052, 0.9991452840932439, 0.9993208061098098, 0.9995517046377339, 
    0.9996324212841479, 0.99965697871233, 0.9996922005216742, 0.9997311401886716, 
    0.9997724279762918, 0.9997924848399462, 0.999823793114919, 0.9998262390739013, 
    0.9998428715949805, 0.9998521662391129, 0.999870755527378, 0.9998747669001091, 
    0.9998792674646364]

x_ha=np.arange(10,600,10)
f=interpolate.interp1d(x1,y1,kind="quadratic")
y_ha=f(x_ha) 

x_xi=np.arange(10,600,10)
f=interpolate.interp1d(x2,y2,kind="quadratic")
y_xi=f(x_xi)  

f_family='Arial'
fig3	=	plt.figure(figsize=(8, 6))
ax3	=	fig3.add_subplot(1,	1,	1)
plt.yticks(fontproperties = f_family)
plt.xticks(fontproperties = f_family) 
plt.tick_params(labelsize=18)

ax3.scatter(x1,y1,color='b')
ax3.plot(x_ha,y_ha,color='b',label='satisfactory')
ax3.scatter(x2,y2,color='orange')
ax3.plot(x_xi,y_xi,color='orange',label='nearest')
ax3.legend(loc=4,handletextpad=0.1,prop={'size':18,'family':f_family})
ax3.set_xlabel( 'd(m)',size=18,family=f_family) 
ax3.set_ylabel('P(d)',size=18,family=f_family) 
#plt.savefig('C:/python/摩拜单车/draw2/sf_10_a.pdf',bbox_inches='tight')



