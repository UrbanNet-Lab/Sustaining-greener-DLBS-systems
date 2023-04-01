#仿真，统计每次真实选的车在范围内的排名按车数分组
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy.optimize import curve_fit
from rtree import index
import math
from math import radians, cos, sin, asin, sqrt



#deadline1=pd.datetime(2016, 8, 10, 0, 0)   #星期三
#deadline2=pd.datetime(2016, 8, 11, 0, 0)
#data=pd.read_csv('C:/python/MobikeData/lak3.csv')
#data['time111'] = pd.to_datetime(data['start_time'])
#data_new=data[ (data['time111']>=deadline1) & (data['time111']<deadline2) ]
#data_ha=data_new[['orderid','userid','bikeid','start_time','end_time','start_location_x','start_location_y',
#                  'end_location_x','end_location_y','d']]
#data_ha.to_csv('C:/python/MobikeData/lak3_day1.csv',index=False)


nodes=10


#data=pd.read_csv('C:/python/MobikeData/dyn/dyn_sh.csv') 
#rows=data.index
#rank_all=[]
#for row in rows:
#    for tt in range(1,24):
#        time_start='t'+str(tt)
#        time_end='t'+str(tt+1)
#        rank_start=eval(data.loc[row,time_start])
#        rank_end=eval(data.loc[row,time_end])
#        rank_this=math.ceil(rank_start[2]/rank_start[3]*nodes)-1
#        rank_that=math.ceil(rank_end[2]/rank_end[3]*nodes)-1
#        rank_all.append((rank_this,rank_that))
#degree=np.zeros((nodes,nodes),dtype=int)
#rank_full=pd.Series(rank_all).value_counts()
#for i,j in zip(rank_full.index,rank_full.values):
#    degree[i[0]][i[1]]=j
#np.savetxt('C:/python/MobikeData/dyn/dyn_sh_degree_%d.txt'%nodes,degree,fmt='%d',delimiter=',')



c_r=10
angle=360/nodes
c_dict={}
for i in range(nodes):
    c_dict[i]=[c_r*cos((90-angle*i)/180*np.pi),c_r*sin((90-angle*i)/180*np.pi)]
c_dict_se={}
c_r_se=c_r
angle_se=7
for i in range(nodes):
    c_dict_se[i]=[(c_r_se*cos((90-angle*i-angle_se)/180*np.pi),c_r_se*sin((90-angle*i-angle_se)/180*np.pi)),
                  (c_r_se*cos((90-angle*i+angle_se)/180*np.pi),c_r_se*sin((90-angle*i+angle_se)/180*np.pi))]


degree_bj=np.loadtxt('C:/python/MobikeData/dyn/dyn_sh_degree_%d.txt'%nodes,dtype=int,delimiter=',')
degree_no=[]
degree_no_index=[]
degree_no_se=[]
for i in range(nodes):
    for j in range(nodes):
        if(i!=j):
            degree_no.append(degree_bj[i][j])
            degree_no_index.append((i,j))
        if(i==j):
            degree_no_se.append(degree_bj[i][j])
degree_no_new,degree_no_index_new = (list(t) for t in zip(*sorted(zip(degree_no,degree_no_index)))) 
line_num=int(len(degree_no_new)*0.3)
degree_good=degree_no_new[::-1][:line_num]
degree_index_good=degree_no_index_new[::-1][:line_num]
line_wid_max=5
degree_max=degree_good[0]

degree_max_se=max(degree_no_se)   

colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
        '#7f7f7f', '#bcbd22', '#17becf']
f_family='Arial'
fig=plt.figure(figsize=(8, 8))
ax=fig.add_subplot(1,1,1)

for i in range(nodes):
    for j in range(nodes):
        if(i!=j):
            if( (i,j) in degree_index_good): 
                plt.arrow(c_dict_se[i][0][0],c_dict_se[i][0][1],c_dict_se[j][1][0]-c_dict_se[i][0][0],c_dict_se[j][1][1]-c_dict_se[i][0][1],
                          length_includes_head = True,head_width=0.5,head_length=2, 
                          lw=degree_bj[i][j]/degree_max*line_wid_max,color=colors[0],alpha=degree_bj[i][j]/degree_max)
        if(i==j):
            ax.plot(c_dict[i][0],c_dict[i][1],marker='o',markersize=40,c=colors[0],
                    linewidth=0,markeredgewidth=0,alpha=degree_bj[i][j]/degree_max_se)   

xx=np.array(list(c_dict.values()))
#ax.plot(xx[:,0],xx[:,1],marker='o',markersize=30,markerfacecolor='none',c='g',markeredgewidth=2,linewidth=0)  #40
for i in range(nodes):
    if(i!=nodes-1):
        ax.text(xx[i][0]-0.7,xx[i][1]-0.4, int((i+1)*100/nodes),size=25,family=f_family,c='w')
    else:
        ax.text(xx[i][0]-1.1,xx[i][1]-0.4, int((i+1)*100/nodes),size=25,family=f_family,c='w')
lim_xy=12
ax.set_ylim([-lim_xy,lim_xy])
ax.set_xlim([-lim_xy,lim_xy])
plt.axis('off')
#plt.savefig('C:/python/摩拜单车/draw2/dyn_a.pdf',bbox_inches='tight') 







#c_r=10
#angle=360/nodes
##c_dict={0:[c_r*cos(angle*2/180*np.pi),c_r*sin(angle*2/180*np.pi)],1:[c_r*cos(angle*1/180*np.pi),c_r*sin(angle*1/180*np.pi)],
##           2:[c_r,0],3:[c_r*cos(angle*9/180*np.pi),c_r*sin(angle*9/180*np.pi)],
##           4:[c_r*cos(angle*8/180*np.pi),c_r*sin(angle*8/180*np.pi)],5:[c_r*cos(angle*7/180*np.pi),c_r*sin(angle*7/180*np.pi)],
##           6:[c_r*cos(angle*6/180*np.pi),c_r*sin(angle*6/180*np.pi)],7:[c_r*cos(angle*5/180*np.pi),c_r*sin(angle*5/180*np.pi)],
##           8:[c_r*cos(angle*4/180*np.pi),c_r*sin(angle*4/180*np.pi)],9:[c_r*cos(angle*3/180*np.pi),c_r*sin(angle*3/180*np.pi)]}
#c_dict={}
#for i in range(nodes):
#    c_dict[i]=[c_r*cos((90-angle*i)/180*np.pi),c_r*sin((90-angle*i)/180*np.pi)]
#c_dict_se={}
#c_r_se=c_r+1
#for i in range(nodes):
#    c_dict_se[i]=[c_r_se*cos((90-angle*i)/180*np.pi),c_r_se*sin((90-angle*i)/180*np.pi)]
#
#
#degree_bj=np.loadtxt('C:/python/MobikeData/dyn/dyn_sh_degree_%d.txt'%nodes,dtype=int,delimiter=',')
##degree_bj=np.loadtxt('C:/python/MobikeData/dyn/dyn_sh_degree_%d_200.txt'%nodes,dtype=int,delimiter=',')
#degree_no=[]
#degree_no_se=[]
#for i in range(nodes):
#    for j in range(nodes):
#        if(i!=j):
#            degree_no.append(degree_bj[i][j])
#        if(i==j):
#            degree_no_se.append(degree_bj[i][j])
##start=4000
##bins=16000
#line_wid_num=5
#start=min(degree_no)
#bins=math.ceil((max(degree_no)-start)/line_wid_num)+1
##d_group=[start,start+bins,start+bins*2,start+bins*3,start+bins*4,start+bins*5]
#d_group=[start-1]
#for i in range(1,line_wid_num+1):
#    d_group.append(d_group[i-1]+bins)
#    
#line_wid_num_se=5
#start_se=min(degree_no_se)
#bins_se=math.ceil((max(degree_no_se)-start_se)/line_wid_num_se)+1
#d_group_se=[start_se-1]
#for i in range(1,line_wid_num_se+1):
#    d_group_se.append(d_group_se[i-1]+bins_se)
#
#
#f_family='Arial'
#fig=plt.figure(figsize=(8, 8))
#ax=fig.add_subplot(1,1,1)
#for kk in range(1,len(d_group)):
#    for i in range(nodes):
#        for j in range(nodes):
#            if(i!=j):
#                for m in range(len(d_group)):
#                    if(degree_bj[i][j]<d_group[m]):
#                        line_wid=m
#                        break
#                if(line_wid==kk):
#                    if(line_wid>=3):
#                        if(i<j):
#                            plt.arrow(c_dict[i][0]-0.5,c_dict[i][1]-0.2,c_dict[j][0]-c_dict[i][0],c_dict[j][1]-c_dict[i][1],
#                                      length_includes_head = True,head_width=0.5,head_length=2, 
#                                      lw=line_wid,color='g',alpha=line_wid/line_wid_num)
#                        else:
#                            plt.arrow(c_dict[i][0]+0.5,c_dict[i][1]+0.2,c_dict[j][0]-c_dict[i][0],c_dict[j][1]-c_dict[i][1],
#                                      length_includes_head = True,head_width=0.5,head_length=2, 
#                                      lw=line_wid,color='g',alpha=line_wid/line_wid_num)
#            if(i==j):
#                for m in range(len(d_group_se)):
#                    if(degree_bj[i][j]<d_group_se[m]):
#                        line_wid=m
#                        break
#                if(line_wid==kk):
#                    if(line_wid>=1):
#                        ax.plot(c_dict_se[i][0],c_dict_se[i][1],marker='o',markersize=40,markerfacecolor='none',c='r',
#                                markeredgewidth=line_wid,linewidth=0,alpha=line_wid/line_wid_num)      #60
#
#
#xx=np.array(list(c_dict.values()))
#ax.plot(xx[:,0],xx[:,1],marker='o',markersize=30,markerfacecolor='none',c='g',markeredgewidth=2,linewidth=0)  #40
#for i in range(nodes):
#    ax.text(xx[i][0]-0.7,xx[i][1]-0.5, int((i+1)*100/nodes),size=30,family=f_family)
#lim_xy=13
#ax.set_ylim([-lim_xy,lim_xy])
#ax.set_xlim([-lim_xy,lim_xy])
#plt.axis('off')







#data=pd.read_csv('C:/python/MobikeData/lak3_day1.csv')
#rows=data.index
#
#DR1=0.001                                        
#bike_all=[]
##初始位置
#dict_init={}
#for i,j,k in zip(data['bikeid'],data['start_location_x'],data['start_location_y']):  
#    if i not in dict_init.keys():
#        dict_init[i]=[j,k] 
#        bike_all.append(i)
#    else:
#        continue 
#
##初始次数，时间        
#dict_init2={}
#dict_init3={}
#time_init=pd.to_datetime('2016/08/09 23:30:00')
#for i in bike_all:  
#    if i not in dict_init2.keys():
#        dict_init2[i]=1
#    if i not in dict_init3.keys():
#        dict_init3[i]=time_init
#        
#data['time111'] = pd.to_datetime(data['start_time'])
#data['time222'] = pd.to_datetime(data['end_time'])
#                                
#print(len(dict_init))
#
#idx = index.Index()
#for key,value in dict_init.items():
#    idx.insert(key,[value[0],value[1],value[0],value[1]])
#dict_change=dict_init.copy()
#dict_change2=dict_init2.copy()
#dict_change3=dict_init3.copy()
#t1,t2,t3,t4,t5,t6=[],[],[],[],[],[]
#t7,t8,t9,t10,t11,t12=[],[],[],[],[],[]
#t13,t14,t15,t16,t17,t18=[],[],[],[],[],[]
#t19,t20,t21,t22,t23,t24=[],[],[],[],[],[]
#deci=6
##    xx = list(idx.intersection( [120.8,30.6,122.1,31.6] ))
##    print(len(xx))
#flag_time=0
#for row in rows:
#    limit_x=data.loc[row,'start_location_x']
#    limit_y=data.loc[row,'start_location_y']
#    time_start=data.loc[row,'time111']
#    time_end=data.loc[row,'time222'] 
#    bike_true=data.loc[row,'bikeid']  
#    old_x=dict_change[bike_true][0]
#    old_y=dict_change[bike_true][1]
#    new_x=data.loc[row,'end_location_x']
#    new_y=data.loc[row,'end_location_y']
#    idx.delete(bike_true,[old_x,old_y,old_x,old_y])
#    idx.insert(bike_true,[limit_x,limit_y,limit_x,limit_y])  
#    dict_change[bike_true]=[limit_x,limit_y]
#    bike_this=bike_true
#    
#    if (flag_time==time_start.hour):
#        for bike_number in bike_all:            
#            number_x=dict_change[bike_number][0]
#            number_y=dict_change[bike_number][1]    
#            intersecs= list(idx.intersection( [round(number_x-DR1,deci),round(number_y-DR1,deci),
#                                               round(number_x+DR1,deci),round(number_y+DR1,deci)] ))         
#            intersecs_count=[]
#            intersecs_good=[] 
#            intersecs_time=[]
#            choose_time=[]
#            choose=[]
#            for intersecs_this in intersecs:
#                intersecs_count.append(dict_change2[intersecs_this]) 
#                intersecs_time.append(dict_change3[intersecs_this])
#            for time_this in intersecs_time:
#                choose_time.append( (time_start.day-time_this.day)*24*60+(time_start.hour-time_this.hour)*60+(time_start.minute-time_this.minute) )
#        #        choose_time.append( (time_start.month-time_this.month)*30*24*60+(time_start.day-time_this.day)*24*60+(time_start.hour-time_this.hour)*60+(time_start.minute-time_this.minute) )
#            for m,n in zip(intersecs_count,choose_time):
#                choose.append(m*n)
#            choose2,test_combine = (list(t) for t in zip(*sorted(zip(choose,intersecs))))                 
#            combine_this=choose2[test_combine.index(bike_number)]
#            start_rank2=choose2.index(combine_this)
#            end_rank2=start_rank2+choose2.count(combine_this)        
#            gg2=random.choice(range(start_rank2,end_rank2))+1   
#            res=(number_x,number_y,gg2,len(intersecs))
#            if(flag_time==0):
#                t1.append(res)
#            if(flag_time==1):
#                t2.append(res)
#            if(flag_time==2):
#                t3.append(res)
#            if(flag_time==3):
#                t4.append(res)
#            if(flag_time==4):
#                t5.append(res)
#            if(flag_time==5):
#                t6.append(res)
#            if(flag_time==6):
#                t7.append(res)
#            if(flag_time==7):
#                t8.append(res)
#            if(flag_time==8):
#                t9.append(res)
#            if(flag_time==9):
#                t10.append(res)
#            if(flag_time==10):
#                t11.append(res)
#            if(flag_time==11):
#                t12.append(res)
#            if(flag_time==12):
#                t13.append(res)
#            if(flag_time==13):
#                t14.append(res)
#            if(flag_time==14):
#                t15.append(res)
#            if(flag_time==15):
#                t16.append(res)
#            if(flag_time==16):
#                t17.append(res)
#            if(flag_time==17):
#                t18.append(res)
#            if(flag_time==18):
#                t19.append(res)
#            if(flag_time==19):
#                t20.append(res)
#            if(flag_time==20):
#                t21.append(res)
#            if(flag_time==21):
#                t22.append(res)
#            if(flag_time==22):
#                t23.append(res)
#            if(flag_time==23):
#                t24.append(res)
#            
#        flag_time=flag_time+1
#            
#        
#    idx.delete(bike_this,[limit_x,limit_y,limit_x,limit_y])
#    idx.insert(bike_this,[new_x,new_y,new_x,new_y])
#    dict_change[bike_this]=[new_x,new_y]          
#    dict_change2[bike_this]=dict_change2[bike_this]+1
#    dict_change3[bike_this]=time_end
#
#
#data_new=pd.DataFrame({"bikeid":bike_all,"t1":t1,"t2":t2,"t3":t3,"t4":t4,"t5":t5,
#                       "t6":t6,"t7":t7,"t8":t8,"t9":t9,"t10":t10,
#                       "t11":t11,"t12":t12,"t13":t13,"t14":t14,"t15":t15,
#                       "t16":t16,"t17":t17,"t18":t18,"t19":t19,"t20":t20,
#                       "t21":t21,"t22":t22,"t23":t23,"t24":t24})
#data_new.to_csv('C:/python/MobikeData/dyn/dyn_sh.csv',index=False)




