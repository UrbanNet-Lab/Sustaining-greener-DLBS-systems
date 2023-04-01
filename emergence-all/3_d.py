#仿真，统计每次真实选的车的范围内有多少车，按每周分组，方法在simu_rank里
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy.optimize import curve_fit
from rtree import index



def log_pdf(x,y):
    bins=np.logspace(0, 4, 20)
    bins2=list(bins)
    bins_all={}
    for i in range(len(bins)-1):
        bins_all[bins[i]]=[]
    widths = (bins[1:] - bins[:-1])
    for i in range(len(x)):
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
            index_this=bins2.index(key)
            y_new.append(np.sum(value)/widths[index_this])
#            y_new.append(np.mean(value))
            x_new.append(key)
        
    return x_new,y_new


def func1(x, a,b1,b2,c):
    return a*pow(x+c,-b1)*np.exp((-b2)*x)

def func2(x, a, b):
    return a*pow(x,b)

def func3(x, a,b1,b2):
    return a*pow(x,-b1)*np.exp((-b2)*x)



#data=pd.read_csv('F:/python/MOBIKE_CUP_2017/Mobike_bj.csv')
#rows=data.index
#
#DR=0.002                                               
#bike_all=[]
##初始位置
#dict_init={}
#for i,j,k in zip(data['bikeid'],data['start_x'],data['start_y']):  
#    if i not in dict_init.keys():
#        dict_init[i]=[j,k] 
#        bike_all.append(i)
#    else:
#        continue 
#
#        
#                          
#print(len(dict_init))
#deadline1=pd.datetime(2017, 5, 10, 0, 0)
#deadline2=pd.datetime(2017, 5, 17, 0, 0)
#deadline3=pd.datetime(2017, 5, 25, 0, 0)
#data['time111'] = pd.to_datetime(data['starttime'])
#
#idx = index.Index()
#for key,value in dict_init.items():
#    idx.insert(key,[value[0],value[1],value[0],value[1]])
#dict_change=dict_init.copy()
#bikeid_near=[]
#sta_bikes=[]
#sta_bikes_1=[]
#sta_bikes_2=[]
##    xx = list(idx.intersection( [120.8,30.6,122.1,31.6] ))
##    print(len(xx))
#for row in rows:
#    limit_x=data.loc[row,'start_x']
#    limit_y=data.loc[row,'start_y']
#    time_this=data.loc[row,'time111']
#    bike_true=data.loc[row,'bikeid']  
#    old_x=dict_change[bike_true][0]
#    old_y=dict_change[bike_true][1]
#    new_x=data.loc[row,'end_x']
#    new_y=data.loc[row,'end_y']
#    idx.delete(bike_true,[old_x,old_y,old_x,old_y])
#    idx.insert(bike_true,[limit_x,limit_y,limit_x,limit_y])  
#    dict_change[bike_true]=[limit_x,limit_y]
#    intersecs = list(idx.intersection( [limit_x-DR,limit_y-DR,limit_x+DR,limit_y+DR] ))
##    if(len(intersecs)>800):
##        print(row)
#    if(len(intersecs)>0):
#        if(time_this>=deadline1 and time_this<deadline2):
#            sta_bikes_1.append(len(intersecs))
#        if(time_this>=deadline2 and time_this<deadline3):
#            sta_bikes_2.append(len(intersecs))
#        sta_bikes.append(len(intersecs))
#                
#        bike_this=bike_true
#              
#        idx.delete(bike_this,[limit_x,limit_y,limit_x,limit_y])
#        idx.insert(bike_this,[new_x,new_y,new_x,new_y])
#        dict_change[bike_this]=[new_x,new_y]          
##    else:
##        bikeid_near.append(np.nan)
#
#
##data_haha.to_csv('F:/python/MobikeData/lak888888.csv',index=False)
#
#
#sta111=pd.Series(sta_bikes_1).value_counts()
#sta222=pd.Series(sta_bikes_2).value_counts()
#
#x1,y1 = (list(t) for t in zip(*sorted(zip(list(sta111.index),list(sta111.values)))))
#x2,y2 = (list(t) for t in zip(*sorted(zip(list(sta222.index),list(sta222.values)))))
#np.savetxt('F:/python/MOBIKE_CUP_2017/near400_bikes_1.txt',np.array([x1,y1]).T,fmt='%d',delimiter=',')
#np.savetxt('F:/python/MOBIKE_CUP_2017/near400_bikes_2.txt',np.array([x2,y2]).T,fmt='%d',delimiter=',')






#100_150
rank_bj=np.loadtxt('C:/python/MOBIKE_CUP_2017/weeks/near400_bikes.txt',dtype=int,delimiter=',')
x_bj=rank_bj[:,0]
y_bj=rank_bj[:,1]/np.sum(rank_bj[:,1])
rank_bj1=np.loadtxt('C:/python/MOBIKE_CUP_2017/weeks/near400_bikes_1.txt',dtype=int,delimiter=',')
x_bj1=rank_bj1[:,0]
y_bj1=rank_bj1[:,1]/np.sum(rank_bj1[:,1])
rank_bj2=np.loadtxt('C:/python/MOBIKE_CUP_2017/weeks/near400_bikes_2.txt',dtype=int,delimiter=',')
x_bj2=rank_bj2[:,0]
y_bj2=rank_bj2[:,1]/np.sum(rank_bj2[:,1])
rank_sh=np.loadtxt('C:/python/MobikeData/weeks/near200_bikes.txt',dtype=int,delimiter=',')
x_sh=rank_sh[:,0]
y_sh=rank_sh[:,1]/np.sum(rank_sh[:,1])
rank_sh1=np.loadtxt('C:/python/MobikeData/weeks/near200_bikes_1.txt',dtype=int,delimiter=',')
x_sh1=rank_sh1[:,0]
y_sh1=rank_sh1[:,1]/np.sum(rank_sh1[:,1])
rank_sh2=np.loadtxt('C:/python/MobikeData/weeks/near200_bikes_2.txt',dtype=int,delimiter=',')
x_sh2=rank_sh2[:,0]
y_sh2=rank_sh2[:,1]/np.sum(rank_sh2[:,1])
rank_sh3=np.loadtxt('C:/python/MobikeData/weeks/near200_bikes_3.txt',dtype=int,delimiter=',')
x_sh3=rank_sh3[:,0]
y_sh3=rank_sh3[:,1]/np.sum(rank_sh3[:,1])
rank_sh4=np.loadtxt('C:/python/MobikeData/weeks/near200_bikes_4.txt',dtype=int,delimiter=',')
x_sh4=rank_sh4[:,0]
y_sh4=rank_sh4[:,1]/np.sum(rank_sh4[:,1])
rank_nj=np.loadtxt('C:/python/Nanjing2/weeks/near250_bikes.txt',dtype=int,delimiter=',')
x_nj=rank_nj[:,0]
y_nj=rank_nj[:,1]/np.sum(rank_nj[:,1])
rank_nj1=np.loadtxt('C:/python/Nanjing2/weeks/near250_bikes_1.txt',dtype=int,delimiter=',')
x_nj1=rank_nj1[:,0]
y_nj1=rank_nj1[:,1]/np.sum(rank_nj1[:,1])
rank_nj2=np.loadtxt('C:/python/Nanjing2/weeks/near250_bikes_2.txt',dtype=int,delimiter=',')
x_nj2=rank_nj2[:,0]
y_nj2=rank_nj2[:,1]/np.sum(rank_nj2[:,1])
rank_nj3=np.loadtxt('C:/python/Nanjing2/weeks/near250_bikes_3.txt',dtype=int,delimiter=',')
x_nj3=rank_nj3[:,0]
y_nj3=rank_nj3[:,1]/np.sum(rank_nj3[:,1])
rank_nj4=np.loadtxt('C:/python/Nanjing2/weeks/near250_bikes_4.txt',dtype=int,delimiter=',')
x_nj4=rank_nj4[:,0]
y_nj4=rank_nj4[:,1]/np.sum(rank_nj4[:,1])
data_nj_mo=pd.read_csv('C:/python/Nanjing/bikes_group_combine3.txt')
ha=data_nj_mo['bikes_350'].value_counts(normalize=True)
i_nj4,j_nj4 = (list(t) for t in zip(*sorted(zip(list(ha.index),list(ha.values)))))
data_cd3=pd.read_csv('C:/python/Mobai/Chengdu/bikes_group_combine.txt')
cd=data_cd3['bikes_150'].value_counts(normalize=True)
i_cd2,j_cd2 = (list(t) for t in zip(*sorted(zip(list(cd.index),list(cd.values)))))
data_xa3=pd.read_csv('C:/python/Mobai/Xian/bikes_group_combine2.txt')
xa=data_xa3['bikes_200'].value_counts(normalize=True)
i_xa2,j_xa2 = (list(t) for t in zip(*sorted(zip(list(xa.index),list(xa.values)))))
data_xm3=pd.read_csv('C:/python/Xiamen/bikes_group_combine.txt')
xm=data_xm3['bikes_150'].value_counts(normalize=True)
i_xm2,j_xm2 = (list(t) for t in zip(*sorted(zip(list(xm.index),list(xm.values)))))
data_sgp3=pd.read_csv('C:/python/Singapore/bikes_group_combine.txt')
sgp=data_sgp3['bikes_300'].value_counts(normalize=True)
i_sgp2,j_sgp2 = (list(t) for t in zip(*sorted(zip(list(sgp.index),list(sgp.values)))))


#50-75
#rank_bj=np.loadtxt('C:/python/MOBIKE_CUP_2017/weeks/near200_bikes.txt',dtype=int,delimiter=',')
#x_bj=rank_bj[:,0]
#y_bj=rank_bj[:,1]/np.sum(rank_bj[:,1])
#rank_bj1=np.loadtxt('C:/python/MOBIKE_CUP_2017/weeks/near200_bikes_1.txt',dtype=int,delimiter=',')
#x_bj1=rank_bj1[:,0]
#y_bj1=rank_bj1[:,1]/np.sum(rank_bj1[:,1])
#rank_bj2=np.loadtxt('C:/python/MOBIKE_CUP_2017/weeks/near200_bikes_2.txt',dtype=int,delimiter=',')
#x_bj2=rank_bj2[:,0]
#y_bj2=rank_bj2[:,1]/np.sum(rank_bj2[:,1])
#rank_sh=np.loadtxt('C:/python/MobikeData/weeks/near100_bikes.txt',dtype=int,delimiter=',')
#x_sh=rank_sh[:,0]
#y_sh=rank_sh[:,1]/np.sum(rank_sh[:,1])
#rank_sh1=np.loadtxt('C:/python/MobikeData/weeks/near100_bikes_1.txt',dtype=int,delimiter=',')
#x_sh1=rank_sh1[:,0]
#y_sh1=rank_sh1[:,1]/np.sum(rank_sh1[:,1])
#rank_sh2=np.loadtxt('C:/python/MobikeData/weeks/near100_bikes_2.txt',dtype=int,delimiter=',')
#x_sh2=rank_sh2[:,0]
#y_sh2=rank_sh2[:,1]/np.sum(rank_sh2[:,1])
#rank_sh3=np.loadtxt('C:/python/MobikeData/weeks/near100_bikes_3.txt',dtype=int,delimiter=',')
#x_sh3=rank_sh3[:,0]
#y_sh3=rank_sh3[:,1]/np.sum(rank_sh3[:,1])
#rank_sh4=np.loadtxt('C:/python/MobikeData/weeks/near100_bikes_4.txt',dtype=int,delimiter=',')
#x_sh4=rank_sh4[:,0]
#y_sh4=rank_sh4[:,1]/np.sum(rank_sh4[:,1])




x_bj,y_bj=log_pdf(x_bj,y_bj)
x_bj1,y_bj1=log_pdf(x_bj1,y_bj1)
x_bj2,y_bj2=log_pdf(x_bj2,y_bj2)
x_sh,y_sh=log_pdf(x_sh,y_sh)
x_sh1,y_sh1=log_pdf(x_sh1,y_sh1)
x_sh2,y_sh2=log_pdf(x_sh2,y_sh2)
x_sh3,y_sh3=log_pdf(x_sh3,y_sh3)
x_sh4,y_sh4=log_pdf(x_sh4,y_sh4)
x_nj,y_nj=log_pdf(x_nj,y_nj)
x_nj1,y_nj1=log_pdf(x_nj1,y_nj1)
x_nj2,y_nj2=log_pdf(x_nj2,y_nj2)
x_nj3,y_nj3=log_pdf(x_nj3,y_nj3)
x_nj4,y_nj4=log_pdf(x_nj4,y_nj4)
i_nj4,j_nj4=log_pdf(i_nj4,j_nj4)
i_cd2,j_cd2=log_pdf(i_cd2,j_cd2)
i_xa2,j_xa2=log_pdf(i_xa2,j_xa2)
i_xm2,j_xm2=log_pdf(i_xm2,j_xm2)
i_sgp2,j_sgp2=log_pdf(i_sgp2,j_sgp2)


f_family='Arial'
fig	=	plt.figure(figsize=(8, 6))
ax1	=	fig.add_subplot(1,	1,	1)
plt.yticks(fontproperties = f_family)
plt.xticks(fontproperties = f_family) 
plt.tick_params(labelsize=18)


alpha_xy=1
#ax1.scatter(x_sh,y_sh,alpha=alpha_xy,label='sh_all')
ax1.scatter(x_sh1,y_sh1,alpha=alpha_xy,label='D1 (1)')
ax1.scatter(x_sh2,y_sh2,alpha=alpha_xy,label='D1 (2)')
ax1.scatter(x_sh3,y_sh3,alpha=alpha_xy,label='D1 (3)')
ax1.scatter(x_sh4,y_sh4,alpha=alpha_xy,label='D1 (4)')
#ax1.scatter(x_bj,y_bj,alpha=alpha_xy,label='bj_all')
ax1.scatter(x_bj1,y_bj1,alpha=alpha_xy,label='D2 (1)')
ax1.scatter(x_bj2,y_bj2,alpha=alpha_xy,label='D2 (2)')
ax1.scatter(x_nj1,y_nj1,alpha=alpha_xy,label='D3 (1)')
ax1.scatter(x_nj2,y_nj2,alpha=alpha_xy,label='D3 (2)')
ax1.scatter(x_nj3,y_nj3,alpha=alpha_xy,label='D3 (3)')
ax1.scatter(x_nj4,y_nj4,alpha=alpha_xy,label='D3 (4)')
ax1.scatter(i_nj4,j_nj4,alpha=alpha_xy,label='D4')
ax1.scatter(i_cd2,j_cd2,alpha=alpha_xy,label='D5')
ax1.scatter(i_xa2,j_xa2,alpha=alpha_xy,label='D6')
ax1.scatter(i_xm2,j_xm2,alpha=alpha_xy,label='D7')
ax1.scatter(i_sgp2,j_sgp2,alpha=alpha_xy,label='D8')

ax1.legend(loc=3,handletextpad=0.1,ncol=3,columnspacing=1,prop={'size':18,'family':f_family})
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_ylim([0.00000001,1])
ax1.set_xlabel('#bikes',size=18,family=f_family)  
ax1.set_ylabel('$P$(#)',size=18,family=f_family)
#plt.savefig('C:/python/摩拜单车/draw2/3_d.pdf',bbox_inches='tight')  






