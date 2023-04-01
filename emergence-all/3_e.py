#仿真，统计每次真实选的车在范围内的排名，按每周分组，方法在simu_rank里
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

def func2(x, a,b1,b2):
    return a*pow(x,-b1)*np.exp((-b2)*x)

def func3(x, a, b):
    return a*pow(x,b)



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
##初始次数，时间        
#dict_init2={}
#dict_init3={}
#time_init=pd.to_datetime('2017/05/09 23:30:00')
#for i in bike_all:  
#    if i not in dict_init2.keys():
#        dict_init2[i]=1
#    if i not in dict_init3.keys():
#        dict_init3[i]=time_init
#        
#data['time111'] = pd.to_datetime(data['starttime'])
#data['time222'] = pd.to_datetime(data['endtime'])
#deadline1=pd.datetime(2017, 5, 10, 0, 0)
#deadline2=pd.datetime(2017, 5, 17, 0, 0)
#deadline3=pd.datetime(2017, 5, 25, 0, 0)
#                                
#print(len(dict_init))
#
#idx = index.Index()
#for key,value in dict_init.items():
#    idx.insert(key,[value[0],value[1],value[0],value[1]])
#dict_change=dict_init.copy()
#dict_change2=dict_init2.copy()
#dict_change3=dict_init3.copy()
#bikeid_near=[]
#sta_count=[]
#sta_combine=[]
#sta_count_1=[]
#sta_combine_1=[]
#sta_count_2=[]
#sta_combine_2=[]
##    xx = list(idx.intersection( [120.8,30.6,122.1,31.6] ))
##    print(len(xx))
#for row in rows:
#    limit_x=data.loc[row,'start_x']
#    limit_y=data.loc[row,'start_y']
#    time_start=data.loc[row,'time111']
#    time_end=data.loc[row,'time222'] 
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
#        intersecs_count=[]
#        intersecs_good=[] 
#        intersecs_time=[]
#        choose_time=[]
#        choose=[]
#        for intersecs_this in intersecs:
#            intersecs_count.append(dict_change2[intersecs_this]) 
#            intersecs_time.append(dict_change3[intersecs_this])
#        for time_this in intersecs_time:
#            choose_time.append( (time_start.day-time_this.day)*24*60+(time_start.hour-time_this.hour)*60+(time_start.minute-time_this.minute) )
#        for m,n in zip(intersecs_count,choose_time):
#            choose.append(m*n)
#        intersecs_count2,test_count = (list(t) for t in zip(*sorted(zip(intersecs_count,intersecs))))
#        choose2,test_combine = (list(t) for t in zip(*sorted(zip(choose,intersecs))))
#
#                
#        bike_this=bike_true
#        
#        count_this=intersecs_count2[test_count.index(bike_this)]
#        combine_this=choose2[test_combine.index(bike_this)]
#        start_rank1=intersecs_count2.index(count_this)
#        end_rank1=start_rank1+intersecs_count2.count(count_this)
#        start_rank2=choose2.index(combine_this)
#        end_rank2=start_rank2+choose2.count(combine_this)        
##        sta_count.append(test_count.index(bike_this)+1) 
##        sta_combine.append(test_combine.index(bike_this)+1)
#        gg1=random.choice(range(start_rank1,end_rank1))+1
#        gg2=random.choice(range(start_rank2,end_rank2))+1
#        if(time_start>=deadline1 and time_start<deadline2):
#            sta_count_1.append(gg1)
#            sta_combine_1.append(gg2)
#        if(time_start>=deadline2 and time_start<deadline3):
#            sta_count_2.append(gg1)
#            sta_combine_2.append(gg2)
#        sta_count.append(gg1)
#        sta_combine.append(gg2)
#        
#        idx.delete(bike_this,[limit_x,limit_y,limit_x,limit_y])
#        idx.insert(bike_this,[new_x,new_y,new_x,new_y])
#        dict_change[bike_this]=[new_x,new_y]          
#        dict_change2[bike_this]=dict_change2[bike_this]+1
#        dict_change3[bike_this]=time_end
##    else:
##        bikeid_near.append(np.nan)
#
#
##data_haha.to_csv('F:/python/MobikeData/lak888888.csv',index=False)
#
#
#new_all=pd.Series(sta_count).value_counts()
#new111=pd.Series(sta_count_1).value_counts()
#new222=pd.Series(sta_count_2).value_counts()
#con_all=pd.Series(sta_combine).value_counts()
#con111=pd.Series(sta_combine_1).value_counts()
#con222=pd.Series(sta_combine_2).value_counts()
#
#
#x,y = (list(t) for t in zip(*sorted(zip(list(new_all.index),list(new_all.values)))))
#x1,y1 = (list(t) for t in zip(*sorted(zip(list(new111.index),list(new111.values)))))
#x2,y2 = (list(t) for t in zip(*sorted(zip(list(new222.index),list(new222.values)))))
#m,n = (list(t) for t in zip(*sorted(zip(list(con_all.index),list(con_all.values)))))
#m1,n1 = (list(t) for t in zip(*sorted(zip(list(con111.index),list(con111.values)))))
#m2,n2 = (list(t) for t in zip(*sorted(zip(list(con222.index),list(con222.values)))))
#np.savetxt('F:/python/MOBIKE_CUP_2017/weeks/rank400_count.txt',np.array([x,y]).T,fmt='%d',delimiter=',')
#np.savetxt('F:/python/MOBIKE_CUP_2017/weeks/rank400_count_1.txt',np.array([x1,y1]).T,fmt='%d',delimiter=',')
#np.savetxt('F:/python/MOBIKE_CUP_2017/weeks/rank400_count_2.txt',np.array([x2,y2]).T,fmt='%d',delimiter=',')
#np.savetxt('F:/python/MOBIKE_CUP_2017/weeks/rank400_combine.txt',np.array([m,n]).T,fmt='%d',delimiter=',')
#np.savetxt('F:/python/MOBIKE_CUP_2017/weeks/rank400_combine_1.txt',np.array([m1,n1]).T,fmt='%d',delimiter=',')
#np.savetxt('F:/python/MOBIKE_CUP_2017/weeks/rank400_combine_2.txt',np.array([m2,n2]).T,fmt='%d',delimiter=',')






#100_150
#count_sh=np.loadtxt('C:/python/MobikeData/weeks/rank200_count.txt',dtype=int,delimiter=',')
#x_sh=count_sh[:,0]
#y_sh=count_sh[:,1]/np.sum(count_sh[:,1])
#count_sh1=np.loadtxt('C:/python/MobikeData/weeks/rank200_count_1.txt',dtype=int,delimiter=',')
#x_sh1=count_sh1[:,0]
#y_sh1=count_sh1[:,1]/np.sum(count_sh1[:,1])
#count_sh2=np.loadtxt('C:/python/MobikeData/weeks/rank200_count_2.txt',dtype=int,delimiter=',')
#x_sh2=count_sh2[:,0]
#y_sh2=count_sh2[:,1]/np.sum(count_sh2[:,1])
#count_sh3=np.loadtxt('C:/python/MobikeData/weeks/rank200_count_3.txt',dtype=int,delimiter=',')
#x_sh3=count_sh3[:,0]
#y_sh3=count_sh3[:,1]/np.sum(count_sh3[:,1])
#count_sh4=np.loadtxt('C:/python/MobikeData/weeks/rank200_count_4.txt',dtype=int,delimiter=',')
#x_sh4=count_sh4[:,0]
#y_sh4=count_sh4[:,1]/np.sum(count_sh4[:,1])
#count_bj=np.loadtxt('C:/python/MOBIKE_CUP_2017/weeks/rank400_count.txt',dtype=int,delimiter=',')
#x_bj=count_bj[:,0]
#y_bj=count_bj[:,1]/np.sum(count_bj[:,1])
#count_bj1=np.loadtxt('C:/python/MOBIKE_CUP_2017/weeks/rank400_count_1.txt',dtype=int,delimiter=',')
#x_bj1=count_bj1[:,0]
#y_bj1=count_bj1[:,1]/np.sum(count_bj1[:,1])
#count_bj2=np.loadtxt('C:/python/MOBIKE_CUP_2017/weeks/rank400_count_2.txt',dtype=int,delimiter=',')
#x_bj2=count_bj2[:,0]
#y_bj2=count_bj2[:,1]/np.sum(count_bj2[:,1])

combine_sh=np.loadtxt('C:/python/MobikeData/weeks/rank200_combine.txt',dtype=int,delimiter=',')
m_sh=combine_sh[:,0]
n_sh=combine_sh[:,1]/np.sum(combine_sh[:,1])
combine_sh1=np.loadtxt('C:/python/MobikeData/weeks/rank200_combine_1.txt',dtype=int,delimiter=',')
m_sh1=combine_sh1[:,0]
n_sh1=combine_sh1[:,1]/np.sum(combine_sh1[:,1])
combine_sh2=np.loadtxt('C:/python/MobikeData/weeks/rank200_combine_2.txt',dtype=int,delimiter=',')
m_sh2=combine_sh2[:,0]
n_sh2=combine_sh2[:,1]/np.sum(combine_sh2[:,1])
combine_sh3=np.loadtxt('C:/python/MobikeData/weeks/rank200_combine_3.txt',dtype=int,delimiter=',')
m_sh3=combine_sh3[:,0]
n_sh3=combine_sh3[:,1]/np.sum(combine_sh3[:,1])
combine_sh4=np.loadtxt('C:/python/MobikeData/weeks/rank200_combine_4.txt',dtype=int,delimiter=',')
m_sh4=combine_sh4[:,0]
n_sh4=combine_sh4[:,1]/np.sum(combine_sh4[:,1])
combine_bj=np.loadtxt('C:/python/MOBIKE_CUP_2017/weeks/rank400_combine.txt',dtype=int,delimiter=',')
m_bj=combine_bj[:,0]
n_bj=combine_bj[:,1]/np.sum(combine_bj[:,1])
combine_bj1=np.loadtxt('C:/python/MOBIKE_CUP_2017/weeks/rank400_combine_1.txt',dtype=int,delimiter=',')
m_bj1=combine_bj1[:,0]
n_bj1=combine_bj1[:,1]/np.sum(combine_bj1[:,1])
combine_bj2=np.loadtxt('C:/python/MOBIKE_CUP_2017/weeks/rank400_combine_2.txt',dtype=int,delimiter=',')
m_bj2=combine_bj2[:,0]
n_bj2=combine_bj2[:,1]/np.sum(combine_bj2[:,1])
#combine_nj=np.loadtxt('C:/python/Nanjing2/weeks/rank250_combine.txt',dtype=int,delimiter=',')
#m_nj=combine_nj[:,0]
#n_nj=combine_nj[:,1]/np.sum(combine_nj[:,1])
data_nj2=pd.read_csv('C:/python/Nanjing2/bikes_group_combine3.txt')
ha=data_nj2['rank_250'].value_counts(normalize=True)
m_nj,n_nj = (list(t) for t in zip(*sorted(zip(list(ha.index),list(ha.values)))))
combine_nj1=np.loadtxt('C:/python/Nanjing2/weeks/rank250_combine_1.txt',dtype=int,delimiter=',')
m_nj1=combine_nj1[:,0]
n_nj1=combine_nj1[:,1]/np.sum(combine_nj1[:,1])
combine_nj2=np.loadtxt('C:/python/Nanjing2/weeks/rank250_combine_2.txt',dtype=int,delimiter=',')
m_nj2=combine_nj2[:,0]
n_nj2=combine_nj2[:,1]/np.sum(combine_nj2[:,1])
combine_nj3=np.loadtxt('C:/python/Nanjing2/weeks/rank250_combine_3.txt',dtype=int,delimiter=',')
m_nj3=combine_nj3[:,0]
n_nj3=combine_nj3[:,1]/np.sum(combine_nj3[:,1])
combine_nj4=np.loadtxt('C:/python/Nanjing2/weeks/rank250_combine_4.txt',dtype=int,delimiter=',')
m_nj4=combine_nj4[:,0]
n_nj4=combine_nj4[:,1]/np.sum(combine_nj4[:,1])
data_nj_mo=pd.read_csv('C:/python/Nanjing/bikes_group_combine3.txt')
haha=data_nj_mo['rank_350'].value_counts(normalize=True)
i_nj4,j_nj4 = (list(t) for t in zip(*sorted(zip(list(haha.index),list(haha.values)))))
data_cd3=pd.read_csv('C:/python/Mobai/Chengdu/bikes_group_combine.txt')
cd=data_cd3['rank_150'].value_counts(normalize=True)
i_cd2,j_cd2 = (list(t) for t in zip(*sorted(zip(list(cd.index),list(cd.values)))))
data_xa3=pd.read_csv('C:/python/Mobai/Xian/bikes_group_combine2.txt')
xa=data_xa3['rank_200'].value_counts(normalize=True)
i_xa2,j_xa2 = (list(t) for t in zip(*sorted(zip(list(xa.index),list(xa.values)))))
data_xm3=pd.read_csv('C:/python/Xiamen/bikes_group_combine.txt')
xm=data_xm3['rank_150'].value_counts(normalize=True)
i_xm2,j_xm2 = (list(t) for t in zip(*sorted(zip(list(xm.index),list(xm.values)))))
data_sgp3=pd.read_csv('C:/python/Singapore/bikes_group_combine.txt')
sgp=data_sgp3['rank_300'].value_counts(normalize=True)
i_sgp2,j_sgp2 = (list(t) for t in zip(*sorted(zip(list(sgp.index),list(sgp.values)))))


#50-75
#count_sh=np.loadtxt('C:/python/MobikeData/weeks/rank100_count.txt',dtype=int,delimiter=',')
#x_sh=count_sh[:,0]
#y_sh=count_sh[:,1]/np.sum(count_sh[:,1])
#count_sh1=np.loadtxt('C:/python/MobikeData/weeks/rank100_count_1.txt',dtype=int,delimiter=',')
#x_sh1=count_sh1[:,0]
#y_sh1=count_sh1[:,1]/np.sum(count_sh1[:,1])
#count_sh2=np.loadtxt('C:/python/MobikeData/weeks/rank100_count_2.txt',dtype=int,delimiter=',')
#x_sh2=count_sh2[:,0]
#y_sh2=count_sh2[:,1]/np.sum(count_sh2[:,1])
#count_sh3=np.loadtxt('C:/python/MobikeData/weeks/rank100_count_3.txt',dtype=int,delimiter=',')
#x_sh3=count_sh3[:,0]
#y_sh3=count_sh3[:,1]/np.sum(count_sh3[:,1])
#count_sh4=np.loadtxt('C:/python/MobikeData/weeks/rank100_count_4.txt',dtype=int,delimiter=',')
#x_sh4=count_sh4[:,0]
#y_sh4=count_sh4[:,1]/np.sum(count_sh4[:,1])
#count_bj=np.loadtxt('C:/python/MOBIKE_CUP_2017/weeks/rank200_count.txt',dtype=int,delimiter=',')
#x_bj=count_bj[:,0]
#y_bj=count_bj[:,1]/np.sum(count_bj[:,1])
#count_bj1=np.loadtxt('C:/python/MOBIKE_CUP_2017/weeks/rank200_count_1.txt',dtype=int,delimiter=',')
#x_bj1=count_bj1[:,0]
#y_bj1=count_bj1[:,1]/np.sum(count_bj1[:,1])
#count_bj2=np.loadtxt('C:/python/MOBIKE_CUP_2017/weeks/rank200_count_2.txt',dtype=int,delimiter=',')
#x_bj2=count_bj2[:,0]
#y_bj2=count_bj2[:,1]/np.sum(count_bj2[:,1])
#
#combine_sh=np.loadtxt('C:/python/MobikeData/weeks/rank100_combine.txt',dtype=int,delimiter=',')
#m_sh=combine_sh[:,0]
#n_sh=combine_sh[:,1]/np.sum(combine_sh[:,1])
#combine_sh1=np.loadtxt('C:/python/MobikeData/weeks/rank100_combine_1.txt',dtype=int,delimiter=',')
#m_sh1=combine_sh1[:,0]
#n_sh1=combine_sh1[:,1]/np.sum(combine_sh1[:,1])
#combine_sh2=np.loadtxt('C:/python/MobikeData/weeks/rank100_combine_2.txt',dtype=int,delimiter=',')
#m_sh2=combine_sh2[:,0]
#n_sh2=combine_sh2[:,1]/np.sum(combine_sh2[:,1])
#combine_sh3=np.loadtxt('C:/python/MobikeData/weeks/rank100_combine_3.txt',dtype=int,delimiter=',')
#m_sh3=combine_sh3[:,0]
#n_sh3=combine_sh3[:,1]/np.sum(combine_sh3[:,1])
#combine_sh4=np.loadtxt('C:/python/MobikeData/weeks/rank100_combine_4.txt',dtype=int,delimiter=',')
#m_sh4=combine_sh4[:,0]
#n_sh4=combine_sh4[:,1]/np.sum(combine_sh4[:,1])
#combine_bj=np.loadtxt('C:/python/MOBIKE_CUP_2017/weeks/rank200_combine.txt',dtype=int,delimiter=',')
#m_bj=combine_bj[:,0]
#n_bj=combine_bj[:,1]/np.sum(combine_bj[:,1])
#combine_bj1=np.loadtxt('C:/python/MOBIKE_CUP_2017/weeks/rank200_combine_1.txt',dtype=int,delimiter=',')
#m_bj1=combine_bj1[:,0]
#n_bj1=combine_bj1[:,1]/np.sum(combine_bj1[:,1])
#combine_bj2=np.loadtxt('C:/python/MOBIKE_CUP_2017/weeks/rank200_combine_2.txt',dtype=int,delimiter=',')
#m_bj2=combine_bj2[:,0]
#n_bj2=combine_bj2[:,1]/np.sum(combine_bj2[:,1])


#x_bj,y_bj=log_pdf(x_bj,y_bj)
#x_bj1,y_bj1=log_pdf(x_bj1,y_bj1)
#x_bj2,y_bj2=log_pdf(x_bj2,y_bj2)
#x_sh,y_sh=log_pdf(x_sh,y_sh)
#x_sh1,y_sh1=log_pdf(x_sh1,y_sh1)
#x_sh2,y_sh2=log_pdf(x_sh2,y_sh2)
#x_sh3,y_sh3=log_pdf(x_sh3,y_sh3)
#x_sh4,y_sh4=log_pdf(x_sh4,y_sh4)

m_bj,n_bj=log_pdf(m_bj,n_bj)
m_bj1,n_bj1=log_pdf(m_bj1,n_bj1)
m_bj2,n_bj2=log_pdf(m_bj2,n_bj2)
m_sh,n_sh=log_pdf(m_sh,n_sh)
m_sh1,n_sh1=log_pdf(m_sh1,n_sh1)
m_sh2,n_sh2=log_pdf(m_sh2,n_sh2)
m_sh3,n_sh3=log_pdf(m_sh3,n_sh3)
m_sh4,n_sh4=log_pdf(m_sh4,n_sh4)
m_nj,n_nj=log_pdf(m_nj,n_nj)
m_nj1,n_nj1=log_pdf(m_nj1,n_nj1)
m_nj2,n_nj2=log_pdf(m_nj2,n_nj2)
m_nj3,n_nj3=log_pdf(m_nj3,n_nj3)
m_nj4,n_nj4=log_pdf(m_nj4,n_nj4)
i_nj4,j_nj4=log_pdf(i_nj4,j_nj4)
i_cd2,j_cd2=log_pdf(i_cd2,j_cd2)
i_xa2,j_xa2=log_pdf(i_xa2,j_xa2)
i_xm2,j_xm2=log_pdf(i_xm2,j_xm2)
i_sgp2,j_sgp2=log_pdf(i_sgp2,j_sgp2)


f_family='Arial'
#fig	=	plt.figure(figsize=(8, 6))
#ax1	=	fig.add_subplot(1,	1,	1)
#plt.yticks(fontproperties = f_family)
#plt.xticks(fontproperties = f_family) 
#plt.tick_params(labelsize=18)

fig2	=	plt.figure(figsize=(8, 6))
ax2	=	fig2.add_subplot(1,	1,	1)
plt.yticks(fontproperties = f_family)
plt.xticks(fontproperties = f_family) 
plt.tick_params(labelsize=18)

alpha_gg=1   
#popt, pcov = curve_fit(func2, m_sh[4:],n_sh[4:],maxfev = 10000)        #50-75
popt, pcov = curve_fit(func2, m_sh[1:],n_sh[1:],maxfev = 10000)        #100_150
y888 = [func2(i, *popt) for i in m_sh] 
ax2.scatter(m_sh,n_sh,label=r'D1',alpha=alpha_gg)
ax2.plot(m_sh,y888,color='b',linestyle='-',label=r'$\alpha$=%.2f'%popt[1])
ax2.scatter(m_sh1,n_sh1,label='D1 (1)',alpha=alpha_gg)
ax2.scatter(m_sh2,n_sh2,label='D1 (2)',alpha=alpha_gg)
ax2.scatter(m_sh3,n_sh3,label='D1 (3)',alpha=alpha_gg)
ax2.scatter(m_sh4,n_sh4,label='D1 (4)',alpha=alpha_gg)
ax2.scatter(m_bj,n_bj,label=r'D2',alpha=alpha_gg)
ax2.scatter(m_bj1,n_bj1,label='D2 (1)',alpha=alpha_gg)
ax2.scatter(m_bj2,n_bj2,label='D2 (2)',alpha=alpha_gg)
ax2.scatter(m_nj,n_nj,label=r'D3',alpha=alpha_gg)
ax2.scatter(m_nj1,n_nj1,label='D3 (1)',alpha=alpha_gg)
ax2.scatter(m_nj2,n_nj2,label='D3 (2)',alpha=alpha_gg)
ax2.scatter(m_nj3,n_nj3,label='D3 (3)',alpha=alpha_gg)
ax2.scatter(m_nj4,n_nj4,label='D3 (4)',alpha=alpha_gg)
ax2.scatter(i_nj4,j_nj4,alpha=alpha_gg,label='D4')
ax2.scatter(i_cd2,j_cd2,alpha=alpha_gg,label='D5')
ax2.scatter(i_xa2,j_xa2,alpha=alpha_gg,label='D6')
ax2.scatter(i_xm2,j_xm2,alpha=alpha_gg,label='D7')
ax2.scatter(i_sgp2,j_sgp2,alpha=alpha_gg,label='D8')
ax2.legend(loc=3,handletextpad=0.1,ncol=3,columnspacing=1,prop={'size':18,'family':f_family})
ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.set_ylim([0.00000001,1])
ax2.set_xlabel(r'rank(times$^{-1}$*$\Delta$t$^{-1}$)',size=18,family=f_family)  
ax2.set_ylabel(r'$P$(rank)',size=18,family=f_family)
#plt.savefig('C:/python/摩拜单车/draw2/3_e.pdf',bbox_inches='tight')  






#rank_all=np.loadtxt('F:/python/MOBIKE_CUP_2017/rank200_count.txt',dtype=int,delimiter=',')
#rank_x=rank_all[:,0]
#rank_y=rank_all[:,1]/np.sum(rank_all[:,1])
#rank_sh=np.loadtxt('F:/python/MobikeData/rank200_count.txt',dtype=int,delimiter=',')
#x_sh=rank_sh[:,0]
#y_sh=rank_sh[:,1]/np.sum(rank_sh[:,1])
#
#rank_all2=np.loadtxt('F:/python/MOBIKE_CUP_2017/rank200_combine.txt',dtype=int,delimiter=',')
#rank_x2=rank_all2[:,0]
#rank_y2=rank_all2[:,1]/np.sum(rank_all2[:,1])
#rank_sh2=np.loadtxt('F:/python/MobikeData/rank200_combine.txt',dtype=int,delimiter=',')
#x_sh2=rank_sh2[:,0]
#y_sh2=rank_sh2[:,1]/np.sum(rank_sh2[:,1])
#
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
#
#cut1=20
#cut2=100
#no=16
#popt1, pcov1 = curve_fit(func2, rank_x[:cut1],rank_y[:cut1],maxfev = 10000)
#popt2, pcov2 = curve_fit(func2, rank_x[cut2:],rank_y[cut2:],maxfev = 10000)
#y777 = [func2(i, *popt1) for i in rank_x[:cut1]]
#y888 = [func2(i, *popt2) for i in rank_x[cut1:]]
#ax1.scatter(rank_x,rank_y,alpha=0.2)
#ax1.plot(rank_x[:cut1],y777,color='blue',linewidth = 3,linestyle='-',label=r'bj1, $\alpha$=%.2f'%popt1[1])
#ax1.plot(rank_x[cut1:],y888,color='blue',linewidth = 3,linestyle='-',label=r'bj2, $\alpha$=%.2f'%popt2[1])
#popt1, pcov1 = curve_fit(func2, x_sh[:cut2],y_sh[:cut2],maxfev = 10000)
#popt2, pcov2 = curve_fit(func2, x_sh[cut2:],y_sh[cut2:],maxfev = 10000)
#y777 = [func2(i, *popt1) for i in x_sh[:cut2]]
#y888 = [func2(i, *popt2) for i in x_sh[cut2:]]
#ax1.scatter(x_sh,y_sh,alpha=0.2)
#ax1.plot(x_sh[:cut2],y777,color='orange',linewidth = 3,linestyle='-',label=r'sh1, $\alpha$=%.2f'%popt1[1])
#ax1.plot(x_sh[cut2:],y888,color='orange',linewidth = 3,linestyle='-',label=r'sh2, $\alpha$=%.2f'%popt2[1])
#
#ax1.legend(loc=3,handletextpad=0.1,prop={'size':20,'family':'Times New Roman'})
#ax1.set_yscale('log')
#ax1.set_xscale('log')
#ax1.set_ylim([0.0000001,1])
#ax1.set_xlabel('rank_count',size=18,family='Times New Roman')  
#ax1.set_ylabel('p_rank',size=18,family='Times New Roman')
#
#
#
#popt, pcov = curve_fit(func1, rank_x2[2:],rank_y2[2:],maxfev = 10000)
#y777 = [func1(i, *popt) for i in rank_x2]
#ax2.scatter(rank_x2,rank_y2,label=r'bj, $\alpha$=%.2f'%popt[1],alpha=0.2)
#ax2.plot(rank_x2,y777,color='blue',linewidth = 3,linestyle='-')
#popt, pcov = curve_fit(func1, x_sh2,y_sh2,maxfev = 10000)
#y888 = [func1(i, *popt) for i in x_sh2]
#ax2.scatter(x_sh2,y_sh2,label=r'sh, $\alpha$=%.2f'%popt[1],alpha=0.2)
#ax2.plot(x_sh2,y888,color='orange',linewidth = 3,linestyle='-')
#
#ax2.legend(loc=3,handletextpad=0.1,prop={'size':20,'family':'Times New Roman'})        
#ax2.set_yscale('log')
#ax2.set_xscale('log')
#ax2.set_ylim([0.0000001,1])
#ax2.set_xlabel('rank_combine',size=18,family='Times New Roman')  
#ax2.set_ylabel('p_rank',size=18,family='Times New Roman')


