#范围内车数的分布，计算方法在group_bikes里
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


bikes_sh1=np.loadtxt('C:/python/MobikeData/grid2/near50_bikes.txt',dtype=int,delimiter=',')
i_sh1=bikes_sh1[:,0]
j_sh1=bikes_sh1[:,1]/np.sum(bikes_sh1[:,1])
bikes_sh2=np.loadtxt('C:/python/MobikeData/grid2/near100_bikes.txt',dtype=int,delimiter=',')
i_sh2=bikes_sh2[:,0]
j_sh2=bikes_sh2[:,1]/np.sum(bikes_sh2[:,1])
#bikes_sh3=np.loadtxt('C:/python/MobikeData/grid2/near200_bikes.txt',dtype=int,delimiter=',')
#i_sh3=bikes_sh3[:,0]
#j_sh3=bikes_sh3[:,1]/np.sum(bikes_sh3[:,1])
#bikes_sh4=np.loadtxt('C:/python/MobikeData/grid2/near300_bikes.txt',dtype=int,delimiter=',')
#i_sh4=bikes_sh4[:,0]
#j_sh4=bikes_sh4[:,1]/np.sum(bikes_sh4[:,1])
#bikes_sh5=np.loadtxt('C:/python/MobikeData/grid2/near400_bikes.txt',dtype=int,delimiter=',')
#i_sh5=bikes_sh5[:,0]
#j_sh5=bikes_sh5[:,1]/np.sum(bikes_sh5[:,1])
#bikes_sh6=np.loadtxt('C:/python/MobikeData/grid2/near500_bikes.txt',dtype=int,delimiter=',')
#i_sh6=bikes_sh6[:,0]
#j_sh6=bikes_sh6[:,1]/np.sum(bikes_sh6[:,1])
#bikes_sh7=np.loadtxt('C:/python/MobikeData/grid2/near600_bikes.txt',dtype=int,delimiter=',')
#i_sh7=bikes_sh7[:,0]
#j_sh7=bikes_sh7[:,1]/np.sum(bikes_sh7[:,1])
#bikes_bj1=np.loadtxt('C:/python/MOBIKE_CUP_2017/grid2/near75_bikes.txt',dtype=int,delimiter=',')
#i_bj1=bikes_bj1[:,0]
#j_bj1=bikes_bj1[:,1]/np.sum(bikes_bj1[:,1])
bikes_bj2=np.loadtxt('C:/python/MOBIKE_CUP_2017/grid2/near150_bikes.txt',dtype=int,delimiter=',')
i_bj2=bikes_bj2[:,0]
j_bj2=bikes_bj2[:,1]/np.sum(bikes_bj2[:,1])
#bikes_bj3=np.loadtxt('C:/python/MOBIKE_CUP_2017/grid2/near300_bikes.txt',dtype=int,delimiter=',')
#i_bj3=bikes_bj3[:,0]
#j_bj3=bikes_bj3[:,1]/np.sum(bikes_bj3[:,1])
#bikes_bj4=np.loadtxt('C:/python/MOBIKE_CUP_2017/grid2/near450_bikes.txt',dtype=int,delimiter=',')
#i_bj4=bikes_bj4[:,0]
#j_bj4=bikes_bj4[:,1]/np.sum(bikes_bj4[:,1])
#bikes_bj5=np.loadtxt('C:/python/MOBIKE_CUP_2017/grid2/near600_bikes.txt',dtype=int,delimiter=',')
#i_bj5=bikes_bj5[:,0]
#j_bj5=bikes_bj5[:,1]/np.sum(bikes_bj5[:,1])
data_nj3=pd.read_csv('C:/python/Nanjing2/bikes_group_combine3.txt')
xi=data_nj3['bikes_250'].value_counts(normalize=True)
i_nj2,j_nj2 = (list(t) for t in zip(*sorted(zip(list(xi.index),list(xi.values)))))
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


#scale_1=np.sum(np.array(i_sh2)*np.array(j_sh2))
#scale_2=np.sum(np.array(i_bj2)*np.array(j_bj2))
#print(scale_1)
#print(scale_2)
#print(np.mean(data_nj3['bikes_250']))
#print(np.mean(data_nj_mo['bikes_350']))
#print(np.mean(data_xm3['bikes_150']))


#i_bj1,j_bj1=log_pdf(i_bj1,j_bj1)
i_bj2,j_bj2=log_pdf(i_bj2,j_bj2)
#i_bj3,j_bj3=log_pdf(i_bj3,j_bj3)
#i_bj4,j_bj4=log_pdf(i_bj4,j_bj4)
#i_bj5,j_bj5=log_pdf(i_bj5,j_bj5)
#i_sh1,j_sh1=log_pdf(i_sh1,j_sh1)
i_sh2,j_sh2=log_pdf(i_sh2,j_sh2)
#i_sh3,j_sh3=log_pdf(i_sh3,j_sh3)
#i_sh4,j_sh4=log_pdf(i_sh4,j_sh4)
#i_sh5,j_sh5=log_pdf(i_sh5,j_sh5)
#i_sh6,j_sh6=log_pdf(i_sh6,j_sh6)
#i_sh7,j_sh7=log_pdf(i_sh7,j_sh7)
i_nj2,j_nj2=log_pdf(i_nj2,j_nj2)
i_nj4,j_nj4=log_pdf(i_nj4,j_nj4)
i_cd2,j_cd2=log_pdf(i_cd2,j_cd2)
i_xa2,j_xa2=log_pdf(i_xa2,j_xa2)
i_xm2,j_xm2=log_pdf(i_xm2,j_xm2)
i_sgp2,j_sgp2=log_pdf(i_sgp2,j_sgp2)




f_family='Arial'
fig3=	plt.figure(figsize=(8, 6))
ax111=	fig3.add_subplot(1,	1,	1) 

def func1(x, a,b1,b2,c):
    return a*pow(x+c,-b1)*np.exp((-b2)*x)

def func2(x, a,b1,b2):
    return a*pow(x,-b1)*np.exp((-b2)*x)

no=6
alpha_xy=0.8
#ax111.plot(i_sh1,j_sh1,alpha=alpha_xy,marker='o',linestyle='--',label='sh r=50')
ax111.plot(i_sh2,j_sh2,alpha=alpha_xy,marker='o',markersize=10,markerfacecolor='none',markeredgewidth=2,linestyle='--',label='D1 (r=100)')
#ax111.plot(i_bj1,j_bj1,alpha=alpha_xy,marker='s',linestyle='--',label='bj r=75')
ax111.plot(i_bj2,j_bj2,alpha=alpha_xy,marker='s',markersize=10,markerfacecolor='none',markeredgewidth=2,linestyle='--',label='D2 (r=150)')
ax111.plot(i_nj2,j_nj2,alpha=alpha_xy,marker='^',markersize=10,markerfacecolor='none',markeredgewidth=2,linestyle='--',label='D3 (r=250)')
#popt, pcov = curve_fit(func2, i_nj2[:1]+i_nj2[no:], j_nj2[:1]+j_nj2[no:],maxfev = 10000)
#y777 = [func2(i, *popt) for i in i_nj2]
#ax111.plot(i_nj2,j_nj2,alpha=alpha_xy,marker='^',markersize=10,markerfacecolor='none',markeredgewidth=2,linestyle='--',label='D3 (r=250)')
#ax111.plot(i_nj2,y777,color='green',linestyle='-',label=r'$\alpha$=%.2f'%popt[1])
ax111.plot(i_nj4,j_nj4,alpha=alpha_xy,marker='^',markersize=10,markerfacecolor='none',markeredgewidth=2,linestyle='--',label='D4 (r=350)')
ax111.plot(i_cd2,j_cd2,alpha=alpha_xy,marker='^',markersize=10,markerfacecolor='none',markeredgewidth=2,linestyle='--',label='D5 (r=150)')
ax111.plot(i_xa2,j_xa2,alpha=alpha_xy,marker='^',markersize=10,markerfacecolor='none',markeredgewidth=2,linestyle='--',label='D6 (r=200)')
ax111.plot(i_xm2,j_xm2,alpha=alpha_xy,marker='^',markersize=10,markerfacecolor='none',markeredgewidth=2,linestyle='--',label='D7 (r=150)')
ax111.plot(i_sgp2,j_sgp2,alpha=alpha_xy,marker='^',markersize=10,markerfacecolor='none',markeredgewidth=2,linestyle='--',label='D8 (r=300)')
ax111.set_xticklabels([i for i in ax111.get_xticks()],family=f_family,size=18)
ax111.set_yticklabels([i for i in ax111.get_yticks()],family=f_family,size=18)
ax111.set_yscale('log')
ax111.set_xscale('log')
ax111.set_ylim([0.00000001,1])
ax111.legend(loc=3,handletextpad=0.1,ncol=2,columnspacing=1,prop={'size':18,'family':f_family})
ax111.set_xlabel('#bikes',size=18,family=f_family)  
ax111.set_ylabel('$P$(#)',size=18,family=f_family)
#plt.savefig('C:/python/摩拜单车/draw2/3_a.pdf',bbox_inches='tight')        
        
        




