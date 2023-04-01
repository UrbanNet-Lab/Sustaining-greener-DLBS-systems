#范围内出行车的排名的分布，计算方法在group_bikes里
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




combine_sh1=np.loadtxt('C:/python/MobikeData/weeks/rank100_combine.txt',dtype=int,delimiter=',')
#combine_sh1=np.loadtxt('C:/python/MobikeData/grid2/rank50_combine.txt',dtype=int,delimiter=',')
m_sh1=combine_sh1[:,0]
n_sh1=combine_sh1[:,1]/np.sum(combine_sh1[:,1])
combine_sh2=np.loadtxt('C:/python/MobikeData/weeks/rank200_combine.txt',dtype=int,delimiter=',')
#combine_sh2=np.loadtxt('C:/python/MobikeData/grid2/rank100_combine.txt',dtype=int,delimiter=',')
m_sh2=combine_sh2[:,0]
n_sh2=combine_sh2[:,1]/np.sum(combine_sh2[:,1])
combine_sh3=np.loadtxt('C:/python/MobikeData/grid2/rank200_combine.txt',dtype=int,delimiter=',')
m_sh3=combine_sh3[:,0]
n_sh3=combine_sh3[:,1]/np.sum(combine_sh3[:,1])
combine_sh4=np.loadtxt('C:/python/MobikeData/grid2/rank300_combine.txt',dtype=int,delimiter=',')
m_sh4=combine_sh4[:,0]
n_sh4=combine_sh4[:,1]/np.sum(combine_sh4[:,1])
combine_sh5=np.loadtxt('C:/python/MobikeData/grid2/rank400_combine.txt',dtype=int,delimiter=',')
m_sh5=combine_sh5[:,0]
n_sh5=combine_sh5[:,1]/np.sum(combine_sh5[:,1])
combine_sh6=np.loadtxt('C:/python/MobikeData/grid2/rank500_combine.txt',dtype=int,delimiter=',')
m_sh6=combine_sh6[:,0]
n_sh6=combine_sh6[:,1]/np.sum(combine_sh6[:,1])
combine_sh7=np.loadtxt('C:/python/MobikeData/grid2/rank600_combine.txt',dtype=int,delimiter=',')
m_sh7=combine_sh7[:,0]
n_sh7=combine_sh7[:,1]/np.sum(combine_sh7[:,1])
combine_bj1=np.loadtxt('C:/python/MOBIKE_CUP_2017/grid2/rank75_combine.txt',dtype=int,delimiter=',')
m_bj1=combine_bj1[:,0]
n_bj1=combine_bj1[:,1]/np.sum(combine_bj1[:,1])
combine_bj2=np.loadtxt('C:/python/MOBIKE_CUP_2017/grid2/rank150_combine.txt',dtype=int,delimiter=',')
m_bj2=combine_bj2[:,0]
n_bj2=combine_bj2[:,1]/np.sum(combine_bj2[:,1])
combine_bj3=np.loadtxt('C:/python/MOBIKE_CUP_2017/grid2/rank300_combine.txt',dtype=int,delimiter=',')
m_bj3=combine_bj3[:,0]
n_bj3=combine_bj3[:,1]/np.sum(combine_bj3[:,1])
combine_bj4=np.loadtxt('C:/python/MOBIKE_CUP_2017/grid2/rank450_combine.txt',dtype=int,delimiter=',')
m_bj4=combine_bj4[:,0]
n_bj4=combine_bj4[:,1]/np.sum(combine_bj4[:,1])
combine_bj5=np.loadtxt('C:/python/MOBIKE_CUP_2017/grid2/rank600_combine.txt',dtype=int,delimiter=',')
m_bj5=combine_bj5[:,0]
n_bj5=combine_bj5[:,1]/np.sum(combine_bj5[:,1])
data_nj2=pd.read_csv('C:/python/Nanjing2/bikes_group_combine3.txt')
ha=data_nj2['rank_250'].value_counts(normalize=True)
m_nj2,n_nj2 = (list(t) for t in zip(*sorted(zip(list(ha.index),list(ha.values)))))
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




#m_bj1,n_bj1=log_pdf(m_bj1,n_bj1)
m_bj2,n_bj2=log_pdf(m_bj2,n_bj2)
#m_bj3,n_bj3=log_pdf(m_bj3,n_bj3)
#m_bj4,n_bj4=log_pdf(m_bj4,n_bj4)
#m_bj5,n_bj5=log_pdf(m_bj5,n_bj5)
#m_sh1,n_sh1=log_pdf(m_sh1,n_sh1)
m_sh2,n_sh2=log_pdf(m_sh2,n_sh2)
#m_sh3,n_sh3=log_pdf(m_sh3,n_sh3)
#m_sh4,n_sh4=log_pdf(m_sh4,n_sh4)
#m_sh5,n_sh5=log_pdf(m_sh5,n_sh5)
#m_sh6,n_sh6=log_pdf(m_sh6,n_sh6)
#m_sh7,n_sh7=log_pdf(m_sh7,n_sh7)
m_nj2,n_nj2=log_pdf(m_nj2,n_nj2)
i_nj4,j_nj4=log_pdf(i_nj4,j_nj4)
i_cd2,j_cd2=log_pdf(i_cd2,j_cd2)
i_xa2,j_xa2=log_pdf(i_xa2,j_xa2)
i_xm2,j_xm2=log_pdf(i_xm2,j_xm2)
i_sgp2,j_sgp2=log_pdf(i_sgp2,j_sgp2)


f_family='Arial'
fig2=	plt.figure(figsize=(8, 6))
ax11=	fig2.add_subplot(1,	1,	1) 


def func1(x, a,b1,b2,c):
    return a*pow(x+c,-b1)*np.exp((-b2)*x)

def func2(x, a,b1,b2):
    return a*pow(x,-b1)*np.exp((-b2)*x)


alpha_xy=1
#popt, pcov = curve_fit(func2, m_sh1[4:], n_sh1[4:],maxfev = 10000)
#y777 = [func2(i, *popt) for i in m_sh1]
#ax11.scatter(m_sh1,n_sh1,alpha=alpha_xy,marker='o',label=r'sh r=50')
#ax11.plot(m_sh1,y777,color='blue',linestyle='-',label=r'$\alpha$=%.2f'%popt[1])
popt, pcov = curve_fit(func2, m_sh2[1:], n_sh2[1:],maxfev = 10000)
y777 = [func2(i, *popt) for i in m_sh2]
ax11.scatter(m_sh2,n_sh2,alpha=alpha_xy,marker='o',label=r'D1 (r=100)')
ax11.plot(m_sh2,y777,color='blue',linestyle='-',label=r'$\alpha$=%.2f'%popt[1])
print([round(i,2) for i in popt])
#popt, pcov = curve_fit(func2, m_bj1[4:], n_bj1[4:],maxfev = 10000)
#y888 = [func2(i, *popt) for i in m_bj1]
#ax11.scatter(m_bj1,n_bj1,alpha=alpha_xy,marker='s',label=r'bj r=75')
#ax11.plot(m_bj1,y888,color='g',linestyle='-',label=r'bj r=75($\alpha$=%.2f)'%popt[1])
#popt, pcov = curve_fit(func2, m_bj2[1:], n_bj2[1:],maxfev = 10000)
#y888 = [func2(i, *popt) for i in m_bj2]
ax11.scatter(m_bj2,n_bj2,alpha=alpha_xy,marker='s',label='D2 (r=150)')
ax11.scatter(m_nj2,n_nj2,alpha=alpha_xy,marker='^',label='D3 (r=250)')
ax11.scatter(i_nj4,j_nj4,alpha=alpha_xy,marker='^',label='D4 (r=350)')
ax11.scatter(i_cd2,j_cd2,alpha=alpha_xy,marker='^',label='D5 (r=150)')
ax11.scatter(i_xa2,j_xa2,alpha=alpha_xy,marker='^',label='D6 (r=200)')
ax11.scatter(i_xm2,j_xm2,alpha=alpha_xy,marker='^',label='D7 (r=150)')
ax11.scatter(i_sgp2,j_sgp2,alpha=alpha_xy,marker='^',label='D8 (r=300)')
#ax11.plot(m_bj2,y888,color='r',linestyle='-',label=r'bj r=150($\alpha$=%.2f)'%popt[1])
ax11.set_xticklabels([i for i in ax11.get_xticks()],family=f_family,size=18)
ax11.set_yticklabels([i for i in ax11.get_yticks()],family=f_family,size=18)
ax11.set_yscale('log')
ax11.set_xscale('log')
ax11.set_ylim([0.00000001,1])
ax11.legend(loc=3,handletextpad=0.1,ncol=2,columnspacing=1,prop={'size':18,'family':f_family})
ax11.set_xlabel(r'rank(times$^{-1}$*$\Delta$t$^{-1}$)',size=18,family=f_family)  
ax11.set_ylabel(r'$P$(rank)',size=18,family=f_family)
#plt.savefig('C:/python/摩拜单车/draw2/3_b.pdf',bbox_inches='tight') 

        
        



