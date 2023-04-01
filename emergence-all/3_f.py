#重标度
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
from scipy.optimize import curve_fit
import time
from math import radians, cos, sin, asin, sqrt
import math



def log_pdf(x,y):
    bins=np.logspace(-3, 2, 20)
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
#            index_this=bins2.index(key)
#            y_new.append(np.sum(value)/widths[index_this])
            y_new.append(np.mean(value))
            x_new.append(key)
        
    return x_new,y_new


count_sh1=np.loadtxt('C:/python/MobikeData/weeks/rank100_count.txt',dtype=int,delimiter=',')
#count_sh1=np.loadtxt('C:/python/MobikeData/grid2/rank50_count.txt',dtype=int,delimiter=',')
x_sh1=count_sh1[:,0]
y_sh1=count_sh1[:,1]/np.sum(count_sh1[:,1])
count_sh2=np.loadtxt('C:/python/MobikeData/weeks/rank200_count.txt',dtype=int,delimiter=',')
#count_sh2=np.loadtxt('C:/python/MobikeData/grid2/rank100_count.txt',dtype=int,delimiter=',')
x_sh2=count_sh2[:,0]
y_sh2=count_sh2[:,1]/np.sum(count_sh2[:,1])
count_sh3=np.loadtxt('C:/python/MobikeData/grid2/rank200_count.txt',dtype=int,delimiter=',')
x_sh3=count_sh3[:,0]
y_sh3=count_sh3[:,1]/np.sum(count_sh3[:,1])
count_sh4=np.loadtxt('C:/python/MobikeData/grid2/rank300_count.txt',dtype=int,delimiter=',')
x_sh4=count_sh4[:,0]
y_sh4=count_sh4[:,1]/np.sum(count_sh4[:,1])
count_sh5=np.loadtxt('C:/python/MobikeData/grid2/rank400_count.txt',dtype=int,delimiter=',')
x_sh5=count_sh5[:,0]
y_sh5=count_sh5[:,1]/np.sum(count_sh5[:,1])
count_sh6=np.loadtxt('C:/python/MobikeData/grid2/rank500_count.txt',dtype=int,delimiter=',')
x_sh6=count_sh6[:,0]
y_sh6=count_sh6[:,1]/np.sum(count_sh6[:,1])
count_sh7=np.loadtxt('C:/python/MobikeData/grid2/rank600_count.txt',dtype=int,delimiter=',')
x_sh7=count_sh7[:,0]
y_sh7=count_sh7[:,1]/np.sum(count_sh7[:,1])


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


scale_sh1=np.sum(np.array(x_sh1)*np.array(y_sh1))
scale_sh2=np.sum(np.array(x_sh2)*np.array(y_sh2))
scale_sh3=np.sum(np.array(x_sh3)*np.array(y_sh3))
scale_sh4=np.sum(np.array(x_sh4)*np.array(y_sh4))
scale_sh5=np.sum(np.array(x_sh5)*np.array(y_sh5))
scale_sh6=np.sum(np.array(x_sh6)*np.array(y_sh6))
scale_sh7=np.sum(np.array(x_sh7)*np.array(y_sh7))

scale_sh11=np.sum(np.array(m_sh1)*np.array(n_sh1))
scale_sh22=np.sum(np.array(m_sh2)*np.array(n_sh2))
scale_sh33=np.sum(np.array(m_sh3)*np.array(n_sh3))
scale_sh44=np.sum(np.array(m_sh4)*np.array(n_sh4))
scale_sh55=np.sum(np.array(m_sh5)*np.array(n_sh5))
scale_sh66=np.sum(np.array(m_sh6)*np.array(n_sh6))
scale_sh77=np.sum(np.array(m_sh7)*np.array(n_sh7))






x_sh1=np.array(x_sh1)/scale_sh1
y_sh1=np.array(y_sh1)*scale_sh1
x_sh2=np.array(x_sh2)/scale_sh2
y_sh2=np.array(y_sh2)*scale_sh2
x_sh3=np.array(x_sh3)/scale_sh3
y_sh3=np.array(y_sh3)*scale_sh3
x_sh4=np.array(x_sh4)/scale_sh4
y_sh4=np.array(y_sh4)*scale_sh4
x_sh5=np.array(x_sh5)/scale_sh5
y_sh5=np.array(y_sh5)*scale_sh5
x_sh6=np.array(x_sh6)/scale_sh6
y_sh6=np.array(y_sh6)*scale_sh6
x_sh7=np.array(x_sh7)/scale_sh7
y_sh7=np.array(y_sh7)*scale_sh7

m_sh1=np.array(m_sh1)/scale_sh11
n_sh1=np.array(n_sh1)*scale_sh11
m_sh2=np.array(m_sh2)/scale_sh22
n_sh2=np.array(n_sh2)*scale_sh22
m_sh3=np.array(m_sh3)/scale_sh33
n_sh3=np.array(n_sh3)*scale_sh33
m_sh4=np.array(m_sh4)/scale_sh44
n_sh4=np.array(n_sh4)*scale_sh44
m_sh5=np.array(m_sh5)/scale_sh55
n_sh5=np.array(n_sh5)*scale_sh55
m_sh6=np.array(m_sh6)/scale_sh66
n_sh6=np.array(n_sh6)*scale_sh66
m_sh7=np.array(m_sh7)/scale_sh77
n_sh7=np.array(n_sh7)*scale_sh77


m_sh1,n_sh1=log_pdf(m_sh1,n_sh1)
m_sh2,n_sh2=log_pdf(m_sh2,n_sh2)
m_sh3,n_sh3=log_pdf(m_sh3,n_sh3)
m_sh4,n_sh4=log_pdf(m_sh4,n_sh4)
m_sh5,n_sh5=log_pdf(m_sh5,n_sh5)
m_sh6,n_sh6=log_pdf(m_sh6,n_sh6)
m_sh7,n_sh7=log_pdf(m_sh7,n_sh7)


def func1(x, a,b1,b2,c):
    return a*pow(x+c,-b1)*np.exp((-b2)*x)

def func2(x, a,b1,b2):
    return a*pow(x,-b1)*np.exp((-b2)*x)


f_family='Arial'
#fig=	plt.figure(figsize=(8, 6))
#ax1	=	fig.add_subplot(1,	1,	1) 

fig2=	plt.figure(figsize=(8, 6))
ax11=	fig2.add_subplot(1,	1,	1) 

alpha_xy=1
ax11.plot(m_sh1,n_sh1,alpha=alpha_xy,marker='o',linestyle='--',label='D1 (r=50)')
ax11.plot(m_sh2,n_sh2,alpha=alpha_xy,marker='o',linestyle='--',label='D1 (r=100)')
ax11.plot(m_sh3,n_sh3,alpha=alpha_xy,marker='o',linestyle='--',label='D1 (r=200)')
ax11.plot(m_sh4,n_sh4,alpha=alpha_xy,marker='o',linestyle='--',label='D1 (r=300)')
#ax11.plot(m_sh5,n_sh5,alpha=alpha_xy,marker='o',linestyle='--',label='D1 (r=400)')
#ax11.plot(m_sh6,n_sh6,alpha=alpha_xy,marker='o',linestyle='--',label='D1 (r=500)')
#ax11.plot(m_sh7,n_sh7,alpha=alpha_xy,marker='o',linestyle='--',label='sh r=600')
ax11.set_xticklabels([i for i in ax11.get_xticks()],family=f_family,size=18)
ax11.set_yticklabels([i for i in ax11.get_yticks()],family=f_family,size=18)
ax11.set_yscale('log')
ax11.set_xscale('log')
ax11.set_ylim([0.000001,10])
ax11.legend(loc=3,handletextpad=0.1,prop={'size':18,'family':f_family})
ax11.set_xlabel(r'rank(times$^{-1}$*$\Delta$t$^{-1}$)/$\langle$rank$\rangle$',size=18,family=f_family)  
ax11.set_ylabel(r'$\langle$rank$\rangle P$(rank)',size=18,family=f_family)
#plt.savefig('C:/python/摩拜单车/draw2/3_f.pdf',bbox_inches='tight')





