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



data=pd.read_csv('C:/python/Nanjing2/bikes_group_combine1.txt')
ha=data['rank_50'].value_counts(normalize=True)
m_bj1,n_bj1 = (list(t) for t in zip(*sorted(zip(list(ha.index),list(ha.values)))))
ha2=data['rank_100'].value_counts(normalize=True)
m_bj2,n_bj2 = (list(t) for t in zip(*sorted(zip(list(ha2.index),list(ha2.values)))))
data2=pd.read_csv('C:/python/Nanjing2/bikes_group_combine2.txt')
ha3=data2['rank_150'].value_counts(normalize=True)
m_bj3,n_bj3 = (list(t) for t in zip(*sorted(zip(list(ha3.index),list(ha3.values)))))
ha4=data2['rank_200'].value_counts(normalize=True)
m_bj4,n_bj4 = (list(t) for t in zip(*sorted(zip(list(ha4.index),list(ha4.values)))))
data3=pd.read_csv('C:/python/Nanjing2/bikes_group_combine3.txt')
ha5=data3['rank_250'].value_counts(normalize=True)
m_bj5,n_bj5 = (list(t) for t in zip(*sorted(zip(list(ha5.index),list(ha5.values)))))
ha6=data3['rank_300'].value_counts(normalize=True)
m_bj6,n_bj6 = (list(t) for t in zip(*sorted(zip(list(ha6.index),list(ha6.values)))))


scale_sh11=np.sum(np.array(m_bj1)*np.array(n_bj1))
scale_sh22=np.sum(np.array(m_bj2)*np.array(n_bj2))
scale_sh33=np.sum(np.array(m_bj3)*np.array(n_bj3))
scale_sh44=np.sum(np.array(m_bj4)*np.array(n_bj4))
scale_sh55=np.sum(np.array(m_bj5)*np.array(n_bj5))
scale_sh66=np.sum(np.array(m_bj6)*np.array(n_bj6))


m_bj1=np.array(m_bj1)/scale_sh11
n_bj1=np.array(n_bj1)*scale_sh11
m_bj2=np.array(m_bj2)/scale_sh22
n_bj2=np.array(n_bj2)*scale_sh22
m_bj3=np.array(m_bj3)/scale_sh33
n_bj3=np.array(n_bj3)*scale_sh33
m_bj4=np.array(m_bj4)/scale_sh44
n_bj4=np.array(n_bj4)*scale_sh44
m_bj5=np.array(m_bj5)/scale_sh55
n_bj5=np.array(n_bj5)*scale_sh55
m_bj6=np.array(m_bj6)/scale_sh66
n_bj6=np.array(n_bj6)*scale_sh66
#
#
m_bj1,n_bj1=log_pdf(m_bj1,n_bj1)
m_bj2,n_bj2=log_pdf(m_bj2,n_bj2)
m_bj3,n_bj3=log_pdf(m_bj3,n_bj3)
m_bj4,n_bj4=log_pdf(m_bj4,n_bj4)
m_bj5,n_bj5=log_pdf(m_bj5,n_bj5)
m_bj6,n_bj6=log_pdf(m_bj6,n_bj6)



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
ax11.plot(m_bj1,n_bj1,alpha=alpha_xy,marker='o',linestyle='--',label='D3 (r=50)')
#ax11.plot(m_bj2,n_bj2,alpha=alpha_xy,marker='o',linestyle='--',label='D3 (r=100)')
#ax11.plot(m_bj3,n_bj3,alpha=alpha_xy,marker='o',linestyle='--',label='D3 (r=150)')
#ax11.plot(m_bj4,n_bj4,alpha=alpha_xy,marker='o',linestyle='--',label='D3 (r=200)')
#ax11.plot(m_bj5,n_bj5,alpha=alpha_xy,marker='o',linestyle='--',label='D3 (r=250)')
#ax11.plot(m_bj6,n_bj6,alpha=alpha_xy,marker='o',linestyle='--',label='D3 (r=300)')
ax11.plot(m_bj2[1:],n_bj2[1:],alpha=alpha_xy,marker='o',linestyle='--',label='D3 (r=100)')
ax11.plot(m_bj3[1:],n_bj3[1:],alpha=alpha_xy,marker='o',linestyle='--',label='D3 (r=150)')
ax11.plot(m_bj4[2:],n_bj4[2:],alpha=alpha_xy,marker='o',linestyle='--',label='D3 (r=200)')
ax11.plot(m_bj5[3:],n_bj5[3:],alpha=alpha_xy,marker='o',linestyle='--',label='D3 (r=250)')
ax11.plot(m_bj6[3:],n_bj6[3:],alpha=alpha_xy,marker='o',linestyle='--',label='D3 (r=300)')
ax11.set_xticklabels([i for i in ax11.get_xticks()],family=f_family,size=18)
ax11.set_yticklabels([i for i in ax11.get_yticks()],family=f_family,size=18)
ax11.set_yscale('log')
ax11.set_xscale('log')
ax11.set_ylim([0.000001,10])
ax11.legend(loc=3,handletextpad=0.1,prop={'size':18,'family':f_family})
ax11.set_xlabel(r'rank(times$^{-1}$*$\Delta$t$^{-1}$)/$\langle$rank$\rangle$',size=18,family=f_family)  
ax11.set_ylabel(r'$\langle$rank$\rangle P$(rank)',size=18,family=f_family)
#plt.savefig('C:/python/摩拜单车/draw2/sf_8_nj_di.pdf',bbox_inches='tight')





