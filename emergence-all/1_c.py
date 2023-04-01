#一条订单数据一个骑行时间，骑行速度，所有时间list，速度list的分布
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as sci
from scipy.optimize import curve_fit


#data=pd.read_csv('F:/python/MobikeData/lak3.csv')
#data2=pd.read_csv('F:/python/MobikeData/lak.csv')
#dict2={}
#for i,j in zip(data2['orderid'],data2['t']):
#    if i not in dict2.keys():
#        dict2[i]=j
#print(len(dict2))
#a=[]
#for i in data['orderid']:
#    a.append(dict2[i])
#data['t']=a
#data_haha=data[['orderid','userid',	'bikeid','t','d']]
#data_haha.to_csv('F:/python/MobikeData/lak3_v.csv',index=False)


##data=pd.read_csv('C:/python/Nanjing/mobai33.csv')
#data=pd.read_csv('C:/python/Nanjing2/orange_Oct19_NJ33.csv')
#rows=data.index
#tt=[]
#vv=[]
#for row in rows:
#    t=int(data.loc[row,'dt(s)']//60)
#    v=round(data.loc[row,'d']/t,3)
#    tt.append(t)
#    vv.append(v)
#data['t']=tt
#data['v']=vv
#data_haha=data[['bikeid','t','d','v']]
##data_haha.to_csv('C:/python/Nanjing/mobai33_v.csv',index=False)
#data_haha.to_csv('C:/python/Nanjing2/orange_Oct19_NJ33_v.csv',index=False)




#data=pd.read_csv('C:/python/MobikeData/lak3_v.csv')
#rows=data.index
#v=[]
#for row in rows:
#    v.append(round(data.loc[row,'d']/data.loc[row,'t'],3))
#data['v']=v
##ax1 = sns.jointplot(x='t',y='v',data = data,height = 6)
#ax1 = sns.jointplot(x='t',y='v',data = data,kind='kde',shade=False,height = 6)
##ax1.plot_joint(plt.scatter, c = 'b', marker='+',alpha=0.1)
#plt.yticks(fontproperties = 'Times New Roman')
#plt.xticks(fontproperties = 'Times New Roman') 
#plt.tick_params(labelsize=18)
#ax1.set_axis_labels('duration(min)','v(km/min)',fontsize = 18,family='Times New Roman')
#
#dict2={}
#for i,j,k in zip(data['userid'],data['t'],data['v']):
#    if i not in dict2.keys():
#        dict2[i]=[[j],[k]]
#    else:
#        dict2[i][0].append(j)
#        dict2[i][1].append(k)
#t_mean=[]
#v_mean=[]
#for key,value in dict2.items():
#    t_mean.append(round(np.mean(value[0]),3))
#    v_mean.append(round(np.mean(value[1]),3))
#df_user={"duration":t_mean,"v":v_mean}
#df_user=pd.DataFrame(df_user)
#ax2 = sns.jointplot(x='duration',y='v',data = df_user,kind='kde',shade=False,height = 6)
##ax2.plot_joint(plt.scatter, c = 'b', marker='+',alpha=0.1)
#plt.yticks(fontproperties = 'Times New Roman')
#plt.xticks(fontproperties = 'Times New Roman') 
#plt.tick_params(labelsize=18)
#ax2.set_axis_labels('duration(min)','v(km/min)',fontsize = 18,family='Times New Roman')



#幂律 
def func4(x, a, b):
    return a*pow(x,b)


def pdf1(data,max,step):
    all_num=len(data)    
    x_range=np.arange(0,max,step)
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


data=pd.read_csv('C:/python/MobikeData/lak3_v.csv')
#data=pd.read_csv('C:/python/Nanjing/mobai3_v.csv')
#data2=pd.read_csv('C:/python/Nanjing2/orange_Oct19_NJ3_v.csv')
x1,y1=pdf1(data['t'],1500,2)
x2,y2=pdf1(data['v'],1.2,0.012)
#x11,y11=pdf1(data2['t'],1500,2)

f_family='Arial'
fig=	plt.figure(figsize=(6, 6))
ax1	=	fig.add_subplot(2,	1,	1) 
ax2	=	fig.add_subplot(2,	1,	2)
ax1.plot(x1,y1)

x_no=range(30,200)
popt, pcov = curve_fit(func4, x1[30:300],y1[30:300],maxfev = 10000)
print(popt)
popt_no=[popt[0]-2200,popt[1]]
y777 = [func4(i, *popt_no) for i in x_no]
#ax1.errorbar(rank_x,rank_y,fmt="o",yerr = rank_std,label=r'user, $\xi$=%.2f'%-popt[1])
ax1.plot([40,40],[0.00005,0.0005],'k-',linewidth=2)
ax1.plot([40,90],[0.00005,0.00005],'k-',linewidth=2)
ax1.plot(x_no,y777,'k-',linewidth=2)
ax1.text(30, 0.000005,r'%.2f'%popt[1], size = 18,family=f_family)

ax1.set_xticklabels([int(i) for i in ax1.get_xticks()],family=f_family,size=18)
ax1.set_yticklabels([round(i,4) for i in ax1.get_yticks()],family=f_family,size=18)
ax1.set_yscale('log')
ax1.set_xscale('log')
#ax1.set_ylim([0.00001,1])
ax1.set_xlabel('duration(min)',size=18,family=f_family)  
ax1.set_ylabel('$P$',size=18,family=f_family)

ax2.plot(x2,y2)
ax2.set_xticklabels([round(i,2) for i in ax2.get_xticks()],family=f_family,size=18)
ax2.set_yticklabels([round(i,4) for i in ax2.get_yticks()],family=f_family,size=18)
ax2.set_xlabel('v(km/min)',size=18,family=f_family)  
ax2.set_ylabel('$P$',size=18,family=f_family)
plt.subplots_adjust(hspace=0.5)
#plt.savefig('C:/python/摩拜单车/draw2/1_c.pdf',bbox_inches='tight')  



#南京数据
#fig	=	plt.figure(figsize=(8, 6))
#ax1	=	fig.add_subplot(1,	1,	1)
#plt.yticks(fontproperties = f_family)
#plt.xticks(fontproperties = f_family) 
#plt.tick_params(labelsize=18)
#ax1.scatter(x1,y1,label='Mobike')
#ax1.scatter(x11,y11,label='DiDi')
#ax1.legend(loc=3,handletextpad=0.1,prop={'size':18,'family':f_family})
#ax1.set_yscale('log')
#ax1.set_xscale('log')
#ax1.set_ylim([0.0000001,1])
#ax1.set_xlim([0.7,2000])
#ax1.set_xlabel('duration(min)',size=18,family=f_family)  
#ax1.set_ylabel('$P$',size=18,family=f_family)



#dict2={}
#for i,j,k in zip(data['userid'],data['t'],data['v']):
#    if i not in dict2.keys():
#        dict2[i]=[[j],[k]]
#    else:
#        dict2[i][0].append(j)
#        dict2[i][1].append(k)
#t_mean=[]
#v_mean=[]
#for key,value in dict2.items():
#    t_mean.append(round(np.mean(value[0]),3))
#    v_mean.append(round(np.mean(value[1]),3))
#x11,y11=pdf1(t_mean,500,2)
#x22,y22=pdf1(v_mean,1,0.01)
#fig2=	plt.figure(figsize=(8, 6))
#ax11	=	fig2.add_subplot(2,	1,	1) 
#ax22	=	fig2.add_subplot(2,	1,	2)
#ax11.plot(x11,y11)
#ax11.set_xticklabels([int(i) for i in ax11.get_xticks()],family=f_family,size=18)
#ax11.set_yticklabels([round(i,4) for i in ax11.get_yticks()],family=f_family,size=18)
##ax11.set_yscale('log')
##ax11.set_xscale('log')
#ax11.set_xlabel('duration(min)',size=18,family=f_family)  
#ax11.set_ylabel('$P$',size=18,family=f_family)
#
#ax22.plot(x22,y22)
#ax22.set_xticklabels([round(i,2) for i in ax22.get_xticks()],family=f_family,size=18)
#ax22.set_yticklabels([round(i,4) for i in ax22.get_yticks()],family=f_family,size=18)
#ax22.set_xlabel('v(km/min)',size=18,family=f_family)  
#ax22.set_ylabel('$P$',size=18,family=f_family)
#plt.subplots_adjust(hspace=0.5)




