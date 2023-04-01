#骑行时间分布
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as sci
import time


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


#data=pd.read_csv('C:/python/Xiamen/dc_xm3.csv')
#rows=data.index
#tt=[]
#for row in rows:
#    t_s=pd.to_datetime(data.loc[row,'start_time'])
#    t_e=pd.to_datetime(data.loc[row,'end_time'])
#    dt=int(time.mktime(t_e.timetuple())-time.mktime(t_s.timetuple()))
#    t=int(dt//60)
#    tt.append(t)
#data['t']=tt
#data_haha=data[['bikeid','t']]
#data_haha.to_csv('C:/python/Xiamen/dc_xm3_v.csv',index=False)





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






def pdf1(data,max,step):
    all_num=len(data)    
    x_range=np.arange(0,max,step)
#    y=np.zeros((len(x_range)-1), dtype=np.int)
    y=np.zeros(len(x_range)-1)
    x=x_range[:-1]+step/2
        
    for data1 in data:
        if(data1<max-step):
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


#data=pd.read_csv('C:/python/MobikeData/lak3_v.csv')
data=pd.read_csv('C:/python/Nanjing/mobai3_v.csv')
data2=pd.read_csv('C:/python/Nanjing2/orange_Oct19_NJ3_v.csv')
data3=pd.read_csv('C:/python/Xiamen/dc_xm3.csv')
data4=pd.read_csv('C:/python/Mobai/Chengdu/mobai_cd3_v.csv')
data5=pd.read_csv('C:/python/Mobai/Xian/mobai_xa3_v.csv')
data6=pd.read_csv('C:/python/Singapore/bike_sgp.csv')
x1,y1=pdf1(data['t'],1500,2)
x11,y11=pdf1(data2['t'],1500,2)
x111,y111=pdf1([int(i//60) for i in data3['dt(s)']],1500,2)
x_cd,y_cd=pdf1(data4['t'],1500,2)
x_xa,y_xa=pdf1(data5['t'],1500,2)
x_sgp,y_sgp=pdf1([int(i//60) for i in data6['dt(s)']],1500,2)
x_cd_new,y_cd_new,x_xa_new,y_xa_new=[],[],[],[]
x1_new,y1_new=[],[]
for i in range(len(x_cd)):
    if(x_cd[i]<120):
        x_cd_new.append(x_cd[i])
        y_cd_new.append(y_cd[i])
for i in range(len(x_xa)):
    if(x_xa[i]<120):
        x_xa_new.append(x_xa[i])
        y_xa_new.append(y_xa[i])
for i in range(len(x1)):
    if(x1[i]<120):
        x1_new.append(x1[i])
        y1_new.append(y1[i])

f_family='Arial'
fig	=	plt.figure(figsize=(6, 6))
ax1	=	fig.add_subplot(1,	1,	1)
plt.yticks(fontproperties = f_family)
plt.xticks(fontproperties = f_family) 
plt.tick_params(labelsize=18)
ax1.scatter(x11,y11,label='D3 (NJ)')
#ax1.scatter(x1,y1,label='D4 (NJ)')
ax1.scatter(x1_new,y1_new,label='D4 (NJ)')
#ax1.scatter(x_cd,y_cd,label='D5 (CD)')
#ax1.scatter(x_xa,y_xa,label='D6 (XA)')
ax1.scatter(x_cd_new,y_cd_new,label='D5 (CD)')
ax1.scatter(x_xa_new,y_xa_new,label='D6 (XA)')
ax1.scatter(x111,y111,label='D7 (XM)')
ax1.scatter(x_sgp,y_sgp,label='D8 (SGP)')
ax1.legend(loc=3,handletextpad=0.1,prop={'size':18,'family':f_family})
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_ylim([0.0000001,1])
ax1.set_xlim([0.7,2000])
ax1.set_xlabel('duration(min)',size=18,family=f_family)  
ax1.set_ylabel('$P$',size=18,family=f_family)
#plt.savefig('C:/python/摩拜单车/draw2/sf_1_c.pdf',bbox_inches='tight')



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





