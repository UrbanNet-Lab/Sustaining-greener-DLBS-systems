#人和车的间隔时间的分布，一个人的多次出行的间隔list的分布，n次出行n-1个间隔，一个人一个分布，对所有人的分布求平均
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import datetime
import pandas as pd
from scipy.optimize import curve_fit

def pdf1(data,step):
    all_num=len(data)    
    x_range=np.arange(0,1000,step)
#    y=np.zeros((len(x_range)-1), dtype=np.int)
    y=np.zeros(len(x_range)-1)
    x=x_range[:-1]+step/2
#    x=x_range[:-1]             
        
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

def time_gap(userid,time1):
    DELTA=1
    dict1={}
    for i,j in zip(userid,time1):
        if i not in dict1.keys():
            dict1[i]=[j]
        else:
            dict1[i].append(j)
    u_all2=0    
    dict_all1={}  
    for key,value in dict1.items():
        if(len(value)>1):
            u_all2=u_all2+1
            user_this=[]
            for m in range(len(value)-1):
                user_this.append(((time.mktime(value[m+1].timetuple())-time.mktime(value[m].timetuple()))/60)//60)
#                user_this.append( (value[m+1].month-value[m].month)*31*24+(value[m+1].day-value[m].day)*24+(value[m+1].hour-value[m].hour) )
            x1,y1=pdf1(user_this,DELTA)
            for i,j in zip(x1,y1):
                if i not in dict_all1.keys():
                    dict_all1[i]=[j]
                else:
                    dict_all1[i].append(j) 
    all1=[]
    all1_x=[]
#    bar1=[]
#    print(u_all2)
    for key,value in dict_all1.items():
#        all1.append(np.mean(value))
        all1.append(np.sum(value)/u_all2)
        all1_x.append(key)
#        bar1.append(np.std(value))
    all1_x,all1 = (list(t) for t in zip(*sorted(zip(all1_x,all1)))) 
    return all1_x,all1

def log_pdf(x,y):
#    bins=np.logspace(0, 4, 20,base=2)
    bins=np.logspace(0, 3, 20)
    bins2=list(bins)
    bins_all={}
    for i in range(len(bins)-1):
        bins_all[bins[i]]=[]
    widths = (bins[1:] - bins[:-1])
    for i in range(len(x)):
        if(x[i]>=1):
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

#幂律 
def func1(x, a, b):
    return a*pow(x,b)

def func2(x, a,b1,b2,c):
    return a*pow(x+c,-b1)*np.exp((-b2)*x)
def func3(x, a,b1,b2):
    return a*pow(x,-b1)*np.exp((-b2)*x)

#幂律 
def func4(x, a, b):
    return a*pow(x,b)

#data1=pd.read_csv('C:/python/MobikeData/lak3.csv')
#time1 = pd.to_datetime(data1['start_time'])
#data2=pd.read_csv('C:/python/MOBIKE_CUP_2017/Mobike_bj.csv')
#time2 = pd.to_datetime(data2['starttime'])
#userid1=data1['userid']
#userid2=data2['userid']
#all1_x,all1=time_gap(userid1,time1)
#all2_x,all2=time_gap(userid2,time2)
#
#all1_x,all1=log_pdf(all1_x,all1)
#all2_x,all2=log_pdf(all2_x,all2)



##data1=pd.read_csv('C:/python/Nanjing/mobai33.csv')
#data1=pd.read_csv('C:/python/Nanjing2/orange_Oct19_NJ33.csv')
#time1 = pd.to_datetime(data1['starttime'])
#userid1=data1['bikeid']
#x,y=time_gap(userid1,time1)
#x,y=log_pdf(x,y)
#print(x)
#print(y)

    

#取平均值
#all1_x=[1.4384498882876628, 2.0691380811147897, 2.9763514416313179, 4.2813323987193934,
#        6.1584821106602643, 8.8586679041008249, 12.742749857031335, 18.329807108324356,
#        26.366508987303583, 37.926901907322495, 54.555947811685172, 78.475997035146108, 
#        112.88378916846884, 162.37767391887209, 233.57214690901213, 335.98182862837808, 
#        483.29302385717523, 695.19279617756058]
#all1=[0.15573829877526493, 0.092448375830355353, 0.06080170001742994, 0.037082053380180058,
#      0.025649175198652386, 0.029547111336823851, 0.013892264043953393, 0.0080903815592121529,
#      0.0022565882799270172, 0.001067665183772422, 0.00040083378037674344, 0.00013642141924236,
#      5.0351680415774728e-05, 1.6305657660752666e-05, 5.0460749134406213e-06,
#      3.3744773526483093e-06, 6.6729336412383954e-06, 1.8911456560384279e-05]
#all2_x=[1.4384498882876628, 2.0691380811147897, 2.9763514416313179, 4.2813323987193934,
#        6.1584821106602643, 8.8586679041008249, 12.742749857031335, 18.329807108324356,
#        26.366508987303583, 37.926901907322495, 54.555947811685172, 78.475997035146108,
#        112.88378916846884, 162.37767391887209, 233.57214690901213, 335.98182862837808]
#all2=[0.046212390453992641, 0.026828204064717961, 0.022081789325327637, 0.017882209463054344,
#      0.011973991447080884, 0.016465124117125599, 0.011492240937878677, 0.020617130488040515,
#      0.0054195402834368899, 0.006574791883596086, 0.0037608353193310145, 0.0019315975675730805,
#      0.0011016228543328898, 0.00050858638614495607, 0.00011068583744119481, 1.8116091900138284e-05]


#人的间隔时间
all1_x=[1.4384498882876628, 2.0691380811147897, 2.976351441631318, 4.281332398719393,
        6.158482110660264, 8.858667904100825, 12.742749857031335, 18.329807108324356,
        26.366508987303583, 37.926901907322495, 54.55594781168517, 78.47599703514611, 
        112.88378916846884, 162.3776739188721, 233.57214690901213, 335.9818286283781,
        483.2930238571752, 695.1927961775606]
all1=[0.24693390576594032, 0.10190367542395895, 0.04659202089285838, 0.03950889281157999,
      0.028497122599075334, 0.030428926778469013, 0.012432541335367812, 0.00805343453674531, 
      0.0023423995660417315, 0.0010914822310623, 0.0003854163033921716, 0.0001387694291741092,
      4.9849235977639844e-05, 1.6261117545933283e-05, 4.188240412833384e-06, 
      1.5805922759033918e-06, 2.015423369089539e-06, 6.204399477185863e-08]
all2_x=[1.4384498882876628, 2.0691380811147897, 2.976351441631318, 4.281332398719393,
        6.158482110660264, 8.858667904100825, 12.742749857031335, 18.329807108324356,
        26.366508987303583, 37.926901907322495, 54.55594781168517, 78.47599703514611, 
        112.88378916846884, 162.3776739188721, 233.57214690901213, 335.9818286283781]
all2=[0.07327295956951516, 0.02957209983045569, 0.016921158278509117, 0.019052512806306866,
      0.01330351949429046, 0.016956515662294704, 0.01028469945893163, 0.02052297651300628,
      0.005625629150426514, 0.006721459707547617, 0.0036161803655393133, 0.0019648431553846747,
      0.001090630087626569, 0.0005143407664174748, 0.00011024304762453831, 1.7216972967194826e-06]


#车的间隔时间
all1_b_x=[1.4384498882876628, 2.0691380811147897, 2.976351441631318, 4.281332398719393,
        6.158482110660264, 8.858667904100825, 12.742749857031335, 18.329807108324356, 
        26.366508987303583, 37.926901907322495, 54.55594781168517, 78.47599703514611, 
        112.88378916846884, 162.3776739188721, 233.57214690901213, 335.9818286283781, 
        483.2930238571752, 695.1927961775606]
all1_b=[0.08016675350598472, 0.03176009279748958, 0.013141812661826496, 0.01098328235492936,
      0.008638177974359139, 0.012059623970817359, 0.006902759443114711, 0.0150323955020236,
      0.0045556092327184335, 0.005693028159845615, 0.0034353217342567455, 0.002300980776893493,
      0.0017697821274951539, 0.0011103335128480134, 0.0005538675038447193, 0.0002453204799900025,
      7.093010451306931e-05, 1.219284022321325e-06]
all2_b_x=[1.4384498882876628, 2.0691380811147897, 2.976351441631318, 4.281332398719393, 
        6.158482110660264, 8.858667904100825, 12.742749857031335, 18.329807108324356, 
        26.366508987303583, 37.926901907322495, 54.55594781168517, 78.47599703514611,
        112.88378916846884, 162.3776739188721, 233.57214690901213, 335.9818286283781]
all2_b=[0.06260436965082568, 0.03062201830455592, 0.020330239095598793, 0.02196046361420634,
      0.01652804985350593, 0.017686724918987953, 0.011161754691682967, 0.013687938383823085,
      0.008102801706362258, 0.006869177452399766, 0.0042207232511086165, 0.0024191809866363594,
      0.0011554507576607925, 0.0004936760049794026, 0.00012272256533156395, 1.8410755132516904e-06]
all3_b_x=[1.4384498882876628, 2.0691380811147897, 2.976351441631318, 4.281332398719393,
          6.158482110660264, 8.858667904100825, 12.742749857031335, 18.329807108324356,
          26.366508987303583, 37.926901907322495, 54.55594781168517, 78.47599703514611, 
          112.88378916846884]
all3_b=[0.2479091639250119, 0.07843417858163684, 0.03394872813725505, 0.024638628128714788,
        0.01389626652612102, 0.015908953517164752, 0.008425948956322656, 0.003983727590695268,
        0.0015579520800519753, 0.0006409110476087453, 0.0002820772065526961, 0.00016675289709160875,
        5.2459778150454134e-05]
all4_b_x=[1.4384498882876628, 2.0691380811147897, 2.976351441631318, 4.281332398719393, 
          6.158482110660264, 8.858667904100825, 12.742749857031335, 18.329807108324356,
          26.366508987303583, 37.926901907322495, 54.55594781168517, 78.47599703514611, 
          112.88378916846884, 162.3776739188721, 233.57214690901213, 335.9818286283781, 
          483.2930238571752, 695.1927961775606]
all4_b=[0.11934232866493077, 0.05308082447786007, 0.028583076821902493, 0.029373624280513842,
        0.021447078041730155, 0.02575175250809226, 0.01568478338979045, 0.013377665659850433, 
        0.0061047393870209314, 0.003980622778544628, 0.0018741331591206503, 0.0009549093412195825,
        0.00046069140777180243, 0.00019930066594220743, 8.26646246224595e-05, 3.4578504500260564e-05,
        1.1789533539943417e-05, 5.610523668421514e-07]


f_family='Arial'
fig	=	plt.figure(figsize=(6, 6))
ax	=	fig.add_subplot(1,	1,	1)
colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
        '#7f7f7f', '#bcbd22', '#17becf']

#正文
ax.plot(all1_x,all1,marker='o',markersize=10,markerfacecolor='none',c=colors[0],markeredgewidth=2,label=r'D1 (user)')
ax.plot(all2_x,all2,marker='o',markersize=10,markerfacecolor='none',c=colors[1],markeredgewidth=2,label=r'D2 (user)')

ax.plot(all1_b_x,all1_b,marker='s',markersize=10,markerfacecolor='none',c=colors[0],markeredgewidth=2,label=r'D1 (bike)')
ax.plot(all2_b_x,all2_b,marker='s',markersize=10,markerfacecolor='none',c=colors[1],markeredgewidth=2,label=r'D2 (bike)')
#ax.plot(all3_b_x,all3_b,marker='s',markersize=10,markerfacecolor='none',c=colors[2],markeredgewidth=2,label=r'nj (Mobike)')
#ax.plot(all4_b_x,all4_b,marker='s',markersize=10,markerfacecolor='none',c=colors[3],markeredgewidth=2,label=r'nj (DiDi)')


x_no=range(20,100)
popt, pcov = curve_fit(func4, all1_x[9:15],all1[9:15],maxfev = 10000)
print(popt)
popt_no=[popt[0]-25,popt[1]]
y777 = [func4(i, *popt_no) for i in x_no]
#ax1.errorbar(rank_x,rank_y,fmt="o",yerr = rank_std,label=r'user, $\xi$=%.2f'%-popt[1])
ax.plot([24,24],[0.00005,0.0002],'k-',linewidth=2)
ax.plot([24,60],[0.00005,0.00005],'k-',linewidth=2)
ax.plot(x_no,y777,'k-',linewidth=2)
ax.text(25, 0.00001,r'%.2f'%popt[1], size = 18,family=f_family)


#补充材料
#popt, pcov = curve_fit(func1, all1_x[:],all1[:],maxfev = 10000)
#y777 = [func1(i, *popt) for i in all1_x]
#ax.scatter(all1_x,all1,label=r'sh, slope=%.2f'%popt[1],marker='o',c='',s=100,edgecolor='#1f77b4',linewidths=2)
#ax.plot(all1_x,y777,color='#1f77b4',linestyle='--')
#popt, pcov = curve_fit(func3, all2_x[1:],all2[1:],maxfev = 10000)
#y888 = [func3(i, *popt) for i in all2_x]
#ax.scatter(all2_x,all2,label=r'bj, $\alpha$=%.2f'%popt[1],marker='s',c='',s=100,edgecolor='orange',linewidths=2)
#ax.plot(all2_x,y888,color='orange',linestyle='--')


ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim([0.00000001,1])
#ax.set_xlim([2,1000])
ax.legend(loc=3,handletextpad=0.1,prop={'size':18,'family':f_family})
ax.set_xlabel(r'waiting interval $\tau$ (h)',size=18,family=f_family)  
ax.set_ylabel(r'$\log_{10}P(\tau)$',size=18,family=f_family)
plt.yticks(fontproperties = f_family)
plt.xticks(fontproperties = f_family) 
plt.tick_params(labelsize=18)
#plt.savefig('C:/python/摩拜单车/draw2/1_d.pdf',bbox_inches='tight')    




     
        