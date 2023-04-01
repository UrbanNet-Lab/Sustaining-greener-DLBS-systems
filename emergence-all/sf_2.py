#统计车的终点和下一次起点不一样的情况，终点和下次起点间距离分布
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy.optimize import curve_fit
from math import radians, cos, sin, asin, sqrt



def func1(x, a,b1,b2,c):
    return a*pow(x+c,-b1)*np.exp((-b2)*x)
#def func1(x, a,b1,b2,c):
#    return a*pow(x+c,-b1)*np.exp(-x/b2)

def func2(x, a, b):
    return a*pow(x,-b)

def func3(x, a,b1,b2):
    return a*pow(x,-b1)*np.exp((-b2)*x)

def geodistance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) 
    #lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2 
    dis=2*asin(sqrt(a))*6371*1000 # 地球平均半径，6371km
    dis=round(dis/1000,3)
    return dis

def pdf1(data,step):
    all_num=len(data)    
    x_range=np.arange(0,100,step)
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

def move(bikeid_random1,startx,starty,endx,endy):
    dict1={}
    for i,j,k,m,n in zip(bikeid_random1,startx,starty,endx,endy):
        if i not in dict1.keys():
            dict1[i]=[[j,k,m,n]]
        else:
            dict1[i].append([j,k,m,n])
    dis_all=[]
    for key,value in dict1.items():
        for i in range(len(value)-1):
            if((round(value[i][2],3)!=round(value[i+1][0],3)) or (round(value[i][3],3)!=round(value[i+1][1],3))):
                dis=geodistance(value[i][2],value[i][3],value[i+1][0],value[i+1][1])
                dis_all.append(dis)
    move_x,move_y=pdf1(dis_all,1)
    move_count=len(dis_all)
    return move_x,move_y,move_count

def log_pdf(x,y,left,right):
    bins=np.logspace(left, right, 20)
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


#data=pd.read_csv('F:/python/MOBIKE_CUP_2017/Mobike_bj4.csv')
#startx=data['start_x']
#starty=data['start_y']
#endx=data['end_x']
#endy=data['end_y']
#bikeid=data['bikeid']
#dis_x,dis_y,nonono=move(bikeid,startx,starty,endx,endy)
#np.savetxt('F:/python/MOBIKE_CUP_2017/move3.txt',np.array([dis_x,dis_y]).T,delimiter=',')
    

##data=pd.read_csv('C:/python/Nanjing/mobai33.csv')
#data=pd.read_csv('C:/python/Nanjing2/orange_Oct19_NJ33.csv')
#startx=data['startpositionx']
#starty=data['startpositiony']
#endx=data['endpositionx']
#endy=data['endpositiony']
#bikeid=data['bikeid']
#dis_x,dis_y,nonono=move(bikeid,startx,starty,endx,endy)
#print(nonono)
#print(len(data))
##np.savetxt('C:/python/Nanjing/move.txt',np.array([dis_x,dis_y]).T,delimiter=',')
#np.savetxt('C:/python/Nanjing2/move.txt',np.array([dis_x,dis_y]).T,delimiter=',')
    

#data=pd.read_csv('C:/python/Singapore/bike_sgp.csv')
#startx=data['lng0']
#starty=data['lat0']
#endx=data['lng1']
#endy=data['lat1']
#bikeid=data['bike_id']
#dis_x,dis_y,nonono=move(bikeid,startx,starty,endx,endy)
#print(nonono)
#print(len(data))
#np.savetxt('C:/python/Singapore/move.txt',np.array([dis_x,dis_y]).T,delimiter=',')
    





f_family='Arial'
fig	=	plt.figure(figsize=(6, 6))
ax1	=	fig.add_subplot(1,	1,	1)
plt.yticks(fontproperties = f_family)
plt.xticks(fontproperties = f_family) 
plt.tick_params(labelsize=18)


#cdf
#move_bj=np.loadtxt('C:/python/MOBIKE_CUP_2017/move.txt',delimiter=',')
#move_sh=np.loadtxt('C:/python/MobikeData/move.txt',delimiter=',')
#bj_x=move_bj[:,0]-0.5
#bj_y=move_bj[:,1]
#sh_x=move_sh[:,0]-0.5
#sh_y=move_sh[:,1]
#bj_cdf=[1.0]
#for i in range(len(bj_x)-1):
#    bj_cdf.append(bj_cdf[i]-bj_y[i])
#sh_cdf=[1.0]
#for i in range(len(sh_x)-1):
#    sh_cdf.append(sh_cdf[i]-sh_y[i])
#g=1
#popt, pcov = curve_fit(func1, sh_x[g:-30], sh_cdf[g:-30],maxfev = 1000000)
#y888 = [func1(i, *popt) for i in sh_x]
#ax1.scatter(sh_x,sh_cdf,label=r'sh, $\alpha$=%.2f'%popt[1])
#ax1.plot(sh_x,y888,color='blue',linestyle='-')
#
#popt, pcov = curve_fit(func1, bj_x[0:], bj_cdf[0:],maxfev = 10000)
#y777 = [func1(i, *popt) for i in bj_x]
#ax1.scatter(bj_x,bj_cdf,label=r'bj, $\alpha$=%.2f'%popt[1])
#ax1.plot(bj_x,y777,color='orange',linestyle='-')
#
#ax1.legend(loc=3,handletextpad=0.1,prop={'size':18,'family':f_family})
#ax1.set_yscale('log')
#ax1.set_xscale('log')
#ax1.set_ylim([0.0000001,1])
##ax1.set_xlim([0.3,100])
#ax1.set_xlabel('log'+r'$_{10}d$'+'(km)',size=18,family=f_family)  
#ax1.set_ylabel('log'+r'$_{10}p(d)$',size=18,family=f_family)
            

 
    

#pdf
move_bj=np.loadtxt('C:/python/MOBIKE_CUP_2017/move.txt',delimiter=',')
move_sh=np.loadtxt('C:/python/MobikeData/move.txt',delimiter=',')
move_nj1=np.loadtxt('C:/python/Nanjing/move.txt',delimiter=',')
move_nj2=np.loadtxt('C:/python/Nanjing2/move.txt',delimiter=',')
move_xm=np.loadtxt('C:/python/Xiamen/move.txt',delimiter=',')
move_sgp=np.loadtxt('C:/python/Singapore/move.txt',delimiter=',')
bj_x=move_bj[:,0]
bj_y=move_bj[:,1]
sh_x=move_sh[:,0]
sh_y=move_sh[:,1]    
nj_x1=move_nj1[:,0]
nj_y1=move_nj1[:,1]
nj_x2=move_nj2[:,0]
nj_y2=move_nj2[:,1] 
xm_x=move_xm[:,0]
xm_y=move_xm[:,1]
sgp_x=move_sgp[:,0]
sgp_y=move_sgp[:,1]
#bj_x=move_bj[:,0]+0.5
#bj_y=move_bj[:,1]
#sh_x=move_sh[:,0]+0.5
#sh_y=move_sh[:,1]    
#nj_x1=move_nj1[:,0]+0.5
#nj_y1=move_nj1[:,1]
#nj_x2=move_nj2[:,0]+0.5
#nj_y2=move_nj2[:,1] 
#xm_x=move_xm[:,0]+0.5
#xm_y=move_xm[:,1]
sh_x,sh_y=log_pdf(sh_x,sh_y,0,2)
bj_x,bj_y=log_pdf(bj_x,bj_y,0,2)
nj_x1,nj_y1=log_pdf(nj_x1,nj_y1,0,2)
nj_x2,nj_y2=log_pdf(nj_x2,nj_y2,0,2)
xm_x,xm_y=log_pdf(xm_x,xm_y,0,2)
sgp_x,sgp_y=log_pdf(sgp_x,sgp_y,0,2)
g=2
g2=4
colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
        '#7f7f7f', '#bcbd22', '#17becf']
#popt, pcov = curve_fit(func1, sh_x[:g]+sh_x[g2:], sh_y[:g]+sh_y[g2:],maxfev = 1000000)
#y888 = [func1(i, *popt) for i in sh_x]

#ax1.scatter(sh_x,sh_y,label=r'sh')
#popt, pcov = curve_fit(func1, bj_x[:], bj_y[:],maxfev = 10000)
#y777 = [func1(i, *popt) for i in bj_x]
#ax1.scatter(bj_x,bj_y,label=r'bj')
#ax1.plot(bj_x,y777,color='orange',linestyle='-',label=r'$\alpha$=%.2f'%popt[1])
#popt, pcov = curve_fit(func2, nj_x1[1:], nj_y1[1:],maxfev = 10000)
#y888 = [func2(i, *popt) for i in nj_x1]
#ax1.scatter(nj_x1,nj_y1,label=r'nj (Mobike)')
#ax1.plot(nj_x1,y888,color='g',linestyle='-',label=r'$\alpha$=%.2f'%popt[1])
#popt, pcov = curve_fit(func1, nj_x2[:], nj_y2[:],maxfev = 1000000)
#y999 = [func1(i, *popt) for i in nj_x2]
#ax1.scatter(nj_x2,nj_y2,label=r'nj (DiDi)')
#ax1.plot(nj_x2,y999,color='r',linestyle='-',label=r'$\alpha$=%.2f'%popt[1])
#ax1.scatter(sh_x,sh_y,label=r'sh')
#ax1.scatter(bj_x,bj_y,label=r'bj')
#ax1.scatter(nj_x1,nj_y1,label=r'nj (Mobike)')
#ax1.scatter(nj_x2,nj_y2,label=r'nj (DiDi)')
ax1.plot(sh_x,sh_y,marker='s',markersize=10,markerfacecolor='none',markeredgewidth=2,label=r'D1 (SH)')
ax1.plot(bj_x,bj_y,marker='s',markersize=10,markerfacecolor='none',markeredgewidth=2,label=r'D2 (BJ)')
ax1.plot(nj_x2,nj_y2,marker='s',markersize=10,markerfacecolor='none',markeredgewidth=2,label=r'D3 (NJ)')
#ax1.plot(nj_x1,nj_y1,marker='s',markersize=10,markerfacecolor='none',markeredgewidth=2,label=r'D4 (NJ)')
ax1.plot(xm_x,xm_y,marker='s',markersize=10,markerfacecolor='none',markeredgewidth=2,label=r'D7 (XM)')
ax1.plot(sgp_x,sgp_y,marker='s',markersize=10,markerfacecolor='none',markeredgewidth=2,label=r'D8 (SGP)')


ax1.legend(loc=3,handletextpad=0.1,prop={'size':18,'family':f_family})
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_ylim([0.00000001,1])
ax1.set_xlim([1,80])
#ax1.set_xlim([0.5,80])
ax1.set_xlabel('log'+r'$_{10}d$'+'(km)',size=18,family=f_family)  
ax1.set_ylabel('log'+r'$_{10}P$($d$)',size=18,family=f_family)
#plt.savefig('C:/python/摩拜单车/draw2/sf_2.pdf',bbox_inches='tight')





