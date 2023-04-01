#拟合，用明确函数形式拟合数据x和y，二维函数，相当于一种预测，图为直线时用幂律，直线降落速度一直一样。
#图前面降落较慢后面较快用幂指，有无c参数会让后面降落快慢一些，两者形式都可试下，可以少拟合前面几个点，从第2,3,4点拟合效果
#可能好些，少拟合后面的点基本没影响，或者跳着点拟合，避开和规则函数形式太远的点，其实就是多试下看看哪种像。
#图前面降落较慢后面很快用泊松这些，这些在后面降落会更快些。
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import factorial
from sklearn.metrics import r2_score

def pdf1(data,step):
    all_num=len(data)    
    x_range=np.arange(0,130,step)
#    y=np.zeros((len(x_range)-1), dtype=np.int)
    y=np.zeros(len(x_range)-1)
#    x=x_range[:-1]
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

def cdf1(data,step):
    all_num=len(data)    
    x_range=np.arange(0,130,step)
#    y=np.zeros((len(x_range)), dtype=np.int)
    y=np.zeros(len(x_range))
    x=x_range
        
    for data1 in data:
        a=int(data1//step) 
        for i in range(a+1):
            y[i]=y[i]+1
    y=y/all_num    
    x1=[]
    y1=[]
    for i in range(len(x)):
        if(y[i]!=0):
            x1.append(x[i])
            y1.append(y[i])
        
    return x1,y1

#直线   
def func(x, a, b):
    return a * x + b

#泊松    
def func1(x, a):
    return (a**x/factorial(x)) * np.exp(-a)

#normal
def func2(x,mu,sigma):
    return np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))

#指数
def func3(x, a, b):
    return a * np.exp(-b * x) 

#log_normal
def func4( x, a, mu, sigma ):
    return a / x * 1. / (sigma * np.sqrt( 2. * np.pi ) ) * np.exp( -( np.log( x ) - mu )**2 / ( 2. * sigma**2 ) )    

#幂律 
def func5(x, a, b):
    return a*pow(x,b)

#幂律和指数
def func6(x, a,b1,b2,c):
    return a*pow(x+c,-b1)*np.exp((-b2)*x)

def func7(x, a,b1,b2):
    return a*pow(x,-b1)*np.exp((-b2)*x)

def r2(y_true,y_no):
    y_true = np.array(y_true)
    y_no = np.array(y_no)  
    ybar=np.mean(y_true)
    ssreg = np.sum((y_no - y_true)**2)
    sstot = np.sum((y_true - ybar)**2)
    return  1-ssreg/sstot

data1=pd.read_csv('F:/python/MobikeData/lak-user.csv')
data2=pd.read_csv('F:/python/MobikeData/lak-bike.csv')

d_user = data1['user_distance_aver']
d_bike = data2['bike_distance_aver']
DELTA = 1
left=0
right=200

left2=1
right2=100

fig	=	plt.figure(figsize=(10, 9))
x,y=cdf1(d_user,DELTA)
#x,y=pdf1(d_bike,DELTA)
x_int=x

ax	=	fig.add_subplot(3,	2,	1)    
ax.scatter(x_int, y,label='user',alpha=0.6)
x_good=x_int[:]
y_good=y[:]
popt, pcov = curve_fit(func1, x_good, y_good,maxfev = 10000)
y777 = [func1(i, *popt) for i in x_good]
R2=r2_score(y_good,y777)
print(R2)
ax.plot(x_good,y777,'b--')
ax.text(6, 0.2,'$R^{2}=%.4f$'%R2, size = 18)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim([np.min(y)*0.5,np.max(y)])

ax	=	fig.add_subplot(3,	2,	2)    
ax.scatter(x_int, y,label='user',alpha=0.6)
x_good=x_int[:]
y_good=y[:]
#for j in range(len(x_int)):
##    if( x_int[j]<right and x_int[j]>left ):
#    if(  x_int[j]>left ):
#        x_good.append(x_int[j])
#        y_good.append(y[j])
popt, pcov = curve_fit(func2, x_good, y_good,maxfev = 10000)
y777 = [func2(i, *popt) for i in x_good]
R2=r2_score(y_good,y777)
print(R2)
ax.plot(x_good,y777,'b--')
ax.text(6, 0.2,'$R^{2}=%.4f$'%R2, size = 18)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim([np.min(y)*0.5,np.max(y)])

ax	=	fig.add_subplot(3,	2,	3)    
ax.scatter(x_int, y,label='user',alpha=0.6)
x_good=x_int[:]
y_good=y[:]
popt, pcov = curve_fit(func3, x_good, y_good,maxfev = 10000)
y777 = [func3(i, *popt) for i in x_good]
R2=r2_score(y_good,y777)
print(R2)
ax.plot(x_good,y777,'b--')
ax.text(6, 0.2,'$R^{2}=%.4f$'%R2, size = 18)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim([np.min(y)*0.5,np.max(y)])

#ax	=	fig.add_subplot(3,	2,	4)    
#ax.scatter(x_int, y,label='user',alpha=0.6)
#x_good=x_int[:]
#y_good=y[:]
##for j in range(len(x_int)):
##    if( x_int[j]<right and x_int[j]>left ):
###    if(  x_int[j]>left ):
##        x_good.append(x_int[j])
##        y_good.append(y[j])
#popt, pcov = curve_fit(func4, x_good, y_good,maxfev = 10000)
#y777 = [func4(i, *popt) for i in x_good]
#R2=r2_score(y_good,y777)
#print(R2)
#ax.plot(x_good,y777,'b--')
#ax.text(6, 0.2,'$R^{2}=%.4f$'%R2, size = 18)
#ax.set_yscale('log')
#ax.set_xscale('log')
#ax.set_ylim([np.min(y)*0.5,np.max(y)])

ax	=	fig.add_subplot(3,	2,	5)    
ax.scatter(x_int, y,label='user',alpha=0.6)
x_good=x_int[2:-6]
y_good=y[2:-6]
popt, pcov = curve_fit(func5, x_good, y_good,maxfev = 10000)
y777 = [func5(i, *popt) for i in x_good]
R2=r2_score(y_good,y777)
print(R2)
ax.plot(x_good,y777,'b--')
ax.text(6, 0.2,'$R^{2}=%.4f$'%R2, size = 18)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim([np.min(y)*0.5,np.max(y)])

#ax	=	fig.add_subplot(3,	2,	6)    
#ax.scatter(x_int, y,label='user',alpha=0.6)
#x_good=x_int[:]
#y_good=y[:]
##for j in range(len(x_int)):
##    if( x_int[j]<right and x_int[j]>left ):
###    if(  x_int[j]>left ):
##        x_good.append(x_int[j])
##        y_good.append(y[j])
#popt, pcov = curve_fit(func6, x_good, y_good,maxfev = 10000)
#y777 = [func6(i, *popt) for i in x_good]
#R2=r2_score(y_good,y777)
#print(R2)
#ax.plot(x_good,y777,'b--')
#ax.text(6, 0.2,'$R^{2}=%.4f$'%R2, size = 18)
#ax.set_yscale('log')
#ax.set_xscale('log')
#ax.set_ylim([np.min(y)*0.5,np.max(y)])

plt.subplots_adjust(wspace=0,	hspace=0)




fig	=	plt.figure(figsize=(10, 9))
#x,y=pdf1(d_user,DELTA)
x,y=cdf1(d_bike,DELTA)
x_int=x

ax	=	fig.add_subplot(3,	2,	1)    
ax.scatter(x_int, y,label='user',alpha=0.6)
x_good=x_int[:]
y_good=y[:]
popt, pcov = curve_fit(func1, x_good, y_good,maxfev = 10000)
y777 = [func1(i, *popt) for i in x_good]
R2=r2_score(y_good,y777)
print(R2)
ax.plot(x_good,y777,'b--')
ax.text(6, 0.1,'$R^{2}=%.4f$'%R2, size = 18)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim([np.min(y)*0.5,np.max(y)])

ax	=	fig.add_subplot(3,	2,	2)    
ax.scatter(x_int, y,label='user',alpha=0.6)
x_good=x_int[:]
y_good=y[:]
popt, pcov = curve_fit(func2, x_good, y_good,maxfev = 10000)
y777 = [func2(i, *popt) for i in x_good]
R2=r2_score(y_good,y777)
print(R2)
ax.plot(x_good,y777,'b--')
ax.text(6, 0.1,'$R^{2}=%.4f$'%R2, size = 18)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim([np.min(y)*0.5,np.max(y)])

ax	=	fig.add_subplot(3,	2,	3)    
ax.scatter(x_int, y,label='user',alpha=0.6)
x_good=x_int[:]
y_good=y[:]
popt, pcov = curve_fit(func3, x_good, y_good,maxfev = 10000)
y777 = [func3(i, *popt) for i in x_good]
R2=r2_score(y_good,y777)
print(R2)
ax.plot(x_good,y777,'b--')
ax.text(6, 0.1,'$R^{2}=%.4f$'%R2, size = 18)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim([np.min(y)*0.5,np.max(y)])

#ax	=	fig.add_subplot(3,	2,	4)    
#ax.scatter(x_int, y,label='user',alpha=0.6)
#x_good=x_int[:]
#y_good=y[:]
#popt, pcov = curve_fit(func4, x_good, y_good,maxfev = 10000)
#y777 = [func4(i, *popt) for i in x_good]
#R2=r2_score(y_good,y777)
#print(R2)
#ax.plot(x_good,y777,'b--')
#ax.text(6, 0.1,'$R^{2}=%.4f$'%R2, size = 18)
#ax.set_yscale('log')
#ax.set_xscale('log')
#ax.set_ylim([np.min(y)*0.5,np.max(y)])

ax	=	fig.add_subplot(3,	2,	5)    
ax.scatter(x_int, y,label='user',alpha=0.6)
x_good=x_int[10:-2]
y_good=y[10:-2]
popt, pcov = curve_fit(func5, x_good, y_good,maxfev = 10000)
y777 = [func5(i, *popt) for i in x_int]
ax.plot(x_int,y777,'b--')
popt, pcov = curve_fit(func5, x_int, y,maxfev = 10000)
y888 = [func5(i, *popt) for i in x_int]
#R2=r2_score(y_good,y777)
#print(R2)
ax.plot(x_int,y888,'b--')
#ax.text(6, 0.1,'$R^{2}=%.4f$'%R2, size = 18)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim([np.min(y)*0.5,np.max(y)])

#ax	=	fig.add_subplot(3,	2,	6)    
#ax.scatter(x_int, y,label='user',alpha=0.6)
#x_good=x_int[:]
#y_good=y[:]
#popt, pcov = curve_fit(func6, x_good, y_good,maxfev = 10000)
#y777 = [func6(i, *popt) for i in x_good]
#R2=r2_score(y_good,y777)
#print(R2)
#ax.plot(x_good,y777,'b--')
#ax.text(6, 0.1,'$R^{2}=%.4f$'%R2, size = 18)
#ax.set_yscale('log')
#ax.set_xscale('log')
#ax.set_ylim([np.min(y)*0.5,np.max(y)])

plt.subplots_adjust(wspace=0,	hspace=0)
#samples1 = np.random.rand(2,10,size=1000)
#samples2 = np.random.rand(2,10,size=1000)
#print(r2(samples1,samples2))


