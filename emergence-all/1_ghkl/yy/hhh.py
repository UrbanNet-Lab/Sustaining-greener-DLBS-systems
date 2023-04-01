import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import radians, cos, sin, asin, sqrt
from matplotlib.colors import LogNorm
from io import StringIO


pr=list(np.load("边界坐标.npy"))              #读取边界坐标
asr=list(np.load("边界坐标（1）.npy"))
br=list(np.load("边界坐标（2）.npy"))
h5=list(np.load("边界坐标（3）.npy"))
c=list(np.load("边界坐标（4）.npy"))
h2=list(np.load("边界坐标（5）.npy"))

pr1=[]                            #边界经纬度坐标
pr2=[]
for i in range(0,len(pr)):
    l=pr[i][1]
    pr1.append(l)
    m=pr[i][0]
    pr2.append(m)

aa1=[]
aa2=[]
for i in range(0,len(asr)):
    r=asr[i][0]
    r1=asr[i][1]
    aa1.append(r)
    aa2.append(r1)
    
b1=[]
b2=[]
for i in range(0,len(br)):
    r=br[i][0]
    r1=br[i][1]
    b1.append(r)
    b2.append(r1)
    
h3=[]
h4=[]
for i in range(0,len(h5)):
    r=h5[i][0]
    r1=h5[i][1]
    h3.append(r)
    h4.append(r1)
    
c1=[]
c2=[]
for i in range(0,len(c)):
    r=c[i][1]
    r1=c[i][0]
    c1.append(r)
    c2.append(r1)
    
h=[]
h1=[]
r=h2[0][0]
h.append(r)
r1=h2[1][0]
h.append(r1)

r3=h2[0][1]
h1.append(r3)
r4=h2[1][1]
h1.append(r4)



fig= plt.figure(figsize=(6,6))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel("longitude",fontdict={'size'   : 18,"family":"Arial"})#经度
ax.set_ylabel("latitude",fontdict={ 'size'   : 18,"family":"Arial"})#纬度

my_x_ticks = np.arange(120.8, 122.2, 0.3)
plt.xticks(my_x_ticks)
my_y_ticks = np.arange(30.6,31.8, 0.3)
plt.yticks(my_y_ticks)
plt.xlim(120.775,122.1)
plt.ylim(30.625,31.575)

fig.set_facecolor('white')
plt.plot(aa1,aa2,color="black",linewidth="3")
plt.plot(b1,b2,color="black",linewidth="3")
plt.plot(c1,c2,color="black",linewidth="3")
plt.plot(pr1,pr2,color="black",linewidth="3")
plt.plot(h,h1,color="black",linewidth="3")
plt.plot(h3,h4,color="black",linewidth="3")

plt.yticks(fontproperties = 'Arial')
plt.xticks(fontproperties = 'Arial')
plt.tick_params(labelsize=18)

plt.text(121.80,30.72, 'km',fontdict=dict(fontsize=18,family='Arial'))
plt.text(121.81,30.65, '23',fontdict=dict(fontsize=18,family='Arial'))
plt.text((122.1-120.775)*0.05+120.775,(31.575-30.625)*0.9+30.625, 'Shanghai',fontdict=dict(fontsize=18,family='Arial'))


