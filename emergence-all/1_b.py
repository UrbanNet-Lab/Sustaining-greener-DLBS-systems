#所有出行的小时的分布，每个小时内的出行量（订单数）
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import datetime
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr


#这是计算方法
##data=pd.read_csv('F:/python/MobikeData/lak3.csv')
##time1 = pd.to_datetime(data['start_time'])
#data=pd.read_csv('F:/python/MOBIKE_CUP_2017/Mobike_bj.csv')
#time1 = pd.to_datetime(data['starttime'])
#hour=[]
#for this in time1:
#    hour.append(this.hour)
#hours=pd.Series(hour).value_counts(normalize=True)
#x1, y1 = (list(t) for t in zip(*sorted(zip(hours.index+0.5, hours.values)))) 



##data=pd.read_csv('C:/python/Nanjing/mobai33.csv')
#data=pd.read_csv('C:/python/Nanjing2/orange_Oct19_NJ33.csv')
#time1 = pd.to_datetime(data['starttime'])
#hour=[]
#for this in time1:
#    hour.append(this.hour)
#hours=pd.Series(hour).value_counts(normalize=True)
#x, y = (list(t) for t in zip(*sorted(zip(hours.index+0.5, hours.values)))) 
#print(y)
 

 

x1=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5,
    13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5] 
y1=[0.009129297305335909, 0.0059867292049459245, 0.003856788123205889,
    0.002472375339254511, 0.002295287908939882, 0.0052891417032092936,
    0.02270045612243101, 0.06233379708715637, 0.08281332245370777, 
    0.05072723252460146, 0.03454770304883895, 0.03638804258708103,
    0.04102264566664123, 0.04051486458192691, 0.03827730130496804,
    0.042793519969787516, 0.0526566049697973, 0.09064234796408158,
    0.09858193082045291, 0.08745575260201116, 0.07291697241153945, 
    0.05942799781624782, 0.03811195447776819, 0.019057934006069893]
y2=[0.004093844200355122, 0.0019288490523186191, 0.001155889973985217, 
    0.0009055493042077879, 0.0013726798323491763, 0.00909936718008011,
    0.038305993723417434, 0.09991657461700458, 0.08999553516125036, 
    0.04629044163191147, 0.039033788247925014, 0.05322255543626378, 
    0.06140346915390015, 0.05073689195606392, 0.04208949291819796, 
    0.04780022814551761, 0.05569789558987488, 0.08544681938307801, 
    0.09066493836974027, 0.06491533581781393, 0.04679370380311351,
    0.03894442695214106, 0.021438968493207252, 0.008746761056282776]
y3=[0.00387607668796888, 0.002264517921644901, 0.001369824951375382,
    0.0008696860238955266, 0.0007661850514031676, 0.0014559599888858015, 
    0.011409419283134204, 0.04998680188941373, 0.08145804390108363, 
    0.05075715476521256, 0.04875798833009169, 0.06416296193387051, 
    0.06983676021116977, 0.06758127257571547, 0.06130661294804112,
    0.06268268963601001, 0.06773061961656016, 0.09500347318699638, 
    0.08896151708808002, 0.059428313420394556, 0.04611350375104196,
    0.03561405946096138, 0.02017227007502084, 0.008434287302028342]
y4=[0.007699915021126281, 0.004643488456789575, 0.0034235826737847577,
    0.0022244034453455562, 0.0018853762313794115, 0.005546574048577122,
    0.022033067737341257, 0.0677032904885841, 0.11929834690922657,
    0.05270318685581128, 0.03934122126781373, 0.048380219760697125,
    0.04974669189384451, 0.05073416417207778, 0.04758668881447069, 
    0.049686732932422546, 0.057203810132916436, 0.09271876138109916,
    0.0875371227396952, 0.05281718290592217, 0.046930101175195645, 
    0.046389730288306376, 0.029803304997172306, 0.013963035670400407]


f_family='Arial'
alpha_xy=0.6
#alpha_xy=1
fig	=	plt.figure(figsize=(6, 6))
ax	=	fig.add_subplot(1,	1,	1)    
ax.plot(x1, y1,marker='o',label='D1 (SH)')
ax.plot(x1, y2,marker='o',label='D2 (BJ)')
ax.plot(x1, y4,marker='o',label='D3 (NJ)')
#ax.plot(x1, y1,marker='o',markersize=7,markeredgewidth=0,alpha=alpha_xy,label='D1 (SH)')
#ax.plot(x1, y2,marker='o',markersize=7,markeredgewidth=0,alpha=alpha_xy,label='D2 (BJ)')
#ax.plot(x1, y4,marker='o',markersize=7,markeredgewidth=0,alpha=alpha_xy,label='D3 (NJ)')
#ax.plot(x1, y3,marker='o',alpha=alpha_xy,label='D4 (NJ)')
#ax.scatter([1,2,3],[0.1,0.1,0.1],s=49,linewidths=0)



#ax.set_xlim([-2,25])
#ax.legend(loc=2,handletextpad=0.1,frameon=False,prop={'size':18,'family':f_family})
ax.legend(bbox_to_anchor=(0.4, 0.3),handletextpad=0.1,frameon=False,prop={'size':18,'family':f_family})
ax.set_xlabel('Time(h)',size=18,family=f_family)  
ax.set_ylabel(r'$P$',size=18,family=f_family)
plt.yticks(fontproperties = f_family)
plt.xticks(fontproperties = f_family)
plt.tick_params(labelsize=18) 
#plt.savefig('C:/python/摩拜单车/draw2/1_b.pdf',bbox_inches='tight')




