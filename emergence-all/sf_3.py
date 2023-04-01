#上海前两周数据
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def log_pdf(x,y):
#    bins=np.logspace(0, 4, 20,base=2)
    bins=np.logspace(0, 3, 30)
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
#            index_this=bins2.index(key)
#            y_new.append(np.sum(value)/widths[index_this])
            y_new.append(np.mean(value))
            x_new.append(key)
        
    return x_new,y_new

#幂律 
def func4(x, a, b):
    return a*pow(x,b)


#user_uni=np.loadtxt('C:/python/MobikeData/uni_user2.txt',delimiter=',')
#bike_uni=np.loadtxt('C:/python/MobikeData/uni_bike2.txt',delimiter=',')
##user_uni=np.loadtxt('C:/python/MobikeData/uni_user.txt',delimiter=',')
##bike_uni=np.loadtxt('C:/python/MobikeData/uni_bike.txt',delimiter=',')
#unique_t_x1=user_uni[:,0]
#unique_t_y1=user_uni[:,1]
#unique_t_x2=bike_uni[:,0]
#unique_t_y2=bike_uni[:,1]
#unique_t_x1,unique_t_y1=log_pdf(unique_t_x1,unique_t_y1)
#unique_t_x2,unique_t_y2=log_pdf(unique_t_x2,unique_t_y2)


unique_t_x1=[1.0, 1.6102620275609394, 2.592943797404667, 3.2903445623126686, 4.1753189365604015,
             5.298316906283708, 6.7233575364993365, 8.531678524172808, 10.826367338740546,
             13.73823795883263, 17.43328822199988, 22.122162910704493, 28.072162039411772,
             35.62247890262442, 45.20353656360243, 57.36152510448679, 72.7895384398315, 
             92.36708571873861, 117.21022975334806, 148.73521072935117, 188.73918221350976,
             239.5026619987486, 303.9195382313198]
unique_t_y1=[2.061002661934339, 2.2801136363636365, 2.486202104427239, 2.5833887043189367, 
             2.763069544364509, 2.8906326932545547, 2.966137965760322, 3.0848470514033424, 
             3.407924050632911, 3.676418715634402, 4.0296541762256775, 4.657113415040245, 
             5.423491714570667, 6.006407974665564, 6.821250784046978, 7.971884820939567,
             9.239916546931786, 10.58201775858266, 12.313303451848775, 14.657051835570975, 
             17.59895556952108, 21.090388405397473, 24.67705592653728]
unique_t_std1=[0.5944408410677576, 0.7288315591631797, 0.8714196396632223, 0.9627833734421148,
               1.0278977533425449, 1.1323486476640712, 1.1942466971474348, 1.2991483166921802, 
               1.436698657410933, 1.4807226883683913, 1.6319033619467083, 1.8622023140358757, 
               2.1481129373456893, 2.4094543377702853, 2.716435753149918, 3.1187249652100926, 
               3.532473834841295, 4.03027181807063, 4.613948861070844, 5.355905296221825, 
               6.210142177132595, 7.2387601222877365, 7.919143025460066]
unique_t_x2=[1.0, 1.6102620275609394, 2.592943797404667, 3.2903445623126686, 4.1753189365604015,
             5.298316906283708, 6.7233575364993365, 8.531678524172808, 10.826367338740546, 
             13.73823795883263, 17.43328822199988, 22.122162910704493, 28.072162039411772, 
             35.62247890262442, 45.20353656360243, 57.36152510448679, 72.7895384398315,
             92.36708571873861, 117.21022975334806, 148.73521072935117, 188.73918221350976,
             239.5026619987486, 303.9195382313198]
unique_t_y2=[1.9241953308468018, 1.9568467550118926, 1.989994441356309, 2.0085403726708075, 
             2.0673469387755103, 2.055487804878049, 2.000912408759124, 1.985611510791367, 
             1.9816844391785151, 1.9960656859391037, 1.9511129895745283, 1.7359258236370112, 
             2.1800858244235948, 2.1433595493083444, 1.9583240386658167, 2.11875092452886,
             2.3771448504732815, 2.317727032152781, 2.402903053949451, 2.53312108463674, 
             2.656941373614156, 2.9435066651623374, 3.378399258343634]
unique_t_std2=[0.3817920125278503, 0.4418321052206451, 0.5193958413304212, 0.5810851755890027, 
               0.6006417503691758, 0.6502582084496565, 0.5576847489057363, 0.44774382340065716,
               0.48212374487401216, 0.5384745781091272, 0.6999513133032748, 0.7583709471419053, 
               0.7054882794347825, 0.7383629881419006, 0.874343486903829, 0.8977380033166386, 
               0.9429669481389568, 1.016680825270023, 1.12070000244533, 1.1972761784647687, 
               1.3383326374195528, 1.518639206622078, 1.7795779301019283]



f_family='Arial'
alpha_xy=1
fig7	=	plt.figure(figsize=(8, 6))
ax7	=	fig7.add_subplot(1,	1,	1)
plt.yticks(fontproperties = f_family)
plt.xticks(fontproperties = f_family) 
plt.tick_params(labelsize=18)

popt, pcov = curve_fit(func4, unique_t_x1[10:],unique_t_y1[10:])
y777 = [func4(i, *popt) for i in unique_t_x1[0:]]
#ax7.scatter(unique_t_x1,unique_t_y1,label=r'user, $\mu$=%.2f'%popt[1],alpha=1)
ax7.errorbar(unique_t_x1,unique_t_y1,fmt="o",yerr = unique_t_std1,label=r'user, $\epsilon$=%.2f'%popt[1],alpha=alpha_xy)
ax7.plot(unique_t_x1[0:],y777,'b-')
popt, pcov = curve_fit(func4, unique_t_x2[10:],unique_t_y2[10:])
y888= [func4(i, *popt) for i in unique_t_x2[0:]]
#ax7.scatter(unique_t_x2,unique_t_y2,label=r'bike, $\mu$=%.2f'%popt[1],alpha=1)
ax7.errorbar(unique_t_x2,unique_t_y2,fmt="o",yerr = unique_t_std2,label=r'bike, $\epsilon$=%.2f'%popt[1],alpha=alpha_xy)
ax7.plot(unique_t_x2[0:],y888,c='orange',linestyle='-')
ax7.plot([24,24],[1,100],color='k',linestyle='--')
ax7.plot([168,168],[1,100],color='k',linestyle='--')
ax7.plot([720,720],[1,100],color='k',linestyle='--')
ax7.text(19, 70,r'Day', size = 18,family=f_family)
ax7.text(120, 70,r'Week', size = 18,family=f_family)
ax7.text(400, 70,r'Month', size = 18,family=f_family)
ax7.set_yscale('log')
ax7.set_xscale('log')
ax7.set_xlim([1,1000])
ax7.set_ylim([1.6,100])
ax7.legend(loc=2,handletextpad=0.1,prop={'size':18,'family':f_family})
ax7.set_xlabel('t(h)',size=18,family=f_family) 
ax7.set_ylabel(r'S(t)',size=18,family=f_family)
#plt.savefig('C:/python/摩拜单车/draw2/sf_3.pdf',bbox_inches='tight')



