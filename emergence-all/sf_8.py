import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm



#user_all=data["userid"].value_counts()
#user_uni=list(user_all.index)
#good_user=user_uni[:10]
#print(good_user)
#good_user=[2240, 7207, 4906, 3369, 6044, 5305, 3984, 1455, 5650, 7045]



def geodistance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) 
    #lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2 
    dis=2*asin(sqrt(a))*6371*1000 # 地球平均半径，6371km
    dis=round(dis/1000,3)
    return dis

def gyration(bikeid_random1,endx,endy):
    dict_gy={}
    dict7={}
    for i,j,k in zip(bikeid_random1,endx,endy):
        if i not in dict_gy.keys():
            dict_gy[i]=[[j,k]]
        else:
            dict_gy[i].append([j,k])
    bike_end=[]  
    for key,value in dict_gy.items():
        bike_point1_array=np.array(value)
        endx_mid=np.mean(bike_point1_array[:,0])
        endy_mid=np.mean(bike_point1_array[:,1])
        dis1=0
        for point1 in value:
            dis1=dis1+geodistance(point1[0],point1[1],endx_mid,endy_mid)    
        gy_this=dis1/len(value)
        if(gy_this<100):
            dict7[key]=gy_this
    return dict7

def pdf1(data,step):
    all_num=len(data)    
    x_range=np.arange(0,240,step)
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

def log_pdf1(x,y):
    bins=np.logspace(-3, 1.2, 20)
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

def log_pdf(x,y):
    bins=np.logspace(0, 4, 20)
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


def gy_time(bikeid,endx,endy,time1):
    ggg=40
    bike_all=bikeid.value_counts()
#    good_bike=bike_all.index[:int(len(bike_all)*0.1)]
    good_bike=[]
    for i,j in zip(bike_all.index,bike_all.values):
        if(j>ggg):
            good_bike.append(i)
    print(ggg)
    dict2={}
    dict_t={}
    for i,j,k,z in zip(bikeid,endx,endy,time1):
        if(i in good_bike):
            if i not in dict2.keys():
                dict2[i]=[[j,k]]
            else:
                dict2[i].append([j,k])
            if i not in dict_t.keys():
                dict_t[i]=[z]
            else:
                dict_t[i].append(z)
    u_all2=0  
    for key,value in dict_t.items():
        if(len(value)>1):
            u_all2=u_all2+1
            user_this=[]
            for m in range(1,len(value)):
                user_this.append(((time.mktime(value[m].timetuple())-time.mktime(value[0].timetuple()))/60)//60)
            dict_t[key]=user_this
    dict_all2={}    
    for key,value in dict2.items():
        if(len(value)>1):
            trip=[]
            for m in range(len(value)):
                value_this=value[:m+1]
                bike_point1_array=np.array(value_this)
                endx_mid=np.mean(bike_point1_array[:,0])
                endy_mid=np.mean(bike_point1_array[:,1])
                dis1=0
                for point1 in value_this:
                    dis1=dis1+geodistance(point1[0],point1[1],endx_mid,endy_mid)                   
                trip.append(dis1/len(value_this))
#            if(len(trip)>200):
#                trip=trip[:200]  
            new_x,new_y=log_pdf(dict_t[key],trip[1:])
#            new_x,new_y=dict_t[key],trip[1:]
            for i,j in zip(new_x,new_y):
                if i not in dict_all2.keys():
                    dict_all2[i]=[j]
                else:
                    dict_all2[i].append(j)  
    all2=[]
    b_x=[]
    bar2=[]
    for key,value in dict_all2.items():
        all2.append(np.mean(value))
#        all2.append(np.sum(value)/u_all2)
        bar2.append(np.std(value))
        b_x.append(key)
    b_x,all2,bar2 = (list(t) for t in zip(*sorted(zip(b_x,all2,bar2)))) 
    return b_x,all2,bar2


def func1(x, a,b1,b2,c):
    return a*pow(x+c,-b1)*np.exp((-b2)*x)

def func2(x, a,b1,b2):
    return a*pow(x,-b1)*np.exp((-b2)*x)

def func4(x, a, b):
    return a*pow(x,b)

def func7(x,mu,sigma):
    return np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))

def func8(x, a, b):
    return a * np.exp(-b * x) 



data1=pd.read_csv('C:/python/MobikeData/lak3.csv')
data2=pd.read_csv('C:/python/MobikeData/bikes_group_combine.txt')
#data1=pd.read_csv('C:/python/MOBIKE_CUP_2017/Mobike_bj2.csv')
#dis_all1=data1['d(m)']/1000
#data2=pd.read_csv('C:/python/MOBIKE_CUP_2017/bikes_group_combine.txt')

f_family='Arial'
fig2=	plt.figure(figsize=(8, 6))
ax11=	fig2.add_subplot(1,	1,	1)
plt.yticks(fontproperties = f_family)
plt.xticks(fontproperties = f_family) 
plt.tick_params(labelsize=18)



#good_user=[2240, 7207, 4906, 3369, 6044, 5305, 3984, 1455, 5650, 7045]
#dict1={}
#for i in good_user:
#    dict1[i]=[]
#for i,j in zip(data1['userid'],data2['rank_100']):
#    if i in good_user:
#        dict1[i].append(j)
#for key,value in dict1.items():
#    ha=pd.Series(value).value_counts(normalize=True)
#    x,y = (list(t) for t in zip(*sorted(zip(list(ha.index),list(ha.values)))))
#    x,y=log_pdf(x,y)
#    ax11.plot(x,y,marker='o')
#ax11.set_yscale('log')
#ax11.set_xscale('log')
#ax11.set_ylim([0.00001,1])
##ax11.legend(loc=3,handletextpad=0.1,ncol=2,columnspacing=1,prop={'size':18,'family':f_family})
#ax11.set_xlabel(r'rank(times$^{-1}$*$\Delta$t$^{-1}$)',size=18,family=f_family)  
#ax11.set_ylabel(r'$P$(rank)',size=18,family=f_family)


   
#dict1={}
#for i,j in zip(data1['userid'],data2['rank_100']):
#    if i not in dict1.keys():
#        dict1[i]=[j]
#    else:
#        dict1[i].append(j)
#dict2={}
#user_all=data1["userid"].value_counts()
#for i,j in zip(user_all.index,user_all.values):
#    dict2[i]=j 
#x,y,z=[],[],[]
#for key,value in dict1.items():
#    limit=np.max(value)*0.95
#    value2=[i for i in value if i<=limit]
#    if(len(value2)>0):
#        x.append(np.mean(value2))
#        y.append(np.std(value2))
#        z.append(dict2[key])
##ax11.scatter(x,y)
##plt.hist2d(x, y, bins=100,norm=LogNorm())
##plt.colorbar() 
#group=[10,20,40,70,100,200]
#colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
#        '#7f7f7f', '#bcbd22', '#17becf']
#dict3={}
#for i in range(len(group)):
#    dict3[group[i]]=[]    
#for i,j,g in zip(z,x,y):
#    for k in group:
#        if i<=k:
#            dict3[k].append([j,g])
#            break
#for key,value in dict3.items():
#    value7=np.array(value)
#    x,y=value7[:,0],value7[:,1]
#    ax11.scatter(x,y,alpha=0.4,label='trips=%d'%key,color=colors[group.index(key)])    
#ax11.legend(loc=2,handletextpad=0.1,ncol=2,columnspacing=1,prop={'size':18,'family':f_family})
#ax11.set_xlabel(r'$\langle$rank$\rangle$',size=18,family=f_family)  
#ax11.set_ylabel(r'STD',size=18,family=f_family)




#dict2={}
#user_all=data1["userid"].value_counts()
#for i,j in zip(user_all.index,user_all.values):
#    dict2[i]=j
#group=[10,20,40,70,100,200]
##dict2=gyration(data1['userid'],data['end_location_x'],data['end_location_y'])
##group=[1,3,6,10,20,30]
#gg=[]
#for i in data1['userid']:
#    gg.append(dict2[i])
#colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
#        '#7f7f7f', '#bcbd22', '#17becf']
#dict3={}
#for i in range(len(group)):
#    dict3[group[i]]=[]    
#for i,j in zip(gg,data2['rank_100']):
#    for k in group:
#        if i<=k:
#            dict3[k].append(j)
#            break
#for key,value in dict3.items():
#    ha=pd.Series(value).value_counts(normalize=True)
#    m,n = (list(t) for t in zip(*sorted(zip(list(ha.index),list(ha.values)))))
#    m,n=log_pdf(m,n)    
#    if(key==200):
#        popt, pcov = curve_fit(func2, m[1:], n[1:],maxfev = 10000)
#        y777 = [func2(i, *popt) for i in m]
#        ax11.scatter(m,n,label='trips=%d'%key,color=colors[group.index(key)])
#        ax11.plot(m,y777,color=colors[group.index(key)],linestyle='-',label=r'$\alpha$=%.2f'%popt[1])
#    else:
#        ax11.scatter(m,n,label='trips=%d'%key,color=colors[group.index(key)])        
##    ax11.scatter(m,n,label=r'r$_g$=%d'%key,color=colors[group.index(key)])
#ax11.text(100, 0.1,r'Shanghai(D1)', size = 18,family=f_family)
##ax11.legend(loc=3,handletextpad=0.1,prop={'size':18,'family':f_family})
#ax11.legend(loc=3,handletextpad=0.1,ncol=2,columnspacing=1,prop={'size':18,'family':f_family})
#ax11.set_yscale('log')
#ax11.set_xscale('log')
#ax11.set_ylim([0.0000001,1])
#ax11.set_xlabel(r'rank(times$^{-1}$*$\Delta$t$^{-1}$)',size=18,family=f_family)  
#ax11.set_ylabel(r'$P$(rank)',size=18,family=f_family)

 


#time1 = pd.to_datetime(data1['start_time'])
##x1,y1,z1=gy_time(data1['userid'],data1['end_location_x'],data1['end_location_y'],time1)
#x2,y2,z2=gy_time(data1['bikeid'],data1['end_location_x'],data1['end_location_y'],time1)
##x1=[1.0, 1.6102620275609394, 2.592943797404667, 3.2903445623126686, 4.1753189365604015, 5.298316906283708, 6.7233575364993365, 8.531678524172808, 10.826367338740546, 13.73823795883263, 17.43328822199988, 22.122162910704493, 28.072162039411772, 35.62247890262442, 45.20353656360243, 57.36152510448679, 72.7895384398315, 92.36708571873861, 117.21022975334806, 148.73521072935117, 188.73918221350976, 239.5026619987486, 303.9195382313198, 385.6620421163472, 489.3900918477494, 621.0169418915616]
##y1=[0.9131036119385022, 1.0561666174423938, 1.1237707405207404, 1.1655331880223179, 1.2492741190339272, 1.2932689359098464, 1.3872575255920134, 1.4864547661133471, 1.6262515850568513, 1.692434541298541, 1.863821542376419, 2.0392106941589248, 2.246920684308187, 2.373371355419562, 2.551430226730322, 2.7792512251022434, 3.024241172046275, 3.262417492965113, 3.5101387399015342, 3.755238588533735, 4.046404100074631, 4.355144508060113, 4.617833167585315, 4.9353641365042416, 5.217885031096348, 5.50350876118669]
##z1=[0.7458895165197764, 0.8539888361448742, 0.867829765592231, 0.8755802191568078, 0.9352165269584883, 0.9016851402594165, 0.9682128049526969, 0.9877741317181364, 1.033868892878932, 1.063783713855353, 1.1169393401282828, 1.151917130662893, 1.1777022789596985, 1.2010265906044393, 1.2484281888334774, 1.2933914918455613, 1.3339296355115609, 1.3578059462225904, 1.408018096588125, 1.4509592762977501, 1.4985398739254305, 1.53807403655943, 1.5376698466678713, 1.5565394553140952, 1.5720774956393633, 1.5993604588440957]
##x2=[1.0, 1.6102620275609394, 2.592943797404667, 3.2903445623126686, 4.1753189365604015, 5.298316906283708, 6.7233575364993365, 8.531678524172808, 10.826367338740546, 13.73823795883263, 17.43328822199988, 22.122162910704493, 28.072162039411772, 35.62247890262442, 45.20353656360243, 57.36152510448679, 72.7895384398315, 92.36708571873861, 117.21022975334806, 148.73521072935117, 188.73918221350976, 239.5026619987486, 303.9195382313198, 385.6620421163472, 489.3900918477494, 621.0169418915616]
##y2=[1.2627726396046168, 1.2521929726892493, 1.4607720843503424, 1.607110978656501, 1.6855832390281495, 1.641696437544867, 1.5159353992180513, 1.4963457479930748, 1.4338168713389385, 1.6428562145588703, 1.5845528313240864, 1.122395178941729, 1.6780987724156147, 1.729795159848014, 1.3687779437475214, 1.508754392990154, 1.826313351093345, 1.6523338629900328, 1.6332868013553081, 1.7120881334084683, 1.7340773745677172, 1.8411975540635046, 1.831142615093936, 1.9398034043050252, 1.9584139555282798, 1.994255902249534]
##z2=[1.8177046960115268, 1.5595498853549237, 1.887743931319092, 2.0994095723912705, 2.1245233114669495, 2.03688493554244, 1.9820494364191228, 1.8278708081639816, 1.8518241175683536, 2.070247950697839, 2.196275952208985, 1.9095330790885903, 2.094974533568146, 2.213923745391695, 2.028610640992852, 2.088074629042953, 2.2169802841476343, 2.12186383234635, 2.1154956679619614, 2.1146619871213863, 2.0890983940602808, 2.11835866284032, 2.0908011547615626, 2.1165009304811795, 2.0788337342576986, 2.0664477686399456]
##ax11.scatter(x1,y1,label=r'user')
#ax11.scatter(x2,y2,label=r'bike')
##print(x2)
##print(y2)
##print(z2)
#ax11.set_yscale('log')
#ax11.set_xscale('log')
##ax11.set_ylim([0.0000001,1])
#ax11.legend(loc=2,handletextpad=0.1,prop={'size':18,'family':f_family})
#ax11.set_xlabel('t(h)',size=18,family=f_family) 
#ax11.set_ylabel(r'$r_g$(t)',size=18,family=f_family)




#time1 = pd.to_datetime(data1['start_time'])
#bikeid=data1['bikeid']
#endx=data1['end_location_x']
#endy=data1['end_location_y']
##x2,y2,z2=gy_time(data1['bikeid'],data1['end_location_x'],data1['end_location_y'],time1)
#good_bike=[]
#bike_all=bikeid.value_counts()
#for i,j in zip(bike_all.index,bike_all.values):
#    if(j>50):
#        good_bike.append(i)
#print(good_bike)
##good_bike=bike_all.index[:int(len(bike_all)*0.1)]
#dict2={}
#dict_t={}
#for i,j,k,z in zip(bikeid,endx,endy,time1):
#    if(i in good_bike):
#        if i not in dict2.keys():
#            dict2[i]=[[j,k]]
#        else:
#            dict2[i].append([j,k])
#        if i not in dict_t.keys():
#            dict_t[i]=[z]
#        else:
#            dict_t[i].append(z)
#u_all2=0  
#for key,value in dict_t.items():
#    if(len(value)>1):
#        u_all2=u_all2+1
#        user_this=[]
#        for m in range(1,len(value)):
#            user_this.append(((time.mktime(value[m].timetuple())-time.mktime(value[0].timetuple()))/60)//60)
#        dict_t[key]=user_this
#dict_all2={}    
#for key,value in dict2.items():
#    if(len(value)>1):
#        trip=[]
#        for m in range(len(value)):
#            value_this=value[:m+1]
#            bike_point1_array=np.array(value_this)
#            endx_mid=np.mean(bike_point1_array[:,0])
#            endy_mid=np.mean(bike_point1_array[:,1])
#            dis1=0
#            for point1 in value_this:
#                dis1=dis1+geodistance(point1[0],point1[1],endx_mid,endy_mid)                   
#            trip.append(dis1/len(value_this))
##            if(len(trip)>200):
##                trip=trip[:200]  
#        new_x,new_y=log_pdf(dict_t[key],trip[1:])
#        ax11.plot(new_x,new_y)
#
#ax11.set_yscale('log')
#ax11.set_xscale('log')
##ax11.set_ylim([0.0000001,1])
#ax11.legend(loc=2,handletextpad=0.1,prop={'size':18,'family':f_family})
#ax11.set_xlabel('t(h)',size=18,family=f_family) 
#ax11.set_ylabel(r'$r_g$(t)',size=18,family=f_family)
    

    
#gggg=[32253, 76787, 99079, 117422, 103893, 111179, 26157, 93787]



#dict1={}
#for i,j in zip(data1['userid'],data2['rank_100']):
#    if i not in dict1.keys():
#        dict1[i]=[j]
#    else:
#        dict1[i].append(j)
#dict2={}
#user_all=data1["userid"].value_counts()
#for i,j in zip(user_all.index,user_all.values):
#    dict2[i]=j 
#x,y,z=[],[],[]
#for key,value in dict1.items():
#    limit=np.max(value)*0.95
#    value2=[i for i in value if i<=limit]
#    if(len(value2)>0):
#        x.append(np.mean(value2))
#x1,y1=pdf1(x,1)
##ax11.scatter(x1,y1)
##popt, pcov = curve_fit(func4, x1[80:],y1[80:],maxfev = 10000)
##y777 = [func4(i, *popt) for i in x1]
##ax11.scatter(x1,y1,label=r'$\xi$=%.2f'%-popt[1])
##ax11.plot(x1,y777,'b-',linewidth=2)
#popt, pcov = curve_fit(func7, x1[:],y1[:],maxfev = 10000)
#y777 = [func7(i, *popt) for i in x1]
#ax11.scatter(x1,y1,label=r'$\mu$=%.2f, $\sigma$=%.2f'%(popt[0],popt[1]))
#ax11.plot(x1,y777,'b-',linewidth=2)
#ax11.set_yscale('log')
#ax11.set_xscale('log')
#ax11.set_ylim([0.00001,1])
#ax11.legend(loc=2,handletextpad=0.1,ncol=2,columnspacing=1,prop={'size':18,'family':f_family})
#ax11.set_xlabel(r'$\langle$rank$\rangle$',size=18,family=f_family)  
#ax11.set_ylabel(r'$P$($\langle$rank$\rangle$)',size=18,family=f_family)



#dict1={}
#for i,j in zip(data1['userid'],data2['rank_100']):
#    if i not in dict1.keys():
#        dict1[i]=[j]
#    else:
#        dict1[i].append(j)
#dict2={}
#for key,value in dict1.items():
#    limit=np.max(value)*0.95
#    value2=value
##    value2=[i for i in value if i<=limit]
#    if(len(value2)>0):
#        dict2[key]=np.mean(value2)
#group=[10,20,40,70,100,200]
#gg=[]
#for i in data1['userid']:
#    gg.append(dict2[i])
#colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
#        '#7f7f7f', '#bcbd22', '#17becf']
#dict3={}
#for i in range(len(group)):
#    dict3[group[i]]=[]    
#for i,j in zip(gg,data2['rank_100']):
#    for k in group:
#        if i<=k:
#            dict3[k].append(j)
#            break
#for key,value in dict3.items():
#    ha=pd.Series(value).value_counts(normalize=True)
#    m,n = (list(t) for t in zip(*sorted(zip(list(ha.index),list(ha.values)))))
#    scale_sh1=np.sum(np.array(m)*np.array(n))
#    m=np.array(m)/scale_sh1
#    n=np.array(n)*scale_sh1
#    m,n=log_pdf1(m,n)    
#    if(key==70):
##        ax11.scatter(m,n,label=r'$\langle$rank$\rangle$=%d'%key,color=colors[group.index(key)])  
#        popt, pcov = curve_fit(func1, m[:], n[:],maxfev = 10000)
#        y777 = [func1(i, *popt) for i in m]
#        ax11.scatter(m,n,label=r'$\langle$rank$\rangle$=%d'%key,color=colors[group.index(key)])
#        ax11.plot(m,y777,color=colors[group.index(key)],linestyle='-',label=r'$\alpha$=%.2f'%popt[1])
#    else:
#        ax11.scatter(m,n,label=r'$\langle$rank$\rangle$=%d'%key,color=colors[group.index(key)])        
##    ax11.scatter(m,n,label=r'r$_g$=%d'%key,color=colors[group.index(key)])
#ax11.text(1, 1,r'Shanghai(D1)', size = 18,family=f_family)
##ax11.legend(loc=3,handletextpad=0.1,prop={'size':18,'family':f_family})
#ax11.legend(loc=3,handletextpad=0.1,ncol=2,columnspacing=1,prop={'size':18,'family':f_family})
#ax11.set_yscale('log')
#ax11.set_xscale('log')
##ax11.set_ylim([0.0000001,1])
##ax11.set_xlabel(r'rank(times$^{-1}$*$\Delta$t$^{-1}$)',size=18,family=f_family)  
##ax11.set_ylabel(r'$P$(rank)',size=18,family=f_family)
#ax11.set_ylim([0.0001,10])
#ax11.set_xlim([0.007,20])
#ax11.set_xlabel(r'rank(times$^{-1}$*$\Delta$t$^{-1}$)/$\langle$rank$\rangle$',size=18,family=f_family)  
#ax11.set_ylabel(r'$\langle$rank$\rangle P$(rank)',size=18,family=f_family)



def pdf2(data_x,data_y,step):
 
    x_range=np.arange(0,100,step)
    y=[[] for i in range(len(x_range)-1)]
    x=x_range[:-1]+step/2
    for i,j in zip(data_x,data_y):
        a=int(i//step)        
        y[a].append(j)
    x1=[]
    y1=[]
    z1=[]
    for i in range(len(x)):
        if(len(y[i])>0):
            x1.append(x[i])
            y1.append(np.mean(y[i]))
            z1.append(np.std(y[i]))
    return x1,y1,z1

#ax11.scatter(data1['d'],data2['rank_100'],alpha=0.1,c='grey',s=4)
xxx,yyy,zzz=pdf2(data1['d'],data2['rank_100'],2)
#xxx,yyy=pdf2(dis_all1,data2['rank_150'],2)
ax11.scatter(xxx,yyy)
ax11.errorbar(xxx,yyy,fmt="none",yerr = zzz,alpha=0.6)
#ax11.errorbar(xxx,yyy,fmt="o",yerr = zzz,alpha=0.6)
#plt.hist2d(data1['d'],data2['rank_100'],bins=100,norm=LogNorm())
#plt.colorbar() 
#ax11.set_yscale('log')
#ax11.set_xscale('log')
ax11.set_ylim([0,180])
ax11.set_xlim([0,100])
ax11.set_xlabel(r'd(km)',size=18,family=f_family)  
ax11.set_ylabel(r'$\langle$rank$\rangle$',size=18,family=f_family)
#plt.savefig('C:/python/摩拜单车/draw7/d_rank.pdf',bbox_inches='tight') 




