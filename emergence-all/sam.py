import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from scipy.optimize import curve_fit
from scipy.special import factorial


all1_x=[1.0, 1.6102620275609394, 2.592943797404667, 3.2903445623126686, 4.1753189365604015, 
        5.298316906283708, 6.7233575364993365, 8.531678524172808, 10.826367338740546, 13.73823795883263, 
        17.43328822199988, 22.122162910704493, 28.072162039411772, 35.62247890262442, 45.20353656360243,
        57.36152510448679, 72.7895384398315, 92.36708571873861, 117.21022975334806, 148.73521072935117, 
        188.73918221350976, 239.5026619987486, 303.9195382313198]
all1=[1.8917066110416232, 1.942069002940697, 1.9617683436869136, 2.0064112152655356, 2.0162209167990106, 
      2.083417188858552, 2.128709715887971, 2.1247736768302525, 2.132811950858918, 2.1034761735566443, 
      2.190846383602471, 2.2347602644526523, 2.5659404808156543, 2.562634618623643, 2.766357285510757, 
      2.8470376255519105, 3.2222257922315607, 3.38788039023688, 3.6358891170000893, 3.838399930544136, 
      4.284043751595284, 4.674215273461309, 4.962926673144386]
all2_x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
all2=[0.6211050870515258, 0.26180246813392594, 0.1511726340474528, 0.10783843755376707,
      0.08582369841340573, 0.07249285694863201, 0.06342503646666375, 0.05644446907092112,
      0.050616646642257124, 0.04631738670409669, 0.04321553955088785, 0.0408068887543931, 
      0.038438330751950085, 0.03654278625700764, 0.03543580995152619, 0.03382030966458074,
      0.03261238152371451, 0.0322439160740071, 0.03040507971230073, 0.02960959364646259,
      0.02879961593085224, 0.028091981574823104, 0.026193046173174396, 0.02586703571902792, 
      0.02235751226618111, 0.02208016162313379, 0.021911236004496272, 0.022628127712079466, 
      0.01910358615118312, 0.01892574062272783, 0.01246422650098145, 0.010380893167648116, 
      0.010113608777056526, 0.010081557495005245, 0.005620773181279754, 0.0051524413347141884, 
      0.0035130970724191065, 0.0035130970724191065, 0.0035130970724191065, 0.0035130970724191065, 
      0.0018181818181818182, 0.0018181818181818182]
all3_x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 
        46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59]
all3=[0.30132941176470585, 0.16301176470588238, 0.11216470588235294, 0.08475294117647059,
      0.06498235294117648, 0.052723529411764704, 0.04205882352941177, 0.033452941176470595,
      0.026676470588235295, 0.02203529411764706, 0.017552941176470587, 0.014205882352941176,
      0.011676470588235293, 0.009164705882352943, 0.0071529411764705885, 0.005847058823529412, 
      0.005164705882352941, 0.0040411764705882355, 0.003111764705882353, 0.002829411764705882, 
      0.002247058823529412, 0.0019470588235294115, 0.0018823529411764704, 0.001435294117647059,
      0.0012764705882352943, 0.0010294117647058824, 0.0008470588235294117, 0.0008176470588235294, 
      0.0006941176470588236, 0.0006176470588235294, 0.0005176470588235295, 0.0003882352941176471,
      0.0002470588235294118, 0.0002176470588235294, 0.0003470588235294118, 0.0002, 0.0002235294117647059,
      0.00017647058823529413, 0.0001352941176470588, 0.00014117647058823534, 6.470588235294118e-05,
      0.0001, 6.470588235294118e-05, 7.058823529411765e-05, 6.470588235294118e-05, 7.647058823529412e-05, 
      5.2941176470588244e-05, 1.7647058823529414e-05, 2.3529411764705884e-05, 5.882352941176471e-06, 
      1.7647058823529414e-05, 5.882352941176471e-06, 5.882352941176471e-06, 1.1764705882352942e-05, 
      6.470588235294118e-05, 5.882352941176471e-06, 1.7647058823529414e-05, 5.882352941176471e-06]
all4_x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66]
all4=[0.10710588235294119, 0.08674117647058824, 0.07494705882352941, 0.06801176470588235, 
      0.06388823529411766, 0.06174705882352941, 0.05964705882352941, 0.05692941176470588,
      0.052335294117647056, 0.04795294117647059, 0.04318235294117647, 0.03771764705882353, 
      0.03360588235294118, 0.027600000000000003, 0.025317647058823528, 0.02151764705882353,
      0.018323529411764707, 0.014711764705882351, 0.013488235294117645, 0.011876470588235296,
      0.009111764705882354, 0.008111764705882355, 0.007511764705882354, 0.005888235294117647, 
      0.00551764705882353, 0.004347058823529411, 0.003811764705882352, 0.0036647058823529415, 
      0.002929411764705882, 0.0024294117647058826, 0.002435294117647059, 0.002335294117647059,
      0.0018882352941176467, 0.0016176470588235296, 0.0012647058823529412, 0.0012470588235294117,
      0.0011294117647058825, 0.001023529411764706, 0.0007823529411764707, 0.0007647058823529412, 
      0.0007470588235294119, 0.0006705882352941176, 0.0004117647058823529, 0.0004294117647058823,
      0.00046470588235294114, 0.0003647058823529412, 0.0003117647058823529, 0.0002941176470588235,
      0.0003117647058823529, 0.00020588235294117648, 0.0002823529411764706, 0.0001882352941176471,
      0.0001647058823529412, 0.00017058823529411769, 0.00012941176470588237, 0.00010588235294117649,
      4.705882352941177e-05, 4.11764705882353e-05, 4.705882352941177e-05, 4.7058823529411774e-05, 
      4.7058823529411774e-05, 2.3529411764705884e-05, 1.1764705882352942e-05, 1.1764705882352942e-05,
      5.882352941176471e-06, 5.882352941176471e-06]
all5_x=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5,
        16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 28.5, 29.5, 30.5, 32.5, 34.5, 38.5]
all5=[0.47761764705882354, 0.1069, 0.08128823529411766, 0.06854117647058823, 0.059364705882352944,
      0.050376470588235286, 0.0397, 0.0315, 0.02414117647058824, 0.01757058823529412, 
      0.012658823529411764, 0.009088235294117649, 0.006276470588235295, 0.004929411764705882,
      0.0031764705882352945, 0.0021470588235294116, 0.0016, 0.0009823529411764707, 
      0.0007411764705882353, 0.0004411764705882353, 0.0003176470588235294, 0.00019411764705882354,
      0.0001176470588235294, 8.823529411764706e-05, 9.411764705882355e-05, 7.058823529411765e-05,
      3.529411764705883e-05, 5.882352941176471e-06, 5.882352941176471e-06, 5.882352941176471e-06, 
      1.1764705882352942e-05, 5.882352941176471e-06, 5.882352941176471e-06]
all6_x=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 18.5]
all6=[0.29014117647058824, 0.5884705882352941, 0.0886235294117647, 0.021411764705882352,
      0.00658235294117647, 0.0026117647058823534, 0.0010470588235294118, 0.0005235294117647059, 
      0.00025294117647058827, 0.0001235294117647059, 3.529411764705883e-05, 7.058823529411765e-05,
      3.529411764705883e-05, 2.3529411764705884e-05, 1.7647058823529414e-05, 1.7647058823529414e-05, 
      1.1764705882352942e-05]

all11_x=[1.0, 1.6102620275609394, 2.592943797404667, 3.2903445623126686, 4.1753189365604015, 
         5.298316906283708, 6.7233575364993365, 8.531678524172808, 10.826367338740546, 
         13.73823795883263, 17.43328822199988, 22.122162910704493, 28.072162039411772, 
         35.62247890262442, 45.20353656360243, 57.36152510448679, 72.7895384398315, 
         92.36708571873861, 117.21022975334806, 148.73521072935117, 188.73918221350976,
         239.5026619987486, 303.9195382313198]
all11=[1.7114076416177042, 1.7365509094383758, 1.7484556098457973, 1.7780383259269061,
       1.7865023734715149, 1.806536750370731, 1.8244733731851615, 1.8407049104843742, 
       1.8282352358452179, 1.873396506070554, 1.9344284330290633, 1.9535522898528197, 
       2.0011399508155234, 2.0234652339070265, 2.0527619734869473, 2.086853216143128, 
       2.119041351631492, 2.155545216073438, 2.187825130799477, 2.2304143794690505, 
       2.2810940521681142, 2.336582363694449, 2.3910319276499896]
all22_x=[1, 2, 3, 4, 5, 6, 7]
all22=[0.908261562197189, 0.46488504793990665, 0.3136753947231772, 0.23587724274742095,
       0.18586655727989793, 0.1533225108225108, 0.07468253968253968]
all33_x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 16, 17]
all33=[0.8979807436728249, 0.08917078701892331, 0.01095875150613854, 0.0015272382108540317, 
       0.0002611443668357637, 6.01563496267546e-05, 1.8938640837024607e-05, 8.234724258065001e-06, 
       5.760838032556382e-06, 1.6544862302853788e-06, 3.3032018031467674e-06, 8.145710468867095e-07,
       1.6499765331184417e-06, 8.224360555966774e-07]
all44_x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 23]
all44=[0.800081961883763, 0.16369226537017417, 0.02960841962594297, 0.005177370003868406, 
       0.0009952572759887722, 0.00024950817717303575, 8.32547186788079e-05, 4.529238952727788e-05,
       1.9731267909207092e-05, 1.728722557925163e-05, 8.24793746757379e-06, 6.587420994450777e-06, 
       4.124500472675788e-06, 8.248715262597851e-07, 1.6364663374266946e-06, 1.6499765331184417e-06,
       3.294953596835378e-06, 1.6489273642833697e-06, 8.145710468867095e-07, 8.224360555966774e-07]
all55_x=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 
         17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5]
all55=[0.8719948253817374, 0.033575158588595444, 0.027435797342491207, 0.02087027833635957, 
       0.015271155675414174, 0.010925470812605618, 0.007448341957032407, 0.004926487364933341,
       0.00302206520071693, 0.0018886465423366093, 0.0010997338572579579, 0.0006845405668033781,
       0.00036658942278099333, 0.00021582727587372052, 0.00013098692684046108, 6.671442007154366e-05, 
       3.290678652664927e-05, 1.6476364587817137e-05, 1.565451404144254e-05, 5.759342264615491e-06, 
       1.6417052674596956e-06, 2.4692028723964477e-06, 1.6499765331184417e-06, 8.224360555966774e-07]
all66_x=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5,
         16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 29.5, 35.5, 37.5]
all66=[0.4244364603905325, 0.44291944889135665, 0.08955685745864878, 0.0255357335583959, 
       0.00931117036195884, 0.004103171127176538, 0.001920800850393644, 0.0009852553322935525,
       0.00047705455537091715, 0.00030822071757129763, 0.00012849186353060363, 8.406594129374187e-05,
       6.513058086095834e-05, 5.4410235840025205e-05, 3.868024135676521e-05, 2.638026440549643e-05,
       1.4006595574737522e-05, 9.08463669587505e-06, 3.297811280171981e-06, 2.4807469861313924e-06, 
       2.456276314346405e-06, 8.260517726093939e-06, 8.325909397453936e-07, 8.259206950948571e-07, 
       2.4692028723964477e-06, 8.325909397453936e-07, 1.654826291036535e-06, 8.289055959416783e-07, 
       8.224360555966774e-07, 8.145710468867095e-07]


#normal
def func1(x,mu,sigma):
    return np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))

def func2(x, a,b1,b2,c):
    return a*pow(x+c,-b1)*np.exp((-b2)*x)

#泊松    
def func3(x, a):
    return (a**x/factorial(x)) * np.exp(-a)

#幂律 
def func4(x, a, b):
    return a*pow(x,b)

def func5(x, a,b1,b2):
    return a*pow(x,-b1)*np.exp((-b2)*x)


f_family='Arial'
size_x=8
size_y=6
alpha_xy=1
fig	=	plt.figure(figsize=(size_x, size_y))
ax1	=	fig.add_subplot(1,	1,	1)
plt.yticks(fontproperties = f_family)
plt.xticks(fontproperties = f_family) 
plt.tick_params(labelsize=18)
popt, pcov = curve_fit(func4, all1_x[10:],all1[10:])
y777 = [func4(i, *popt) for i in all1_x[0:]]
ax1.scatter(all1_x,all1,label=r'user, $\epsilon$=%.2f'%popt[1],alpha=alpha_xy)
ax1.plot(all1_x[0:],y777,'b-')
popt, pcov = curve_fit(func4, all11_x[10:],all11[10:])
y888= [func4(i, *popt) for i in all11_x[0:]]
ax1.scatter(all11_x,all11,label=r'bike, $\epsilon$=%.2f'%popt[1],alpha=alpha_xy)
ax1.plot(all11_x[0:],y888,c='orange',linestyle='-')
ax1.plot([24,24],[1,100],color='k',linestyle='--')
ax1.plot([168,168],[1,100],color='k',linestyle='--')
ax1.plot([720,720],[1,100],color='k',linestyle='--')
ax1.text(19, 70,r'Day', size = 18,family=f_family)
ax1.text(120, 70,r'Week', size = 18,family=f_family)
ax1.text(400, 70,r'Month', size = 18,family=f_family)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_xlim([1,1000])
ax1.set_ylim([1.6,100])
ax1.legend(loc=2,handletextpad=0.1,prop={'size':18,'family':f_family})
ax1.set_xlabel('t(h)',size=18,family=f_family) 
ax1.set_ylabel(r'S(t)',size=18,family=f_family)
#plt.savefig('C:/python/摩拜单车/draw2/sam_1.pdf',bbox_inches='tight') 


fig2	=	plt.figure(figsize=(size_x, size_y))
ax2	=	fig2.add_subplot(1,	1,	1)
plt.yticks(fontproperties = f_family)
plt.xticks(fontproperties = f_family) 
plt.tick_params(labelsize=18)
popt, pcov = curve_fit(func4, all2_x[4:-10],all2[4:-10],maxfev = 10000)
y777 = [func4(i, *popt) for i in all2_x]
ax2.scatter(all2_x,all2,label=r'user, $\xi$=%.2f'%-popt[1],alpha=alpha_xy)
ax2.plot(all2_x,y777,'b-')
popt, pcov = curve_fit(func4, all22_x,all22,maxfev = 10000)
y888 = [func4(i, *popt) for i in all22_x]
ax2.scatter(all22_x,all22,label=r'bike, $\xi$=%.2f'%-popt[1],alpha=alpha_xy)
ax2.plot(all22_x,y888,color='orange',linestyle='-')
ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.set_ylim([0.001,1])
ax2.legend(loc=3,handletextpad=0.1,prop={'size':18,'family':f_family})
ax2.set_xlabel('log'+r'$_{10}L$',size=18,family=f_family) 
ax2.set_ylabel('log'+r'$_{10}P(L)$',size=18,family=f_family) 
#plt.savefig('C:/python/摩拜单车/draw2/sam_2.pdf',bbox_inches='tight') 


fig3	=	plt.figure(figsize=(size_x, size_y))
ax3	=	fig3.add_subplot(1,	1,	1)
plt.yticks(fontproperties = f_family)
plt.xticks(fontproperties = f_family) 
plt.tick_params(labelsize=18)
ax3.scatter(all3_x,all3,marker='o',label=r'uesr',alpha=alpha_xy)
ax3.scatter(all33_x,all33,marker='o',label=r'bike',alpha=alpha_xy)
#ax3.set_yscale('log')
#ax3.set_xscale('log')
#ax3.set_ylim([0.000001,1])
#ax3.set_xlim([0.5,150])
#ax3.legend(loc=3,handletextpad=0.1,prop={'size':18,'family':f_family})
ax3.set_xlabel(r'i$^{th}$ trip',size=18,family=f_family)  
ax3.set_ylabel(r'$P(i)$',size=18,family=f_family)
#plt.savefig('C:/python/摩拜单车/draw2/sam_3.pdf',bbox_inches='tight') 


fig4	=	plt.figure(figsize=(size_x, size_y))
ax4	=	fig4.add_subplot(1,	1,	1)
plt.yticks(fontproperties = f_family)
plt.xticks(fontproperties = f_family) 
plt.tick_params(labelsize=18)
popt, pcov = curve_fit(func5, all4_x[12:],all4[12:],maxfev = 10000)
y777 = [func5(i, *popt) for i in all4_x[:12]]
y888 = [func5(i, *popt) for i in all4_x[12:]]
ax4.scatter(all4_x,all4,label=r'user, $\alpha$=%.2f'%popt[1],alpha=alpha_xy)
ax4.plot(all4_x[:12],y777,'b--')
ax4.plot(all4_x[12:],y888,'b-')
popt, pcov = curve_fit(func4, all44_x[2:],all44[2:],maxfev = 10000)
y888 = [func4(i, *popt) for i in all44_x]
ax4.scatter(all44_x,all44,label=r'bike, $\xi$=%.2f'%-popt[1],alpha=alpha_xy)
ax4.plot(all44_x,y888,color='orange',linestyle='-')
ax4.set_yscale('log')
ax4.set_xscale('log')
ax4.set_ylim([0.0000001,1])
ax4.set_xlim([0.8,160])
ax4.legend(loc=3,handletextpad=0.1,prop={'size':20,'family':f_family})
ax4.set_xlabel('log'+r'$_{10}$'+'#trips',size=18,family=f_family)  
ax4.set_ylabel('log'+r'$_{10}P$(#)',size=18,family=f_family)
#plt.savefig('C:/python/摩拜单车/draw2/sam_4.pdf',bbox_inches='tight') 


fig5	=	plt.figure(figsize=(size_x, size_y))
ax5	=	fig5.add_subplot(1,	1,	1)
plt.yticks(fontproperties = f_family)
plt.xticks(fontproperties = f_family) 
plt.tick_params(labelsize=18)
popt, pcov = curve_fit(func5, all5_x[:1]+all5_x[7:],all5[:1]+all5[7:],maxfev = 10000)
y777 = [func5(i, *popt) for i in all5_x]
ax5.scatter(all5_x,all5,label=r'user, $\alpha$=%.2f'%popt[1],alpha=alpha_xy)
ax5.plot(all5_x[:],y777,'b-')
popt, pcov = curve_fit(func5, all55_x[:1]+all55_x[8:],all55[:1]+all55[8:],maxfev = 10000)
y888 = [func5(i, *popt) for i in all55_x]
ax5.scatter(all55_x,all55,label=r'bike, $\alpha$=%.2f'%popt[1],alpha=alpha_xy)
ax5.plot(all55_x[:],y888,color='orange',linestyle='-')
ax5.set_yscale('log')
ax5.set_xscale('log')
ax5.set_ylim([0.0000001,1])
ax5.legend(loc=3,handletextpad=0.1,prop={'size':18,'family':f_family})
ax5.set_xlabel('log'+r'$_{10}r_g$'+'(km)',size=18,family=f_family) 
ax5.set_ylabel('log'+r'$_{10}P(r_g)$',size=18,family=f_family)
#plt.savefig('C:/python/摩拜单车/draw2/sam_5.pdf',bbox_inches='tight') 


fig6	=	plt.figure(figsize=(size_x, size_y))
ax6	=	fig6.add_subplot(1,	1,	1)
plt.yticks(fontproperties = f_family)
plt.xticks(fontproperties = f_family) 
plt.tick_params(labelsize=18)
popt, pcov = curve_fit(func4, all6_x[5:],all6[5:])
y777 = [func4(i, *popt) for i in all6_x]
ax6.scatter(all6_x,all6,label=r'user, $\gamma$=%.2f'%-popt[1],alpha=alpha_xy)
ax6.plot(all6_x,y777,'b-')
popt, pcov = curve_fit(func4, all66_x[5:],all66[5:],maxfev = 10000)
y888 = [func4(i, *popt) for i in all66_x]
ax6.scatter(all66_x,all66,label=r'bike, $\gamma$=%.2f'%-popt[1],alpha=alpha_xy)
ax6.plot(all66_x,y888,color='orange',linestyle='-')
ax6.set_yscale('log')
ax6.set_xscale('log')
ax6.set_ylim([0.0000001,1])
ax6.set_xlim([0.1,50])
ax6.legend(loc=1,handletextpad=0.1,prop={'size':18,'family':f_family})
ax6.set_xlabel('log'+r'$_{10}\langle d \rangle$'+'(km)',size=18,family=f_family) 
ax6.set_ylabel('log'+r'$_{10}P(\langle d \rangle)$',size=18,family=f_family)
#plt.savefig('C:/python/摩拜单车/draw2/sam_6.pdf',bbox_inches='tight') 







#dict_all1={}
#dict_all2={}
#dict_all3={}
#dict_all4={}
#dict_all5={}
#dict_all6={}
#
#dict_all11={}
#dict_all22={}
#dict_all33={}
#dict_all44={}
#dict_all55={}
#dict_all66={}
#
#for kk in range(1,11,1):
#    file='C:/python/MOBIKE_CUP_2017/sample/sample_bj'+str(kk)+'.csv'
#    data=pd.read_csv(file)
#
#    time1 = pd.to_datetime(data['starttime'])
#    endx=data['end_x']
#    endy=data['end_y']
#    dis_all=data['d(m)']/1000
#    bikeid_random1=data['userid']
#    bikeid_random2=data['bikeid']
#    trips_x1,trips_y1,dis_x1,dis_y1=trips_d(bikeid_random1,dis_all)    
#    gy_x1,gy_y1=gyration(bikeid_random1,endx,endy)
#    unique_t_x1,unique_t_y1=uni_time(bikeid_random1,endx,endy,1,time1)
#    unique_t_x1,unique_t_y1=log_pdf(unique_t_x1,unique_t_y1)
#    rank_x1,rank_y1,rank_std1=rank(bikeid_random1,endx,endy,1)
#    ith_x1,ith_y1=i_trip(bikeid_random1,dis_all)
#    
#    trips_x2,trips_y2,dis_x2,dis_y2=trips_d(bikeid_random2,dis_all)    
#    gy_x2,gy_y2=gyration(bikeid_random2,endx,endy)
#    unique_t_x2,unique_t_y2=uni_time(bikeid_random2,endx,endy,1,time1)
#    unique_t_x2,unique_t_y2=log_pdf(unique_t_x2,unique_t_y2)
#    rank_x2,rank_y2,rank_std2=rank(bikeid_random2,endx,endy,1)
#    ith_x2,ith_y2=i_trip(bikeid_random2,dis_all)
#    
#    for i,j in zip(trips_x1,trips_y1):
#        if i not in dict_all4.keys():
#            dict_all4[i]=[j]
#        else:
#            dict_all4[i].append(j) 
#    for i,j in zip(dis_x1,dis_y1):
#        if i not in dict_all6.keys():
#            dict_all6[i]=[j]
#        else:
#            dict_all6[i].append(j)
#    for i,j in zip(gy_x1,gy_y1):
#        if i not in dict_all5.keys():
#            dict_all5[i]=[j]
#        else:
#            dict_all5[i].append(j)
#    for i,j in zip(unique_t_x1,unique_t_y1):
#        if i not in dict_all1.keys():
#            dict_all1[i]=[j]
#        else:
#            dict_all1[i].append(j)
#    for i,j in zip(rank_x1,rank_y1):
#        if i not in dict_all2.keys():
#            dict_all2[i]=[j]
#        else:
#            dict_all2[i].append(j)
#    for i,j in zip(ith_x1,ith_y1):
#        if i not in dict_all3.keys():
#            dict_all3[i]=[j]
#        else:
#            dict_all3[i].append(j)
#            
#            
#    for i,j in zip(trips_x2,trips_y2):
#        if i not in dict_all44.keys():
#            dict_all44[i]=[j]
#        else:
#            dict_all44[i].append(j) 
#    for i,j in zip(dis_x2,dis_y2):
#        if i not in dict_all66.keys():
#            dict_all66[i]=[j]
#        else:
#            dict_all66[i].append(j)
#    for i,j in zip(gy_x2,gy_y2):
#        if i not in dict_all55.keys():
#            dict_all55[i]=[j]
#        else:
#            dict_all55[i].append(j)
#    for i,j in zip(unique_t_x2,unique_t_y2):
#        if i not in dict_all11.keys():
#            dict_all11[i]=[j]
#        else:
#            dict_all11[i].append(j)
#    for i,j in zip(rank_x2,rank_y2):
#        if i not in dict_all22.keys():
#            dict_all22[i]=[j]
#        else:
#            dict_all22[i].append(j)
#    for i,j in zip(ith_x2,ith_y2):
#        if i not in dict_all33.keys():
#            dict_all33[i]=[j]
#        else:
#            dict_all33[i].append(j)
# 
#           
##print(unique_x)
##print(unique_y)
##print(rank_x)
##print(rank_y)
##print(rank_std)
##print(ith_x)
##print(ith_y)
##print(trips_x)
##print(trips_y)
##print(gy_x)
##print(gy_y)
##print(dis_x1)
##print(dis_y1)
#
#
#
#all1=[]
#all1_x=[]
#for key,value in dict_all1.items():
#    all1.append(np.sum(value)/10)
#    all1_x.append(key)
#all2=[]
#all2_x=[]
#for key,value in dict_all2.items():
#    all2.append(np.sum(value)/10)
#    all2_x.append(key)  
#all3=[]
#all3_x=[]
#for key,value in dict_all3.items():
#    all3.append(np.sum(value)/10)
#    all3_x.append(key)
#all4=[]
#all4_x=[]
#for key,value in dict_all4.items():
#    all4.append(np.sum(value)/10)
#    all4_x.append(key)
#all5=[]
#all5_x=[]
#for key,value in dict_all5.items():
#    all5.append(np.sum(value)/10)
#    all5_x.append(key)
#all6=[]
#all6_x=[]
#for key,value in dict_all6.items():
#    all6.append(np.sum(value)/10)
#    all6_x.append(key)
#
#
#all11=[]
#all11_x=[]
#for key,value in dict_all11.items():
#    all11.append(np.sum(value)/10)
#    all11_x.append(key)
#all22=[]
#all22_x=[]
#for key,value in dict_all22.items():
#    all22.append(np.sum(value)/10)
#    all22_x.append(key)  
#all33=[]
#all33_x=[]
#for key,value in dict_all33.items():
#    all33.append(np.sum(value)/10)
#    all33_x.append(key)
#all44=[]
#all44_x=[]
#for key,value in dict_all44.items():
#    all44.append(np.sum(value)/10)
#    all44_x.append(key)
#all55=[]
#all55_x=[]
#for key,value in dict_all55.items():
#    all55.append(np.sum(value)/10)
#    all55_x.append(key)
#all66=[]
#all66_x=[]
#for key,value in dict_all66.items():
#    all66.append(np.sum(value)/10)
#    all66_x.append(key)
#
#all1_x,all1 = (list(t) for t in zip(*sorted(zip(all1_x,all1)))) 
#all2_x,all2 = (list(t) for t in zip(*sorted(zip(all2_x,all2))))
#all3_x,all3 = (list(t) for t in zip(*sorted(zip(all3_x,all3))))
#all4_x,all4 = (list(t) for t in zip(*sorted(zip(all4_x,all4))))
#all5_x,all5 = (list(t) for t in zip(*sorted(zip(all5_x,all5))))
#all6_x,all6 = (list(t) for t in zip(*sorted(zip(all6_x,all6))))
#
#all11_x,all11 = (list(t) for t in zip(*sorted(zip(all11_x,all11)))) 
#all22_x,all22 = (list(t) for t in zip(*sorted(zip(all22_x,all22))))
#all33_x,all33 = (list(t) for t in zip(*sorted(zip(all33_x,all33))))
#all44_x,all44 = (list(t) for t in zip(*sorted(zip(all44_x,all44))))
#all55_x,all55 = (list(t) for t in zip(*sorted(zip(all55_x,all55))))
#all66_x,all66 = (list(t) for t in zip(*sorted(zip(all66_x,all66))))





