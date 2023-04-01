#人和车的指标对比，计算方法在compare_data里
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy.optimize import curve_fit
from scipy.special import factorial
import time


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
            index_this=bins2.index(key)
#            y_new.append(np.sum(value)/widths[index_this])
            y_new.append(np.mean(value))
            x_new.append(key)
        
    return x_new,y_new

trips_x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
         21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
         39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 
         57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
trips_y=[0.10767570112507285, 0.08609003453591822, 0.0742163389630038,
         0.06751872489830134, 0.06389862574031172, 0.06231393522864615, 
         0.0599225453804262, 0.05738761472499017, 0.05279717971021982,
         0.048218227960026755, 0.04272922749208372, 0.03766797862964462,
         0.03314644320233799, 0.028469883703237994, 0.02489858841970184, 
         0.02108901539618698, 0.01818087864198913, 0.015321545762244749, 
         0.01322584997688993, 0.011537810084028788, 0.009353118998199999,
         0.00821914662119294, 0.007418188916927193, 0.005893785544292387,
         0.005483258835654388, 0.004518664611162308, 0.0038411520011023934, 
         0.0034421085570416816, 0.0030086153192491094, 0.0025119641262814608, 
         0.0022507198571481888, 0.0021387580275196434, 0.00173684376731461, 
         0.0016277527538303865, 0.0014526329690267646, 0.0013177046102436463,
         0.0011913886998934928, 0.0010105272828012275, 0.0008210534172759974, 
         0.0007779911751111723, 0.0007349289329463473, 0.0006574168970496623, 
         0.0005110052736892572, 0.00047081384766875376, 0.00044497650236985874, 
         0.0004133975247823204, 0.0003043065112980969, 0.00034736875346292196, 
         0.00031004814358674026, 0.0002239236592570902, 0.00019234468166955183,
         0.0001693781525149785, 0.00016650733637065682, 0.00015215325564904846,
         0.00012057427806151011, 0.0001033493811955801, 7.46412197523634e-05, 
         4.306224216482504e-05, 4.5933058309146705e-05, 6.0287139030755056e-05, 
         4.8803874453468376e-05, 2.0095713010251684e-05, 1.4354080721608346e-05, 
         1.1483264577286676e-05, 1.4354080721608346e-05, 8.612448432965008e-06,
         5.741632288643338e-06, 2.870816144321669e-06, 2.870816144321669e-06, 
         2.870816144321669e-06]
trips_x2=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
          20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
trips_y2=[0.07254862135857941, 0.08290708470810605, 0.09423963324656376, 
          0.10126755425861754, 0.10345406462951602, 0.10064994279840167, 
          0.09284068111352445, 0.08177341534992456, 0.068500986520319,
          0.056372589657287815, 0.04337166116757581, 0.03265257904597682,
          0.02350032331338186, 0.016439241954470843, 0.010961567157992473,
          0.007007195795268019, 0.00470462421036924, 0.002754381310828511, 
          0.001687032646361481, 0.0011025815330028353, 0.0006134664168587203,
          0.0002839354699650158, 0.00019481703778621525, 7.87558237859168e-05,
          3.93779118929584e-05, 2.4870260142921094e-05, 1.036260839288379e-05, 
          6.2175650357302736e-06, 8.290086714307032e-06, 4.145043357153516e-06]


dis_x1=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5,
        13.5, 14.5, 16.5, 17.5, 18.5]
dis_y1=[0.7362208002112921, 0.2253935171229829, 0.027267011738767215,
        0.006953116701547083, 0.0023454567899108035, 0.0009703358567807242,
        0.00044210568622553704, 0.0001866030493809085, 7.177040360804174e-05,
        5.454550674211171e-05, 2.8708161443216692e-05, 3.157897758753836e-05,
        8.612448432965008e-06, 2.870816144321669e-06, 1.4354080721608346e-05, 
        2.870816144321669e-06, 2.870816144321669e-06, 2.870816144321669e-06]
dis_x2=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 
        15.5, 17.5, 19.5]
dis_y2=[0.7957571336196176, 0.1929911461873891, 0.008928423391308673,
        0.0014528376966823073, 0.0005036227678941521, 0.0001782368643576012,
        6.217565035730274e-05, 6.424817203587949e-05, 3.316034685722813e-05, 
        8.290086714307032e-06, 6.2175650357302736e-06, 4.145043357153516e-06,
        4.145043357153516e-06, 2.072521678576758e-06, 2.072521678576758e-06, 
        2.072521678576758e-06]
dis_x11=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5,
         13.5, 14.5, 15.5, 16.5, 18.5, 19.5, 20.5, 21.5]
dis_y11=[0.2900471675092512, 0.5873259208860487, 0.0897445834876397, 
         0.021433513333505582, 0.006671776719403559, 0.0026067010590440755,
         0.0010449770765330877, 0.0005483258835654388, 0.0002669859014219152,
         0.0001291867264944751, 4.306224216482504e-05, 4.8803874453468376e-05, 
         3.444979373186003e-05, 1.1483264577286676e-05, 1.1483264577286676e-05,
         1.4354080721608346e-05, 2.870816144321669e-06, 5.741632288643338e-06,
         2.870816144321669e-06, 2.870816144321669e-06, 2.870816144321669e-06]
dis_x22=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5,
         13.5, 14.5, 15.5, 19.5, 24.5]
dis_y22=[0.2423751927445161, 0.69331031452589, 0.054542553015104536,
         0.0068082337141246495, 0.0018362542072190074, 0.0006155389385372971, 
         0.00023419494967917364, 0.00011606121400029844, 6.0103128678725976e-05, 
         4.5595476928688673e-05, 1.6580173428614064e-05, 1.6580173428614064e-05,
         6.2175650357302736e-06, 6.2175650357302736e-06, 4.145043357153516e-06,
         2.072521678576758e-06, 2.072521678576758e-06, 2.072521678576758e-06]


gy_x=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5,
      13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 
      24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 31.5, 32.5, 34.5, 38.5, 40.5]
gy_y=[0.47807414169774326, 0.10683168117864228, 0.0807617997720572, 
      0.06822781648594879, 0.05915029583760367, 0.049777081126393424,
      0.0402746796886887, 0.03153017371308489, 0.023661266661499196, 
      0.018066045996216265, 0.013042117743653343, 0.009310056756035173,
      0.0064363697955691825, 0.004943545400521914, 0.0031263187811662978,
      0.0021244039467980353, 0.0015933029600985265, 0.0009789483052136893,
      0.0006746417939155923, 0.00046794303152443206, 0.0003157897758753836, 
      0.00021818202696844685, 0.00010622019733990177, 0.00010909101348422343, 
      5.7416322886433384e-05, 4.5933058309146705e-05, 2.5837345298895024e-05, 
      1.1483264577286676e-05, 1.4354080721608346e-05, 8.612448432965008e-06, 
      5.741632288643338e-06, 2.870816144321669e-06, 1.1483264577286676e-05, 
      5.741632288643338e-06, 2.870816144321669e-06, 5.741632288643338e-06]
gy_x2=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 
       13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 24.5, 26.5, 31.5]
gy_y2=[0.1595717341203389, 0.1480153532405949, 0.1801601644753204,
       0.1656338600301759, 0.12995125429011986, 0.08943552799562283, 
       0.05652388373982392, 0.03227538010047585, 0.01776358330708139,
       0.009647588413774808, 0.0051087659376917085, 0.0027875416576857394,
       0.0013968796113607348, 0.0006942947623232139, 0.0004393745958582727,
       0.000265282774857825, 0.0001430039958217963, 9.119095385737735e-05,
       4.5595476928688673e-05, 1.865269510719082e-05, 8.290086714307032e-06,
       8.290086714307032e-06, 4.145043357153516e-06, 4.145043357153516e-06, 
       4.145043357153516e-06, 2.072521678576758e-06]


days_x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
days_y=[0.12845753919381742, 0.10491971762652405, 0.09692162384844387, 
        0.09644506836848647, 0.09722593035974197, 0.09704793975879403,
        0.09432066442168845, 0.08402017609586229, 0.07125652751820816,
        0.05450244449994689, 0.03716558580438833, 0.022208633692472432,
        0.011325369689348984, 0.004182779122276672]
days_x2=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
days_y2=[0.0928178833750601, 0.11961144363570043, 0.14607754547112564,
         0.15976862367980368, 0.1565479249912954, 0.13099995025947972, 
         0.09421890802977799, 0.057222323545504286, 0.027918939532107504,
         0.01091182663770663, 0.003131580256329481, 0.0006694245021802929,
         0.00010155356225026114, 2.072521678576758e-06]


unique_x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
          19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 
          35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 
          51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
unique_y=[1.0, 1.6194044256271998, 2.0714290801102413, 2.4356945087181257, 
          2.743220415869217, 3.0132260084413196, 3.2579585393300374, 
          3.4878173198103584, 3.7033142389525366, 3.9021824731576853,
          4.098974455600119, 4.277076494527229, 4.453695714696577, 
          4.619370810269533, 4.782560296846011, 4.940163565426171, 
          5.095918722534047, 5.240658037494008, 5.387071378586424, 
          5.541244231870389, 5.692286716010908, 5.830760306978404, 
          5.967977901682951, 6.094724432655467, 6.213452253218884, 
          6.333564067066606, 6.452222611130557, 6.546977205153618,
          6.6689911146102805, 6.812189213311233, 6.8972445464982775,
          7.00646830530401, 7.081816510387939, 7.17832023169218, 
          7.295992500585892, 7.365594256846584, 7.502119927316778, 
          7.599584343609283, 7.717554240631164, 7.820809248554913,
          7.879676440849343, 7.934378629500581, 8.043536503683859, 
          7.959695817490494, 8.100781928757602, 8.21285140562249, 
          8.322769953051644, 8.371313672922252, 8.5424, 8.586073500967117, 
          8.535307517084282, 8.64516129032258, 8.843450479233226, 
          8.901960784313726, 9.133663366336634, 8.98125, 9.387096774193548, 
          9.306122448979592, 9.301204819277109, 9.149253731343284]
unique_std=[0.0, 0.48553329766416486, 0.7182439645016108, 0.9028614535164982,
            1.0655077451173365, 1.2107217223466178, 1.342716952457698, 
            1.4626822250917804, 1.5768597150121975, 1.6839633812709418, 
            1.7942072064569057, 1.9025630428854485, 2.014694960707279, 
            2.1187682475988585, 2.219253085426121, 2.327098448311181,
            2.4311074770724126, 2.533018668557851, 2.6376753214285467,
            2.7440942218497386, 2.8611442104205698, 2.9637065894527357, 
            3.073485325316518, 3.181852256238784, 3.2822106092137453, 
            3.383096464639754, 3.4894825919733066, 3.5849900830290964, 
            3.6891174173059804, 3.7984467508626514, 3.8959343798639874,
            4.021926909491963, 4.090369850115404, 4.182491234320997,
            4.307958894460668, 4.388786680107311, 4.550127465434194, 
            4.662830476255672, 4.744259225228991, 4.892475967045077,
            4.905649896438985, 4.942384936927696, 5.080871112060882,
            4.929376816698954, 5.060735318139471, 5.20009996640936, 
            5.385844586025935, 5.552593543300759, 5.681848487948266, 
            5.928232900082032, 5.912652495249406, 6.120027340739973, 
            6.524304297060855, 7.012747222473178, 7.725293127544077, 
            7.545256684666201, 8.304399997183145, 8.533736367686167,
            8.258963644194187, 8.10058728540445]
unique_x2=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 
           18, 19, 20, 21, 22, 23, 24, 25, 26]
unique_y2=[1.0, 1.8686745668705405, 2.7036854349490547, 3.5195055590083557,
           4.321966515839994, 5.1106037698578515, 5.886334205941812, 
           6.64845837797111, 7.399049298474277, 8.13103575756331, 
           8.848713271965574, 9.548821343729061, 10.236864103176499, 
           10.915331497946474, 11.576028119507908, 12.213406445837064, 
           12.836183618361837, 13.424657534246576, 13.982617586912065, 
           14.531523642732049, 15.050819672131148, 15.643312101910828, 
           16.124293785310734, 16.36144578313253, 16.666666666666668, 16.807692307692307]
unique_std2=[0.0, 0.33775592332751087, 0.5063981579797935, 0.6493940411350639,
             0.7756697384053591, 0.893757725123888, 1.0047592825219633, 
             1.1130098393203554, 1.2219223555482808, 1.3250803399278728,
             1.4309951075231235, 1.5303296958732724, 1.6395619442160738, 
             1.7459120189229298, 1.8562017947680092, 1.9753979775293906, 
             2.0731417386070117, 2.2243021062540396, 2.3782639823002514, 
             2.6245448523535373, 2.7750128016730073, 3.212263005416616, 
             3.801029846119512, 4.24428265542114, 5.366563145999495, 6.152043005058943]


rank_x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
      20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
      37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
rank_y=[0.6211325821639487, 0.2618635757831092, 0.151161358091383, 
        0.1079527810091704, 0.08578695173633478, 0.07250311194831235,
        0.06317261972891978, 0.05652256326649286, 0.0507826838633238,
        0.046559246132569135, 0.04337603411719515, 0.04070925286880684,
        0.03843997969489675, 0.036652117398675375, 0.03527383716391619, 
        0.03382993418756046, 0.03241679864894437, 0.031534248517360336, 
        0.029939230448917362, 0.029057432845793176, 0.028064512775721268,
        0.026998223226407755, 0.02600877786354551, 0.025353888931685616, 
        0.024483254928012302, 0.024043402170745833, 0.02384831863334255, 
        0.023281101014816983, 0.0232372480278224, 0.022821934302950438, 
        0.021886488166459387, 0.02168711279516211, 0.02057121257589567, 
        0.020254349259325042, 0.019961553215138983, 0.01857603924806707,
        0.018794298910578693, 0.018794298910578693, 0.018288644069375987,
        0.01746078242202392, 0.017567920585161965, 0.017567920585161965, 
        0.017363288052943224, 0.017363288052943224, 0.017363288052943224,
        0.01695402298850575, 0.016666666666666666, 0.016666666666666666]
rank_std=[0.24565104626227138, 0.11460734074476517, 0.06990465143456871,
          0.04879244330356198, 0.037302755816859885, 0.03041417333662599, 
          0.02567292794645066, 0.022367354837239842, 0.01962774671245816, 
          0.017478491235058517, 0.01567363566115989, 0.014260541323362975,
          0.013061063820541665, 0.011676815431528603, 0.01087438444114637,
          0.010098478095227084, 0.009289120803030404, 0.008470302890258432,
          0.007709922948568124, 0.007108607569727138, 0.006524991032754633, 
          0.006016010115866707, 0.005195600455303276, 0.004825147612947614,
          0.00456051507735063, 0.004426172510162348, 0.004358308122054194,
          0.004439371787413362, 0.004493472221960304, 0.004228233193076677,
          0.004077753882713918, 0.004011494207348284, 0.00357130543135277,
          0.003441587024275589, 0.0034369105259081643, 0.0024061979604266368,
          0.002417593308263585, 0.002417593308263585, 0.0020966679603637533, 
          0.0005755011803549205, 0.0006466506684684722, 0.0006466506684684722,
          0.0006245357393605399, 0.0006245357393605399, 0.0006245357393605399,
          0.00028735632183908046, 0.0, 0.0]
rank_x2=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
       20, 21, 22, 23, 24, 25, 26]
rank_y2=[0.35757702826642324, 0.2187646481201412, 0.16920092673315965,
         0.14274972175152958, 0.12508417339055222, 0.11158238532794493,
         0.1006227088269854, 0.09160593192697165, 0.0839870117414486,
         0.0774776881712089, 0.07167250998594021, 0.0666404503991565,
         0.062343518422224235, 0.058456102553896896, 0.055126829026077975,
         0.052061156462632464, 0.04928904538609132, 0.046746650603509556,
         0.04467232383458039, 0.04163039214234205, 0.03976478020039572,
         0.0374193624934921, 0.0339080459770115, 0.0339080459770115, 
         0.0339080459770115, 0.03333333333333333]
rank_std2=[0.23840216592622113, 0.11078606778134936, 0.07182743129565303, 
           0.051310412718528285, 0.03835015447730573, 0.029506198717684087,
           0.023257932175483165, 0.018728815446384262, 0.015415689349780099,
           0.012938589012569618, 0.010970916439205532, 0.00937006392367222, 
           0.00817555702665939, 0.0071527067215670205, 0.006314230165611096,
           0.005617887949664721, 0.005095471154481484, 0.004755504968051454,
           0.004544434199267107, 0.0038672505496691574, 0.003346123077119606, 
           0.00297727439460276, 0.0005747126436781609, 0.0005747126436781609,
           0.0005747126436781609, 0.0]


#user_uni=np.loadtxt('C:/python/MOBIKE_CUP_2017/uni_user.txt',delimiter=',')
#bike_uni=np.loadtxt('C:/python/MOBIKE_CUP_2017/uni_bike.txt',delimiter=',')
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
unique_t_y1=[1.8729928847554487, 1.9296944397593747, 1.9563905099843595, 1.9820856873822976, 
             1.9937676395391308, 2.0710853927795365, 2.0915459103083576, 2.0901525865806247,
             2.0780703246143326, 2.0362588801723844, 2.094924715469739, 1.8726914547420919, 
             2.4379655182147517, 2.429503124148452, 2.3278689299525315, 2.586890410064001, 
             2.948083741799663, 2.9814151816496177, 3.062894877796925, 3.4441214719599706,
             3.5636348745185806, 4.067186140946676, 4.3023287460218915]
unique_t_std1=[0.5194795404200772, 0.6234775075870623, 0.6762570208637936, 0.7228350197446178,
               0.7600401781879038, 0.8023115838932898, 0.7975349506622742, 0.7188118217946023, 
               0.7159374446521611, 0.7231764710227244, 0.8656136637875104, 0.9307132635195926, 
               1.046728274877102, 1.0650223221762904, 1.2122477883146043, 1.30569179919039, 
               1.4486225661960637, 1.5714599336613548, 1.7074084976005182, 1.8680336583678996, 
               2.0109922704905148, 2.332632849797069, 2.5982652853815558]
unique_t_x2=[1.0, 1.6102620275609394, 2.592943797404667, 3.2903445623126686, 4.1753189365604015,
             5.298316906283708, 6.7233575364993365, 8.531678524172808, 10.826367338740546, 
             13.73823795883263, 17.43328822199988, 22.122162910704493, 28.072162039411772, 
             35.62247890262442, 45.20353656360243, 57.36152510448679, 72.7895384398315,
             92.36708571873861, 117.21022975334806, 148.73521072935117, 188.73918221350976,
             239.5026619987486, 303.9195382313198]
unique_t_y2=[1.8158125543041688, 1.8697639076178858, 1.8971769173306774, 1.961301879140192, 
             2.0263640531866116, 2.064982392333449, 2.11451553814052, 2.1268111700743977, 
             2.129142044437098, 2.187300439560439, 2.337019246012873, 2.4882823350023697, 
             2.7205085652262824, 2.811310596196936, 3.0805075596132645, 3.3215560286044865, 
             3.6655377789128254, 4.08924074822837, 4.647964600923781, 5.195871712633977, 
             5.935788920142842, 6.848676182838339, 7.762999273093031]
unique_t_std2=[0.5085299036412826, 0.5502658258157114, 0.572495968675389, 0.5884663653802118,
               0.6216228719623538, 0.633989351384878, 0.6561768950403689, 0.6585114121265445, 
               0.6625807209587843, 0.667114121611588, 0.7432591765710018, 0.8511849354557597, 
               0.9490613281298786, 0.9983798911514615, 1.1308218576706257, 1.246620093394402, 
               1.3887560329882354, 1.5627644584573035, 1.7857358596343311, 2.020246184166073, 
               2.282509749675598, 2.6322746711970844, 2.884940258200458]


user_ith=np.loadtxt('C:/python/MOBIKE_CUP_2017/ith_user.txt',delimiter=',')
bike_ith=np.loadtxt('C:/python/MOBIKE_CUP_2017/ith_bike.txt',delimiter=',')
ith_x=user_ith[:,0]
ith_y=user_ith[:,1]
ith_x2=bike_ith[:,0]
ith_y2=bike_ith[:,1]



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
popt, pcov = curve_fit(func5, trips_x[12:],trips_y[12:],maxfev = 10000)
y777 = [func5(i, *popt) for i in trips_x[:12]]
y888 = [func5(i, *popt) for i in trips_x[12:]]
ax1.scatter(trips_x,trips_y,label=r'user, $\alpha$=%.2f'%popt[1],alpha=alpha_xy)
ax1.plot(trips_x[:12],y777,'b--')
ax1.plot(trips_x[12:],y888,'b-')
popt, pcov = curve_fit(func1, trips_x2[6:],trips_y2[6:],maxfev = 10000)
y888 = [func1(i, *popt) for i in trips_x2]
ax1.scatter(trips_x2,trips_y2,label=r'bike, $\mu$=%.2f, $\sigma$=%.2f'%(popt[0],popt[1]),alpha=alpha_xy)
ax1.plot(trips_x2,y888,color='orange',linestyle='-')
ax1.text(33, 0.2,r'Beijing', size = 18,family=f_family)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_ylim([0.000001,1])
ax1.set_xlim([0.8,80])
ax1.legend(loc=3,handletextpad=0.1,prop={'size':18,'family':f_family})
ax1.set_xlabel('log'+r'$_{10}$'+'#trips',size=18,family=f_family)  
ax1.set_ylabel('log'+r'$_{10}P$(#)',size=18,family=f_family)
#plt.savefig('C:/python/摩拜单车/draw2/2_bj4.pdf',bbox_inches='tight') 
        

fig2	=	plt.figure(figsize=(size_x, size_y))
ax2	=	fig2.add_subplot(1,	1,	1)
plt.yticks(fontproperties = f_family)
plt.xticks(fontproperties = f_family) 
plt.tick_params(labelsize=18)
popt, pcov = curve_fit(func4, dis_x11[4:],dis_y11[4:])
y777 = [func4(i, *popt) for i in dis_x11]
ax2.scatter(dis_x11,dis_y11,label=r'user, $\gamma$=%.2f'%-popt[1],alpha=alpha_xy)
ax2.plot(dis_x11,y777,'b-')
popt, pcov = curve_fit(func4, dis_x22[4:],dis_y22[4:],maxfev = 10000)
y888 = [func4(i, *popt) for i in dis_x22]
ax2.scatter(dis_x22,dis_y22,label=r'bike, $\gamma$=%.2f'%-popt[1],alpha=alpha_xy)
ax2.plot(dis_x22,y888,color='orange',linestyle='-')
ax2.text(13, 0.2,r'Beijing', size = 18,family=f_family)
ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.set_ylim([0.000001,1])
ax2.set_xlim([0.3,30])
ax2.legend(loc=3,handletextpad=0.1,prop={'size':18,'family':f_family})
ax2.set_xlabel('log'+r'$_{10}\langle d \rangle$'+'(km)',size=18,family=f_family) 
ax2.set_ylabel('log'+r'$_{10}P(\langle d \rangle)$',size=18,family=f_family)
#plt.savefig('C:/python/摩拜单车/draw2/2_bj6.pdf',bbox_inches='tight') 


no=6
fig3	=	plt.figure(figsize=(size_x, size_y))
ax3	=	fig3.add_subplot(1,	1,	1)
plt.yticks(fontproperties = f_family)
plt.xticks(fontproperties = f_family) 
plt.tick_params(labelsize=18)
#popt, pcov = curve_fit(func2, gy_x[:1]+gy_x[no:],gy_y[:1]+gy_y[no:],maxfev = 100000)
popt, pcov = curve_fit(func5, gy_x[:1]+gy_x[6:],gy_y[:1]+gy_y[6:],maxfev = 100000)
y777 = [func5(i, *popt) for i in gy_x]
ax3.scatter(gy_x,gy_y,label=r'user, $\alpha$=%.2f'%popt[1])
ax3.plot(gy_x,y777,'b-')
print(popt)
popt, pcov = curve_fit(func1, gy_x2[7:],gy_y2[7:],maxfev = 10000)
y888 = [func1(i, *popt) for i in gy_x2]
ax3.scatter(gy_x2,gy_y2,label=r'bike, $\mu$=%.2f, $\sigma$=%.2f'%(popt[0],popt[1]),alpha=alpha_xy)
ax3.plot(gy_x2,y888,color='orange',linestyle='-')
#popt, pcov = curve_fit(func3, gy_x2[2:],gy_y2[2:],maxfev = 100000)
#y888 = [func3(i, *popt) for i in gy_x2]
#ax3.scatter(gy_x2,gy_y2,label=r'bike, $\lambda$=%.2f'%popt[0])
#ax3.plot(gy_x2,y888,color='orange',linestyle='-')
ax3.text(20, 0.2,r'Beijing', size = 18,family=f_family)
ax3.set_yscale('log')
ax3.set_xscale('log')
ax3.set_ylim([0.000001,1])
ax3.set_xlim([0.3,50])
ax3.legend(loc=3,handletextpad=0.1,prop={'size':18,'family':f_family})
ax3.set_xlabel('log'+r'$_{10}r_g$'+'(km)',size=18,family=f_family) 
ax3.set_ylabel('log'+r'$_{10}P(r_g)$',size=18,family=f_family)
#plt.savefig('C:/python/摩拜单车/draw2/2_bj5.pdf',bbox_inches='tight') 


fig6	=	plt.figure(figsize=(size_x, size_y))
ax6	=	fig6.add_subplot(1,	1,	1)
plt.yticks(fontproperties = f_family)
plt.xticks(fontproperties = f_family) 
plt.tick_params(labelsize=18)
popt, pcov = curve_fit(func4, rank_x[1:],rank_y[1:],maxfev = 10000)
y777 = [func4(i, *popt) for i in rank_x]
ax6.errorbar(rank_x,rank_y,fmt="o",yerr = rank_std,label=r'user, $\xi$=%.2f'%-popt[1],alpha=alpha_xy)
ax6.plot(rank_x,y777,'b-',linewidth=2)
popt, pcov = curve_fit(func4, rank_x2,rank_y2,maxfev = 10000)
y888 = [func4(i, *popt) for i in rank_x2]
ax6.errorbar(rank_x2,rank_y2,fmt="o",yerr = rank_std2,label=r'bike, $\xi$=%.2f'%-popt[1],alpha=alpha_xy)
ax6.plot(rank_x2,y888,color='orange',linestyle='-',linewidth=2)
ax6.text(25, 0.6,r'Beijing', size = 18,family=f_family)
ax6.set_yscale('log')
ax6.set_xscale('log')
ax6.set_ylim([0.01,1])
ax6.legend(loc=3,handletextpad=0.1,prop={'size':18,'family':f_family})
ax6.set_xlabel('log'+r'$_{10}L$',size=18,family=f_family) 
ax6.set_ylabel('log'+r'$_{10}P(L)$',size=18,family=f_family) 
#plt.savefig('C:/python/摩拜单车/draw2/2_bj2.pdf',bbox_inches='tight') 


fig7	=	plt.figure(figsize=(size_x, size_y))
ax7	=	fig7.add_subplot(1,	1,	1)
plt.yticks(fontproperties = f_family)
plt.xticks(fontproperties = f_family) 
plt.tick_params(labelsize=18)
popt, pcov = curve_fit(func4, unique_t_x1[10:],unique_t_y1[10:])
y777 = [func4(i, *popt) for i in unique_t_x1[0:]]
#ax7.scatter(unique_t_x1,unique_t_y1,label=r'user, $\mu$=%.2f'%popt[1],alpha=alpha_xy)
ax7.errorbar(unique_t_x1,unique_t_y1,fmt="o",yerr = unique_t_std1,label=r'user, $\epsilon$=%.2f'%popt[1],alpha=alpha_xy)
ax7.plot(unique_t_x1[0:],y777,'b-')
popt, pcov = curve_fit(func4, unique_t_x2[10:],unique_t_y2[10:])
y888= [func4(i, *popt) for i in unique_t_x2[0:]]
#ax7.scatter(unique_t_x2,unique_t_y2,label=r'bike, $\mu$=%.2f'%popt[1],alpha=alpha_xy)
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
#plt.savefig('C:/python/摩拜单车/draw2/2_bj1.pdf',bbox_inches='tight') 


fig8	=	plt.figure(figsize=(size_x, size_y))
ax8	=	fig8.add_subplot(1,	1,	1)
plt.yticks(fontproperties = f_family)
plt.xticks(fontproperties = f_family) 
plt.tick_params(labelsize=18)
popt, pcov = curve_fit(func5, ith_x[:], ith_y[:],maxfev = 10000)
y777 = [func5(i, *popt) for i in ith_x]
ax8.scatter(ith_x,ith_y,label=r'user, $\alpha$=%.2f'%popt[1])
ax8.plot(ith_x,y777,color='blue',linestyle='-')
popt, pcov = curve_fit(func5, ith_x2[0:],ith_y2[0:],maxfev = 1000000)
y888 = [func5(i, *popt) for i in ith_x2]
ax8.scatter(ith_x2,ith_y2,label=r'bike, $\alpha$=%.2f'%popt[1])
ax8.plot(ith_x2,y888,color='orange',linestyle='-')
ax8.text(55, 0.55,r'Beijing', size = 18,family=f_family)
ax8.set_ylim([-0.04,0.64])
#ax8.legend(loc=3,handletextpad=0.1,prop={'size':18,'family':f_family})
#ax8.set_yscale('log')
#ax8.set_xscale('log')
#ax8.set_ylim([0.0000001,1])
ax8.set_xlabel(r'i$^{th}$ trip',size=18,family=f_family)  
ax8.set_ylabel(r'$P(i)$',size=18,family=f_family)
#plt.savefig('C:/python/摩拜单车/draw2/2_bj3.pdf',bbox_inches='tight') 









#fig4	=	plt.figure(figsize=(size_x, size_y))
#ax4	=	fig4.add_subplot(1,	1,	1)
#plt.yticks(fontproperties = f_family)
#plt.xticks(fontproperties = f_family) 
#plt.tick_params(labelsize=18)
#popt, pcov = curve_fit(func1, days_x,days_y)
#y777 = [func1(i, *popt) for i in days_x]
#ax4.scatter(days_x,days_y,label=r'user, $\mu$=%.2f, $\sigma$=%.2f'%(popt[0],popt[1]),alpha=0.6)
#ax4.plot(days_x,y777,'b-')
##popt, pcov = curve_fit(func1, days_x2[:],days_y2[:],maxfev = 10000)
##y888 = [func1(i, *popt) for i in days_x2]
##ax4.scatter(days_x2,days_y2,label=r'bike, $\mu$=%.2f, $\sigma$=%.2f'%(popt[0],popt[1]),alpha=0.6)
##ax4.plot(days_x2,y888,color='orange',linestyle='-')
#ax4.set_yscale('log')
#ax4.set_xscale('log')
#ax4.set_ylim([0.0000001,1])
#ax4.set_ylim([0.001,1])
#ax4.legend(loc=3,handletextpad=0.1,prop={'size':18,'family':f_family})
#ax4.set_xlabel('#Active Days',size=18,family=f_family) 
#ax4.set_ylabel(r'$P$(#)',size=18,family=f_family)
##plt.savefig('C:/python/摩拜单车/draw2/2_bj7.pdf',bbox_inches='tight') 
#
#        
#fig5	=	plt.figure(figsize=(size_x, size_y))
#ax5	=	fig5.add_subplot(1,	1,	1)
#plt.yticks(fontproperties = f_family)
#plt.xticks(fontproperties = f_family) 
#plt.tick_params(labelsize=18)
#ax5.errorbar(unique_x,unique_y,fmt="o",yerr = unique_std,label='user')
#ax5.errorbar(unique_x2,unique_y2,fmt="o",yerr = unique_std2,label='bike')
#ax5.plot([0,65],[0,65],'k-',linewidth=2)      
#ax5.set_ylim([0,65])
#ax5.set_xlim([0,65])
#ax5.legend(loc=2,handletextpad=0.1,prop={'size':18,'family':f_family})
#ax5.set_xlabel('#trips',size=18,family=f_family)  
#ax5.set_ylabel('#Unique Locations',size=18,family=f_family)
#plt.savefig('C:/python/摩拜单车/draw2/2_bj8.pdf',bbox_inches='tight') 
               
               
               
               