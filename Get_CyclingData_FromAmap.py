import csv
import requests
import pandas as pd

# 调用高德地图web服务API，计算出发地到目的地的骑行距离与时间
def get_dis_time(origin, destination, key):
    '''
    @param origin: 起点经纬度，格式X,Y  采用","分隔，例如“ 117.500244, 40.417801 ”
    @param destination: 终点经纬度，格式X,Y  采用","分隔，例如“ 117.500244, 40.417801 ”
    @param key: 请求服务权限标识，需要用户在高德地图官网申请
    @return: Paths, Paths包含骑行距离（单位：m）和骑行时间(单位：s)，例如"[266, 1108]"
    '''
    parameters = {'output': 'json', 'origin': origin, 'destination': destination, 'key': key}
    base = 'https://restapi.amap.com/v4/direction/bicycling?parameters'
    response = requests.get(base, parameters)
    answer = response.json()
    Paths = []
    if int(answer['errcode']) == 0:
        if len(answer['data']['paths']) > 0:
            Paths.append(answer['data']['paths'][0]['duration'])
            Paths.append(answer['data']['paths'][0]['distance'])
        else:
            Paths = []
    else:
        Paths.append(answer['errdetail'])  # 出错返回错误原因
    return Paths

# 将结果写入csv数据文件，存储格式每一行为对应的id, [time, distance]
def write_csv(sql_data):
    """
    @param sql_data: 一条记录的返回数据，对应格式为[id, [time, distance]],例如[['549189', [416, 1733]]]
    """
    file_name = 'Cycling_Time.csv'
    try:
        with open(file_name, 'a+', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for row in sql_data:
                writer.writerow(row)
    except UnicodeEncodeError:
        print("编码错误, 该数据无法写到文件中, 直接忽略该数据")

if __name__ == '__main__':
    # 读取文件
    filename = '../data/Mobike_Cup_2017_Beijing_LONLAT_final.csv'
    csv_data = pd.read_csv(filename, encoding='gbk')  # 读取训练数据
    orderid_list = csv_data['orderid'].values.tolist()
    startx_list = csv_data['start_x'].values.tolist()
    starty_list = csv_data['start_y'].values.tolist()
    endx_list = csv_data['end_x'].values.tolist()
    endy_list = csv_data['end_y'].values.tolist()

    # 共 3214096 条记录
    start = []  # 起点经纬度列表
    end = []  # 重点经纬度列表
    for i in range(len(orderid_list)):
        start.append(str(startx_list[i]) + ',' + str(starty_list[i]))
        end.append(str(endx_list[i]) + ',' + str(endy_list[i]))
    # print(len(start))  # 3214096
    # 调用API获取骑行距离和骑行时间
    number = 3214096  # 需要计算的记录总个数，对应文件Mobike_Cup_2017_Beijing_LONLAT_final.csv中记录个数
    # 在https://lbs.amap.com/dev/申请key
    key = ['此处填入key', '此处填入key', '此处填入key'] #Load real keys from a file or put them here mannually
    k = 0  # 用于更换key的值
    key_k = 0  #  用于检测单个用户的调用是否达到上限30000
    for i in range(number):
        row = []
        row_all = []
        if key_k < 29000:  #Maybe smaller than 30000 would be better
            Dis_time = get_dis_time(start[i], end[i], key[k])
            key_k = key_k + 1
        else:
            k = k + 1
            key_k = 0
            Dis_time = get_dis_time(start[i], end[i], key[k])
            key_k = key_k + 1
        row.append(str(orderid_list[i]))
        row.append(Dis_time)
        row_all.append(row)
        write_csv(row_all)
        # print(i)  # 用于检测调用到哪条记录，防止出现数据不对应的情况
