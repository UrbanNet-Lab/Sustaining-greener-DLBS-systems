#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import geopandas as gpd
import numpy as np
import transbigdata as tbd
import matplotlib.pyplot as plt
tbd.set_mapboxtoken('pk.eyJ1Ijoia2VpdGh5dWEiLCJhIjoiY2xlY3JiZ3pqMDBjdzNwbGFscHN1ZTlxOSJ9.pVZdLE3nHEfXOlElsUp9lg')


# # readme
# 1. 处理订单数据
# 2. 结合栅格信息，确定每个订单的起点栅格、终点栅格
# 3. 导出CSV文件

# In[18]:


def getGridId(x):
    sid = ''+str(x['start_lng_col'])+','+str(x['start_lat_col'])
    eid = ''+str(x['end_lng_col'])+','+str(x['end_lat_col'])
    return sid ,eid


# In[11]:


COLUMNS = ['ORDER_ID','USER_ID','BIKE_ID','BIKE_TYPE','START_TIME','END_TIME','START_X','START_Y','END_X','END_Y','SID','EID']


# In[12]:


path = './Mobike_bj2.csv'
df = pd.read_csv(path)
df.columns


# In[13]:


df1 = df[['orderid','userid','bikeid','biketype','starttime','endtime','start_x','start_y','end_x','end_y']]
df1
df1.rename({'orderid':'ORDER_ID'})


# In[30]:



df1[:3000].to_csv('SAMPLE_BEIJING.CSV',index=False)


# In[4]:


district = gpd.read_file(r'./beijing.geojson')
district.crs = None
district.head()

fig = plt.figure(1, (7, 5), dpi=150)
ax1 = plt.subplot(111)
district.plot(ax=ax1)
bounds = [119, 28, 121, 30]
# tbd.plot_map(plt, bounds, zoom=11)


# In[7]:


fig.savefig('1.png')


# In[14]:


# 剔除起点、终点不在这里面的地方
data = tbd.clean_outofshape(df1, district, col=['start_x', 'start_y'], accuracy=1000)
data = tbd.clean_outofshape(df1, district, col=['end_x', 'end_y'], accuracy=1000)
data


# In[15]:


# 地图栅格化
grid_rec, params_rec = tbd.area_to_grid(district, accuracy=1000)
print(len(grid_rec))


# In[19]:


data = pd.read_csv('beijing-1000m.csv')
data['starttime'] = pd.to_datetime(data['starttime'])
data['endtime'] = pd.to_datetime(data['endtime'])
data['start_lng_col'], data['start_lat_col'] = tbd.GPS_to_grid(data['start_x'], data['start_y'], params_rec)
data['end_lng_col'], data['end_lat_col'] = tbd.GPS_to_grid(data['end_x'], data['end_y'], params_rec)
data[['sid','eid']] = data.apply(getGridId, axis=1, result_type='expand')


# In[32]:


group_counts = data.groupby(['sid']).size()
print(group_counts)


# In[16]:


# data['start_time'] = pd.to_datetime(data['start_time'])
# data['end_time'] = pd.to_datetime(data['end_time'])
# data.dtypes


# In[30]:


# data['start_time'] = pd.to_datetime(data['start_time'])
# data['end_time'] = pd.to_datetime(data['end_time'])
# data = data.sort_values(by = 'start_time')
data = data.sort_values(by = 'starttime')
# 存储地址
save_path = 'beijing-1000m.csv'
data.to_csv(save_path, columns=data.columns, index=False)
data


# In[8]:


data = pd.read_csv('beijing-1000m.csv')
data['starttime'] = pd.to_datetime(data['starttime'])
data['endtime'] = pd.to_datetime(data['endtime'])
one_minute = pd.Timedelta(minutes=1)
data['t_delta']=data['endtime'] - data['starttime']
data[data['t_delta']>one_minute]


# In[31]:


# st_time = ''
# ed_time = ''
data2 = data[(data['starttime'] >= '2017-5-10 00:00:00') & (data['starttime'] <'2017-5-17 00:00:00')].sort_values('starttime',ascending=True)
data2.to_csv('bj-1000m-10-16.csv',columns=data2.columns,index=False)
data2
# save_path = 'shanghai-1000m-16-22.csv'
# data2.to_csv(save_path, columns=data.columns, index=False)


# In[12]:


# data2 = data[(data['starttime'] >= '2017-5-16 00:00:00')  ].sort_values('starttime',ascending=True)
# data2.to_csv('bj-1000m-17-24.csv',columns=data2.columns,index=False)
# data2


# In[42]:


data1 = data[(data['starttime'] < '2017-5-17 00:00:00')]
data2 = data[(data['starttime'] >= '2017-5-17 00:00:00')]
data2["starttime"] = data2["starttime"] - pd.Timedelta(days=1)
data2["endtime"] = data2["endtime"] - pd.Timedelta(days=1)

data3 = pd.concat([data1,data2])
data3
data3.to_csv('bj_1000m_2.csv', columns=data.columns, index=False)
# data[(data['starttime'] >= '2017-5-17 00:00:00')]["starttime"] =time


# In[30]:


data[(data['starttime'] >= '2017-5-17 00:00:00')]


# In[ ]:


data2 = data[(data['starttime'] >= '2017-5-17 00:00:00') ].sort_values('starttime',ascending=True)
data2.to_csv('bj-1000m-17-24.csv',columns=data2.columns,index=False)
data2


# In[ ]:


tbd.visualization_od(data, col=['StartLng', 'StartLat', 'EndLng', 'EndLat'], mincount=100, accuracy=1000)


# In[22]:


a= tbd.visualization_data(data,['start_lng_col','start_lat_col'],accuracy=1000)


# In[25]:


a.save_to_html()

