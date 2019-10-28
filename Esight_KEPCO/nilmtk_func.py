#!/usr/bin/env python
# coding: utf-8

# # import
# 
# ---

# In[1342]:


from __future__ import print_function, division
import time

from bokeh.core.property.dataspec import value
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from six import iteritems
from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
# from nilmtk.legacy.disaggregate import CombinatorialOptimisation, FHMM
import nilmtk.utils
from math import pi
import pandas as pd
from bokeh.io import output_file, show
from bokeh.palettes import Category20c
from bokeh.plotting import figure
from bokeh.transform import cumsum
import nilmtk
from nilmtk.dataset_converters import convert_redd
from nilmtk import DataSet
from nilmtk.utils import print_dict
from bokeh.io import output_file, show
from bokeh.layouts import gridplot
from bokeh.palettes import Viridis3
from bokeh.plotting import figure
from bokeh.palettes import Dark2_5 as palette
import itertools  
from bokeh.palettes import Category10, Set3, Paired, Pastel2
from bokeh.palettes import brewer
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.embed import components

from flask import Flask, render_template


# # 데이터 컨버터 (return : DataSet)
# 
# ---

# In[ ]:


# 데이터 컨버트 
# .dat ==> .h5
convert_redd('C:\\Users\\dlsrk\\Desktop\\nilm\\low_freq', 'C:\\Users\\dlsrk\\Desktop\\nilm\\data\\redd.h5')
# .h5 데이터(컨버트된) read
# redd = DataSet('C:\\Users\\Kim-Taesu\\Documents\\nilm\\data\\redd.h5')
redd = DataSet('C:\\Users\\dlsrk\\Desktop\\nilm\\data\\redd.h5')


# date load 함수
def getData(inputPath, convertOutputPath):
    convert_redd(inputPath, convertOutputPath)
    return DataSet(convertOutputPath)


# # 시각화 데이터 준비
# 
# ---

# ### 변수 

# > buildings_count : 전체 빌딩 수 (int)
# 
# > buildingsElec : 각 빌딩의 전력 정보 (dict{ 빌딩 번호 : 빌딩 전력 정보 })
# 
# > buildingAppliances : 각 빌딩의 가전기기 정보( dict{ 빌딩 번호 : 해당 빌딩에 있는 가전기기 목록(list) })
# 
# > buildingsPower : 각 빌딩의 총 전력량 정보( dict{ 빌딩 번호 : 빌딩 총전력량(dataframe) })

# ###  전체 빌딩 개수 (return : buildingCount(int))

# In[626]:


# 빌딩 총 count 
def getBuildingsCnt(data):
    return len(data)
buildings_count = getBuildingsCnt(redd.buildings)
print(buildings_count)


# ### 각 빌딩 전력 정보 (return : dict{ building(int) : elecInfo })

# In[625]:


def getBuildinsElec(buildingsNum):
    buildingsElec = {}
    for x in range(1, buildingsNum+1):
        buildingsElec[x]=redd.buildings[x].elec
    return buildingsElec
buildingsElec = getBuildinsElec(buildings_count)
print_dict(buildingsElec)


# ### 각 빌딩 가전기기 종류 (return : dict{ building(int) : list })

# In[526]:


def getBuildingAppliances(buildingsNum):
    buildingAppliances={}
    for i in range(1,buildingsNum+1):
        appliancesLen = len(buildingsElec[i].appliances)
        buildingAppliances[i]=[]
        for j in range(appliancesLen):
            buildingAppliances[i].append(buildingsElec[i].appliances[j].type['type'])
    return buildingAppliances
buildingAppliances=getBuildingAppliances(buildings_count)
# print_dict(buildingAppliances)


# ### 각 빌딩 전력데이터 (return : Dict{ building(int) : DataFrame })

# In[535]:


def getBuildingsPower(buildingsNum):
    buildingsPower={}
    for x in range(1, buildingsNum+1):
        buildingsPower[x]=next(buildingsElec[x].load())['power'].dropna()
    return buildingsPower
buildingsPower = getBuildingsPower(buildings_count)
print_dict(buildingsPower)


# # 시각화
# 
# ---

# ###  특정 building의 시간당 에너지 사용량 시각화 (output file : html/{빌딩 번호}__usage_per_hour.html

# In[1293]:


def building_usage_per_hour(buildingNum, flag):
    tmp = buildingsPower[buildingNum].resample(rule='H').mean()
    p = figure(x_axis_label='timestamp', y_axis_label ='power', x_axis_type='datetime', title='building'+str(buildingNum))
    p.line(tmp.index, tmp['apparent'], color="firebrick", legend="apparent")
    p.line(tmp.index, tmp['active'], color="navy", legend="active")

    if flag:
        pass
        # output_file('html/building'+str(buildingNum)+'_usage_per_hour.html')
        # show(p)
    return p
#
# # example
# building_usage_per_hour(1,True)


# ###  각 building의 시간당 에너지 사용량 시각화 종합 (output file : html/all_buildings_usage_per_hour.html)

# In[1295]:


def all_buildings_usage_per_hour(x_size):
    tmp=[]
    tmp_x = []
    for i in range(1, buildings_count+1):
        tmp_x.append(building_usage_per_hour(i, False))
        if i % x_size==0:
            tmp.append(tmp_x)
            tmp_x=[]
    grid = gridplot(tmp)
    # output_file('html/all_buildings_usage_per_hour.html')
    # show(grid)
    
    
# example
# all_buildings_usage_per_hour(3)


# ### 모든 building의 시간당 에너지 사용량 시각화 (output : html/total_usage_per_hour.html)

# In[1296]:

#############################################################
def total_usage_per_hour(buildingNum, flag):
    tmp = pd.DataFrame()
    
    for i in range(1,buildingNum+1):
        tmp = pd.concat((tmp , buildingsPower[i].resample(rule='H').mean()))
    tmp = tmp.groupby(tmp.index).mean()
    
    if flag:
        p = figure(x_axis_label='timestamp', y_axis_label ='power', x_axis_type='datetime', title='Total Energy Status per hour')
        p.line(tmp.index, tmp['apparent'], color="firebrick", legend="apparent")
        p.line(tmp.index, tmp['active'], color="navy", legend="active")
        # output_file('total_usage_per_hour.html')
        # show(p)
    else:
        return tmp
    

# # example
# total_usage_per_hour(buildings_count,True)


# ### 각 지역의 시간당 에너지 사용량 시각화 (output : html/district_usage_per_hour.html)

# In[1297]:


def district_usage_per_hour(buildingNum, flag):
    tmp = pd.DataFrame()
    for i in range(1,(buildingNum//2)+1):
        tmp = pd.concat((tmp , buildingsPower[i].resample(rule='H').mean()))
    tmp = tmp.groupby(tmp.index).mean()
    
    tmp2 = pd.DataFrame()
    for i in range((buildingNum//2)+1, buildingNum+1):
        tmp2 = pd.concat((tmp2 , buildingsPower[i].resample(rule='H').mean()))
    tmp2 = tmp2.groupby(tmp2.index).mean()
    
    p1 = figure(x_axis_label='timestamp', y_axis_label ='power', x_axis_type='datetime', title='District 1 Total Energy Status per hour')
    p1.line(tmp.index, tmp['apparent'], color="firebrick", legend="apparent")
    
    p2 = figure(x_axis_label='timestamp', y_axis_label ='power', x_axis_type='datetime', title='District 2 Total Energy Status per hour')
    p2.line(tmp2.index, tmp2['apparent'], color="firebrick", legend="apparent")
    
    result = row(p1,p2)
    # output_file('district_usage_per_hour.html')
    # show(result)
    
# example
# district_usage_per_hour(buildings_count,True)


# ### 특정 빌딩의 가전기기 사용량 확인 (output : html/{빌딩번호}_appliances_usage.html)

# In[1298]:


# buildingsPower
def building_appliances_usage(buildingNum, flag):
    
    buildingP = figure(x_axis_label='timestamp', y_axis_label ='active', x_axis_type='datetime', title='building'+str(buildingNum))    
    buildingElec = buildingsElec[buildingNum]
    buildingAppliance = buildingAppliances[buildingNum]
    colors = Category20c[20]
    
    legendList={}
    
    for index in range(3,len(buildingAppliance)):
        if buildingAppliance[index-3] in legendList:
            legendList[buildingAppliance[index-3]]=legendList[buildingAppliance[index-3]]+1
        else:
            legendList[buildingAppliance[index-3]]=1
        
        buildingDf= next(buildingElec[index].load())['power'].dropna().resample(rule='H').mean()
        buildingP.line(buildingDf.index, buildingDf['active'], color=colors[index-3%20], legend=buildingAppliance[index-3]+str(legendList[buildingAppliance[index-3]]))
        
    buildingP.legend.location = "top_left"
    buildingP.legend.click_policy="hide"
    if flag:
        pass
        # output_file('html/building'+str(buildingNum)+'_appliances_usage.html')
        # show(buildingP)
    return buildingP

# example
# buildingDf = building_appliances_usage(3, True)


# ### 각 building내 가전기기의 시간당 에너지 사용량 시각화 종합 
# ### (output file : html/all_buildings_appliances_usage.html)

# In[1300]:


def all_buildings_appliances_usage(x_size):
    tmp=[]
    tmp_x = []
    for i in range(1, buildings_count+1):
        tmp_x.append(building_appliances_usage(i, False))
        if i % x_size==0:
            tmp.append(tmp_x)
            tmp_x=[]
    print(tmp)
    grid = gridplot(tmp)
    # output_file('html/all_buildings_appliances_usage.html')
    # show(grid)
    
# example
# all_buildings_appliances_usage(3)


# ### 모든 빌딩의 가전기기 사용량 확인 (output : html/building{Num}.html)

# In[727]:


# buildingsPower
def buildingsVisualization(x_size):
    tmp=[]
    tmp_x = []
    for i in range(1, buildings_count+1):
        tmp_x.append(buildingVisualization(i, False))
        if i % x_size==0:
            tmp.append(tmp_x)
            tmp_x=[]
    grid = gridplot(tmp)
    # output_file('html/buildingsVisualization.html')
    # show(grid)
    
# example
# buildingsVisualization(3)


# ### 각 빌딩의 가전제품 에너지 소모 비율 (output : buildings_fraction.html)

# In[1886]:


def buildingApplianceFraction(buildingNum):
    elec = buildingsElec[buildingNum]
    fraction = elec.submeters().fraction_per_meter().dropna()
    labels = elec.get_labels(fraction.index)
    
    labelsDup={}
    for l in labels:
        labelsDup[l]=1

    for x in range(len(labels)):
        tmp = labels.pop(0)
        if tmp in labels:
            labels.append(tmp+str(labelsDup[tmp]))
            labelsDup[tmp]=labelsDup[tmp]+1
        else:
            labels.append(tmp)

    x={}
    for i in range(len(labels)):
        x[labels[i]]=fraction[i]

    data = pd.Series(x).reset_index(name='value').rename(columns={'index':'country'})

    data['angle'] = data['value']/data['value'].sum() * 2*pi
    
    if len(x)>20:
        colorTmp = Category20c[20][:len(data['angle'])]
        for i in range(len(x)-20-2):
            colorTmp.append(colorTmp[i])
        data['color'] = colorTmp
        
    else:
        data['color'] = Category20c[len(x)]
    data.head()

    p = figure(title="building"+str(buildingNum)+" fraction per meter", toolbar_location=None,
               tools="hover", tooltips="@country: @value", x_range=(-0.5, 1.0))

    p.wedge(x=0, y=1, radius=0.4,
            start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
            line_color="white", fill_color='color', legend='country', source=data)

    p.axis.axis_label=None
    p.axis.visible=False
    p.grid.grid_line_color = None
    
    
    return p

def buildingApplianceFractionView(x_size):
    list1=[]
    list2=[]
    for i in range(1,4):
        list1.append(buildingApplianceFraction(i))
    for i in range(4,7):
        list2.append(buildingApplianceFraction(i))
    grid = gridplot([list1, list2])
    output_file('html/buildings_fraction.html')
    show(grid)
# buildingApplianceFractionView(1)


# # 공모전 시각화
# ---

# ### 시간별 평균 활동 (전국)

# - startDate, endDate 설정

# In[1859]:


# start='2011-04-18'
# end='2011-04-18'


# -  모든 건물의 시간당 총 전력량

# In[1860]:


# total = total_usage_per_hour(buildings_count,False).loc[start:end]


# - 시각화 차트의 x 좌표가 될 시간 리스트

# In[1861]:


# timeData=total.index.strftime('%H').tolist()


# -  시각화할 가전기기들

# In[1862]:


# appliances = ["fridge", "light", "microwave","electric oven", "washer dryer", "dish washer", "electric space heater"]
#
#
# # - data 설정
#
# # In[1863]:
#
#
# data={'timestamp' : timeData}


# -  가전기기들의 시간당 사용량 계산 함수

# In[1864]:


def applianceTimeUsage(appliance, buildingNum, start, end):
    tmp= pd.DataFrame()
    for i in range(1,buildingNum+1):
        try:
            tmpDf = next(buildingsElec[i][appliance].load())['power']
        except KeyError:
            continue
        tmp=pd.concat((tmp,tmpDf))
    tmp=tmp.resample(rule='H').mean()
    return tmp


# -  가전기기들의 전체 전력 사용량 비율 계산

# In[1865]:

#
# for a in appliances:
#     tmp = applianceTimeUsage(a,buildings_count,start,end)
#     tmp = tmp.loc[start:end]
#     tmp = (tmp['active']/total['active']).values
#     data[a]=np.nan_to_num(tmp, copy=False).tolist()


# - 가전기기들의 사용량 비율 총합 계산

# In[1866]:


# tmpTotal=[0]*len(timeData)
# for appliance in appliances:
#     for l in range(len(timeData)):
#         tmpTotal[l] += data[appliance][l]


# - 시각화할 가전기기들 사이의 비율 계산 (y값 max값이 1로 만들기 위해)

# In[1867]:


# for appliance in appliances:
#     for l in range(len(timeData)):
#         data[appliance][l]=data[appliance][l]/tmpTotal[l]
#
#
# # -  시각화 색깔 설정
#
# # In[1869]:
#
#
# colors = Pastel2[len(appliances)]


# - 시각화 데이터 및 축 설정

# In[1870]:


# p = figure(x_range=data['timestamp'], plot_height=750,plot_width=1750, title="시간별 평균 활동(전국)",
#            toolbar_location=None, tools="hover", tooltips="$name @timestamp: @$name")
# p.vbar_stack(appliances, x='timestamp', width=0.5, color=colors, source=data,
#              legend=[value(x) for x in appliances])
#
#
# # - 시각화 옵션 설정
#
# # In[1871]:
#
#
# p.y_range.start = 0
# p.y_range.end = 1.0
# p.x_range.range_padding = 0.5
# p.xgrid.grid_line_color = None
# p.axis.minor_tick_line_color = None
# p.outline_line_color = None
# p.legend.location = "top_left"
# p.legend.orientation = "vertical"
# p.legend.click_policy="hide"


# - 시각화 출력 및 저장

# In[1872]:


# output_file("html/appliance_per_hour.html")
# show(p)


# > 전체 코드

# In[1888]:

#############################################################
def applianceTimeUsage(appliance, buildingNum, start, end):
    tmp= pd.DataFrame()
    for i in range(1,buildingNum+1):
        try:
            tmpDf = next(buildingsElec[i][appliance].load())['power']
        except KeyError:
            continue
        tmp=pd.concat((tmp,tmpDf))
    tmp=tmp.resample(rule='H').mean()
    return tmp

def appliancesTimeUsage(d,start,end,buildings_total):
    total = total_usage_per_hour(buildings_total,False).loc[start:end]
    timeData=total.index.strftime('%H').tolist()
    appliances = ["fridge", "light", "microwave","electric oven", "washer dryer", "dish washer", "electric space heater"]
    data={'timestamp' : timeData}
    for a in appliances:
        tmp = applianceTimeUsage(a,buildings_total,start,end) 
        tmp = tmp.loc[start:end]
        tmp = (tmp['active']/total['active']).values
        data[a]=np.nan_to_num(tmp, copy=False).tolist()
    tmpTotal=[0]*len(timeData)
    for appliance in appliances:
        for l in range(len(timeData)):
            tmpTotal[l] += data[appliance][l]
    for appliance in appliances:
        for l in range(len(timeData)):
            data[appliance][l]=data[appliance][l]/tmpTotal[l]

    colors = Pastel2[len(appliances)]
    p = figure(x_range=data['timestamp'], plot_height=750,plot_width=1750, title="시간별 평균 활동(전국)",
           toolbar_location=None, tools="hover", tooltips="$name @timestamp: @$name")
    p.vbar_stack(appliances, x='timestamp', width=0.5, color=colors, source=data,
                 legend=[value(x) for x in appliances])
    p.y_range.start = 0
    p.y_range.end = 1.0
    p.x_range.range_padding = 0.5
    p.xgrid.grid_line_color = None
    p.axis.minor_tick_line_color = None
    p.outline_line_color = None
    p.legend.location = "top_left"
    p.legend.orientation = "vertical"
    p.legend.click_policy="hide"
    # output_file("html/appliance_per_hour.html")
    # show(p)
    
# appliancesTimeUsage(data,start,end,buildings_total)


# > 파이차트

# In[1889]:


def getPieChart(d):
    import copy
    x = copy.deepcopy(d)
    del x['timestamp']
    for k in x.keys():
        tmp = x[k]
        x[k] = sum(tmp,0.0)/len(tmp)
    data = pd.Series(x).reset_index(name='value').rename(columns={'index':'country'})
    data['angle'] = data['value']/data['value'].sum() * 2*pi
    data['color'] = Category20c[len(x)]

    p = figure(plot_height=350, title="Pie Chart", toolbar_location=None,
               tools="hover", tooltips="@country: @value", x_range=(-0.5, 1.0))

    p.wedge(x=0, y=1, radius=0.4,
            start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
            line_color="white", fill_color='color', legend='country', source=data)

    p.axis.axis_label=None
    p.axis.visible=False
    p.grid.grid_line_color = None
    output_file("html/시간별 평균 활동(pie).html")
    show(p)
    
# getPieChart(data)


# ### 가전제품 소비 패턴

# - startDate, endDate, appliance 설정

# In[1785]:


# start='2011-04-18'
# end='2011-04-18'
# appliance='light'
#
#
# # - 특정 가전기기의 비율 계산
#
# # In[1786]:
#
#
# tmp = applianceTimeUsage(appliance,buildings_count,start,end)
# tmp = tmp.loc[start:end]
# tmp = (tmp['active']/total['active']).values
# result = np.nan_to_num(tmp, copy=False).tolist()
#
#
# # - 시각화 색깔 및 시각화 데이터 설정
#
# # In[1787]:
#
#
# fruits = timeData
# years = [appliance]
# colors = ["#718dbf"]
# data = {'fruits' : fruits,
#         appliance   : result}
#
#
# # - 시각화 figure 및 데이터 추가
#
# # In[1788]:
#
#
# p = figure(x_range=fruits, plot_height=750, plot_width=1750, title=appliance+" 소비 패턴",
#            toolbar_location=None, tools="hover", tooltips="$name @fruits: @$name")
#
# p1 = p.vbar_stack(years, x='fruits', width=0.9, color=colors, source=data,
#              legend=[value(x) for x in years])
#
#
# # - 시각화 옵션 설정
#
# # In[1789]:
#
#
# p.y_range.start = 0
# p.x_range.range_padding = 0.5
# p.xgrid.grid_line_color = None
# p.axis.minor_tick_line_color = None
# p.outline_line_color = None
# p.legend.location = "top_left"
# p.legend.orientation = "horizontal"
#
#
# # - 시각화 출력
#
# # In[1790]:
#
#
# output_file("html/"+appliance+" 소비 패턴.html")
# show(p)


# > 전체 코드

# In[1890]:


# start='2011-04-18'
# end='2011-04-18'
# appliance='light'

def applianceUsagePattern(start, end, appliance):
    tmp = applianceTimeUsage(appliance,buildings_count,start,end)
    tmp = tmp.loc[start:end]
    tmp = (tmp['active']/total['active']).values
    result = np.nan_to_num(tmp, copy=False).tolist()
    fruits = timeData
    years = [appliance]
    colors = ["#718dbf"]
    data = {'fruits' : fruits,
            appliance   : result}
    p = figure(x_range=fruits, plot_height=750, plot_width=1750, title=appliance+" 소비 패턴",
               toolbar_location=None, tools="hover", tooltips="$name @fruits: @$name")

    p1 = p.vbar_stack(years, x='fruits', width=0.9, color=colors, source=data,
                 legend=[value(x) for x in years])
    p.y_range.start = 0
    p.x_range.range_padding = 0.5
    p.xgrid.grid_line_color = None
    p.axis.minor_tick_line_color = None
    p.outline_line_color = None
    p.legend.location = "top_left"
    p.legend.orientation = "horizontal"
    output_file("html/"+appliance+" 소비 패턴.html")
    show(p)
    
# applianceUsagePattern(start,end,appliance)


# ___

# ### 지역별 평균 활동

# - startDate, endDate 설정

# In[1791]:


# start='2011-04-18'
# end='2011-04-18'


# - 특정 빌딩의 모든 가전기기 사용량 비율 get

# In[1792]:


def applianceLocationUsage(buildingNum, start, end):
    buildingPower = buildingsPower[buildingNum]['active'].resample(rule='H').mean().to_frame().fillna(0).loc[start:end]
    appliances = buildingAppliances[buildingNum]
    timeData=buildingPower.index.strftime('%Hh').tolist()
    data={'timestamp' : timeData}
    dup={}

    print('applianceLocationUsage buildingPower', buildingPower)
    print('applianceLocationUsage appliances',appliances)
    
    for index in range(len(appliances)):
        tmpDf = next(buildingsElec[buildingNum][3+index].load())['power'].resample(rule='H').mean().fillna(0).loc[start:end]
        tmpDf['active'] = tmpDf['active']/buildingPower['active']
        
        if appliances[index] in dup:
            data[appliances[index]+str(dup[appliances[index]])]= tmpDf['active'].values.tolist()
            dup[appliances[index]]+=1
            pass
        else:
            data[appliances[index]]= tmpDf['active'].values.tolist()
            dup[appliances[index]]=2
            pass
    print('applianceLocationUsage data',data)
    return data


# - 시각화

# In[1826]:


def applianceLocationUsageVisualization(buildingNum,start,end):
    # data
    print(start,end)
    t2 = applianceLocationUsage(buildingNum,start,end)

    print(t2)

    # appliances = buildingAppliances[buildingNum]
        
    
    p1 = figure(x_range=t2['timestamp'], plot_height=700,plot_width=1500, title="지역별 평균 활동",
               toolbar_location=None, tools="hover", tooltips="$name @timestamp: @$name")

    legendKeyTmp = t2.keys()
    legendTmp = []
    for k in legendKeyTmp:
        if k =='timestamp': continue
        legendTmp.append(k)
    if len(legendTmp)>20:
        colors = Category20c[20][:len(legendTmp)]
    else:
        colors = Category20c[len(legendTmp)] 
        
    print(len(legendTmp))
    print(len(colors))
    
    p1.vbar_stack(legendTmp, x='timestamp', width=0.3, color=colors, source=t2,
                 legend=[value(x) for x in legendTmp])

    p1.y_range.start = 0
    p1.y_range.end = 1.0
    p1.x_range.range_padding = 0.5
    p1.xgrid.grid_line_color = None
    p1.axis.minor_tick_line_color = None
    p1.outline_line_color = None
    p1.legend.location = "top_left"
    p1.legend.orientation = "vertical"
    p1.legend.click_policy="hide"

    script_bok, div_bok = components(p1)

    return render_template('update_content.html', div_bok=div_bok, script_bok=script_bok)


    # output_file("html/building"+str(buildingNum)+"_avg_usage.html")
    # show(p1)
    
# applianceLocationUsageVisualization(6)



def applianceStatus(applianceName):
    x_axis_size=10000
    applianceP = figure(x_axis_label='timestamp', y_axis_label ='active', x_axis_type='datetime', title=applianceName)    
        
    colors = list(brewer["Set1"][buildings_count])
    for num, color in zip(range(1,buildings_count+1),colors):
        buildingElec = buildingsElec[num]
        print(num)
        try:
            applianceDf= next(buildingElec[applianceName].load())['power'].head(x_axis_size)
            applianceP.line(applianceDf.index[:], applianceDf['active'], color=color, legend="building"+str(num))
        except Exception:
            print('not exist')
    
#     output_file('html/applianceStatus_'+applianceName+'.html')
#     show(applianceP)

# applianceStatus('fridge')


def getApplianceMeter(building, applianceName):
    return next(building[applianceName].load())
# getApplianceMeter(buildingsElec[1],'microwave')

def getApplianceTotal(building, applianceName):
    return building[applianceName].total_energy()
# getApplianceTotal(buildingsElec[1],'microwave')


# result_origin_df = pd.read_csv('data/result_cutted.csv',index_col='timestamp', parse_dates=True)
# result_predict_df = pd.read_csv('data/result_predict.csv',index_col='timestamp', parse_dates=True)
#
# del result_origin_df['Unnamed: 3']
# del result_predict_df['Unnamed: 3']

# origindf = pd.read_csv('result_cutted.csv', index_col='timestamp', parse_dates=True)
# predictdf = pd.read_csv('result_predict.csv', index_col='timestamp', parse_dates=True)

def getMlResult(origin, predict, mode):
    p = figure(x_axis_label='timestamp', y_axis_label ='power', x_axis_type='datetime', title='compare '+mode)
    p.line(origin.index, origin[mode], color=Category20c[5][0], legend="origin")
    p.line(predict.index, predict[mode], color=Category20c[5][4], legend="predict")
    

