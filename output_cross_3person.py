# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:38:55 2019

@author: USER
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
###受測者資料取出###############################################################
df = pd.read_csv("output_test_cross0501_On1.csv", header=None)
size = df.shape  #(row,colom)
row_number = len(df.index)  #number of row
print(row_number)

#begin=df.iloc[0,1] #取出row data,型態為str
for i in range(0, row_number):
    trackid_xy = df.iloc[i, 1]
    trackid_name = df.iloc[i, 0]
    #print(trackid_xy)
    print(trackid_name)
    globals()[str(trackid_name)] = list(
        eval(trackid_xy))  #建立各trackerid的list  #轉str為list **important step
    num = len(globals()[str(trackid_name)]) - 1
    delete_list = []
    for a in range(0, num):
        if a != num:
            if abs(globals()[str(trackid_name)][a][0] -
                   globals()[str(trackid_name)][a + 1][0]) <= 2 and abs(
                       globals()[str(trackid_name)][a][1] -
                       globals()[str(trackid_name)][a + 1][1]) <= 2:
                #print(globals()[str(trackid_name)][a])
                delete_element = globals()[str(trackid_name)][a]
                #print(a)
                delete_list.append(a)
    for index in sorted(delete_list, reverse=True):
        #print(globals()[str(trackid_name)][index])
        del globals()[str(trackid_name)][index]
    print(globals()[str(trackid_name)])
##############################################################################
####取出原始座標################################################################
#cf = pd.read_csv("video transform data\experiment data\output_XYBegin.csv", header=None)
#cf = pd.read_csv("video transform data\experiment data\output_XYBegin_0416.csv", header=None)
#cf = pd.read_csv("video transform data\experiment data\output_XYBegin_0427.csv", header=None)
cf = pd.read_csv("output_XYBegin_0501.csv", header=None)

size = cf.shape  #(row,colom)
row_number = len(df.index)  #number of row
print(row_number)

begin = cf.iloc[0, 1]  #取出row data,型態為str
org_xy = list(eval(begin))  #轉str為list **important step
#print(org_xy)
number = len(org_xy) - 1
#print(num)

delete_element_list = []
for a in range(0, number):
    if a != number:
        if abs(org_xy[a][0] - org_xy[a + 1][0]) <= 3 and abs(
                org_xy[a][1] - org_xy[a + 1][1]) <= 3:
            #print(org_xy[a])
            delete_element = org_xy[a]
            #print(a)
            delete_element_list.append(a)
#print(delete_list)
for index in sorted(delete_element_list, reverse=True):
    #print(org_xy[index])
    del org_xy[index]

#print(org_xy)
print(len(org_xy))
new_num = len(org_xy)
###原始座標繪製#################################################################
video_x = []
video_y = []
for n in range(0, new_num):
    video_x.append(org_xy[n][0])
    video_y.append(org_xy[n][1])

#print(video_x)
#print(video_y)
#print(len(video_x))

plt.scatter(video_x, video_y)
###############################################################################
####繪圖區#####################################################################

for i in range(0, row_number):
    video_x = []
    video_y = []
    trackid_name = df.iloc[i, 0]
    new_num = len(globals()[str(trackid_name)])
    for n in range(0, new_num):
        video_x.append(globals()[str(trackid_name)][n][0])
        video_y.append(globals()[str(trackid_name)][n][1])
    plt.plot(video_x, video_y, label=str(trackid_name))
    #plt.legend(loc='upper left')
    #plt.show()

#print(video_x)
#print(video_y)
#print(len(video_x))

#plt.scatter(video_x, video_y)
plt.show()

real_x = []
real_y = []
for n in range(0, 5):
    for m in range(0, 5):
        real_x.append(n)
        real_y.append(m)

#print(real_x)
#print(real_y)
#plt.scatter(real_x,real_y,c = 'r',marker = 'o')
#plt.show()
###############################################################################
###model訓練與預測##############################################################
realxy_list = [list(x) for x in zip(real_x, real_y)]
#print(realxy_list)
#print(org_xy)
realxy_array = np.array(realxy_list)
videoxy_array = np.array(org_xy)

bestdegree = 2  #最佳回歸階數

model = make_pipeline(PolynomialFeatures(bestdegree), LinearRegression())
model.fit(videoxy_array, realxy_array)
for i in range(0, row_number):
    trackid_name = df.iloc[i, 0]
    track_line = np.array(globals()[str(trackid_name)])
    test_predict = model.predict(track_line)
    globals()["predict_list" + str(i)] = test_predict.tolist()
    print("predict_list" + str(i) + ":", globals()["predict_list" + str(i)])
#print("test predict:",test_predict)
##model訓練結果繪製#############################################################
#predict_list=test_predict.tolist()
#print("predict list:",predict_list)
for i in range(0, row_number):
    predict_number = len(globals()["predict_list" + str(i)])
    #for i in (0,row_number):
    predict_x = []
    predict_y = []
    trackid_name = df.iloc[i, 0]
    new_num = len(globals()[str(trackid_name)])
    for n in range(0, predict_number):
        predict_x.append(globals()["predict_list" + str(i)][n][0])
        predict_y.append(globals()["predict_list" + str(i)][n][1])
    plt.plot(predict_x, predict_y, label=str(trackid_name))
    #plt.legend(loc='upper left')

plt.scatter(real_x, real_y, c='r', marker='o')
plt.show()
