# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:38:55 2019

@author: USER
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

df = pd.read_csv("output_XYBegin_0614_G.csv", header=None)
size=df.shape #(row,colom)
row_number=len(df.index)-1 #number of row
#print(row_number)

begin=df.iloc[0,1] #取出row data,型態為str
org_xy=list(eval(begin)) #轉str為list **important step
#print(org_xy)
num=len(org_xy)-1
#print(num)

delete_list=[]
for a in range(0,num):
    if a!=num:
        if abs(org_xy[a][0]-org_xy[a+1][0])<=1 and abs(org_xy[a][1]-org_xy[a+1][1])<=2:
            #print(org_xy[a])
            delete_element=org_xy[a]
            #print(a)
            delete_list.append(a)           
#print(delete_list)
for index in sorted(delete_list, reverse=True):
    #print(org_xy[index])
    del org_xy[index]


#print(org_xy)
print(len(org_xy))
new_num=len(org_xy)

#for i in (0,row_number):
#    s=df.iloc[i,1]
#    print(s)

####繪圖區#####################################################################
video_x=[]
video_y=[]
for n in range(0,new_num):
    video_x.append(org_xy[n][0])
    video_y.append(org_xy[n][1])

#print(video_x)
#print(video_y)
#print(len(video_x))

plt.scatter(video_x, video_y) 
plt.show()

real_x=[]
real_y=[]
for n in range(0,6):
    for m in range(0,6):
        real_x.append(n)
        real_y.append(m)
        
#print(real_x)
#print(real_y)
plt.scatter(real_x,real_y,c = 'r',marker = 'o')
###############################################################################
#####查詢最佳回歸階數##############################################################
realxy_list=[list(x) for x in zip(real_x, real_y)]
realxy_array=np.array(realxy_list)
videoxy_array=np.array(org_xy)

print(len(realxy_array))
print(len(videoxy_array))
print(realxy_list)
print(org_xy)

x_train, x_test, y_train, y_test = train_test_split(videoxy_array, realxy_array, test_size=0.3)
rmses = []
degrees = np.arange(1, 10)
min_rmse, min_deg,score = 1e10, 0 ,0
 
for deg in degrees:
	# 生成多项式特征集(如根据degree=3 ,生成 [[x,x**2,x**3]] )
	poly = PolynomialFeatures(degree=deg, include_bias=False)
	x_train_poly = poly.fit_transform(x_train)
 
	# 多项式拟合
	poly_reg = LinearRegression()
	poly_reg.fit(x_train_poly, y_train)
	#print(poly_reg.coef_,poly_reg.intercept_) #系数及常数
	
	# 测试集比较
	x_test_poly = poly.fit_transform(x_test)
	y_test_pred = poly_reg.predict(x_test_poly)
	
	#mean_squared_error(y_true, y_pred) #均方误差回归损失,越小越好。
	poly_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
	rmses.append(poly_rmse)
	# r2 范围[0，1]，R2越接近1拟合越好。
	r2score = r2_score(y_test, y_test_pred)
	
	# degree交叉验证
	if min_rmse > poly_rmse:
		min_rmse = poly_rmse
		min_deg = deg
		score = r2score
	print('degree = %s, RMSE = %.2f ,r2_score = %.2f' % (deg, poly_rmse,r2score))
		
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(degrees, rmses)
ax.set_yscale('log')
ax.set_xlabel('Degree')
ax.set_ylabel('RMSE')
ax.set_title('Best degree = %s, RMSE = %.2f, r2_score = %.2f' %(min_deg, min_rmse,score))  
plt.show()

######各階數畫圖################################################################

for degree in [1,2,3,4,5]:
    print("degree:",degree)
    linreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    
    predicted = cross_val_predict(linreg, videoxy_array, realxy_array, cv=10)
    
    iris_y_list=realxy_array.tolist()
    yt=[]
    yt1=[]
    for i in range(0,len(iris_y_list)):
        g=iris_y_list[i][0]
        yt.append(g)
        h=iris_y_list[i][1]
        yt1.append(h)
    #print(yt)
    #print(yt1)
    
    c=predicted.tolist()
    d=[]
    d1=[]
    for i in range(0,len(c)):
        e=c[i][0]
        d.append(e)
        f=c[i][1]
        d1.append(f)
    #print(d)
    #print(d1)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    ax1.scatter(d, d1, s=10, c='b', marker="s", label='predict data')
    ax1.scatter(yt,yt1, s=10, c='r', marker="o", label='orginal data')
    plt.legend(loc='upper left');
    plt.show()

##############################################################################
