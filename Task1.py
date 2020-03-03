# Code for Task 1
# Name   : Jayanth Harsha G
# RollNo.: 17CS10013

import pandas as pd
import csv
from sklearn import preprocessing
import numpy as np
import os

df=pd.read_csv('../data/winequality-red.csv',delimiter=';')
def convert_list_to_float(test_list):
    for i in range(0, len(test_list)): 
        test_list[i] = float(test_list[i]) 
    return test_list


l1=[]
l2=[]
l3=[]
with open("../data/winequality-red.csv",'r') as csvfile:
    readcsv = csv.reader(csvfile, delimiter=';')
    
    next(readcsv)
    for line in readcsv:
        if int(line[11])<=6:
            l2.append([0])
        else:
            l2.append([1])
        if int(line[11])<5:
            l3.append([0])
        elif int(line[11])==5 or int(line[11])==6:
            l3.append([1])
        else :
            l3.append([2])
        l1.append(convert_list_to_float(line[0:11]))
        
minmaxscaler = preprocessing.MinMaxScaler(feature_range=(0,1))
standardscaler = preprocessing.StandardScaler()

DataSet_A = minmaxscaler.fit_transform(np.array(l1))
DataSet_B = standardscaler.fit_transform(np.array(l1))


Label_A=np.array(l2)
Label_B=np.array(l3)

for i in range(11):
    DataSet_B[:,i] = pd.cut(DataSet_B[:,i], bins=4, labels=False)


df1 = pd.DataFrame(np.append(DataSet_A, Label_A, axis=1))
df2 = pd.DataFrame(np.append(DataSet_B, Label_B, axis=1))

df1.to_csv('../data/temp1.csv', header=False)
df2.to_csv('../data/temp2.csv', header=False)

df3=pd.read_csv('../data/temp1.csv', names=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"])
df4=pd.read_csv('../data/temp2.csv', names=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"])

os.remove("../data/temp1.csv")      # removing temporary csv s
os.remove("../data/temp2.csv")

df3.to_csv('../data/DataSet_A.csv')
df4.to_csv('../data/DataSet_B.csv')
























