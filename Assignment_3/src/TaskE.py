# Code for Task E
# Name   : Jayanth Harsha G
# RollNo.: 17CS10013

import pandas as pd
import numpy as np
import math

#+++++++++++++ Data Initialization++++++++++++++

df=pd.read_csv('../data/tfIdf_DataSet.csv')
a=np.array(df)
true_label=(a[:,1])
N=len(list(true_label))
(names,counts)=np.unique(true_label, return_counts=True)
indices=np.array(range(len(names)))
name_to_index=dict(zip(names, indices))
class_freq = dict(zip(indices, counts))
HC=0
for j in range(8):
    if class_freq[j]!=0:
        HC=HC-(class_freq[j]/N)*math.log(class_freq[j]/N,2)
        
#++++++++++ function for NMI evaluation+++++++++#
        
def NMIevaluation(fileName):
    #fileName='kmeans.txt'
    inF=open(fileName)
    cluster_list=[]
    flag=0
    for line in inF:
        if flag==0:
            cluster_list.append(list(map(int, line.split(','))))
            flag=1
        else:
            flag=0
    inF.close()
    # +++++++++++++++++ initialization of w(k,j)+++++++++++++++
    w=np.zeros((8,8), dtype='float64')
    for k in range(8):
        for i in range(len(cluster_list[k])):
            j=name_to_index[true_label[cluster_list[k][i]]]
            w[k][j]=w[k][j]+1
    MI=0
    HW=0
    for k in range(8):
        if len(cluster_list[k])!=0:
            HW=HW-((len(cluster_list[k])/N)*math.log(len(cluster_list[k])/N,2))
        for j in range(8):
            c=len(cluster_list[k])*class_freq[j]
            b=w[k][j]
            if b==0:
                continue
            else:
                MI=MI+ (b/N)*math.log((N*b)/c,2)
                
    NMI=(2*MI)/(HW+HC)
    return NMI

kmeans_nmi=NMIevaluation('../clusters/kmeans.txt')
agglomerative_nmi=NMIevaluation('../clusters/agglomerative.txt')
kmeans_reduced_nmi=NMIevaluation('../clusters/kmeans_reduced.txt')
agglomerative_reduced_nmi=NMIevaluation('../clusters/agglomerative_reduced.txt')

#print(kmeans_nmi)
#print(agglomerative_nmi)
#print(kmeans_reduced_nmi)
#print(agglomerative_reduced_nmi)

outF = open("../data/ResultFile_for_TaskE.txt", "w")      # opening result file to write results 
outF.write("NMI score for Agglomerative clustering before applying PCA : ")
outF.write(str(agglomerative_nmi))
outF.write("\nNMI score for Agglomerative clustering after applying PCA   :  ")
outF.write(str(agglomerative_reduced_nmi))
outF.write("\nNMI score for K means clustering before applying PCA : ")
outF.write(str(kmeans_nmi))
outF.write("\nNMI score for Agglomerative clustering after applying PCA : ")
outF.write(str(kmeans_reduced_nmi))
outF.close()