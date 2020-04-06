# Code for Task B
# Name   : Jayanth Harsha G
# RollNo.: 17CS10013

import pandas as pd
import numpy as np

df=pd.read_csv('../data/tfIdf_DataSet.csv')
a=np.array(df)
corpus=a[:,2:].astype('float64')
(no_of_docs, no_of_features) = corpus.shape

#+++++++++++++++++ Initialization of needed data ++++++++++++++++++++++#

#+++++++++++++++++ SIMILARITY MATRIX ++++++++++++++++++++#
similarity_matrix = np.ones((no_of_docs,no_of_docs), dtype='float64')
for i in range(no_of_docs):
    j=i+1
    while j<no_of_docs:
        similarity_matrix[i][j]=np.sum(corpus[i]*corpus[j])
        similarity_matrix[j][i]=similarity_matrix[i][j]
        j=j+1

#++++++++++++++++ Cluster list for representing clusters +++++++++++++#
cluster_list=[]
for i in range(no_of_docs):
    cluster_list.append([i])

#++++++++++++++++ Agglomerative clustering Main Algorithm +++++++++++++#
    
while len(cluster_list)>8:
    size=len(cluster_list)
    max_similarity=-2
    for i in range(size):
        j=i+1
        while j<size:
            single_link_max_similarity=-2
            for k in range(len(cluster_list[i])):
                for t in range(len(cluster_list[j])):
                    if similarity_matrix[k][t]>single_link_max_similarity:
                        single_link_max_similarity=similarity_matrix[k][t]
            if single_link_max_similarity>max_similarity:
                index1=i
                index2=j
            j=j+1
    cluster_list[index1].extend(cluster_list[index2])
    cluster_list.pop(index2)

#++++++++++++++ Writing into text file ++++++++#
    
outF = open("../clusters/agglomerative.txt", "w")      # Opening result file to write clusters 
cluster_list.sort(key=min)                             # Sorting clusters with respect to minimum element 
for i in range(len(cluster_list)):
    cluster_list[i].sort()
    for j in range(len(cluster_list[i])):
        if j!=0:
            outF.write(","+str(cluster_list[i][j]))
        else:
            outF.write(str(cluster_list[i][j]))
    outF.write("\n\n")
outF.close()
