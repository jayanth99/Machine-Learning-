# Code for Task C
# Name   : Jayanth Harsha G
# RollNo.: 17CS10013

import pandas as pd
import numpy as np
import random

df=pd.read_csv('../data/tfIdf_DataSet.csv')
a=np.array(df)
corpus=a[:,2:].astype('float64')
(no_of_docs, no_of_features) = corpus.shape

#+++++++++++++++++ Initialization of needed data ++++++++++++++++#

#+++++++++++++++++ Random Initialization of centroid points +++++#

prev_centroid = np.ones((8,no_of_features))
for i in range(8):
    index=random.randrange(no_of_docs)
    prev_centroid[i]=prev_centroid[i]*corpus[index]

curr_centroid=prev_centroid

#+++++++++ Cluster index list to represent clusters ++++++++++++#
cluster_list=[]

for itr in range(100):
    cluster_list.clear()
    for i in range(8):
        cluster_list.append([])
    for i in range(no_of_docs):
        max_similarity=-2
        for j in range(8):
            similarity=np.sum(corpus[i]*curr_centroid[j])
            if similarity>max_similarity:
                max_similarity=similarity
                closest_index=j
        cluster_list[closest_index].append(i)
    prev_centroid=curr_centroid
    curr_centroid = np.zeros((8,no_of_features))
    for i in range(8):
        size=0
        for j in range(len(cluster_list[i])):
            size=size+1
            index=cluster_list[i][j]
            curr_centroid[i]=curr_centroid[i]+corpus[index]
        curr_centroid[i]=curr_centroid[i]/size
    if (curr_centroid==prev_centroid).all():
        break

#++++++++++++++ Writing into text file ++++++++#
       
outF = open("../clusters/kmeans.txt", "w")      # Opening result file to write clusters 
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
           