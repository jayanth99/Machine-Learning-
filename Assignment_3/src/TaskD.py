# Code for Task D
# Name   : Jayanth Harsha G
# RollNo.: 17CS10013

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import numpy as np
import random

df=pd.read_csv('../data/tfIdf_DataSet.csv')
a=np.array(df)
corpus=a[:,2:].astype('float64')
pca = PCA(n_components=100)
X_reduced=pca.fit_transform(corpus)
cv_tfidf=TfidfTransformer(norm='l2')
corpus=cv_tfidf.fit_transform(X_reduced).toarray()
(no_of_docs, no_of_features) = corpus.shape


#++++++++++++++ AGGLOMERATIVE CLUSTERING +++++++++++++++++++++++++++++#

# Initialization of needed data

#++++++++++++++++++++++++++++++ SIMILARITY MATRIX ++++++++++++++++++++++++++++#
similarity_matrix = np.ones((no_of_docs,no_of_docs), dtype='float64')
for i in range(no_of_docs):
    j=i+1
    while j<no_of_docs:
        similarity_matrix[i][j]=np.sum(corpus[i]*corpus[j])
        similarity_matrix[j][i]=similarity_matrix[i][j]
        j=j+1

#+++++++++++ Cluster list for representing clusters +++++++++++++++++++#
cluster_list=[]
for i in range(no_of_docs):
    cluster_list.append([i])

#++++++++++++ Agglomerative clustering Main ALGORITHM +++++++++++++++++#
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

outF = open("../clusters/agglomerative_reduced.txt", "w")      # Opening result file to write clusters     
cluster_list.sort(key=min)                                     # Sorting clusters with respect to minimum element
for i in range(len(cluster_list)):
    cluster_list[i].sort()
    for j in range(len(cluster_list[i])):
        if j!=0:
            outF.write(","+str(cluster_list[i][j]))
        else:
            outF.write(str(cluster_list[i][j]))
    outF.write("\n\n")
outF.close()
cluster_list.clear()
#+++++++++++++++++END OF AGGLOMERATIVE CLUSTERING ++++++++++++++++#


#++++++++++++++++ K MEANS CLUSTERING +++++++++++++++++++++++++++++#

#++++++++++++++ Initialization of needed data +++++++++++#

#+++++++++++++++++++ Random Initialization of centroid points +++++++++++++++++#

prev_centroid = np.ones((8,no_of_features))
for i in range(8):
    index=random.randrange(no_of_docs)
    prev_centroid[i]=prev_centroid[i]*corpus[index]

curr_centroid=prev_centroid

#+++++++++++++++++++ Cluster index list +++++++++++++++++++++#
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

#+++++++++++ writing into text file +++++++++++++++++#
outF = open("../clusters/kmeans_reduced.txt", "w")      # opening result file to write results 
cluster_list.sort(key=min)
for i in range(len(cluster_list)):
    cluster_list[i].sort()
    for j in range(len(cluster_list[i])):
        if j!=0:
            outF.write(","+str(cluster_list[i][j]))
        else:
            outF.write(str(cluster_list[i][j]))
    outF.write("\n\n")
outF.close()
#+++++++++++++ End of kmeans clustering ++++++++++++#
           