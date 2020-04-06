# Code for Task A
# Name   : Jayanth Harsha G
# RollNo.: 17CS10013

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer

df=pd.read_csv('../data/AllBooks_baseline_DTM_Labelled.csv')
a=np.array(df)
(r,c)=a.shape
for i in range(r):
    a[i][0]=a[i][0].split('_')[0]
a=np.delete(a,13,0)
cv_tfidf=TfidfTransformer(norm='l2')
a[:,1:]=cv_tfidf.fit_transform(a[:,1:].astype('float64')).toarray()
df=pd.DataFrame(a)
df.to_csv('../data/tfIdf_DataSet.csv')