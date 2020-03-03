# Code for Task 2
# Name   : Jayanth Harsha G
# RollNo.: 17CS10013

from sklearn import preprocessing
import numpy as np
import csv
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


outF = open("../data/ResultFile_for_Task2.txt", "w")      # opening result file to write results 
outF.write("Task 2 :\n\n")
coeff=[]                                                    # coeff and temp are null valued initiated lists
temp=[]

def CustomLogisticRegression(TrainSet, Trainlabel, Trainsize, TestSet, Testlabel, Testsize):
    coeff.clear()
    temp.clear()
    for i in range(12):        
        coeff.append(0.5)
        temp.append(0)
    ones_array=np.ones((Trainsize,1),dtype=float)
    xj=np.append(ones_array, TrainSet, axis=1)    # xj contains attribute values of all training examples including first column of 1's
    
    l1=np.sum(coeff*xj, axis = 1)
    l1 = (ones_array+np.exp(-l1).reshape(Trainsize,1))**(-1)        # hypothesis function calculation
    prev_cost_func=((-(Trainlabel*np.log(l1)+(ones_array-Trainlabel)*np.log(ones_array-l1))).sum())/Trainsize
    for itr in range(100000):                           # iterating 1,00,000 times for accuracy
        l1=l1-Trainlabel
        for k in range(12):
            l4=l1*(xj[:,k].reshape(Trainsize,1))             # computing sigma i=0 to 1599 h(xi)-y(i) multiplied by xi,k for kth coefficient
            temp[k]=coeff[k]-((0.05/Trainsize)*(l4.sum())) # Applying gradient descent and storing updated coefficients in temporary list 
        for k in range(12):
            coeff[k]=temp[k]                            # Updating coefficients for n degree polynomial
        l1=np.sum((coeff*xj), axis = 1)
        l1 = (ones_array+np.exp(-l1).reshape(Trainsize,1))**(-1)        # Hypothesis function calculation
        curr_cost_func=((-(Trainlabel*np.log(l1)+(ones_array-Trainlabel)*np.log(ones_array-l1))).sum())/Trainsize
        #print("Cost function is "+ str(cost_func))
        if abs(prev_cost_func-curr_cost_func) < 10e-8:  # checking for convergence test ie < 10^-8
           break
        else:
           prev_cost_func=curr_cost_func   
    ones_array_Test=np.ones((Testsize,1),dtype=float)
    xj_Test=np.append(ones_array_Test, TestSet, axis=1)    # xj contains attribute values of all test examples including first column of 1's
    l1=np.sum(coeff*xj_Test, axis = 1)
    l1 = (ones_array_Test+np.exp(-l1).reshape(Testsize,1))**(-1)     # hypothesis function calculation for all test examples
    
    TruePositive=0
    TrueNegative=0
    FalsePositive=0
    FalseNegative=0
    for i in range(Testsize):
        if l1[i] >= 0.5 and Testlabel[i] == 1 : 
            TruePositive=TruePositive+1
        elif l1[i] < 0.5 and Testlabel[i] == 0:
            TrueNegative=TrueNegative+1
        elif l1[i] >= 0.5 and Testlabel[i]==0 :
            FalsePositive=FalsePositive+1
        else:
            FalseNegative=FalseNegative+1
    Accuracy = (TruePositive+TrueNegative)/Testsize
    Precision = TruePositive/(TruePositive+FalsePositive)
    Recall = TruePositive/(TruePositive+FalseNegative)
    return (Accuracy, Precision, Recall)
   

def scikitlearnLogisticRegression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(penalty='none', dual=False, max_iter=500, tol=0.00000001, solver='saga' )
    model=model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    return score, precision, recall
    
def convert_list_to_float(test_list):
    for i in range(0, len(test_list)): 
        test_list[i] = float(test_list[i]) 
    return test_list

l1=[]                                               # Empty lists for processing data
l2=[]
with open('../data/winequality-red.csv','r') as csvfile:
    readcsv = csv.reader(csvfile, delimiter=';')
    
    next(readcsv)
    for line in readcsv:
        if int(line[11])<=6:
            l2.append(0)
        else:
            l2.append(1)
        l1.append(convert_list_to_float(line[0:11]))
        
minmaxscaler = preprocessing.MinMaxScaler(feature_range=(0,1))

DataSet_A = minmaxscaler.fit_transform(np.array(l1))
Label_A=np.array(l2)

TrainSetPart1 = DataSet_A[0:533, :]
TrainSetPart2 = DataSet_A[533:1066, :]
TrainSetPart3 = DataSet_A[1066:1599, :]

Trainlabel=Label_A.reshape(1599,1)

LabelSetPart1 = Trainlabel[0:533, :]
LabelSetPart2 = Trainlabel[533:1066, :]
LabelSetPart3 = Trainlabel[1066:1599, :]

TrainSet = DataSet_A
CustomAccuracy, CustomPrecision, CustomRecall = CustomLogisticRegression(TrainSet, Trainlabel, 1599, TrainSet, Trainlabel, 1599)
outF.write("coefficients of logistic regression function using self written logistic regression classifier trained on entire data are:\n\n")
for i in range(12):
    outF.write(str(coeff[i]))
    outF.write("\n")
y_train=Label_A.reshape(1599,)
scikitAccuracy, scikitPrecision, scikitRecall = scikitlearnLogisticRegression(DataSet_A, y_train, DataSet_A, y_train)

# 1st fold
TrainSet = np.append(TrainSetPart1, TrainSetPart2, axis=0)
Trainlabel = np.append(LabelSetPart1, LabelSetPart2, axis=0)
TestSet = TrainSetPart3
Testlabel = LabelSetPart3
CustomAccuracy1, CustomPrecision1, CustomRecall1 = CustomLogisticRegression(TrainSet, Trainlabel, 1066, TestSet, Testlabel, 533)
scikitAccuracy1, scikitPrecision1, scikitRecall1 = scikitlearnLogisticRegression(TrainSet, Trainlabel.reshape(1066,), TestSet, Testlabel.reshape(533,))

# 2nd fold
TrainSet = np.append(TrainSetPart1, TrainSetPart3, axis=0)
Trainlabel = np.append(LabelSetPart1, LabelSetPart3, axis=0)
TestSet = TrainSetPart2
Testlabel = LabelSetPart2
CustomAccuracy2, CustomPrecision2, CustomRecall2 = CustomLogisticRegression(TrainSet, Trainlabel, 1066, TestSet, Testlabel, 533)
scikitAccuracy2, scikitPrecision2, scikitRecall2 = scikitlearnLogisticRegression(TrainSet, Trainlabel.reshape(1066,), TestSet, Testlabel.reshape(533,))

# 3rd fold
TrainSet = np.append(TrainSetPart2, TrainSetPart3, axis=0)
Trainlabel = np.append(LabelSetPart2, LabelSetPart3, axis=0)
TestSet = TrainSetPart1
Testlabel = LabelSetPart1
CustomAccuracy3, CustomPrecision3, CustomRecall3 = CustomLogisticRegression(TrainSet, Trainlabel, 1066, TestSet, Testlabel, 533)
scikitAccuracy3, scikitPrecision3, scikitRecall3 = scikitlearnLogisticRegression(TrainSet, Trainlabel.reshape(1066,), TestSet, Testlabel.reshape(533,))

Custom_3_fold_Average_Accuracy = 100*(CustomAccuracy1 + CustomAccuracy2 + CustomAccuracy3)/3
Custom_3_fold_Average_Precision = 100*(CustomPrecision1 + CustomPrecision2 + CustomPrecision3)/3
Custom_3_fold_Average_Recall  = 100*(CustomRecall1 + CustomRecall2 + CustomRecall3)/3

scikit_3_fold_Average_Accuracy = 100*(scikitAccuracy1 + scikitAccuracy2 + scikitAccuracy3)/3
scikit_3_fold_Average_Precision = 100*(scikitPrecision1 + scikitPrecision2 + scikitPrecision3)/3
scikit_3_fold_Average_Recall = 100*(scikitRecall1 + scikitRecall2 + scikitRecall3)/3

print("\n\nFor Self Written Logistic Regression Classifier:\n")
print("Mean Accuracy after implementing 3 fold cross validation is :"+str(Custom_3_fold_Average_Accuracy)+"%")
print("Mean Precision after implementing 3 fold cross validation is:"+str(Custom_3_fold_Average_Precision)+"%")
print("Mean Recall after implementing 3 fold cross validation is   :"+str(Custom_3_fold_Average_Recall)+"%")

print("\n\nFor scikit learn Logistic Regression Classifier:\n")
print("Mean Accuracy after implementing 3 fold cross validation is :"+str(scikit_3_fold_Average_Accuracy)+"%")
print("Mean Precision after implementing 3 fold cross validation is:"+str(scikit_3_fold_Average_Precision)+"%")
print("Mean Recall after implementing 3 fold cross validation is   :"+str(scikit_3_fold_Average_Recall)+"%")

outF.write("\n\nFor Self Written Logistic Regression Classifier:\n")
outF.write("\nMean Accuracy after implementing 3 fold cross validation is :"+str(Custom_3_fold_Average_Accuracy)+"%")
outF.write("\nMean Precision after implementing 3 fold cross validation is:"+str(Custom_3_fold_Average_Precision)+"%")
outF.write("\nMean Recall after implementing 3 fold cross validation is   :"+str(Custom_3_fold_Average_Recall)+"%")

outF.write("\n\nFor scikit learn Logistic Regression Classifier:\n")
outF.write("\nMean Accuracy after implementing 3 fold cross validation is :"+str(scikit_3_fold_Average_Accuracy)+"%")
outF.write("\nMean Precision after implementing 3 fold cross validation is:"+str(scikit_3_fold_Average_Precision)+"%")
outF.write("\nMean Recall after implementing 3 fold cross validation is   :"+str(scikit_3_fold_Average_Recall)+"%")
outF.close()                                # Closing output file


    
        
    



