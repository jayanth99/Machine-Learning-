# Code for Task 3
# Name   : Jayanth Harsha G
# RollNo.: 17CS10013

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sb
#from sklearn.model_selection import train_test_split
from sklearn import tree
#%matplotlib inline
#import random
from sklearn import metrics
#from pprint import pprint

## checking if that node contains all examples of same class or not
def check_purity(data):
    
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False

## different types of labels and label with maximum frequency    
def classify_data(data):
    
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    
    if len(unique_classes)==0:
        classification="NULL"
        return classification
    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
    
    return classification

## Splitting the data on given column (ie given attribute)
def split_data(data, split_column):
    
    split_column_values = data[:, split_column]

    data_equal_zero = data[split_column_values == 0]
    data_equal_one  = data[split_column_values == 1]
    data_equal_two  = data[split_column_values == 2]
    data_equal_three  = data[split_column_values == 3]
    
    
    return data_equal_zero, data_equal_one, data_equal_two, data_equal_three


## Calculating entropy
def calculate_entropy(data):
    
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
     
    return entropy

## Calculating Overall entropy 
def calculate_overall_entropy(data_equal_zero, data_equal_one, data_equal_two, data_equal_three):
    
    n = len(data_equal_zero) + len(data_equal_one) + len(data_equal_two) + len(data_equal_three)
    p_data_equal_zero = len(data_equal_zero)/n
    p_data_equal_one = len(data_equal_one)/n
    p_data_equal_two = len(data_equal_two)/n
    p_data_equal_three = len(data_equal_three)/n

    overall_entropy =  (p_data_equal_zero * calculate_entropy(data_equal_zero) 
                      + p_data_equal_one * calculate_entropy(data_equal_one)
                      + p_data_equal_two * calculate_entropy(data_equal_two)
                      + p_data_equal_three * calculate_entropy(data_equal_three))
    
    return overall_entropy

## Determining best split
def determine_best_split(data, potential_splits):
    
    overall_entropy = 999999
    for column_index in potential_splits:
        data_equal_zero, data_equal_one, data_equal_two, data_equal_three = split_data(data, split_column=column_index)
        n = len(data_equal_zero) + len(data_equal_one) + len(data_equal_two) + len(data_equal_three)
        if(n==0):
            continue
        current_overall_entropy = calculate_overall_entropy(data_equal_zero, data_equal_one, data_equal_two, data_equal_three)

        if current_overall_entropy <= overall_entropy:
            overall_entropy = current_overall_entropy
            best_split_column = column_index
    
    return best_split_column


## Decision Tree Algorithm
def decision_tree_algorithm(df1, counter=0, min_samples=10):
    
    # data preparations
    if counter == 0:
        global COLUMN_HEADERS
        COLUMN_HEADERS = df1.columns
        data = df1.values
    else:
        data = df1          
    
    # base cases
    if (check_purity(data)) or (len(data) < min_samples) :
        classification = classify_data(data)
        
        return classification
    
    # recursive part
    else:    
        counter += 1
        # helper functions 
        potential_splits = [0,1,2,3,4,5,6,7,8,9,10]
        split_column = determine_best_split(data, potential_splits)
        data_equal_zero, data_equal_one, data_equal_two, data_equal_three = split_data(data, split_column)
        
        # instantiate sub-tree
        feature_name = COLUMN_HEADERS[split_column]
        question1 = "{} == {}".format(feature_name, 0)
        question2 = "{} == {}".format(feature_name, 1)
        question3 = "{} == {}".format(feature_name, 2)
        question4 = "{} == {}".format(feature_name, 3)
        sub_tree={question1 : [], question2 : [], question3 : [], question4 : []}
        null_count=0
        if len(data_equal_zero) == 0:
            null_count+=1
        if len(data_equal_one) == 0:
            null_count+=1
        if len(data_equal_two) == 0:
            null_count+=1
        if len(data_equal_three) == 0:
            null_count+=1
        if null_count>=3:
            del sub_tree[question1]
            del sub_tree[question2]
            del sub_tree[question3]
            del sub_tree[question4]
            sub_tree = classify_data(data)
            return  sub_tree
        
        if len(data_equal_zero) != 0:
            if len(data_equal_zero) < min_samples:
                classification = classify_data(data_equal_zero)
                answer1 = classification
                sub_tree[question1].append(answer1)
            else :
                answer1 = decision_tree_algorithm(data_equal_zero, counter)
                sub_tree[question1].append(answer1)
        else:
            answer1="NULL"
            sub_tree[question1].append("NULL")
        if len(data_equal_one) != 0:
            if len(data_equal_one) < min_samples:
                classification = classify_data(data_equal_one)
                answer2 = classification
                sub_tree[question2].append(answer2)
            else :
                answer2 = decision_tree_algorithm(data_equal_one, counter)
                sub_tree[question2].append(answer2)
        else:
            answer2="NULL"
            sub_tree[question2].append("NULL")
        if len(data_equal_two) != 0:
            if len(data_equal_two) < min_samples:
                classification = classify_data(data_equal_two)
                answer3 = classification
                sub_tree[question3].append(answer3)
            else :
                answer3 = decision_tree_algorithm(data_equal_two, counter)
                sub_tree[question3].append(answer3)
        else:
            answer3="NULL"
            sub_tree[question3].append("NULL")
        if len(data_equal_three) != 0:
            if len(data_equal_three) < min_samples:
                classification = classify_data(data_equal_three)
                answer4 = classification
                sub_tree[question4].append(answer4)
            else :
                answer4 = decision_tree_algorithm(data_equal_three, counter)
                sub_tree[question4].append(answer4)
        else:
            answer4="NULL"
            sub_tree[question4].append("NULL")
        return sub_tree
    

## Classification     
def classify_example(example, tree):
    question = list(tree.keys())[0]
    if len(question.split()) == 3:
        feature_name = (question.split())[0]
    elif len(question.split()) == 4:
        feature_name = (question.split())[0] + " " + (question.split())[1]
    else:
        feature_name = (question.split())[0] + " " + (question.split())[1] + " " + (question.split())[2]
    
    # asking question    
    if example[feature_name] == 0:
        question1 = list(tree.keys())[0]
        answer = tree[question1][0]
    elif example[feature_name] == 1:
        question2 = list(tree.keys())[1]
        answer = tree[question2][0]
    elif example[feature_name] == 2:
        question3 = list(tree.keys())[2]
        answer = tree[question3][0]
    else :
        question4 = list(tree.keys())[3]
        answer = tree[question4][0]

    # base case
    if not isinstance(answer, dict):
        return answer
    
    # recursive part
    else:
        residual_tree = answer
        return classify_example(example, residual_tree)

# calculate accuracy
def calculate_accuracy(testdf, tree):

    testdf["classification"] = testdf.apply(classify_example, axis=1, args=(tree,))
    testdf["classification_correct"] = (testdf["classification"] == testdf["label"])
    accuracy = testdf["classification_correct"].mean()
    y_actual=np.array(testdf["label"])
    y_pred=np.array(testdf["classification"])
    
    zeros = np.zeros(y_actual.shape)
    ones = np.ones(y_actual.shape)
    twos = 2*np.ones(y_actual.shape)
    
    TP_for_0 = np.sum(np.logical_and(y_pred == zeros, y_actual == zeros))
    FP_for_0 = np.sum(np.logical_and(y_pred == zeros, np.logical_not(y_actual == zeros)))
    FN_for_0 = np.sum(np.logical_and(y_actual == zeros, np.logical_not(y_pred == zeros)))
    TP_for_1 = np.sum(np.logical_and(y_pred == ones, y_actual == ones))
    FP_for_1 = np.sum(np.logical_and(y_pred == ones, np.logical_not(y_actual == ones)))
    FN_for_1 = np.sum(np.logical_and(y_actual == ones, np.logical_not(y_pred == ones)))
    TP_for_2 = np.sum(np.logical_and(y_pred == twos, y_actual == twos))
    FP_for_2 = np.sum(np.logical_and(y_pred == twos, np.logical_not(y_actual == twos)))
    FN_for_2 = np.sum(np.logical_and(y_actual == twos, np.logical_not(y_pred == twos)))
    
    precision_for_0=0
    precision_for_1=0
    precision_for_2=0
    
    if TP_for_0 + FP_for_0 != 0:
        precision_for_0 = TP_for_0/(TP_for_0 + FP_for_0)
    if TP_for_1 + FP_for_1 != 0:
        precision_for_1 = TP_for_1/(TP_for_1 + FP_for_1)
    if TP_for_2 + FP_for_2 != 0:
        precision_for_2 = TP_for_2/(TP_for_2 + FP_for_2)
        
    recall_for_0=0
    recall_for_1=0
    recall_for_2=0
    
    if TP_for_0 + FN_for_0 != 0:
        recall_for_0 = TP_for_0/(TP_for_0 + FN_for_0)
    if TP_for_1 + FN_for_1 != 0:
        recall_for_1 = TP_for_1/(TP_for_1 + FN_for_1)
    if TP_for_2 + FN_for_2 != 0:
        recall_for_2 = TP_for_2/(TP_for_2 + FN_for_2)

    macro_precision = (precision_for_0 + precision_for_1 + precision_for_2)/3.0
    macro_recall = (recall_for_0 + recall_for_1 + recall_for_2)/3.0
    return accuracy, macro_precision, macro_recall

def scikit_learn_DT_classifier(x_train, y_train, x_test, y_test):
    clf=tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=10)
    clf=clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    a, counts = np.unique(y_pred, return_counts=True)
    score = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average="macro")
    recall = metrics.recall_score(y_test, y_pred, average="macro")
    return (score, precision, recall)

# Data Reading 
df=pd.read_csv('../data/DataSet_B.csv')
df = df.drop(df.columns[0],  axis='columns')
df = df.rename(columns={"quality": "label"})
df=df.astype(int)

train_df = df.copy()
test_df = df.copy()
x=np.array(train_df.iloc[:, 0:11])
y=np.array(train_df.iloc[:,11])

y=y.astype(int)

selfwrittentree = decision_tree_algorithm(train_df)
accuracy, precision, recall = calculate_accuracy(test_df, selfwrittentree)

sklearn_accuracy, sklearn_precision, sklearn_recall=scikit_learn_DT_classifier(x, y.astype(int), x, y.astype(int))

TrainSetPart1 = x[0:533, :]
TrainSetPart2 = x[533:1066, :]
TrainSetPart3 = x[1066:1599, :]

LabelSetPart1 = y[0:533]
LabelSetPart2 = y[533:1066]
LabelSetPart3 = y[1066:1599]

# 1st fold
traindf=df.loc[0:1066].copy()
testdf=df.loc[1066:1599].copy()
x_train=np.append(TrainSetPart1, TrainSetPart2, axis=0)
y_train=np.append(LabelSetPart1, LabelSetPart2, axis=0)
x_test=TrainSetPart3
y_test=LabelSetPart3
selfwrittentree = decision_tree_algorithm(traindf)
accuracy1, precision1, recall1 = calculate_accuracy(testdf, selfwrittentree)
sklearn_accuracy1, sklearn_precision1, sklearn_recall1=scikit_learn_DT_classifier(x_train, y_train, x_test, y_test)

# 2nd fold
traindf=df.loc[0:533].append(df.loc[1066:1599]).copy()
testdf=df.loc[533:1066].copy()
x_train=np.append(TrainSetPart1, TrainSetPart3, axis=0)
y_train=np.append(LabelSetPart1, LabelSetPart3, axis=0)
x_test=TrainSetPart2
y_test=LabelSetPart2
selfwrittentree = decision_tree_algorithm(traindf)
accuracy2, precision2, recall2 = calculate_accuracy(testdf, selfwrittentree)
sklearn_accuracy2, sklearn_precision2, sklearn_recall2=scikit_learn_DT_classifier(x_train, y_train, x_test, y_test)

# 3rd fold
traindf=df.loc[533:1599].copy()
testdf=df.loc[0:533].copy()
x_train=np.append(TrainSetPart2, TrainSetPart3, axis=0)
y_train=np.append(LabelSetPart2, LabelSetPart3, axis=0)
x_test=TrainSetPart1
y_test=LabelSetPart1
selfwrittentree = decision_tree_algorithm(traindf)
accuracy3, precision3, recall3 = calculate_accuracy(testdf, selfwrittentree)
sklearn_accuracy3, sklearn_precision3, sklearn_recall3=scikit_learn_DT_classifier(x_train, y_train, x_test, y_test)

# Final Calculations
self_written_mean_accuracy = (accuracy1+accuracy2+accuracy3)/3
self_written_mean_precision = (precision1+precision2+precision3)/3
self_written_mean_recall = (recall1+recall2+recall3)/3
print("\nFor Self Written Decision Tree : \n")
print("Mean Macro Accuracy : " + str(100*self_written_mean_accuracy) + "%")
print("Mean Macro Precision: " + str(100*self_written_mean_precision) + "%")
print("Mean Macro Recall   : " + str(100*self_written_mean_recall) + "%")

sklearn_mean_accuracy = (sklearn_accuracy1+sklearn_accuracy2+sklearn_accuracy3)/3
sklearn_mean_precision = (sklearn_precision1+sklearn_precision2+sklearn_precision3)/3
sklearn_mean_recall = (sklearn_recall1+sklearn_recall2+sklearn_recall3)/3
print("\nFor Scikit learn Decision Tree : \n")
print("Mean Macro Accuracy : " + str(100*sklearn_mean_accuracy) + "%")
print("Mean Macro Precision: " + str(100*sklearn_mean_precision) + "%")
print("Mean Macro Recall   : " + str(100*sklearn_mean_recall) + "%")

outF = open("../data/ResultFile_for_Task3.txt", "w")      # opening result file to write results 
outF.write("Task 3 :\n")

outF.write("\nFor Self Written Decision Tree : \n")
outF.write("\nMean Macro Accuracy : " + str(100*self_written_mean_accuracy) + "%")
outF.write("\nMean Macro Precision: " + str(100*self_written_mean_precision) + "%")
outF.write("\nMean Macro Recall   : " + str(100*self_written_mean_recall) + "%")

outF.write("\n\nFor Scikit learn Decision Tree : \n")
outF.write("\nMean Macro Accuracy : " + str(100*sklearn_mean_accuracy) + "%")
outF.write("\nMean Macro Precision: " + str(100*sklearn_mean_precision) + "%")
outF.write("\nMean Macro Recall   : " + str(100*sklearn_mean_recall) + "%")

outF.close()















