import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import sys
sys.path.append("C:/Users10710/OneDrive/桌面/HW8/")
from kmeans import k_means, Cal_Distance, Cal_Accuracy
import math
import matplotlib.pyplot as plt
import random
import numpy as np
import copy
import pandas as pd
from pandas import Series,DataFrame


def Cal_Accuracy_testdata(result,testcount):
    #For iris dataset
    classify_data = [[] for i in range(3)]

    classify_data[0] = result[0:test_count[0]]
    classify_data[1] = result[test_count[0]:test_count[0]+test_count[1]]
    classify_data[2] = result[test_count[0]+test_count[1]:test_count[0]+test_count[1]+test_count[2]]
    numberlabel = [[],[],[]]
    count = [[],[],[]]
    acc = [[],[],[]]
    for i in range(0,3):
        numberlabel[i] = max(classify_data[i],key=classify_data[i].count)
        count[i] = classify_data[i].count(numberlabel[i])
    for i in range(0,3):
        acc[i] = count[i]/len(classify_data[i])

    acc = np.mean(acc)

    return acc

def Cal_Distance_digitnumber(feature, center):
    dimension = len(feature)
    dist = 0.0
    for i in range(dimension):
        dist += (feature[i] - center[i]) ** 2
    return math.sqrt(dist)

def testdata_classify(testdata,center):
    data = testdata[:,1:]
    labels = [-1 for i in range(len(data))]

    for labelIndex, item in enumerate(data):
        classIndex = -1
        minDist = math.inf            
        for i, point in enumerate(center):
            dist = Cal_Distance_digitnumber(item, point)
            #Minimize the distance ,select the cluster
            if(dist < minDist):
                classIndex = i
                minDist = dist
        labels[labelIndex] = classIndex
    return labels

train_dataset =  pd.read_csv('C:/Users/10710/OneDrive/桌面/xuezhengyuansb/ziptrain.csv', header=None)
train_dataset = train_dataset.drop([0],axis =0)
train_dataset = train_dataset.drop([0],axis =1)
train_dataset = train_dataset[train_dataset[1].isin([1,4,8,'1','4','8'])]

test_dataset = pd.read_csv('C:/Users/10710/OneDrive/桌面/xuezhengyuansb/ziptest.csv', header=None)
test_dataset = test_dataset.drop([0],axis = 0)
test_dataset = test_dataset.drop([0],axis = 1)
test_dataset = test_dataset[test_dataset[1].isin([1,4,8,'1','4','8'])]


train_dataset_np = train_dataset.values
test_dataset_np = test_dataset.values

for i in range(0,len(train_dataset_np)):
    for j in range(0,len(train_dataset_np[0])):
        train_dataset_np[i][j] =float(train_dataset_np[i][j])

for i in range(0,len(test_dataset_np)):
    for j in range(0,len(test_dataset_np[0])):
        test_dataset_np[i][j] =float(test_dataset_np[i][j])

#The first number is class [1,4,8]
#print(train_dataset_np)
#print(test_dataset_np)

#Count 1,4,8:
train_count = [0,0,0]

for i in range(0,len(train_dataset_np)):
    if train_dataset_np[i][0] == 1:
        train_count[0] += 1
    if train_dataset_np[i][0] == 4:
        train_count[1] += 1
    if train_dataset_np[i][0] == 8:
        train_count[2] += 1
#print(train_count)

test_count = [0,0,0]

for i in range(0,len(test_dataset_np)):
    if test_dataset_np[i][0] == 1:
        test_count[0] += 1
    if test_dataset_np[i][0] == 4:
        test_count[1] += 1
    if test_dataset_np[i][0] == 8:
        test_count[2] += 1

#print(train_count)
train_dataset_np_order = train_dataset_np[train_dataset_np[:,0].argsort()]#The number:1:1005, 4:652, 8:542
test_dataset_np_order = test_dataset_np[test_dataset_np[:,0].argsort()]

#print(train_dataset_np_order)

result = k_means(train_dataset_np_order, 3, 1000)
print(result[1])

#A method to assign class labels to each of your clusters：
acc = Cal_Accuracy_testdata(result[1], train_count)
print("The acc of training is :",acc)

#Predict labels on the zip.test data:
labels = testdata_classify(test_dataset_np_order, result[2])
print("Print labels", labels)

#Acc:
acc_test = Cal_Accuracy_testdata(labels, test_count)
print("The acc of test is :",acc_test)

#PCA:
train_data_pca = train_dataset_np_order[:,1:]
train_pca = PCA(n_components= 2)
pca_data = train_pca.fit_transform(train_data_pca)
print(pca_data)

pca_data_label = train_dataset_np_order[:,0:1]
pca_train_data = np.hstack((pca_data_label, pca_data))
print(pca_train_data)

#pca kmeans:
result_pca = k_means(pca_train_data, 3, 1000)
acc_pca = Cal_Accuracy_testdata(result_pca[1], train_count)
print("The acc of training is :",acc_pca)

#acc_loss:
print("The acc loss is :", acc-acc_pca)

if __name__ == '__main__':
    pass