
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import math
import matplotlib.pyplot as plt
import random
import numpy as np
import copy
import random



def Cal_Accuracy(result):
    #For iris dataset
    classify_data = [[] for i in range(3)]

    classify_data[0] = result[0:50]
    classify_data[1] = result[50:100]
    classify_data[2] = result[100:150]
    numberlabel = [[],[],[]]
    count = [[],[],[]]
 
    for i in range(0,3):
        numberlabel[i] = max(classify_data[i],key=classify_data[i].count)
        count[i] = classify_data[i].count(numberlabel[i])
    acc = np.array(count)/50
    acc = np.mean(acc)
    #print("%.4f" % acc)
    return acc

def Cal_Distance(feature, center):
    dimension = len(feature)
    dist = 0.0
    for i in range(dimension):
        dist += (feature[i] - center[i]) ** 2
    return math.sqrt(dist)

def k_means(input_data, k ,iteration ):

    data = input_data[:,1:]
    eval_sum_disatance = []

    #Intialize random sample
    index= random.sample(list(range(len(data))), k)
    
    #Intialize node & label
    #intialized_node = [data[i] for i in index]
    #labels = [-1 for i in range(len(data))]
    
    center = [data[i] for i in index]
    labels = [-1 for i in range(len(data))]

    center_list = []
    distance = []
    #Iteration
    for x in range(0,iteration):
        C = [[] for i in range(k)]

        for labelIndex, item in enumerate(data):
            classIndex = -1
            minDist = math.inf            
            for i, point in enumerate(center):
                dist = Cal_Distance(item, point)
                #Minimize the distance ,select the cluster
                if(dist < minDist):
                    classIndex = i
                    minDist = dist
            C[classIndex].append(item)
            labels[labelIndex] = classIndex
        distance.append(copy.deepcopy(minDist))

        for i, cluster in enumerate(C):
            clusterHeart = []
            dimension = len(data[0])
            for j in range(dimension):
                clusterHeart.append(0)
            for item in cluster:
                for j, coordinate in enumerate(item):
                    clusterHeart[j] += coordinate / len(cluster)
            center[i] = clusterHeart
        center_list.append(copy.deepcopy(center))
    
        
        if  x >=1 and center_list[x] == center_list[x-1]:
            print("The k-means converge at loop ", x)
            #print("The cluster centers are:", center_list[x])
            return C, labels, center_list[x]

    return C, labels, center_list[x]

if __name__ == "__main__":
    # import some data to play with
    iris = datasets.load_iris()

    feature = iris.data[:, :4] 
    groundtruth = iris.target.reshape(len(feature),-1)
    input = np.hstack((groundtruth, feature))

    #print(catalog)
    res = k_means(input, 3, 100)
    print("-------------------------------")
    print("Output predicted label:", res[1])
    print("Output centers:", res[2])

    acc = Cal_Accuracy(res[1])


    print("-------------------------------")
    #Sklearn
    kmeans = KMeans(n_clusters=3)#新建KMeans对象，并传入参数
    kmeans.fit(feature)
    #print(kmeans.labels_)
    print("The accuracy is : ", acc)
    print("The  built-in function accuracy is: ", Cal_Accuracy(kmeans.labels_.tolist()))
    #print(kmeans.predict([[0, 0], [4, 4]]))
    #print(kmeans.cluster_centers_)

    #Run 20 times:
    acc_list = []
    print("-------------------------------")
    for times in range(0,20):
        res = k_means(input, 3, 100)
        acc = Cal_Accuracy(res[1])
        acc_list.append(acc)
    print("20 times acc are:",acc_list)
"""
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()
"""