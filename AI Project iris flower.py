#AI Project iris flower- Kmeans algorithm
#import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
from sklearn import metrics

#read dataset file
data = pd.read_csv("iris.csv")

#To make sure the data is read correctly, 
#you can try printing the first five columns
#print(data.head(5))

#splitting the data and select it
dataSplit = 130
selectedData = data[["SepalLength","PetalLength","Species"]]

#to determine the training data
trainingData = selectedData[:dataSplit]
X_train = np.array(trainingData[["SepalLength","PetalLength"]].values)
y_train_flower_name = np.array(trainingData[["Species"]].values)


#to determine the testing  data
testingtData = selectedData[dataSplit:] 
X_test = np.array(testingtData[["SepalLength","PetalLength"]].values)
y_test_flower_name = np.array(testingtData[["Species"]].values)

#create model and train it
k = 3
model = KMeans(n_clusters=k)
model = model.fit((X_train))

#the centroid the algorithm generated it 
centroidsArray = model.cluster_centers_
print("Centoids:")
print(centroidsArray)

# We can look at the clusters each data point was assigned to
print("Training Model Classes Result:")
y_result = model.labels_
print(y_result)

#we create dictionary for flower classes
i=0
flower_to_class = {} 
for flower in y_train_flower_name[:,0]:
    flower_to_class[flower] = y_result[i]
    i = i+1

#print the dictionary 
print("Mapped Flower Name to Class Number:")  
print(flower_to_class)  

#test the model
print("Testing Classes With Predict (y_pred):")
y_pred = model.predict(X_test)
print(y_pred) 

# What true predict ?
y_test = [0] * len(y_test_flower_name)
i = 0
for flower in y_test_flower_name[:,0]:
    for key in flower_to_class: 
        if(y_test_flower_name[i] == key):
            y_test[i] = flower_to_class[key]
            i = i + 1
            break

#print true predict           
print("True Testing Classes (y_test):")
print(np.array(y_test))

#validate the model
print("Accuracy:")
score = accuracy_score(y_test, y_pred, normalize=True, sample_weight=None) * 100
print(score)





# And we'll visualize it:
plt.figure(figsize=(8, 6))
#for training    
scatter_x = X_train[:,0]
scatter_y = X_train[:,1]
group = y_train_flower_name
colors = {1: 'red', 2: 'blue', 3: 'green'}

i=1
fig, ax = plt.subplots()
for flowerName in np.unique(group):
    filteredData = trainingData.loc[trainingData['Species'] == flowerName]
    scatter_x = np.array(filteredData.iloc[:,0])
    scatter_y = np.array(filteredData.iloc[:,1])
    ax.scatter(scatter_x, scatter_y, c =colors[i], label = flowerName, s = 100)
    i = i +1
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
#for centroid 
plt.scatter(centroidsArray[:,0] ,centroidsArray[:,1], color='yellow')
plt.show()

#for testing 
scatter_x = X_test[:,0]
scatter_y = X_test[:,1]
group = y_test_flower_name
colors = {1: 'red', 2: 'blue', 3: 'green'}

i=1
fig, ax = plt.subplots()
for flowerName in np.unique(group):
    filteredData = testingtData.loc[testingtData['Species'] == flowerName]
    scatter_x = np.array(filteredData.iloc[:,0])
    scatter_y = np.array(filteredData.iloc[:,1])
    ax.scatter(scatter_x, scatter_y, c =colors[i], label = flowerName, s = 100)
    i = i +1
      
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
#for centroid 
plt.scatter(centroidsArray[:,0] ,centroidsArray[:,1], color='yellow')
plt.show()