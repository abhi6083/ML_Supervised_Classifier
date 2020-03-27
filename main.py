#loading neccessary modules
from sklearn import datasets,neighbors
import numpy as np
#loading iris datasets
iris=datasets.load_iris()
features=iris.data
#print(features)
#print(iris.DESCR)
labels=iris.target
model_classifier=neighbors.KNeighborsClassifier()
#training the classifier
model_classifier.fit(features,labels)
output=model_classifier.predict([[float(input("Sepal length:")),float(input("Sepal width:")),float(input("Petal length:")),float(input("Petal width:"))]])
flower_name={0:"- Iris-Setosa",1:"- Iris-Versicolour",2:"- Iris-Virginica"}
print(flower_name[output[0]])

