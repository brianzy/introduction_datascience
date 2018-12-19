import matplotlib.pyplot as plt  
import pandas as pd  

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class'] # Assign column names to the dataset
dataset = pd.read_csv(url, names=names) # Read dataset to pandas dataframe
X = dataset.iloc[:, :-1].values # the feature set 'sepal-length', 'sepal-width', 'petal-length', 'petal-width'
y = dataset.iloc[:, 4].values # 'Class'

# percentage split 80% training and 20% testing
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  

# K Neighbors classifier with k=13
from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=13)  #knn classifier
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)  

# display the classificaiton report, confusion matrix and accuracy of the classifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
print(accuracy_score(y_test,y_pred))


#create list from range 1 to 100
numbers=list(range(1,101))
neighbors=[num for num in numbers if num % 2 == 1]
error=[]
from sklearn.model_selection import cross_val_score
for k in neighbors:
    classifier = KNeighborsClassifier(n_neighbors=k)
    accuracy=cross_val_score(estimator=classifier, X=X, y=y, scoring='accuracy',cv=10)
    avg_accuracy=accuracy.mean()
    error.append(1-avg_accuracy)
# determining best k
optimal_k = neighbors[error.index(min(error))]
print("The optimal number of neighbors is %d" % optimal_k) 

# plot misclassification error vs k
plt.plot(neighbors, error)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()



