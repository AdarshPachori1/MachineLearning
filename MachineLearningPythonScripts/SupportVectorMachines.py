#Trains a Support Vector Machine on seaborn's buildin dataset iris 
#which will very likely predict the type of flower based on it's characteristics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

def SupportVectorMachines():   
    #loading dataset
    iris = sns.load_dataset('iris')
    
    #data visualization
    sns.pairplot(data = iris, hue = 'species')
    
    #splitting data to training and testing
    x_train, x_test, y_train, y_test = train_test_split(iris.drop('species', axis = 1), iris['species'], test_size = 0.3)
    
    #training model
    svc = SVC()
    svc.fit(x_train, y_train)
    
    predictions = svc.predict(x_test)
    
    #printing accuracy of our model
    print("\n\nConfusion Matrix\n")
    print(confusion_matrix(y_test, predictions))
    print("\n\nClassification Report\n")
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    SupportVectorMachines()