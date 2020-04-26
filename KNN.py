import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

def KNNClassification():
    #loading data
    df = pd.read_csv("data/Classified Data",index_col=0)
    
    #data visualization
    sns.pairplot(data = df, hue = 'TARGET CLASS')
    
    #scaling data
    df_scaled = StandardScaler()
    df_scaled.fit(df.drop('TARGET CLASS', axis = 1))
    transformed_features = df_scaled.transform(df.drop('TARGET CLASS', axis =1))
    dataframe_scaled = pd.DataFrame(transformed_features, columns = df.columns.drop("TARGET CLASS"))
   
    #Splitting Data 
    x_train, x_test, y_train, y_test = train_test_split(dataframe_scaled, df['TARGET CLASS'], test_size = 0.3)
    knn = KNeighborsClassifier(n_neighbors = 1)
    knn.fit(x_train, y_train)
    predictions = knn.predict(x_test)
    
    #printing accuracy of our model with K-value 1
    print("\n\nUSING AN ARBITRARY K-VALUE\n")
    print("Confusion Matrix\n")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report\n")
    print(classification_report(y_test, predictions))
    
    #choosing an appropriate K-value
    error_rates = []
    for i in range(1,40):
        knn = KNeighborsClassifier(n_neighbors = i)
        knn.fit(x_train, y_train)
        predictions = knn.predict(x_test)
        error_rates.append(np.mean(predictions!=y_test))
    
    #visualizing which K-value minimizes error
    plt.scatter(np.array(range(1,40)), error_rates, lw = 0.2)
    
    #Retraining model with ideal K-value to mimize the error
    knn = KNeighborsClassifier(n_neighbors = 26)
    knn.fit(x_train, y_train)
    predictions = knn.predict(x_test)
    
    #printing accuracy of our model with ideal K-value
    print("\n\nUSING THE IDEAL K-VALUE\n")
    print("Confusion Matrix\n")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report\n")
    print(classification_report(y_test, predictions))
    
if __name__ == "__main__":
    KNNClassification()
