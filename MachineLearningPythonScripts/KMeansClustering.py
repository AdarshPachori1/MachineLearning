# Clusters alike data into K clusters based on similar features between points
# Clustering technique is applied here to determine whether or not a college is public or private

import numpy as np
import pandas as pd
import matplotlib.pyplot
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,classification_report

def KMeansClustering():
    
    #loading data
    colleges = pd.read_csv("data/College_Data")
    
    #reorganizing dataset
    colleges.set_index(colleges.columns[0], inplace=True)
    colleges.index.name = "Colleges"
    
    #Data Visualization
    sns.scatterplot(x = 'Room.Board', y = 'Grad.Rate', data = colleges, hue = "Private")

    km = KMeans(2)
    
    km.fit(colleges.drop("Private", axis = 1))

    colleges["Cluster"] = colleges["Private"].apply(lambda x: 0 if x=="Yes" else 1)
    print("\n\nConfusion Matrix\n")
    print(confusion_matrix(colleges['Cluster'],km.labels_))
    print("\nClassification Report\n")
    print(classification_report(colleges['Cluster'],km.labels_))

    
if __name__ == "__main__":
    KMeansClustering()
