#Performs principal component analysis which will reduce the dimensionality of datasets
#without losing any information. In this program, we have used principal component analysis
#to reduce the number of variables to only two variables.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def PrincipalComponentAnalysis():
    
    #loading the file
    cancer = load_breast_cancer()
    
    df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
    
    #scaling the data
    scaler = StandardScaler()
    scaler.fit(df)
    scaled_data = scaler.transform(df)
    
    pca = PCA(n_components=2)
    #fitting and transformign the scaled data
    pca.fit(scaled_data)
    x_pca = pca.transform(scaled_data)
    
    plt.figure(figsize=(8,6))
    plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='plasma')
    plt.xlabel('First principal component')
    plt.ylabel('Second Principal Component')
    
    df_comp = pd.DataFrame(pca.components_,columns=cancer['feature_names'])
    plt.figure(figsize=(12,6))
    sns.heatmap(df_comp,cmap='plasma')

    

if __name__ == "__main__":
    PrincipalComponentAnalysis()
