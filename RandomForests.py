#Analyzes loan data and predicts based on features whether a loan
#will get paid back in completion using a Random Forest Classifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def DecisionTreeClassification():
    #loading data
    loans = pd.read_csv("data/loan_data.csv")
    
    #data vizualization
    plt.figure(figsize=(11,7))
    sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1')
    

    cat_feats = ['purpose']
    
    #transforming categorical data into dummy variables
    final_data = pd.get_dummies(loans, columns = cat_feats, drop_first = True)
    
    x_train, x_test, y_train, y_test = train_test_split(final_data.drop('not.fully.paid', axis = 1), final_data['not.fully.paid'], test_size=0.3, random_state=42)
    
    #training model
    rfc = RandomForestClassifier(n_estimators=600)
    rfc.fit(x_train,y_train)
    predict = rfc.predict(x_test)
    
    #printing accuracy
    print("\n\nConfusion Matrix\n")
    print(confusion_matrix(y_test, predict))
    print("\nClassification Report\n")
    print(classification_report(y_test, predict))
    
if __name__ == "__main__":
    DecisionTreeClassification()
