import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)
    pred = dtc.predict(x_test)
    
    print("\n\nConfusion Matrix\n")
    print(confusion_matrix(y_test, pred))
    print("\nClassification Report\n")
    print(classification_report(y_test, pred))

if __name__ == "__main__":
    DecisionTreeClassification()
