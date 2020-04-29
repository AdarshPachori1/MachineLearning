#Predicting the number of stars given (either 1 or 5) a yelp review based on the text of the yelp review
#Uses a pipeline to streamline the process of fitting and transforming text through a count vectorizer,
#tf-idf transformer, and a multinomialdb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline


def NaturalLanguageProcessing():
    
    #loading data
    yelp = pd.read_csv("data/yelp.csv")
    
    #visualizing data
    g = sns.FacetGrid(data = yelp, col = "stars")
    
    yelp_class=yelp[(yelp['stars'] == 1) | (yelp['stars'] ==5)]
    
    #pipelining the training process
    pipeline = Pipeline([
    ('bow',  CountVectorizer()),
    ('tfidf', TfidfTransformer()), 
    ('classifier', MultinomialNB())])
   
    x_train, x_test, y_train, y_test = train_test_split(yelp_class['text'],yelp_class['stars'], test_size = 0.3)
    
    #training the model
    pipeline.fit(x_train, y_train)
    
    #printing out accuracy of our model
    predictions = pipeline.predict(x_test)
    print("\n\nConfusion Matrix\n")
    print(confusion_matrix(y_test, predictions))
    print("\n\nClassification Report\n")
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    
    NaturalLanguageProcessing()
    
