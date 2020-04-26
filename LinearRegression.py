import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


def LinearRegressionClassification():
    #loading file
    df = pd.read_csv("data/Ecommerce Customers")
    
    #data analysis
    sns.jointplot(x= 'Time on App', y = "Yearly Amount Spent", data = df)
    sns.lmplot(x="Length of Membership", y = "Yearly Amount Spent", data = df)
    
    X=df[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
    y = df["Yearly Amount Spent"]


    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)
    
    #training the model
    lm = LinearRegression()
    lm.fit(x_train, y_train)
    predictions = lm.predict(x_test)
    
    #printing out the effectiveness/error of our model 
    print("\n\nYearly Amount Spent (feature that is being predicted) Mean: " + str(df["Yearly Amount Spent"].mean()), end = "")
    print("\n\nOur error values in estimating the Yearly Amount Spent", end ="")
    print("\n\n\nMean Absolute Error: ", end = "")
    print(metrics.mean_absolute_error(y_test, predictions))
    print("\nMean Squared Error: ", end = "")
    print(metrics.mean_squared_error(y_test, predictions))
    print("\nRoot Mean Squared Error: ", end = "")
    print(np.sqrt(metrics.mean_squared_error(y_test, predictions)))
    print()
    
    #visualizing correlation between predictions and actual data
    plt.scatter(predictions, y_test)

if __name__ == "__main__":
    LinearRegressionClassification()
