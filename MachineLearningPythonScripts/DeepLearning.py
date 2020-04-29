#Uses DeepLearning via TensorFlow in order to determine the class of bank notes. 
#Runs for 600 epochs in order to minimize loss, implemented with Early Stopping and Dropout layers
#for Efficiency and better results. 

from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

def DeepLearning():
    
    #loading data
    df = pd.read_csv("data/bank_note_data.csv")
    df[df.isnull()==False]
    
    #splitting data into training and testing
    x_train, x_test, y_train, y_test = train_test_split(df.drop('Class', axis =1), df['Class'], test_size = 0.3)
    
    #scales the data
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    #Deep learning model with Dropout layers
    model = Sequential()
    model.add(Dense(30, activation = "relu"))
    model.add(Dropout(0.2))
    model.add(Dense(15, activation = "relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = "sigmoid"))

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    early_stop = EarlyStopping(monitor = "val_loss", mode = "min", verbose = 1, patience = 25)
    
    #training the model up to 600 times in order to minimize loss
    
    model.fit(x_train, y_train.values, epochs = 600, validation_data = (x_test, y_test.values), verbose =1 ,callbacks=[early_stop])
    
    #printing results
    predictions = model.predict_classes(x_test)
    print("\n\nConfusion Matrix\n")
    print(confusion_matrix(y_test, predictions))
    print("\n\nClassification Report\n")
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    DeepLearning()
