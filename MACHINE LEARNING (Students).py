#SUPERVISED MACHINE LEARNING
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

##LOADING OUR DATA
placement_datasets = pd.read_csv('data sets/Placement_Dataset.csv', sep=',')
print(placement_datasets.head())  ##prints out the first five rows of our datasets
#Looking for Missing values on our datasets
print(placement_datasets.isnull().sum())  ##gives us the missing values that we have on our datasets
##Dropping our salary column
data_sets = placement_datasets.drop(columns='salary',
                                    axis=0)  ## to remove our salary column from our datasets
print(data_sets.head())  ##To give us the Datasets that we have Except the salary column
##Renaming "Placed" and "Not Placed" renaming placed as 1 and not placed as 0
data_sets['status'] = data_sets['status'].map({
    "Placed": 1,
    "Not Placed": 0
})  ## changing our target variables of "Placed and Not Placed" to 1 and 0
print(data_sets['status'].value_counts())  ##This will help us to see how many replaced targets of 0 and 1 we have
##We have 148 placed rows and 67 Not placed rows
print(data_sets.shape)  ##This will help us to know how many rows and features we have on our data
print(data_sets.describe())  ##This is going to give us the statistical Informattion about our data
##To see the satatistical info about our target values
##WELL TAKE THE STATUS COLUMN AS OUR TARGET
X=data_sets.drop(columns=['status']) ## TO DROP THE "status" column and have only our data
Y=data_sets['status']##This is our Target variable of 0 and 1
print(X.head())##This is going to print the first 5 rows of our data
##Splitting our data to Training Data and Testing data
X=pd.get_dummies(X,drop_first=True)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=42)##the test size is the data that we will be using for our testing of our model
#The smaller the test size the Greater the training size (test_size=0.2)we are training 80% of data (given to the model to learn )and 20% of our data for testing
print(X.shape,X_train.shape,X_test.shape)##This is to Give us how many data points we had and how many is going for training and how many for testing our ML model
##LOADING OUR MACHINE LEARNING MODEL(LOGISTIC REGRESSION)
MODEL=LogisticRegression(max_iter=1000)##This is the model we are going to give the training sets of data
MODEL.fit(X_train,Y_train)##<-----this is how our model is going to learn from the given x_train sets and y_train sets
x_train_prediction=MODEL.predict(X_train)##This is to tell how good our prediction is
##seeing how accurate our x_train_prediction is
Accuracy_score_train=accuracy_score(Y_train,x_train_prediction)##this gives our Accuracy score of the predicted x_train values
print(f"The accuracy of the training set is {Accuracy_score_train*100:.2f}%")
y_test_prediction=MODEL.predict(X_test)##This is now the given x_test data and asked to predict the target of
#print(y_test_prediction)##The prediction made by our Model after learning and trying to predict our x_test target
#print(Y_test)##The test target that is 0.2 of test_size represening our test sets
Accuracy_score=accuracy_score(Y_test,y_test_prediction)##How accurate our testting is from our prediction
print(f"The accuracy of the test set is: {Accuracy_score*100:.2f}%")
compairison=pd.DataFrame({
    "Actual":Y_test,
    "Predicted":y_test_prediction##How many correct predictions we hadve
})
print(compairison)
##now lets give it some data to see its prediction
