import numpy as np # linear algebra
import pandas as pd 
#import matplotlib.pyplot as plt

# Reading the dataset or loading the dataset
train_data=pd.read_csv('train.csv')
train_data.head()
test_data=pd.read_csv('test.csv')
test_data.head()

# checking the sample submission holds a true pateren or not
from sklearn.ensemble import RandomForestClassifier

# Selecting a target variable or column for prediction
y_train = train_data["Survived"] 

 # defining the features basically input information for the model

features = ["Pclass", "Sex", "SibSp", "Parch"]

# Feature encoding basically converting the categorical data to  to some numerical format knows as one-hot-encoding

X_train = pd.get_dummies(train_data[features])
# feature encoding of test_data
X_test = pd.get_dummies(test_data[features])
'''
'n_estimators=100' specifies that the random forest should consist of 100 decision trees.
'max_depth=5' sets the maximum depth of the tree. This means each decision tree in the forest can grow up to a depth of 5. It helps in controlling overfitting.
'random_state=1' ensures that the splits that you generate are reproducible. Different random states can lead to slightly different tree structures.

'''
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

'''
model.fit(X, y) is used to train the model on the training data. X represents the features (input variables) of the training data, and y is the target variable (the output you're trying to predict).
The fit method adjusts the parameters of the model so that it can accurately map the input data X to the target y.
'''

model.fit(X_train, y_train)
'''
After the model is trained, it's used to make predictions on new, unseen data.
X_test is the feature data for which you want to predict the target variable. It should be processed and prepared in the same way as the training data (X).
model.predict(X_test) is used to get the predictions from the model for X_test. These predictions are stored in the predictions variable.
'''

predictions = model.predict(X_test)

# generating the csv file as a submission

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
