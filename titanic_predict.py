#!/usr/bin/python

#predicting titanic surviving probability based on titanic data set

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import csv as csv
import plotly.plotly as py
import plotly.graph_objs as go

#reading the training data into a dataframe
trainData=pd.read_csv("train.csv", header=0)

####converting all strings to integers####

#female=0, male=1
trainData['Gender']=trainData['Sex'].map({'female':0,'male':1}).astype(int)

# for missing Embarked -> just make them embark from most common place
if len(trainData.Embarked[ trainData.Embarked.isnull() ]) > 0:
    trainData.Embarked[ trainData.Embarked.isnull() ] = trainData.Embarked.dropna().mode().values
# determine all values of Embarked
Ports = list(enumerate(np.unique(trainData['Embarked'])))    
# set up a dictionary in the form  Ports : index
Ports_dict = { name : i for i, name in Ports }
# Convert all Embark strings to int 
trainData.Embarked = trainData.Embarked.map( lambda x: Ports_dict[x]).astype(int)     
# All the ages with no data -> make the median of all Ages
median_age = trainData['Age'].dropna().median()
if len(trainData.Age[ trainData.Age.isnull() ]) > 0:
    trainData.loc[ (trainData.Age.isnull()), 'Age'] = median_age

# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
trainData = trainData.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 


# Testing data set
#reading the testing data into a dataframe
testData=pd.read_csv("test.csv", header=0)

#shaping the same data in the same way as the training data
#female=0, male=1
testData['Gender']=testData['Sex'].map({'female':0,'male':1}).astype(int)

# for missing Embarked -> just make them embark from most common place
if len(testData.Embarked[ testData.Embarked.isnull() ]) > 0:
    testData.Embarked[ testData.Embarked.isnull() ] = testData.Embarked.dropna().mode().values
# determine all values of Embarked
Ports = list(enumerate(np.unique(testData['Embarked'])))    
# set up a dictionary in the form  Ports : index
Ports_dict = { name : i for i, name in Ports }
# Convert all Embark strings to int 
testData.Embarked = testData.Embarked.map( lambda x: Ports_dict[x]).astype(int)     
# All the ages with no data -> make the median of all Ages
median_age = testData['Age'].dropna().median()
if len(testData.Age[ testData.Age.isnull() ]) > 0:
    testData.loc[ (testData.Age.isnull()), 'Age'] = median_age

# All the missing Fares -> assume median of their respective class
if len(testData.Fare[ testData.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = testData[ testData.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        testData.loc[ (testData.Fare.isnull()) & (testData.Pclass == f+1 ), 'Fare'] = median_fare[f]


# Collect the test data's PassengerIds and sex before dropping it
ids = testData['PassengerId'].values
sex_test = testData['Gender'].values
# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
testData = testData.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 

#converting both datasets into numpy array
train=trainData.values
test=testData.values


print 'Training...'
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit( train[0::,1::], train[0::,0] )

print 'Predicting...'
output = forest.predict(test).astype(int)

predictions_file = open("predictions.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived","Gender"])
open_file_object.writerows(zip(ids, sex_test, output))
predictions_file.close()
print 'Done.'

print 'plotting survival data of Men and Women in Plotly'
male_survived=0
female_survived=0
male_died=0
female_died=0
for i in range(len(trainData['Gender'])):
	saved=trainData['Survived'][i]
	gender=trainData['Gender'][i]
	if saved==1 and gender==1:
		male_survived +=1			
	elif saved==0 and gender==1:
		male_died +=1
	elif saved==0 and gender==0:
		female_died +=1
	elif saved==1 and gender==0:
		female_survived +=1
survival=[(male_survived+male_died),(male_died),(female_survived+female_died),(female_died)]
print survival
"""
#using plotly to plot survival data by gender (only training data set)
data=[go.Bar(x=['total male','male died','total female', 'female died'],y=survival)]
py.iplot(data, filename='titanic_survival')
"""

# plotting for the output created by random forest model in test data set
print "plotting predicted survival data"
male_test_alive=0
female_test_alive=0
male_test_died=0
female_test_died=0

for i in range(len(sex_test)):
	saved=output[i]
	gender=sex_test[i]
	if saved==1 and gender==1:
		male_test_alive +=1			
	elif saved==0 and gender==1:
		male_test_died +=1
	elif saved==0 and gender==0:
		female_test_died +=1
	elif saved==1 and gender==0:
		female_test_alive +=1
survival_test=[(male_test_alive+male_test_died),(male_test_died),(female_test_alive+female_test_died),(female_test_died)]
data_test=[go.Bar(x=['total male','male died','total female', 'female died'],y=survival_test)]
py.iplot(data_test, filename='titanic_survival_test')

