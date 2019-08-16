from faker import Faker 
import json            # To create a json file                 
from random import randint
fake = Faker() 
def input_data(x): 
  
    # dictionary 
    student_data ={} 
    for i in range(0, x): 
        student_data[i]={} 
        student_data[i]['CGPA']= randint(5, 10) 
        student_data[i]['10th_percentage']= randint(30,100)
        student_data[i]['12th_percentage']= randint(30,100)
        avg=(50*student_data[i]['CGPA']+2*student_data[i]['10th_percentage']+3*student_data[i]['12th_percentage'])/3;
        if(avg>240):
            student_data[i]['JOB']=1;
        else:
            student_data[i]['JOB']=0;
    #print(student_data) 
  
    # dictionary dumped as json in a json file 
    with open('students.json', 'w') as fp: 
        json.dump(student_data, fp) 
      
  
def main(): 
  
    # Enter number of students 
    number_of_students = 400  # For the above task make this 100 
    input_data(number_of_students) 
main() 
# The folder or location where this python code 
# is save there a students.json will be created  
# having 10 students data
import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

data=pd.read_json('students.json')
data=np.transpose(data)
#datadata=data.values
data
X = data[:,0:3]
y = data[:,3:4]
validation_size = 0.10
seed = 9
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=validation_size, random_state = seed)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

#Step 1 - Create prediction model
model = LogisticRegression()
#Step 2 - Fit model
model.fit(X_train, Y_train)
#Step 3 - Predictions 
predictions = model.predict(X_test)
#Step 4 - Check Accuracy
print("Model --- LogisticRegression")
print("Accuracy: {} ".format(accuracy_score(Y_test,predictions) * 100))
print(classification_report(Y_test, predictions))

new_data = [(95,86,7.36), (80,82,8.7) , (84,85,8.3) ]
#Convert to numpy array
new_array = np.asarray(new_data)
#Output Labels
labels=["NOT PLACED","PLACED"]
#Let's make some kickass predictions
prediction=model.predict(new_array)
#Get number of test cases used
no_of_test_cases, cols = new_array.shape
for i in range(no_of_test_cases):
 print("Status of Student with 10TH % = {}, 12TH % = {}, CPI = {} will be ----- {}".format(new_data[i][0],new_data[i][1],new_data[i][2], labels[int(prediction[i])]))