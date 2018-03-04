#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 19:06:20 2018

@author: dell
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier



#reading the csv files
X_train=pd.read_csv("/home/dell/Desktop/loan_predictor/train.csv")
X_test=pd.read_csv("/home/dell/Desktop/loan_predictor/test.csv")



#Converting the non-numeric data into numeric ones
X_train.loc[X_train['Education']=='Graduate', 'Education']=1
X_train.loc[X_train['Education']=='Not Graduate', 'Education']=0
X_test.loc[X_test['Education']=='Graduate', 'Education']=1
X_test.loc[X_test['Education']=='Not Graduate', 'Education']=0


X_train.loc[X_train['Loan_Status']=='Y', 'Loan_Status']=1
X_train.loc[X_train['Loan_Status']=='N', 'Loan_Status']=0
#sns.barplot(x='Education',y='Loan_Status',  data=X_train)
#plt.show()

#X_train[['Prop_area1','Prop_area2']]=pd.get_dummies(X_train['Property_Area'], drop_first=True)
#print(X_train['Prop_area1'].head())

X_train.loc[X_train['Property_Area']=='Urban', 'Property_Area']=1
X_train.loc[X_train['Property_Area']=='Rural', 'Property_Area']=0
X_train.loc[X_train['Property_Area']=='Semiurban', 'Property_Area']=0.5

X_test.loc[X_test['Property_Area']=='Urban', 'Property_Area']=1
X_test.loc[X_test['Property_Area']=='Rural', 'Property_Area']=0
X_test.loc[X_test['Property_Area']=='Semiurban', 'Property_Area']=0.5



#sns.boxplot(x='Loan_Status', y='CoapplicantIncome', data=X_train)
#plt.show()


#filling the missing values after some visualisation
X_train['Credit_History'].fillna(1, inplace=True)
X_test['Credit_History'].fillna(1, inplace=True)


X_train['Loan_Amount_Term'].fillna(X_train['Loan_Amount_Term'].mean(), inplace=True)
X_test['Loan_Amount_Term'].fillna(X_test['Loan_Amount_Term'].mean(), inplace=True)


X_train['LoanAmount'].fillna(X_train['LoanAmount'].mean(), inplace=True)
X_test['LoanAmount'].fillna(X_test['LoanAmount'].mean(), inplace=True)


X_train.loc[X_train['Self_Employed']=='Yes', 'Self_Employed']=1
X_train.loc[X_train['Self_Employed']=='No', 'Self_Employed']=0

X_test.loc[X_test['Self_Employed']=='Yes', 'Self_Employed']=1
X_test.loc[X_test['Self_Employed']=='No', 'Self_Employed']=0


X_train['Self_Employed'].fillna(0, inplace=True)
X_test['Self_Employed'].fillna(0, inplace=True)


X_train['Dependents'].replace("3+", 3, inplace=True)
pd.to_numeric(X_train['Dependents'])


X_test['Dependents'].replace("3+", 3, inplace=True)

X_train['Dependents']=X_train.Dependents.astype(float)
X_test['Dependents']=X_test.Dependents.astype(float)

X_train['Dependents'].fillna(X_train['Dependents'].mean(), inplace=True)
X_test['Dependents'].fillna(X_test['Dependents'].mean(), inplace=True)


X_train.loc[X_train['Married']=='Yes', 'Married']=1
X_train.loc[X_train['Married']=='No', 'Married']=0

X_test.loc[X_test['Married']=='Yes', 'Married']=1
X_test.loc[X_test['Married']=='No', 'Married']=0


X_train['Married'].fillna(X_train['Married'].mean(), inplace=True)
X_test['Married'].fillna(X_test['Married'].mean(), inplace=True)

X_train.loc[X_train['Gender']=='Male', 'Gender']=1
X_train.loc[X_train['Gender']=='Female', 'Gender']=0

X_test.loc[X_test['Gender']=='Male', 'Gender']=1
X_test.loc[X_test['Gender']=='Female', 'Gender']=0

X_train['Gender'].fillna(1, inplace=True)
X_test['Gender'].fillna(1, inplace=True)

y_train=X_train['Loan_Status'].copy()
X_train.drop(['Loan_Status'],axis=1, inplace=True)

#z_train=X_train['Loan_ID'].copy()
#z_test=X_test['Loan_ID'].copy()

z_test=X_test['Loan_ID'].copy()
X_train.drop(['Loan_ID'],1, inplace=True)
X_test.drop(['Loan_ID'],1, inplace=True)

#Converting the type of features for model implementation
y_train=y_train.astype('float')
X_train['Education']=X_train['Education'].astype('float')
X_train['Property_Area']=X_train['Property_Area'].astype('float')
X_train['Gender']=X_train['Gender'].astype('float')
X_train['Self_Employed']=X_train['Self_Employed'].astype('float')
X_train['ApplicantIncome']=X_train['ApplicantIncome'].astype('float')



X_test['Education']=X_test['Education'].astype('float')
X_test['Property_Area']=X_test['Property_Area'].astype('float')
X_test['Gender']=X_test['Gender'].astype('float')
X_test['Self_Employed']=X_test['Self_Employed'].astype('float')
X_test['ApplicantIncome']=X_test['ApplicantIncome'].astype('float')



#Using the Random Forest Method
rfc=RandomForestClassifier()

tr=rfc.fit(X_train, y_train)
pred=rfc.predict(X_test).astype(int)

print(rfc.score(X_train, y_train))

final_pred=np.where(pred>0.5, 'Y', 'N')


#Creating the submission file
submissions=pd.DataFrame({
        "Loan_ID": z_test,
        "Loan_Status": final_pred
        })

submissions.to_csv("Loan.csv", sep=',', index=False)