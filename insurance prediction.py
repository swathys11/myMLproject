#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import python libraries and load the csv file
import pandas as pd
data = pd.read_csv("TravelInsurancePrediction.csv")
data.head()


# In[3]:


#remove all the rows that contain null values:
data.drop(columns=["Unnamed: 0"], inplace=True)


# In[4]:


data.isnull().sum()


# In[5]:


data.info()


# In[6]:


#convert 1 and 0 to purchased and not purchased:
data["TravelInsurance"] = data["TravelInsurance"].map({0: "Not Purchased", 1: "Purchased"})


# In[7]:


#Now let’s start by looking at the age column to see how age affects the purchase of an insurance policy:
import plotly.express as px
data = data
figure = px.histogram(data, x = "Age", 
                      color = "TravelInsurance", 
                      title= "Factors Affecting Purchase of Travel Insurance: Age")
figure.show()


# In[8]:


#According to the visualization above, people around 34 are more likely to buy an 
#insurance policy and people around 28 are very less likely to buy an insurance policy. 
#Now let’s see how a person’s type of employment affects the purchase of an insurance policy
import plotly.express as px
data = data
figure = px.histogram(data, x = "Employment Type", 
                      color = "TravelInsurance", 
                      title= "Factors Affecting Purchase of Travel Insurance: Employment Type")
figure.show()


# In[9]:


#According to the visualization above, people working in the private sector or the self-employed
# are more likely to have an insurance policy. Now let’s see how a person’s annual income affects#
#the purchase of an insurance policy:

import plotly.express as px
data = data
figure = px.histogram(data, x = "AnnualIncome", 
                      color = "TravelInsurance", 
                      title= "Factors Affecting Purchase of Travel Insurance: Income")
figure.show()


# In[10]:


#convert all categorical values to 1 and 0 first because all columns are important for training the insurance prediction model:
import numpy as np
data["GraduateOrNot"] = data["GraduateOrNot"].map({"No": 0, "Yes": 1})
data["FrequentFlyer"] = data["FrequentFlyer"].map({"No": 0, "Yes": 1})
data["EverTravelledAbroad"] = data["EverTravelledAbroad"].map({"No": 0, "Yes": 1})
x = np.array(data[["Age", "GraduateOrNot", 
                   "AnnualIncome", "FamilyMembers", 
                   "ChronicDiseases", "FrequentFlyer", 
                   "EverTravelledAbroad"]])
y = np.array(data[["TravelInsurance"]])


# In[29]:


#Now let’s split the data and train the model by using the decision tree classification algorithm:

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)


# In[ ]:


0.8190954773869347
#The model gives a score of over 80% which is not bad for this kind of problem

