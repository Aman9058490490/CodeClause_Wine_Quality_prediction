#!/usr/bin/env python
# coding: utf-8

# In[64]:


#IMPORT LIBRARIES FOR GRAPHING AND VISUALIZATION:
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import seaborn as sns
import sklearn.base
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[2]:


df = pd.read_csv('WineQT.csv')


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.head()


# In[7]:


df.shape


# In[65]:


plt.figure(figsize = (12,6)) # It's describe that,how many wine which are belong to which                              # to which quality:
sns.countplot(df['quality'])#quality:
plt.show()


# In[8]:


plt.figure(figsize = (12,6))
sns.barplot(x='quality',y='alcohol',data=df,palette='inferno')
plt.show()


# In[26]:


plt.figure(figsize = (12,6))
sns.pairplot(df)
plt.show()


# In[10]:


plt.figure(figsize=(12,6))
sns.heatmap(df.corr(),annot = True)
plt.show()


# In[12]:





# In[46]:


sns.lmplot(x='alcohol',y='quality',data=df)


# # Data pre-processing 

# In[47]:


#separate the data and label
x=df.drop('quality',axis =1)
y=df['quality']  


#  Label Binarization

# In[53]:


y=df['quality'].apply(lambda y_value:1 if y_value>=7 else 0)


# In[54]:


y


# Train & Test data

# In[55]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[56]:


y.shape,y_train.shape,y_test.shape


# MODEL Training:
# Random forest Classifier

# In[57]:


model= RandomForestClassifier()


# In[58]:


model.fit(x_train,y_train) 


# Model Evaluation:
# Accuracy Score
# 

# In[60]:


#Accuracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction,y_test)


# In[61]:


print('Accuracy:',test_data_accuracy)


# Building a predictive System

# In[69]:


input_data=(7.4,0.66,0.0,1.8,0.075,13.0,40.0,0.9978,3.51,0.56,9.4,5)


#change the input data to a numpy array
input_data_as_numpy_array =np.asarray(input_data)

#reshape the data as we are predicting the label for only  one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==1):
    print('Good Quality wine')
else:
    print('Bad Quality wine')


# In[ ]:




