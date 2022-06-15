#!/usr/bin/env python
# coding: utf-8

# ## Data Science & Business Analytics intern at The Sparks Foundation 2022.
# ## INTERN: NIKESH GOTAL
# ## TASK 3 - Prediction using Decision Tree Algorithm
# ## Create the Decision Tree classifier and visualize it graphically.

# In[1]:


import numpy as np
import pandas as pd
import sklearn.metrics as sm
import seaborn as sns
import matplotlib.pyplot as mt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report


# In[2]:


data=pd.read_csv('Iris.csv',index_col=0)
data.head()


# In[3]:


data.info()


# In[4]:


data.describe()


# ## Data preprocessing

# In[5]:


target=data['Species']
df=data.copy()
df=df.drop('Species', axis=1)
df.shape


# In[6]:


#defingi the attributes and labels
X=data.iloc[:, [0,1,2,3]].values
le=LabelEncoder()
data['Species']=le.fit_transform(data['Species'])
y=data['Species'].values
data.shape


# ## Trainig the model
# ## We will now split the data into test and train.

# In[7]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("Traingin split:",X_train.shape)
print("Testin spllit:",X_test.shape)


# ## Defining Decision Tree Algorithm

# In[8]:


dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)
print("Decision Tree Classifier created!")


# ## Classification Report and Confusion Matrix

# In[9]:


y_pred=dtree.predict(X_test)
print("Classification report:\n",classification_report(y_test,y_pred))


# In[10]:


print("Accuracy:",sm.accuracy_score(y_test,y_pred))


# ## The accuracy is 1 or 100% since i took all the 4 features of the iris dataset.

# In[11]:


#confusion matrix
cm=confusion_matrix(y_test,y_pred)
cm


# ## Visualization of trained model

# In[12]:


#visualizing the graph
mt.figure(figsize=(20,10))
tree=plot_tree(dtree,feature_names=df.columns,precision=2,rounded=True,filled=True,class_names=target.values)


# ## The Descision Tree Classifier is created and is visaulized graphically. Also the prediction was calculated using decision tree algorithm and accuracy of the model was evaluated.
