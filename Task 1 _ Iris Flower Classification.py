#!/usr/bin/env python
# coding: utf-8

# # Oasis Infobyte 
# # Data Science Internship
# # Name : Vora Gautam Kalyanbhai
# # Task 1 : Iris Flower Classification

# # Loading Packages and Data

# In[112]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[113]:


df=pd.read_csv("C:\\Users\\A\\Oasis Infobyte\\Iris.csv")
df


# In[114]:


df.head()


# In[115]:


df.tail()


# In[116]:


df.info()


# In[117]:


df.isnull().sum()


# In[118]:


df.shape


# In[119]:


df.describe()


# In[120]:


df.Species.value_counts()


# # Data Preparation

# In[121]:


# Delete Unwanted Columns.
df.drop("Id",axis=1,inplace=True)


# In[122]:


df.head()


# # Exploratory Data Analysis

# In[123]:


sns.set()


# In[124]:


#distplot for Distribution checking.

#SepalLengthCm distribution
plt.figure(figsize=(6,6))
sns.distplot(df['SepalLengthCm'])
plt.show()


# In[125]:


# SepalWidthCm distribution
plt.figure(figsize=(6,6))
sns.distplot(df['SepalWidthCm'])
plt.show()


# In[126]:


# PetalLengthCm distribution
plt.figure(figsize=(6,6))
sns.distplot(df['PetalLengthCm'])
plt.show()


# In[127]:


# PetalWidthCm distribution
plt.figure(figsize=(6,6))
sns.distplot(df['PetalWidthCm'])
plt.show()


# In[128]:


# Boxplot for Outlier Checking.

# Box Plot
sns.set(style= "darkgrid")
fig,axs1=plt.subplots(2,2,figsize=(15,15))
sns.boxplot(data=df,y="SepalLengthCm",ax=axs1[0,0],color='green')
sns.boxplot(data=df,y="SepalWidthCm",ax=axs1[0,1],color='skyblue')
sns.boxplot(data=df,y="PetalLengthCm",ax=axs1[1,0],color='orange')
sns.boxplot(data=df,y="PetalWidthCm",ax=axs1[1,1],color='yellow')


# In[129]:


# Interpretation: Using boxplot In SepalLengthCm outliers are not present.and we can see that there are some outlier predict in SepalWidthCm,PetalLengthCm and PetalWidthCm.


# In[130]:


# pairplot


# In[131]:


sns.pairplot(df,hue="Species",palette="hls")


# In[132]:


sns.heatmap(df.corr(),annot = True)


# In[133]:


# Interpretation: heatmap show that a correlation of PetalLengthCm and PetalWidthCm is high.


# # Splitting Feactures and Target

# In[134]:



x=df.drop(columns='Species',axis=1)
y=df['Species']


# In[135]:


print(x)


# In[136]:


print(y)


# # Splitting data into Train and Test

# In[137]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[138]:


X_train


# In[139]:


X_test


# In[140]:


print(x.shape,X_train.shape,X_test.shape)


# In[141]:


print('training data shape is:{}.'.format(X_train.shape))
print('training label shape is:{}.'.format(y_train.shape))
print('testing data shape is:{}.'.format(X_test.shape))
print('testing labelshape is:{}.'.format(y_test.shape))


# # Model Building

# # 1)Support Vector Machine Algorithm

# In[142]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC


# In[143]:


m_svm = SVC()


# In[144]:


# Fit the Support Vector Machine model on Training dataset
m_svm.fit(X_train,y_train)


# In[145]:


pred = m_svm.predict(X_test)
pred


# In[146]:


accuracy_score(y_test,pred)


# # Accuracy For our model is 93.33%.

# In[147]:


print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))


# In[148]:


## Let's check the classification report aslo

print("Classification Report")
print(classification_report(y_test, pred))


# # 2) Logistic Regression.

# In[149]:


from sklearn.linear_model import LogisticRegression


# In[150]:


logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)


# In[151]:


from sklearn. metrics import accuracy_score, confusion_matrix
predictions= logmodel. predict(X_test)


# In[152]:


percentage= logmodel.score(X_test,y_test)
res=confusion_matrix(y_test,predictions)
print("confusion matrix")
print(res)


# In[153]:


#check the accuracy on the training set
print(logmodel.score(X_train,y_train))


# In[154]:


print(f"Test set:{len(X_test)}")


# In[155]:


print(f"Accuracy={percentage*100}%")


# # Conclusion:
# # 1)Support Vector Machine Algorithm gives 93.33 % Accuracy.
# # 2)Logistic Regression gives 95.55 % Accuracy.
# # Thank You!!!

# In[ ]:




