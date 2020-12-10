#!/usr/bin/env python
# coding: utf-8

# <b> Installation section </b>

# In[ ]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install numpy')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install sklearn')


# <b> Import libraries </b>

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import matplotlib.ticker as ticker
from sklearn import preprocessing 
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# data set to dataframe 
df=pd.read_csv('teleCust1000t.csv')
df.head()


# <b> Data visulation and analysis </b>

# In[ ]:


df['custcat'].value_counts()
'''3:- Plus services 1:- Basic services 2:- E-service customers 4:-Total services  '''


# In[ ]:


df.hist(column='income',bins=50)
df.hist(column='age')


# In[ ]:


df.columns # to see columns in  data set
'''colums = 'region', 'tenure', 'age', 'marital', 'address', 'income', 'ed',
       'employ', 'retire', 'gender', 'reside', 'custcat'],'''
df[0:5]


# In[ ]:


y=df['custcat'].values # dependent variable 
y[0:5]


# In[ ]:


X=df[['region','tenure','age','marital','address','income','ed','employ','retire','gender','reside']].values # dependet variables
X[0:5]


# In[ ]:


X=preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]


# <b> Data fitting </b>

# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=4)
print('Train Set : X=',x_train.shape,'Y=',y_train.shape)
print('Test Set: x_test=',x_test.shape,'y_test=',y_test.shape)


# <b> Traing and testing of model </b>
# 

# In[ ]:


# Here taking k=3 as random for basic model
k=3
model_3=KNeighborsClassifier(n_neighbors=k).fit(x_train,y_train)
model_3


# In[ ]:


#Model prediction for k=3 
y_predict_model_3=model_3.predict(x_test)
y_predict_model_3[0:5]


# In[ ]:


print("Train set Accuracy for k=3: ", metrics.accuracy_score(y_train, model_3.predict(x_train)))
print("Test set Accuracy for k=3: ", metrics.accuracy_score(y_test, y_predict_model_3))


# In[ ]:


# now here i am taking k=4 to see what is effect on accuaracy
k=4
model_4=KNeighborsClassifier(n_neighbors=k).fit(x_train,y_train)
model_4


# In[ ]:


y_predict_model_4=model_4.predict(x_test)
y_predict_model_4[0:5]
#accuaracy of model
print("Train set Accuracy for k=3: ", metrics.accuracy_score(y_train, model_4.predict(x_train)))
print("Test set Accuracy for k=3: ", metrics.accuracy_score(y_test, y_predict_model_4))


# <p> As here we can see that accuaracy of model is changeed when we change k so we need to find best k for best accuarracy </p>

# In[ ]:


Ks=10 #800 #100  # max K= 800 because data has total 800 records 
mean_accuaracy = np.zeros((Ks-1))
#print(mean_accuaracy)
std_accuaracy=np.zeros((Ks-1))
Confustion_Matrix = []
for n in range (1,Ks):
    model=KNeighborsClassifier(n_neighbors=n).fit(x_train,y_train)
    y_predict=model.predict(x_test)
    mean_accuaracy[n-1]=metrics.accuracy_score(y_test,y_predict)
    std_accuaracy[n-1]=np.std(y_predict==y_test)/np.sqrt(y_predict.shape[0])
mean_accuaracy    


# In[ ]:


# here we created visulation about it
plt.plot(range(1,Ks),mean_accuaracy,'g')
plt.fill_between(range(1,Ks),mean_accuaracy-1 * std_accuaracy,mean_accuaracy +1 * std_accuaracy,alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()


# In[ ]:


print( "The best accuracy was with", mean_accuaracy.max(), "with k=", mean_accuaracy.argmax()+1) 

