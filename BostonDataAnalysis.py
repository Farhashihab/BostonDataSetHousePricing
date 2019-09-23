#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error #to evaluate our result
boston = load_boston() #load boston data
# print(type(boston))
# print(boston.DESCR)
# print(boston.feature_names)
# print(boston.target)
data = boston.data #load the data
print(type(data))
data_frame = pd.DataFrame(data,columns=boston.feature_names) #convert into pandas data frame
data_frame['price'] = boston.target # adding price column into the data frame
# print(data_frame.head())
# print(data_frame.describe()) # description of data frame
# print(data_frame.info()) #every information about each column
print(sns.pairplot(data_frame))


# In[2]:


sns.pairplot(data_frame)


# In[3]:


rows = 2
cols = 7
fig,ax = plt.subplot(nrows=rows,ncols=cols,figsize(16,4))


# In[2]:


rows = 2
cols = 7
fig, ax = plt.subplots(nrows=rows,ncols=cols,figsize=(16,4))

col = data_frame.columns
index = 0
for i in range(rows):
    for j in range(cols):
        sns.distplot(data_frame[col[index]], ax = ax[i][j])
        index = index + 1

plt.tight_layout()#to remove the overlapping


# In[4]:


corrmat = data_frame.corr() #to check the correlation with the target data .Negative correlation indicates inversly proportional to each other
corrmat


# In[134]:


fig, ax = plt.subplots(figsize= (20,10))
sns.heatmap(corrmat,annot=True,annot_kws = {'size':12})


# In[5]:


corrmat.index.values


# In[6]:


def getCorelateddata(corrdata, threshold):
    features = []
    value = []
    for i,index in enumerate(corrdata.index):
        if abs(corrdata[index]) > threshold:
            features.append(index)
            value.append(corrdata[index])
            
    df = pd.DataFrame(data=value, index=features, columns=['Corr values'])
    return df

threshold = 0.50
corr_value = getCorelateddata(corrmat['price'], threshold)
corr_value


# In[8]:


corr_value.index


# In[9]:


#find out the corr_value data 
correlated_data = data_frame[corr_value.index] #get the values from our main data-frame
correlated_data.head()


# In[10]:


sns.set(style='ticks')
sns.pairplot(correlated_data)
# plt.tight_layout()


# In[11]:


sns.heatmap(correlated_data.corr(),annot=True,annot_kws={'size':12})


# In[12]:


#we need to drop our price column

X = correlated_data.drop(labels=["price"],axis = 1)
y = correlated_data["price"]
y.shape


# In[13]:


X.head()


# In[14]:


X_test,X_train,y_test,y_train = train_test_split(X,y,test_size=.2,random_state = 0)


# In[15]:


X_test.shape


# In[16]:


X_train.shape


# In[17]:


model = LinearRegression()
model.fit(X_train,y_train)


# In[20]:


y_pred = model.predict(X_test)
y_pred


# In[21]:


#show the predict value and test value
df = pd.DataFrame(data = [y_pred,y_test])
df.T


# In[22]:


# y_test
from sklearn.metrics import r2_score
score = r2_score(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)

print(score)
print(mae)
print(mse)


# In[23]:


#Store features performance ,here we will change the threshold and check the feature
total_feature_name = []
total_feature = []
selected_corelation_value = []
r2_scores = []
mae_value = []
mse_value = []


# In[24]:


#function to store them into a dataframe
def performance_feature(feature,th,y_true,y_pred):
    score = r2_score(y_true,y_pred)
    mae = mean_absolute_error(y_true,y_pred)
    mse = mean_squared_error(y_true,y_pred)
    
    total_feature.append(len(feature)-1)
    total_feature_name.append(str(feature))
    selected_corelation_value.append(th)
    r2_scores.append(score)
    mae_value.append(mae)
    mse_value.append(mse)
    
    
    df2 = pd.DataFrame(data= [total_feature_name, total_feature,selected_corelation_value,r2_scores,mae_value,mse_value],
                      index =['Features Name','No. of features','Threshold','R2_score','MAE','MSE'])
    return df2.T


# In[31]:


performance_feature(correlated_data.columns.values, 0.90, y_test, y_pred)


# In[32]:


correlated_data.columns


# In[153]:


#Regression Plot

rows = 2
cols = 2
fig,ax = plt.subplots(nrows = rows,ncols=cols,figsize=(16,4))

col = correlated_data.columns
index = 0


for i in range(rows):
    for j in range(cols):
        sns.regplot(x = correlated_data[col[index]], y=correlated_data['price'],ax = ax[i][j])
        index = index + 1
fig.tight_layout()


# In[154]:




corrmat['price']
# In[33]:


corrmat['price']


# In[34]:


#for 0.60 threshold we will get the correlated data 
threshold = 0.60
corr_value = getCorelateddata(corrmat['price'],threshold)
corr_value


# In[36]:


correlated_data = data_frame[corr_value.index]
correlated_data.head()


# In[37]:


# A function that will split train test data and automatically predict the y_pred

def get_y_predict(corr_data):
    X = corr_data.drop(labels='price',axis =1)
    y = corr_data['price']
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=0)
    model = LinearRegression()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    return y_pred,y_test


# In[38]:


from sklearn.metrics import r2_score
get_y_predict(correlated_data)
# y_test
# y_predict

performance_feature(correlated_data.columns.values, 0.50, y_test, y_pred)


# In[39]:


correlated_data.columns.values


# In[41]:


#for 0.70 threshold we will get the correlated data 
threshold = 0.70
corr_value = getCorelateddata(corrmat['price'],threshold)
corr_value


# In[43]:


correlated_value = data_frame[corr_value.index]
correlated_value.head()


# In[44]:


#predication and split of data
get_y_predict(correlated_value)


# In[47]:


correlated_value.columns.values


# In[48]:


performance_feature(correlated_value.columns.values, threshold, y_test, y_pred)


# In[49]:


# lets select only RM and see what are the performance value

correlated_value = data_frame[['RM','price']]
correlated_value.head()


# In[50]:


get_y_predict(correlated_value)
performance_feature(correlated_value.columns.values, threshold, y_test, y_pred)


# In[51]:


#Let's find the feature which are correlated more then 40%

threshold = 0.40
corr_value = getCorelateddata(corrmat['price'],threshold)
corr_value


# In[52]:


correlated_value = data_frame[corr_value.index]
correlated_value.head()


# In[53]:


get_y_predict(correlated_value)
performance_feature(correlated_value.columns.values, threshold, y_test, y_pred)


# In[ ]:


#Normalization and standarization

