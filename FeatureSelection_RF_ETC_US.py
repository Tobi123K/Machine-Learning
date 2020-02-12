#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# In[50]:


# Read Whole Dataset
df= pd.read_csv('All_Data_Clean.csv')


# In[51]:


# Display whole dataset
df.head()


# In[52]:


# Convert all variables to the category before applying one hote encoding. 
# Category variable is for categorical variables.
df = df.astype('category')


# In[53]:


# Check your data
df.info()


# In[54]:


# Create the data set with data for y, by dropping targetted columns
df5 = df.drop(['F_XACTIV02','F_SOC','F_XSIC202'], axis=1)


# In[55]:


df1 = pd.read_csv('All_Data_Clean.csv')


# In[56]:


# Create the data set with data for x, by dropping targetted columns
#df2 = df1.drop(['ACYEAR','F_INSTID','F_ZDLEV501','F_XQMODE01','F_XAGRPJ01','F_SEXID','f_XJACS01_1level','f_XJACS01_1level',
             #'F_XCL6SS01','F_XSTUDIS01','F_XQUALENT01','F_TARIFF_numeric'], axis= 1)
#df_activity = df.drop(['ACYEAR','F_INSTID','F_ZDLEV501','F_XQMODE01','F_XAGRPJ01','F_SEXID','f_XJACS01_1level','f_XJACS01_1level',
             #'F_XCL6SS01','F_XSTUDIS01','F_XQUALENT01','F_TARIFF_numeric','F_SOC','F_XSIC202'], axis= 1)
df_job = df.drop(['ACYEAR','F_INSTID','F_ZDLEV501','F_XQMODE01','F_XAGRPJ01','F_SEXID','f_XJACS01_1level','f_XJACS01_1level',
             'F_XCL6SS01','F_XSTUDIS01','F_XQUALENT01','F_TARIFF_numeric','F_XACTIV02','F_XSIC202'], axis= 1)
#df_industry = df.drop(['ACYEAR','F_INSTID','F_ZDLEV501','F_XQMODE01','F_XAGRPJ01','F_SEXID','f_XJACS01_1level','f_XJACS01_1level',
             #'F_XCL6SS01','F_XSTUDIS01','F_XQUALENT01','F_TARIFF_numeric','F_XACTIV02','F_SOC',], axis= 1)


# In[57]:


# Check if you selected proper data
df2.info()


# In[58]:


#df_activity.info()
#df_job.info()
#df_industry.info()


# In[59]:


# Create a y dataset by converting df2 to the dummies (one-hot)
#y = pd.get_dummies(df2)
# Creat a dummies for just activity
#y = pd.get_dummies(df_activity)
# Create a dummies for just job
y = pd.get_dummies(df_job)
# Create a dummies for just Industry
#y = pd.get_dummies(df_industry)


# In[60]:


y.info()


# In[33]:


# Drop Other form the Activity data set
#y = y.drop(['F_XACTIV02_Other'], axis=1)


# In[61]:


# Drop unknown form the Job data set
y = y.drop(['F_SOC_Uknown'], axis=1)


# In[62]:


# drop F_XSIC202_Not known/ not applicable from industry dataset
#y = y.drop(['F_XSIC202_Not known/ not applicable'], axis=1)


# In[63]:


# drop F_XSIC202_Not known/ not applicable and F_SOC_Uknown from all 3 dependent from industry dataset
#y = y.drop(['F_XSIC202_Not known/ not applicable','F_SOC_Uknown','F_XACTIV02_Other'], axis=1)


# In[64]:


y.info()


# In[65]:


# Save the y data in the csv file
#y.to_csv('onehote_dependent_RandomF.csv')


# In[66]:


# Read the activity
#y.to_csv('onehote_dependent_Activity.csv')


# In[67]:


#y.to_csv('onehote_dependent_Job.csv')


# In[68]:


#y.to_csv('onehote_dependent_Industry.csv')


# In[69]:


# Read the X data from the csv file
#y = pd.read_csv('onehote_dependent_RandomF.csv', index_col=False)
#y = pd.read_csv('onehote_dependent_Activity.csv', index_col=False)
#y = pd.read_csv('onehote_dependent_Job.csv', index_col=False)
#y = pd.read_csv('onehote_dependent_Industry.csv', index_col=False)


# In[70]:


# Display the y data
y.info()


# In[71]:


# Display the df5 data
df5.info()


# In[72]:


X = pd.get_dummies(df5)


# In[73]:


X.info()


# In[74]:


# Save the y data in the csv file
#X.to_csv('onehote_dependent_RandomF.csv')


# In[75]:


# read the y dataset
#X = pd.read_csv('onehote_dependent_RandomF.csv', index_col=False)


# In[76]:


# =============================================================================
# RANDOM FOREST
# =============================================================================
model = RandomForestRegressor(random_state=1, max_depth=20)
# Rum the model
model.fit(X,y)
features = X.columns
importances = model.feature_importances_
indices = np.argsort(importances)[-20:]  # top features
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[208]:


# =============================================================================
# Feature Importance - Extra Tree Classifier 
# =============================================================================
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()


# In[77]:


# =============================================================================
# Univariate Selection
# =============================================================================
df.info()
#apply SelectKBest class to extract top 40 best features
bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(40,'Score'))  #print 40 best features


# In[ ]:




