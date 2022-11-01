#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
#from sklearn.neighbors import KNeighborsClassifier #KNN Classifier
from sklearn.metrics import confusion_matrix 
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
#from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier



applications = pd.read_csv("C:/BI/CIND 820/Files/application_record.csv", encoding = 'utf-8') 
credit_record = pd.read_csv("C:/BI/CIND 820/Files/credit_record.csv", encoding = 'utf-8') 

applications.head()


# In[2]:


applications.info()


# In[4]:


applications.describe()


# In[5]:


credit_record.info()


# In[6]:


credit_record.describe()


# In[7]:


applications.isnull().sum()


# In[8]:


applications.FLAG_MOBIL.value_counts()


# In[9]:


credit_record.head(10)


# In[11]:


credit_record.STATUS.value_counts()


# In[2]:


#Convert status values to binary (2,3,4,5 --> 1 'Bad'; Else 0 'Good')

credit_record['STATUS'] = np.where((credit_record['STATUS'] == '2') | 
                                   (credit_record['STATUS'] == '3' )| 
                                   (credit_record['STATUS'] == '4' )| 
                                   (credit_record['STATUS'] == '5'), 1, 0)


# In[3]:


credit_record.head(10)


# In[6]:



credit_record.STATUS.value_counts()


# In[ ]:





# In[5]:


applications.FLAG_MOBIL.value_counts()


# In[ ]:





# In[7]:


#Drop unwanted data
applications.drop( columns = ['FLAG_MOBIL'],inplace=True)
applications.dropna(subset=['OCCUPATION_TYPE'],inplace=True)
applications.drop_duplicates(subset=applications.columns[1:],inplace=True)


# In[9]:


applications.info()


# In[93]:


#Optional

merged_data2 = pd.merge(applications, credit_record, how = "inner" , on='ID')


# In[94]:


#Optional

merged_data2['Age']= -(merged_data2['DAYS_BIRTH'])//365
merged_data2['Years_of_employment']= -(merged_data2['DAYS_EMPLOYED'])//365
merged_data2.drop( columns = ['DAYS_BIRTH'],inplace=True)
merged_data2.drop( columns = ['DAYS_EMPLOYED'],inplace=True)
merged_data2


# In[95]:


#Optional

import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

sns.heatmap(merged_data2.corr(), ax=ax, annot=True)


# In[96]:


#Optional


corr = merged_data2.corr()[['STATUS']].sort_values(by='STATUS', ascending=False)

sns.heatmap(corr, annot=True)


# In[12]:


#creating a DF with the most recent month in each status for all applications

credit_classified = pd.DataFrame(pd.unique(credit_record.ID),columns = ['ID'])


# In[13]:


#creating a DF with the most recent month in each status for all applications
credit_classified['Max_Mnth_Good'] = [max(credit_record[(credit_record.ID == i) & (credit_record.STATUS == 0)].MONTHS_BALANCE) for i in credit_classified.ID]
credit_classified['Max_Mnth_Bad'] = [max(credit_record[(credit_record.ID == i) & (credit_record.STATUS == 1)].MONTHS_BALANCE ,default=1) for i in credit_classified.ID]


# In[14]:


#creating a DF with the most recent month in each status for all applications

credit_classified['Status'] = ["Good" if (credit_classified.Max_Mnth_Good.iloc[i] > credit_classified.Max_Mnth_Bad.iloc[i]) or (credit_classified.Max_Mnth_Bad.iloc[i] == 1) else "Bad" for i in range(len(credit_classified.ID))]


# In[15]:


credit_classified.Status.value_counts()


# In[16]:


credit_classified.head()


# In[17]:


credit_classified


# In[18]:


credit_classified[credit_classified["Status"]=="Bad"]


# In[11]:


#Merging all data


# In[19]:


merged_data = pd.merge(applications, credit_classified, how = "inner" , on='ID')


# In[20]:


merged_data.info()


# In[21]:


merged_data.describe()


# In[22]:


#Handling Outliers
#the function to define the whiskers 
def drop_outlier(x):
    q75,q25 = np.percentile(merged_data[x],[75,25])
    intr_qr = q75-q25
    mx = q75+(1.5*intr_qr)
    mn = q25-(1.5*intr_qr)
    return mx,mn


# In[106]:


#pip install seaborn==0.11.0


# In[23]:


sns.displot(merged_data, x="AMT_INCOME_TOTAL")


# In[24]:


merged_data.boxplot('AMT_INCOME_TOTAL')


# In[25]:


mx,mn = drop_outlier('AMT_INCOME_TOTAL')
mx,mn


# In[26]:


merged_data.drop(merged_data[merged_data.AMT_INCOME_TOTAL > mx].index,inplace=True)
#merged_data.shape[0]
len(merged_data.index)


# In[27]:


sns.displot(merged_data, x="AMT_INCOME_TOTAL")


# In[28]:


sns.displot(merged_data, x="CNT_CHILDREN")


# In[30]:


merged_data.boxplot('CNT_CHILDREN')


# In[31]:


mx,mn = drop_outlier('CNT_CHILDREN')
mx,mn


# In[32]:


merged_data.drop(merged_data[merged_data.CNT_CHILDREN > 3].index,inplace=True)
len(merged_data.index)


# In[35]:


#"Customer Distribution by number of children
sns.displot(merged_data, x="CNT_CHILDREN")


# In[ ]:





# In[36]:


merged_data.boxplot('DAYS_EMPLOYED')


# In[37]:


mx,mn = drop_outlier('DAYS_EMPLOYED')
mx,mn


# In[38]:


merged_data.drop(merged_data[merged_data.DAYS_EMPLOYED > mx].index,inplace=True)
merged_data.drop(merged_data[merged_data.DAYS_EMPLOYED < mn].index,inplace=True)
len(merged_data.index)


# In[39]:


sns.displot(merged_data, x="DAYS_EMPLOYED")


# In[40]:


merged_data.boxplot('CNT_FAM_MEMBERS')


# In[41]:


mx,mn = drop_outlier('CNT_FAM_MEMBERS')
mx,mn


# In[42]:


merged_data.drop(merged_data[merged_data.CNT_FAM_MEMBERS > 6].index,inplace=True)
len(merged_data.index)


# In[133]:


#"Customer Distribution by family members

sns.displot(merged_data, x="CNT_FAM_MEMBERS")


# In[43]:


merged_data.info()


# In[ ]:





# In[44]:


fig, axes = plt.subplots(1,3)

g1= merged_data['CODE_GENDER'].value_counts().plot.pie(explode=[0.1,0.1], ax=axes[0])
g1.set_title("Customer Distribution by Gender")

g2= merged_data['FLAG_OWN_CAR'].value_counts().plot.pie(explode=[0.1,0.1], ax=axes[1])
g2.set_title("Car Ownership")

g3= merged_data['FLAG_OWN_REALTY'].value_counts().plot.pie(explode=[0.1,0.1], ax=axes[2])
g3.set_title("Realty Ownership")

fig.set_size_inches(14,5)

plt.tight_layout()

plt.show()


# In[45]:


#Customer Distribution by Income Type

merged_data['NAME_INCOME_TYPE'].hist()
sns.set(rc={'figure.figsize':(15,3)})


# In[137]:


#Customer Distribution by family status
merged_data['NAME_FAMILY_STATUS'].hist()
sns.set(rc={'figure.figsize':(15,3)})


# In[138]:


#Customer Distribution by Housing type
merged_data['NAME_HOUSING_TYPE'].hist()
sns.set(rc={'figure.figsize':(15,3)})


# In[46]:


#Customer Distribution by Education Type
merged_data['NAME_EDUCATION_TYPE'].hist()
sns.set(rc={'figure.figsize':(15,3)})


# In[139]:


#Income type Distribution in realty ownership
from pylab import rcParams
sns.set(rc={'figure.figsize':(15,3)})
sns.countplot(x='NAME_INCOME_TYPE',hue='FLAG_OWN_REALTY',data=merged_data)


# In[140]:


#Income type Distribution in gender
sns.set(rc={'figure.figsize':(15,3)})
S=sns.countplot(x='NAME_INCOME_TYPE',hue='CODE_GENDER',data=merged_data)
S.axes.set_title("NAME_INCOME_TYPE",fontsize=20)


# In[141]:


sns.set(rc={'figure.figsize':(15,3)})
plt.xticks(fontsize=15,rotation='vertical')
P=sns.countplot(x='OCCUPATION_TYPE',hue='CODE_GENDER',data=merged_data)
P.axes.set_title("OCCUPATION_TYPE",fontsize=20)


# In[145]:


#customer distribution by age
sns.set(rc={'figure.figsize':(10,3)})
merged_data['Age']= -(merged_data['DAYS_BIRTH'])//365
merged_data['Age']= merged_data['Age'].astype(int)
#print(merged_data['Age'].value_counts(bins=10,normalize=True,sort=False))
#merged_data['Age'].plot(kind='hist',bins=20,density=True)
#plt.show()


# In[146]:


merged_data['Age'].hist()


# In[147]:


#customer distribution by years of employment

sns.set(rc={'figure.figsize':(10,3)})
merged_data['Years_of_employment']= -(merged_data['DAYS_EMPLOYED'])//365
#merged_data['Years_of_employment']= merged_data['Years_of_employment'].astype(int)
#print(merged_data['Age'].value_counts(bins=10,normalize=True,sort=False))
#merged_data['Years_of_employment'].plot(kind='hist',bins=20,density=True)
#plt.show()
merged_data['Years_of_employment'].hist()


# In[ ]:





# In[148]:


sns.set(rc={'figure.figsize':(10,3)})
merged_data['AMT_INCOME_TOTAL']=merged_data['AMT_INCOME_TOTAL'].astype(object)
#merged_data['AMT_INCOME_TOTAL'] = merged_data['AMT_INCOME_TOTAL']/10000
#print(merged_data['AMT_INCOME_TOTAL'].value_counts(bins=10,sort=False))
#merged_data['AMT_INCOME_TOTAL'].plot(kind='hist',bins=60,density=True)
#plt.show()

merged_data['AMT_INCOME_TOTAL'].hist()


# In[149]:


#optional

merged_data.groupby(['Status']).count().plot(kind='pie', y='ID')


# In[150]:


#import seaborn as sns
#import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

sns.heatmap(merged_data.corr(), ax=ax, annot=True)


# In[153]:


merged_data.drop( columns = ['DAYS_BIRTH'],inplace=True)
merged_data.drop( columns = ['DAYS_EMPLOYED'],inplace=True)

fig, ax = plt.subplots(figsize=(10, 6))

sns.heatmap(merged_data.corr(), ax=ax, annot=True)


# In[154]:


#optional

corr = merged_data.corr()[['Max_Mnth_Good']].sort_values(by='Max_Mnth_Good', ascending=False)

sns.heatmap(corr, annot=True)


# In[155]:


#optional

corr = merged_data.corr()[['Max_Mnth_Bad']].sort_values(by='Max_Mnth_Bad', ascending=False)

sns.heatmap(corr, annot=True)


# In[ ]:




