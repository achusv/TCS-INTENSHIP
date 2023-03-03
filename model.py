# In[1]:


#import necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Libraries for data visualization
import matplotlib.pyplot as plt  
import seaborn as sns 
from pandas.plotting import scatter_matrix
from collections import Counter
def printmd(string):
    display(Markdown(string))


# Import sys and warnings to ignore warning messages 
import sys
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')

if not sys.warnoptions:
    warnings.simplefilter("ignore")


# In[2]:


data=pd.read_csv('salarydata.csv')


# In[3]:


data.head()


# In[4]:


data.tail()


# ### Understanding the dataset

# In[5]:


data.shape


# In[6]:



data.info()


# In[7]:


def data_info(data):
    print('Number of Columns in the dataset: ',data.shape[1])
    print('Number of rows in the dataset: ',data.shape[0])
    print('Total number of datapoints in the dataset:',data.size)
    numerical_features = [f for f in data.columns if data[f].dtypes!='O']
    print('Count of Numerical Features:',len(numerical_features))
    cat_features = [c for c in data.columns if data[c].dtypes=='O']
    print('Count of Categorical Features:',len(cat_features))
data_info(data)


# In[8]:


def unique_checker(data):
    """loops and prints unique values in each column"""
    for col in data.columns:
        print("Unique values in {} feature.".format(col))
        print(data[col].unique(),"\n")
        print("*"*40)


# In[9]:


unique_checker(data) #scroll / toggle output to view all outputs

# Note:
## this step is just for the Data Understanding part
### Not intergral to the straightforward analysis 


# In[10]:


def count_checker(data):
    """count of each value under each feature in the data"""
    for col in data.columns:
        print("Count for each category of values in {} feature.".format(col))
        print(data[col].value_counts(),"\n")
        print("*"*40)


# In[11]:


count_checker(data) #scroll / toggle output to view all outputs
# Note:
## this step is just for the Data Understanding part
### Not intergral to the straightforward analysis 


# **Findings**
# 
# - The dataset contains 48,842 entries with a total of 15 columns representing different attributes of the people. Here’s the list;
# 
# 1. Age: Discrete (from 17 to 90)
# 2. Work class (Private, Federal-Government, etc): Nominal (9 categories)
# 3. Final Weight (the number of people the census believes the entry represents): Discrete
# 4. Education (the highest level of education obtained): Ordinal (16 categories)
# 5. Education Number (the number of years of education): Discrete (from 1 to 16)
# 6. Marital Status: Nominal (7 categories)
# 7. Occupation (Transport-Moving, Craft-Repair, etc): Nominal (15 categories)
# 8. Relationship in family (unmarried, not in the family, etc): Nominal (6 categories)
# 9. Race: Nominal (5 categories)
# 10. Sex: Nominal (2 categories)
# 11. Capital Gain: Continous
# 12. Capital Loss: Continous
# 13. Hours (worked) per week: Discrete (from 1 to 99)
# 14. Native Country: Nominal (42 countries)
# 15. Salary (whether or not an individual makes more than 50,000 dollar annually):     Boolean (≤50k, >50k)

# In[12]:


#creating a Dataframe from the given dataset
df = pd.DataFrame(data)
df.columns


# ### Renaming the columns

# In[13]:



#replacing some special character columns names with proper names 
df.rename(columns={'capital-gain': 'capital_gain', 'capital-loss': 'capital_loss', 'native-country': 'country','hours-per-week': 'hours_per_week','marital-status': 'marital'}, inplace=True)
df.columns


# ### Data Cleaning

# ### 1. Missing Values

# In[14]:


#check the missing value
df.isnull().sum()


# In[ ]:





# **Above sum shows there are no null values in the dataset.**

# In[15]:


#we can see that there are some special characters in the data like ‘?’.
#Finding the special characters in the data frame
df.isin(['?']).sum(axis=0)


# **Findings**
# 
# - we see that there is a special character as " ?" for columns workcalss, Occupation, and country, we need to clean those data. 
# - In this case, as the missing value fall into the categorical features, we will use the pandas DataFrame mode() method to fill the missing value.

# In[16]:


#Handling missing values
# the code will replace the special character to nan  
df['country'] = df['country'].replace('?',np.nan)
df['workclass'] = df['workclass'].replace('?',np.nan)
df['occupation'] = df['occupation'].replace('?',np.nan)


# In[17]:



df.isnull().sum()


# In[18]:


#we will use the pandas DataFrame mode() method to fill the missing value.
df = df.fillna(df.mode().iloc[0])


# In[19]:


df.isnull().sum()


# ### 2. Remove duplicate data 

# In[20]:


#Checking for duplicated entries
sum(df.duplicated(df.columns))


# In[21]:


#Delete the duplicates and check that it worked
df = df.drop_duplicates(df.columns, keep='last')
sum(df.duplicated(df.columns))


# In[22]:


df.shape


# In[23]:


df.columns


# ### 3. Handling Outliers

# In[24]:


## checking outliers
for i in ['age',
       'capital_gain','capital_loss','hours_per_week'] :
    plt.title(i)
    sns.boxplot(data=df[i])
    plt.show()  


# In[25]:


#df1=df
#df1.head()


# ### Handling Outliers with age

# In[26]:


q1 = np.percentile(df['age'],25,interpolation='midpoint')
q3 = np.percentile(df['age'],75,interpolation='midpoint')

IQR = q3-q1
low_limit=q1-1.5*IQR
high_limit=q3+1.5*IQR

index=df['age'][(df['age']<low_limit)|(df['age']>high_limit)].index
df.drop(index,inplace=True)


# ### Handling Outliers with capital_gain

# In[27]:


q1 = np.percentile(df['hours_per_week'],25,interpolation='midpoint')
q3 = np.percentile(df['hours_per_week'],75,interpolation='midpoint')

IQR = q3-q1
low_limit=q1-1.5*IQR
high_limit=q3+1.5*IQR

index=df['hours_per_week'][(df['hours_per_week']<low_limit)|(df['hours_per_week']>high_limit)].index
df.drop(index,inplace=True)


# ### 6. Feature Reduction 

# - While analyzing the dataset, 
# - As we can see in 'descriptive statistics - Numerical columns',
#     - 'capital-gain'and 'capital-loss' both columns have 75% data as 0.00
#             - So, we can drop 'capital-gain'& 'capital-loss' both columns. 
# - The column,education-num is the numerical version of the column education, so we also drop it.

# In[28]:


df.drop(['capital_gain','capital_loss','education-num'], axis = 1,inplace = True)
df.head()


# In[29]:


df.shape


# Now, we need to convert the categorical values to numeric for modeling. Looking at the Marital-status col, there are nearly 6 different values which would mean the same as two values of being married ot no married, therefore we convert them into only two values.

# In[30]:


df.replace(['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent','Never-married','Separated','Widowed'],
             ['divorced','married','married','married','not married','not married','not married'], inplace = True)


# In[31]:


df['marital'].value_counts()


# Before we do further analysis, we will separate the data as numeric and categorical so that our analysis becomes easy.

# In[32]:


# NUMERIC FEATURES:

numeric_data = df.select_dtypes(include=np.number) # select_dtypes selects data with numeric features
numeric_col = numeric_data.columns 

print('Numeric Features: ')
print(numeric_data.head(5))
print('----'*20)


# In[33]:



# CATEGORICAL FEATURES:

categorical_data = df.select_dtypes(exclude=np.number) # we will exclude data with numeric features
categorical_col = categorical_data.columns

print('Categorical Features: ')
print(categorical_data.head(5))
print('----'*20)


# ## 8. Feature Engineering

# In[34]:


# education Category
df.education= df.education.replace(['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th','10th', '11th', '12th'], 'school')
df.education = df.education.replace('HS-grad', 'high school')
df.education = df.education.replace(['Assoc-voc', 'Assoc-acdm', 'Prof-school', 'Some-college'], 'higher')
df.education = df.education.replace('Bachelors', 'undergrad')
df.education = df.education.replace('Masters', 'grad')
df.education = df.education.replace('Doctorate', 'doc')


# In[35]:


# Salary
df.Salary = df.salary.replace('<=50K', 0)
df.Salary = df.salary.replace('>50K', 1)


# In[36]:


df.corr()


# In[37]:


sns.heatmap(df.corr(), annot=True);


# In[38]:


# Salary
df.Salary = df.Salary.replace( 0,'<=50K')
df.Salary = df.Salary.replace( 1,'>50K')


# In[39]:


df['salary'].value_counts()


# In[40]:


df.info()


# In[41]:


#Covert workclass Columns Datatype To Category Datatype
df['workclass'] = df['workclass'].astype('category')


#  I chose not to use the 'Fnlwgt' attribute that is used by the census, as the inverse of sampling fraction adjusted for non-response and over or under sampling of particular groups. This attribute does not convey individual related meaning.

# ## 9. Encoding

# In[42]:


df.columns


# In[43]:


unique_checker(df)


# ### One-Hot Encoding

# In[44]:


#Select the variables to be one-hot encoded
#one_hot_features = ['marital','sex']
#df1 = pd.get_dummies(df1, columns=one_hot_features)


# In[45]:


#df1.head()


# ### Label Encoding

# In[46]:


#label encoding
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
#convert the categorical columns into numeric 
df['workclass']  = le.fit_transform(df['workclass'])
df['education']  = le.fit_transform(df['education'])
df['occupation']  = le.fit_transform(df['occupation'])
df['relationship']  = le.fit_transform(df['relationship'])
df['race']  = le.fit_transform(df['race'])
df['country']  = le.fit_transform(df['country'])
df['marital']  = le.fit_transform(df['marital'])
df['sex']  = le.fit_transform(df['sex'])


# In[47]:


df.head()


# ### Sampling

# In[48]:


# In[51]:


X=df.drop(columns=['salary'],axis=1)
Y=df['salary']
X.head()


# In[54]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=24) # 80% training and 20% test


# ## RANDOM FOREST

# In[58]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score

rf_model=RandomForestClassifier()
rf_model.fit(X_train,Y_train)
Y_pred=rf_model.predict(X_test)
print('Accuracy on training data is:',rf_model.score(X_train,Y_train))
print('Accuracy is:',accuracy_score(Y_test,Y_pred))
print('Precision is:',precision_score(Y_test,Y_pred,average='weighted'))
print('Recall is:',recall_score(Y_test,Y_pred,average='weighted'))
print('f1 score is:',f1_score(Y_test,Y_pred,average='weighted'))
print(classification_report(Y_test,Y_pred))
print(confusion_matrix(Y_test,Y_pred))


# In[59]:


df


# In[60]:


# save the model
import pickle
filename = 'model.pkl'
pickle.dump(rf_model, open(filename, 'wb'))


# In[61]:


load_model = pickle.load(open(filename,'rb'))


# In[62]:


load_model.predict([[52,4,11,2,3,5,4,0,40,38]])


# In[ ]:





# In[ ]:





# In[ ]:




