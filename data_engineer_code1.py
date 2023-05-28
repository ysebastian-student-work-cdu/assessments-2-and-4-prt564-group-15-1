
# ---
#
# A COMPAREHESIVE ANALYSIS ON HIBP BREACH DATASET USING MACHINE LEARNING AND VISUALIZATION.
# ---
#
# - Cybersecurity is an essential aspect for day-to-day activities for modern-day technollogy organizations.
# - Data breaches can have a devastating impact on individuals and organizations alike.
# - In this project, we will be analyzing data from the Have I Been Pwned (HIBP) breach dataset, which contains information on over 600 data breaches.
# - Our goal will be to gain insights into common patterns and trends in cyber attacks and explore ways to better protect against them.
# - Also aside from analysis, We will also create both clustering and Classification mOdel to observe if a data breach was verified or not.
#
#
#
# - We will use Python programming language and various libraries such as Pandas for data manipulation, Matplotlib and Seaborn for data visualization, and Scikit-learn for machine learning.
# - Our analysis will involve various steps such as data cleaning, feature engineering, and machine learning modeling.
# - By finishing this project, we will be having a good understanding on trends on data breachs from HIBP data and  we can make carefully decisions on counter measures
#
#
# ---
# ###
#
#



import pandas as pd
import numpy as np
import os, re



# plotting
import seaborn as sns
import matplotlib.pyplot as plt


# features for text processing
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer



# feature engineering..
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# split data
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GroupKFold

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


import warnings
warnings.filterwarnings('ignore')
# create a  stopwords that will be used later
all_stopwords = nltk.corpus.stopwords.words('english')


# In[2]:


# read the dataset
df = pd.read_csv("databreaches650.csv")



# check the sample of the dataset
df.sample(5)


# In[4]:


# check the coolumns available
df.columns


# In[5]:


# check the sample of the dataset dimension
df.shape


# In[6]:


# check the data types
df.dtypes


# In[7]:


# ## check the number of nulls present in the dataset
df.isna().sum()


# In[8]:


# check the distribution of domain column like how many instances are present in each domain
df.Domain.value_counts()[:25]


# In[9]:


# check the frequency of each record's name
df.Name.value_counts()[:5]


# In[10]:


# check the frequency of each record's Title
df.Title.value_counts()[:5]


# - There are a total of 650 records each with  about 16 features.
# - Most of the features are boolean and Strings.
# - We see that domain has `26 nulls`. This column is not going to be used for this process.
# - THis is because most of the records have its own unique domain. When observed as from above, the domain with the most number of records is `ogusers.com` and has 3 records while the others has 2 and 1.
# - In this process, we will just drop this column since it does not play a much important role in our analysis.
# - As observed above, also `Title, LogoPath and Name will also be droped.`. This is becuase, for each record, they appear to be unique hence some observable features cannot be done from them.

# In[11]:


# drop domain column
df.drop(columns=["Domain", "Name", "Title", "LogoPath"], inplace=True)


# In[12]:


# check the data remainings
df.head()


# ### Feature Engineering: Working on Dates.
# - On date features, we will extracts some other features like the time, day, week of the year or month, year etc.
# - These features will help in more analysis and coming up with more advanced analysis using machine learning.

# In[13]:


# function to create date features
def extract_date_features(data, col, update_index):
    """
    The function helps in extracting features from a date series and append them to the dataframe
    """
    #convert the col to a datetime
    x = pd.to_datetime(data[col])
    data[f'{update_index}_{col}_day'] = x.dt.day
    data[f'{update_index}_{col}_year'] = x.dt.year
    data[f'{update_index}_{col}_dayofweek'] = x.dt.dayofweek
    data[f'{update_index}_{col}_yearquater'] = x.dt.quarter
    data[f'{update_index}_{col}_month'] = x.dt.month
    data[f'{update_index}_{col}_Quater']  = x.dt.quarter

    if "breach" not in col.lower():
        data[f'{update_index}_{col}_hour'] = x.dt.hour
        data[f'{update_index}_{col}_minute'] = x.dt.minute


    #drop the column since it is not required now
    data.drop(col, axis=1, inplace=True)
    return data


# In[14]:


df.columns


# In[15]:


"breach" in "BreachDate".lower()


# In[16]:


#Appply the above function to the two columns for Project start and project end

# for Breach date
df = extract_date_features(df, "BreachDate", "breach")
# for Added date
df = extract_date_features(df, "AddedDate", "added")

# for modiefied date
df = extract_date_features(df, "ModifiedDate", "modified")


# In[17]:


print(df.head())



# save the results for data engineer 1.
df.to_csv("./data/save1.csv", index=False)

# check added Date features
print(df[[col for col in df.columns if "date" in col.lower()]].head())


# ### Observation.
# - As from the above process, using feature engieneering on 3 date features, we have been able to comeup with 22 features which we did not have.
# - These new features can help us in coming up with a more robust analysis and models.
#
#
#
# - In the next steps we will be carrying out Analysis using these features before we dive into other Feature engineering methods.

# ### 1. What Year did the most of the breachs OCCUR?
#
# - So for this, we will plot the number of breachs on each year.
# - This will help us in knowing the years that had most of the breach and least of the breach

# In[19]:
