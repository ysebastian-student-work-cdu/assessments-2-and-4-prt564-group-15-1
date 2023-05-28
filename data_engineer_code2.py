# import the code for plotting
from business_analysis import *


# 

# ### Feature Engineeering 2: Text Vectorization and Categorical Encoding.
# - THis process now aims to process data for machine learning.
# - We will convert all binary data to numerical and also vectorizer textual data from the description column

# In[40]:


df.head(2)


# ### 1. Encoding Binary Data To numerical.
# - Here all False values will be changed to 0 while True values will be changed to 1

# In[41]:


# get all columns that are numerical
binary_columns = ["IsVerified", "IsFabricated", "IsSensitive", "IsRetired", "IsSpamList", "IsMalware"]


# In[42]:


# check sample of them
df[binary_columns].head()


# In[43]:


# changing them to numerical
df[binary_columns].head().astype(int)


# In[44]:


# convert them to numerical
df[binary_columns] = df[binary_columns].astype(int)


# - FOr Data class column, We are only going to create a new feature called NumberOfDataClass which will count number of classes per record.
# - The DataClass will then be dropped after wards

# In[45]:


# get the number of data classes per record
df["NumDataClasses"] = df.DataClasses.apply(len)

# drop the data class column

df.drop(columns=["DataClasses"], inplace=True)


# In[46]:


# Now check the columns that remains which belongs to object data type
df.select_dtypes(include='object')


# ### 2. Text Vectorization.
# - As any machine Learning model accepts inputs as Numerical, but based on above, we are remaining with textual data which is still string.
# - We will do some data processing on it before converting it to numerical using TFIDF.
# - TFIDF uses Term Frequency and Inverse Document Frequency to get how important acertain word is to the document.
# -

# In[47]:


df.loc[645]["Description"]


# In[48]:


# I will create the function to perform the cleaning process.
# In this function, it will clean the text, remove stopwords and any words less than 3 characters 

# define lemmatizer object 
word_lemmatizer = WordNetLemmatizer()

def text_processing(text):
    """
    This function removes any stopwords in the text, pucntuations, words <3 characters etc
    and lastly it lemmatixe the word into its root form
    """
    #stemming object
    # stemmer = PorterStemmer()
    import string
    from html import unescape
    
    # remove HTML tags
    text = unescape(text)
    text = re.sub('<[^>]+>', '', text)
    

    #remove new lines escape characters
    text = re.sub(r"\n", "", text)
    
    # convert to lowercase
    text = text.lower()
    
    # remove non letters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # tokenize the text into words
    words = text.split()
    # remove stopwords
    #words = [w for w in words if w not in all_stopwords]
    words = [word_lemmatizer.lemmatize(w) for w in words if w not in all_stopwords and len(w)>3]
    # return a new sentense with the applied functions
    return " ".join(words)


# In[49]:


df.loc[645]["Description"]


# In[50]:


text_processing(df.loc[645]["Description"])


# In[51]:


# get the description column from the df
description = df["Description"]

# drop the dolumn from the dataframe
df.drop(columns=["Description"], inplace=True)


# In[52]:


# apply the function to the whole description column
description = description.apply(text_processing)


# ### Perform TFIDF Vectorization

# In[53]:


# create vectorizer object and fit it with description data
tfidf_vec = TfidfVectorizer()

tfidf_vec.fit(description)


# In[54]:


# transform the description to numerical using the vectorizer.
X_desc = tfidf_vec.transform(description).toarray()

