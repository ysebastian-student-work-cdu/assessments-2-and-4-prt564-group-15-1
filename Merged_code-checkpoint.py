
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

# In[1]:


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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


import warnings
warnings.filterwarnings('ignore')
# create a  stopwords that will be used later
all_stopwords = nltk.corpus.stopwords.words('english')


# read the dataset
df = pd.read_csv("databreaches650.csv")


# - There are a total of 650 records each with  about 16 features.
# - Most of the features are boolean and Strings.
# - We see that domain has `26 nulls`. This column is not going to be used for this process.
# - THis is because most of the records have its own unique domain. When observed as from above, the domain with the most number of records is `ogusers.com` and has 3 records while the others has 2 and 1.
# - In this process, we will just drop this column since it does not play a much important role in our analysis.
# - As observed above, also `Title, LogoPath and Name will also be droped.`. This is becuase, for each record, they appear to be unique hence some observable features cannot be done from them.

# In[11]:


# drop domain column
df.drop(columns=["Domain", "Name", "Title", "LogoPath"], inplace=True)


# ### Feature Engineering: Working on Dates.
# - On date features, we will extracts some other features like the time, day, week of the year or month, year etc.
# - These features will help in more analysis and coming up with more advanced analysis using machine learning.


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



#Appply the above function to the two columns for Project start and project end

# for Breach date
df = extract_date_features(df, "BreachDate", "breach")
# for Added date
df = extract_date_features(df, "AddedDate", "added")

# for modiefied date
df = extract_date_features(df, "ModifiedDate", "modified")


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


# plot the distribution of  total counts of breach by year
plt.figure(figsize=(10,7))
sns.countplot(x = df['breach_BreachDate_year'])
plt.xticks(fontweight='bold', rotation=50)
plt.title("DIstribution Of Breachs By Year", fontweight='bold', fontsize=18)
plt.xlabel("Year of Breach Occurence ", fontweight='bold', fontsize=18)
plt.ylabel("Frequency Distribution", fontweight='bold', fontsize=18)


# ### Observation.
# - The information stored is from the year 2007 upto current year.
# - As from the above, It is clearly observed that, data breaches occured mostly in the year 2015, 2016, 2018 and 2020 based on provided data. These were the years with most of the breaches
# - We cannot say that 2023 has the least items since it is still quarterway. In this case, we can Ignore 2007 since it also might be data for half the year. In this case, we therefore observe 2009 as the year with least data breaches.

# ### Among All years Observed, What Year had Most of the Breachs Occuring as Malware?

plt.figure(figsize=(14,7))
sns.countplot(x =df['breach_BreachDate_year'], hue= df.IsMalware)
plt.xticks(fontweight='bold', rotation=50)
plt.title("DIstribution Of Breachs By Year", fontweight='bold', fontsize=18)
plt.xlabel("Year of Breach Occurence ", fontweight='bold', fontsize=18)
plt.ylabel("Frequency Distribution", fontweight='bold', fontsize=18)


# ### Observation.
# - Only 3 years has with data breachs that were malware that are observable.
# - THese years are 2018, 2020 and 2021.

# ### Can we be able to determine which Month was the most Occurence Of Data Breach Happened?

plt.figure(figsize=(12,5))
un, count = np.unique(df.breach_BreachDate_month.values.astype('int32'),return_counts=True)
# un, count = zip(*sorted(zip(un, count)))
plt.title("Breach By Month", fontsize=16, fontweight='bold')
plt.xlabel("Month Number of THe Year", fontsize=15, fontweight='bold')
plt.ylabel("Counts", fontsize=15, fontweight='bold')
sns.countplot(x = df.breach_BreachDate_month)
plt.xticks(range(len(un)), ['Jan',"Feb","March","April","May","June","July","Aug","sept","Oct","Nov","Dec"], fontsize=15,  fontweight='bold', rotation=50)
plt.show()


# ### Observation.
# - Based on above visualization, It is clear that the month of January has the most of breaches followed by Feb, May and then December.
# - September appears to be the month with the least breaches.

# ### Based the Total number of user Accounts Breached, what was the Distribution of these numbers?
# - This is the  PwnCount attribute which stores the total number of breached user account. This is usually less than the total number reported by the media due to duplication or other data integrity issues in the source data.

# In[22]:


# Check how different months had different pawned accounts
plt.figure(figsize=(14,5))
sns.barplot(data=df,x='breach_BreachDate_month',y='PwnCount',palette='OrRd')
plt.xticks(rotation=45)
plt.title("User's Accounts Pawned Against Each Month of the Year", fontsize=17, c='g')
plt.xticks(range(len(un)), ['Jan',"Feb","March","April","May","June","July","Aug","sept","Oct","Nov","Dec"], fontsize=15,  fontweight='bold', rotation=50)
plt.xlabel("Accounts Pawned Against Month", fontsize=15, fontweight='bold')
plt.ylabel("Average Pawned Accounts", fontsize=15, fontweight='bold')
plt.show()


# ### Observation.
# - October had the largest average number of accounts Pawned followed by July then February.
# -  March Appears to have the least number of accounts pawned..
#
# - Organizations can therefore be more vigilant on september to avoid data breachs as it appears to have the largest number of accounts.


plt.figure(figsize=(10,7))
sns.boxplot(data=df, x='PwnCount', palette=['#00876c']);
plt.title("Pawned Accounts Counts Distriution" , c ='r', fontweight='bold')
plt.text(80,0.23,"     Mean value: {:.2f}".format(df.PwnCount.mean()))
plt.text(80,0.29,"     Median value: {:.2f}".format(df.PwnCount.median()))
plt.text(80,0.35,"     Frequent PwnCount : {:.2f}".format(df.PwnCount.mode().max()))


# ### Observation.
# - Total number of Accounts Pawned appears not to be equally distributed.
# - There are plenty of outliers. Most of values are concentrated between 0e8 to 1e8 with the mean about 19million

# ### By using the day of the week, when was the most of the Breach Occuring?.


# analysis by week
day_df =df.groupby(["breach_BreachDate_dayofweek", "IsMalware"]).mean(numeric_only=True).reset_index()
fig = plt.figure(figsize=(12,7))
g = sns.catplot(data=day_df,
            x='breach_BreachDate_dayofweek',
            y='PwnCount',
            kind='bar',
            hue='IsMalware',
            height=6,
            aspect=1.5,
            palette='viridis',
            ax=fig)

plt.title('Day By Weekly Number of Breachs By Malware',color='r', fontsize=20);
g.set_xticklabels(["Mon", "Tue", "Wed", "Thur","Fri", "Sart", "Sun"], rotation=50, fontweight='bold')


# ### Observation.
# - Mondat and Thursday has most of the Breaches.
# - Breaches that are Malware appears to might be occuring only on Wenesday and Thursday with Thursday having the most of the Malware Breachs
# ### What Percentage of Data Breach Were Sensitive?
# - This can help see, if most of the targets were having some sensitive information


# explode = (0.1,0)
plt.figure(figsize=(5,4))
explode = (0, 0.1)
plt.pie(df['IsSensitive'].value_counts() ,
        labels = df['IsSensitive'].value_counts().index.tolist(),
        autopct='%1.1f%%',
        explode = (0, 0.1)  ,
        shadow=True, startangle=90)
plt.title("DIstribution of of Information that was Sensitive Among the Breachs" , c='red' , fontsize =16)
# draw as an equal circle
plt.axis('equal')
plt.tight_layout()
plt.legend()
plt.show()


# ### Observation.
# - Only few information are sensitive among the breaches.
# - It caters for about 8% of all.

# ### Were the Information Already In spamList? What Percentage?

# explode = (0.1,0)
plt.figure(figsize=(5,4))
explode = (0, 0.1)
plt.pie(df['IsSpamList'].value_counts() ,
        labels = df['IsSpamList'].value_counts().index.tolist(),
        autopct='%1.1f%%',
        explode = (0, 0.1)  ,
        shadow=True, startangle=90)
plt.title("DIstribution of of Information that was In SpamList Among the Breachs" , c='red' , fontsize =16)
# draw as an equal circle
plt.axis('equal')
plt.tight_layout()
plt.legend()
plt.show()


# ### Observation.
# - About 2% of the data breaches were in spam list.
# - This indicates that most of them are not spam hence legit information that is not blocked by spamlist

# ### Was the Breach Done to verified Informations? What Percentage was this?

# explode = (0.1,0)
plt.figure(figsize=(5,4))
explode = (0, 0.1)
plt.pie(df['IsVerified'].value_counts() ,
        labels = df['IsVerified'].value_counts().index.tolist(),
        autopct='%1.1f%%',
        explode = (0, 0.1)  ,
        shadow=True, startangle=90)
plt.title("DIstribution of of Information that was Verified Among the Breachs" , c='red' , fontsize =16)
# draw as an equal circle
plt.axis('equal')
plt.tight_layout()
plt.legend()
plt.show()


# ### Observation.
# - Only about 6% of the breaches was not verified
#
# ### Among the Malwares, How was the Disribution for those information breachs that were Fabricated


#type of contact and target
plt.figure(figsize=(8,7))
plt.title("Were Information Fabricated or Not")
sns.countplot(x = df['IsFabricated'] , hue =df['IsMalware'])



df[["IsFabricated", "IsMalware"]].value_counts()


# ### Observation.
# - Most of the information is not Fabricated hence comes as the original Information.
# - Among those Fabricated, there is not any of them that is considered malware

# - From above, we have mostly dealt with the Breachs By date.
# - Below, we will check the various data classes that were available in the information that was breached.
# - By this, we are looking on the Data Classes Columns.
# - Since each record has a list, We will split each of them and count them
#
# ### What We the common Data Classes Used in the data collected for Pawned Accounts?

# In[31]:


# df.DataClasses.apply(len)
df.DataClasses.values[0]


# In[32]:


# this library will help get the actual list of values
import ast
# test it
df.DataClasses.head().apply(ast.literal_eval)



# Get the actual Lsit of of values from it
df['DataClasses']  = df.DataClasses.apply(ast.literal_eval)


# plot the distribution of the number of classes per record
plt.figure(figsize=(12,7))
sns.displot(x = df.DataClasses.apply(len), kde=True)
plt.title("DIstribution of Number of Data Classes Among the Breach Records", fontsize=16, fontweight='bold')


# ### Observation.
# - Most of the records has between 2 and 7 data classes.
# - There are some few of them that contains more than 20 data classes.
# get the number of occurance per Item from the List
df['DataClasses'].explode().value_counts()[:40]


# g
# get the top 40 items and count their occurrences from the list
top_data_classes = df["DataClasses"].explode().value_counts().nlargest(40)


# In[39]:


sns.set(style="darkgrid")
plt.figure(figsize=(10, 15))
# sns.countplot(y=df["DataClasses"].explode()[:40],  order = df['DataClasses'].explode().value_counts().index)
sns.countplot(y=df["DataClasses"].explode(), order=top_data_classes.index, palette='viridis')
plt.title("TOP 40 Data Classes APpearing On the HIBP Breach Data", fontsize=14, fontweight='bold')
plt.xlabel("Number Of OCcurence Count", fontsize=12, fontweight='bold')
plt.ylabel("Data Class", fontsize=12, fontweight='bold')
plt.show()


# ### Observation.
# - As from the above, `Email Address, Usernames, Names and Ip addresses` were among the common most data classes in the breachs


# ### Feature Engineeering 2: Text Vectorization and Categorical Encoding.
# - THis process now aims to process data for machine learning.
# - We will convert all binary data to numerical and also vectorizer textual data from the description column


# ### 1. Encoding Binary Data To numerical.
# - Here all False values will be changed to 0 while True values will be changed to 1

# get all columns that are numerical
binary_columns = ["IsVerified", "IsFabricated", "IsSensitive", "IsRetired", "IsSpamList", "IsMalware"]



# changing them to numerical
df[binary_columns].head().astype(int)

# convert them to numerical
df[binary_columns] = df[binary_columns].astype(int)


# - FOr Data class column, We are only going to create a new feature called NumberOfDataClass which will count number of classes per record.
# - The DataClass will then be dropped after wards


# get the number of data classes per record
df["NumDataClasses"] = df.DataClasses.apply(len)

# drop the data class column

df.drop(columns=["DataClasses"], inplace=True)


# Now check the columns that remains which belongs to object data type
df.select_dtypes(include='object')


# ### 2. Text Vectorization.
# - As any machine Learning model accepts inputs as Numerical, but based on above, we are remaining with textual data which is still string.
# - We will do some data processing on it before converting it to numerical using TFIDF.
# - TFIDF uses Term Frequency and Inverse Document Frequency to get how important acertain word is to the document.
# -


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


df.loc[645]["Description"]


text_processing(df.loc[645]["Description"])


# get the description column from the df
description = df["Description"]

# drop the dolumn from the dataframe
df.drop(columns=["Description"], inplace=True)



# apply the function to the whole description column
description = description.apply(text_processing)


# ### Perform TFIDF Vectorization

# create vectorizer object and fit it with description data
tfidf_vec = TfidfVectorizer()

tfidf_vec.fit(description)


# In[54]:


# transform the description to numerical using the vectorizer.
X_desc = tfidf_vec.transform(description).toarray()



pd.DataFrame(X_desc).head(5)


# ### Observation.
# - As from above, we are able to see that from the textual feature, we have created 2901 other features.
# - These features are mostly sparse matrix as it contains most zeros.
# - We will merge the above data and then we will perform dimension reduction on them.
#
#
#
# - In this project we will be trying to predict whether breach is considered unverified.
# - An unverified breach may not have been hacked from the indicated website. An unverified breach is still loaded into HIBP when there's sufficient confidence that a significant portion of the data is legitimate.
#


final_df = pd.concat([df, pd.DataFrame(X_desc)], axis=1)

# Extract the Label and Training Features
X = final_df.drop("IsVerified", axis=1)
Y = final_df.IsVerified.values


# ### Preliminary Random Forest classifier with Original Dataset.
# - At first we will start by scalling the dataset in order to normalize them
# - I will split the data into twice, i.e Training set to be 70% while the other 30% as test data


# scaling the dataset
from sklearn.preprocessing import StandardScaler

# define scaler object
scaler = StandardScaler()
# scaler the dataset
X = scaler.fit_transform(X)



# split the data into train and test
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y, test_size=0.3, random_state=42, stratify=Y)


# define the Random forest classifier object
random_clf1 = RandomForestClassifier(class_weight='balanced', random_state=42)
# train the Random forest classifier model
random_clf1.fit(Xtrain, Ytrain)
# get the general score for Random forest classifiermodel
random_clf1.score(Xtest, Ytest)


# get the score of the model
random_clf1.score(Xtest, Ytest)

# define a function to run evaluation on the model on 4 basic scoring metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
def evaluate_model_perfomance(Ytrue, Ypred, model_name):
    print("Accuracy Score is  {:4f}".format(accuracy_score(Ytrue, Ypred)))
    print("F1 Score is  {:4f}".format(f1_score(Ytrue, Ypred, average='weighted')))
    print("Recall Score is  {:4f}".format(recall_score(Ytrue, Ypred, average='weighted')))
    print("Precision Score is  {:4}".format(precision_score(Ytrue, Ypred, average='weighted')))
    print()
    #get the heat map plot of the model predictions on test set
    cm = confusion_matrix(Ytrue, Ypred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt='', cbar=False, cmap="Blues")
    plt.title(f"Confusion Matricx for  {model_name}", fontweight='bold', fontsize =16)
    plt.xlabel("True Labels", fontweight='bold', fontsize =16)
    plt.ylabel("Predicted Output", fontweight='bold', fontsize =16)



# run the function for Evaluation
evaluate_model_perfomance(Ytest, random_clf1.predict(Xtest), "Base Random Forest Classifier")


# ### Observation.
# - We have achieved and F1 score of about 98% where we have 5 mis clasified labels among the test labels

# ### Data Dimension Reduction.
# - We will use PCA to reduce the dimension of the dataset into x dimensions which are suitable and contain the atleast 70% of the information.
#


# in order to work well , try reducing the dimension of the data points
# Apply PCA by fitting the data with the same number of dimensions as features


pca = PCA(random_state=42)
pca.fit(X)

#transform the df using the PCA fit above
pca_df = pca.transform(X)



#  - To observe the information stored by each principal components, A cumulative explained variance will be used to observe the percentage for each as below.

# get explained variance for components
for i in range(pca.n_components_):
    first_n = pca.explained_variance_ratio_[0:i+1].sum()*100

    if i % 50 == 0:
        print(f'Percent variance explained by first {i+1} components: {round(first_n , 4)}%')



# ### Observation.
# - It can be seen that the majority of the variance in our data over (90%) can be encoded in atleast first 540 principal components 7 of our 2930 dimensions.
# - We will therefore decompose the whole features into first 500 features and use it for model training and dimension reduction.


# reduccing the data to the 500 dimensions

pca = PCA(n_components=500, random_state=42)
pca.fit(X)


# Transform all the data into the required data sizes
Xtrain_transformed = pca.transform(Xtrain)
Xtest_transformed = pca.transform(Xtest)


# ### Retrain the Random forest classifier Model Again with this data.

# ### 2. Test Using a Tree Based Model.
# - In this case, I will use Random forest classifier Model to observe how the model performs on the same dataset.

#
# ### 3. Random FOrest Classifier.

# define the Random forest classifier object
random_clf = RandomForestClassifier(class_weight='balanced', random_state=42)
# train the Random forest classifier model
random_clf.fit(Xtrain_transformed, Ytrain)
# get the general score for Random forest classifiermodel
random_clf.score(Xtest_transformed, Ytest)


# Run evaluation on the test dataset for the model
evaluate_model_perfomance(Ytest, random_clf.predict(Xtest_transformed), "Random Classifier  Model")


# ### Observation.
# - The model did not any false negative but only 5 false positive.

# ### 4. Naive Bayes Algorithm

# define the naive bayes using gausian nb object
bayes_clf = GaussianNB()

# train the naive bayes model
bayes_clf.fit(Xtrain_transformed, Ytrain)

# get the general score for naive bayes
bayes_clf.score(Xtest_transformed, Ytest)


# Run evaluation on the test dataset for the model
evaluate_model_perfomance(Ytest, bayes_clf.predict(Xtest_transformed), "Naive Bayes Model")


# ### Observation.
# - It had about 30 false positive and 4 false negative.
# - The model had a lower porfomance of about 93%.

# ### 5. SVM algorithms


# define the SVM object
svm_clf = SVC(class_weight='balanced', random_state=42)

# train the svm model
svm_clf.fit(Xtrain_transformed, Ytrain)

# get the general score for svm based model
svm_clf.score(Xtest_transformed, Ytest)


# Run evaluation on the test dataset for the model
evaluate_model_perfomance(Ytest, svm_clf.predict(Xtest_transformed), "SVM  Model")


# ### Observation.
# - Perfomed similar to randomforest model with a score of about 97%.


# ### Data CLustering using Kmeans.
# - In this case, we will use Kmeans to cluster data based on their characteristics.
#
#
# - Clustering algorithms will be used to group these customers datapoints into various segments based on their similarities.
# - AN analytical methods will be used to determine the number of groups to be used.
#
#
# ### Steps for Analytical Methods.
# - Fit clustering algorithm with various number of k groups i,e use a range from 2 to 20.
# - Using elbow curve method and sum of squared distance errors, determine the best or rather the best range of k values using a plot.
# - Use the determined number of clusters to determine the best values.
#
# A list holds the SSE values for each k
sse = []
for k in range(2, 20):
   kmeans = KMeans(n_clusters=k)
   kmeans.fit(pca.transform(X))
   sse.append(kmeans.inertia_)

# plot the elbow curve
plt.plot(range(2, 20), sse)
plt.xticks(range(2, 20))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()


# ### Observation
# - From the curve above, there is no observable values that can be used to identify the optimum number of clusters.
# - In this case, we will therefore use the original 2 classes for IsVerified column.
# - For easier analysis, both 2 clusters are going to be used in the analysis.

# Clustering with KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
# Fit KMeans model to reduced feature data
kmeans.fit(pca.transform(X))


# get the clusters labels for original dataframe for the clusters
cluster_labels = kmeans.labels_



# Compaire the Clustered Data to the Original Labels
evaluate_model_perfomance(Y, cluster_labels, "Clustering Results")


# ### Observation
# - As It can be seen, the clustering was able to group the data into two classes although not accurated for the class labelled 0.
# - Most of the values were classfied as 1 as seen above.

# ---
# ### COnclusion.
# - In conclusion, this project has demonstrated how to process data for machine learning, work with date features for feature engineering, Perform Analytical analysis from attributes using charts and graphs, clean textual data, perform textual data vectorization, clustering and classification, and evaluate the performance of the models using confusion matrix for imbalanced classes.
# - Through these steps, we were able to gain insights from the HIBP dataset and identify patterns and trends in cyber attacks.
# - The results obtained from the analysis can be used for further decision making, such as identifying when do most data breaches occur, data classes which are mostly targeted etc that are most susceptible to cyber attacks and what measures can be taken to prevent them.
# - Overall, this project has shown the importance of data preprocessing and analysis in the field of cybersecurity, and the potential impact it can have on the industry.
#
# ---




# #### PERFOMING CLROSS VALIDATION AND HYPER PARAMETER TUNING.

# In[80]:


# transform the datasets
Xdata = pca.transform(X)
Ydata = Y.copy()


# In[81]:


# get the data we have
print("DATA SHAPES ARE  : ", Xdata.shape, Ydata.shape)


# ### FUnction to perfom the task,.
# - In this case, I will create a fuction to takes the models, itparam and data to be used.
# - The function will first split the data into 80% and 20% splits and then after that It will perfom a hypa-parameter tuning with 5 folds and using gridsearchCV.
# - Since we are dividing the data into 5 folds, 4 of the folds will be used for training i.e 80% of the dataset while the other 1 fold or 20% will be used for evaluation,
# - The function then returns the best estimator with the best parameters selected to be used to evaluate the model for comparison..
# 
# - Here is the implimentation of the function


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report



# We fisrt need to split the data into 80:20 rule as suggested before starting the tuning methods..

X_train, X_test, Y_train, Y_test = train_test_split(Xdata, Ydata, test_size=0.2, random_state=42)

print(f"New shapes for Training are {X_train.shape} , {Y_train.shape}")
print(f"New shapes for Testing are {X_test.shape} , {Y_test.shape}")
    
def kfold_cross_validation(X, Y, model, params):
    # split the data into 80% training and 20% testing sets
    
    # Perform k-fold cross-validation
    cv = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)  # You can change the scoring metric as needed
    
    # Fit the model on the training data
    cv.fit(X,Y)
    return cv


# 
# ### Lets now do the TUning on the 3 models we trained above.
# 
# 
# ### NOTE:  
# - For Naive Bayes, We will not do hyperameter tuning since the model is not parametric and is based on probs hence we will only do for the two models i.e SVM and RANDOM CLASSIFIER>...
# 
# #### 1. Random Forest Classifier.


# lets define parameters for  grid with random forest classifier
rf_params = {
    'n_estimators': [50, 100,250, 500],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}



# create the classifier with gridsearchCV
from sklearn.ensemble import RandomForestClassifier

# we call the  function for RF optimization of features
grid_rf_model = kfold_cross_validation(X_train, Y_train, RandomForestClassifier(random_state=42), rf_params)


# get to print the best parameterst

print(f"Best Parameters for Random FOrest Classifier are :\n\t    {grid_rf_model.best_params_}")


# get the best estimator and evaluate it...
grid_rf_model.best_estimator_



# evaluate random forest model witht the best model gotten..

print("Here are the results evaluation of the random forst model with the best model after tuning")
evaluate_model_perfomance(Y_test, grid_rf_model.best_estimator_.predict(X_test), "Tuned Random forest  Model")



# ### 2. Suppport Vector Machine ((SVM) with Hyperparameter tuning.
# - Here is the code for tuning of SVM model..


# define parameters for svm...
svm_params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': [0.1, 1, 10]
}



# lets create the svm object and run it with the grid classfier..

from sklearn.svm import SVC

# call the validation and tuning function
grid_svm_model = kfold_cross_validation(X_train, Y_train, SVC(class_weight='balanced', random_state=42), svm_params)



# get the best parameter for svm..
print(f"Best Parameters for Random SVM Classifier are :\n\t    {grid_svm_model.best_params_}")


# evaluate SVM model witht the best model gotten..

print("Here are the results evaluation of the SVM model with the best model after tuning")
evaluate_model_perfomance(Y_test, grid_svm_model.best_estimator_.predict(X_test), "Tuned SVM Model")

