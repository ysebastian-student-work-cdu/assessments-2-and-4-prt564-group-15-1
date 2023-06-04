# import the code for data engineer
from data_engineer_code1 import *


# <<<<PART FOR DATA ANALYSIS  <<<BUSINESS_ANALYSIT>>>
# plot the distribution of  total counts of breach by year
plt.figure(figsize=(10,7))
sns.countplot(x = df['breach_BreachDate_year'])
plt.xticks(fontweight='bold', rotation=50)
plt.title("DIstribution Of Breachs By Year", fontweight='bold', fontsize=18)
plt.xlabel("Year of Breach Occurence ", fontweight='bold', fontsize=18)
plt.ylabel("Frequency Distribution", fontweight='bold', fontsize=18)
plt.savefig("./imgs/breach_BreachDate_year_count.png")
plt.show()


# ### Observation.
# - The information stored is from the year 2007 upto current year.
# - As from the above, It is clearly observed that, data breaches occured mostly in the year 2015, 2016, 2018 and 2020 based on provided data. These were the years with most of the breaches
# - We cannot say that 2023 has the least items since it is still quarterway. In this case, we can Ignore 2007 since it also might be data for half the year. In this case, we therefore observe 2009 as the year with least data breaches.

# ### Among All years Observed, What Year had Most of the Breachs Occuring as Malware?

# In[20]:


plt.figure(figsize=(14,7))
sns.countplot(x =df['breach_BreachDate_year'], hue= df.IsMalware)
plt.xticks(fontweight='bold', rotation=50)
plt.title("DIstribution Of Breachs By Year", fontweight='bold', fontsize=18)
plt.xlabel("Year of Breach Occurence ", fontweight='bold', fontsize=18)
plt.ylabel("Frequency Distribution", fontweight='bold', fontsize=18)
plt.savefig("./imgs/breach_BreachDate_year_Malware.png")
plt.show()

# ### Observation.
# - Only 3 years has with data breachs that were malware that are observable.
# - THese years are 2018, 2020 and 2021.

# ### Can we be able to determine which Month was the most Occurence Of Data Breach Happened?

# In[21]:


plt.figure(figsize=(12,5))
un, count = np.unique(df.breach_BreachDate_month.values.astype('int32'),return_counts=True)
# un, count = zip(*sorted(zip(un, count)))
plt.title("Breach By Month", fontsize=16, fontweight='bold')
plt.xlabel("Month Number of THe Year", fontsize=15, fontweight='bold')
plt.ylabel("Counts", fontsize=15, fontweight='bold')
sns.countplot(x = df.breach_BreachDate_month)
plt.xticks(range(len(un)), ['Jan',"Feb","March","April","May","June","July","Aug","sept","Oct","Nov","Dec"], fontsize=15,  fontweight='bold', rotation=50)
plt.savefig("./imgs/Breach_By_Mont.png")
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
plt.savefig("./imgs/breach_BreachDate_month_pwnCount.png")
plt.show()


# ### Observation.
# - October had the largest average number of accounts Pawned followed by July then February.
# -  March Appears to have the least number of accounts pawned..
# 
# - Organizations can therefore be more vigilant on september to avoid data breachs as it appears to have the largest number of accounts.

# In[23]:


plt.figure(figsize=(10,7))
sns.boxplot(data=df, x='PwnCount', palette=['#00876c']);
plt.title("Pawned Accounts Counts Distriution" , c ='r', fontweight='bold')
plt.text(80,0.23,"     Mean value: {:.2f}".format(df.PwnCount.mean()))
plt.text(80,0.29,"     Median value: {:.2f}".format(df.PwnCount.median()))
plt.text(80,0.35,"     Frequent PwnCount : {:.2f}".format(df.PwnCount.mode().max()))
plt.savefig("./imgs/PwnCount.png")
plt.show()


# ### Observation.
# - Total number of Accounts Pawned appears not to be equally distributed.
# - There are plenty of outliers. Most of values are concentrated between 0e8 to 1e8 with the mean about 19million

# ### By using the day of the week, when was the most of the Breach Occuring?.

# In[24]:


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
plt.savefig("./imgs/breach_BreachDate_dayofweek_isMalware.png")
plt.show()

# ### Observation.
# - Mondat and Thursday has most of the Breaches.
# - Breaches that are Malware appears to might be occuring only on Wenesday and Thursday with Thursday having the most of the Malware Breachs

# In[25]:


df.head(1)


# ### What Percentage of Data Breach Were Sensitive?
# - This can help see, if most of the targets were having some sensitive information

# In[26]:


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
plt.savefig("./imgs/IsSensitive.png")
plt.show()


# ### Observation.
# - Only few information are sensitive among the breaches.
# - It caters for about 8% of all.

# ### Were the Information Already In spamList? What Percentage?

# In[27]:


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
plt.savefig("./imgs/IsSpamList.png")
plt.show()


# ### Observation.
# - About 2% of the data breaches were in spam list.
# - This indicates that most of them are not spam hence legit information that is not blocked by spamlist

# ### Was the Breach Done to verified Informations? What Percentage was this?

# In[28]:


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
plt.savefig("./imgs/IsVerified.png")
plt.show()


# ### Observation.
# - Only about 6% of the breaches was not verified
# 
# ### Among the Malwares, How was the Disribution for those information breachs that were Fabricated

# In[29]:


#type of contact and target
plt.figure(figsize=(8,7))
plt.title("Were Information Fabricated or Not")
sns.countplot(x = df['IsFabricated'] , hue =df['IsMalware'])
plt.savefig("./imgs/IsFabricated.png")
plt.show()

# In[30]:


print(df[["IsFabricated", "IsMalware"]].value_counts())

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


# In[33]:


# test it
df.DataClasses.head().apply(ast.literal_eval)


# In[34]:


# Get the actual Lsit of of values from it
df['DataClasses']  = df.DataClasses.apply(ast.literal_eval)


# In[35]:


print(df.head(1))


# In[36]:


# plot the distribution of the number of classes per record
plt.figure(figsize=(12,7))
sns.displot(x = df.DataClasses.apply(len), kde=True)
plt.title("DIstribution of Number of Data Classes Among the Breach Records", fontsize=16, fontweight='bold')
plt.savefig("./imgs/DataClasses.png")
plt.show()

# ### Observation.
# - Most of the records has between 2 and 7 data classes.
# - There are some few of them that contains more than 20 data classes.

# In[37]:


# get the number of occurance per Item from the List
df['DataClasses'].explode().value_counts()[:40]


# In[38]:


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
plt.savefig("./imgs/DataClasses_occurentce.png")
plt.show()


# ### Observation.
# - As from the above, `Email Address, Usernames, Names and Ip addresses` were among the common most data classes in the breachs

# In[ ]:

# save the data for business analysis
df.to_csv("./data/saved.csv", index=False)