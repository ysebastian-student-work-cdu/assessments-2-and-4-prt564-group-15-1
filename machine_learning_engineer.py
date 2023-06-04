# import the code from data engineer
from data_engineer_code2 import *

print("I'm THE MACHINE LEARNING ENGINEER STARTING")
print(X_desc[:4, :10])


# In[56]:


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


# check their shapes
X.shape, Y.shape


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


# define a function to run evaluation on the model on 4 basic scoring metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
def evaluate_model_perfomance(Ytrue, Ypred, model_name):
    print(f"Outputing Results for model  {model_name}")
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
    plt.savefig(f"./imgs/{model_name}.png")
    plt.show()


# define the Random forest classifier object
random_clf1 = RandomForestClassifier(class_weight='balanced', random_state=42)
# train the Random forest classifier model
random_clf1.fit(Xtrain, Ytrain)
# get the general score for Random forest classifiermodel
random_clf1.score(Xtest, Ytest)



# Run evaluation on the test dataset for the model
evaluate_model_perfomance(Ytest, random_clf1.predict(Xtest), "Random Classifier  Model On Original Data")



# ### Observation.
# - We have achieved and F1 score of about 98% where we have 4 mis clasified labels among the test labels

# ### Data Dimension Reduction.
# - We will use PCA to reduce the dimension of the dataset into x dimensions which are suitable and contain the atleast 70% of the information.
# 

# in order to work well , try reducing the dimension of the data points
# Apply PCA by fitting the data with the same number of dimensions as features


pca = PCA(random_state=42)
pca.fit(X)

#transform the df using the PCA fit above
pca_df = pca.transform(X)


pca.n_components_


#  - To observe the information stored by each principal components, A cumulative explained variance will be used to observe the percentage for each as below.


# get explained variance for components
for i in range(pca.n_components_):
    first_n = pca.explained_variance_ratio_[0:i+1].sum()*100
    
    if i % 50 == 0:
        print(f'Percent variance explained by first {i+1} components: {round(first_n , 4)}%')
    


# ### Observation.
# - It can be seen that the majority of the variance in our data over (90%) can be encoded in atleast first 540 principal components 7 of our 2930 dimensions. 
# - We will therefore decompose the whole features into first 500 features and use it for model training and dimension reduction.

# In[69]:


# reduccing the data to the 500 dimensions

pca = PCA(n_components=500, random_state=42)
pca.fit(X)


# In[70]:


# Transform all the data into the required data sizes
Xtrain_transformed = pca.transform(Xtrain)
Xtest_transformed = pca.transform(Xtest)




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

# In[ ]:





# ### 4. Naive Bayes Algorithm

# In[78]:


# define the naive bayes using gausian nb object
bayes_clf = GaussianNB()

# train the naive bayes model
bayes_clf.fit(Xtrain_transformed, Ytrain)

# get the general score for naive bayes
bayes_clf.score(Xtest_transformed, Ytest)


# In[79]:


# Run evaluation on the test dataset for the model
evaluate_model_perfomance(Ytest, bayes_clf.predict(Xtest_transformed), "Naive Bayes Model")


# ### Observation.
# - It had about 30 false positive and 4 false negative.
# - The model had a lower porfomance of about 93%.

# In[ ]:





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

# In[ ]:








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
plt.savefig("./imgs/sse.png")
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

# In[ ]:






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

