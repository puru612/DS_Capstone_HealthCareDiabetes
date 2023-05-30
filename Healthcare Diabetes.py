#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[2]:


#loading the dataset
data = pd.read_csv('health care diabetes.csv')


# In[3]:


#exploring the data
data.head()


# In[4]:


data.shape


# In[5]:


data.isna().sum()


# In[6]:


#exploring the value counts for each variable
for i in data.columns:
    print(i, ':', data[i].value_counts())
    print('-'*50)


# In[7]:


# visually exploring the variables using histograms
variables = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for i in variables:
    sns.histplot(data=data[i])
    plt.show()


# In[8]:


data.describe().T


# In[9]:


# treating the missing values with the mean value of each variable
variables = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for i in variables:
    data[i].replace(0,data[i].mean(),inplace=True)
   # data[i].fillna(data[i].median(),inplace=True)


# In[10]:


#saving the dataset after cleaning all the missing values
data.to_excel("D:\Puru\Data-Science-Capstone-Projects-master\Project_2\Project 2\Healthcare - Diabetes/health care diabetes.xlsx")


# In[11]:


# creating a count plot describing the data types and the count of variables
sns.countplot(data.dtypes)


# In[12]:


# Checking the balance of the data by plotting the count of outcomes by their value.
positive=data[data['Outcome']==1]
positive.hist(figsize=(12,12))


# We can observe that there are some outliers in SkinThickness, Insulin, BMI and DiabetesPedigreeFunction. Also, it can be seen that majority of the people tested positive for Diabetes are between the Age Group of 20-45 and have low SkinThickness with Glucose levels above 100.

# In[13]:


# Creating scatter charts between the pair of variables to understand the relationships.
sns.pairplot(data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age','Outcome']],data=data,hue='Outcome')
plt.show()


# The three major observations that can be drawn upon looking at the scatter plots above are:
# 1. A higher glucose results in higher chances of developing diabetes
# 2. SkinThickness and BMI have high correlation.
# 3. More than 12 pregnancies have resulted almost everytime in developing diabetes.

# In[14]:


# analyzing the correlation between variables
data.corr()


# In[15]:


# plotting the correlation between variables using a heatmap
sns.heatmap(data=data.corr(), annot=True)


# #### Machine Learning model building ####

# In[16]:


x=data.drop('Outcome',axis=1)
y=data.Outcome


# In[17]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[18]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x=scaler.fit_transform(x)


# In[19]:


from sklearn.model_selection import train_test_split as tst
xtrain,xtest,ytrain,ytest=tst(x,y,random_state=10, test_size=0.2)


# In[20]:


print(xtrain.shape)
print(xtest.shape)


# In[21]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[22]:


logreg=LogisticRegression()


# In[23]:


logreg.fit(xtrain,ytrain)


# In[24]:


logreg_pred=logreg.predict(xtest)


# In[25]:


print(metrics.classification_report(ytest,logreg_pred))


# In[26]:


print('Logistic Regression Score: {}'.format(logreg.score(xtrain,ytrain)))
print('Logistic Regression Accuracy Score: {}'.format(accuracy_score(ytest,logreg_pred)))


# In[27]:


#checking the accuracy and Cross Validation Scores of each classifier model
accuracyscores=[]
modelscores=[]
models=[]
names=[]
cvscores=[]
models.append(('LogReg', LogisticRegression()))
models.append(('SVC', SVC()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('XGB', XGBClassifier()))
models.append(('KNN', KNeighborsClassifier()))


# In[28]:


for name, model in models:
    model.fit(xtrain,ytrain)
    modelscores.append(model.score(xtrain,ytrain))
    ypred=model.predict(xtest)
    accuracyscores.append(accuracy_score(ytest,ypred))
    kfold=KFold(n_splits=10)
    score=cross_val_score(model, x, y, cv=kfold, scoring='accuracy').mean()
    cvscores.append(score)
    names.append(name)

models_comp = pd.DataFrame({'Name':names,'Model Score':modelscores,'Accuracy Score':accuracyscores,'Cross Validation Score':cvscores})
print(models_comp)


# In[29]:


# plotting the accuracy and cross validation score of each model
fig, axes = plt.subplots(1,2,sharey=True,figsize=(12,7))
viz=sns.barplot(x='Name',y='Accuracy Score',data=models_comp, ax=axes[0])
viz.set(xlabel='Classifier Name', title='Accuracy Score')

for p in viz.patches:
    height = p.get_height()
    viz.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.3f}'.format(height), ha='center' )

viz2=sns.barplot(x='Name',y='Cross Validation Score',data=models_comp, ax=axes[1])
viz2.set(xlabel='Classifier Name', title='Cross Validation Score')

for p in viz2.patches:
    height = p.get_height()
    viz2.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.3f}'.format(height), ha='center' )


# We can observe that Logistic Regression, Random Forest, Decision Tree and XGBoost have performed better than others. And among all, Logistic Regression has performed the best with an accuracy of 73.4% and cross validation score of 0.776.
# KNN Classifier has not performed well compared to other models.

# In[30]:


# analyzing the sensitivity, specificity, AUC (ROC curve) of each model
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[31]:


logreg_probs=logreg.predict_proba(x)

knn=KNeighborsClassifier()
knn.fit(xtrain,ytrain)
knn_probs=knn.predict_proba(x)


# In[32]:


#plotting the roc_curve
lrfpr, lrtpr, lrthreshold = roc_curve(y, logreg_probs[:,1], pos_label=1)
knnfpr, knntpr, knnthreshold = roc_curve(y, knn_probs[:,1], pos_label=1)

"""random_probs=[0 for i in range(len(y))]
randomfpr, randomtpr, _ = roc_curve(y, random_probs, pos_label=1)
plt.plot(randomfpr, randomtpr, linestyle='--', color='blue', label='No Skill')"""

plt.plot((0,1), (0,1), linestyle='--', color='blue', label='No Skill')
plt.plot(lrfpr, lrtpr, linestyle='--', color='green', label='Logistic Regression')
plt.plot(knnfpr, knntpr, linestyle='--', color='red', label='KNN')

plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

#calculating the auc score
lr_auc= roc_auc_score(y, logreg_probs[:,1])
knn_auc= roc_auc_score(y, knn_probs[:,1])
print('Logistic Regression AUC: {}'.format(lr_auc))
print('KNN Classifier AUC: {}'.format(knn_auc))


# In[33]:


#plotting the ROC curve and finding out the AUC Score of different classifier models in comparison with KNN Classifier
models1=[('LogReg', LogisticRegression()),('DT', DecisionTreeClassifier()),('RF', RandomForestClassifier()),('XGB',XGBClassifier())]
for name, model in models1:
    model.fit(xtrain,ytrain)
    model_probabs=model.predict_proba(x)
    knn=KNeighborsClassifier()
    knn.fit(xtrain,ytrain)
    knn_probabs=knn.predict_proba(x)
    
    modelfpr, modeltpr, modelthreshold = roc_curve(y, model_probabs[:,1], pos_label=1)
    knnfpr, knntpr, knnthreshold = roc_curve(y, knn_probabs[:,1], pos_label=1)
    
    plt.plot((0,1),(0,1), linestyle='--', color='blue', label='No Skill')
    plt.plot(modelfpr, modeltpr, linestyle='--', color='red', label=name)
    plt.plot(knnfpr, knntpr, linestyle='--', color='green', label='KNN')
    
    plt.title(str(name) + '-KNN ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.show()
    
    modelauc= roc_auc_score(y, model_probabs[:,1])
    knnauc= roc_auc_score(y, knn_probabs[:,1])
    print(str(name)+'AUC: {}'.format(modelauc))
    print('KNN AUC: {}'.format(knnauc))
    
    print('=='*50)


# It is hence observed that Decision Tree, Random Forest and XGBoost perform more precisely than Logistic Regression and KNN Classifier as they have higher AUC score and we can observe the ROC Curve for the said classifiers passing through the top left corner i.e. FPR=0,TPR=1.

# In[34]:


#plotting the precision recall curve for all classifiers
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score

models2=[]
models2.append(('LogReg', LogisticRegression()))
models2.append(('DT', DecisionTreeClassifier()))
models2.append(('RF', RandomForestClassifier()))
models2.append(('XGB', XGBClassifier()))
models2.append(('KNN', KNeighborsClassifier()))

def generate_graph(recall, precision, name):
    plt.figure(figsize=(8,5))
    plt.plot((0,1), (0.5,0.5), linestyle='--', color='blue')
    plt.plot(recall, precision, marker='.', label=name)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(name)
    plt.legend()
    plt.show()
    
for name, model in models2:
    print('=='*50)
    print('Precision-Recall curve for {}'.format(name))
    print('=='*50)
    
    model.fit(xtrain,ytrain)
    probs=model.predict_proba(x)
    probs=probs[:,1]
    yhat=model.predict(x)
    
    precision,recall,thresholds=precision_recall_curve(y, probs)
    f1=f1_score(y, yhat)
    auc_score=auc(recall,precision)
    ap=average_precision_score(y, probs)
    
    generate_graph(recall, precision, name)
    
    print('Calculated Values for' + str(name) +':')
    print('F1 Score: {}'.format(f1))
    print('Area Under the Curve: {}'.format(auc_score))
    print('Average Precision Score: {}'.format(ap))
    print('--'*50)


# Used F1 Score, AUC and Average Precision Score to evaluate various classifiers and following observations are drawn:
# 1. Decision Tree, XGBoost and Random Forest Classifier have shown promising precision as can be observed from the precision-recall curves plotted for each classifier respectively. They have higher area under the curve and the precision-recall curve is passing through the top right corner i.e. Precision=1, Recall=1
# 2. On the other hand, Logistic Regression and KNN Classifier have similar results in all aspects.

# #### Conclusion: ####
# ##### It is observed that Logistic Regression gave the most accurate results, although it lacked on precision which can be accounted in the form of consistency of a model to tackle false positives. Decision Tree, Random Forest Classifier and XGBoost Classifier had higher precision than all other classifiers. 

# # Tableau Data Report #

# ![image.png](attachment:image.png)

# Cick on the following link to view the viz on Tableau Public:
# https://public.tableau.com/views/DiabetesDataAnalysis_16837196392100/DiabetesDataAnalysis?:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link
