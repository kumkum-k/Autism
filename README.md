# Autism
Utilizing supervised learning techniques, this project aims to enhance the early diagnosis of Autistic Spectrum Disorder (ASD) by analyzing behavioral attributes and individual traits. By leveraging a neural network created with the Keras API, the goal is to develop a diagnostic tool for ASD, potentially leading to improved treatment strategies and reduced healthcare costs associated with neurodevelopmental disorders.

for numeric computing
import numpy as np
#for dataframe
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

#Machine Learning Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.svm import SVC,LinearSVC
#Regression
from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV,ElasticNet,LogisticRegression
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.svm import SVR

#Modelling Helpers
from sklearn.preprocessing import Normalizer,scale
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV,KFold,cross_val_score,ShuffleSplit,cross_validate

#Preprocessing
from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder
from sklearn.impute import SimpleImputer

#Classification
from sklearn.metrics import accuracy_score,recall_score,f1_score,fbeta_score,r2_score,roc_auc_score,roc_curve,auc,cohen_kappa_score

#to display all the interactive output without using print()
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactive="all"
df=pd.read_csv('Toddler Autism dataset July 2018.csv')
df.info()
df.head(20)
df.shape

df.describe()

print("COLUMNS")
df.columns
#remove unwanted columns
df.drop(['Case_No','Who completed the test'],axis=1,inplace=True)
df.columns

#Calculating the percentage of babies shows the symptoms of autism
yes_autism = df[df['Class/ASD Traits '] == 'Yes']
no_autism = df[df['Class/ASD Traits '] == 'No']

print("Toddlers:",round(len(yes_autism)/len(df)==100,2))
print("Toddlers:",round(len(no_autism)/len(df)==100,2))
#Displaying The Content of the target column
df['Class/ASD Traits '].value_counts()

fig=plt.gcf()
fig.set_size_inches(7,7)
plt.pie(df['Class/ASD Traits '].value_counts(),labels=('no_autism','yes_autism'),explode=[0.1,0],autopct='%1.1f%%',shadow=True,startangle=90,labeldistance=1.1)
plt.show()

#Checking null data
df.isnull().sum()
df.dtypes
corr=df.corr()
plt.figure(figsize=(15,15))
sns.heatmap(data=corr,annot=True,square=True,cbar=True)

#Visualizing Juandiace occurance in males and females
plt.figure(figsize=(16,8))
plt.style.use('dark_background')
sns.countplot(x='Jaundice',hue="Sex",data=yes_autism)
sns.countplot(x='Qchat-10-Score',hue='Sex',data=df)
#Visualizing the age Distribution of Positive ASD among Todllers
f,ax=plt.subplots(figsize=(12,8))
sns.countplot(x="Age_Mons",data=yes_autism,color="r")

plt.style.use('dark_background')
ax.set_xlabel('Toddlers age in monts')
ax.set_title('Age Distribution of ASD positive')

plt.figure(figsize=(16,8))
sns.countplot(x='Ethnicity',data=yes_autism)

#Visualize Positive ASD among Todllers Based on Ethnicity
plt.figure(figsize=(20,6))
sns.countplot(x='Ethnicity',data=yes_autism,order=yes_autism['Ethnicity'].value_counts().index[:11],hue="Sex",palette='Paired')
plt.xlabel('Ethnicity')
plt.tight_layout()

#displaying the no. of positive cases of ASD with regared Ethnicity
yes_autism['Ethnicity'].value_counts()

#Lets Visualize the distribution of autsim in family within different ethnicity
f,ax=plt.subplots(figsize=(12,8))
sns.countplot(x='Family_mem_with_ASD',data=yes_autism,hue='Ethnicity',palette='rainbow',ax=ax)
ax.set_xlabel('Toddlers Relatives with ASD')
ax.set_title('POsitive ASD Toddler Relatives with Autism Distribution for Different Ethnicities')
plt.tight_layout()

#removing 'Qchat-10-score'
df.drop('Qchat-10-Score',axis=1,inplace=True)

le=LabelEncoder()
columns=['Ethnicity','Family_mem_with_ASD','Class/ASD Traits ','Sex','Jaundice']










