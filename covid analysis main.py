import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Import the dataset
df=pd.read_csv("https://raw.githubusercontent.com/SR1608/Datasets/main/covid-data.csv")

# Data Understanding and Cleaning
df.shape
df.dtypes
df.info()
pd.set_option('display.max_columns',None)
df.describe(include="all").round(2)

df['location'].nunique()
df['continent'].value_counts()
df['total_cases'].max()
df['total_cases'].mean()
df['total_cases'].min()
df['total_deaths'].describe().round(2)
df.groupby("continent").agg({"human_development_index":"max"}).head(1)
df.groupby('continent').agg({'gdp_per_capita':'min'}).head(1)

df=df[['continent','location','date','total_cases','total_deaths','gdp_per_capita','human_development_index']]
df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.isnull().sum()
df.dropna(subset=['continent'], inplace=True)
df=df.fillna(0)

# Datetime format
df['date']=pd.to_datetime(df['date'])
df['month']=pd.DatetimeIndex(df['date']).month

# Data Aggregation
df_groupby=df.groupby("continent").max().reset_index()

# Feature Engineering
df_groupby["total_deaths_to_total_cases"]=df_groupby["total_deaths"]/df_groupby["total_cases"]

# Data Visualization
sns.distplot(df["gdp_per_capita"])
sns.jointplot(data=df_groupby,x="total_cases",y="gdp_per_capita",kind="scatter")
sns.pairplot(data=df_groupby)
sns.catplot(data=df_groupby,x='continent',y='total_cases',kind='bar')
df.to_csv("covid_Data.csv")


# Loan dataset analysis
df = pd.read_csv("/content/loan_data_set.csv")
df.isnull().sum()
df.fillna(method="ffill", inplace=True)
df.fillna(method="bfill", inplace=True)

df1 = pd.get_dummies(df, columns=["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area", "Loan_Status"])
df1.dtypes
df1.drop("Gender_Female", axis=1, inplace=True)
df1.drop("Married_No", axis=1, inplace=True)
df1.drop("Education_Graduate", axis=1, inplace=True)
df1.drop("Self_Employed_No", axis=1, inplace=True)
df1.drop("Property_Area_Rural", axis=1, inplace=True)
df1.drop("Loan_Status_N", axis=1, inplace=True)
df1.drop("Dependents_3+", axis=1, inplace=True)

x = df1.iloc[:, 2:15]
y = df1.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

m1 = LogisticRegression()
m1.fit(x_train, y_train)
y_pred = m1.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
c = confusion_matrix(y_test, y_pred)
print("accuracy=",accuracy)
print("confusion matrix=",c)