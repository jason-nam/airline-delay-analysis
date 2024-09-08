# Databricks notebook source
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# COMMAND ----------

df = spark.read.option("header", "true").option("inferSchema", "true").csv("/FileStore/tables/airlines_delay.csv").toPandas()

# COMMAND ----------

df.head()

# COMMAND ----------

y = df.iloc[:,7]
x = df.iloc[:,1:7]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

# COMMAND ----------

# Pre-process data: Numerical Xs

step_1a = Pipeline(
    [
        ("1A_i", KNNImputer(n_neighbors = 6)),
        ("1A_ii", StandardScaler())
    ])

# COMMAND ----------

# Pre-process data: Categorical Xs

step_1b = Pipeline(
    [
        ("1B_i", SimpleImputer(strategy = 'most_frequent')),
        ("1B_ii", OneHotEncoder())
    ])

# COMMAND ----------

# Pre-process data: Column transforms

num_x = ['Time', 'Length', 'DayOfWeek']

cat_x = ['Airline', 'AirportFrom', 'AirportTo']

step_1c = ColumnTransformer(
    [
        ('1C_i', step_1a, num_x),
        ('1C_ii', step_1b, cat_x)
    ])

# COMMAND ----------

# Decision Tree

max_depth = 4

model_dt = Pipeline(
    [
        ('1C', step_1c),
        ('2_RF', DecisionTreeClassifier(max_depth = max_depth, criterion = "entropy"))
    ])

model_dt.fit(x_train, y_train)

# COMMAND ----------

# Random Forest Tree

max_depth = 4

model_rf = Pipeline(
    [
        ('1C', step_1c),
        ('2_RF', RandomForestClassifier(max_depth = max_depth))
    ])

model_rf.fit(x_train, y_train)

# COMMAND ----------

# K-Nearest Neighbors

n_neighbors = 5

model_knn = Pipeline(
    [
        ('1C', step_1c),
        ('2_RF', KNeighborsClassifier(n_neighbors = n_neighbors))
    ])

model_knn.fit(x_train, y_train)

# COMMAND ----------

# Logistic Regression

n_neighbors = 5

model_lr = Pipeline(
    [
        ('1C', step_1c),
        ('2_RF', LogisticRegression())
    ])

model_lr.fit(x_train, y_train)

# COMMAND ----------

print("Accuracy score of the Decision Tree model is " + str(model_dt.score(x_test, y_test)))
print("Accuracy score of the Random Forest Tree model is " + str(model_rf.score(x_test, y_test)))
print("Accuracy score of the K-Nearest Neighbors model is " + str(model_knn.score(x_test, y_test)))
print("Accuracy score of the Logistic Regression model is " + str(model_lr.score(x_test, y_test)))

# COMMAND ----------


