#Create KNN Model from MongoDB connection

#%% 
# Dependencies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from flask import Flask
from pymongo import MongoClient


#%% 
# Initial Data Load - To Be Replaced by MongoDB Load
traindf = pd.read_csv('fraudTrain.csv')
testdf = pd.read_csv('fraudTest.csv')

# MongoDB Data Import
# from flask import Flask
# from pymongo import MongoClient

# app = Flask(__name__)

# client = MongoClient('localhost', 27017)

# db = client.flask_db
# todos = db.todos



#%%
# Pre-Processing
# Data Modeling Techniques
#pulling out target features/columns
target_columns =['amt', 'city_pop', 'zip', 'category', 'unix_time']
#target_columns = ['first', 'last', 'dob', 'gender', 'street', 'city', 'state', 'zip', 'job', 'amt']

#full_df = pd.concat([traindf, testdf])
#Training Dataset
X_raw = traindf[target_columns]
y_raw = traindf['is_fraud']

#Validation Dataset
X_val = testdf[target_columns]
y_val = testdf['is_fraud']


#%%
#restructuring data for model training
#Training dataset
dummy_categories = pd.get_dummies(data=traindf['category'])
X_newraw = X_raw.drop(['category'], axis=1)

for category in dummy_categories:
  X_newraw[category] = dummy_categories[category]

#Validation dataset
X_newval = X_val.drop(['category'], axis=1)

for category in dummy_categories:
  X_newval[category] = dummy_categories[category]

print(X_newval)
#%%
# Dimensionality Reduction
#Training Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X_normalized = X_newraw.values
X_normalized = StandardScaler().fit_transform(X_normalized)

print(X_normalized)
pca = PCA(n_components=6)
principal_components = pca.fit_transform(X_normalized)
principal_df = pd.DataFrame(data = principal_components, columns = ['principal component 1', 'principal component 2',
                                                                    'principal component 3', 'principal component 4',
                                                                    'principal component 5', 'principal component 6'])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(principal_df, y_raw, test_size=0.3, random_state=42)


#Validation Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X_normalized = X_newval.values
X_normalized = StandardScaler().fit_transform(X_normalized)

pca = PCA(n_components=6)
principal_components = pca.fit_transform(X_normalized)
principal_df_val = pd.DataFrame(data = principal_components, columns = ['principal component 1', 'principal component 2',
                                                                    'principal component 3', 'principal component 4',
                                                                    'principal component 5', 'principal component 6'])

#%%
# KNN Modeling
from sklearn.neighbors import KNeighborsClassifier
#X_train, X_test, y_train, y_test = train_test_split(principal_df, y_raw, test_size=0.3, random_state=42)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

pickle.dump(knn_model, open('knn_model.pkl','wb'))

#%%
#Verification of Model Creation
model = pickle.load(open('knn_model.pkl','rb'))
y_pred_log = knn_model.predict(X_test)

#%%
from sklearn.metrics import confusion_matrix, classification_report
print("K Nearest Neighbors Model Performance:")
print("-"*50)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))
print("-"*50)
print("Classification Report:")
print(classification_report(y_test, y_pred_knn))
print("-"*50)
print("R^2: {}".format(knn_model.score(X_test, y_test)))

# %%
