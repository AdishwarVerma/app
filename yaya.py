import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV


df = pd.read_csv("balanced.csv", encoding = "utf-8") # raw csv file
out= pd.read_csv("out.csv", encoding = "utf-8") # raw csv file
df=df[list(out)]

# In[6]:


df = df.replace(-1.0, np.NaN)
limitPer = len(df) * .50
df = df.dropna(thresh=limitPer, axis=1)


# In[7]:


drop_cols = ["RecordID", "MechVent", 'SAPS-I', 'SOFA','Length_of_stay',"Survival"]
df = df.drop(columns=drop_cols)


# In[8]:


from sklearn.impute import SimpleImputer


# imputer that fills in NA with most frequent values
imputer = SimpleImputer(strategy='most_frequent',
                        missing_values=np.nan)

# impute on appropriate variables
imputer = imputer.fit(df[['Gender', "ICUType"]])
df[['Gender', "ICUType"]] = imputer.transform(df[['Gender', "ICUType"]])

# impute with mean values
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(df)
df.iloc[:,:] = imp.transform(df)

y = df['In-hospital_death']
x = df.drop(['In-hospital_death'], axis=1)

trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)


# In[11]:
#################################################

from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()

scaler = sc.fit(trainX)
trainX_scaled = scaler.transform(trainX)
testX_scaled = scaler.transform(testX)

mlp_clf = MLPClassifier(hidden_layer_sizes=(150,100,50),
                        max_iter = 300,activation = 'relu',
                        solver = 'adam')

mlp_clf.fit(trainX_scaled, trainY)
########################
# df = pd.read_csv("142841.txt", encoding = "utf-8")
# df = df.pivot_table(index='Time', values='Value', columns='Parameter')         .reset_index()         .rename_axis(None, axis=1)
# new = df
# names = list(df.columns)
# names = names[1:]
# for name in names:
#     new[name] = new[name].mean()
#
# #get the first record
# df = new.iloc[[0]]
# # drop time column
# df = df.iloc[: , 1:]
# for col in x.columns:
#     if col not in df.columns:
#         df[col] = x[col].mean()
# names = list(x.columns)
# df = df[names]
#
# train_X, test_X, train_Y, test_Y = train_test_split(x, y, test_size = 0.2)
# train_X = train_X.append(df)
# sc=StandardScaler()
# scaler = sc.fit(train_X)
# train_X = pd.DataFrame(sc.fit_transform(train_X), columns=list(x))
# df = pd.DataFrame(train_X.iloc[-1:])
# #prob of survival
# mlp_clf.predict_proba(df)[0][0]
# #prob of death
# mlp_clf.predict_proba(df)[0][1]
# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression()
# lr.fit(trainX,trainY)
# Y_pred = lr.predict(testX)
# print(Y_pred[0])
# print(lr.predict_proba(testX))
def pivot(df):

    df = df.pivot_table(index='Time', values='Value', columns='Parameter').reset_index().rename_axis(None, axis=1)
    new = df
    names = list(df.columns)
    names = names[1:]
    for name in names:
        new[name] = new[name].mean()

    # get the first record
    df = new.iloc[[0]]
    # drop time column
    df = df.iloc[:, 1:]
    for col in x.columns:
        if col not in df.columns:
            df[col] = x[col].mean()
    names = list(x.columns)
    df = df[names]
    return df

# def scaling(df):
#     train_X, test_X, train_Y, test_Y = train_test_split(x, y, test_size=0.2)
#     train_X = train_X.append(df)
#     sc = StandardScaler()
#     scaler = sc.fit(train_X)
#     train_X = pd.DataFrame(sc.fit_transform(train_X), columns=list(x))
#     df = pd.DataFrame(train_X.iloc[-1:])
#     return df


import shap
from shap import Explanation
def mortality(file):

    df = pd.read_csv(file, encoding="utf-8")



    zz=pivot(df)

    train_X, test_X, train_Y, test_Y = train_test_split(x, y, test_size=0.2)
    train_X = train_X.append(zz)
    sc = StandardScaler()
    scaler = sc.fit(train_X)
    train_X = pd.DataFrame(sc.fit_transform(train_X), columns=list(x))
    df = pd.DataFrame(train_X.iloc[-1:])
    #prob of survival
    survival=mlp_clf.predict_proba(df)[0][0]
    #######################################
    explainer = shap.Explainer(mlp_clf.predict, testX)
    shap_values = explainer(df)
    kk = shap_values
    zz=shap_values[0]


    return survival,kk,zz


#############


# def new_func(file):
#     df = pd.read_csv(file, encoding="utf-8")
#     zz = pivot(df)
#     df = scaling(zz)
#     explainer = shap.Explainer(mlp_clf.predict, testX)
#     shap_values = explainer(df)


    # exp = Explanation(shap_values.values,
    #               shap_values.base_values,
    #               data=df.values,
    #               feature_names=df.columns)



   # return kk



