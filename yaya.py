import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
import shap
from shap import Explanation
import sys
sys.tracebacklimit=0

df = pd.read_csv("xvz.csv", encoding="utf-8")  # raw csv file
df = df.replace(-1.0, np.NaN)
limitPer = len(df) * .50
df = df.dropna(thresh=limitPer, axis=1)

drop_cols = ["RecordID", 'SAPS-I', 'SOFA', 'Length_of_stay', "Survival", 'Gender', 'HCT',
             'Height', 'K', 'Mg', 'Na', 'Temp', 'FiO2', 'MechVent', 'pH', 'Time']

df = df.drop(columns=drop_cols)

# imputer that fills in NA with most frequent values
imputer = SimpleImputer(strategy='most_frequent',
                        missing_values=np.nan)
# impute on appropriate variables
imputer = imputer.fit(df[["ICUType"]])
df[["ICUType"]] = imputer.transform(df[["ICUType"]])
# impute with mean values
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(df)
df.iloc[:, :] = imp.transform(df)

y = df['In-hospital_death']
x = df.drop(['In-hospital_death'], axis=1)

trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.2)

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()

scaler = sc.fit(trainX)
trainX_scaled = scaler.transform(trainX)
testX_scaled = scaler.transform(testX)

best = MLPClassifier(hidden_layer_sizes=(100, 50, 30),
                     max_iter=100, activation='relu',
                     solver='adam', alpha=0.0001, learning_rate='adaptive',
                     early_stopping=True)

best.fit(trainX_scaled, trainY)


def pivot(df):
        try:
            df = df.pivot_table(index='Time', values='Value', columns='Parameter').reset_index().rename_axis(None,
                                                                                                             axis=1)
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
            df2 = df.to_dict()

            for i in range(len(df2)):

                if (list(df2.values())[i][0]) < -1:
                    raise Exception("Sorry, no negative values.")

            return df
        except KeyError:
            raise Exception("Please Enter the file in a valid format")


def mortality(file):
    df = pd.read_csv(file, encoding="utf-8")

    try:
        lst = list(df['Value'])
        lst = [float (i) for i in lst]

        zz = pivot(df)



        train_X, test_X, train_Y, test_Y = train_test_split(x, y, test_size=0.2)
        train_X = train_X.append(zz)
        sc = MinMaxScaler()
        scaler = sc.fit(train_X)
        train_X = pd.DataFrame(sc.fit_transform(train_X), columns=list(x))
        df = pd.DataFrame(train_X.iloc[-1:])
        # prob of survival
        survival = best.predict_proba(df)[0][0]
        #######################################
        explainer = shap.Explainer(best.predict, testX)
        shap_values = explainer(df)
        kk = shap_values
        zz = shap_values[0]

        return survival, kk, zz
    except ValueError as e:
        raise Exception("please Enter a Valid file format")
    except KeyError:
        raise Exception("Please Enter a valid file format")





