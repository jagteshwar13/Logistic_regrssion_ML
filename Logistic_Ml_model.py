import os
import pandas as pd 
import numpy as np
from datetime import datetime 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore")
import yfinance as yf
sns.set() # When you call sns.set(), it uses the default style 'darkgrid' unless you specify otherwise.

data = yf.download("SNTI", start = "2024-12-02", end = "2024-12-03", interval = "1m", auto_adjust= True)

data.columns = [ col[0]for col in data.columns.to_flat_index()]

data.index = data.index.tz_convert("US/Eastern") # pandas use it to convert the datetime

data["Close"].plot()


data["returns"] = np.log(data["Close"] / data["Open"])

data["target"] = np.where(data["returns"].shift(-1)>0,1,0) # here we defined the target variable

features = ["Volume","returns"]

plt.figure(figsize = (10,6))
sns.scatterplot(x = data.returns, y = data.Volume)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data[features].iloc[:-1], data.iloc[:-1]["target"], test_size = 0.25, shuffle = False)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() # making a class instance 

x_trn = scaler.fit_transform(x_train) # each row contains volume and returns
x_tst = scaler.transform(x_test)

x_trn_df = pd.DataFrame(x_trn, columns = x_train.columns)

sns.pairplot(x_trn_df[["Volume", "returns"]])
plt.show()

print(x_trn_df.describe().T)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state =1)

model.fit(x_trn, y_train)

y_pred = model.predict(x_trn)

y_trn = model.predict(x_tst)

print("model accuracy on training data: ", model.score(x_trn, y_train))
print("model accuracy on test data:", model.score(x_tst, y_test))

from sklearn.metrics import accuracy_score

print("model accuracy on training data:", accuracy_score(y_pred, y_train))
print("model accuracy on test data:", accuracy_score(y_test, y_trn))

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_trn))

cm = confusion_matrix(y_test, y_trn)
df = pd.DataFrame(cm, index = ["squareoff", "long"], columns = ["squareoff", "long"])  
print(df)    
plt.figure(figsize = (12,7))
sns.heatmap(df, annot = True, fmt="g")
plt.xlabel("predicted")
plt.ylabel("Actual")
plt.show()


      














