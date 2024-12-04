import pandas as pd 
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore")
sns.set()
import yfinance as yf

df = yf.download("CTNT", start = "2024-12-03", end = "2024-12-04", interval = "1m", auto_adjust=True)
df.columns = [col[0] for col in df.columns.to_flat_index()]
df.index = df.index.tz_convert("US/Eastern")
# create a intraday graph on close prices
df["Close"].plot()

df["returns"] = np.log(df["Close"] / df["Open"])
df["high-low"] = df["High"] - df["Low"]

df["target"] = np.where(df["Volume"].shift(-1)>df["Volume"],-1,1) # if the volume of the next candle is greate than the next candle

features = ["returns", "high-low"]

plt.figure(figsize = (12,5))
sns.scatterplot(x = df["returns"], y = df["high-low"])
plt.show()

sns.pairplot(df[features])
plt.show()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df[features].iloc[:-1], df["target"].iloc[:-1], test_size = 0.25, shuffle = False)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=3, random_state = 2)

model.fit(x_train, y_train)

y_pred = model.predict(x_train)

print("the accuracy of the model on the trained data is :", model.score(x_train, y_train))

y_predtrain = model.predict(x_test)

print("The accuracy of the model in predicting the test data is :", model.score(x_test,y_test))

from sklearn.metrics import accuracy_score

print("The accuracy on the training data:", accuracy_score(y_train, y_pred))

print("The accuracy on the testing dataset is :", accuracy_score(y_test, y_predtrain))

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, y_predtrain))

cm = confusion_matrix(y_test,y_predtrain)

data = pd.DataFrame(cm, index = ["short", "Long"], columns = ["short", "long"])

plt.figure(figsize = (12,5))

sns.heatmap(data, annot = True, fmt = "g")

plt.xlabel("predicted")

plt.ylabel("ActuaL")

plt.show()

# print classification report

print(classification_report(y_test, y_predtrain))

### visualise the tree

from sklearn.tree import plot_tree

fig = plt.figure(figsize = (25,20))

_ = plot_tree(model, feature_names = features, class_names =["-1","1"], filled = True)

print(model.feature_importances_)

plot_df = pd.DataFrame({"importances": model.feature_importances_}, index = features)
plot_df.sort_values("importances", ascending = False).plot.bar();
plt.show()


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators= 20 , max_depth=3, max_leaf_nodes=5, random_state=2,
                               max_features=4, min_samples_leaf=1)

model.fit(x_train, y_train)

y_pred_train = model.predict(x_train)

print("Model accuracy on training data:", model.score(x_train, y_train))

y_pred_train = model.predict(x_train)

print("model accuracy on training data:", model.score(x_train, y_train))

print("model accuracy on the test data.", model.score(x_test, y_test))


      






      
      
      







      
      
      
      




