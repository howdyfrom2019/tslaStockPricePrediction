import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("TSLA_Data(20121203-20221103).csv")
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="raise")
df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month
df["day"] = df["Date"].dt.day


X = df[["Open", "High", "Low", "Adj Close", "Volume"]]
y = df[["Close"]]

# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)
mlr = LinearRegression()
mlr.fit(X, y)

df_predict = pd.read_csv("TSLA_Train.csv")
X_predict = df_predict[["Open", "High", "Low", "Adj Close", "Volume"]]

y_predict = mlr.predict(X_predict)

plt.scatter(df_predict["Close"], y_predict, alpha=0.4)
plt.xlabel("Actual Close")
plt.ylabel("Predicted close")
plt.show()
