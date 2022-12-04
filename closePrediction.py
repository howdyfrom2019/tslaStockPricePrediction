import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("TSLA_Data(20121203-20221103).csv")
test = pd.read_csv("TSLA_Train.csv")

data["time"] = data.sort_index(ascending=False).index + 1
test["time"] = test.sort_index(ascending=False).index + 1

close = data["Close"]
time = data["time"]
time = sm.add_constant(time)
mod = sm.OLS(close, time)
res = mod.fit()

close_test = test["Close"]
time_test = test["time"]
time_test = sm.add_constant(time_test)

# test_result = res.predict(time_test)
# plt.plot(test_result, label="prediction")
# plt.plot(close_test, label="real")


result = res.predict(time)
plt.plot(data["time"], result, label="prediction")
plt.plot(data["time"], close, label="real")
# plt.xlabel("time")
# plt.ylabel("close_price")
# plt.legend()
plt.show()

