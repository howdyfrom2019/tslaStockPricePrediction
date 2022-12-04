import torch
import torch.optim as optim
import pandas as pd
import torch.autograd as autograd

data = pd.read_csv("TSLA_Data(20121203-20221103).csv")
test = pd.read_csv("TSLA_Train.csv")

data[["Volume"]] = data[["Volume"]].div(500000)
test[["Volume"]] = test[["Volume"]].div(500000)

d1 = torch.FloatTensor(data[["Open"]].values)
d2 = torch.FloatTensor(data[["High"]].values)
d3 = torch.FloatTensor(data[["Low"]].values)
d4 = torch.FloatTensor(data[["Adj Close"]].values)
d5 = torch.FloatTensor(data[["Volume"]].values)
y_train = torch.FloatTensor(data[["Close"]].values)

t1 = torch.FloatTensor(test[["Open"]].values)
t2 = torch.FloatTensor(test[["High"]].values)
t3 = torch.FloatTensor(test[["Low"]].values)
t4 = torch.FloatTensor(test[["Adj Close"]].values)
t5 = torch.FloatTensor(test[["Volume"]].values)
y_test = torch.FloatTensor(test[["Close"]].values)

w1 = torch.zeros(1, requires_grad=True)
w2 = torch.zeros(1, requires_grad=True)
w3 = torch.zeros(1, requires_grad=True)
w4 = torch.zeros(1, requires_grad=True)
w5 = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer1 = optim.SGD([w1], lr=0.00001)
optimizer2 = optim.SGD([w2], lr=0.00001)
optimizer3 = optim.SGD([w3], lr=0.00001)
optimizer4 = optim.SGD([w4], lr=0.00001)
optimizer5 = optim.SGD([w5], lr=0.00001)
epochs = 1000

for epoch in range(epochs + 1):
    hypothesis = w1 * d1 + b
    cost = torch.mean((hypothesis - y_train) ** 2)
    autograd.set_detect_anomaly(True)

    optimizer1.zero_grad()
    cost.backward()
    optimizer1.step()

for epoch in range(epochs + 1):
    hypothesis = w2 * d2 + b
    cost = torch.mean((hypothesis - y_train) ** 2)
    autograd.set_detect_anomaly(True)

    optimizer2.zero_grad()
    cost.backward()
    optimizer2.step()

for epoch in range(epochs + 1):
    hypothesis = w3 * d3 + b
    cost = torch.mean((hypothesis - y_train) ** 2)
    autograd.set_detect_anomaly(True)

    optimizer3.zero_grad()
    cost.backward()
    optimizer3.step()

for epoch in range(epochs + 1):
    hypothesis = w4 * d4 + b
    cost = torch.mean((hypothesis - y_train) ** 2)
    autograd.set_detect_anomaly(True)

    optimizer4.zero_grad()
    cost.backward()
    optimizer4.step()

for epoch in range(epochs + 1):
    hypothesis = w5 * d5 + b
    cost = torch.mean((hypothesis - y_train) ** 2)
    autograd.set_detect_anomaly(True)

    optimizer5.zero_grad()
    cost.backward()
    optimizer5.step()

print(w1, w2, w3, w4, w5, b)

predict_open = w1 * t1 + b
predict_high = w2 * t2 + b
predict_low = w3 * t3 + b
predict_adj_close = w4 * t4 + b
predict_volume = w5 * t5 + b
avg_predict = (predict_open + predict_high + predict_low + predict_adj_close + predict_volume) / 5
for i in range(21):
    print(y_test[i], avg_predict[i])

