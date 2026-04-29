import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import math

df = pd.read_csv("../data/good2.csv", encoding="UTF-8")

x_data = []
y_data = []

for i in range(len(df) - 1):
    y_val = df["y"].iloc[i]
    yt_val = df["y"].iloc[i + 1]
    x_data.append(math.log(y_val))
    y_data.append(math.log(yt_val))

x_data = pd.Series(x_data)
y_data = pd.Series(y_data)


start = 75
lenght = 150
x = x_data[start:lenght].values.reshape(-1, 1)
y = y_data[start:lenght]


model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

plt.scatter(x, y, label='Данные')
plt.plot(x, y_pred, 'r', label='Регрессия')
plt.xlabel('ln y t')
plt.ylabel('ln y t+1')
plt.legend()
plt.show()

print("Коэффициент:", model.coef_[0])
print("Свободный член:", model.intercept_)