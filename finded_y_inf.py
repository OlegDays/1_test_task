import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import math

df = pd.read_csv("good2.csv", encoding="UTF-8")

x_data = []
y_data = []

for i in range(1, len(df)):
    dt = df["t"].iloc[i] - df["t"].iloc[i-1]
    if dt <= 0:
        continue
    dy = df["y"].iloc[i] - df["y"].iloc[i-1]
    if dy <= 0:
        continue
    y_val = df["y"].iloc[i]
    tempo = (1 / y_val) * (dy / dt)
    x_data.append(math.log(y_val))
    y_data.append(tempo)

x_data = pd.Series(x_data)
y_data = pd.Series(y_data)

span = 51
y_data = y_data.ewm(span=span, adjust=False).mean()

start = 75
lenght = 120
x = x_data[start:lenght].values.reshape(-1, 1)
y = y_data[start:lenght]

model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

plt.scatter(x, y, label='Данные')
plt.plot(x, y_pred, 'r', label='Регрессия')
plt.xlabel('ln y')
plt.ylabel('(1/y) dy/dt')
plt.legend()
plt.show()

print("Коэффициент:", model.coef_[0])
print("Свободный член:", model.intercept_)