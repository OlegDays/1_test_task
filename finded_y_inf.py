import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import math

df = pd.read_csv("good2.csv", encoding="UTF-8")

x_data = []
y_data = []
lenght = 103
y_helped = 0
for i in range(1, df.size // 2):
    und_log = (df["y"][i] - df["y"][i - 1]) / 2
    if und_log > 0:
        y = math.log((1 / df["y"][i]) * und_log)
        y_helped = y
        y_data.append(y)
    else:
        y_data.append(y_helped)

for i in range(1, df.size // 2):
    x_data.append(math.log(df["y"][i]))

x_data = pd.Series(x_data)
y_data = pd.Series(y_data)

span = 51
y_data = y_data.ewm(span=span, adjust=False).mean()

plt.figure(figsize=(12, 5))
plt.plot(x_data[66:lenght], y_data[66:lenght], marker='o', linestyle='-', markersize=3)
plt.title('Визуализация анаморфозы')
plt.xlabel('lny')
plt.ylabel('ln(1/y * dy/dt)')
plt.grid(True)
plt.show()

x = x_data[66:lenght].values.reshape(-1, 1)
y = y_data[66:lenght]

model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)
plt.scatter(x, y)
plt.plot(x, y_pred, 'r')
plt.legend()
plt.show()

print("Коэффициент:", model.coef_[0])
print("Свободный член:", model.intercept_)

