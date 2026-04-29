import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import math

df = pd.read_csv("../data/good2.csv", encoding="UTF-8")
y_inf = 15638.693994919746066740527
y_data = []


for i in range(75, 150):
    y_data.append(math.log(math.log(y_inf / df["y"][i])))

print(y_data)
print(len(y_data))
plt.figure(figsize=(12, 5))
plt.plot(df["t"][75:150], y_data, marker='o', linestyle='-', markersize=3)
plt.title('Визуализация проверки')
plt.xlabel('t')
plt.ylabel('ln(ln(y_inf / y))')
plt.grid(True)
plt.show()

x = df["t"].values.reshape(-1, 1)
y = y_data
start = 75
end = 150

model = LinearRegression()
model.fit(x[start:end], y)
y_pred = model.predict(x[start:end])
plt.scatter(x[start:end], y)
plt.plot(x[start:end], y_pred, 'r')
plt.legend()
plt.show()

print("Коэффициент:", model.coef_[0])
print("Свободный член:", model.intercept_)

