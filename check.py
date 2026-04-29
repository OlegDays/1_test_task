import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import math

df = pd.read_csv("good2.csv", encoding="UTF-8")
y_inf = 44283.071778024462169320169259635
k = 0.40144
a = 18.81209
y_data = []

for i in range(0, len(df)):
    y_data.append(math.log(math.log(y_inf / df["y"][i])))

print(y_data)
print(len(y_data))
plt.figure(figsize=(12, 5))
plt.plot(df["t"], y_data, marker='o', linestyle='-', markersize=3)
plt.title('Визуализация проверки')
plt.xlabel('t')
plt.ylabel('ln(ln(y_inf / y))')
plt.grid(True)
plt.show()

x = df["t"].values.reshape(-1, 1)
y = y_data
start = 70
end = 140

model = LinearRegression()
model.fit(x[start:end], y[start:end])
y_pred = model.predict(x[start:end])
plt.scatter(x[start:end], y[start:end])
plt.plot(x[start:end], y_pred, 'r')
plt.legend()
plt.show()

print("Коэффициент:", model.coef_[0])
print("Свободный член:", model.intercept_)

