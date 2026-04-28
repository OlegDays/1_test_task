import pandas as pd
import matplotlib.pyplot as plt
import math

df = pd.read_csv("good2.csv", encoding="UTF-8")

lenght = df.size // 2
dat_t = df["t"]
dat_y = df["y"]
x = []
ylog = []
count = 0
for i in range(1, lenght):
    und_log = (df["y"][i] - df["y"][i - 1]) / (df["t"][i] - df["t"][i - 1])
    if und_log > 0:
        y = math.log((1 / df["y"][i]) * und_log)
        y_helped = y
        ylog.append(y)
    else:
        ylog.append(y_helped)
    x.append(df["t"][i])

data = pd.Series(ylog)
data = data.rolling(window=21, center=True).mean()
x_data = pd.Series(data)

print(x_data.size)

plt.figure(figsize=(12, 5))
plt.plot(x_data.index, data, marker='o', linestyle='-', markersize=3)
plt.title('Визуализация анаморфозы')
plt.xlabel('t')
plt.ylabel('ln(1/y * dy/dt)')
plt.grid(True)
plt.show()