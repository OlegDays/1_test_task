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
y_helped = 0
for i in range(1, lenght):
    und_log = (df["y"][i] - df["y"][i - 1]) / 2
    if und_log > 0:
        y = math.log((1 / df["y"][i]) * und_log)
        y_helped = y
        ylog.append(y)
    else:
        ylog.append(y_helped)
    x.append(df["t"][i])

data = pd.Series(ylog)
y_data = data.rolling(window=21, center=True, min_periods=1).mean()
x_data = pd.Series(x)

print(x_data.size)
df = pd.DataFrame({"t": x_data, "y": y_data})
df.set_index('t', inplace=True)
df.to_csv("anamorf.csv", encoding="UTF-8")

plt.figure(figsize=(12, 5))
plt.plot(x_data.index, y_data, marker='o', linestyle='-', markersize=3)
plt.title('Визуализация анаморфозы')
plt.xlabel('t')
plt.ylabel('ln(1/y * dy/dt)')
plt.grid(True)
plt.show()

