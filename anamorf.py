import pandas as pd
import matplotlib.pyplot as plt
import math

df = pd.read_csv("good2.csv", encoding="UTF-8")

lenght = len(df)
dat_t = df["t"]
dat_y = df["y"]
x = []
ylog = []
for i in range(1, lenght):
    und_log = (df["y"].iloc[i] - df["y"].iloc[i - 1]) / (df["t"].iloc[i] - df["t"].iloc[i - 1])
    y = math.log((1 / df["y"].iloc[i]) * und_log)
    ylog.append(y)
    x.append((df["t"].iloc[i] + df["t"].iloc[i-1])/2)

y_data = pd.Series(ylog)

x_data = pd.Series(x)

span = 17
y_data = y_data.ewm(span=span, adjust=False).mean()

print(x_data.size)
df = pd.DataFrame({"t": x_data, "y": y_data})
df.set_index('t', inplace=True)
df.to_csv("anamorf.csv", encoding="UTF-8")

plt.figure(figsize=(12, 5))
plt.plot(x_data, y_data, marker='o', linestyle='-', markersize=3)
plt.title('Визуализация анаморфозы')
plt.xlabel('t')
plt.ylabel('ln(1/y * dy/dt)')
plt.grid(True)
plt.show()

