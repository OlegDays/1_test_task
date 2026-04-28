import pandas as pd
import matplotlib.pyplot as plt
import math


df = pd.read_csv("task2.csv", encoding="UTF-8")
print(df.head())
print("=========")
print(df.tail())
print("=========")
df.info()

plt.figure(figsize=(12, 5))
plt.plot(df.index, df['y'], marker='o', linestyle='-', markersize=3)
plt.title('Визуализация модели гомперца')
plt.xlabel('t')
plt.ylabel('y')
plt.grid(True)
plt.show()

good_df = df.copy()
span = 51
good_df["y"] = df["y"].ewm(span=span, adjust=False).mean()


plt.figure(figsize=(12, 5))
plt.plot(good_df.index, good_df['y'], marker='o', linestyle='-', markersize=3)
plt.title('Визуализация модели гомперца c обработкой выбросов')
plt.xlabel('t')
plt.ylabel('y')
plt.grid(True)
plt.show()

good_df.to_csv("good2.csv", index=False)
