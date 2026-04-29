import pandas as pd
import matplotlib.pyplot as plt

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
good_df["y_sgl"] = good_df["y"].ewm(span=span, adjust=False).mean()

# ------------------ принудительная монотонизация (неубывание) ------------------
y_mono = good_df["y_sgl"].values
for i in range(1, len(y_mono)):
    if y_mono[i] < y_mono[i - 1]:
        y_mono[i] = y_mono[i - 1] * 1.001
good_df["y_mono"] = y_mono

# ------------------ отбор точек с положительной производной (по монотонному ряду) ------------------
x_data = []
y_data = []
for i in range(1, len(good_df)):
    diff = good_df["y_mono"].iloc[i] - good_df["y_mono"].iloc[i-1]
    if diff > 0:
        x_data.append(good_df["t"].iloc[i])
        y_data.append(good_df["y_mono"].iloc[i])

done_df = pd.DataFrame({"t": x_data, "y": y_data})

plt.figure(figsize=(12, 5))
plt.plot(done_df.index, done_df['y'], marker='o', linestyle='-', markersize=3)
plt.title('Визуализация модели гомперца c обработкой выбросов')
plt.xlabel('t')
plt.ylabel('y')
plt.grid(True)
plt.show()

done_df.to_csv("good2.csv", index=False)