from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("anamorf.csv", encoding="UTF-8")
x = df["t"].values.reshape(-1, 1)
y = df["y"]
start = 75
end = 120

model = LinearRegression()
model.fit(x[start:end], y[start:end])
y_pred = model.predict(x[start:end])
plt.scatter(x[start:end], y[start:end])
plt.plot(x[start:end], y_pred, 'r')
plt.legend()
plt.show()

print("Коэффициент:", model.coef_[0])
print("Свободный член:", model.intercept_)
