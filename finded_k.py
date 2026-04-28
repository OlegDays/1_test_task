from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("anamorf.csv", encoding="UTF-8")
x = df["t"].values.reshape(-1, 1)
y = df["y"]
lenght = 103

model = LinearRegression()
model.fit(x[66:lenght], y[66:lenght])
y_pred = model.predict(x[66:lenght])
plt.scatter(x[66:lenght], y[66:lenght])
plt.plot(x[66:lenght], y_pred, 'r')
plt.legend()
plt.show()

print("Коэффициент:", model.coef_[0])
print("Свободный член:", model.intercept_)
