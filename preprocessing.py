import pandas as pd

file = pd.read_csv("task2.csv", encoding="UTF-8")
print(file.head())
print("=========")
print(file.tail())
print("=========")
file.info()