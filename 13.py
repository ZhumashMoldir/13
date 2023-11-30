# 13

# Three lines to make our compiler able to draw:
import sys
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

full_univer_data = pd.read_csv("data.csv", header=0, sep=",")

x = full_univer_data["Science"]
y = full_univer_data["English"]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
    return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.figure(figsize=(10, 6))

plt.scatter(x, y)
plt.plot(x, mymodel, color='red')  
plt.ylim(ymin=0, ymax=100)  
plt.xlim(xmin=0, xmax=200)
plt.xlabel("Science")
plt.ylabel("English")
plt.title("Scatter Plot and Linear Regression")
plt.grid(True)  

# Two lines to make our compiler able to draw:
plt.savefig(sys.stdout.buffer)
sys.stdout.flush()



import pandas as pd
import statsmodels.api as sm

# Загрузка данных
df = pd.read_csv("C:/Users/XE/Downloads/datasetage_kz.csv")

# Преобразование категориальных переменных в фиктивные переменные (dummy variables)
df = pd.get_dummies(df, columns=['Gender', 'Married'], prefix=['Gender', 'Married'])

# Выделение независимой переменной (предиктора) и зависимой переменной
X = df["Age"]
y = df["Income"]

# Добавление константы к переменным для оценки свободного члена (intercept)
X = sm.add_constant(X)

# Создание модели
model = sm.OLS(y, X).fit()

# Вывод таблицы коэффициентов регрессии
coefficients_table = model.summary().tables[1]
print(coefficients_table)



import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("C:/Users/XE/Downloads/datasetage_kz.csv")
df = pd.get_dummies(df, columns=['Gender', 'Married'], prefix=['Gender', 'Married'])

X = df["Age"]
y = df["Income"]

# Добавление константы к переменным для оценки свободного члена (intercept)
X = sm.add_constant(X)

# Создание модели
model = sm.OLS(y, X).fit()

# Вывод результатов
print(model.summary())



import sys
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

full_yoga_data = pd.read_csv("data.csv", header=0, sep=",")

x = full_yoga_data["Yoga_Duration"]
y = full_yoga_data["Calories"]

# Визуализация данных
plt.scatter(x, y)
plt.xlabel("Yoga_Duration")
plt.ylabel("Calories")
plt.title("Scatter Plot of Data")
plt.show()

# Статистика и регрессия
slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
    return slope * x + intercept

mymodel = list(map(myfunc, x))

# Calculate R-squared
r_squared = r**2

print(f"R-squared: {r_squared}")

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.ylim(ymin=0, ymax=2000)
plt.xlim(xmin=0, xmax=200)
plt.xlabel("Yoga_Duration")
plt.ylabel("Calories")
plt.title("Linear Regression with R-squared")
plt.annotate(f"R-squared = {r_squared:.2f}", xy=(0.7, 0.9), xycoords="axes fraction", fontsize=10, color="red")
plt.show()

# Two lines to make our compiler able to draw:
plt.savefig(sys.stdout.buffer)
sys.stdout.flush()




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/XE/Downloads/datasetage_kz.csv")

df = pd.get_dummies(df, columns=['Gender', 'Married'], prefix=['Gender', 'Married'])

X = df[["Age"]]
y = df["Income"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)

# Создание модели линейной регрессии
model = LinearRegression()

# Обучение модели на обучающем наборе
model.fit(X_train, y_train)

# Предсказания на тестовом наборе
y_pred = model.predict(X_test)

# Оценка качества модели
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Визуализация результатов
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Linear Regression - Data Science Case')
plt.show()
