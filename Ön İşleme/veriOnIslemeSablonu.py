import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('satislar.csv')

print(veriler)

# Bağımsız değişkenler (X) ve bağımlı değişken (y) olarak verileri ayırır
x = veriler.iloc[:,:1].values
y = veriler.iloc[:,1:].values 
print(y)

# Verileri eğitim ve test setlerine böler
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# Verileri ölçeklendirir
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# Doğrusal regresyon modeli oluşturur ve eğitir
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Test seti üzerinde tahmin yapar
tahmin = lr.predict(x_test)
print(tahmin)

# Verileri Görselleştirme
x_train = x_train.sort_index()
y_train = y_train.sort_index()
plt.plot(x_train, y_train)
plt.plot(x_test, lr.predict(x_test))
plt.show()