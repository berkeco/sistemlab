

#Aashir Javed kodundan esinlendim


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
url = 'https://raw.githubusercontent.com/irhallac/SisLab/main/hw_data/data_0123.csv'
res = requests.get(url, allow_redirects=True)
with open('data0123.csv','wb') as file:
    file.write(res.content)
df = pd.read_csv('data0123.csv')

print(df)

degercikisi = df.iloc[0:100, 4].values
degercikisi = np.where(degercikisi == 'Iris-setosa', -1, 1)
degergirisi = df.iloc[0:100, [0, 2]].values
plt.title('2 Boyutlu Görüntüsü', fontsize=13)

plt.scatter(degergirisi[:50, 0], degergirisi[:50, 1], color='black', marker='o', label='setosa')
plt.scatter(degergirisi[50:100, 0], degergirisi[50:100, -1], color='green', marker='x', label='versicolor')
plt.xlabel('sapel uzunluğu')
plt.ylabel('petal uzunluğu')
plt.legend(loc='upper left')

plt.show()

class Perceptron(object):
    def __init__(self, ogr=0.1, veri=10):
        self.ogri = ogr
        self.veri = veri

    def ogren(self, X, y):
        self.w = np.zeros(1 + X.shape[1])

        self.hatalar = []
        for _ in range(self.veri):
            hata = 0
            for xi, hedef in zip(X, y):
                degisim = self.ogr * (hedef - self.tahmin(xi))
                self.w[1:] += degisim * xi
                self.w[0] += degisim
                hata += int(degisim != 0.0)
            self.hatalar.append(hata)
        return self

    def net_input(self, X):
        return np.dot(X, self.w[1:]) + self.w[0]

    def tahmin(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)


sinif = Perceptron(ogr=0.1, veri=10)
sinif.ogren(degergirisi, degercikisi)
sinif.w
sinif.hatalar