import numpy as np
#import pandas as pd
#import nltk
#import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#data_test = pd.read_csv("./Wine/wine_test.csv")
#data_train = pd.read_csv("./Wine/wine_train.csv")

data_test = np.loadtxt("./Wine/wine_test.csv", delimiter=";", unpack=True, skiprows=1).T
data_train = np.loadtxt("./Wine/wine_train.csv", delimiter=";", unpack=True, skiprows=1).T
X_train = []
Y_train = []

for line in data_train:
    print(len(line))
    Y_train.append(line[-1]) # last item is output
    X_train.append(line[:-1]) # all but last

print(X_train[0])

print(type(X_train[0]))
lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)

print("Trained!")

test_index = 300

X_test_1 = data_test[test_index][:-1].reshape(1, -1)
print(X_test_1)


res_real = [Y for Y in (data_test[i][-1] for i in range(len(data_test)))]
res_predict = [lin_reg.predict(line[:-1].reshape(1, -1)) for line in data_test]

for i in range(200,210):
    print(res_real[i], res_predict[i])

res_diff = []
for i in range(len(res_real)):
    res_diff.append(res_real[i] - res_predict[i])

#plt.scatter(res_predict)
#plt.scatter(res_real)
plt.plot(res_diff)
plt.show()