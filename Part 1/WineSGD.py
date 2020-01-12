import numpy as np
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt

data_test = np.loadtxt("./Wine/wine_test.csv", delimiter=";", unpack=True, skiprows=1).T
data_train = np.loadtxt("./Wine/wine_train.csv", delimiter=";", unpack=True, skiprows=1).T

X_train = []
Y_train = []
for line in data_train:
    Y_train.append(line[-1])  # last item is output
    X_train.append(line[:-1])  # all but last

sgd_reg = SGDRegressor(max_iter=200, tol=-np.infty, penalty=None, eta0=0.0001, random_state=42)

sgd_reg.fit(X_train, Y_train)

print("Trained!")

res_real = [Y for Y in (data_test[i][-1] for i in range(len(data_test)))]
res_predict = [sgd_reg.predict(line[:-1].reshape(1, -1)) for line in data_test]

for i in range(200, 210):
    print(res_real[i], res_predict[i])

res_diff = []
for i in range(len(res_real)):
    res_diff.append(abs(res_real[i] - res_predict[i]))

print("mean distance: {}".format(np.mean(res_diff)))

plt.hist(res_real)
plt.show()
plt.plot(res_diff)
plt.show()
