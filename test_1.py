import sklearn
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import numpy as np
import time

start_time = time.time()

boston = datasets.load_boston()
print(boston.keys())

#class sklearn.linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
#fit(X, y, sample_weight=None)
#predict(X)
lr = LinearRegression()
lr.fit(boston.data, boston.target)
hoho = lr.predict(boston.data[2].reshape(1,-1))
print(boston.data[2].reshape(1,-1))
print(boston.target[2])
print(hoho)
print([x for x in zip(boston.feature_names, lr.coef_)])

print(lr._residues)

end_time = time.time()
print('time spent:', end_time - start_time)
