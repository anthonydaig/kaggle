import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd
from sklearn.model_selection import cross_val_score
from numpy import mean
from sklearn.neural_network import MLPRegressor


df = pd.read_hdf('train.h5')
df = df.fillna(0)
cols = list(df.columns)
cols.remove('y')
cols.remove('id')
X = df[cols].values
y = df.y.values

def test_lr_fit(X, y):
	lr = linear_model.LinearRegression()
	scores = cross_val_score(lr, X, y, cv=10, n_jobs = 5, scoring = 'neg_mean_squared_error')
	print(scores)
	return lr

#### using all features
# [-0.00070571 -0.00070313 -0.00168687 -0.00028591 -0.00025608 -0.00025862
#  -0.00022209 -0.00046262 -0.00122381 -0.00053382]

lr.fit(X, y)
kk = lr.coef_
jj = [abs(i) for i in kk]


new_cols = ['timestamp',
 'derived_0',
 'derived_1',
 'derived_2',
 'derived_3',
 'derived_4',
 'technical_36',
 'technical_37',
 'technical_38',
 'technical_39',
 'technical_40',
 'technical_41',
 'technical_42',
 'technical_43',
 'technical_44']


X = df[new_cols].values

test_lr_fit(X, y)
 # [-0.00070571 -0.00070313 -0.00037573 -0.00028591 -0.00025608 -0.00025862
 # -0.00022209 -0.00046262 -0.00122381 -0.00053382]
model2 = MLPRegressor
def test_mlp_fit(X, y):
	lr = MLPRegressor((200, 150, 100, 50))
	scores = cross_val_score(lr, X, y, cv=4, n_jobs = 5, scoring = 'neg_mean_squared_error')
	print(scores)
model2.fit(X, y)





from sklearn.decomposition import PCA
pca = PCA()

pca.fit(X)


# 'fundamental_17', 'fundamental_61', 'derived_1'

for i in pca.components_:
	print(list(i).index(max(i)))













