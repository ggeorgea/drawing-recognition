import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC	
import numpy as np
import pandas as pd
import json
from pandas.io.json import json_normalize
import ast
import functionMisc
import os
import sys


data = np.load('justDescriptors.npy')
np.random.shuffle(data)


out = np.array(data[:,0])
inp = []

for inIndex in range(0 ,len(data)):
	inputs = np.array(data[:,1][inIndex]).ravel()
	inputs = inputs[:350]
	if len(inputs)<350:
		extraneeded = 350 - len(inputs)
		inputs = np.concatenate((inputs,np.zeros(extraneeded)))
	inp.append(np.array(inputs))
inp = np.vstack(inp)


print(inp.shape)
print(out.shape)


split = (len(out)*9//10)


X_train, X_test = inp[:split], inp[split:]


y_train, y_test = out[:split], out[split:]


# linC = LogisticRegression()
# linOnevOne = OneVsOneClassifier(linC)
# linOnevOne.fit(X_train, y_train)
# print(", 1versus1 logistic, ")
# print(" %f" % linOnevOne.score(X_train, y_train))
# print(",  %f" % linOnevOne.score(X_test, y_test))
# # for x in range(split + ((len(y)-split)//5), split + ((len(y)-split)//3)):
# # 	print(x, y[x], "vs ", linOnevOne.predict((X.loc[x]).values.reshape(1,-1)))
# sys.stdout.flush()

# linC = LogisticRegression()
# linOnevOne = OneVsRestClassifier(linC)
# linOnevOne.fit(X_train, y_train)
# print("\n	,1versus rest logistic, ")
# print(" %f" % linOnevOne.score(X_train, y_train))
# print(",  %f" % linOnevOne.score(X_test, y_test))
# # for x in range(split + ((len(y)-split)//5), split + ((len(y)-split)//3)):
# # 	print(x, y[x], "vs ", linOnevOne.predict((X.loc[x]).values.reshape(1,-1)))
	
# sys.stdout.flush()


#add a second layer of 10 to get perfect
mlpMC = MLPClassifier(hidden_layer_sizes=(100,100), random_state  = 4, max_iter=5000)
onevOne = OneVsOneClassifier(mlpMC)
onevOne.fit(X_train, y_train)
print("\n	,1versus1 MLP, ")
print(" %f" % onevOne.score(X_train, y_train))
print(",  %f" % onevOne.score(X_test, y_test))
# for x in range(split + ((len(y)-split)//5), split + ((len(y)-split)//3)):
# 	print(x, y[x], "vs ", onevOne.predict((X.loc[x]).values.reshape(1,-1)))
	
sys.stdout.flush()

# #class sklearn.svm.SVR(kernel=’rbf’, degree=3, gamma=’auto’, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
# sv1 = SVC(kernel = 'linear')
# onevOne = OneVsOneClassifier(sv1)
# onevOne.fit(X_train, y_train)
# print("\n	,1versus1 SVM, ")
# print("%f" % onevOne.score(X_train, y_train))
# print(", %f" % onevOne.score(X_test, y_test))
# # for x in range(split + ((len(y)-split)//5), split + ((len(y)-split)//3)):
# # 	print(x, y[x], "vs ", sv1.predict((X.loc[x]).values.reshape(1,-1)))
	
