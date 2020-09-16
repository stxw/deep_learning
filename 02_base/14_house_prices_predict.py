#!/usr/bin/python3

from mxnet import nd, gluon, autograd, init
from mxnet.gluon import nn
from mxnet.gluon import loss as g_loss
from mxnet.gluon import data as g_data
import numpy as np
import pandas as pd

# 获取数据
train_data = pd.read_csv("02_base/house-prices-data/train.csv")
test_data = pd.read_csv("02_base/house-prices-data/test.csv")
print("train_data.shape =", train_data.shape)
print("test_data.shape =", test_data.shape)
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

# 对数据预处理
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
print("all_features.shape =", all_features.shape)
mumeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
print(mumeric_features)
all_features[mumeric_features] = all_features[mumeric_features].apply(
	lambda x: (x - x.mean()) / (x.std()))
all_features[mumeric_features] = all_features[mumeric_features].fillna(0)
all_features = pd.get_dummies(all_features, dummy_na=True)
print("all_feature.shape =", all_features.shape)

n_train = train_data.shape[0]
train_features = nd.array(all_features[:n_train].values)
test_features = nd.array(all_features[n_train:].values)
train_labels = nd.array(train_data.SalePrice.values).reshape((-1, 1))
