#!/usr/bin/python3

from mxnet import init, nd, gluon
from mxnet.gluon import nn

# 多输入通道
def corr2d(x, k):
	h, w = k.shape
	y = nd.zeros((x.shape[0] - h + 1, x.shape[1] - w + 1))
	for i in range(y.shape[0]):
		for j in range(y.shape[1]):
			y[i, j] = (x[i: i + h, j: j + w] * k).sum()
	return y

def corr2d_multi_in(x, k):
	y = list()
	for x_, k_ in zip(x, k):
		y.append(corr2d(x_, k_))
	return nd.add_n(*y)

x = nd.array([[[0,1,2], [3,4,5], [6,7,8]],[[1,2,3], [4,5,6], [7,8,9]]])
k = nd.array([[[0,1], [2,3]], [[1,2], [3,4]]])
print(x.shape, k.shape)
y = corr2d_multi_in(x, k)
print(y)

# 多输出通道
def corr2d_multi_in_out(x, k):
	y = list()
	for k_ in k:
		y.append(corr2d_multi_in(x, k_))
	return nd.stack(*y)

k = nd.stack(k, k + 1, k + 2)
y = corr2d_multi_in_out(x, k)
print(y)