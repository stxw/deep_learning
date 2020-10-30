#!/usr/bin/python3

import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as g_loss
from mxnet.gluon import nn
import time

# LeNet模型
net = nn.Sequential()
net.add(
	nn.Conv2D(channels=6, kernel_size=5, activation='sigmoid'),
	nn.MaxPool2D(pool_size=2, strides=2),
	nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
	nn.MaxPool2D(pool_size=2, strides=2),
	nn.Dense(120, activation='sigmoid'),
	nn.Dense(84, activation='sigmoid'),
	nn.Dense(10)
)

x = nd.random.uniform(shape=(1, 1, 28, 28))
net.initialize()
for layer in net:
	x = layer(x)
	print(layer.name, 'output shape:\t', x.shape)

# 获取数据和训练模型
def try_gpu():
	try:
		ctx = mx.gpu()
		_ = nd.zeros((1,), ctx=ctx)
	except mx.base.MXNetError:
		ctx = mx.cpu()
	return ctx

ctx = try_gpu()
print(ctx)