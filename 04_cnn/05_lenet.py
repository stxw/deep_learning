#!/usr/bin/python3

import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as g_loss
from mxnet.gluon import data as g_data
from mxnet.gluon import nn
import time
import d2lzh as d2l

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
mnist_train = g_data.vision.FashionMNIST(train=True)
mnist_test = g_data.vision.FashionMNIST(train=False)
transformer = g_data.vision.transforms.ToTensor()
batch_size = 256
train_iter = g_data.DataLoader(mnist_train.transform_first(transformer), \
	batch_size=batch_size, shuffle=True, num_workers=0)
test_iter = g_data.DataLoader(mnist_test.transform_first(transformer), \
	batch_size=batch_size, shuffle=False, num_workers=0)

def try_gpu():
	try:
		ctx = mx.gpu()
		_ = nd.zeros((1,), ctx=ctx)
	except mx.base.MXNetError:
		ctx = mx.cpu()
	return ctx

ctx = try_gpu()
print(ctx)

def evaluate_accuracy(data_iter, net, ctx):
	acc_sum, n = nd.array([0], ctx=ctx), 0
	for x, y in data_iter:
		x, y = x.as_in_context(ctx), y.as_in_context(ctx).astype('float32')
		acc_sum += (net(x).argmax(axis=1) == y).sum()
		n += y.size
		return acc_sum.asscalar() / n

def train(net, train_iter, test_iter, batch_size, trainer, num_epochs, ctx):
	print("training on", ctx)
	loss = g_loss.SoftmaxCrossEntropyLoss()
	for epoch in range(num_epochs):
		train_l_sum = 0.0
		train_acc_sum = 0.0
		n = 0
		start = time.time()
		for x, y in train_iter:
			x = x.as_in_context(ctx)
			y = y.as_in_context(ctx)
			with autograd.record():
				y_hat = net(x)
				l = loss(y_hat, y)
			l.backward()
			trainer.step(batch_size)
			train_l_sum += l.sum().asscalar()

			y = y.astype('float32')
			train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
			n += y.size
		test_acc = evaluate_accuracy(test_iter, net, ctx)
		print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
			% (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc, 
			time.time() - start))

lr = 0.9
num_epochs = 5
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train(net, train_iter, test_iter, batch_size, trainer, num_epochs, ctx)