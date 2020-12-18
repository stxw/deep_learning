# coding=utf-8
import os, sys
import time
import mxnet as mx
from mxnet import gluon, init, nd, autograd
from mxnet.gluon import nn
from mxnet.gluon import data as g_data
from mxnet.gluon import loss as g_loss

def batch_norm(x, gamma, beta, moving_mean, moving_var, momentum=0.9, eps=1e-5):
	if not autograd.is_training():
		x_hat = (x - moving_mean) / nd.sqrt(moving_var + eps)
	else:
		assert len(x.shape) in (2, 4)
		if len(x.shape) == 2:
			mean = x.mean(axix=0)
			var = ((x - mean) ** 2).mean(axis=0)
		else:
			mean = x.mean(axix=(0, 2, 3), keepdims=True)
			var = ((x - mean) ** 2).mean(axix=(0, 2, 3), keepdims=True)
		x_hat = (x - mean) / nd.sqrt(var + eps)
		moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
		moving_var = momentum * moving_var + (1.0 - momentum) * var
	y = gamma * x_hat + beta
	return y, moving_mean, moving_var

class BatchNorm(nn.Block):
	def __init__(self, mun_features, num_dims, **kwargs):
		super(BatchNorm, self).__init__(**kwargs)
		if num_dims = 2:
			shape = (1, num_features)
		else:
			shape = (1, num_features, 1, 1)
		self.gamma = self.params.get("gamma", shape=shape, init=init.One())
		self.beta = self.params.get("beta", shape=shape, init=init.Zero())
		self.moving_mean = nd.zeros(shape)
		self.moving_var = nd.zeros(shape)
	
	def forward(self, x):
		if self.moving_mean.context != x.context:
			self.moving_mean = self.moving_mean.copyto(x.context)
			self.moving_var = self.moving_var.copyto(x.context)
		y, self.moving_mean, self.moving_var = batch_norm(
			x, self.gamma.data(), self.beta.data(), self.moving_mean, self.moving_var)
		return y
