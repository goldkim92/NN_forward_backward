# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 00:38:53 2018

@author: USER
"""
#%%
import numpy as np
import random

#%%
''' Forward activation function'''
def sigmoid(x):
    s = 1 / (1+np.exp(-x))

    return s

#%%
''' Backward activation function'''
def sigmoid_grad(s):
    """ input: s = sigmoid(x) """
    ds = s * (1 - s)
    
    return ds

#%%
''' Softmax function '''
def softmax(x):
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        const = np.amax(x, 1, keepdims=True)  # along col
        exp = np.exp(x - const)
        exp_sum = np.sum(exp, 1, keepdims=True)  # along col
        x = exp / exp_sum
    else:
        # Vector
        const = -x.max()
        normal = x + const
        exp = np.exp(normal)
        exp_sum = np.sum(exp)
        x = exp / exp_sum

    assert x.shape == orig_shape
    return x

#%%
def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    [Inputs]
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### forward propagation
    z1 = np.matmul(data,W1) + b1 # M x H
    h = sigmoid(z1) # M x H
    z2 = np.matmul(h, W2) + b2 # M x Dy
    y = labels # M x Dy
    CE = np.sum(y * np.log(z2), axis=1) # M
    cost = np.sum(CE) # scalar

    ### backward propagation
    d1 = z2 - y # M x Dy
    gradW2 = np.matmul(np.transpose(h), d1) # H x Dy
    gradb2 = np.sum(d1, axis=0) # 1 x Dy
    d2 = np.matmul(d1, np.transpose(W2)) # M x H
    d3 = d2 * sigmoid_grad(sigmoid(z1))  # M x H
    gradW1 = np.matmul(data.T, d3) # Dx x H
    gradb1 = np.sum(d3, axis=0) # 1 x H

    ### Stack gradients 
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad

#%%
N = 20
dimensions = [10, 5, 10]
data = np.random.randn(N, dimensions[0])   # each row will be a datum
labels = np.zeros((N, dimensions[2]))
for i in range(N):
    labels[i, random.randint(0,dimensions[2]-1)] = 1

params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
    dimensions[1] + 1) * dimensions[2], )

cost, grad = forward_backward_prop(data, labels, params, dimensions)
