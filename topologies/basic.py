#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2019, RISE ETC AB
#
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL RISE 
# ETC AB BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION 
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Except as contained in this notice, the name of RISE ETC AB shall not be
# used in advertising or otherwise to promote the sale, use or other dealings 
# in this Software without prior written authorization from RISE ETC AB.

"""
.. module:: basic
   :platform: Unix, Windows
   :synopsis: Contains low-level convenience wrappers for neural network 
       blocks.

.. moduleauthor:: Paul Lucas <pal.toth@ri.se>

Contains low-level convenience wrappers for neural network building blocks. All
building blocks return Callables. When called on Tensorflow tensors, these
return tensors that are the inputs acted upon by the block. When used in Keras
models, these blocks list their layers separately.
The module provides the following functionality:

================================  =============  ==============================
name                              type           summary
================================  =============  ==============================
input_layer                       method         An alias for the Input layer 
                                                 of Keras.
conv2d                            method         2D convolution layer with 
                                                 optional activation and batch.
                                                 normalization.
compression                       method         1 X 1 convolution layer with  
                                                 optional activation and batch
                                                 normalization.
================================  =============  ==============================

"""

from tensorflow import Tensor
from tensorflow.keras import layers
from typing import Tuple, Callable, Optional


def input_layer(shape: Tuple):
    """ An alias for keras.layers.Input, just to be consistent.
    
    :param shape: The input shape.
    :type shape: Tuple
    :returns: An Input layer.
    :rtype: keras.layers.Layer
    
    """
    return layers.Input(shape)


def conv2d(
        filters: int = 64,
        kernel_size: Tuple[int, int] = (3, 3),
        padding: str = "same",
        strides: Tuple[int, int] = (1, 1),
        batchnorm: bool = True,
        activation: Optional[str] = "relu",
        upsample: bool = False
    ) -> Callable:
    """ Basic convolutional layer with optional batchnorm and upsampling.
    
    Returns a Callable that behaves like a Keras layer.
    
    :param filters: Number of convolution filters.
    :type filters: int
    :param kernel_size: Tuple of kernel size.
    :type kernel_size: Tuple[int, int]
    :param padding: Padding keyword argument of tf.keras.layers.Conv2D.
    :type padding: str
    :param strides: Strides keyword argument of tf.keras.layers.Conv2D.
    :type strides: Tuple[int, int]
    :param batchnorm: If True, adds a batch normalization layer after the
        activation layer.
    :type batchnorm: bool
    :param upsample: If True, adds transposed convolution instead of regular
        convolution ("learnable upsampling").
    :type upsample: bool
    :returns: A Callable that acts like a Keras layer. Calling it on a
        Tensorflow tensor returns a tensor that is the input acted upon be the
        block.
    :rtype: Callable
    
    """

    def inner(input_: Tensor) -> Tensor:
        if upsample:
            output = layers.Conv2DTranspose(
                    filters,
                    kernel_size,
                    strides=strides,
                    padding=padding
                )(input_)
        else:
            output = layers.Conv2D(
                    filters,
                    kernel_size,
                    strides=strides,
                    padding=padding
                )(input_)
        if activation is not None:
            output = layers.Activation(activation)(output)
        if batchnorm:
            output = layers.BatchNormalization(momentum=0.9)(output)
        return output

    return inner


def compression(
        filters: int = 64,
        padding: str = "same",
        strides: Tuple[int, int] = (1, 1),
        batchnorm: bool = True,
        activation: Optional[str] = None,
        upsample: bool = False
    ) -> Callable:
    """ Basic 1 X 1 convolutional layer with optional batchnorm and upsampling.
    
    Returns a Callable that behaves like a Keras layer.
    
    :param filters: Number of convolution filters.
    :type filters: int
    :param kernel_size: Tuple of kernel size.
    :type kernel_size: Tuple[int, int]
    :param padding: Padding keyword argument of tf.keras.layers.Conv2D.
    :type padding: str
    :param strides: Strides keyword argument of tf.keras.layers.Conv2D.
    :type strides: Tuple[int, int]
    :param batchnorm: If True, adds a batch normalization layer after the
        activation layer.
    :type batchnorm: bool
    :param upsample: If True, adds transposed convolution instead of regular
        convolution ("learnable upsampling").
    :type upsample: bool
    :returns: A Callable that acts like a Keras layer. Calling it on a
        Tensorflow tensor returns a tensor that is the input acted upon be the
        block.
    :rtype: Callable
    
    """

    def inner(input_: Tensor) -> Tensor:
        if upsample:
            output = layers.Conv2DTranspose(
                    filters,
                    (1,1),
                    strides=strides,
                    padding=padding
                )(input_)
        else:
            output = layers.Conv2D(
                    filters,
                    (1,1),
                    strides=strides,
                    padding=padding
                )(input_)
        if activation is not None:
            output = layers.Activation(activation)(output)
        if batchnorm:
            output = layers.BatchNormalization(momentum=0.9)(output)
        return output

    return inner