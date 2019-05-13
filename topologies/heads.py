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
.. module:: heads
   :platform: Unix, Windows
   :synopsis: Contains top-level blocks ('task blocks') of neural networks.

.. moduleauthor:: Paul Lucas <pal.toth@ri.se>

This module contains implementations of 'task blocks' of neural networks. Task
blocks determine the task that the network will carry out. Tasks are normally
one of classification, regression or segmentation (a special case of 
classification). 

Currently, the module contains namespaces for main machine learning tasks.

.. note::
    All blocks return Callables. The block methods are used to parametrize the
    returned Callables. The returned Callables act on Tensorflow tensors and
    syntactically behave like Keras layers. Unlike custom Keras layers, 
    summaries of Keras models made from the blocks still list all low-level 
    layers individually.

.. todo::
    Refactor namespaces into submodules.           

The module provides the following functionality:
    
================================  =============  ==============================
name                              type           summary
================================  =============  ==============================
Classification                    class          A namespace that contains task
                                                 blocks for classification.
~.global_aggr_mlp                 method         A task block for classifying
                                                 image data in a fully
                                                 convolutional manner.
Segmentation                      class          A namespace that contains task
                                                 blocks for image segmentation.
~.fcn                             method         A task block for pixelwise
                                                 classification.
Regression                        class          A namespace that contains task
                                                 blocks for regression.
~.scalar_regression_mlp           method         A simple MLP block for 
                                                 continuous scalar regression.
================================  =============  ==============================

"""

import tensorflow as tf
from tensorflow.keras import layers
from typing import Tuple, Callable, Optional


class Classification:
    """ A namespace for classification task blocks.
    
    """
    
    @staticmethod
    def global_aggr_mlp(
            num_classes: int = 3,
            num_steps: int = 3,
            num_units: Optional[int] = None,
            batchnorm: bool = True,
            dropout: Optional[float] = None,
            aggr_fn: str = "max",
            activation: str = "relu",
            aux_inputs: Optional[Tuple[tf.Tensor, ...]] = None
        ) -> Callable:
        """ Global maximum pooling head that can be placed on top of existing
        models. Flattens global maxed features and projects by an MLP. This
        block is for classifying image data.
        
        Global pooling reduces tensors feature-wise, returning the highest 
        activation. Global pooling discards spatial localization information, 
        thus it is used with fully convolutional architectures. It introduces
        translation invariance for the same reason. The output of the head is a
        logistic layer for classification. An MLP connects the global pooled
        features with the logistic layer. Optional Dropout strategy is based on 
        `Li et al. <https://arxiv.org/pdf/1801.05134.pdf/>`_.
        
        :param num_classes: Number of classes in the final logistic layer.
        :type num_classes: int
        :param num_steps: The number of hidden layers in the MLP.
        :type num_steps: int
        :param num_units: Optional, the number of units in the first MLP layer.
            If not provided, num_units in the first layer is calculated by 
            geometric scaling based on the incoming features.
        :type num_units: Optional[int]
        :param batchnorm: If True, batch normalization layers are inserted 
            after the MLP layers' activation.
        :param aggr_fn: String, specifies the type of global pooling. Accepted 
            values are: "max" (maxpool) and "avg" (average pool).
        :type aggr_fn: str
        :param aux_inputs: An optional tuple of Tensorflow Tensor objects that 
            are merged into the first MLP layer.
        :type aux_inputs: Optional[tensorflow.Tensor]
        :param activation: Type of activation function to use on MLP layers. 
            Accepted values: "relu", "elu", "selu", etc (valid keras 
            activations).
        :type activation: str
        :returns: A Callable. If called on a Tensorflow tensor, it returns
            another Tensorflow tensor.
        :rtype: Callable
        
        """
        aggr_fns = {"max": layers.GlobalMaxPooling2D, 
                    "avg": layers.GlobalAveragePooling2D}
        if aggr_fn in aggr_fns:
            aggr_fn = aggr_fns[aggr_fn]()
        else:
            raise ValueError("Aggregating function not supported. "
                             "Valid values for aggr_fn are: {}.".format(
                                ", ".join(aggr_fns.keys())))

        def inner(input_):
            output = aggr_fn(input_)
            if aux_inputs is not None:
                output = layers.Concatenate()([output, *aux_inputs])
            compression = ((output.shape.as_list()[-1] / num_classes)
                           ** (-1 / num_steps))
            for i in range(num_steps - 1):
                if i == 0:
                    if num_units is None:
                        num_features = int(output.shape.as_list()[-1]
                                           * compression)
                    else:
                        num_features = num_units
                else:
                    num_features = int(output.shape.as_list()[-1] 
                        * compression)
                output = layers.Dense(num_features)(output)
                output = layers.Activation(activation)(output)
                if batchnorm:
                    output = layers.BatchNormalization(momentum=0.9)(output)
            if dropout is not None:
                output = layers.Dropout(dropout)(output)
            output = layers.Dense(num_classes)(output)
            output = layers.Activation("softmax")(output)
            return output
    
        return inner


class Segmentation:
    
    @staticmethod
    def fcn(
            num_classes: int = 3,
            num_steps: int = 3,
            num_units: Optional[int] = None,
            batchnorm: bool = True,
            dropout: Optional[float] = None,
            activation: str = "relu"
        ) -> Callable:
        """Adds a network-in-a-network-style CNN classification top layer. This
        is for segmenting image data.
        
        Network-in-network-stype FCN head. Conserves spatial input 
        dimensionality. Based on 
        `Lin et al. <https://arxiv.org/pdf/1312.4400.pdf/>`_. Optional Dropout 
        strategy is based on 
        `Li et al. <https://arxiv.org/pdf/1801.05134.pdf/>`_.
        
        :param num_classes: Number of classes in the final logistic layer.
        :type num_classes: int
        :param num_steps: The number of hidden layers in the MLP.
        :type num_steps: int
        :param num_units: Optional, the number of units in the first MLP layer.
            If not provided, num_units in the first layer is calculated by 
            geometric scaling based on the incoming features.
        :type num_units: Optional[int]
        :param batchnorm: if True, batch normalization layers are inserted 
            after the MLP layers' activation.
        :type batchnorm: bool
        :param activation: Type of activation function to use on NIN layers. 
            Accepted values: "relu", "elu", "selu", etc (valid keras 
            activations).
        :type activation: str
        :returns: A Callable. If called on a Tensorflow tensor, it returns
            another Tensorflow tensor.
        :rtype: Callable
        
        """
        def inner(input_):
            output = input_
            compression = ((output.shape.as_list()[-1] / num_classes)
                           ** (-1 / num_steps))
            for i in range(num_steps - 1):
                if i == 0:
                    if num_units is None:
                        num_features = int(output.shape.as_list()[-1]
                                           * compression)
                    else:
                        num_features = num_units
                else:
                    num_features = int(output.shape.as_list()[-1] 
                        * compression)
                output = layers.Conv2D(num_features, (1,1), 
                                       padding="same")(output)
                output = layers.Activation(activation)(output)
                if batchnorm:
                    output = layers.BatchNormalization(momentum=0.9)(output)
            if dropout is not None:
                output = layers.Dropout(dropout)(output)
            output = layers.Conv2D(num_classes, (1,1), padding="same")(output)
            output = layers.Activation("softmax")(output)
            return output
    
        return inner


class Regression:
    
    @staticmethod
    def scalar_regression_mlp(
            num_out: int = 1,
            flatten: bool = True,
            num_units: Optional[int] = 50,
            num_layer: int = 2,
            activation: str = "relu",
            batchnorm: bool = True,
            dropout: Optional[float] = 0.2
        ) -> Callable:
        """ Simple MLP regression head. The number of hidden MLP units is
        calculated by geometric scaling.
        
        :param num_out: The number of scalars to regress.
        :type num_out: int
        :param flatten: If True, flattens the input tensor.
        :type flatten: bool
        :param num_units: The number of units in the hiddent layers. If not
            specified, the number of units in each layer will be calculated
            as a geometric progression between the numbers of features in the
            input and output.
        :type num_units: Optional[int]
        :param num_layer: The number of hidden layers.
        :type num_layer: int
        :param activation: The name of the activation on the hidden layers.
        :type activation: str
        :param batchnorm: If True, batchnorm layers are inserted after the
            activations of hidden layers.
        :type batchnorm: bool
        :param dropout: Optional, if specified, a single Dropout layer will be
            added before the output with the specified rate.
        :type dropout: Optional[float]
        
        """
        def inner(input_):
            if flatten:
                output = layers.Flatten()(input_)
            else:
                output = input_
            compression = ((output.shape.as_list()[-1] / num_out)
                           ** (-1 / num_layer))
            for i in range(num_layer - 1):
                if i == 0:
                    if num_units is None:
                        num_features = int(output.shape.as_list()[-1]
                                           * compression)
                    else:
                        num_features = num_units
                else:
                    num_features = int(output.shape.as_list()[-1] 
                        * compression)
                output = layers.Dense(num_features)(output)
                output = layers.Activation(activation)(output)
                if batchnorm:
                    output = layers.BatchNormalization(momentum=0.9)(output)
            if dropout is not None:
                output = layers.Dropout(dropout)(output)
            output = layers.Dense(num_out)(output)
            return output
        
        return inner
