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
.. module:: bodies
   :platform: Unix, Windows
   :synopsis: Contains mid-level blocks ('transform blocks') of neural
       networks.

.. moduleauthor:: Paul Lucas <pal.toth@ri.se>

This module contains implementations of 'transform blocks' of neural networks.
Transform blocks carry out feature extraction, dimensionality reduction and
other data transformation before passing the tensor to task blocks. The type of
transformation is primarily dictated by the type of data and secondarily by the
task.

Currently, the module contains namespaces that each contain blocks for 
transforming different types of data.

.. note::
    Most blocks return Callables. The block methods are used to parametrize the
    returned Callables. The returned Callables act on Tensorflow tensors and
    syntactically behave like Keras layers. Unlike custom Keras layers, 
    summaries of Keras models made from the blocks still list all low-level 
    layers individually. Blocks that do not return a Callable but act otherwise
    have this explicitly stated in their documentation.

.. note::
    In the docstrings of many blocks, it is practical to describe the block as
    a sequence of layers. In those cases, layers will be abbreviated as
    follows:
        * Convolution 2D -- ``C2``
        * Convolution 1D -- ``C1``
        * Strided convolution 2D -- ``C2s``
        * Strided convolution 1D -- ``C1s``
        * Transposed convolution 2D -- ``C2t``
        * Transposed convolution 1D -- ``C1t``
        * 1 X 1 convolution 2D -- ``NIN2``
        * 1 X 1 convolution 1D -- ``NIN1``
        * Dense -- ``D``
        * BatchNormalization -- ``bn``
        * Activation, any except softmax or sigmoid -- ``A``
        * Activation, softmax or sigmoid -- ``AC``
        * Dropout -- ``dr``
        * Pooling 2D -- ``P2``
        * Pooling 1D -- ``P1``
        * Upsampling 2D -- ``U2``
        * Upsampling 1D -- ``U1``
        * Input -- ``I``
        * Add -- ``add``
        * Concatenation -- ``concat``
        * Repeating layers -- ``k X ( ... )`` -- the structure in parentheses 
          is repeated sequentially k times.
        * Additional named layer inputs on top of the sequential input are 
          listed in angle brackets.
        * Referencing is given in square brackets. E.g., ``add<ref1>[ref2]`` 
          means that the result of the sequential input and a previously 
          referenced tensor ``ref1`` are added, then the result is named 
          ``ref2``.
        
.. note::
    BatchNorm - Activation orders are reversed compared to base 
    implementations. See 
    `here <https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md/>`_

.. todo::
    Refactor namespaces into submodules.           

The module provides the following functionality:
    
================================  =============  ==============================
name                              type           summary
================================  =============  ==============================
ImageData                         class          A namespace that contains 
                                                 transform blocks for working
                                                 with image data.
~.dcnn                            method         A vanilla deep convolutional
                                                 reduction block.
~.dcnn_upsampling                 method         Deep convolutional upsampling
                                                 block that increases spatial
                                                 size.
~.resnet                          method         Residual network reduction
                                                 block.
~.resnet_upsampling               method         Residual network upsampling
                                                 block that increases spatial
                                                 size.
~.segnet                          method         Deep convolutional reduction
                                                 followed by deep convolutional
                                                 upsampling. Used for 
                                                 segmentation.
~.pyramid_pooling_net             method         Spatial Pyramid Pooling
                                                 reduction block.
~.nin                             method         Network-in-network reduction
                                                 block.
~.dense_net                       method         DenseNet reduction and
                                                 (optionally) upsampling.
ProcessData                       class          A namespace that contains 
                                                 transform blocks for handling
                                                 process data.
~.feature_reduction_mlp           method         Reduces features 
                                                 observation-wise using a
                                                 shared MLP.
~.temporal_reduction_conv         method         Reduces both features and
                                                 temporal neighborhoods using
                                                 1D convolution. Utilizes 
                                                 information in the interaction
                                                 of features and their temporal 
                                                 patterns.
~.take_feature                    method         Special block that takes a
                                                 single feature (column) from
                                                 input data.
~.split_features                  method         Special block that separates
                                                 features (columns) so that
                                                 interaction between features
                                                 is not propagated down the
                                                 network.
~.pca                             method         Special block that reduces
                                                 features by using an
                                                 autoencoder. This block must
                                                 be trained separately.
~.conv_leg                        method         1D convolutional reduction
                                                 with no interaction between
                                                 features, only temporal 
                                                 patterns.
~.randomized_feature_bagger       method         Dropout for inputs, meant to
                                                 reduce effects of 
                                                 collinearity.
~.reducer_regressor               method         Combines feature_reduction_mlp
                                                 and temporal_reduction_conv
                                                 with optional PCA and bagging.
~.memoriless_engineeredstats      method         Extracts the mean and std
                                                 columnwise, from temporal 
                                                 histories of features.
~.memoriless_mlp                  method         Simple MLP on present values
                                                 only, no memory.
================================  =============  ==============================

"""

import basic
from tensorflow import Tensor
from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Tuple, Callable, Optional


class ImageData:
    """ A namespace that contains blocks for transforming image data. Image
    data follows Keras' 'channels-last' format convention, i.e., the size of
    image data is n_batch X width X height X channels.
    
    """
    
    
    @staticmethod
    def dcnn(
            num_filters: int = 16,
            kernel_shape: Tuple[int, int] = (3, 3),
            num_block: int = 1,
            num_pool: int = 3,
            pool_expansion: float = 2.0,
            activation: str = "elu",
            batchnorm: bool = True
        ) -> Callable:
        """ Returns a vanilla DCNN feature extractor.
        
        Block structure:
            I -> num_pool X (num_block X (C2, A, BN), P2), C2, A, BN 
        
        Adds subsequent convolutional blocks with num_block convolutional 
        units. After each convolutional block, a pooling layer is added. This
        block reduces image size (spatial size). Each pooling block outputs
        a spatial size of (width/2, height/2), where
        (width, height) is the spatial size output by the previous block. The
        final image size is therefore
        (width_original / (2*num_pool),
        height_original / (2*num_pool)). The block grows the number of features
        simultaneously. The total 'volume' of the tensors is conserved if
        pool_expansion == 2.
        
        :param num_filters: The number of filters in the first convolutional
            unit. The number of filters in subsequent layers are calculated
            as a geometric series with the exponent pool_expansion.
        :type num_filters: int
        :param kernel_shape: The size of the convolutional kernels.
        :type kernel_shape: Tuple[int, int]
        :param num_block: The number of convolutional blocks between pooling
            layers.
        :type num_block: int
        :param num_pool: The number of pooling layers.
        :type num_pool: int
        :param pool_expansion: The exponent in the geometric series of the
            number of convolutional filters.
        :type pool_expansion: float
        :param activation: The name of the activation function to use.
        :type activation: str
        :param batchnorm: If True, BatchNormalization layers will be inserted
            after activation layers.
        :type batchnorm: bool
        :returns: A Callable. Takes a Tensorflow tensor and returns another
            tensor that is the input acted upon by the block.
        :rtype: Callable

        """
        def inner(input_: Tensor) -> Tensor:
            for j in range(num_pool):
                for i in range(num_block):
                    if i == 0 and j == 0:
                        output = layers.Conv2D(
                                int(num_filters * pool_expansion ** j),
                                kernel_shape,
                                padding="same"
                            )(input_)
                    else:
                        output = layers.Conv2D(
                                int(num_filters * pool_expansion ** j),
                                kernel_shape,
                                padding="same"
                            )(output)
                    output = layers.Activation(activation)(output)
                    if batchnorm:
                        output = layers.BatchNormalization(
                                momentum=0.9)(output)
                output = layers.AveragePooling2D(pool_size=(2, 2), 
                                                 padding="same")(output)
            for _ in range(num_block):
                output = layers.Conv2D(
                        int(num_filters * pool_expansion ** num_pool),
                        kernel_shape,
                        padding="same"
                    )(output)
                output = layers.Activation(activation)(output)
                if batchnorm:
                    output = layers.BatchNormalization(momentum=0.9)(output)
            return output
        
        return inner
    
    
    @staticmethod
    def dcnn_upsampling(
            num_features: Optional[int] = 16,
            kernel_shape: Tuple[int, int] = (3, 3),
            num_block: int = 1,
            num_pool: int = 3,
            pool_expansion: float = 2.0,
            activation: str = "relu",
            batchnorm: bool = True
        ) -> Callable:
        """ Returns a vanilla DCNN upsampler ('learnable upsampling').
        
        Block structure:
            I -> num_pool X (U2, num_block X (C2, A, BN))
        
        Adds num_pool upsampling blocks that consist of an upsampling layer
        followed by num_block convolutional units. This block grows image size.
        Each pooling block grows the image size so that the output image size
        is (width*2, height*2), if (width, height) is the image size output by
        the previous block. The number of features is reduced simultaneously.
        The 'volume' of the tensors is conserved if pool_expansion == 2.
        
        :param num_features: The number of filters in the first convolutional
            unit. The number of filters in subsequent layers are calculated
            as a geometric series with the exponent pool_expansion. If not
            specified, the number of filters is calculated based on the number
            of features in the input tensor.
        :type num_features: Optional[int]
        :param kernel_shape: The size of the convolutional kernels.
        :type kernel_shape: Tuple[int, int]
        :param num_block: The number of convolutional blocks between upsampling
            layers.
        :type num_block: int
        :param num_pool: The number of upsampling layers.
        :type num_pool: int
        :param pool_expansion: The exponent in the geometric series of the
            number of convolutional filters.
        :type pool_expansion: float
        :param activation: The name of the activation function to use.
        :type activation: str
        :param batchnorm: If True, BatchNormalization layers will be inserted
            after activation layers.
        :type batchnorm: bool
        :returns: A Callable. Takes a Tensorflow tensor and returns another
            tensor that is the input acted upon by the block.
        :rtype: Callable

        """
        
        def inner(input_: Tensor) -> Tensor:
            if num_features is None:
                num_filters_ = input_.shape.as_list()[-1]
            else:
                num_filters_ = num_features
            for j in range(num_pool):
                if j == 0:
                    output = input_
                output = layers.UpSampling2D()(output)
                for _ in range(num_block):
                    if num_features is None:
                        num_features_ = int(num_filters_ / pool_expansion 
                                            ** (j + 1))
                    else:
                        num_features_ = int(num_filters_ * pool_expansion 
                                            ** (num_pool-j-1))
                    # num_units is calculated by a geometric scaling rule.
                    output = layers.Conv2D(
                            num_features_,
                            kernel_shape,
                            padding="same"
                        )(output)
                    output = layers.Activation(activation)(output)
                    if batchnorm:
                        output = layers.BatchNormalization(
                                momentum=0.9)(output)
            return output
        
        return inner
    
    
    @staticmethod
    def resnet(
            stem: bool = True,
            stem_num_features: int = 64,
            stem_pool: bool = True,
            stem_activation: str = "relu",
            num_res_block: int = 3,
            res_conv_num_features: int = 64,
            res_conv_kernel_size: Tuple[int, int] = (3, 3),
            res_conv_activation: str = "relu",
            res_bottleneck_num_features: int = 128,
            res_bottleneck_activation: str = "relu",
        ) -> Callable:
        """ Residual block. The implementation follows `He at al.
        <https://arxiv.org/abs/1406.4729>`_ in the sense that no subsequent
        reduction blocks are included apart from a stem block. Use a resnet
        block i.e., to extract features for a pyramid pooling network.
        
        Block structure (without stem block):
            I[inp] -> num_res_block X (C2, NIN2, NIN2, add<inp>[inp])
        
        Stem block structure:
            I -> C2s, P2
        
        :param stem: If True, a stem block is inserted after the input.
        :type stem: bool
        :param stem_num_features: The number of filters in the strided 
            convolution in the stem block.
        :type stem_num_features: int
        :param stem_pool: If True, a 2X2 maxpooling layer is added after the
            strided convolution in the stem block.
        :type stem_pool: bool
        :param stem_activation: The name of the activation function to use
            after the strided convolution in the stem block.
        :type stem_activation: str
        :param num_res_block: The number of residual blocks.
        :type num_res_block: int
        :param res_conv_num_features: The number of filters in convolutional
            layers in residual blocks.
        :type res_conv_num_features: int
        :param res_conv_kernel_size: The kernel size in convolutional layers in
            residual blocks.
        :type res_conv_kernel_size: Tuple[int, int]
        :param res_conv_activation: The name of activation used after
            convolutional layers in residual blocks.
        :type res_conv_activation: str
        :param res_bottleneck_num_features: The number of filters in bottleneck
            layers in the residual blocks.
        :type res_bottleneck_num_features: int
        :param res_bottleneck_activation: The name of activation to use after
            bottleneck layers in residual blocks.
        :type res_bottleneck_activation: str
        :returns: A Callable. Takes a Tensorflow tensor and returns another
            tensor that is the input acted upon by the block.
        :rtype: Callable
        
        """
        
        def res_stem() -> Callable:
            
            def inner(input_: Tensor) -> Tensor:
                output = basic.conv2d(
                        stem_num_features,
                        strides=(2,2),
                        activation=stem_activation
                    )(input_)
                if stem_pool:
                    output = layers.MaxPooling2D(padding="same")(output)
                return output
            
            return inner
        
        def res_block() -> Callable:
            
            def inner(input_: Tensor) -> Tensor:
                input_num_features = input_.shape.as_list()[-1]
                output = basic.conv2d(
                        res_conv_num_features,
                        res_conv_kernel_size,
                        activation=res_conv_activation
                    )(input_)
                output = basic.compression(
                        res_bottleneck_num_features,
                        activation=res_bottleneck_activation
                    )(output)
                output = basic.compression(
                        input_num_features,
                        activation=res_bottleneck_activation
                    )(output)
                return layers.Add()([input_, output])
            
            return inner
        
        def inner(input_: Tensor) -> Tensor:
            
            if stem:
                output = res_stem()(input_)
            else:
                output = input_
            for _ in range(num_res_block):
                output = res_block()(output)
            return output
        
        return inner
                    
   
    @staticmethod
    def resnet_upsampling(
            num_upsampling: int = 2,
            res_conv_kernel_size: Tuple[int, int] = (3, 3),
            res_upsample_activation: str = "relu",
            res_upsample_num_features: int = 64
        ) -> Callable:
        """ ResNet-like upsampling block.
        
        This block grows image size. Each block outputs a (width*2, height*2)
        image size tensor, where (width, height) is the image size of the
        output of the previous layer. The number of features stays constant.
        
        Block structure:
            I -> U2[upsampled]
            I -> num_upsampling X (C2s, NIN2, NIN2, add<upsampled>[upsampled])
        
        :param num_upsampling: The number of upsampling blocks.
        :type num_upsampling: int
        :param res_conv_kernel_size: A the kernel size of the convolutional
            layer (non-compression) in each block.
        :type res_conv_kernel_size: Tuple[int, int]
        :param res_upsample_activation: The name of the activation function
            used after convolutional layers in each block.
        :type res_upsample_activation: str
        :param res_upsample_num_features: The number of filters in the
            middle compression layer in each block.
        :type res_upsample_num_features: int
        :returns: A Callable. Takes a Tensorflow tensor and returns another
            tensor that is the input acted upon by the block.
        :rtype: Callable
            
        """
        
        def res_upsampling() -> Callable:
            
            def inner(input_: Tensor) -> Tensor:
                input_num_features = input_.shape.as_list()[-1]
                output = basic.conv2d(
                        int(input_num_features/2),
                        res_conv_kernel_size,
                        activation=res_upsample_activation,
                        upsample=True,
                        strides=(2,2)
                    )(input_)
                output = basic.compression(
                        res_upsample_num_features,
                        activation=res_upsample_activation
                    )(output)
                output = basic.compression(
                        input_num_features,
                        activation=res_upsample_activation
                    )(output)
                upsampled = layers.UpSampling2D()(input_)
                output = layers.Add()([upsampled, output])
                output = basic.compression(
                        int(input_num_features/2),
                        activation=res_upsample_activation
                    )(output)
                return output
            
            return inner
        
        def inner(input_: Tensor) -> Tensor:
            output = input_
            for _ in range(num_upsampling):
                output = res_upsampling()(output)
            return output
        
        return inner
    
    
    @staticmethod
    def segnet(
            num_filters: int = 16,
            kernel_shape: Tuple[int, int] = (3, 3),
            num_block: int = 1,
            num_pool: int = 3,
            pool_expansion: float = 2.0,
            activation: str = "elu",
            batchnorm: bool = True
        ) -> Callable:
        """ Returns a SegNet body.
        
        A SegNet is a DCNN feature extractor with an upsampling branch added
        on top. The upsampling block returns the spatial dimensionality to that
        of the original image input. A feature set is computed for every pixel.
        
        :param num_filters: Number of convolutional features in the first 
            convolutional layer. The number of features in consequent 
            layers are calculated so that the number of tensor elements is 
            preserved if pool_expansion = 2.
        :type num_filters: int
        :param kernel_shape: Convolution kernel shape.
        :type kernel_shape: Tuple[int, int]
        :param num_block: number of convolutional layers between two pooling 
            layers.
        :type num_block: int
        :param num_pool: Number of pooling layers.
        :type num_pool: int
        :param pool_expansion: The number of filters in convolutional layers is 
            multiplied by this amount after each pooling layer.
        :type pool_expansion: float
        :param batchnorm: If True, batch normalization layers are inserted 
            after.
        :returns: A Callable. Takes a Tensorflow tensor and returns another
            tensor that is the input acted upon by the block.
        :rtype: Callable
        
        """
        def inner(input_: Tensor) -> Tensor:
            output = ImageData.dcnn(
                    num_filters, 
                    kernel_shape, 
                    num_block, 
                    num_pool, 
                    pool_expansion, 
                    activation,
                    batchnorm
                )(input_)
            output = ImageData.dcnn_upsampling(
                    num_filters, 
                    kernel_shape, 
                    num_block, 
                    num_pool, 
                    pool_expansion, 
                    activation,
                    batchnorm
                )(output)
            return output
        
        return inner
    
    
    @staticmethod
    def pyramid_pooling_net(
            stem_num_features: int = 64,
            stem_pool: bool = True,
            stem_activation: str = "relu",
            stem_num_res_block: int = 3,
            stem_res_conv_num_features: int = 64,
            stem_res_conv_kernel_size: Tuple = (3, 3),
            stem_res_conv_activation: str = "relu",
            stem_res_bottleneck_num_features: int = 128,
            stem_res_bottleneck_activation: str = "relu",
            num_pooled_features: int = 128,
            num_pyramid_levels: int = 4,
            pool_activation: str = "relu",
            aggr_fn: str = "max",
            pool_compression: bool = True,
            pool_compression_num_features: int = 48,
            upsampling_res_conv_kernel_size: Tuple = (3, 3),
            upsampling_res_upsample_activation: str = "relu",
            upsampling_res_upsample_num_features: int = 64
        ) -> Callable:
        """ `Spatial pyramid pooling network 
        <https://arxiv.org/abs/1406.4729>`_. Pyramid pooling operates on
        ResNet features.
        
        Block structure (excluding the ResNet):
            I -> P2, NIN2, U2[scale1]
            I -> P2, NIN2, U2[scale2]
            I -> P2, NIN2, U2[scale3]
            ...
            I -> P2[scale0], NIN2, concat<scale1, scale2,...>
        
        .. note::
            Pooling is used instead of bilinear subsampling. This is a
            simplification justified by the fixed image size in this case.
        
        :param stem_num_features: The number of filters in the strided 
            convolution in the stem block.
        :param stem_pool: If True, a 2X2 maxpooling layer is added after the
            strided convolution in the stem block.
        :param stem_activation: The name of the activation function to use
            after the strided convolution in the stem block.
        :param stem_num_res_block: The number of residual blocks.
        :param stem_res_conv_num_features: The number of filters in convolutional
            layers in residual blocks.
        :param stem_res_conv_kernel_size: The kernel size in convolutional layers in
            residual blocks.
        :param stem_res_conv_activation: The name of activation used after
            convolutional layers in residual blocks.
        :param stem_res_bottleneck_num_features: The number of filters in bottleneck
            layers in the residual blocks.
        :param stem_res_bottleneck_activation: The name of activation to use after
            bottleneck layers in residual blocks.
        :param num_pooled_features: The number of filters in the bottleneck
            layers after pooling.
        :param num_pyramid_levels: The number of pyramid levels.
        :param pool_activation: The name of the activation after the pooling
            bottleneck layers.
        :param aggr_fn: The name of aggregating function to use in the pooling
            layers. Can be either 'max' or 'avg'.
        :param pool_compression: If True, the concatenated pooled features will
            be compressed.
        :param pool_compression_num_features: Number of features in the final
            (optional) compression layer.
        :param upsampling_res_conv_kernel_size: The kernel size of the 
            convolutional layer (non-compression) in each block.
        :param upsampling_res_upsample_activation: The name of the activation 
            function used after convolutional layers in each block.
        :param upsampling_res_upsample_num_features: The number of filters in 
            the middle compression layer in each block.
        :type stem_num_features: int = 64,
        :type stem_pool: bool,
        :type stem_activation: str
        :type stem_num_res_block: int
        :type stem_res_conv_num_features: int
        :type stem_res_conv_kernel_size: Tuple[int, int]
        :type stem_res_conv_activation: str
        :type stem_res_bottleneck_num_features: int
        :type stem_res_bottleneck_activation: str
        :type num_pooled_features: int
        :type num_pyramid_levels: int
        :type pool_activation: str
        :type aggr_fn: str
        :type pool_compression: bool
        :type pool_compression_num_features: int
        :type upsampling_res_conv_kernel_size: Tuple[int, int]
        :type upsampling_res_upsample_activation: str
        :type upsampling_res_upsample_num_features: int
        :returns: A Callable. Takes a Tensorflow tensor and returns another
            tensor that is the input acted upon by the block.
        :rtype: Callable
        
        """
        
        aggr_fns = {
            "max": layers.MaxPooling2D, 
            "avg": layers.AveragePooling2D
        }
        
        def pooling_module() -> Callable:
            
            if aggr_fn in aggr_fns:
                aggr_fn_ = aggr_fns[aggr_fn]
            else:
                raise ValueError("Aggregating function not supported. "
                                 "Valid values for aggr_fn are: {}.".format(
                                 ", ".join(aggr_fns.keys())))
            
            def inner(input_: Tensor) -> Tensor:
                features = [input_,]
                for k in range(num_pyramid_levels):
                    pooled = aggr_fn_(pool_size=(2**(k+1), 2**(k+1)))(input_)
                    pooled = basic.compression(
                            filters=int(
                                    num_pooled_features/num_pyramid_levels),
                            activation=pool_activation
                        )(pooled)
                    pooled = layers.UpSampling2D(
                            size=(2**(k+1), 2**(k+1))
                        )(pooled)
                    features.append(pooled)
                output = layers.Concatenate()(features)
                if pool_compression:
                    output = basic.compression(
                            filters=pool_compression_num_features,
                            activation=pool_activation
                        )(output)
                return output
            
            return inner
        
        def inner(input_: Tensor) -> Tensor:
            
            output = ImageData.resnet(
                    True,
                    stem_num_features,
                    stem_pool,
                    stem_activation,
                    stem_num_res_block,
                    stem_res_conv_num_features,
                    stem_res_conv_kernel_size,
                    stem_res_conv_activation,
                    stem_res_bottleneck_num_features,
                    stem_res_bottleneck_activation
                )(input_)
            output = pooling_module()(output)
            if stem_pool:
                num_upsampling = 2
            else:
                num_upsampling = 1
            output = ImageData.resnet_upsampling(
                    num_upsampling,
                    upsampling_res_conv_kernel_size,
                    upsampling_res_upsample_activation,
                    upsampling_res_upsample_num_features
                )(output)
            return output
            
        return inner
    
    
    
    @staticmethod
    def nin(
            stem: bool = True,
            stem_conv_num_features: int = 64,
            stem_conv_kernel_size: Tuple[int, int] = (3, 3),
            stem_bottleneck_num_units: int = 64,
            stem_bottleneck_num_steps: int = 2,
            stem_bottleneck_compression: float = 0.75,
            stem_activation: str = "relu",
            num_block: int = 2,
            pool_type = "max",
            pool_size: Tuple[int, int] = (3, 3),
            pool_stride: Tuple[int, int] = (2, 2),
            block_conv_kernel_size: Tuple[int, int] = (3, 3),
            block_conv_num_features: int = 64,
            block_bottleneck_num_units: int = 64,
            block_bottleneck_num_steps: int = 2,
            block_bottleneck_compression: float = 1.0,
            block_activation: str = "relu",
            neck: bool = True,
            neck_conv_kernel_size: Tuple[int, int] = (3, 3),
            neck_conv_num_features: int = 64,
            neck_bottleneck_num_units: int = 64,
            neck_bottleneck_num_steps: int = 1,
            neck_bottleneck_compression: float = 1.0,
            neck_activation: str = "relu",
            batchnorm: bool = True
        ) -> Callable:
        """ `Network-in-network <https://arxiv.org/abs/1312.4400>`_ block. This
        is a deep convolutional neural network with pixelwise shared MLP layers
        between convolutional layers.
        
        Main block structure:
            I -> C2, A, BN, (block_bottleneck_num_steps X NIN2), P2
        
        
        :param stem: If True, a stem block with pooling is added before the
            first block.
        :param stem_conv_num_features: Number of convolutional features in the
            stem block.
        :param stem_conv_kernel_size: Convolutiona kernel size in the stem
            block.
        :param stem_bottleneck_num_units: Number of bottleneck features in the
            stem block.
        :param stem_bottleneck_num_steps: Effective number of layers in
            bottleneck MLP in the stem block.
        :param stem_bottleneck_compression: Compression (geometric sequence
            exponent) in the bottleneck MLP in the stem block.
        :param stem_activation: Name of the activation in the stem block.
        :param num_block: Number of NIN blocks.
        :param pool_type: Pooling type. Acceptable values are 'max' and 'avg'.
        :param pool_size: Pool size between NIN blocks.
        :param pool_stride: Pool stride between NIN blocks.
        :param block_conv_kernel_size: Convolution kernel size in NIN blocks.
        :param block_conv_num_features: Number of convolutional features in NIN
            blocks.
        :param block_bottleneck_num_units: Number of bottleneck features in the
            NIN blocks.
        :param block_bottleneck_num_steps: Effective number of layers in
            bottleneck MLP in the NIN blocks.
        :param block_bottleneck_compression: Compression (geometric sequence
            exponent) in the bottleneck MLP in the NIN blocks.
        :param block_activation: Name of the activation in the NIN block.
        :param neck: If True, a 'neck' block is added after the last NIN block.
            A Neck block is practically a final NIN block after the last
            pooling layer.
        :param neck_conv_kernel_size: Convolution kernel size in the neck 
            block.
        :param neck_conv_num_features: Number of convolutional features in the
            neck blocks.
        :param neck_bottleneck_num_units: Number of bottleneck features in the
            neck block.
        :param neck_bottleneck_num_steps: Effective number of layers in the
            bottleneck MLP in the neck block.
        :param neck_bottleneck_compression: Compression (geometric sequence
            exponent) in the bottleneck MLP in the neck block.
        :param neck_activation: Name of the activation in the neck block.
        :param batchnorm: If True, BatchNormalization layers are inserted after
            convolutional layers. This affects all blocks.
        :type stem: bool
        :type stem_conv_num_features: int
        :type stem_conv_kernel_size: Tuple[int, int]
        :type stem_bottleneck_num_units: int
        :type stem_bottleneck_num_steps: int
        :type stem_bottleneck_compression: float
        :type stem_activation: str
        :type num_block: int
        :type pool_type: str
        :type pool_size: Tuple[int, int]
        :type pool_stride: Tuple[int, int]
        :type block_conv_kernel_size: Tuple[int, int]
        :type block_conv_num_features: int
        :type block_bottleneck_num_units: int
        :type block_bottleneck_num_steps: int
        :type block_bottleneck_compression: float
        :type block_activation: str
        :type neck: bool
        :type neck_conv_kernel_size: Tuple[int, int]
        :type neck_conv_num_features: int
        :type neck_bottleneck_num_units: int
        :type neck_bottleneck_num_steps: int
        :type neck_bottleneck_compression: float
        :type neck_activation: str
        :type batchnorm: bool
        :returns: A Callable. Takes a Tensorflow tensor and returns another
            tensor that is the input acted upon by the block.
        :rtype: Callable
        
        
        """
        
        def nin_block() -> Callable:
            
            def inner(input_: Tensor) -> Tensor:
                output = input_
                output = basic.conv2d(
                        block_conv_num_features, 
                        block_conv_kernel_size,                         
                        padding="same", 
                        batchnorm=batchnorm,
                        activation=block_activation
                    )(input_)
                for k in range(block_bottleneck_num_steps):
                    num_units = int(block_bottleneck_num_units 
                                    * block_bottleneck_compression ** (k + 1))
                    output = basic.compression(
                            filters=num_units, 
                            batchnorm=batchnorm, 
                            activation=block_activation
                        )(output)
                if pool_type == "max":
                    output = layers.MaxPool2D(
                            pool_size=pool_size, 
                            strides=pool_stride, 
                            padding="same"
                        )(output)
                elif pool_type == "avg":
                    output = layers.AveragePooling2D(
                            pool_size=pool_size, 
                            strides=pool_stride, 
                            padding="same"
                        )(output)
                else:
                    raise ValueError("The value of the pool_type argument "
                                     "must be either 'max' or 'avg' for "
                                     "MaxPooling and AveragePooling, "
                                     "respectively.")
                return output
            
            return inner
        
        def nin_stem() -> Callable:
            
            def inner(input_: Tensor) -> Tensor:
                output = input_
                output = basic.conv2d(
                        stem_conv_num_features, 
                        stem_conv_kernel_size,                         
                        padding="same", 
                        batchnorm=batchnorm,
                        activation=stem_activation
                    )(input_)
                for k in range(stem_bottleneck_num_steps):
                    num_units = int(stem_bottleneck_num_units 
                                    * stem_bottleneck_compression ** (k + 1))
                    output = basic.compression(
                            filters=num_units, 
                            batchnorm=batchnorm, 
                            activation=stem_activation
                        )(output)
                if pool_type == "max":
                    output = layers.MaxPool2D(
                            pool_size=pool_size, 
                            strides=pool_stride, 
                            padding="same"
                        )(output)
                elif pool_type == "avg":
                    output = layers.AveragePooling2D(
                            pool_size=pool_size, 
                            strides=pool_stride, 
                            padding="same"
                        )(output)
                else:
                    raise ValueError("The value of the pool_type argument "
                                     "must be either 'max' or 'avg' for "
                                     "MaxPooling and AveragePooling, "
                                     "respectively.")
                return output
            
            return inner
        
        def nin_neck() -> Callable:
            
            def inner(input_: Tensor) -> Tensor:
                output = input_
                output = basic.conv2d(
                        neck_conv_num_features, 
                        neck_conv_kernel_size,                         
                        padding="same", 
                        batchnorm=batchnorm,
                        activation=neck_activation
                    )(input_)
                for k in range(neck_bottleneck_num_steps):
                    num_units = int(neck_bottleneck_num_units 
                                    * neck_bottleneck_compression ** (k + 1))
                    output = basic.compression(
                            filters=num_units, 
                            batchnorm=batchnorm, 
                            activation=neck_activation
                        )(output)
                return output
            
            return inner
        
        def inner(input_: Tensor) -> Tensor:
            
            if stem:
                output = nin_stem()(input_)
            else:
                output = input_
            for _ in range(num_block):
                output = nin_block()(output)
            if neck:
                output = nin_neck()(output)
            return output
        
        return inner
        
    
    @staticmethod
    def dense_net(
            stem_features: int = 64,
            stem_pool: bool = False,
            num_block: int = 3,
            block_growth_rate: int = 32,
            block_bottleneck_features: int = 64,
            num_pool: int = 3,
            pool_bottleneck_compression: float = 0.5,
            upsampling: str = "dcnn",
            upsample_output_expansion: float = 1,
            grid_activation="relu",
            upsampling_activation="relu"
        ) -> Callable:
        """ DenseNet reduction block with Optional upsampling.
        
        Loosely based on `Huang et al. <https://arxiv.org/abs/1608.06993>`_.
        
        Main block structure:
            I[inp] -> num_block X (NIN2, C2, A, BN, concat<inp>[inp]), NIN2, P2
        
        .. note::
            DenseNet models are memory hungry. A stem block is added after the
            input to reduce the image size. This can help with the memory
            requirements.
        
        :param stem_features: Number of convolutional features in the stem 
            block.
        :param stem_pool: If True, the output of the stem block is pooled 2X2.
        :param num_block: The number of dense blocks between each pooling 
            block.
        :param block_growth_rate: The growth rate in dense blocks. This is the
            number of additional features that each block introduces.
        :param block_bottleneck_features: The number of bottleneck features in
            dense blocks.
        :param num_pool: Number of pooling blocks. The image size is reduced
            this many times.
        :param pool_bottleneck_compression: Compression factor (geometric
            sequence exponent) for 
        :param upsampling: A string that specifies the upsampling type. 
            Accepted values are 'dcnn' and 'dense'.
        :param upsample_output_expansion: Output expansion factor for
            upsampling layers.
        :param grid_activation: Name of the activation to use in dense blocks.
        :param upsampling_activation: Name of the activation to use in 
            upsampling blocks.
        :type stem_features: int
        :type stem_pool: bool
        :type num_block: int
        :type block_growth_rate: int
        :type block_bottleneck_features: int
        :type num_pool: int
        :type pool_bottleneck_compression: float
        :type upsampling: str
        :type upsample_output_expansion: float
        :type grid_activation: str
        :type upsampling_activation: str
        :returns: A Callable. Takes a Tensorflow tensor and returns another
            tensor that is the input acted upon by the block.
        :rtype: Callable

        """
        
        def dense_grid() -> Callable:
        
            def inner(input_: Tensor) -> Tensor:
                input_channels = input_.shape.as_list()[-1]
                if block_bottleneck_features < 1:
                    bottleneck_features = int(block_bottleneck_features 
                                              * input_channels)
                else:
                    bottleneck_features = block_bottleneck_features
                compression = basic.compression(bottleneck_features)
                branch_convolution = basic.conv2d(
                        block_growth_rate, 
                        (3, 3),
                        activation=grid_activation
                    )
                branch_1 = compression(input_)
                branch_1 = branch_convolution(branch_1)
                return layers.Concatenate()([branch_1, input_])
        
            return inner
        
        def dense_pool() -> Callable:
        
            def inner(input_: Tensor) -> Tensor:
                input_channels = input_.shape.as_list()[-1]
                if pool_bottleneck_compression < 1:
                    bottleneck_features = int(pool_bottleneck_compression 
                                              * input_channels)
                else:
                    bottleneck_features = pool_bottleneck_compression
                compression = basic.compression(bottleneck_features)
                branch_1 = compression(input_)
                return layers.AveragePooling2D(pool_size=(2,2),
                                               padding="same")(branch_1)
        
            return inner
        
        def dense_upsampling() -> Callable:
            
            def inner(input_: Tensor) -> Tensor:
                input_channels = input_.shape.as_list()[-1]
                filters = int(input_channels * upsample_output_expansion)
                output = layers.Conv2DTranspose(
                        filters,
                        kernel_size=(2,2),
                        strides=(2,2),
                        padding="same"
                    )(input_)
                output = layers.Activation(upsampling_activation)(output)
                return layers.BatchNormalization(momentum=0.9)(output)
        
            return inner
        
        def dense_stem() -> Callable:

            def inner(input_: Tensor) -> Tensor:
                output = basic.conv2d(
                        stem_features, 
                        kernel_size=(7, 7), 
                        padding="same",
                        activation=grid_activation
                    )(input_)
                if stem_pool:
                    output = layers.MaxPool2D(pool_size=(2, 2))(output)
                return output
        
            return inner
        
        # main dense_net inner function
        def inner(input_: Tensor) -> Tensor:
            # stem
            output = dense_stem()(input_)
            # downsample
            for _ in range(num_pool):
                # grid
                for _ in range(num_block):
                    output = dense_grid()(output)
                output = dense_pool()(output)
            for _ in range(num_block):
                output = dense_grid()(output)
            # upsample
            if upsampling == "dense":
                for _ in range(num_pool):
                    output = dense_upsampling()(output)
                    for _ in range(num_block):
                        output = dense_grid()(output)
            elif upsampling == "dcnn":
                if stem_pool:
                    num_pool_ = num_pool + 1
                else:
                    num_pool_ = num_pool
                output = ImageData.dcnn_upsampling(
                        num_filters=None,
                        num_pool=num_pool_,
                        activation=grid_activation,
                        batchnorm=True
                    )(output)
            else:
                raise ValueError("Upsampling is optional, however, if a string"
                                 " is specified, it must be either 'dense' or "
                                 "'dcnn'.")
            # output
            return output
    
        return inner
    
    
class ProcessData:
    """ Namespace for bodies that handle process data.
    
    ProcessData bodies assume that process data has a shape of num_batch X 
    num_history X num_features (without the batch axis, this is the 'table of 
    scalars' type)
    
    """
    
    @staticmethod
    def feature_reduction_mlp(
            compression: float = 0.3,
            num_steps: int = 2,
            activation: str = "relu",
            batchnorm: bool = True,
            dropout: Optional[float] = None
        ) -> Callable:
        """ Feature-wise reduction layer. Extracts a representation of features
        with no consideration for temporal behavior. Features are learned
        end-to-end.
        
        This is an MLP acting observation-wise (row-wise in the table).
        
        :param compression: Compression factor (geometric sequence exponent) in
            the MLP layers.
        :type compression: float
        :param num_steps: Number of layers in the MLP.
        :type num_steps: int
        :param activation: Name of the activation in the MLP.
        :type activation: str
        :param batchnorm: If True, BatchNormalization layers are added after
            the activation in each layer in the MLP.
        :type batchnorm: bool
        :param dropout: Optional, the rate of the dropout to add after the
            output of each layer. If not specified, no dropout is used.
        :type dropout: str
        :returns: A Callable. Takes a Tensorflow tensor and returns another
            tensor that is the input acted upon by the block.
        :rtype: Callable
        
        """
        def inner(input_: Tensor) -> Tensor:
            output = input_
            num_input = input_.shape.as_list()[-1]
            for i in range(num_steps):
                num_features = int(num_input * compression ** (i + 1))
                output = layers.Conv1D(num_features, 1, padding="same")(output)
                output = layers.Activation(activation)(output)
                if batchnorm:
                    output = layers.BatchNormalization(momentum=0.9)(output)
            if dropout is not None:
                output = layers.Dropout(dropout)(output)
            return output
        return inner
    
    
    @staticmethod
    def temporal_reduction_conv(
            num_steps: int,
            activation: str = "relu",
            batchnorm: bool = True,
            neighborhood_size: int = 5,
            stride: Optional[int] = None,
            compression: Optional[float] = None
        ) -> Callable:
        """ This block extracts projected samples feature-wise and 
        convolutional features along the temporal dimension. The data are 
        reduced both feature-wise and temporally. Temporal patterns and feature
        interaction both affect the extracted features.
        
        Features are learned end-to-end. Features are extracted by 1D 
        convolutional layers. Convolutions are strided by default to extend the
        temporal receptive window.
        
        :param num_steps: The number of 1D convolutional layers to add.
        :type num_steps: int
        :param activation: Name of activation to use after convolutional 
            layers.
        :type activation: int
        :param batchnorm: If True, BatchNormalization layers are inserted after
            the activations of convolutional layers.
        :type batchnorm: bool
        :param neighborhood_size: The kernel size of the convolutional layers.
            This many timesteps will be seen by one kernel.
        :type neighborhood_size: int
        :param stride: The stride parameter. If not specified, 1D maximum 1X2
            pooling will be used to extend the receptive window.
        :type stride: Optional[int]
        :param compression: The compression factor for subsequent layers. If
            not provided, the same number of features are used in all layers.
        :type compression: Optional[float]
        :returns: A Callable. Takes a Tensorflow tensor and returns another
            tensor that is the input acted upon by the block.
        :rtype: Callable
        
        """
        
        def inner(input_: Tensor) -> Tensor:
            output = input_
            num_features = output.shape.as_list()[-1]
            for i in range(num_steps):
                
                if stride is not None:
                    num_features *= stride
                else:
                    num_features *= 2
                if compression is not None:
                    num_features = int(num_features * compression)
                if stride is not None:
                    output = layers.Conv1D(
                            num_features, 
                            neighborhood_size,
                            strides=stride
                        )(output)
                else:
                    output = layers.MaxPool1D(padding="same")(output)
                    output = layers.Conv1D(
                            num_features,
                            neighborhood_size
                        )(output)
                if batchnorm:
                    output = layers.BatchNormalization(momentum=0.9)(output)
            return output
        
        return inner
    
    
    @staticmethod
    def take_feature(ind_feature, num_steps) -> layers.Layer:
        """ Special block that extracts a single feature (along the last
        dimension) and returns it. This is used in the conv_leg block. You
        will probably not need it.
        
        :param ind_feature: The index of the feature to take.
        :type ind_feature: int
        :param num_steps: The time history lookback of the input layer. The 
            block must be told this in the current implementation.
        :returns: A Keras Lambda layer that does the slicing.
        :rtype: keras.layers.Layer
        
        """
        output = layers.Lambda(
                lambda x: x[:, :, ind_feature:ind_feature+1],
                output_shape=(num_steps, 1)
            )
        return output
    
    
    @staticmethod
    def split_features(input_: Tensor) -> List:
        """ A special block that splits features and returns a list of layers.
        This removes all feature interaction between the data. This is used by
        the conv_leg block, you will probably not use it on its own.
        
        :param input_: The input tensor.
        :type input_: tensorflow.Tensor
        :returns: A List of layers. Each layer is a separate time series of a
            single feature.
        :rtype: List[keras.layers.Layer,...]
            
        """
        num_features = input_.shape.as_list()[-1]
        return [ProcessData.take_feature(k, num_features)(input_) 
                    for k in range(num_features)]
        
    
    @staticmethod
    def pca(
            input_: Tensor,
            num_components: int = 3
        ) -> Tuple[layers.Layer, layers.Layer]:
        """ A linear autoencoder that realizes top-k PCA feature-wise. This is
        not temporal or functional PCA. It extracts a more compact 
        representation of features utilizing feature interaction information.
        
        This block must be trained separately as it is trained to reconstruct
        its input. Use an appropriate training protocol for that.
        
        The block returns a Tuple of layers. The first element is the PCA
        components (the compact representation) and the second element is the
        reconstructed input.
        
        :param input_: The input tensor.
        :type input_: tensorflow.Tensor
        :param num_components: The number of PCA components to learn.
        :type num_components: int
        :returns: A 2-Tuple of layers. The first element is the PCA
            components (the compact representation) and the second element is 
            the reconstructed input.
        :rtype: Tuple[keras.layers.Layer, keras.layers.Layer]
        
        """
        pca_components = layers.Conv1D(
                num_components, 
                1, 
                padding="same",
                use_bias=False,
                name="pca_components"
            )(input_)
        pca_reconstruction = layers.Conv1D(
                input_.shape.as_list()[-1], 
                1, 
                padding="same",
                use_bias=False,
                name="pca_reconstruction"
            )(pca_components)
        return pca_components, pca_reconstruction
    
    
    @staticmethod
    def conv_leg(
            num_steps: int = 3,
            num_features: int = 4,
            compression: float = 2.0,
            stride: int = 3,
            kernel_size: int = 21,
            activation: str = "relu"
        ) -> Callable:
        """ Convolutional feature extraction feature-wise, using only temporal
        reduction. This separates the input to features and applies 
        convolutions on each feature.
        
        Each feature will be processed by a deep 1D convolutional net.
        
        :param num_steps: The number of stacked convolutional layers over each
            feature.
        :type num_steps: int
        :param num_features: The number of filters in each convolutional layer.
        :type num_features: int
        :param compression: Compression factor (sequence exponent) for 
            subsequent convolutional layers.
        :type compression: float
        :param stride: Stride for 1D convolutions.
        :type stride: int
        :param kernel_size: Kernel size (temporal receptive window size).
        :type kernel_size: int
        :param activation: Name of activation on convolutional layers.
        :type activation: str
        :returns: A Callable. Takes a Tensorflow tensor and returns another
            tensor that is the input acted upon by the block.
        :rtype: Callable
        
        """
        
        def inner(input_: Tensor) -> Tensor:
            features = ProcessData.split_features(input_)
            for i in range(num_steps):
                num_features_ = int(num_features * compression**i)
                features = [layers.Conv1D(
                        padding="same",
                        filters=num_features_, 
                        kernel_size=kernel_size, 
                        strides=stride
                    )(feature) for feature in features]
                features = [layers.Activation(activation)(feature)
                    for feature in features]
                features = [layers.BatchNormalization(momentum=0.9)(feature) 
                    for feature in features]
            output = layers.Concatenate()(features)
            return output
        
        return inner
    
    
    @staticmethod
    def randomized_feature_bagger(
            num_batch: int, 
            rate: float = 0.33
        ) -> Callable:
        """ Adds a dropout layer to the inputs. This randomly masks features.
        The entire time series of the masked feature will be masked. This is
        supposed to suppress collinearity effects when assessing sensitivity.
        
        :param num_batch: The number of batches in the input. The block has to 
            be told this.
        :type num_batch: int
        :param rate: The dropout rate. This is the probability to mask a 
            feature in every pass.
        :type rate: float
        :returns: A Callable. Takes a Tensorflow tensor and returns another
            tensor that is the input acted upon by the block.
        :rtype: Callable
        
        """
        def inner(input_: Tensor) -> Tensor:
            noise_shape = input_.shape.as_list()
            # featurewise
            noise_shape[0] = num_batch
            noise_shape[1] = 1
            return layers.Dropout(rate, noise_shape=noise_shape)(input_)
        return inner
    
    
    @staticmethod
    def reducer_regressor(
            reducer_steps: int = 2,
            reducer_compression: float = 0.3,
            reducer_activation: str = "relu",
            temporal_steps: int = 3,
            temporal_stride: int = 5,
            temporal_compression: Optional[float] = 0.1,
            temporal_activation: str = "relu",
            temporal_size: int = 5,
            bagger: Optional[float] = None,
            bagger_num_batch: Optional[int] = None,
            pca_components: Optional[int] = 4
        ) -> Callable:
        """ This block is a combination of feature_reduction_mlp and 
        temporal_reduction_conv with optional random feature bagging and PCA.
        
        :param reducer_steps: Number of layers in the feature-wise MLP.
        :type reducer_steps: int
        :param reducer_compression: Compression factor (geometric sequence 
            exponent) in the feature-wise MLP layers.
        :type reducer_compression: float
        :param reducer_activation: Name of the activation in the MLP.
        :type reducer_activation: str
        :param temporal_steps: The number of 1D convolutional layers to add.
        :type temporal_steps: int
        :param temporal_activation: Name of activation to use after 
            convolutional layers.
        :type temporal_activation: int
        :param temporal_stride: The stride parameter. If not specified, 1D
            maximum 1X2 pooling will be used to extend the receptive window.
        :type temporal_stride: Optional[int]
        :param temporal_compression: The compression factor for subsequent 
            layers. If not provided, the same number of features are used in 
            all layers.
        :type temporal_compression: Optional[float]
        :param temporal_size: The kernel size of the convolutional layers.
            This many timesteps will be seen by one kernel.
        :type temporal_size: int
        :param bagger_num_batch: The number of batches in the input.
        :type bagger_num_batch: int
        :param rate: The dropout rate for bagging. This is the probability to 
            mask a feature in every pass.
        :type rate: float
        :returns: A Callable. Takes a Tensorflow tensor and returns another
            tensor that is the input acted upon by the block.
        :rtype: Callable
        
        """        
        def inner(input_: Tensor) -> Tensor:
            output = input_
            if bagger is not None:
                output = ProcessData.randomized_feature_bagger(
                        bagger_num_batch, 
                        bagger
                    )(output)
            if pca_components is not None:
                output, pca_rec = ProcessData.pca(output, pca_components)
            output = ProcessData.feature_reduction_mlp(
                    compression=reducer_compression,
                    num_steps=reducer_steps,
                    activation=reducer_activation,
                    batchnorm=True,
                    dropout=None
                )(output)
            output = ProcessData.temporal_reduction_conv(
                    num_steps=temporal_steps,
                    activation=temporal_activation,
                    batchnorm=True,
                    neighborhood_size=temporal_size,
                    stride=temporal_stride,
                    compression=temporal_compression
                )(output)
            if pca_components is not None:
                return output, pca_rec
            return output
        
        return inner
    
    
    @staticmethod
    def memoriless_engineeredstats() -> layers.Layer:
        """ Special layer that extracts the mean and std from temporal 
        histories. A mean and an std will be extracted for each feature.
        
        :returns: A Keras Lambda layer.
        :rtype: keras.layers.Layer
        
        """
        return layers.Lambda(
                lambda x: keras.backend.concatenate(
                        [
                            keras.backend.mean(x, axis=1, keepdims=True),
                            keras.backend.std(x, axis=1, keepdims=True)
                        ],
                        axis=1
                    ),
                    name="moments"
                )
    
    @staticmethod
    def memoriless_mlp(
            num_units: int = 100,
            num_steps: int = 3,
            activation: str = "relu",
            compression: Optional[float] = 0.5,
            dropout: Optional[float] = 0.5
        ) -> Callable:
        """ A regular MLP block that processes scalar features without time
        histories.
        
        :param num_units: The number of units in the first Dense layer.
        :type num_units: int
        :param num_steps: The number of layers in the MLP.
        :type num_steps: int
        :param activation: The name of the activation after Dense layers.
        :type activation: str
        :returns: A Callable. Takes a Tensorflow tensor and returns another
            tensor that is the input acted upon by the block.
        :rtype: Callable
        
        """
        def inner(input_: Tensor) -> Tensor:
            output = input_
            for i in range(num_steps):
                if compression is not None:
                    num_units_ = num_units * compression ** i
                else:
                    num_units_ = num_units
                output = layers.Dense(num_units_)(output)
                output = layers.Activation(activation)(output)
                output = layers.BatchNormalization(momentum=0.9)(output)
            return output
        
        return inner   
