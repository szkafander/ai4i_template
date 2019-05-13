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
.. module:: pre_post
   :platform: Unix, Windows
   :synopsis: Contains low-level implementations of pre- and postprocessing 
       functionality.

.. moduleauthor:: Paul Lucas <pal.toth@ri.se>

This module contains low-level implementations of pre- and postprocessing 
functionality. The module abstractions.pre_post uses these implementations. 

.. note::
    Functions or the __call__ methods of Callables implemented as pre- or
    postprocessors in this module must be in the form method(data, **kwargs).
    Only keyword arguments can be read from recipes. Even if you want to add an
    argument besides data, make it a keyword argument. 
    

The module provides the following functionality:
    
================================  =============  ==============================
name                              type           summary
================================  =============  ==============================
rescale                           method         Rescales an input array 
                                                 between a lower and upper
                                                 bound.
subtract_mean                     method         Subtracts the mean from the 
                                                 input array (normalizes).
standardize                       method         Divides the input array by its
                                                 standard deviation 
                                                 (standardization).
normalize_columns                 method         Subtract_mean for table of 
                                                 scalars data.
standardize_columns               method         Standardize for table of
                                                 scalars data.
take_channel                      method         Splices an input array along a
                                                 singular axis.
mask_image                        method         Masks an input image.
sum_values                        method         Sums an array over an axis or
                                                 multiple axes.
PersistentPrePostFunction         class          Base class for persistent pre-
                                                 or postprocessing functions.
SpatialTransformAugmentation      class          Random spatial transform
                                                 augmentation for images.
MaskImage                         class          Masks images by a stored mask
                                                 loaded from disk.
SlagSignalExtractor               class          Project-specific signal 
                                                 extractor.
================================  =============  ==============================

"""

import abc
import numpy as np
from scipy import misc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Any, Optional, Tuple, Union
from misc import utils


# A cache for e.g., masks
_cache = {}


def rescale(
        data: np.array, 
        output_min: float = 0, 
        output_max: float = 1
    ) -> np.array:
    """ Rescales the input argument data so that the returned array is bound
    between output_min and output_max.
    
    :param data: A numpy array of numerical values.
    :type data: numpy.array
    :param output_min: The lower bound of the elements in the returned array.
    :type output_min: float
    :param output_max: The upper bound of the elements in the returned array.
    :type output_max: float
    :returns: A rescaled numpy array.
    :rtype: numpy.array
    
    """
    return ((data - data.min()) / (data.max() - data.min()) 
            * (output_max - output_min) + output_min)


def subtract_mean(
        data: np.array, 
        mean: Optional[Union[float, np.array]] = None,
        channelwise: bool = False
    ) -> np.array:
    """ Subtracts the mean (or specified value(s)) from the argument data. Data
    is normally image data.
    
    :param data: A numpy array of numerical values.
    :type data: numpy.array
    :param mean: Optional value(s) to subtract from data. This can be a single 
        scalar float and a numpy array. If no value is specified, subtract_mean
        calculates the mean based on data. If channelwise is True, means are
        calculated for each channel (the last dimension in data). Otherwise
        a single mean is calculated. If channelwise is True, and a single float
        is specified, the behavior is the same as if channelwise was False. If
        channelwise is True and mean is a numpy array, it must be 0D and
        len(mean) == data.shape[-1].
    :type mean: Optional[Union[float, np.array]]
    :param channelwise: If True, a separate mean is subtracted from each
        channel (the last dimension in data).
    :type channelwise: bool
    :returns: A mean-normalized numpy array.
    :rtype: numpy.array
    
    """
    if mean is None:
        if channelwise:
            mean = data.mean(axis=tuple(range(data.ndim - 1)))
        else:
            mean = data.mean()
    if channelwise:
        if not utils._is_iterable(mean):
            mean = np.ones(data.shape[-1]) * mean
        return data - np.tile(mean, (*data.shape[:-1], 1))
    return data - mean


def standardize(
        data: np.array, 
        std: Optional[Union[float, Tuple[float,...]]] = None,
        channelwise: bool = False
    ) -> np.array:
    """ Divides the argument data by the standard deviation (std, or specified 
    value(s)). Data is normally image data.
    
    :param data: A numpy array of numerical values.
    :type data: numpy.array
    :param std: Optional value(s) to divide data by. This can be a single 
        scalar float and a numpy array. If no value is specified, standardize
        calculates the std based on data. If channelwise is True, stds are
        calculated for each channel (the last dimension in data). Otherwise
        a single std is calculated. If channelwise is True, and a single float
        is specified, the behavior is the same as if channelwise was False. If
        channelwise is True and std is a numpy array, it must be 0D and
        len(std) == data.shape[-1].
    :type std: Optional[Union[float, np.array]]
    :param channelwise: If True, a separate std is subtracted from each
        channel (the last dimension in data).
    :type channelwise: bool
    :returns: A standardized (unit variance) numpy array.
    :rtype: numpy.array
    
    """
    if std is None:
        if channelwise:
            std = data.std()
        else:
            std = data.std(axis=tuple(range(data.ndim - 1)))
    if channelwise:
        if not utils._is_iterable(std):
            std = np.ones(data.shape[-1]) * std
        return data / np.tile(std, (*data.shape[:-1], 1))
    return data / std


def normalize_columns(
        data: np.array,
        mean: Optional[Union[float, np.array]] = None,
        skip_column: Optional[int] = 0
    ) -> np.array:
    """ Subtracts the mean (or specified value(s)) from the argument data. Data
    is normally process (table of scalars) data.
    
    :param data: A numpy array of numerical values. A 2D array. Rows are
        observations, columns are variables.
    :type data: numpy.array
    :param mean: Optional value(s) to subtract from the data. If a single float
        provided, it will be subtracted from all columns. If a numpy array is
        provided, len(mean) == data.shape[-1] must be True. Then it mean is
        subtracted columnwise. If no value is provided, the columnwise mean
        will be calculated and subtracted columnwise.
    :type mean: Optional[Union[float, np.array]]
    :param skip_column: Optionally, a single column can be specified from which
        the mean will not be subtracted. This is used if e.g., a temporal
        coordinate is present in data.
    :type channelwise: int
    :returns: A column-normalized numpy array.
    :rtype: numpy.array
    
    """
    # if mean is not given, calculate columnwise mean
    data = data.copy()
    num_rows = len(data)
    if mean is None:
        mean = np.tile(np.mean(data, axis=0), (num_rows, 1))
    else:
        # if mean is scalar, apply to all columns
        if len(mean) == 1:
            mean = np.tile(mean, data.shape)
        # if mean is vector, apply columnwise
        if len(mean) != data.shape[1]:
            raise ValueError("If mean is a vector, its length must match the "
                             "number of columns in data.")
        else:
            mean = np.tile(mean, (num_rows, 1))
    # subtract mean
    if skip_column is not None:
        data[:, [i for i in range(data.shape[1]) if i != skip_column]] -= \
            mean[:, [i for i in range(data.shape[1]) if i != skip_column]]
    else:
        data -= mean
    return data


def standardize_columns(
        data: np.array,
        std: Optional[Union[float, np.array]] = None,
        skip_column: Optional[int] = 0
    ) -> np.array:
    """ Subtracts the mean (or specified value(s)) from the argument data. Data
    is normally process (table of scalars) data.
    
    :param data: A numpy array of numerical values. A 2D array. Rows are
        observations, columns are variables.
    :type data: numpy.array
    :param mean: Optional value(s) to subtract from the data. If a single float
        provided, it will be subtracted from all columns. If a numpy array is
        provided, len(mean) == data.shape[-1] must be True. Then it mean is
        subtracted columnwise. If no value is provided, the columnwise mean
        will be calculated and subtracted columnwise.
    :type mean: Optional[Union[float, np.array]]
    :param skip_column: Optionally, a single column can be specified from which
        the mean will not be subtracted. This is used if e.g., a temporal
        coordinate is present in data.
    :type channelwise: int
    :returns: A column-normalized numpy array.
    :rtype: numpy.array
    
    """
    # if std is not given, calculate columnwise std
    data = data.copy()
    num_rows = len(data)
    if std is None:
        std = np.tile(np.std(data, axis=0), (num_rows, 1))
    else:
        # if std is scalar, apply to all columns
        if len(std) == 1:
            std = np.tile(std, data.shape)
        # if std is vector, apply columnwise
        if len(std) != data.shape[1]:
            raise ValueError("If std is a vector, its length must match the "
                             "number of columns in data.")
        else:
            std = np.tile(std, (num_rows, 1))
    # divide by std
    if skip_column is not None:
        data[:, [i for i in range(data.shape[1]) if i != skip_column]] /= \
            std[:, [i for i in range(data.shape[1]) if i != skip_column]]
    else:
        data /= std
    #replace nans with zeros
    data[np.isnan(data)] = 0
    return data
            

def take_channel(
        data: np.array, 
        channel: Union[str, int] = "blue",
        squeeze: bool = False
    ) -> np.array:
    """ Takes a specific array axis from data and returns the spliced array.
    
    :param channel: A string or integer that specifies the axis to take. If
        a string, values can be 'red', 'green' and 'blue' (this assumes a 
        color image). The channel is the last dimension of the argument.
    :type channel: Union[str, int]
    :param squeeze: If True, the returned array is flattened.
    :type squeeze: bool
    :returns: A numpy array, a slice of the input array data.
    :rtype: numpy.array
    
    """
    if isinstance(channel, str):
        if channel == "red":
            channel = 0
        elif channel == "green":
            channel = 1
        elif channel == "blue":
            channel = 2
        else:
            raise ValueError("If channel is a string, it must be 'red', " 
                             "'green' or 'blue'.")
    if squeeze:
        return data[...,channel]
    return data[...,channel:channel+1]


def mask_image(
        data: np.array, 
        mask: Optional[np.array] = None
    ) -> np.array:
    """ Masks the input array data by a binary mask. Values in the masked image
    are zero wherever the mask is not True.
    
    :param mask: A binary mask. The size (array.shape[:2]) of the mask and the 
        argument must be the same. If mask is None, the argument will be 
        returned unchanged.
    :type mask: Optional[numpy.array]
    :returns: A masked array.
    :rtype: numpy.array
    
    """
    if mask is not None:
        if isinstance(mask, str):
            # read from cache if exists
            # this avoids reloading at every call
            if mask not in _cache:
                _cache[mask] = misc.imread(mask)
            mask = _cache[mask]
        if mask.ndim < 3:
            # assume mask missing channels
            mask = mask[...,np.newaxis]
        if data.ndim < 4:
            # assume image missing batch dim
            data = data[np.newaxis,...]
        if data.dtype != np.dtype("bool"):
            # attempt to multiply if not logical
            return data * mask[np.newaxis,:,:]
        return np.logical_and(data, mask[np.newaxis,:,:,np.newaxis])
    return data


def sum_values(
        data: np.array,
        axis: Union[int, Tuple[int,...]] = -1,
        squeeze: bool = True
    ) -> np.array:
    """ Sums values over an axis or multiple axes of data.
    
    :param data: The input data array to sum.
    :type data: numpy.array
    :param axis: A single integer axis or a Tuple of integers specifying
        multiple axes. The default is -1, the last axis.
    :type axis: Union[int, Tuple[int,...]]
    :param squeeze: If True, the returned array will be flattened.
    :type squeeze: bool
    :returns: A numpy array, the summed input array. The size of the returned
        array is the same as that of data (if squeeze is True), or the same of
        that of data except with the summation axes removed (if squeeze if 
        False).
    :rtype: numpy.array
    
    """
    return np.sum(data, axis=axis, keepdims=not squeeze)


# Persistent functions - these store some internal state, normally for
# performance gains.
class PersistentPrePostFunction(abc.ABC):
    """ Abstract base class for PersistentPrePostFunctions. Child classes must
    implement __init__ (this will init and store the state) and __call__ (the 
    object must be Callable so it can used as a function).
    
    """
    
    @abc.abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        pass
    
    @abc.abstractmethod
    def __call__(self, data: Any) -> Any:
        pass


class SpatialTransformAugmentation(PersistentPrePostFunction):
    """ This is a wrapper around Keras' ImageDataGenerator. Instances of this
    class store spatial transformation augmentation parameters and an
    ImageDataGenerator instance.
    
    **Usage**
    
    >>> sta = SpatialTransformAugmentation()
    >>> transformed_image = sta(image)
    
    .. note::
        This class will be mostly used in Processors, not by itself.
    
    """
    
    def __init__(
            self,
            rotation_range: float = 0,
            width_shift_range: float = 0,
            height_shift_range: float = 0,
            zoom_range: float = 0
        ) -> None:
        """ Constructor.
        
        :param rotation_range: The +/- range of random rotation in angles.
        :type rotation_range: float
        :param width_shift_range: The +/- range of random horizontal shift in 
            pixels.
        :type width_shift_range: float
        :param height_shift_range: The +/- range of random vertical shift in 
            pixels.
        :type height_shift_range: float
        :param zoom_range: The +/- range of random zooming. If this is e.g., 
            0.2, the image will be rescaled to 80...120% of its original size.
            The resulting image will be cropped back to the original size.
        :type zoom_range: float
        
        """
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.zoom_range = zoom_range
        self._idg = ImageDataGenerator(
                rotation_range=self.rotation_range,
                width_shift_range=self.width_shift_range,
                height_shift_range=self.height_shift_range,
                zoom_range=self.zoom_range)
    
    def __call__(self, data: np.array) -> np.array:
        """ Call this object as a regular function.
        
        :param data: A numpy array that follows image format conventions. The
            transformation object will act on this and return a randomly
            transformed version of data.
        :type data: numpy.array
        :returns: A randomly transformed image.
        :rtype: numpy.array
        
        """
        return np.array([self._idg.random_transform(x) for x in data])
    

class MaskImage(PersistentPrePostFunction):
    """ This is a persistent function that stores a binary mask. The mask is
    loaded from disk when MaskImage is initialized.
    
    Values in the masked image are zero wherever the mask is not True.
    
    **Usage**
    
    >>> mask_image = MaskImage(path_to_mask)
    >>> masked_image = mask_image(image)
    
    .. note::
        This class will be mostly used in Processors, not by itself.
    
    """
    
    def __init__(
            self,
            mask: str = "mask.png"
        ) -> None:
        """ Constructor.
        
        :param mask: Path to a binary image on disk. This is loaded and stored
            at __init__. The size (array.shape[:2]) of the loaded mask and the 
            argument must be the same.
        :type mask: string
        
        """
        self.mask = utils.read_image(mask, batch_dim=0)
        if self.mask.ndim < 4:
            # assume mask missing channels
            self.mask = self.mask[...,np.newaxis]
        self.mask /= 255.0
    
    def __call__(self, data: np.array) -> np.array:
        """ Call this object as a regular function.
        
        :param data: A numpy array that follows image format conventions. The
            masked version of this image will be returned.
        :type data: numpy.array
        :returns: A randomly transformed image.
        :rtype: numpy.array
        
        """
        return self._mask_image(data, self.mask)
    
    # This function does the work
    def _mask_image(
            self, 
            data: np.array,
            mask: Optional[np.array] = None
        ) -> np.array:
        if mask is None:
            mask = self.mask
        return data * self.mask
    
    
class SlagSignalExtractor(MaskImage):
    """ This is a child class of MaskImage. It loads a mask, applies it, then
    returns a sum over the spatial dimensions. This class is specific for
    returning segmentation area. The segmentation map is masked first.
    
    **Usage**
    
    >>> sse = SlagSignalExtractor(path_to_mask)
    >>> slag_signal = sse(segmentation)
    
    .. note::
        This class will be mostly used in Processors, not by itself.
        
    .. warning::
        This class is specific to the slag problem and even to the implemented
        segmentation models. As such, the class is also a good example of how
        to implement specific persistent postprocessors.
        
    """
    
    def __init__(
            self,
            mask: str = "mask.png",
            class_threshold: float = 0.1
        ) -> None:
        """ Constructor.
        
        :param mask: Path to a binary image on disk. This is loaded and stored
            at __init__. The size (array.shape[:2]) of the loaded mask and the 
            argument must be the same. The path is relative to the project
            folder (see utils.relative_path).
        :type mask: string
        :param class_threshold: The threshold above which the masked 
            segmentation map will be considered 'object class'. I.e., pixels
            for which the thresholded segmentation map is True, are considered 
            to belong to an object of interest.
        :type class_threshold: float
        
        """
        super(SlagSignalExtractor, self).__init__(utils.relative_path(mask))
        self.class_threshold = class_threshold
    
    def __call__(self, data: np.array) -> np.array:
        """ Call this object as a regular function.
        
        :param data: A numpy array, a segmentation map.
        :type data: numpy.array
        :returns: A numpy array, normally singular elements are scalars - these
            are 'object class' sums.
        :rtype: numpy.array
        
        """
        # assume first channel is slag
        slag = data[...,0:1]
        # assume second channel is hole
        hole = data[...,1:2] < self.class_threshold
        # mask out slag
        slag = self._mask_image(slag)
        # get active area coverage
        slag = self._mask_image(slag, hole.astype(float))
        return sum_values(slag, axis=None) / sum_values(hole, axis=None)
        

# This is for compatibility with recipes. If you implement more functions, add
# their keyword arguments here.
_fns_kwargs = {
        rescale: ("min", "max",),
        take_channel: ("channel",),
        subtract_mean: ("mean", "channelwise",),
        standardize: ("std", "channelwise",),
        normalize_columns: ("mean", "skip_column",),
        standardize_columns: ("std", "skip_column",),
        mask_image: ("mask",),
        sum_values: ("axis",),
        SpatialTransformAugmentation: ("rotation_range", "width_shift_range",
            "height_shift_range", "zoom_range",),
        MaskImage: ("mask",),
        SlagSignalExtractor: ("mask", "class_threshold",)
    }
