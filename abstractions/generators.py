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
.. module:: generators
   :platform: Unix, Windows
   :synopsis: Contains low-level implementations of data generators for serving
       data to models.

.. moduleauthor:: Paul Lucas <pal.toth@ri.se>

This module contains low-level implementations of data generators for serving
data to models. The module provides the following functionality:
    
======================  =============  ========================================
name                    type           summary
======================  =============  ========================================
SpatioTemporal          class          A general generator object that can 
                                       serve image and scalar data and 
                                       histories of both.
TableOfScalars          class          A generator to serve tabular scalar 
                                       data.
HistorySampler          class          A helper object that generates history 
                                       indices.
test_sequence_text      method         A helper method that generates dummy 
                                       sequence data (of strings).
test_images_text        method         A helper method that generates dummy
                                       image data (of strings).
test_images_image       method         A helper method that generates dummy
                                       image data.
======================  =============  ========================================

"""

import abc
import functools
import numpy as np
import matplotlib.pyplot as pl
from string import ascii_lowercase
from typing import Any, Callable, List, Optional, Tuple, Union


_VALID_DIRECTIONS = ["forward", "backward"]
_VALID_MODES = ["random", "sequential"]


def _unknown_direction() -> None:
    raise ValueError("Unknown direction. Valid directions are: " 
                     + ", ".join(_VALID_DIRECTIONS))


def _unknown_mode() -> None:
    raise ValueError("Unknown mode. Valid modes are: " 
                     + ", ".join(_VALID_MODES))


def _index_error() -> None:
    raise ValueError("The requested indices lie outside the range of data.")


def test_sequence_text(n_length: int, n_feature: int) -> np.array:
    """ A helper function that can generate sequence test data. Good for
    testing samplers and reducers on.
    
    :param n_length: The number of observations (rows in the output).
    :type n_length: int
    :param n_feature: The number of features (columns in the output).
    :type n_feature: int
    :returns: A numpy array, the sequence data. This is a two-dimensional
        array, where rows are observations and columns are variables. The
        elements of the matrix are strings in the form of <x><n> where x is a
        letter a...z that identifies the variable and n is an integer that
        specifies the observation.
    
    """
    return np.array([[str(c)+str(a) for c in ascii_lowercase[0:n_feature]] 
                      for a in range(n_length)])


def test_images_text(
        n_images: int, 
        width: int, 
        height: int, 
        n_channels: int, 
        labels: bool = False
    ) -> Union[Tuple, np.array]:
    """ A helper function that can generate image-like data. The values in the
    image matrices are strings. This allows for testing samplers and reducers
    by making transformations and indexing by the samplers or reducers
    traceable.
    
    :param n_images: The number of images to generate.
    :type n_images: int
    :param width: The width of each image.
    :type width: int
    :param height: The height of each image.
    :type height: int
    :param n_channels: The number of channels in each image.
    :type n_channels: int
    :param labels: If True, 'labels' are returned for each image. Each label is
        a string that is the same as the string with which the corresponding
        image is filled up.
    :type labels: bool
    :returns: If labels is False, a numpy array of images is returned. The
        array is size n_images X width X height X n_channels. The value of each
        pixel in an image size width X height X n_channels in the returned 
        array is a string, '<n>' where n is an integer, specifying a sequential
        index of the image. The sequential index is 0...n_images. If labels is
        True, a Tuple r is returned, with r[0] being the numpy array of images
        and r[1] being an 1D numpy array of labels, that are the sequential
        indices of the images.
        
    """
    # one-liner numpy array manipulations are ugly, but this is only a helper
    output = np.expand_dims(
            np.swapaxes(np.array([[["0"]*width]*height]*n_channels), 0, -1), 
            0
        )
    for i in range(n_images-1):
        new_image = np.expand_dims(
                np.swapaxes(
                        np.array([[[str(i+1)]*width]*height]*n_channels), 
                        0, -1
                    ), 0
                )
        output = np.concatenate((output, new_image), axis=0)
    if labels:
        return (output, 
                np.array(list(map(lambda x: str(x), list(range(n_images))))))
    else:
        return output


def test_images_image(
        n_images: int, 
        width: int, 
        height: int, 
        fontsize: int = 25
    ) -> np.array:
    """ A helper function that can generate actual image data on which reducer
    and sampler functions can be tested. The images contain rasterized text so
    that they can be identified after sampling and reduction.
    
    :param n_images: The number of images to generate.
    :type n_images: int
    :param width: The width of the images.
    :type width: int
    :param height: The height of the images.
    :type height: int
    :param fontsize: The size of the font with which the rasterized text is
        written.
    :type fontsize: int
    :returns: A numpy array size n_images X width X height. Each image size
        width X height contains rasterized text. The text shows an integer, the
        sequential index of the image 0...n_images.
        
    .. note::
        Calling this method has a side effect of opening matplotlib figures.
        This is due to the fact that the render engine of matplotlib is used
        for rasterization.
        
    
    """
    def _render_image(i):
        fig = pl.figure(
                num=1, 
                figsize=(width/100, height/100), 
                dpi=100, 
                facecolor="k"
            )
        ax = pl.axes(label=str(i))
        ax.text(0, 0, str(i), fontsize=fontsize, color=[1, 0, 0.5])
        ax.axis("off")
        fig.canvas.draw()
        return np.expand_dims(np.array(fig.canvas.renderer._renderer)[:,:,0:3], 
                              axis=0)
    output = _render_image(0)
    for i in range(n_images-1):
        output = np.concatenate((output, _render_image(i+1)), axis=0)
    return output


def test_reducer_function(data: np.array) -> np.array:
    """ A helper function, a default reducer function. This takes arrays of
    strings, i.e., 'a', 'b',... and returns 'a+b', 'b+a',... along the 
    aggregation axis.
    
    :param data: The numpy array to aggregate.
    :type data: numpy array
    :returns: A numpy array, the reduced version of data. test_reducer_function
        is applied over the zeroth dimension of data.
        
    """
    return functools.reduce(np.vectorize(lambda x, y: x+"+"+y), data)


def identity_reducer_function(*args) -> Any:
    """ A helper reducer function. It simply returns its argument(s).
    
    """
    return args


class Sampler(abc.ABC):
    """ Abstract base class for samplers. A sampler must inherit from this.
    
    """
    
    @abc.abstractmethod
    def get_indices(self, start_index: int) -> np.array:
        """ Abstract method for generating indices.
        
        :param start_index: The index at which the history sample starts or
            ends. This is called the 'current index'.
        :type start_index: int
        :returns: A numpy array of integers, that is, the indices. Pulling from
            a data array at these indices must result in an array that is a
            history sample.
            
        """
        pass
    
    @abc.abstractmethod
    def get_valid_index_range(self, n_data: int) -> List[int]:
        pass


class HistorySampler(Sampler):
    """ A helper class that generates history indices.
    
    Instances of this class do not take any data, they simply output the
    integers that index into spatiotemporal or temporal data.
    
    A history sample is a list of lists or 2D array. Each list within the list 
    or row in the array is an 'observation' or 'history'. Depending on the 
    sampler mode, either the first or last element in each history is the 
    'current' element.
    
    **Usage**
    
    >>> sampler = HistorySampler()
    >>> indices = sampler.get_indices(100)
    
    
    ..note::
        Other members of this module use HistorySampler as a helper, but it can
        be used by itself for low-level manipulation.
    
    """
    
    def __init__(
            self, 
            n_lists: int = 10,
            n_step: int = 0,
            n_lookback: int = 10,
            n_skip: int = 0,
            backwards: bool = False
        ) -> None:
        """ Constructor.
        
        :param n_lists: The number of observations to generate. Each 
            observation is a history.
        :type n_lists: int
        :param n_step: Number of indices skipped between observations. Each
            observation starts or ends at start/end of previous 
            observation + (n_step+1).
        :type n_step: int
        :param n_lookback: The length of each observation. If backwards is
            True, then this is the lookback.
        :type n_lookback: int
        :param n_skip: Skip within an observation. The distance of adjacent
            indices in a history is n_skip+1.
        :type n_skip: int
        :param backwards: If True, indices look backwards, not forwards. I.e.,
            within each observation, the 'current' element is the last one, not
            the first one.
        :type backwards: bool
        
        """
        self.backwards = backwards
        self.n_lists = n_lists
        self.n_step = n_step
        self.n_lookback = n_lookback
        self.n_skip = n_skip
        
    @staticmethod
    def _validate_index_range(index_range: List[int]) -> None:
        """ Validates and index range. A valid index range is a list of
        integers. The first element is smaller than the second one and neither
        is below 0.
        
        :param index_range: A list of integers, [index_low, index_high].
        :type index_range: List
        :raises: AttributeError
        
        """
        if index_range[0] >= index_range[1]:
            raise AttributeError("These sampler arguments lead to an empty "
                                 "index range.")
        if any(index < 0 for index in index_range):
            raise AttributeError("These sampler arguments lead to negative "
                                 "indices.")
    
    def get_valid_index_range(self, n_data: int) -> List[int]:
        """ Returns a valid index range given HistorySampler parameters. An
        index range is a list of integers, [index_low, index_high]. Indices
        can be requested from within this range.
        
        :param n_data: The number of datapoints. Generated indices index into
            this data.
        :type n_data: int
        :returns: List -- A List of two integers, [index_low, index_high].
        
        """
        if not self.backwards:
            index_max = (n_data - 1
                         - (self.n_skip+1) * (self.n_lookback-1) 
                         - (self.n_step+1) * (self.n_lists-1))
            index_min = 0
        else:
            index_min = ((self.n_skip+1) * (self.n_lookback-1) 
                            + (self.n_step+1) * (self.n_lists-1))
            index_max = n_data
        self._validate_index_range([index_min, index_max])
        return [index_min, index_max]
        
    def get_indices(self, start_index: int) -> np.array:
        """ The main method of HistorySampler. This returns the actual indices
        starting or ending at start_index, depending on whether HistorySampler
        is forwards or backwards.
        
        :param start_index: The index at which the returned history sample 
            starts or ends, depending on the mode.
        :type start_index: int
        :returns: numpy array -- A history sample, a 2D numpy array. Each row 
            is a history or observation. See the class docstring for more 
            information.
            
        """
        if not self.backwards:
            indices = [[start_index+(self.n_skip+1)*j + (self.n_step+1)*i 
                        for j in range(self.n_lookback)] 
                    for i in range(self.n_lists)]
        else:
            indices = list(reversed([
                         list(reversed([start_index 
                                        - (self.n_skip+1)*j 
                                        - (self.n_step+1)*i 
                                        for j in range(self.n_lookback)])) 
                                      for i in range(self.n_lists)]))
            if np.min(indices) < 0:
                _index_error()
        return np.array(indices)


class SpatioTemporal:
    """ A general generator class that can pull history samples of data arrays.
    It can apply reducer functions on observations, i.e., to aggregate data or
    form temporal averages.
    
    SpatioTemporal takes a length-n iterable of data arrays as data. Instances
    of SpatioTemporal yield history samples pulled from data. The samplers are
    provided as a length-n iterable of samplers (i.e., HistorySampler objects).
    
    SpatioTemporal can return 'current elements' from an optionally passed 
    labels array. There is only one label array, not an iterable of label 
    arrays.
    
    Optionally, a length-n iterable of reducers can be specified. Elements from
    the iterable of reducers apply on each history sample returned by the
    appropriate sampler. Reducers normally act along the temporal dimension,
    i.e., for aggregation (temporal averaging).
    
    Therefore for the kth element in data, it will be sampled by the kth
    element in samplers and aggregated by the kth elementin reducers. If labels
    is provided, labels[start_index] will be yielded. 'start_index' will match
    the value yielded from labels and the values yielded from data[k], such
    that data[k] will be indexed by samplers[k].get_indices[start_index].
    
    :yields: Batches of history data reduced by optional functions.
    
    **Usage**
    
    >>> spatiotemporal = SpatioTemporal(data, labels, reducers, samplers)
    >>> for x_batch, y_batch in spatiotemporal:
            print(x_batch, y_batch)
    
    
    """
    
    def __init__(
            self,
            data,
            samplers: Tuple[Sampler,...],
            n_batch: int = 50,
            random_mode: bool = False,
            sequential_skip: int = 0,
            labels: Optional[np.array] = None,
            reducers: Optional[Tuple[Callable,...]] = None,
            random_seed: Optional[int] = None
        ) -> None:
        """ Constructor.
        
        :param data: An iterable of numpy arrays. Each array must be indexible
            by its corresponding sampler.
        :type data: Iterable
        :param labels: An optional numpy array. The array must be
            indexible by a single integer index. This integer index is used as
            the 'start_index' argument of i.e., HistorySampler.
        :type labels: numpy array
        :param reducers: An iterable of reducer Callables. The Callables act on
            batch.
        :type reducers: Iterable
        :param samplers: An iterable of sampler objects. 
        :type samplers: Iterable
        :param n_batch: The batch size.
        :type n_batch: int
        :param random_mode: If True, start_index are generated randomly within
            the valid index range. OTherwise start_index proceed sequentially.
        :type random_mode: bool
        :param sequential_skip: The skip between two start indices if
            random_mode is False. I.e., if this is 2, start indices are 0, 3...
        :type sequential_skip: int
        :param random_seed: Optional, an integer specifying the random seed for
            generating start_indices if random_mode is True.
        :type random_seed: int
            
        """
        self.data = data
        if not all(entity.shape[0]==data[0].shape[0] for entity in self.data):
            raise ValueError("All data entities must have the same number "
                             "of entries.")
        self.n_data = len(data)
        if labels is not None:
            if self.n_data != labels.shape[0]:
                raise ValueError("Data must have same number of entries "
                                 "as labels.")
        self.labels = labels
        self.n_batch = n_batch
        if reducers is None:
            reducers = (identity_reducer_function,)
        self.reducers = reducers
        self.samplers = samplers
        index_min, index_max = 0, np.Infinity
        for sampler in self.samplers:
            index_min_, index_max_ = sampler.get_valid_index_range(self.n_data)
            index_min = max([index_min, index_min_])
            index_max = min([index_max, index_max_])
        self.index_min = index_min
        self.index_max = index_max
        self.random_mode = random_mode
        self.sequential_skip = sequential_skip
        self.sequential_index = self.index_min - sequential_skip - 1
        self.random_state = np.random.RandomState(random_seed)
    
    @staticmethod
    def _reduce_neighbors(
            data: np.array, 
            indices: List[int], 
            reduce_function: Callable
        ) -> np.array:
        """ Applies an aggregating function over data sliced from arg. data.
        The zeroth axis is assumed to be the batch axis.
        
        :param data: A numpy array. Its zeroth dimension (rows) will be the
            arguments of reduce_function.
        :type data: numpy array
        :param indices: A list of integers, the indices along the zeroth
            dimension of data.
        :type indices: List
        :param reduce_function: The function to apply on the values pulled from
            data.
        :type reduce_function: Callable
        
        """
        return np.array([reduce_function(data.take(inds, axis=0))
                         for inds in indices])
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """ Returns a batch of history samples, optionally aggregated by
        reducers.
        
        """
        
        output = [None] * len(self.data)
        labels = None
            
        for _ in range(self.n_batch):
            
            if self.random_mode:
                start_index = self.random_state.randint(self.index_min, 
                                                        high=self.index_max, 
                                                        size=1)
                start_index = int(start_index)
            else:
                self.sequential_index += (self.sequential_skip + 1)
                if self.sequential_index >= self.index_max:
                    self.sequential_index = 0
                start_index = self.sequential_index
            
            for k in range(len(self.data)):
                indices = self.samplers[k].get_indices(start_index)
                current_data = np.expand_dims(
                        self._reduce_neighbors(
                                self.data[k],
                                indices,
                                self.reducers[k]),
                            axis=0)
                if output[k] is None:
                    output[k] = current_data
                else:
                    output[k] = np.concatenate((output[k], current_data),
                                               axis=0)
            if self.labels is not None:
                if labels is None:
                    labels = self.labels[start_index:start_index+1]
                else:
                    labels = np.concatenate(
                                (labels,
                                 self.labels[start_index:start_index+1]),
                                 axis=0
                            )

        if self.labels is not None:
            output = (output, labels)
        return output


class TableOfScalars:
    """ A generator object for serving process data to Process Models. Process
    data is a pair of arrays, x and y. The array x is called the exogenous
    variables and the array y is called the endogenous variables. Normally, x
    are the predictor variables and y are the predicted variables; however, it
    is not illegal to store a history of the predicted variable in x as well,
    e.g., in the case of a NARX model.
    
    The arrays x and y are n X (1+p) and m X (1+q) arrays where x[:,0] and
    y[:,0] are time coordinates. The time coordinates must be set up so that
    y[:,0] is a subset of x[:,0].
    
    The array x is sampled by matching along the time dimension. First, y is 
    randomly sampled, then matching x are searched for.
    
    :yields: A Tuple (x_batch, y_batch), where x_batch and y_batch are size
        n_batch X n_history X p and n_batch X 1 X q, respectively.
    
    **Usage**
    
    >>> table_of_scalars = TableOfScalars(x, y)
    >>> for x_batch, y_batch in table_of_scalars:
            print(x_batch, y_batch)
        
    .. warning::
        Make sure that a full history can be found in the time coordinate of x
        given a time in the time coordinate of y and a history length. This
        means that in most cases, the time coordinate of x must start at an
        earlier time than that of y.
    
    .. warning::
        This generator assumes that x and y are regular along the time 
        dimension.
    
    
    """
    
    def __init__(
            self,
            x: np.array,
            y: np.array,
            n_batch: int = 50,
            n_history: int = 1000,
            random_mode: bool = False,
            sequential_skip: int = 0,
            sampling_axis: int = 0,
            random_seed: Optional[int] = None
        ) -> None:
        """ Constructor.
        
        :param x: The exogenous data (predictor variable), size n X (1+p). 
            Rows are observations, columns are variables. x[:,0] is the time 
            coordinate.
        :type x: numpy array
        :param y: The endogenous data (predicted variable), size n X (1+p). 
            Rows are observations, columns are variables. x[:,0] is the time 
            coordinate.
        :type y: numpy array
        :param n_batch: The number of batches to yield at calling __next__.
        :type n_batch: int
        :param n_history: The history length in each element in a batch.
        :type n_history: int
        :param random_mode: If True, start times are sampled randomly within
            the valid range.
        :type random_mode: bool
        :param sequential_skip: If random_mode is False, this is the skip
            between sequentially progressing start indices. If sequential_skip
            is 3, the start indices will be 0, 4...
        :type sequential_skip: int
        :param sampling_axis: The column where the time dimension is stored in
            x and y. This overrides the default 0.
        :type sampling_axis: int
        :param random_seed: If random_mode is True, an optional fixed random
            seed can be passed to allow for reproducibility.
        :type random_seed: int
        
        """
        self.x = x
        self.y = y
        self.n_batch = n_batch
        self.n_history = n_history
        self.random_mode = random_mode
        self.sequential_skip = sequential_skip
        self.sampling_axis = sampling_axis
        self.random_seed = random_seed
        self.random_state = np.random.RandomState(random_seed)
        self._t0_end = self.y[:, self.sampling_axis]
        self._t0_ex = self.x[:, self.sampling_axis]
        self._ind_t0_end = np.array(range(len(self._t0_end)))
        self._ind_t0_ex = np.array(range(len(self._t0_ex)))
        self._k = 0
        
        if not self.random_mode:
            print("TableOfScalars generator is operating in sequential mode. "
                  "The value of the num_steps parameter in the "
                  "training/testing protocol should be {}.".format(
                          len(self.y) // self.n_batch))
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """ Returns a batch of history samples.
        
        """
        
        # TODO - check if t0_end is a subset of t0_ex
        # TODO - check if t0_ex starts at least t0_end[0] - n_history earlier
        x_ = np.zeros((self.n_batch, self.n_history, self.x.shape[1]),
                       dtype=self.x.dtype)
        y_ = np.zeros((self.n_batch, self.y.shape[1]), 
                       dtype=self.y.dtype)
        if self.random_mode:
            for i_batch in range(self.n_batch):
                ind_t0_end_sample = int(self.random_state.choice(
                        self._ind_t0_end, size=1))
                y = self.y[ind_t0_end_sample, :]
                t0_end_sample = self._t0_end[ind_t0_end_sample]
                # match t0 in exogenous
                ind_t0_ex_sample = int(self._ind_t0_ex[
                        self._t0_ex==t0_end_sample])
                x = self.x[
                        ind_t0_ex_sample-self.n_history+1:ind_t0_ex_sample+1,
                        :]
                x_[i_batch,...] = x
                y_[i_batch, :] = y
                # the sampling axis is not yielded
            return (np.delete(x_, self.sampling_axis, axis=-1), 
                    np.delete(y_, self.sampling_axis, axis=-1))
        else:
            if self._k + self.n_batch >= len(self.y):
                self._k = 0
            for i_batch in range(self.n_batch):
                ind_t0_end_sample = self._k
                y = self.y[ind_t0_end_sample, :]
                t0_end_sample = self._t0_end[ind_t0_end_sample]
                # match t0 in exogenous
                ind_t0_ex_sample = int(self._ind_t0_ex[
                        self._t0_ex==t0_end_sample])
                x = self.x[
                        ind_t0_ex_sample-self.n_history+1:ind_t0_ex_sample+1, 
                        :]
                x_[i_batch,...] = x
                y_[i_batch, :] = y
                self._k += 1
                # the sampling axis is not yielded
            return (np.delete(x_, self.sampling_axis, axis=-1), 
                    np.delete(y_, self.sampling_axis, axis=-1))
