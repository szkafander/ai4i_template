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
.. module:: signal_proc
   :platform: Unix, Windows
   :synopsis: Contains low-level methods for processing and visualizing scalar
       signals (time histories).

.. moduleauthor:: Paul Lucas <pal.toth@ri.se>

Contains low-level methods for processing and visualizing scalar signals.
Scalar signals are time series. The time series can be regular or scattered.
This module mostly contains helper functions. The module provides the following
functionality:

================================  =============  ==============================
name                              type           summary
================================  =============  ==============================
scattered_moving_average          method         Box-smooths a scattered time 
                                                 series.
find_contiguous_time              method         Finds indices of edges of 
                                                 subsequent scalar values in a.
                                                 time series that form a
                                                 contiguous sequence.
scattered_average                 method         Box-smooths scattered time  
                                                 series that have holes.    
scattered_average_rate            method         Box smoothed rate of scattered 
                                                 time series. 
plot_data                         method         Plots times scattered time 
                                                 series with holes.
linked_rate_plot                  method         Plots two linked axes. One can
                                                 show a scalar series, the 
                                                 other a rate.
================================  =============  ==============================

.. todo::
    Refactor plotting functions to utils.visualization.

"""

import numpy as np
import matplotlib.pyplot as pl
import datetime
from matplotlib import patches
from matplotlib import axes
from misc.constants import timing
from misc import utils
from typing import Optional, Tuple


def scattered_moving_average(data_: np.array, window_size: int) -> np.array:
    """ Box filtering (smoothing) of scattered temporal data.
    
    :param data_: A numpy array. The first column is time. The second column is
        the time history of scalars.
    :type data_: numpy.array
    :param window_size: The box size.
    :type window_size: int
    :returns: A numpy array, the filtered scalar time series.
    :rtype: numpy.array
    
    """
    def window(x, x_current):
        return (np.abs(x_current - x) < window_size).astype(np.float)
    result = np.zeros(data_[:,1].shape)
    for i, x in enumerate(data_[:,0]):
        w = window(data_[:,0], x)
        result[i] = np.sum(data_[:,1] * w) / np.sum(w)
    return result


def find_contiguous_time(
        time: np.array, 
        d_time_threshold: int
    ) -> Tuple[np.array, np.array]:
    """ Finds 'islands' of contiguous time in an array of time values. Islands
    are subsequent values that are separated by time deltas less than 
    d_time_threshold.
    
    :param time: A numpy array of time values. It should be monotonically
        increasing.
    :type time: numpy.array
    :param d_time_threshold: The threshold time separation between subsequent
        values to be considered still acceptable.
    :type d_time_threshold: int
    :returns: A Tuple of two numpy arrays. The first array holds indices that
        define the edges of islands. The second array is differential time, 
        i.e., the time separation between subsequent values.
    :rtype: Tuple[numpy.array, numpy.array]
    
    """
    d_time = np.diff(time)
    d_time_edges = d_time > d_time_threshold
    d_time_edge_inds = np.array([0, 
                                 *(np.nonzero(d_time_edges)[0].tolist()), 
                                 d_time.size])
    return d_time_edge_inds, d_time


def scattered_average(
        data: np.array, 
        d_time_threshold, 
        window_size
    ) -> Tuple[np.array, np.array]:
    """ Box filtering of scattered data with long temporal holes in it.
    Running averages are calculated only within the 'islands' of contiguous
    data.
    
    :param data: A numpy array. The first column is time. The second column is
        the time history of scalars.
    :type data: numpy.array
    :param window_size: The box size.
    :type window_size: int
    :param d_time_threshold: The threshold time separation between subsequent
        values to be considered still acceptable.
    :type d_time_threshold: int
    :returns: A Tuple of two numpy arrays. The first is the corresponding time 
        coordinate.The second is the filtered series of scalar values. 
    :rtype: Tuple[numpy.array, numpy.array]
    
    """    
    time = data[:, 0]
    values = data[:, 1]
    d_time_edge_inds, _ = find_contiguous_time(time, d_time_threshold)
    times_smoothed = np.array([])
    d_smoothed = np.array([])
    for i, ind in enumerate(d_time_edge_inds[:-1]):
        values_current = values[1:][ind:d_time_edge_inds[i+1]]
        times_current = time[1:][ind:d_time_edge_inds[i+1]]
        times_smoothed = np.append(times_smoothed, times_current)
        d_smoothed = np.append(d_smoothed, 
            scattered_moving_average(
                np.stack((times_current, values_current)).transpose(), 
                window_size))
    return times_smoothed, d_smoothed


def scattered_average_rate(
        data: np.array, 
        d_time_threshold: int, 
        window_size: int
    ) -> Tuple[np.array, np.array, np.array]:
    """ The same as scattered average, except that the scattered average of the
    first time differential is returned (the time rate of change).
    
    :param data: A numpy array. The first column is time. The second column is
        the time history of scalars.
    :type data: numpy.array
    :param window_size: The box size.
    :type window_size: int
    :param d_time_threshold: The threshold time separation between subsequent
        values to be considered still acceptable.
    :type d_time_threshold: int
    :returns: A Tuple of three numpy arrays. The first is the corresponding
        time coordinate. The second is the filtered series of scalar values.
        The third is holds the indices of edges of the contiguous time islands.
    :rtype: Tuple[numpy.array, numpy.array, numpy.array]
    
    """
    time = data[:, 0]
    values = np.diff(data[:, 1])
    
    d_time_edge_inds, d_time = find_contiguous_time(time, d_time_threshold)
    
    times_smoothed = np.array([])
    d_smoothed = np.array([])
    for i, ind in enumerate(d_time_edge_inds[:-1]):
        d_time_current = d_time[ind:d_time_edge_inds[i+1]]
        values_current = values[ind:d_time_edge_inds[i+1]] / d_time_current
        times_current = time[1:][ind:d_time_edge_inds[i+1]]
        times_smoothed = np.append(times_smoothed, times_current)
        d_smoothed = np.append(d_smoothed, 
            scattered_moving_average(
                np.stack((times_current, values_current)).transpose(), 
                window_size))
    
    return times_smoothed, d_smoothed, d_time_edge_inds


def plot_data(
        data: np.array, 
        d_time_threshold: Optional[int] = None, 
        title: str = None, 
        lw: float = 0.25, 
        errors: Optional[np.array] = None, 
        create_fig: bool = True, 
        ylabel: Optional[str] = None, 
        xticklabels: bool = True
    ) -> axes._subplots.Axes:
    """ Plots a time series of one scalar. Holes in the data are masked by grey
    rectangles. Adds an optional title and error bars. Returns the matplotlib
    Axes for further customization.
    
    :param data: A numpy array. The first column is time. The second column is
        the time history of scalars.
    :type data: numpy.array
    :param d_time_threshold: The threshold time separation between subsequent
        values to be considered still acceptable.
    :type d_time_threshold: Optional[int]
    :param title: An optional title to display.
    :type title: Optional[str]
    :param lw: Linewidth.
    :type lw: float
    :param errors: An optional numpy array of errors. If this is provided, it
        has to be the same shape as data.
    :type errors: numpy.array
    :param create_fig: If True, a new figure will be created.
    :type create_fig: bool
    :param ylabel: The Y axis label.
    :type ylabel: str
    :param xticklabels: If True, X tick labels are displayed.
    :type xticklabels: bool
    :returns: A matplotlib Axes object. This can be used for further 
        customization.
    :rtype: matplotlib.axes._subplots.Axes
    
    """
    if create_fig:
        pl.figure(figsize=(12, 4))
    if d_time_threshold is None:
        d_time_threshold = timing.HOLE_THRESHOLD
    time = data[:,0]
    dtime = [utils.timestamp_to_datetime(t) for t in time]
    d_time_edges, _ = find_contiguous_time(time, d_time_threshold)
    if errors is not None:
        max_ = (data[:, 1]+errors).max()
        min_ = (data[:, 1]-errors).min()
    else:
        max_ = data[:, 1].max()
        min_ = data[:, 1].min()
    for i, ind in enumerate(d_time_edges[:-1]):
        from_ = ind
        to_ = d_time_edges[i+1]
        pl.plot(dtime[from_+1:to_], data[from_+1:to_, 1], color="b",
                linewidth=lw)
        if errors is not None:
            pl.fill_between(
                    dtime[from_+1:to_],
                    data[from_+1:to_, 1] - errors[from_+1:to_],
                    data[from_+1:to_, 1] + errors[from_+1:to_],
                    color=(1,0.75,0.75)
                )
        if i < len(d_time_edges)-2:
            from_ = d_time_edges[i+1]
            hole_box = patches.Rectangle(
                    (dtime[from_], min_),
                    dtime[from_+1]-dtime[from_],
                    max_-min_,
                    facecolor=(0.9,0.9,0.9),
                    edgecolor="none"
                )
            pl.gca().add_patch(hole_box)
    pl.xlabel("time")
    pl.ylabel("signal")
    pl.xlim([min(dtime), max(dtime)])
    if title is not None:
        pl.title(title)
    if ylabel is not None:
        pl.ylabel(ylabel)
    if not xticklabels:
        locs, _ = pl.xticks()
        pl.xticks(locs, [])
    return pl.gca()


def linked_rate_plot(
        data: np.array, 
        rate_data: np.array, 
        d_time_edge_inds: np.array
    ) -> None:
    """ Displays a pair of linked plots. The two plots are vertically laid out.
    The top plot shows a time series of a single scalar. The bottom plot shows
    a corresponding, or related, rate. The two axes are linked, i.e., zooming
    and panning affect both axes. This is useful for exploring relationships
    between the scalar signal and the rate.
    
    :param data: A numpy array. The first column is time. The second column is
        the time history of scalars.
    :type data: numpy.array
    :param rate_data: A numpy array. The first column is time. The second 
        column is the time history of rates.
    :type rate_data: numpy.array
    :param d_time_edge_inds: The indices of edges of the contiguous time 
        islands. Use find_contiguous_time to get this.
    :type d_time_edge_inds: numpy.array
        
    """
    dts = [datetime.datetime.utcfromtimestamp(a) for a in data[:,0]]
    ax1 = pl.subplot(2, 1, 1)
    ax1.plot(dts, data[:,1], '.', markersize=0.25, color='b')
    ax1.set_ylabel('slag signal, % of slas area')
    ax1.yaxis.label.set_color('b')
    
    pl.subplot(2, 1, 2, sharex=ax1)
    dts_smoothed = [datetime.datetime.utcfromtimestamp(a) 
        for a in rate_data[:,0]]
    for i, ind in enumerate(d_time_edge_inds[:-1]):
        from_ = ind
        to_ = d_time_edge_inds[i+1]
        pl.fill_between(dts_smoothed[from_:to_], rate_data[from_:to_,1] * 100, 
                        where=rate_data[from_:to_,1]<0,
                        color='g', linewidth=0.1)
        pl.fill_between(dts_smoothed[from_:to_], rate_data[from_:to_,1] * 100, 
                        where=rate_data[from_:to_,1]>0, 
                        color='r', linewidth=0.1)
    pl.ylabel('slag formation rate, %/s')


# Legacy code
def _plot_rate(data, hole_threshold=None, title=None):
    pl.figure(figsize=(12, 4))
    if hole_threshold is None:
        hole_threshold = timing.HOLE_THRESHOLD
    time = data[:,0]
    dtime = [utils.timestamp_to_datetime(t) for t in time]
    d_time_edges, _ = find_contiguous_time(time, hole_threshold)
    max_ = data[:, 1].max()
    min_ = data[:, 1].min()
    for i, ind in enumerate(d_time_edges[:-1]):
        from_ = ind
        to_ = d_time_edges[i+1]
        pl.fill_between(
                dtime[from_:to_], data[:,1][from_:to_], 
                where=data[:,1][from_:to_]<0,
                color='g', linewidth=0.1
            )
        pl.fill_between(
                dtime[from_:to_], data[:,1][from_:to_], 
                where=data[:,1][from_:to_]>0, 
                color='r', linewidth=0.1
            )
        if i < len(d_time_edges)-2:
            from_ = d_time_edges[i+1]
            hole_box = patches.Rectangle(
                    (dtime[from_], min_),
                    dtime[from_+1]-dtime[from_],
                    max_-min_,
                    facecolor=(0.9,0.9,0.9),
                    edgecolor="none"
                )
            pl.gca().add_patch(hole_box)
    pl.xlabel("time")
    pl.ylabel("rate")
    pl.xlim([min(dtime), max(dtime)])
    if title is not None:
        pl.title(title)
