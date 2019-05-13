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
.. module:: utils
   :platform: Unix, Windows
   :synopsis: Contains miscellaneous utility functions.

.. moduleauthor:: Paul Lucas <pal.toth@ri.se>

A collection of utility methods. These methods might be part of workflows.
All other utilities that are unlikely to be part of a workflow are in the
scripts folder.

.. todo::
    Refactor this to IO, visualization, logging,...


"""

import collections
import copy
import datetime
import glob
import json
import matplotlib.pyplot as pl
import numpy as np
import os
from tensorflow import keras
from scipy.misc import imread, imsave
from typing import Callable, List, Optional, Tuple, Union


# Misc. private functions
def _string_to_val(string):
    try:
        return float(string)
    except:
        pass
    
    try:
        if string == "True":
            return True
        elif string == "False":
            return False
        else:
            raise ValueError("Not a bool.")
    except:
        return string


def _is_iterable(obj):
    if type(obj) == str:
        return False
    try:
        (item for item in obj)
    except TypeError:
        return False
    return True


def _make_iterable(obj):
    if not _is_iterable(obj):
        return (obj,)
    return obj


def _arg_or_default(default_args: dict, args: Optional[dict] = None) -> dict:
    if args is None:
        args = {}
    else:
        args = copy.deepcopy(args)
    default_args = copy.deepcopy(default_args)
    exclusive_keys = set(args) - set(default_args)
    default_args.update(args)
    for key in exclusive_keys:
        del default_args[key]
    return default_args
    

# A shortcut to the project path. This is used for project structuring.
project_dir = os.path.abspath(os.path.join(os.path.split(__file__)[0], ".."))

# The default datetime format.
DEFAULT_DATETIME_FORMAT = "%Y-%m-%d-%H-%M-%S-%f"


def relative_path(path: str) -> str:
    """ Returns an absolute path given the argument path. The returned absolute
    path is project_dir/path. Paths should be OS independent.
    
    :param path: The relative path to make absolute.
    :type path: str
    :returns: An absolute path.
    :rtype: str
    
    """
    return os.path.join(project_dir, path)


def datestr_to_timestamp(datestr: str) -> float:
    """ Returns a UNIX timestamp from a datestring. A timestamp is the number
    of seconds passed between the datetime specified by datestr and Jan 01 
    1970 (UTC). The timestamp is a float so that it can handle fractions of a 
    second.
    
    .. warning::
        The datestring format is hardcoded to be %Y-%m-%d-%H-%M-%S-%<f//1000>
        
    :param datestr: A datestring.
    :type datestr: str
    :returns: A UNIX timestamp.
    :rtype: float
    
    """
    datestr = datestr.split('-')
    datestr[-1] = str(int(datestr[-1]) * 1000).zfill(6)
    datestr = '-'.join(datestr)
    return datetime.datetime.strptime(datestr, 
                                      DEFAULT_DATETIME_FORMAT).timestamp()


def filename_to_timestamp(filename: str) -> float:
    """ Returns a timestamp from a filename. Calls datestr_to_timestamp on the
    filename. Filename can be a full path to a file.
    
    :param filename: A full path to a file.
    :type filename: str
    :returns: A UNIX timestamp.
    :rtype: float
    
    """
    filename = filename.split('/')[-1]
    return datestr_to_timestamp('-'.join(filename.split('-')[:-1]))


def timestamp_to_datetime(timestamp: float) -> datetime.datetime:
    """ Returns a datetime object from a timestamp.
    
    :param timestamp: A UNIX timestamp.
    :type timestamp: float
    :returns: A datetime object.
    :rtype: datetime.datetime.
    
    """
    return datetime.datetime.fromtimestamp(timestamp)


def get_frame_number(filename: str) -> int:
    """ Returns the frame number from a filename.
    
    .. warning::
        Filename follows a hardcoded format.
    
    :param filename: A full filaname.
    :returns: The frame number.
    :rtype: int
    
    """
    filename = filename.split('/')[-1].split('.')[0].split('-')[-1].split('_')[0]
    return int(filename)


def read_image(image_path: str, batch_dim: Optional[int] = None) -> np.array:
    """ Reads an image from disk and returns a numpy array. Change this 
    function to switch to other IO backends. This currently uses scipy.
    
    :param image_path: The full path to the image file.
    :type image_path: str
    :param batch_dim: An optional integer that specifies the batch dimension.
        If provided, the returned image dimensions will be extended so that the
        shape can be made consistent with e.g., the Keras image format. Use a
        value of 0 to do that.
    :type batch_dim: Optional[int]
    :returns: A numpy array, the image data. If the image is integer, it is
        converted to a float image.
    :rtype: numpy.array
    
    """
    image = imread(image_path)
    if batch_dim is not None:
        image = np.expand_dims(image, batch_dim)
    return image.astype(float)


def write_image(
        image: np.array, 
        image_path: str,
        scale: bool = False,
        max_: Optional[float] = None
    ) -> None:
    """ Writes an image to disk.
    
    .. note::
        The image should not be size width X height X channels.
    
    :param image: Image data.
    :type image: numpy.array
    :param image_path: The full path to the image file to be written.
    :type image_path: str
    :param scale: If True, the image will be scaled between 0 and 255 before
        writing to disk.
    :type scale: bool
    :param max_: Optional, if provided, the image will be scaled between 0 and
        max_. The argument scale must be True for this to have effect.
    :type max_: bool
    
    """
    if max_ is not None:
        max_value = max_
    else:
        max_value = image.max()
    if scale:
        image = image / max_value * 255.0
    imsave(image_path, image.astype(int))


def read_labels(
        image_path: Union[str, Tuple[str,...]],
        batch_dim: Optional[int] = None,
        on_value: str = "min",
        allow_multiple_classes: bool = False,
        priority_class: int = 0,
        insert_background_class: bool = True
    ) -> np.array:
    """ Reads an image or set of images and converts them to a segmentation 
    map.
    
    A label image is a one-hot encoded image of size w X h X c where w and h
    are width and height and c is the number of classes. Each channel contains
    a map of values bound between 0 and 1 (usually binary) that specifies a
    segmentation map.
    
    Returns a one-hot encoded label map.
    
    :param image_path: A full path to an image or a Tuple of full paths to
        multiple images. If multiple images are specified, they are read as
        individual channels of the resulting label map.
    :type image_path: Union[str, Tuple[str,...]]
    :param batch_dim: Optional, if provided, the output dimensions are extended
        with a new axis at batch_dim. This is used to keep consistency with
        Keras' image format.
    :type batch_dim: Optional[int]
    :param on_value: This specifies how to interpret values (colors) in
        individual label images. If 'min', the lowest values along the last
        axis are taken as 'on' values. If 'max', the highest values along the
        last axis are taken as 'on' values. In practice, use this to interpret
        your hand-labeled label maps. If you use e.g., white to denote
        background and any other color to denote objects, then set this to
        'min' (since white has the highest value in all channels). If you use
        black to denote background and any other color to denote objects, set
        this to 'max'.
    :type on_value: str
    :param allow_multiple_classes: If True, allows multiple classes at a single
        pixel, i.e., the class labels at that particular pixel are no longer 
        one-hot encoded but can take fractional values, summing to unity. If
        False, occasional multi-class pixels are treated by replacing
        conflicting classes by a priority class.
    :type allow_multiple_classes: bool
    :param priority_class: An integer specifying the priority class. Default is
        0.
    :type priority_class: int
    :param insert_background_class: If True, a new channel will be added that
        is will specify the background, i.e., have 'on' values wherever no
        other class has 'on' values.
    :type insert_background_class: bool
    :returns: A label map, a numpy array. Each channel is a binary map for one
        class.
    :rtype: numpy.array
    
    """
    image_path = _make_iterable(image_path)
    # assume scaling between 0 and 1 and 8-bit images with an occasional alpha
    # channel
    labels = np.array([read_image(path) for path in image_path])
    # remove alpha channel
    if labels.shape[-1] == 4:
        labels = labels[:,:,:,0:3]
        
    # if there is only a single unique value (label image is empty), assume
    # all background
    if len(np.unique(labels)) == 1:
        labels = np.zeros(labels.shape[:-1])
    else:
        # handle label selection from color channels
        if on_value == "min":
            labels = np.min(labels, axis=-1)
            labels = labels == labels.min()
        elif on_value == "max":
            labels = np.max(labels, axis=-1)
            labels = labels == labels.max()
        else:
            raise ValueError("on_value must be either 'min' or 'max'.")
    
    # flip batch dim and channel dim
    labels = np.moveaxis(labels, 0, -1)

    # calculate classwise sum
    sums = np.sum(labels, axis=-1)

    # handle overlapping class maps
    if not allow_multiple_classes:
        mask = sums > 1
        for k in range(labels.shape[-1]):
            class_map = labels[:,:,k]
            if k == priority_class:
                class_map[mask] = 1
            else:
                class_map[mask] = 0
            labels[:,:,k] = class_map
    else:
        labels /= np.dstack((sums,) * labels.shape[-1])
        
    
    # fill in background class if requested
    if insert_background_class:
        mask = sums == 0
        labels = np.dstack((labels, mask))
    
    # expand batch dim
    if batch_dim is not None:
        labels = np.expand_dims(labels, batch_dim)
    
    return labels.astype(float)


def one_hot_encode(y: np.array) -> np.array:
    """ One-hot encodes an 1D array of integer class labels.
    
    :param y: The class labels to encode.
    :type y: numpy.array
    :returns: A one-hot encoded numpy array.
    :rtype: numpy.array
    
    """
    num_classes = len(np.unique(y))
    n = len(y)
    categorical = np.zeros((n, num_classes), dtype=np.float)
    categorical[np.arange(n), y.astype(int)] = 1
    return categorical


def list_folder_contents(
        folder_path: str,
        keyword: str = "", 
        format_string: str = "*", 
        sort_frames: bool = False,
        only_filenames: bool = False,
        sorting_fn: Optional[Callable] = None
    ) -> List:
    """ Lists the contents of a folder in form of a List. Useful for getting
    all image filenames in a folder. Has filtering based on a keyword and
    supports sorting by an arbitrary sorting function.
    
    :param folder_path: The path to the folder to list.
    :type folder_path: str
    :param keyword: A keyword based on which filenames are filtered.
    :type keyword: str
    :param sort_frames: If True, filenames are sorted based on frame numbers.
        See get_frame_number for the implementation.
    :type sort_frames: bool
    :param only_filenames: If True, only the basenames are returned. If False,
        full paths are returned.
    :type only_filenames: bool
    :param sorting_fn: A method that is used to sort the filenames. See sorting
        in Python for more details.
    :type sorting_fn: Callable
    :returns: A List of filenames that meet filtering criteria. Filename order
        is determined by sorting, if requested.
    :rtype: List
    
    """
    if sorting_fn is None:
        sorting_fn = filename_to_timestamp
    files = glob.glob(folder_path + '/*' + keyword + '*.' + format_string)
    if sort_frames:
        files.sort(key=lambda x: (sorting_fn(x), 
                                  get_frame_number(x)))
    else:
        files.sort(key=lambda x: (sorting_fn(x)))
    if only_filenames:
        return [os.path.basename(file) for file in files]
    return files


def check_dataset(
        folder_path: str,
        dataset_type: str = "classification"
    ) -> bool:
    """ Not implemented.
    
    """
    raise NotImplementedError("Method not implemented.")


def load_image_data(
        folder_path: str,
        dataset_type: str = "classification",
        x_format_string: str = "jpg",
        y_format_string: str = "png",
        x_folder: str = "images",
        y_folder: str = "labels",
        allow_multiple_classes: bool = False,
        priority_class: int = 0,
        insert_background_class: bool = True
    ) -> Tuple[np.array, np.array]:
    """ Loads image data from a folder structure.
    
    Loads image data and corresponding target labels from a folder structure.
    The dataset_type argument controls how the target labels are loaded.
    
    .. highlight:: none
    
    The folder structures for different dataset types::
        
        "classification": a classification task, i.e., a single scalar
            class is produced from a single image. In this case, the 
            folder structure is as follows:
                top-level
                    /0_class_0
                    /1_class_1
                    /...
            where subfolders contain images. The returned x contains
            all images. The returned y contains one-hot encoded class
            labels.
        "segmentation": an image segmentation task, i.e., a one-hot
            encoded segmentation map is produced from a single image.
            In this case, the folder structure is as follows:
                top-level
                    /x_folder
                    /y_folder
                        /0_class_0
                        /1_class_1
                        /...
            where x_folder contains images with names X, and all 
            subfolders in y_folder contain segmentation masks with 
            names X. The returned x contains all images from x_folder.
            The returned y contains one-hot encoded segmentation maps
            read from the respective subfolders in y_folder. Each class
            is a channel in every sample in y.
            
    .. highlight:: default
    
    :param folder_path: The path to the top-level of the dataset folder 
        structure.
    :type folder_path: str
    :param dataset_type: Specifies the dataset type.
    :type dataset_type: str
    :param x_format_string: The extension of the images in x, without the dot
        ("jpg", "png", etc.).
    :type x_format_string: str
    :param y_format_string: The extension of the images in y, without the dot.
    :type y_format_string: str
    :param x_folder: The name of the folder where the images are. The exact
        meaning depends on dataset_type.
    :type x_folder: str
    :param y_folder: The name of the folder where the labels are. The exact
        meaning depends on dataset_type.
    :type y_folder: str
    :returns: A Tuple of images and target labels. Both are numpy float arrays.
    :rtype: Tuple[numpy.array, numpy.array]
    
    """
    y = None
    if dataset_type == "classification":
        subfolders = sorted(glob.glob(os.path.join(folder_path, "*"+os.sep)))
        for k, subfolder in enumerate(subfolders):
            files = list_folder_contents(
                    subfolder, format_string=x_format_string)
            if k == 0:
                x = np.array([read_image(file, batch_dim=None) 
                    for file in files])
                y = np.ones(len(files)) * k
            else:
                x = np.concatenate(
                        (x, np.array([read_image(file, batch_dim=None) 
                            for file in files])))
                y = np.concatenate(
                        (y, np.ones(len(files)) * k))
        y = one_hot_encode(y)
    elif dataset_type == "segmentation":
        x_folder = os.path.join(folder_path, x_folder)
        y_folder = os.path.join(folder_path, y_folder)
        filenames = list_folder_contents(
                x_folder, format_string=x_format_string, only_filenames=True)
        class_folders = sorted(glob.glob(os.path.join(y_folder, "*"+os.sep)))
        for k, filename in enumerate(filenames):
            filename_y = os.path.splitext(filename)[0] + "." + y_format_string
            filename_labels = tuple([os.path.join(class_folder, filename_y) 
                for class_folder in class_folders])
            x_ = read_image(os.path.join(x_folder, filename), batch_dim=0)
            y_ = read_labels(
                    filename_labels,
                    batch_dim=0,
                    allow_multiple_classes=allow_multiple_classes,
                    priority_class=priority_class,
                    insert_background_class=insert_background_class
                )
            if k == 0:
                x = x_
                y = y_
            else:
                x = np.concatenate((x, x_), axis=0)
                y = np.concatenate((y, y_), axis=0)
    return x, y


def imoverlay(
        *args, 
        channels: Tuple = (0, 1, 2), 
        show_image: bool = True,
        alpha: float = 0.5
    ) -> np.array:
    """ Forms an aggregate color image by combining three mono-channel images.
    This is useful for plotting e.g., segmentation data over the original 
    image.
    
    This takes at most three positional arguments. These are the at most three
    channels to use as the aggregate image's red, green and blue channels,
    respectively.
    
    :param channels: A 3-Tuple of channels. This specifies the order of
        channels in the aggregate image. This is the order in which the
        separate mono-channel images are inserted. The default is (0, 1, 2),
        which is red, green, blue.
    :type channels: Tuple
    :param show_image: If True, the resulting aggregate image will be plotted.
    :type show_image: bool
    :param alpha: The transparency of the two label channels over the image.
        This is 0 (invisible)...1 (opague).
    :type alpha: float
    
    """
    def cast_image(image, alpha):
        image = image.astype(np.float64) - image.min()
        image = image / image.max() * alpha
        return image
    image_overlaid = np.zeros((args[0].shape[0], args[0].shape[1], 3))
    for k, arg in enumerate(args):
        image_overlaid[:,:,channels[k]] = cast_image(arg, 1 if k==0 else alpha)
    if show_image:
        pl.imshow(image_overlaid)
        pl.xticks([])
        pl.yticks([])
    return image_overlaid


def show_thumbnail(image: np.array, title: Optional[str] = None) -> None:
    """ Shows an image as a thumbnail. This means no X or Y ticks.
    
    :param image: Image data, size width X height X channels.
    :type image: numpy.array
    :param title: The title of the thumbnail, optional.
    :type title: Optional[str]
    
    """
    pl.imshow(image)
    pl.xticks([])
    pl.yticks([])
    if title:
        pl.title(title)


def show_segmentation_training_data(
        x, 
        y, 
        index: int,
        image_channel: int = -1, 
        label_channels: Tuple[int, int] = (0, 1),
        alpha: float = 0.5
    ) -> None:
    """ Plots segmentation results over an image. It displays a color image
    formed as an aggregate of the image_channel of the image x, 
    label_channels[0] channel of the y, and the label_channels[1] channel of y
    as the red, green and blue channels, respectively.
    
    .. note::
        Both x and y should follow Keras' image data format convention. 
        Normally, y should be segmentation data produced from x.
    
    :param x: A color image to overlay on.
    :type x: numpy.array
    :param y: A label image.
    :type y: numpy.array
    :param index: This indexes into the batch dimension of both x and y.
    :type index: int
    :param image_channel: The channel of x to plot.
    :type image_channel: int
    :param label_channels: A 2-Tuple specifying two of y's channels to plot.
        Specify the same number if you want to plot only one class.
    :type image_channel: Tuple[int, int]
    :param alpha: The transparency of the two label channels over the image.
        This is 0 (invisible)...1 (opague).
    :type alpha: float
    
    """
    imoverlay(x[index, :, :, image_channel], 
              y[index, :, :, label_channels[0]],
              y[index, :, :, label_channels[1]],
              alpha=alpha)


def log1l(text: str) -> None:
    """ Logs a line to the console and flushes the line. Useful for logging in 
    loops without cramming the console.
    
    :param text: The text to log.
    :type text: str
    
    """
    print("\r"+text, end="", flush=True)
    
    
def save_history(
        history: keras.callbacks.History, 
        filename: str, 
        tag: str = "_history"
    ) -> None:
    """ Saves a Keras history objects as a CSV file.
    
    :param history: A Keras history object.
    :type history: keras.callbacks.History,
    :param filename: A full path to the filename to be written.
    :type filename: str
    :param tag: A tag to attach to the filename. The default is '_history'.
    :type tag: str
    
    """
    data = np.array(list(history.history.values()))
    np.savetxt(
            os.path.splitext(filename)[0]+tag+".csv", 
            data.transpose(),
            header=','.join(history.history.keys())
        )
    
    
def read_recipe(file_path: str) -> collections.OrderedDict:
    """ Reads a text file and return a dictionary that represents a recipe for
    building a ModelManager or Processor object. The text file must follow the
    JSON format. Four top-level fields can be specified in the file:
    "model", "preprocess", "postprocess" and "misc".
    
    A recipe is a text file. Recipes follow the JSON format. Here is an example
    recipe:
        
    .. code-block:: text
    
        {
          "model":
            {
              "path": "segmenter/dcnn_filt24_kernel9x9_pexp135"
            },
          "preprocess":
            {
              "take_channel": "blue",
              "subtract_mean": 119.01272772,
              "standardize": 21.73939139
            },
          "postprocess":
            {
              "extract_slag_signal": ["masks/segmentation_mask.png", 0.1]
            }
        }
    
    Top-level keywords in recipes specify the type of object to return from
    the description under them. For example, 'model' tells that a model or
    ModelManager instance should be returned. 'preprocess' and 
    'postprocess' tell Processor.from_recipe what functions to use to
    instantiate Processors for pre- or postprocessing. Under a top-level
    keyword, a JSON object lists the function names as keywords and
    function parameters as values. To understand parameters, see the module
    processing.pre_post_fns. To understand function names (keywords in the
    recipe), please see pre_post.recipe_names. To understand top-level
    keywords, please see pre_post.Processor._abbr.
    
    :param file_path: Path to the text file to read from. The path should be 
        relative to the project folder (utils.project_dir).
    :type file_path: str
    :returns: An ordered dict representation of the serialized JSON. The output
        is ordered to preserve Processor function chain order.
    :rtype: collections.OrderedDict

    """
    file_path = relative_path(file_path)
    with open(file_path) as file:
        return json.load(file, object_pairs_hook=collections.OrderedDict)


# Legacy functions
def standardize(
        x, 
        means=None, 
        stds=None
    ):
    """ Legacy standardize function. Use functions from the pre_post module
    instead.
    
    """
    x = x.astype(np.float64)
    if means is None or stds is None:
        for channel_id in range(x.shape[-1]):
            x[:,:,:,channel_id] -= np.mean(np.squeeze(x[:,:,:,channel_id]))
            x[:,:,:,channel_id] /= np.std(np.squeeze(x[:,:,:,channel_id]))
    else:
        for channel_id in range(x.shape[-1]):
            x[:,:,:,channel_id] -= means[channel_id]
            x[:,:,:,channel_id] /= stds[channel_id]
    return x


def destandardize(x, means, stds):
    """ Legacy destandardize function. Use functions from the pre_post module
    instead.
    
    """
    for channel_id in range(x.shape[-1]):
        x[:,:,:,channel_id] *= stds[channel_id]
        x[:,:,:,channel_id] += means[channel_id]
    return x
