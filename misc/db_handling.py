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
.. module:: db_handling
   :platform: Unix, Windows
   :synopsis: Contains low-level methods for handling the process database.

.. moduleauthor:: Paul Lucas <pal.toth@ri.se>

This module contains low-level methods for handling the central process 
database. The database is an SQL database. Each table stores one signal. A
signal is a time series. The time coordinate is normally irregular.

"""

import datetime
import functools
import os
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import sqlite3 as sql
from misc import utils
from typing import Any, Iterable, Optional, Tuple, Union


_REQUIRED_COLUMNS = ("VALUE", "TIMESTAMP", "STATUS")

# Folder and file names
FOLDER_MAIN = utils.relative_path("data/process")
FOLDER_CSVS = utils.relative_path("data/process/KK4/Analog/2018")
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

# These are project-specific
FILENAME_IMPORTANT_AREAS = "important_tags.csv"
FILENAME_IMPORTANT_SIGNALS = "tags_viktigaste.csv"
FILENAME_METADATA = "KK4-AnalogDef-2019-01-29_corrected.csv"
DATABASE_NAME = "merged_ip21_db.db"

important_areas = np.genfromtxt(os.path.join(
        FOLDER_MAIN, FILENAME_IMPORTANT_AREAS))
important_signals = np.genfromtxt(
        os.path.join(FOLDER_MAIN, FILENAME_IMPORTANT_SIGNALS),
        dtype=np.str
    )
metadata = pd.read_csv(os.path.join(FOLDER_MAIN, FILENAME_METADATA),
                   encoding="utf-8")

# Connection to the database
conn = sql.connect(os.path.join(FOLDER_MAIN, DATABASE_NAME))


# methods to get signal props for filtering
def _is_string_integer(string: str) -> bool:
    try:
        int(string)
        return True
    except:
        return False


def is_signal_name_valid(signal_name: str, prefix: str = "KK") -> bool:
    """ Returns True if signal name is valid. A valid signal name must start
    with a prefix and end with a three-digit process area identifier number.
    
    .. warning::
        This method is hard-coded to a certain project.
        
    :param signal_name: The name of the signal that we want to validate.
    :type signal_name: str
    :returns: True if the signal name is valid, False otherwise.
    :rtype: bool
        
    """
    if signal_name[:2] == prefix and _is_string_integer(signal_name[2:5]):
        return True
    return False


def get_process_area(signal_name: str) -> int:
    """ Gets process area from a signal name. The process area code is a three-
    digit number in the signal name.
    
    .. warning::
        This method is hard-coded to a certain project.
        
    :param signal_name: The name of the signal.
    :type signal_name: str
    :returns: The process area code.
    :rtype: int
    
    """
    if is_signal_name_valid(signal_name):
        return int(signal_name[2:5])
    else:
        return 0


def clean_metadata(
        dataframe: pd.DataFrame, 
        name_tag: str = "NAME", 
        good_signal_tag: Any = "Good"
    ) -> pd.DataFrame:
    """ Cleans a DataFrame that holds process metadata. Removes tags with 
    invalid names and extracts the process area code.
    
    .. warning::
        This method is hard-coded to a certain project.
    
    :param dataframe: The DataFrame that holds the metadata.
    :type dataframe: pandas.DataFrame
    :param name_tag: The name of the field in dataframe that holds the signal
        name.
    :type name_tag: str
    :param good_signal_tag: A binary identifier of 'good' signals to keep. Can
        be any type, but a string is normally used.
    :type good_signal_tag: Any
    :returns: A cleaned DataFrame.
    :rtype: pandas.DataFrame
    
    """
    dataframe = dataframe.rename(columns=lambda x: x.strip())
    dataframe[name_tag] = dataframe[name_tag].apply(lambda x: x.strip())
    dataframe = dataframe[dataframe[name_tag].apply(
            lambda x: is_signal_name_valid(x))]
    dataframe = dataframe.assign(
            PROCESS_AREA=[get_process_area(a) for a in dataframe[name_tag]])
    dataframe = dataframe[dataframe.IP_VALUE_QUALITY.apply(
        lambda x: x.strip()) == good_signal_tag]
    return dataframe


def filter_metadata(
        data: pd.DataFrame,
        process_areas: Optional[Iterable] = None, 
        signal_names: Optional[Iterable] = None
    ) -> pd.DataFrame:
    """ Filters a metadata DataFrame based on important process areas and
    important signals. The DataFrame is filtered based on process areas first.
    Filtering means dropping unimportant columns.
    
    :param data: A DataFrame that holds metadata information.
    :type data: pandas.DataFrame
    :param process_areas: An Iterable of process area codes. These should be
        three-digit integers. If not provided, metadata will not be filtered
        based on process area.
    :type process_areas: Optional[Iterable]
    :param signal_names: An Iterable of signal names. These should be strings.
        If not provided, metadata will not be filtered based on signal name. 
    :type signal_names: Optional[Iterable]
    :returns: A filtered DataFrame.
    :rtype: pandas.DataFrame
    
    """
    if process_areas is not None:
        data_important = data[data.PROCESS_AREA.isin(process_areas)]
    if signal_names is not None:
        data_important = data_important[
                data_important.NAME.isin(signal_names)]
    return data_important


def timestring_to_datetime(timestring: str) -> datetime.datetime:
    """ Converts a time string into a datetime object. Time string must follow
    DATETIME_FORMAT.
    
    :param timestring: The time string. Must follow DATETIME_FORMAT.
    :type timestring: str
    :returns: A datetime object.
    :rtype: datetime.datetime
    
    """
    if len(timestring.split(".")[0]) == 3:
        timestring = timestring + "000"
    return datetime.datetime.strptime(timestring, DATETIME_FORMAT)


def clean_signal(signal: pd.DataFrame) -> pd.DataFrame:
    """ Cleans a signal DataFrame. A signal DataFrame is normally returned by
    read_signal. This method checks if all fields are present and does some
    reformatting.
    
    Signals are pandas DataFrames with columns _REQUIRED_COLUMNS (VALUE, 
    TIMESTAMP, STATUS by default).
    
    :param signal: The signal DataFrame. Normally returned by read_signal.
    :type signal: pandas.DataFrame
    :returns: A cleaned DataFrame.
    :rtype: pandas.DataFrame
    
    """
    
    def missing_column(col):
        raise ValueError("Signal is missing column {}".format(col))
        
    # check missing columns
    columns = signal.keys()
    for column in _REQUIRED_COLUMNS:
        if column not in columns:
            missing_column(column)
    # drop bad rows
    inds = ((signal.VALUE!="VALUE") 
        & (signal.STATUS!="STATUS") 
        & (signal.TIMESTAMP!="TIMESTAMP"))
    signal = signal[inds]
    signal.VALUE = signal.VALUE.apply(lambda x: float(x))
    signal.TIMESTAMP = pd.to_datetime(
            signal.TIMESTAMP.apply(lambda x: x+"000")
        )
    signal.STATUS = signal.STATUS.apply(
            lambda x: True if x=="0" else False
        )
    signal = signal.set_index("TIMESTAMP")
    return signal


def read_signal(name: str) -> pd.DataFrame:
    """ Reads a signal from the folder of CSV's.
    
    :param name: Signal name.
    :type name: str
    :returns: A pandas DataFrame that contains the signal data.
    :rtype: pandas.DataFrame
    
    """
    signal = clean_signal(pd.read_csv(os.path.join(FOLDER_CSVS, name+".csv"), 
                                      sep=";"))
    signal = signal.rename(columns={"VALUE": name})
    return signal


def values(signal: pd.DataFrame) -> pd.DataFrame:
    """ Extracts the values from a signal DataFrame.
    
    :param signal: The signal DataFrame.
    :type signal: pandas.DataFrame
    :returns: The signal values.
    :rtype: pandas.Series
    
    """
    return signal.drop(["STATUS"], axis=1).values.flatten()


def visualize_signal(signal: pd.DataFrame) -> None:
    """ Visualizes the signal. Highlights good and bad quality values.
    
    :param signal: The signal DataFrame.
    :type signal: pandas.DataFrame
    
    """
    pl.cla()
    pl.plot(signal.index[signal.STATUS], 
            values(signal)[signal.STATUS], 
            'g', linewidth=0.1)
    pl.plot(signal.index[~signal.STATUS], 
            values(signal)[~signal.STATUS], 
            'r.')
    pl.pause(0.01)
    pl.show()
    
    
def visualize_signals(signals: pd.DataFrame, normalize: bool = False) -> None:
    """ Visualizes multiple signals. Plots different signals as normalized
    time series.
    
    :param signals: A DataFrame that holds multiple signals. Each signal is a
        column.
    :type signals: pandas.DataFrame
    :param normalize: If True, the curves are normalized between 0 and 1.
        Normalized curves are displaced vertically.
    :type normalize: bool
    
    """
    if not normalize:
        signals.plot()
    else:
        row_ind = 0
        for signal in signals.items():
            pl.plot(signal[1].index,
                    row_ind + (signal[1].values - signal[1].values.min()) 
                    / (signal[1].values.max()-signal[1].values.min()))
            row_ind += 1
        pl.legend(signals.keys())


def explore_signals(names: pd.DataFrame, num_batch: int) -> None:
    """ Visualizes num_batch signals in one plot. If a key is pressed, it shows
    the next num_batch signals from names.
    
    :param names: A DataFrame with signal names. Signals are read from the csv
        folder.
    :type names: pandas.DataFrame
    :param num_batch: The number of plots to show on one page.
    :type num_batch: int
            
    """
    no_signal = 0
    key_pressed = False
    num_subplots = np.ceil(np.sqrt(num_batch))
    while not key_pressed:
        for no_plot in range(num_batch):
            name = names.iloc[no_signal]
            pl.subplot(num_subplots, num_subplots, no_plot+1)
            visualize_signal(read_signal(name))
            pl.title(name)
            no_signal += 1
        key_pressed = pl.waitforbuttonpress()


def query_datetime_range(
        tables: Iterable
    ) -> Tuple[datetime.datetime, datetime.datetime]:
    """ Returns the datetime range between which data is available in tables.
    
    :param tables: Iterable, contains table (signal) names.
    :type tables: Iterable
    :returns: A 2-Tuple of datetimes. The first element is the from date, the
        second element is the to date. Within this range, all tables contain
        entries.
    :rtype: Tuple
    
    """
    mins = []
    maxes = []
    for table in tables:
        mins.append(
                conn.cursor().execute(
                        "SELECT min(TIMESTAMP) FROM {}".format(
                                table)).fetchall()[0][0])
        maxes.append(
                conn.cursor().execute(
                        "SELECT max(TIMESTAMP) FROM {}".format(
                                table)).fetchall()[0][0])
    return mins, maxes


def query_datetime_bound(
        table: str, 
        datetime_start: str, 
        datetime_end: str
    ) -> Tuple[str, str]:
    """ Queries the exact datetime bound of a single table (signal) given an
    approximate datetime bound.
    
    :param table: The name of the table (signal).
    :type table: str
    :param datetime_start: The from date of the approximate datetime bound.
        A datestring.
    :type datetime_start: str
    :param datetime_start: The to date of the approximate datetime bound. A
        datestring.
    :type datetime_start: str
    :returns: A 2-Tuple of datetimes. The first element is the from date, the
        second element is the to date. Within this range, all tables contain
        entries.
    :rtype: Tuple
    
    """
    t_from = conn.cursor().execute(("SELECT max(TIMESTAMP) FROM '{}' "
                         "WHERE TIMESTAMP <= '{}'").format(
                                 table, datetime_start)).fetchall()
    t_to = conn.cursor().execute(("SELECT min(TIMESTAMP) FROM '{}' "
                         "WHERE TIMESTAMP >= '{}'").format(
                                 table, datetime_end)).fetchall()
    return (t_from[0], t_to[0])


def sql_between(table, datetime_start, datetime_end):
    """ Generates an SQL query to get entries between two datetimes.
    
    :param table: The name of the table (signal).
    :type table: str
    :param datetime_start: The from date. A datestring.
    :type datetime_start: str
    :param datetime_end: The to data. A datestring.
    :type datetime_end: str
    :returns: An SQL query.
    :rtype: str
    
    """
    dates = query_datetime_bound(table, datetime_start, datetime_end)
    print(dates[0][0], dates[1][0])
    return "SELECT * FROM '{}' WHERE TIMESTAMP BETWEEN '{}' and '{}'".format(
            table, dates[0][0], dates[1][0])


def query_between(
        signals: Union[Tuple[str,...], str],
        datetime_start: str,
        datetime_end: str
    ) -> pd.DataFrame:
    """ Returns a DataFrame that contains values queried between datetime_start
    and datetime_end. Columns will be filled up with queries from tables named
    in signals. The time coordiante will be shared, but can be irregular. Holes
    will be filled with NaN's.
    
    :param signals: A table (signal) name or a Tuple of names.
    :type signals: Union[Tuple[str,...], str]
    :param datetime_start: The from date. A datestring.
    :type datetime_start: str
    :param datetime_end: The to data. A datestring.
    :type datetime_end: str
    :returns: A pandas DataFrame. Columns will be the requested signals.
    :rtype: pandas.DataFrame
    
    """
    def pull(signal):
        data = pd.read_sql(sql_between(signal, datetime_start, datetime_end),
                           conn).drop(columns=["STATUS"])
        data.TIMESTAMP = pd.to_datetime(data.TIMESTAMP)
        return data.set_index("TIMESTAMP")
    
    if not isinstance(signals, tuple):
        signals = (signals,)
    results = {signal: pull(signal) for signal in signals}
    data = functools.reduce(lambda x,y: x.join(y, how="outer"),
                            results.values()).apply(
                                    pd.Series.interpolate, args=("time",))
    data = data[~data.index.duplicated(keep="first")]
    return data[(data.index >= datetime_start) 
                & (data.index <= datetime_end)]


def query_at(
        signals: Union[Tuple[str,...], str],
        datetimes: pd.DatetimeIndex
    ) -> pd.DataFrame:
    """ Query signals at certain datetimes. Gets the nearest entry to each
    datetime.
    
    :param signals: A table (signal) name or a Tuple of names.
    :type signals: Union[Tuple[str,...], str]
    :param datetimes: A pandas DatetimeIndex Series.
    :type datetimes: pandas.DatetimeIndex
    :returns: A DataFrame with the nearest queried entries.
    :rtype: pandas.DataFrame
    
    """
    data = query_between(signals, datetimes.min(), datetimes.max())
    return data.reindex(datetimes, method="nearest")


def list_tables() -> Iterable:
    """ Lists all tables in the database.
    
    :returns: An Iterable with the table names.
    :rtype: Iterable
    
    """
    cursor = conn.cursor().execute(
            "SELECT name FROM sqlite_master WHERE type='table';")
    return cursor.fetchall()


def drop_tables(tables: Iterable) -> None:
    """ Drops (deletes) tables from the database.
    
    :param tables: An Iterable of table names (strings).
    :type tables: Iterable
    
    """
    for table in tables:
        conn.cursor().execute("DROP TABLE {}".format(table))
        print("Table {} dropped.".format(table))

        
metadata = clean_metadata(metadata)
metadata = filter_metadata(metadata, important_areas)
