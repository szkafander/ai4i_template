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


from misc import utils

CSV_FOLDER = utils.relative_path("data/process/KK4/Analog/2018")

IP_21_STAGE_1_DT = 60
IP_21_STAGE_1_NUM_HISTORY = 2 * 24 * 60
# number of features (tags) without time
IP_21_STAGE_1_NUM_FEATURES = 1536
IP_21_STAGE_1_REDUCED_NUM_FEATURES = 23

# full time series shape
IP_21_STAGE_1_SHAPE = (IP_21_STAGE_1_NUM_HISTORY, IP_21_STAGE_1_NUM_FEATURES)

# reduced time series shape
IP_21_STAGE_1_REDUCED_SHAPE = (IP_21_STAGE_1_NUM_HISTORY, 
                               IP_21_STAGE_1_REDUCED_NUM_FEATURES)