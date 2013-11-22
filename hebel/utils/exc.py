# Copyright (C) 2013  Hannes Bretschneider

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

__author__ = "Ian Goodfellow"
"""
Exceptions related to datasets
"""

class EnvironmentVariableError(Exception):
    """ An exception raised when a required environment variable is not defined """

    def __init__(self, *args):
        super(EnvironmentVariableError,self).__init__(*args)

class NoDataPathError(EnvironmentVariableError):
    """
    Exception raised when PYLEARN2_DATA_PATH is required but has not been
    defined.
    """
    def __init__(self):
        super(NoDataPathError, self).__init__(data_path_essay)

data_path_essay = """\
You need to define your PYLEARN2_DATA_PATH environment variable. If you are
using a computer at LISA, this should be set to /data/lisa/data.
"""

class NotInstalledError(Exception):
    """
    Exception raised when a dataset appears not to be installed.
    This is different from an individual file missing within a dataset,
    the file not loading correctly, etc.
    This exception is used to make unit tests skip testing of datasets
    that haven't been installed.
    We do want the unit test to run and crash if the dataset is installed
    incorrectly.
    """
