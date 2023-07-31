# ------------------------------------------------------------------
# Copyright (c) 2020 PyInstaller Development Team.
#
# This file is distributed under the terms of the GNU General Public
# License (version 2.0 or later).
#
# The full license is available in LICENSE.GPL.txt, distributed with
# this software.
#
# SPDX-License-Identifier: GPL-2.0-or-later
# ------------------------------------------------------------------
"""
Hook for http://pypi.python.org/pypi/h5py/
"""

hiddenimports = ['h5py._proxy', 'h5py.utils', 'h5py.defs', 'h5py.h5ac']
