#!/usr/bin/python
"""
This script contains tools for estimating local tie vectors with 
single-difference GNSS or VLBI data for observations of GNSS satellites or 
natural radio sources

==============================================================================
  This file is part of the PyLocalTie software package.  It has been prepared 
  under the NASA Open-Source Science initiative.

  This is free software; you can redistribute it and/or modify
  it under the terms of the BSD 3-Clause License. See the LICENSE file
  distributed with this software package for the full license text.

  We are distributing this in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  BSD 3-Clause License for more details.

  This software was developed by Applied Research Laboratories at the
  University of Texas at Austin, under NASA Grants 80NSSC24K0828 and 
  80NSSC20K1732.

  Copyright 2025, The Board of Regents of The University of Texas System
==============================================================================
"""

import LAMBDA
import json
import xarray as xr
import datetime
import itertools
import random
import re
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import least_squares, nnls
from scipy.interpolate import make_interp_spline, interp1d, make_smoothing_spline
from scipy.signal import savgol_filter, medfilt
from scipy.stats import linregress
from scipy.stats import median_abs_deviation as mad     
from scipy.spatial.transform import Rotation as rot
from scipy.spatial.transform import Slerp
from scipy.linalg import pinv
import scipy.constants as const
from copy import deepcopy
import argparse
import matplotlib.dates as mdates
from hashlib import sha256
from scipy.stats import norm
from pandas import Timestamp, Timedelta, to_datetime, to_timedelta, DataFrame, read_csv
from collections.abc import Iterable
from gnsstk import IonosphereFreeRange, RinexSatID, WGS84Ellipsoid, Position, AntexData, \
                  computeSolidEarthTides, MJD, TimeSystem, asTimeSystem,\
                  AntexStream, EphTime, CommonTime, TimeSystem, northEastUpGeodetic,\
                  CivilTime, getTimeSystemCorrection, SatelliteSystem, NavSearchOrder, NavSatelliteID, Triple,\
                  Xvt, NavLibrary, NavDataFactory, MultiFormatNavDataFactory, SVHealth,\
                  NavMessageID, NavMessageType, NavValidityType, SatID
# from gnsstk import PreciseRange, PhaseWindup
import re
from typing import Dict, Iterable, Tuple, Optional
from pathlib import Path

# MJD at the Unix epoch (1970-01-01T00:00:00Z)
MJD_UNIX = 40587.0
MJD_GPS = 44244.0
SEC_PER_DAY = 86400.0
TAI2GPS = -19
GPS2UTC = -18
MIN_SLICE = 1
MU_EARTH = 3.9860044188e14
MU_SUN=1.327124400189e20
L_G = 6.969290134e-10
SECONDS_PER_DAY=86400
FACTOR_RW=1e24/const.c**2*3600 # ps^2/hr --> m^2/s
FACTOR_IRW=1e24/const.c**2*3600**3 # ps^2/hr^3 --> m^2/s
ALPHA_IONO=1.345e9
#WEIGHT_RANGE = 1 
#WEIGHT_PHASE = 10
ELEV_WEIGHT=False
MJD_EPOCH_IN_DT = datetime.datetime(1858, 11, 17)
USE_SUN=False
BLOCK_MAP = { # map for satellite blocks
    "BLOCK IIIA": 0,
    "BLOCK IIF": 1,
    "BLOCK IIR": 2,
    "BLOCK IIA": 3,
    "BLOCK II": 4,
    "BLOCK I": 5,
    "GLONASS-M": 6,
    "GLONASS-K1": 7,
    "GLONASS" : 8,
    "BEIDOU-2G": 9,
    "BEIDOU-2I": 10,
    "BEIDOU-2M": 11,
    "GALILEO-1": 12,
    "GALILEO-2": 13,
    "GALILEO-0A": 14,
    "GALILEO-0B": 15
}
mapping_dict = {
    'C1': ['C1D', 'C1B', 'C1C', 'C1P'],
    'L1': ['L1D', 'L1B', 'L1C', 'L1P'],
    'P2': ['C2S'],
    'L2': ['L2S']
}
planet_mu = {
    1: 2.20329e13, # mercury
    2: 3.248599e14, # venus
    3: 3.9860044188e14, # earth
    4: 4.2828372e13, # mars
    5: 1.266865349e17, # jupiter
    6: 3.79311879e16, # saturn
    7: 5.7939399e15, # uranus
    8: 6.8365299e15, # neptune
}

import matplotlib.font_manager as fm
available_fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')
for font in available_fonts: 
    try:
        fm.fontManager.addfont(font)
    except (NotImplementedError, RuntimeError, OSError):
        pass

# set publication-quality plotting defaults
from cycler import cycler
#custom_colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666'] # Color Brewer Dark2
custom_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']
plt.rcParams.update({
    'font.size': 12,                      # generic font size 
    'legend.fontsize': 11,                # legend font size
    'axes.titlesize': 12,                 # Title font size
    'axes.labelsize': 12,                 # Axis label font size
    'xtick.labelsize': 12,                # X-axis tick label size
    'ytick.labelsize': 12,                # Y-axis tick label size
    'font.family': 'Nimbus Roman',        # Use serif fonts (e.g., for LaTeX-like text)
    'figure.figsize': [6.4, 4.8],         # Figure size in inches (width, height)
    'figure.dpi': 300,                    # High resolution for publication
    'axes.linewidth': 1,                  # Axis line width
    'axes.grid': True,                    # Show grid by default
    'grid.alpha': 0.7,                    # Grid transparency
    'grid.linewidth': 0.6,                # Grid line width
    'xtick.major.size': 4,                # Major tick length (X-axis)
    'ytick.major.size': 4,                # Major tick length (Y-axis)
    'xtick.minor.size': 2,                # Minor tick length (X-axis)
    'ytick.minor.size': 2,                # Minor tick length (Y-axis)
    'xtick.direction': 'in',              # Major tick direction (inward)
    'ytick.direction': 'in',              # Major tick direction (inward)
    'lines.linewidth': 1.5,               # Default line width for plots
    'lines.markersize': 4,                # Marker size in plots
    'legend.loc': 'best',                 # Automatically place the legend
    'savefig.dpi': 300,                   # DPI for saving figures
    'mathtext.fontset': 'cm',             # set math text to computer modern
    'savefig.format': 'png',              # Save figures as PNG (can change to 'pdf' if needed)
    'savefig.bbox': 'tight',              # Ensure tight bounding box when saving figures
    'savefig.pad_inches': 0.05,           # Padding around the figure when saving
    'axes.unicode_minus': False,          # Use proper minus signs in axes labels
    'axes.prop_cycle': cycler(color=custom_colors), # use non-default color scheme for uniqueness 
})

def _convert_system(system='Any'):
    """
    Args:
        system (str or int): gnsstk GNSS time system indicator (ex: 'GPS', gnsstk.TimeSystem.GPS, 2)

    Returns:
        gnsstk.TimeSystem: gnsstk time system object
    """
    if isinstance(system, str):
        tsys = asTimeSystem(system)
    else:
        tsys = TimeSystem(system)
    return tsys

def standardize_dates(date_time):
    """
    Convert numpy.datetime64 object to pd.Timestamp to for date_to_something functions to allow them to be called
        with numpy.datetime64 datetimes

    Args:
        date_time (Union(date-like, Iterable(date-like))): Python datetime object(s)

    Returns:
        array_like(datetime.datetime) if of type datetime.datetime else is of type pd.Timestamp
    """
    # check if is a single date_time
    if isinstance(date_time, np.datetime64):
        date_time = Timestamp(date_time)
    else:
        # check if is an array of date_times
        if isinstance(date_time, np.ndarray):
            if isinstance(date_time[0], np.datetime64):
                # get the length of the numpy array
                date_time = np.array(list(map(lambda x: Timestamp(x), date_time)))

    return date_time

def date_to_common(date_time, system='Any'):
    """
    Convert datetime.datetime object to gnsstk.CommonTime

    Args:
        date_time (Union(date-like, Iterable(date-like))) or np.datetime64 (Union(date-like, Iterable(date-like))): Python datetime object(s)
        system (str or int): gnsstk GNSS time system indicator (ex: 'GPS', gnsstk.TimeSystem.GPS, 2)

    Returns:
        array_like(gnsstk.CommonTime): CommonTime object(s) with gnsstk.TimeSystem
    """
    tsys = _convert_system(system)
    date_time = standardize_dates(date_time)
    def _map_date_to_common(date_time):
        civil_time = CivilTime(date_time.year, date_time.month, date_time.day,
                                     date_time.hour, date_time.minute, date_time.second + date_time.microsecond * 1e-6,
                                     tsys)
        return civil_time.toCommonTime()
    return _duck_map(date_time, _map_date_to_common)

def date_to_mjd(date_time):
    """
    Convert Python datetime.datetime object(s) to MJD time(s)

    Args:
        date_time (Union(date-like, Iterable(date-like))) or np.datetime64 (Union(date-like, Iterable(date-like))): Python datetime object(s)

    Returns:
        array_like(float): Modified Julian Day, as float days counted from MJD Epoch
    """
    date_time = standardize_dates(date_time)
    def _map_date_to_mjd(date_time):
        date_diff = date_time - MJD_EPOCH_IN_DT
        return date_diff.days + (date_diff.seconds + date_diff.microseconds/1e6)/SECONDS_PER_DAY
    return _duck_map(date_time, _map_date_to_mjd)

def _duck_map(input_data, map_func):
    """
    A mapping function that is "duck-type friendly".
        Given an input and a mapping function, map the elements of the input to an output
        This works for input types: int, float, list, tuple, np.ndarray
        In the case of iterables, the output will be a generator

    Args:
        input_data (array_like): either a container or a singular element of *MOST* of the time formats
        map_func (function): Function that converts the input time from one format to another

    Returns:
        array_like: output_data, either a singular element or a container of time(s) in the desired format

    Note:
        Note that the np.vectorize() is claimed to create a func that actually uses broadcasting
    """
    if isinstance(input_data, np.ndarray):
        if len(input_data) == 0:
            output_data = input_data
        else:
            vector_map_func = np.vectorize(map_func)
            output_data = vector_map_func(input_data)
    elif isinstance(input_data, Iterable):
        output_data = _duck_zip(*list(map(map_func, input_data)))
    else:
        output_data = map_func(input_data)
    return output_data

def _duck_zip(*args):
    """
    For conversions such as date_to_dow, the output element is actually a tuple of time elements (e.g., GPS week,
        day-of-week, and second-of-day). Some users have stated that the preferred output structure is a tuple of lists,
        rather than a list of tuples (e.g. ([week1, week2, ...], [dow1, dow2, ...], [sod1, sod2, ...]) instead of
        [(week1, dow1, sod1), (week2, dow2, sod2), ...]. This function is used to generate the desired output structure.

    Args:
        *args (int, int, float) or (int, float): Time components of the tuple for dow, doy, sow, or soy

    Returns:
        tuple(list) or list: A tuple is returned if there are multiple times, and will be a tuple of lists.
                             A list is returned if there is just one time being converted.
    """
    if isinstance(args[0], np.ndarray):
        output = np.vstack(args).T
    elif isinstance(args[0], Iterable):
        output_zip = tuple(zip(*args))
        output = tuple(list(i) for i in output_zip)
    else:
        output = list(args)
    return output

def calc_ipp(rx_pos, sv_pos, iono_ht):
    """
    Computes the Ionosphere pierce Point.
    Args:
        rx_pos (list): Receiver position. ECEF [km]
        sv_pos (list): Satellite position. ECEF [km]
        iono_ht (float): VTEC grid height [km]

    Returns:
        xyz_Il (array): IPP position ECEF [km]
    """
    pos = Position(rx_pos[0], rx_pos[1], rx_pos[2])
    r_e = 1e-3 * Position.radiusEarth(pos)
    xyz_1 = np.array(rx_pos)
    xyz_2 = np.array(sv_pos)
    iono_ht = iono_ht + r_e
    d = xyz_2 - xyz_1
    a = np.dot(d, d)
    b = 2 * np.dot(xyz_1, d)
    c = np.dot(xyz_1, xyz_1) - iono_ht * iono_ht
    t1 = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    xyz_I1 = xyz_1 + (t1 * d.T).T
    return xyz_I1


def form_double_differences(antenna_handles, log_file=None):
    """
    Forms double differences from xarray data in antenna_handles
    Returns:
        baseline_handles (array): the baselineInfo objects for double-differenced data
    """

    return baseline_handles

def xyz_to_rll(xyz, in_deg=True):
    """
    Converts a cartesian point to a spherical radius/latitude/longitude
    Args:
        xyz (float, array_like) cartesian coordinates x, y, z

    Keyword Args:
        in_deg (bool): flag for angular units (default=True)

    Returns:
        r, lat, lon (float, tuple): spherical coordinates radius, latitude, longitude
    """
    x, y, z = np.asarray(xyz)
    rad = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    lat = np.arcsin(z / rad)
    lon = np.arctan2(y, x)
    if in_deg:
        lat = np.rad2deg(lat)
        lon = np.rad2deg(lon)
    return rad, lat, lon

def hms_to_deg(hms):
    """ convert HH:MM:SS to decimal degrees """
    h, m, s = [float(x) for x in hms.split(':')]
    return (h + m/60 + s/3600) * 15

def dms_to_deg(dms):
    """ convert DD:MM:SS to decimal degrees"""
    sign = -1 if dms.startswith('-') else 1
    d, m, s = [float(x) for x in dms.strip('+-').split(':')]
    return sign * (d + m / 60 + s / 3600)

def import_key_gnss(rinex_files, full_data, key_file, sim_data_rate=1):
    """
    Read key file and output source arrays corresponding to good data.
    NB: currently supports only single-frequency
    """
    sources_ra_dec, full_source_names, duration_array, source_array, ra_array, dec_array, datetime_array, scan_nums = read_key(key_file)

    # find all data within duration and with relevant quantities
    source_array_full = []
    datetime_array_full = []
    point_ra_dec_full = []
    point_ra_dec = []
    nscans=0
    for time_idx, time_obs in enumerate(datetime_array):
        times_arr = []
        src = source_array[time_idx]
        if len(rinex_files)>0:
            for ant_idx, rinex_file in enumerate(rinex_files):
                rinex_obs = full_data[rinex_file]
                beg_time = time_obs 
                end_time = time_obs + to_timedelta(duration_array[time_idx], unit='s')
                try: 
                    obs_sv = rinex_obs.sel(sv=src).dropna(dim='time',how='all')
                    obs_time = obs_sv.sel(time=slice(beg_time,end_time))
                except: # no observations of the satellite at this epoch
                    times_arr = []
                    break
                # check that data is good
                SNR_vars = [var for var in obs_time.data_vars if var.startswith('S')]
                idxs_good = np.zeros(len(obs_time.C1.values), dtype=bool)
                for SNR_var in SNR_vars:
                    idxs_good = np.bitwise_or(idxs_good, ~np.isnan(obs_time[str(SNR_var)].values))
                    
                #idxs_good = np.bitwise_and(~np.isnan(obs_time.C1.values), ~np.isnan(obs_time.S1C.values))
                if ant_idx == 0: 
                    times_arr = np.array(obs_time.time)
                else:
                    times_arr = np.union1d(times_arr, np.array(obs_time.time[idxs_good]))
        else:
            if sim_data_rate != 0: 
                duration = duration_array[time_idx]
                time_dt = np.datetime64(time_obs, 'ns')
                end_time = time_dt  + np.timedelta64(int(duration*1e9), 'ns')
                times_arr = np.arange(time_dt, end_time + np.timedelta64(int(sim_data_rate*1e9), 'ns'), np.timedelta64(int(sim_data_rate*1e9), 'ns'))
            else:
                times_arr = [np.datetime64(time_obs, 'ns')]

        full_source = full_source_names[time_idx]
        if len(sources_ra_dec)>0:
            source_idx = np.argwhere(sources_ra_dec==full_source)
            if len(source_idx) == 0 and len(ra_array) > 0:
                raise Exception('Source ' + full_source + ' in key file has no dish pointing angle')
        if len(ra_array) > 0:
            ra = ra_array[source_idx[0][0]]
            dec = dec_array[source_idx[0][0]]
            point_ra_dec.append((ra,dec))

        if len(times_arr)>0:
            # we have observations of this source
            nscans+=1
        for time in times_arr:
            datetime_array_full.append(time)
            source_array_full.append(src)
            if len(ra_array) > 0:
                point_ra_dec_full.append((ra, dec))
    print('Number of scans: ' + str(nscans))
    if len(point_ra_dec_full) == 0:
        point_ra_dec_full = None

    return datetime_array_full, source_array_full, point_ra_dec_full, datetime_array, duration_array, source_array, point_ra_dec


def read_thermal_deformation_coeffs(ant_info_file, antenna_handles):
    """
    Read an antenna info file as specified in Nothnagel (2008)  10.1007/s00190-008-0284-z
    """
    # Define the format of fields based on the file description
    def parse_antenna_line(line):
        return {
            "antenna_name": line[14:22].strip(),
            "fo_type": line[24:31].strip(),
            "ref_temp": float(line[57:61].strip()),
            "sin_amp": float(line[62:66].strip()),
            "cos_amp": float(line[67:71].strip()),
            "ref_pressure": float(line[72:78].strip()),
            "antenna_diameter": float(line[80:85].strip()),
            "height_foundation": float(line[86:93].strip()),
            "depth_foundation": float(line[94:100].strip()),
            "foundation_coeff": float(line[102:109].strip()),
            "length_fixed_axis": float(line[111:118].strip()),
            "fixed_axis_coeff": float(line[119:126].strip()),
            "axis_offset_length": float(line[128:135].strip()),
            "axis_offset_coeff": float(line[136:143].strip()),
            "dist_mov_axis_vertex": float(line[145:152].strip()),
            "mov_axis_vertex_coeff": float(line[153:160].strip()),
            "sub_reflector_height": float(line[162:169].strip()),
            "sub_reflector_coeff": float(line[170:177].strip())
        }

    # Open the file and process each line
    with open(ant_info_file, 'r') as ant_file:
        for line in ant_file:
            if line.startswith("ANTENNA_INFO"):
                antenna_data = parse_antenna_line(line)
                antenna_name = antenna_data["antenna_name"]

                # Check if any antenna handle matches the current antenna name
                for antenna_handle in antenna_handles:
                    if antenna_handle.antenna_name in antenna_name:
                        # Save the coefficients using the object's method
                        thermal_coeffs = [
                            antenna_data["ref_temp"],
                            antenna_data["fo_type"],
                            antenna_data["sin_amp"],
                            antenna_data["cos_amp"],
                            antenna_data["ref_pressure"],
                            antenna_data["antenna_diameter"],
                            antenna_data["height_foundation"],
                            antenna_data["depth_foundation"],
                            antenna_data["foundation_coeff"],
                            antenna_data["length_fixed_axis"],
                            antenna_data["fixed_axis_coeff"],
                            antenna_data["axis_offset_length"],
                            antenna_data["axis_offset_coeff"],
                            antenna_data["dist_mov_axis_vertex"],
                            antenna_data["mov_axis_vertex_coeff"],
                            antenna_data["sub_reflector_height"],
                            antenna_data["sub_reflector_coeff"]
                        ]
                        antenna_handle.save_thermal_coeffs(thermal_coeffs)
    return 

def get_spanning_tree(antenna_handles, maximize):
    """
    Spanning-tree baseline set over antenna_handles.

    maximize=True  -> OBS-MAX : maximum spanning tree weighted by the number of
                      common (epoch, satellite) observations the two receivers
                      share (i.e. the SD-able observations, pr_data finite).
    maximize=False -> SHORTEST: minimum spanning tree weighted by geometric
                      ECEF baseline length (from the 'position' attribute).

    Returns `baselines`: list of (i, j) index pairs (i < j) into
    antenna_handles, length len(antenna_handles) - 1.
    """
    n = len(antenna_handles)
    if n < 2:
        return []

    datasets = [h.antenna_data for h in antenna_handles]

    if maximize:
        # OBS-MAX: set of (time_ns, sv) keys with valid pr_data, per antenna
        def obs_keys(ds):
            t = ds['time'].values.astype('datetime64[ns]').astype(np.int64)
            sv = np.asarray(ds['sv'].values, dtype=str)
            pr = np.asarray(ds['pr_data'].values, dtype=float)
            good = np.isfinite(pr)
            return set(zip(t[good].tolist(), sv[good].tolist()))

        keys = [obs_keys(ds) for ds in datasets]

        def weight(edge):
            i, j = edge
            return len(keys[i] & keys[j])              # common-obs count
    else:
        # SHORTEST: ECEF position per antenna
        pos = [np.asarray(ds.attrs['position'], dtype=float) for ds in datasets]

        def weight(edge):
            i, j = edge
            return float(np.linalg.norm(pos[i] - pos[j]))  # baseline length

    # Complete graph, sorted by the chosen weight. Python's sort is stable, so
    # equal-weight edges keep itertools' lexicographic order -> deterministic.
    edges = sorted(itertools.combinations(range(n), 2),
                   key=weight, reverse=maximize)

    # Kruskal + union-find: accept an edge only if it joins two components.
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]              # path compression
            x = parent[x]
        return x

    baselines = []
    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            baselines.append((i, j))
            if len(baselines) == n - 1:
                break

    return baselines

def hms_to_decimal(h, m, s):
    return h + m / 60.0 + s / 3600.0

def dms_to_decimal(d, m, s, sign):
    decimal = d + m / 60.0 + s / 3600.0
    if sign == '-':
        decimal *= -1
    return decimal

def read_src(filename):
    """ Read a source a priori coordinates file from PIMA/pSolve """
    src_dict = {}
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('#') or line.strip() == "":
                continue
            ivs_name = line[0:11].strip()
            ra_h = int(line[25:27].strip())
            ra_m = int(line[28:30].strip())
            ra_s = float(line[31:41].strip())
            dec_sign = line[42].strip()
            dec_d = int(line[43:45].strip())
            dec_m = int(line[46:48].strip())
            dec_s = float(line[49:58].strip())
            ra = hms_to_decimal(ra_h, ra_m, ra_s) * 15  # Convert hours to degrees
            dec = dms_to_decimal(dec_d, dec_m, dec_s, dec_sign)
            src_dict[ivs_name] = (ra,dec)
    return src_dict

def read_key(key_file):
    """
    Read key file and output source arrays corresponding to good data.
    """
    # Regular expressions
    year_pattern = re.compile(r'year=\s*(\d{4})')
    day_pattern = re.compile(r'day=\s*(\d{3})')
    start_pattern = re.compile(r'start=\s*(\d{2}:\d{2}:\d{2})')
    source_pattern = re.compile(r'source=\s*(\w+)')
    duration_pattern = re.compile(r'dur=\s*(\d+\.*\d*)')
    
    datetime_array = []
    source_array = []
    point_ra_dec = []
    full_source_names = []
    duration_array = []

    ra_array = []
    dec_array = []
    sources_ra_dec = []
    scan_nums = []
    UTC2GPS = 0
    # Reading the key file
    with open(key_file, 'r') as f:
        lines = f.readlines()
    
        for line in lines:
            # Extracting year, day and start
            test_year = year_pattern.search(line)
            if test_year is None and not line.startswith("source"): 
                continue # header line -- skip
            elif line.startswith("source"):
                # source right ascension and declination 
                parts = line.split()
                source = parts[1].strip("'")
                hms_split = parts[2].split('=')
                hms = hms_split[1]
                dms_split = parts[3].split('=')
                dms = dms_split[1]
                ra = hms_to_deg(hms)
                dec = dms_to_deg(dms)
           
                sources_ra_dec.append(source)
                ra_array.append(ra)
                dec_array.append(dec) 
            else:
                # datetime of observations
                year = int(year_pattern.search(line).group(1))
                day = int(day_pattern.search(line).group(1))
                start = start_pattern.search(line).group(1)
                hour, minute, second = map(int, start.split(":"))

                parts = line.split()
                scan_nums.append(parts[13])
                
                # Using datetime to combine the information
                dt = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day-1, hours=hour, minutes=minute, seconds=second)
                if UTC2GPS == 0:
                    common_time = date_to_common(dt)
                    civt = CivilTime()
                    civt.convertFromCommonTime(common_time)
                    UTC2GPS = getTimeSystemCorrection(TimeSystem.UTC, TimeSystem.GPS, civt.year, civt.month, civt.day)
                dt = dt + datetime.timedelta(seconds=UTC2GPS)
                datetime_array.append(dt)
    
                # Extracting source
                source = source_pattern.search(line).group(1)
                full_source_names.append(source)
                if len(source) > 3:
                    # E01523203
                    source_array.append(source[0] + source[2] + source[3])
                else:
                    # E15
                    source_array.append(source[0:3])

    
                # Extracting duration
                duration_seconds = float(duration_pattern.search(line).group(1))
                duration_array.append(duration_seconds)
    
    sources_ra_dec = np.array(sources_ra_dec)

    return sources_ra_dec, full_source_names, duration_array, source_array, ra_array, dec_array, datetime_array, scan_nums 

def read_psolve_file(psolve_file):
    """ Read group and phase delays from text file exported by PIMA for pSolve"""
    # Initialize a dictionary to hold the arrays organized by the Sta parameter
    data_by_sta = {}
    
    # Define the keywords to extract
    keywords = ["TIM_BEG", "SNR", "SRT_off", "FRT_off_1", "The_gr", "Tot_gr_1", "Tot_phs_1", "Elev_1", "Azim_1", "Ref_frq_1", "Err_gr_1", "Ef_dur1"]
    baselines = []
    # Open the file and read line by line
    with open(psolve_file, "r") as file:
        for line in file:
            # Skip header lines
            if line.startswith("#"):
                continue
    
            # Extract the Sta, SRT parameters
            sta_match = re.search(r"Sta: (\S+\s*/\s*\S+)", line)
            sou_match = re.search(r"Sou: (\S+\s*)", line)
            srt_match = re.search(r"SRT: (\d{4}\.\d{2}\.\d{2}-\d{2}:\d{2}:\d{2}\.\d{4})", line)

            if sta_match and srt_match:
                sta_pair = sta_match.group(1)
                srt = datetime.datetime.strptime(srt_match.group(1), "%Y.%m.%d-%H:%M:%S.%f")
    
                # Initialize arrays for this Sta if not already present
                if sta_pair not in data_by_sta:
                    data_by_sta[sta_pair] = {key: [] for key in keywords}
                    data_by_sta[sta_pair]["SRT"] = []
                    data_by_sta[sta_pair]["Sou"] = []
                    baselines.append(sta_pair)

                data_by_sta[sta_pair]["SRT"].append(srt)
                data_by_sta[sta_pair]["Sou"].append(sou_match.group(1))
                # Extract and store the values for each keyword
                for key in keywords:
                    value_match = re.search(rf"{key}\s*[:=]\s*([-\d\.]+(?:[DdEe][+-]?\d+)?)", line)
                    if value_match:
                        # Convert D notation to E notation for float conversion
                        value = float(value_match.group(1).replace("D", "E").replace("d", "e"))
                        data_by_sta[sta_pair][key].append(value)
                    if not value_match and key == 'SNR':
                        data_by_sta[sta_pair][key].append(20000) # fix ***** PIMA issue
    return data_by_sta, baselines


def import_data_vlbi(fringe_file, antennas, baselines, rinex_sats, key_file=None):
    """
    Read key file and fringe file, output good data
    """
    data_by_sta, baselines_ps = read_psolve_file(fringe_file)
    
    point_ra_dec_key = []
    if False: #key_file is not None:
        sources_ra_dec, full_source_names, duration_array, source_array_key, ra_array, dec_array, datetime_array_full, scan_nums = read_key(key_file)
        for idx in range(len(ra_array)):
            ra = ra_array[idx]
            dec = dec_array[idx]
            point_ra_dec_key.append((ra,dec))

    # set up array for epochs observed by each antenna
    dt_ant = {}
    for ant in antennas:
        dt_ant[ant] = np.array([],dtype=np.datetime64)

    duration_dict = {}
    sources_dict = {}
    
    if False: # key_file is not None:
        datetime_array_srt = np.array(deepcopy(datetime_array_full))
    baseline_handles = []
    for jdx, baseline in enumerate(baselines):
        antenna1 = antennas[baseline[0]]
        antenna2 = antennas[baseline[1]]
        baseline_use = None
        for b_key in data_by_sta.keys():
            if antenna1 in b_key and antenna2 in b_key:
                baseline_use = b_key
        if baseline_use is None:
            raise ValueError('baseline ' + antenna1 +' — ' + antenna2 +' not found in fringe data')

        data = data_by_sta[baseline_use]
        group_delays = np.array(data['Tot_gr_1'])
        phase_delays = np.array(data['Tot_phs_1'])
        sources = np.array(data['Sou'])
        dt_psolve = np.array(data['SRT'])
        SNR_obs = np.array(data['SNR'])
        SRT_off = np.array(data['SRT_off'])
        FRT = np.array(data['FRT_off_1'])
        #ELEV = np.array(data['Elev_1'])
        frq = np.array(data['Ref_frq_1'])
        dur = np.array(data['Ef_dur1'])
        grdel_err = np.array(data['Err_gr_1'])
        phdel_err = 1/(2*np.pi*frq*SNR_obs)
        
        # take only good data
        use_idxs = np.argwhere(SNR_obs > 100) # limit to 100 SNR
        
        # further restrict with a quick group delay check
        test = (group_delays-np.median(group_delays))
        sigma_test, use_idxs_gr = find_sigmas(group_delays[use_idxs], 10)
        use_idxs = use_idxs[use_idxs_gr]

        group_delays = group_delays[use_idxs] 
        phase_delays = phase_delays[use_idxs] 
        dt_psolve = dt_psolve[use_idxs]
        SRT_off = SRT_off[use_idxs]
        FRT = FRT[use_idxs]
        #ELEV = ELEV[use_idxs]
        frq = frq[use_idxs]
        dur = dur[use_idxs]
        sources = sources[use_idxs]
        grdel_err = grdel_err[use_idxs]
        phdel_err = phdel_err[use_idxs]

        # take only observations with corresponding ephemeris
        idxs_eph = []
        for idx, source in enumerate(sources):
            if len(source.strip())>3:
                sat_id = source.split('_')[1][:3]
                sources[idx] = sat_id # fix this for later
            else:
                sat_id = source.strip()
            if sat_id in rinex_sats:
                idxs_eph.append(idx)
        group_delays = group_delays[idxs_eph] 
        phase_delays = phase_delays[idxs_eph] 
        dt_psolve = dt_psolve[idxs_eph]
        SRT_off = np.round(SRT_off[idxs_eph])
        FRT = FRT[idxs_eph]
        sources = sources[idxs_eph]
        #ELEV = ELEV[idxs_eph]
        frq = frq[idxs_eph]
        dur = dur[idxs_eph]
        grdel_err = grdel_err[idxs_eph]*const.c
        phdel_err = phdel_err[idxs_eph]*const.c
        
        # correct datetime from scan reference time (TAI) to GPS time
        dt_ref = []
        for idx in range(len(dt_psolve)):
            dt_psolve[idx] = dt_psolve[idx] + datetime.timedelta(seconds=TAI2GPS)
            dt_ref.append(dt_psolve[idx] - datetime.timedelta(seconds=SRT_off[idx]))
        dt_ref = np.array(dt_ref)
         
        if False: #key_file is not None:
            datetime_common, idxs_key, idxs_psolve = np.intersect1d(datetime_array_full, dt_ref, return_indices=True)
            datetime_array = np.array(dt_psolve[idxs_psolve], dtype='datetime64[ns]')
            # set output datetime array to scan reference time rather than beginning of scan
            if jdx == 0:
                dt_ref_full = dt_ref
                datetime_array_srt = dt_psolve[idxs_psolve]
                dur = dur[idxs_psolve]
            else:
                dt_ref_full = np.union1d(dt_ref_full, dt_ref)
                datetime_array_srt = np.union1d(datetime_array_srt, dt_psolve[idxs_psolve])

            dt_psolve = dt_psolve[idxs_psolve]
            group_delays = group_delays[idxs_psolve]
            phase_delays = phase_delays[idxs_psolve]
            frq = frq[idxs_psolve]
            sources = sources[idxs_psolve]
        else:
            datetime_array = np.array(dt_psolve, dtype='datetime64[ns]')
            if jdx == 0:
                dt_ref_full = dt_ref
                datetime_array_srt = dt_psolve
            else:
                dt_ref_full = np.union1d(dt_ref_full, dt_ref)
                datetime_array_srt = np.union1d(datetime_array_srt, dt_psolve)

            for idx, time in enumerate(dt_psolve):
                sources_dict[time] = sources[idx].strip()

        for idx, time in enumerate(np.array(dt_psolve, dtype='datetime64[ns]')):
            duration_dict[time] = dur[idx]

        # check if psolve baselines match baselines array
        if baseline_use.split()[0] != antennas[baseline[0]]:
            # need to change the sign of the measurements
            group_delays = -group_delays
            phase_delays = -phase_delays
            
        group_delays = group_delays*const.c # group delay in meters
        # get correlation between ambiguities by adding nearest integer from group delay -- this will cause linked measurements hopefully
        wavelength = const.c/frq
        amb_int = np.rint(group_delays/wavelength)
        phase_delays = (phase_delays/(2*np.pi) + amb_int) * wavelength
        # use frq[0] assuming frq is all the same frequency; could be broken in the future
        baseline_handle = BaselineInfo(datetime_array, frq[0])
        baseline_handle.prepare_vlbi(group_delays, phase_delays, grdel_err, phdel_err)
        #baseline_handle.prepare_vlbi(group_delays, phase_delays, grdel_err, phdel_err, group_delays_dual, phase_delays_dual)
        baseline_handles.append(baseline_handle)

        # add observed epochs to datetimes for each antenna
        dt_ant[antenna1] = np.union1d(dt_ant[antenna1], datetime_array)
        dt_ant[antenna2] = np.union1d(dt_ant[antenna2], datetime_array)
    
    if False: #key_file is not None:
        # get relevant pointing and source array entries
        _, idxs_full, _ = np.intersect1d(datetime_array_full, dt_ref_full, return_indices=True)
        if len(point_ra_dec_key) > 0:
            point_ra_dec_full = np.array(point_ra_dec_key)[idxs_full]
        else:
            point_ra_dec_full = []
        source_array_full = np.array(source_array_key)[idxs_full]
    else:
        point_ra_dec_full = []
        source_array_full = [sources_dict[time] for time in datetime_array_srt]
    
    datetime_array_srt = np.array(datetime_array_srt, dtype='datetime64[ns]')
    print('Number of scans: ' + str(len(datetime_array_srt)))
    return baseline_handles, datetime_array_srt, source_array_full, point_ra_dec_full, dt_ant, duration_dict


TOLERANCE_SECONDS=100
def match_idx_offset(time_array1, source_array1, time_array2, source_array2):
    """
    Matches indices of two time arrays within a specified time tolerance and where the corresponding sources are equal.

    Parameters:
    - time_array1: np.ndarray of datetime64, first set of observation times.
    - source_array1: np.ndarray of str, sources corresponding to time_array1.
    - time_array2: np.ndarray of datetime64, second set of observation times.
    - source_array2: np.ndarray of str, sources corresponding to time_array2.
    - tolerance_seconds: int, maximum allowable time difference in seconds for a match.

    Returns:
    - matched_idx1: np.ndarray of indices in time_array1 that match time_array2.
    - matched_idx2: np.ndarray of indices in time_array2 that match time_array1.
    """
    # Convert time arrays to pandas datetime for easier time manipulation
    time1 = to_datetime(time_array1)
    time2 = to_datetime(time_array2)
    
    # Tolerance for time difference
    tolerance = Timedelta(seconds=TOLERANCE_SECONDS)
    
    # Initialize lists to store matched indices
    matched_idx1 = []
    matched_idx2 = []
    
    # Loop through time_array1 and find matches in time_array2
    for idx1, t1 in enumerate(time1):
        # Calculate time differences
        time_diffs = np.abs(time2 - t1)
        
        # Find indices within tolerance
        close_matches = np.where(time_diffs <= tolerance)[0]
        
        # Check for source equality among close matches
        for idx2 in close_matches:
            if source_array1[idx1] == source_array2[idx2]:
                matched_idx1.append(idx1)
                matched_idx2.append(idx2)
                break  # Stop after the first match for this time entry
    
    return np.array(matched_idx1), np.array(matched_idx2)

    
def import_data_nc_sim(nc_file, antennas, baselines, key_file=None):
    """
    Read key file and fringe file, output good data
    """
    point_ra_dec_full = []
    if key_file is not None:
        sources_ra_dec, full_source_names, duration_array, source_array_full, ra_array, dec_array, datetime_array_full, scan_nums = read_key(key_file)
        for idx in range(len(ra_array)):
            ra = ra_array[idx]
            dec = dec_array[idx]
            point_ra_dec_full.append((ra,dec))

    # set up array for epochs observed by each antenna
    dt_ant = {}
    for ant in antennas:
        dt_ant[ant] = np.array([],dtype=np.datetime64)
    
    nc_data = xr.open_dataset(nc_file)
    freq = nc_data.attrs["frequency"]

    baseline_handles = []
    for baseline in baselines:
        antenna1 = antennas[baseline[0]]
        antenna2 = antennas[baseline[1]]
        try: 
            nc_baseline = nc_data.sel(baseline=antenna1+'-'+antenna2).dropna(dim='time')
            reverse_sign = False
        except: 
            nc_baseline = nc_data.sel(baseline=antenna2+'-'+antenna1).dropna(dim='time')
            reverse_sign = True

        group_delays = nc_baseline.group_delay.values.squeeze()
        phase_delays = nc_baseline.phase_delay.values.squeeze()
        grdel_err = 1e-3*np.ones(len(group_delays))  # Placeholder, no error values from sim
        phdel_err = 1e-4*np.ones(len(phase_delays))
        dt_ref = nc_baseline.time.values  # Extract time coordinates

        if key_file is not None:
            dt_arr_np = np.array(datetime_array_full, dtype='datetime64[ns]')
            #datetime_common, idxs_key, idxs_nc = np.intersect1d(dt_arr_np, dt_ref, return_indices=True)
            idxs_key, idxs_nc = match_idx_offset(dt_arr_np, source_array_full, \
                    nc_baseline.group_delay.time.values, nc_baseline.group_delay.source.values)
            datetime_array = dt_ref[idxs_nc]
            point_ra_dec_full = np.array(point_ra_dec_full)[idxs_key]
            source_array_full =  np.array(source_array_full)[idxs_key]
        else:
            datetime_array = nc_baseline.group_delay.time.values
            source_array_full = nc_baseline.group_delay.source.values
            ra_list = nc_baseline.group_delay.source_ra.values
            dec_list = nc_baseline.group_delay.source_dec.values
            point_ra_dec_full = np.array(list(zip(ra_list, dec_list)))

        # check if psolve baselines match baselines array
        if reverse_sign is True:
            # need to change the sign of the measurements
            group_delays = -group_delays
            phase_delays = -phase_delays
            
        # get correlation between ambiguities by adding nearest integer from group delay -- this will cause linked measurements hopefully
        baseline_handle = BaselineInfo(datetime_array, freq)
        baseline_handle.prepare_vlbi(group_delays, phase_delays, grdel_err, phdel_err)
        #baseline_handle.prepare_vlbi(group_delays, phase_delays, grdel_err, phdel_err, group_delays_dual, phase_delays_dual)
        baseline_handles.append(baseline_handle)

        # add observed epochs to datetimes for each antenna
        dt_ant[antenna1] = np.union1d(dt_ant[antenna1], datetime_array)
        dt_ant[antenna2] = np.union1d(dt_ant[antenna2], datetime_array)

    return baseline_handles, datetime_array, source_array_full, point_ra_dec_full, dt_ant


def import_data_vlbi_farfield(fringe_file, antennas, baselines, src_dict):
    """
    Read key file and fringe file, output good data
    """
    data_by_sta, baselines_ps = read_psolve_file(fringe_file)
 

    # set up array for epochs observed by each antenna
    dt_ant = {}
    for ant in antennas:
        dt_ant[ant] = np.array([],dtype=np.datetime64)

    duration_dict = {}

    datetime_array_srt = np.array([],dtype=np.datetime64)
    src_time_dict = {}
    baseline_handles = []
    for baseline in baselines:
        antenna1 = antennas[baseline[0]]
        antenna2 = antennas[baseline[1]]
        baseline_use = None
        for b_key in data_by_sta.keys():
            if antenna1 in b_key and antenna2 in b_key:
                baseline_use = b_key
        if baseline_use is None:
            raise ValueError('baseline ' + antenna1 +' — ' + antenna2 +' not found in fringe data')

        data = data_by_sta[baseline_use]
        group_delays = np.array(data['Tot_gr_1'])
        phase_delays = np.array(data['Tot_phs_1'])
        sources = np.array(data['Sou'])
        dt_psolve = np.array(data['SRT'])
        SNR_obs = np.array(data['SNR'])
        FRT = np.array(data['FRT_off_1'])
        ELEV = np.array(data['Elev_1'])
        frq = np.array(data['Ref_frq_1'])
        dur = np.array(data['Ef_dur1'])
        grdel_err = np.array(data['Err_gr_1'])
        phdel_err = 1/(2*np.pi*frq*SNR_obs)
        
        # take only good data
        use_idxs = SNR_obs > 10 # limit to 100 SNR
        group_delays = group_delays[use_idxs] 
        phase_delays = phase_delays[use_idxs] 
        dt_psolve = dt_psolve[use_idxs]
        FRT = FRT[use_idxs]
        ELEV = ELEV[use_idxs]
        frq = frq[use_idxs]
        dur = dur[use_idxs]
        sources = sources[use_idxs]
        grdel_err = grdel_err[use_idxs]*const.c
        phdel_err = phdel_err[use_idxs]*const.c

        # correct datetime from scan reference time (TAI) to GPS time
        for idx in range(len(dt_psolve)):
            dt_psolve[idx] = dt_psolve[idx] + datetime.timedelta(seconds=TAI2GPS)
 
        datetime_array = np.array(dt_psolve, dtype='datetime64[ns]')
        
        for idx, source in enumerate(sources):
            src_time_dict[datetime_array[idx]] = source

        # check if psolve baselines match baselines array
        if baseline_use.split()[0] != antennas[baseline[0]]:
            # need to change the sign of the measurements
            group_delays = -group_delays
            phase_delays = -phase_delays
            
        group_delays = group_delays*const.c # group delay in meters
        # get correlation between ambiguities by adding nearest integer from group delay -- this will cause linked measurements hopefully
        wavelength = const.c/frq
        amb_int = np.rint(group_delays/wavelength)
        phase_delays = (phase_delays/(2*np.pi) + amb_int) * wavelength
        # use frq[0] assuming frq is all the same frequency; could be broken in the future
        baseline_handle = BaselineInfo(datetime_array, frq[0])
        baseline_handle.prepare_vlbi(group_delays, phase_delays, grdel_err, phdel_err)
        #baseline_handle.prepare_vlbi(group_delays, phase_delays, grdel_err, phdel_err, group_delays_dual, phase_delays_dual)
        baseline_handles.append(baseline_handle)

        for idx, time in enumerate(np.array(dt_psolve, dtype='datetime64[ns]')):
            duration_dict[time] = dur[idx]

        # add observed epochs to datetimes for each antenna
        dt_ant[antenna1] = np.union1d(dt_ant[antenna1], datetime_array)
        dt_ant[antenna2] = np.union1d(dt_ant[antenna2], datetime_array)
        datetime_array_srt = np.union1d(datetime_array_srt, datetime_array)

    datetime_array_srt = np.array(datetime_array_srt, dtype='datetime64[ns]')
    point_ra_dec_full = []
    source_array_full = []
    for time in datetime_array_srt: 
        source = src_time_dict[time]
        ra, dec = src_dict[source.strip()]
        source_array_full.append(source)
        point_ra_dec_full.append((ra,dec))
    print('Number of scans: ' + str(len(datetime_array_srt)))
    return baseline_handles, datetime_array_srt, source_array_full, point_ra_dec_full, dt_ant, duration_dict

def parse_ra_dec(ra_hours, ra_mins, ra_secs, dec_sign, dec_degs, dec_mins, dec_secs):
    """Convert RA and DEC from hms/dms to decimal degrees."""
    ra_total_seconds = float(ra_hours) * 3600 + float(ra_mins) * 60 + float(ra_secs)
    ra_degrees = (ra_total_seconds / 3600) * 15  # Convert hours to degrees (15 degrees per hour)
    
    dec_total_seconds = float(dec_degs) * 3600 + float(dec_mins) * 60 + float(dec_secs)
    dec_degrees = dec_total_seconds / 3600
    if dec_sign == '-':
        dec_degrees = -dec_degrees
        
    return ra_degrees, dec_degrees

def parse_frequency(frequency_str):
    """Parse frequency considering scientific notation."""
    if 'D' in frequency_str:
        frequency_str = frequency_str.replace('D', 'E')  # Replace D with E for Python notation
    return float(frequency_str)

def parse_ngs_file(filepath, antenna_names):
    source_positions = {}
    baseline_observations = {}
    
    with open(filepath, 'r') as file:
        lines = file.readlines()
        
    i = 0
    while True: # header
        line = lines[i].strip()
        i += 1
        if line.startswith('$END'):
            break
    # Parse Source Position Cards
    while True:
        line = lines[i].strip()
        i += 1
        if line.startswith('$END'):
            break
        if len(line) == 48:  # Ensuring it's a source line
            source = line[0:8].strip()
            ra_h = line[10:12].strip()
            ra_m = line[13:15].strip()
            ra_s = line[16:28].strip()
            dec_sign = line[29].strip()
            dec_d = line[30:32].strip()
            dec_m = line[33:35].strip()
            dec_s = line[36:48].strip()
            ra_degrees, dec_degrees = parse_ra_dec(ra_h, ra_m, ra_s, dec_sign, dec_d, dec_m, dec_s)
            source_positions[source] = (ra_degrees, dec_degrees)

    # Parse Auxiliary Parameters for frequency
    while not lines[i].startswith('$END'):
        line = lines[i]
        i += 1
        if 'GR' in line or 'PH' in line:  # Check for line containing frequency
            frequency = parse_frequency(line.split()[0])*1e6
            break
    
    i += 1  # Skip past $END of Auxiliary Parameters

    # Parse Data Cards
    while i < len(lines):
        line = lines[i].strip()
        site1 = line[0:8].strip()
        site2 = line[10:18].strip()
        if site1 in antenna_names and site2 in antenna_names:
            baseline_key = f'{site1}_{site2}'
            source = line[20:28].strip()
            year = int(line[29:33].strip())
            month = int(line[34:36].strip())
            day = int(line[37:39].strip())
            hour = int(line[40:42].strip())
            minute = int(line[43:45].strip())
            second = float(line[46:60].strip())
            obs_datetime = datetime.datetime(year, month, day, hour, minute, int(second), int((second - int(second)) * 1e6))
            delay = float(lines[i+1][0:20].strip())
            delay_err = float(lines[i+1][20:31].strip())
            quality_flag = float(lines[i+1][60:62].strip())
            fringe_phase = float(lines[i+2][40:60].strip())
            if quality_flag <= 0:
                observation = {
                    'datetime': obs_datetime,
                    'source': source,
                    'delay': delay,
                    'delay_err': delay_err,
                    'fringe_phase': fringe_phase,
                    'frequency': frequency,
                }
                if baseline_key not in baseline_observations:
                    baseline_observations[baseline_key] = []
                baseline_observations[baseline_key].append(observation)
        i += 8  # Skip to the next set of data cards

    return source_positions, baseline_observations

def import_data_vlbi_ngs(ngs_file, antennas, baselines, utc2gps):
    """
    Read key file and fringe file, output good data
    """
    src_dict, baseline_observations = parse_ngs_file(ngs_file, antennas)

    # set up array for epochs observed by each antenna
    dt_ant = {}
    for ant in antennas:
        dt_ant[ant] = np.array([],dtype=np.datetime64)

    datetime_array_srt = np.array([],dtype=np.datetime64)
    src_time_dict = {}
    baseline_handles = []
    for baseline in baselines:
        antenna1 = antennas[baseline[0]]
        antenna2 = antennas[baseline[1]]
        baseline_use = None
        for b_key in baseline_observations.keys():
            if antenna1 in b_key and antenna2 in b_key:
                baseline_use = b_key
        if baseline_use is None:
            raise ValueError('baseline ' + antenna1 +' — ' + antenna2 +' not found in fringe data')

        data = baseline_observations[baseline_use]
        dt_utc = []
        sources = []
        group_delays = []
        grdel_err = []
        phase_delays = []
        frequencies = []
        for obs in data:
            dt_utc.append(obs['datetime'])
            sources.append(obs['source'])
            group_delays.append(obs['delay'])
            grdel_err.append(obs['delay_err'])
            phase_delays.append(obs['fringe_phase'])
            frequencies.append(obs['frequency'])
        dt_utc = np.array(dt_utc)
        sources = np.array(sources)
        group_delays = np.array(group_delays)
        grdel_err = np.array(grdel_err)*const.c/1e9
        phdel_err = grdel_err/WEIGHT_PHASE # what goes here???
        
        phase_delays = np.array(phase_delays)
        frq = np.array(frequencies)

        # correct datetime from scan reference time (UTC) to GPS time
        dt_gps = np.zeros_like(dt_utc)
        for idx in range(len(dt_utc)):
            dt_gps[idx] = dt_utc[idx] + datetime.timedelta(seconds=utc2gps)
 
        datetime_array = np.array(dt_gps, dtype='datetime64[ns]')
        
        for idx, source in enumerate(sources):
            src_time_dict[datetime_array[idx]] = source

        # check if baselines match baselines array
        if baseline_use.split('_')[0] != antennas[baseline[0]]:
            # need to change the sign of the measurements
            group_delays = -group_delays
            phase_delays = -phase_delays
            
        group_delays = group_delays*(const.c/1e9) # group delay in meters
        # get correlation between ambiguities by adding nearest integer from group delay -- this will cause linked measurements hopefully
        wavelength = const.c/frq
        amb_int = np.rint(group_delays/wavelength)

        phase_delays = (phase_delays/(2*np.pi) + amb_int) * wavelength
        # use frq[0] assuming frq is all the same frequency; could be broken in the future
        baseline_handle = BaselineInfo(datetime_array, frq[0])
        baseline_handle.prepare_vlbi(group_delays, phase_delays, grdel_err, phdel_err)
        #baseline_handle.prepare_vlbi(group_delays, phase_delays, group_delays_dual, phase_delays_dual)
        baseline_handles.append(baseline_handle)

        # add observed epochs to datetimes for each antenna
        dt_ant[antenna1] = np.union1d(dt_ant[antenna1], datetime_array)
        dt_ant[antenna2] = np.union1d(dt_ant[antenna2], datetime_array)
        datetime_array_srt = np.union1d(datetime_array_srt, datetime_array)

    datetime_array_srt = np.array(datetime_array_srt, dtype='datetime64[ns]')
    point_ra_dec_full = []
    source_array_full = []
    for time in datetime_array_srt: 
        source = src_time_dict[time]
        ra, dec = src_dict[source.strip()]
        source_array_full.append(source)
        point_ra_dec_full.append((ra,dec))
    print('Number of scans: ' + str(len(datetime_array_srt)))
    return baseline_handles, datetime_array_srt, source_array_full, point_ra_dec_full, dt_ant

from collections import defaultdict
def parse_vda_file(file_path, desired_band, antenna_names):
    antenna_positions_list = []
    source_positions = {}
    baseline_observations = defaultdict(list)

    # Temporary storage for data
    sources = {}
    source_indices = {}
    antenna_indices = {}
    antenna_positions = {}
    band_names = {}
    obs_tab_entries = {}  # Mapping of obs_num to entries
    gr_delay = {}
    gr_delerr = {}
    totphase = {}
    phdelerr = {}
    qualcode = {}
    ref_frq = {}
    utc_mtai = None
    utc_obs = {}
    mjd_obs = {}
    dur_obs = {}
    scan_to_source = {}  # Mapping from scan_idx to source_idx

    with open(file_path, 'r') as vda_file:
        for line in vda_file:
            # Skip empty lines or lines without DATA.1 prefix
            if not line.strip() or not (line.startswith('DATA.1') or line.startswith('DATA.2')):
                continue

            parts = line.strip().split()
            if len(parts) < 7:
                continue  # Not enough parts to process

            data_type = parts[1]
            values = parts[2:]

            # Process different data types
            if data_type == 'SOU_COOR':
                # SOU_COOR 0 0 idx1 idx2 value
                idx1 = int(values[2])
                idx2 = int(values[3])
                value = float(values[4].replace('D', 'E'))  # Handle Fortran 'D' exponent
                if idx2 not in sources:
                    sources[idx2] = {}
                sources[idx2][idx1] = value
            elif data_type == 'SRCNAMES':
                idx1 = int(values[2])
                idx2 = int(values[3])
                name = values[4]
                source_indices[idx2] = name
            elif data_type == 'SITNAMES':
                idx1 = int(values[2])
                idx2 = int(values[3])
                name = values[4]
                antenna_indices[idx2] = name
            elif data_type == 'SIT_COOR':
                idx1 = int(values[2])
                idx2 = int(values[3])
                value = float(values[4].replace('D', 'E'))
                if idx2 not in antenna_positions:
                    antenna_positions[idx2] = {}
                antenna_positions[idx2][idx1] = value
            elif data_type == 'BAND_NAM':
                idx1 = int(values[2])
                idx2 = int(values[3])
                band = values[4]
                band_names[idx2] = band
            elif data_type == 'OBS_TAB':
                idx1 = int(values[2])  # Should be 1, 2, or 3
                obs_num = int(values[3])  # Observation number
                value = int(values[4])
                if obs_num not in obs_tab_entries:
                    obs_tab_entries[obs_num] = {}
                obs_tab_entries[obs_num][idx1] = value
            elif data_type == 'GR_DELAY':
                obs_num = int(values[0])
                _ = int(values[1])  # Unused
                band_idx = int(values[2])
                _ = int(values[3])  # Unused (always 1)
                value = float(values[4].replace('D', 'E'))
                gr_delay[(obs_num, band_idx)] = value
            elif data_type == 'GRDELERR':
                obs_num = int(values[0])
                _ = int(values[1])  # Unused
                band_idx = int(values[2])
                _ = int(values[3])  # Unused (always 1)
                value = float(values[4].replace('D', 'E'))
                gr_delerr[(obs_num, band_idx)] = value
            elif data_type == 'TOTPHASE':
                obs_num = int(values[0])
                _ = int(values[1])  # Unused
                band_idx = int(values[2])
                _ = int(values[3])  # Unused (always 1)
                value = float(values[4].replace('D', 'E'))
                totphase[(obs_num, band_idx)] = value
            elif data_type == 'PHDELERR':
                obs_num = int(values[0])
                _ = int(values[1])  # Unused
                band_idx = int(values[2])
                _ = int(values[3])  # Unused (always 1)
                value = float(values[4].replace('D', 'E'))
                phdelerr[(obs_num, band_idx)] = value
            elif data_type == 'QUALCODE':
                obs_num = int(values[0])
                _ = int(values[1])  # Unused
                _ = int(values[2])  # Unused (always 1)
                band_idx = int(values[3])
                value = values[4]
                qualcode[(obs_num, band_idx)] = value
            elif data_type == 'REF_FREQ':
                obs_num = int(values[0])
                _ = int(values[1])  # Unused
                band_idx = int(values[2])
                _ = int(values[3])  # Unused (always 1)
                value = float(values[4].replace('D', 'E'))
                ref_frq[(obs_num, band_idx)] = value
            elif data_type == 'UTC_MTAI':
                utc_mtai = float(values[4].replace('D', 'E'))
            elif data_type == 'UTC_OBS':
                scan_idx = int(values[0])  # scan number
                _ = int(values[1])  # Unused
                _ = int(values[2])  # Unused
                _ = int(values[3])  # Unused
                value = float(values[4].replace('D', 'E'))
                utc_obs[scan_idx] = value
            elif data_type == 'MJD_OBS':
                scan_idx = int(values[0])  # scan number
                _ = int(values[1])  # Unused
                _ = int(values[2])  # Unused
                _ = int(values[3])  # Unused
                value = int(values[4])
                mjd_obs[scan_idx] = value
            elif data_type == 'SOU_IND':
                # Mapping from scan_idx to source_idx
                scan_idx = int(values[0])  # scan number
                source_idx = int(values[4])
                scan_to_source[scan_idx] = source_idx
            elif data_type == 'SCAN_DUR':
                scan_idx = int(values[0])
                band_idx = int(values[2])
                dur_idx = float(values[4].replace('D','E'))
                dur_obs[(scan_idx, band_idx)] = dur_idx

    # Build source positions
    for idx, coords in sources.items():
        ra = coords.get(1)
        dec = coords.get(2)
        if ra is not None and dec is not None:
            # Convert from radians to degrees using np.degrees()
            ra_deg = np.degrees(ra)
            dec_deg = np.degrees(dec)
            source_name = source_indices.get(idx, f"Source_{idx}")
            source_positions[source_name] = (ra_deg, dec_deg)

    # Build antenna_positions_list in the same order as antenna_names
    # Create a reverse mapping from antenna names to indices
    antenna_names_to_indices = {name: idx for idx, name in antenna_indices.items()}

    for antenna_name in antenna_names:
        idx = antenna_names_to_indices.get(antenna_name)
        if idx is None:
            raise ValueError(f"Antenna '{antenna_name}' not found in the data.")
        coords = antenna_positions.get(idx)
        if not coords:
            raise ValueError(f"Coordinates for antenna '{antenna_name}' not found.")
        x = coords.get(1)
        y = coords.get(2)
        z = coords.get(3)
        if x is None or y is None or z is None:
            raise ValueError(f"Incomplete coordinates for antenna '{antenna_name}'.")
        antenna_positions_list.append([x, y, z])

    # Find the band index corresponding to the desired band
    desired_band_idx = None
    for idx, band in band_names.items():
        if band == desired_band:
            desired_band_idx = idx
            break
    if desired_band_idx is None:
        raise ValueError(f"Desired band '{desired_band}' not found in the data.")

    # Process observations
    for obs_num, entries in obs_tab_entries.items():
        # Extract scan_idx, sta1_idx, sta2_idx
        scan_idx = entries.get(1)  # Scan number associated with the observation
        sta1_idx = entries.get(2)
        sta2_idx = entries.get(3)

        if scan_idx is None or sta1_idx is None or sta2_idx is None:
            raise ValueError(f"Incomplete OBS_TAB entry for observation {obs_num}.")

        site1 = antenna_indices.get(sta1_idx)
        site2 = antenna_indices.get(sta2_idx)

        if site1 not in antenna_names or site2 not in antenna_names:
            continue  # Skip observations not involving desired antennas

        baseline_key = f'{site1}_{site2}'

        # Fetch the quality flag
        quality_flag = qualcode.get((obs_num, desired_band_idx))
        if quality_flag is None:
            raise ValueError(f"Quality flag missing for observation {obs_num}, band {desired_band_idx}.")
        try:
            quality_flag_int = int(quality_flag)
        except ValueError:
            quality_flag_int = 0  # Non-integer quality flags are treated as 0

        if quality_flag_int < 5:
            continue  # Skip observations with low quality

        # Fetch observation data
        delay = gr_delay.get((obs_num, desired_band_idx))
        delay_err = gr_delerr.get((obs_num, desired_band_idx))
        fringe_phase = totphase.get((obs_num, desired_band_idx))
        phase_err = phdelerr.get((obs_num, desired_band_idx))  # Added phase error
        frequency = ref_frq.get((obs_num, desired_band_idx))
        dur = dur_obs.get((obs_num, desired_band_idx))

        if delay is None:
            raise ValueError(f"Group delay missing for observation {obs_num}, band {desired_band_idx}.")
        if delay_err is None:
            raise ValueError(f"Group delay error missing for observation {obs_num}, band {desired_band_idx}.")
        if fringe_phase is None:
            raise ValueError(f"Total fringe phase missing for observation {obs_num}, band {desired_band_idx}.")
        if phase_err is None:
            phase_err = delay_err
            #raise ValueError(f"Phase delay error missing for observation {obs_num}, band {desired_band_idx}.")
        if frequency is None:
            raise ValueError(f"Reference frequency missing for observation {obs_num}, band {desired_band_idx}.")

        # Build datetime using scan_idx
        mjd = mjd_obs.get(scan_idx)
        utc_seconds = utc_obs.get(scan_idx)
        if mjd is None:
            raise ValueError(f"MJD_OBS missing for scan {scan_idx}.")
        if utc_seconds is None:
            raise ValueError(f"UTC_OBS missing for scan {scan_idx}.")
        if utc_mtai is None:
            raise ValueError(f"UTC_MTAI missing in the data.")
        tai_seconds = utc_seconds - utc_mtai
        # Convert MJD and seconds to datetime
        jd = mjd + (tai_seconds / 86400.0)
        obs_datetime = datetime.datetime.utcfromtimestamp((jd - 40587) * 86400.0)

        # Fetch the source index for the scan using scan_to_source
        source_idx = scan_to_source.get(scan_idx)
        if source_idx is None:
            raise ValueError(f"Source index missing for scan {scan_idx}.")
        source_name = source_indices.get(source_idx, f"Source_{source_idx}")

        if obs_datetime.hour< 8:
            # TEMP -- JAXA hack
            continue

        observation = {
            'datetime': obs_datetime,
            'source': source_name,
            'delay': delay,
            'delay_err': delay_err,
            'fringe_phase': fringe_phase,
            'phase_err': phase_err,  # Include phase error
            'frequency': frequency,
            'duration': dur,
            'obs_num': obs_num,
            'baseline_key': baseline_key
        }

        baseline_observations[baseline_key].append(observation)

    return antenna_positions_list, source_positions, baseline_observations


def write_vda_phase(baseline_observations, input_vda, output_vda, int_amb, baseline_handles, baselines, antennas):
    """ Write VDA phase delay ambiguity solution into new output file """
    scan_index = {
        obs['obs_num']: obs
        for baseline in baseline_observations.values()
        for obs in baseline
    }
    # Temporary storage for data
    with open(input_vda, 'r') as vda_file_in:
        with open(output_vda, 'w') as vda_file_out:
            for line in vda_file_in:
                # Skip empty lines or lines without DATA.1 prefix
                if not line.strip() or not (line.startswith('DATA.1') or line.startswith('DATA.2') or line.startswith('DATA.4')):
                    vda_file_out.write(line)
                    continue

                parts = line.strip().split()
                if len(parts) < 7:
                    vda_file_out.write(line)
                    continue  # Not enough parts to process
                data_type = parts[1]
                values = parts[2:]

                # Process different data types
                if data_type == 'N_PHAMB':
                    scan_idx = int(values[0])  # scan number
                    _ = int(values[1])  # Unused
                    _ = int(values[2])  # Unused
                    _ = int(values[3])  # Unused
                    value = int(values[4])
                    try: obs = scan_index[scan_idx]
                    except:
                        # obs not in baseline solution
                        continue
                    dt_obs = np.datetime64(obs['datetime']) + np.timedelta64(TAI2GPS, 's')
                    baseline_key = obs['baseline_key']
                    n_amb = 0
                    for jdx, baseline in enumerate(baselines):
                        antenna1 = antennas[baseline[0]]
                        antenna2 = antennas[baseline[1]]
                        if antenna1 in baseline_key and antenna2 in baseline_key:
                            baseline_handle = baseline_handles[jdx]
                            break
                        n_amb += baseline_handles[jdx].n_amb_state
                    idx_pt = np.argwhere(baseline_handle.datetime_array[baseline_handle.phase_data_idxs] == dt_obs)
                    if len(idx_pt)>0:
                        # get the slip slice to which idx_pt belongs--this gives us which ambiguity to apply
                        slip_idx = next(i for i, arr in enumerate(baseline_handle.slip_slices_arr) if idx_pt in arr)
                        full_amb = int_amb[idx_pt + slip_idx]
                    else:
                        full_amb = 0 # zero out other point so they can be identified as outliers immediately
                    breakpoint()
                    # now put the genie back in the bottle -- write the modified line
                    pos = 0
                    for field in parts[:6]:  # everything before value (DATA.1, N_PHAMB, values[0-3])
                        pos = line.index(field, pos) + len(field)
                    val_start = line.index(str(value), pos)
                    
                    # Overwrite and trim to original length
                    print(f'wrote line for idx {scan_idx}')
                    full_amb_str = str(full_amb)
                    eol = '\n' if line.endswith('\n') else ''
                    line_body = line.rstrip('\n')
                    new_line = (line_body[:val_start] + full_amb_str + line_body[val_start + len(str(value)):])[:len(line_body)] + eol
                    vda_file_out.write(new_line)
                else:
                    vda_file_out.write(line)

    return 


def import_data_vlbi_vda(vda_file, antennas, baselines, band):
    """
    Read key file and fringe file, output good data
    """
    rxpos_all, src_dict, baseline_observations = parse_vda_file(vda_file, band, antennas)

    # set up array for epochs observed by each antenna
    dt_ant = {}
    for ant in antennas:
        dt_ant[ant] = np.array([],dtype=np.datetime64)

    datetime_array_srt = np.array([],dtype=np.datetime64)
    src_time_dict = {}
    duration_dict = {}
    baseline_handles = []
    for baseline in baselines:
        antenna1 = antennas[baseline[0]]
        antenna2 = antennas[baseline[1]]
        baseline_use = None
        for b_key in baseline_observations.keys():
            if antenna1 in b_key and antenna2 in b_key:
                baseline_use = b_key
        if baseline_use is None:
            raise ValueError('baseline ' + antenna1 +' — ' + antenna2 +' not found in fringe data')

        data = baseline_observations[baseline_use]
        dt_tai = []
        sources = []
        group_delays = []
        grdel_err = []
        phdel_err = []
        phase_delays = []
        frequencies = []
        durations = []
        for obs in data:
            dt_tai.append(obs['datetime'])
            sources.append(obs['source'])
            group_delays.append(obs['delay'])
            grdel_err.append(obs['delay_err'])
            phase_delays.append(obs['fringe_phase'])
            phdel_err.append(obs['phase_err'])
            frequencies.append(obs['frequency'])
            durations.append(obs['duration'])
        dt_tai = np.array(dt_tai)
        sources = np.array(sources)
        group_delays = np.array(group_delays)
        grdel_err = np.array(grdel_err)
        phdel_err = np.array(phdel_err)
        dur_arr = np.array(durations)
        
        phase_delays = np.array(phase_delays)
        frq = np.array(frequencies)

        # correct datetime from scan reference time (UTC) to GPS time
        dt_gps = np.zeros_like(dt_tai)
        for idx in range(len(dt_tai)):
            dt_gps[idx] = dt_tai[idx] + datetime.timedelta(seconds=TAI2GPS)
 
        datetime_array = np.array(dt_gps, dtype='datetime64[ns]')
        
        for idx, source in enumerate(sources):
            src_time_dict[datetime_array[idx]] = source

        # check if baselines match baselines array
        if baseline_use.split('_')[0] != antennas[baseline[0]]:
            # need to change the sign of the measurements
            group_delays = -group_delays
            phase_delays = -phase_delays
            
        group_delays = group_delays*const.c # group delay in meters
        grdel_err = grdel_err*const.c
        # get correlation between ambiguities by adding nearest integer from group delay -- this will cause linked measurements hopefully
        wavelength = const.c/frq
        amb_int = np.rint(group_delays/wavelength)

        phase_delays = (phase_delays/(2*np.pi) + amb_int) * wavelength
        phdel_err = phdel_err*const.c
        # use frq[0] assuming frq is all the same frequency; could be broken in the future
        baseline_handle = BaselineInfo(datetime_array, frq[0])
        baseline_handle.prepare_vlbi(group_delays, phase_delays, grdel_err, phdel_err)
        #baseline_handle.prepare_vlbi(group_delays, phase_delays, group_delays_dual, phase_delays_dual)
        baseline_handles.append(baseline_handle)

        for idx, time in enumerate(np.array(dt_gps, dtype='datetime64[ns]')):
            duration_dict[time] = dur_arr[idx]

        # add observed epochs to datetimes for each antenna
        dt_ant[antenna1] = np.union1d(dt_ant[antenna1], datetime_array)
        dt_ant[antenna2] = np.union1d(dt_ant[antenna2], datetime_array)
        datetime_array_srt = np.union1d(datetime_array_srt, datetime_array)

    datetime_array_srt = np.array(datetime_array_srt, dtype='datetime64[ns]')
    point_ra_dec_full = []
    source_array_full = []
    for time in datetime_array_srt: 
        source = src_time_dict[time]
        ra, dec = src_dict[source.strip()]
        source_array_full.append(source)
        point_ra_dec_full.append((ra,dec))
    print('Number of scans: ' + str(len(datetime_array_srt)))
    return baseline_handles, datetime_array_srt, source_array_full, point_ra_dec_full, dt_ant, rxpos_all, duration_dict, baseline_observations

def import_data_vlbi_vgosdb(db_loc, antennas, baselines, band):
    """
    Read key file and fringe file, output good data
    """
    # Create full path to Observables subdirectory
    observables_dir = os.path.join(db_loc, 'Observables')
    
    # Initialize dictionary to hold xarray datasets
    xarray_data = {}

    # Iterate through all files in the Observables subdirectory
    for file_name in os.listdir(observables_dir):
        if file_name.endswith(".nc"):  # Check for netCDF files
            # Create full path to the file
            file_path = os.path.join(observables_dir, file_name)
            
            # Load the netCDF file as an xarray dataset
            key_name = os.path.splitext(file_name)[0]
            xarray_data[key_name] = xr.open_dataset(file_path)
    
    cable_cal = {}
    time_cal = {}
    met_data = {}
    duration_dict = {}
    for antenna in antennas:
        antenna_dir = os.path.join(db_loc, antenna)
        cable_cal_path = os.path.join(antenna_dir, 'Cal-Cable.nc')
        utc_times = os.path.join(antenna_dir, 'TimeUTC.nc')
        time_xr = xr.open_dataset(utc_times)
        time_cal[antenna] = time_xr

        try:
            cable_cal_antenna = xr.open_dataset(cable_cal_path)
            cable_cal[antenna] = cable_cal_antenna['Cal-Cable'].values
        except:
            cable_cal[antenna] = None # no cable cal
        try:
            met_data_antenna = xr.open_dataset(os.path.join(antenna_dir, 'Met.nc'))
            met_data[antenna] = met_data_antenna
        except:
            met_data[antenna] = None

    apriori_sta = os.path.join(db_loc, 'Apriori/Station.nc')
    station_data = xr.open_dataset(apriori_sta)
    sta_names = station_data.AprioriStationList.values.tolist()
    sta_names = [byte_array.decode('utf-8').strip() for byte_array in sta_names]
    
    apriori_sou = os.path.join(db_loc, 'Apriori/Source.nc')
    source_data = xr.open_dataset(apriori_sou)
    source_name_xr = source_data.AprioriSourceList.values
    sources_unique = []
    for source in source_name_xr:
        sources_unique.append(source.decode('utf-8').strip())
    src_dict = {}
    ra_degrees = np.rad2deg(source_data.AprioriSource2000RaDec.values[:,0])
    dec_degrees = np.rad2deg(source_data.AprioriSource2000RaDec.values[:,1])
    for idx, source in enumerate(sources_unique):
        src_dict[source] = (ra_degrees[idx], dec_degrees[idx])

    rxpos_all = []
    for antenna in antennas:
        idx_ant = sta_names.index(antenna)
        rxpos_all.append(station_data.AprioriStationXYZ.values[idx_ant].tolist())
    
    baselines_xr = xarray_data['Baseline'].Baseline.values
    baselines_list = []
    for baseline in baselines_xr:
        baselines_list.append(b''.join(baseline).decode('utf-8').strip())

    filename_phase = 'Phase_b' + band
    filename_freq = 'RefFreq_b' + band
    filename_quality = 'QualityCode_b' + band
    filename_corrinfo= 'CorrInfo-difx_b' + band
    filename_channelinfo= 'ChannelInfo_b' + band
    phase = xarray_data[filename_phase]
    freq = xarray_data[filename_freq]
    quality = xarray_data[filename_quality].QualityCode.values
    quality = list(map(int, quality.tobytes().decode('ascii')))
    polarizations = xarray_data[filename_channelinfo].Polarization.values
    try:
        phase_flag = phase.PhaseDataFlag.values
    except:
        # no phase flags in dataset
        phase_flag = np.zeros(len(quality))
    
    corr_file = xarray_data[filename_corrinfo]
    try:
        corr_startsec = corr_file.StartSec.values
    except: 
        # some vgosdb have it in this format
        corr_startsec = corr_file.STARTSEC.values
    corr_stopsec = corr_file.StopSec.values
    duration = corr_stopsec - corr_startsec 

    time_utc = xarray_data['TimeUTC']

    filename_grdel_obs = 'GroupDelay_b' + band
    filename_grdel_full = 'GroupDelayFull_b' + band
    filename_grdel_full_ivs = 'GroupDelayFull_iIVS_b' + band
    if os.path.exists(os.path.join(db_loc, 'ObsEdit/'+filename_grdel_full+'.nc')):
        grdel = xr.open_dataset(os.path.join(db_loc,'ObsEdit/'+filename_grdel_full+'.nc')).GroupDelayFull.values
    elif os.path.exists(os.path.join(db_loc, 'ObsEdit/'+filename_grdel_full_ivs+'.nc')):
        grdel = xr.open_dataset(os.path.join(db_loc,'ObsEdit/'+filename_grdel_full_ivs+'.nc')).GroupDelayFull.values
    else:
        grdel = xarray_data[filename_grdel_obs].GroupDelay.values

    grdel_errors = xarray_data[filename_grdel_obs].GroupDelaySig.values

    #snr_filename = 'SNR_b' + band + '.nc'
    #snr_path = os.path.join(db_loc, 'Observables/'+snr_filename)
    #snr_xr = xr.open_dataset(snr_path)
    #snr_vals = snr_xr.SNR.values

    sources_xr = xarray_data['Source'].Source.values
    sources = []
    for source in sources_xr:
        sources.append(source.decode('utf-8').strip())

    leap_second = os.path.join(db_loc, 'Session/LeapSecond.nc')
    ls_data = xr.open_dataset(leap_second)
    utc2gps = ls_data.LeapSecond.values[0] - 19
    
    # set up array for epochs observed by each antenna
    dt_ant = {}
    for ant in antennas:
        dt_ant[ant] = np.array([],dtype=np.datetime64)

    datetime_array_srt = np.array([],dtype=np.datetime64)
    src_time_dict = {}
    duration_dict = {}
    baseline_handles = []
    for baseline in baselines:
        antenna1 = antennas[baseline[0]]
        antenna2 = antennas[baseline[1]]
        idxs_use = []
        for idx, baseline_obs in enumerate(baselines_list):
            if antenna1 in baseline_obs and antenna2 in baseline_obs and \
                    quality[idx] == 9 and phase_flag[idx] == 0:
                baseline_use = baseline_obs
                idxs_use.append(idx)
            elif antenna1 in baseline_obs and antenna2 in baseline_obs and \
                    (antenna1 == 'WETTZELL' or antenna1 == 'WETTZ13N') and \
                    (antenna2 == 'WETTZELL' or antenna2 == 'WETTZ13N') and \
                    quality[idx] >=7:
                # wettzell-wettz13n has obs labeled low fringe quality for some reason
                baseline_use = baseline_obs
                idxs_use.append(idx)
            elif antenna1 in baseline_obs and antenna2 in baseline_obs and \
                    (antenna1 == 'WARK12M' or antenna1 == 'WARK30M') and \
                    (antenna2 == 'WARK12M' or antenna2 == 'WARK30M') and \
                    quality[idx] >=6:
                # wark obs labeled low fringe quality
                baseline_use = baseline_obs
                idxs_use.append(idx)
        if len(idxs_use) == 0:
            raise ValueError('baseline ' + antenna1 +' — ' + antenna2 +' not found in fringe data')
        ymdhm_values = time_utc.YMDHM.values  # Shape: (NumObs, 5)
        second_values = time_utc.Second.values  # Shape: (NumObs,)
        if ymdhm_values[0,0] < 2000:
            date_corr = 2000
        else:
            date_corr = 0
        datetime_list = [
            datetime.datetime(
                year=ymdhm[0]+date_corr,  
                month=ymdhm[1],
                day=ymdhm[2],
                hour=ymdhm[3],
                minute=ymdhm[4],
                second=int(second)
            )
            for ymdhm, second in zip(ymdhm_values[idxs_use,:], second_values[idxs_use])
        ]

        dt_utc = np.array(datetime_list)
        # take out bad sections of data
        dt_utc_sec = (dt_utc-dt_utc[0])/datetime.timedelta(seconds=1)
        diff = np.diff(dt_utc_sec)
        #if np.any(diff > 3600):
        #    idx_break = np.argwhere(diff>3600)[0][-1]
        #    idxs_use = idxs_use[idx_break+1:]
        #    dt_utc = dt_utc[idx_break+1:]
        #    datetime_list = datetime_list[idx_break+1:]
        #else:
        #    idx_break = None

        frq = np.array(freq.RefFreq.values)*1e6 # MHz --> Hz
        sources = np.array(sources)[idxs_use]
        group_delays = np.array(grdel[idxs_use])
        grdel_errors_bl = np.array(grdel_errors[idxs_use])
        phase_delays = np.array(phase.Phase.values[idxs_use])
        phase_errors_bl = np.array(phase.PhaseSig.values[idxs_use])
        duration_arr = np.array(duration)[idxs_use]

        # correct datetime from scan reference time (UTC) to GPS time
        dt_gps = np.zeros_like(dt_utc)
        for idx in range(len(dt_utc)):
            dt_gps[idx] = dt_utc[idx] + datetime.timedelta(seconds=float(utc2gps))


        datetime_array = np.array(dt_gps, dtype='datetime64[ns]')
        
        for idx, source in enumerate(sources):
            src_time_dict[datetime_array[idx]] = source
            duration_dict[datetime_array[idx]] = duration_arr[idx]

        # check if baselines match baselines array
        if baseline_use.index(antennas[baseline[0]])>baseline_use.index(antennas[baseline[1]]):
            # need to change the sign of the measurements
            group_delays = -group_delays
            phase_delays = -phase_delays

        group_delays = group_delays*const.c # group delay in meters
        grdel_errors_bl = grdel_errors_bl*const.c
        # get correlation between ambiguities by adding nearest integer from group delay -- this will cause linked measurements hopefully
        wavelength = const.c/frq
        amb_int = np.rint(group_delays/wavelength)
        phase_delays = (phase_delays/(2*np.pi) + amb_int) * wavelength
        phase_errors_bl = phase_errors_bl/(2*np.pi)*wavelength

        #phase_errors_bl = grdel_errors_bl

        # get antenna-specific data
        times_ant1 = time_cal[antenna1]
        ymdhm_vals_ant1 = times_ant1.YMDHM.values  # Shape: (NumObs, 5)
        second_vals_ant1 = times_ant1.Second.values  # Shape: (NumObs,)
        dt_list_ant1 = [
            datetime.datetime(
                year=ymdhm[0],  
                month=ymdhm[1],
                day=ymdhm[2],
                hour=ymdhm[3],
                minute=ymdhm[4],
                second=int(second)
            )
            for ymdhm, second in zip(ymdhm_vals_ant1, second_vals_ant1)
        ]
        times_ant2 = time_cal[antenna2]
        ymdhm_vals_ant2 = times_ant2.YMDHM.values  # Shape: (NumObs, 5)
        second_vals_ant2 = times_ant2.Second.values  # Shape: (NumObs,)
        dt_list_ant2 = [
            datetime.datetime(
                year=ymdhm[0],  
                month=ymdhm[1],
                day=ymdhm[2],
                hour=ymdhm[3],
                minute=ymdhm[4],
                second=int(second)
            )
            for ymdhm, second in zip(ymdhm_vals_ant2, second_vals_ant2)
        ]
        _, idxs_ant1, _ = np.intersect1d(dt_list_ant1, datetime_list, return_indices=True)
        _, idxs_ant2, _ = np.intersect1d(dt_list_ant2, datetime_list, return_indices=True)

        cc_1 = cable_cal[antenna1]
        cc_2 = cable_cal[antenna2]
        if cc_1 is not None:
            group_delays = group_delays - cc_1[idxs_ant1]*const.c
            phase_delays = phase_delays - cc_1[idxs_ant1]*const.c
        if cc_2 is not None:
            group_delays = group_delays + cc_2[idxs_ant2]*const.c
            phase_delays = phase_delays + cc_2[idxs_ant2]*const.c

        # save meteorological data
        met_1 = met_data[antenna1]
        met_2 = met_data[antenna2]
        if met_1 is not None:
            met_1_P = met_1.AtmPres.values[idxs_ant1]
            met_1_H = met_1.RelHum.values[idxs_ant1]
            met_1_T = met_1.TempC.values[idxs_ant1]
        else:
            met_2_P = None
            met_2_H = None
            met_2_T = None
        if met_2 is not None:
            met_2_P = met_2.AtmPres.values[idxs_ant2]
            met_2_H = met_2.RelHum.values[idxs_ant2]
            met_2_T = met_2.TempC.values[idxs_ant2]
        else:
            met_2_P = None
            met_2_H = None
            met_2_T = None

        # use frq[0] assuming frq is all the same frequency; could be broken in the future
        baseline_handle = BaselineInfo(datetime_array, frq[0])
        baseline_handle.prepare_vlbi(group_delays, phase_delays, grdel_errors_bl, phase_errors_bl)
        #baseline_handle.prepare_vlbi(group_delays, phase_delays, grdel_errors_bl, phase_errors_bl, group_delays_dual, phase_delays_dual)
        baseline_handle.save_weather(met_1_P, met_1_H, met_1_T, met_2_P, met_2_H, met_2_T)
        print('using vgosDB meteorological data -- command line inputs will not be used')
        baseline_handles.append(baseline_handle)

        # add observed epochs to datetimes for each antenna
        dt_ant[antenna1] = np.union1d(dt_ant[antenna1], datetime_array)
        dt_ant[antenna2] = np.union1d(dt_ant[antenna2], datetime_array)
        datetime_array_srt = np.union1d(datetime_array_srt, datetime_array)

    datetime_array_srt = np.array(datetime_array_srt, dtype='datetime64[ns]')
    point_ra_dec_full = []
    source_array_full = []
    for time in datetime_array_srt: 
        source = src_time_dict[time]
        ra, dec = src_dict[source.strip()]
        source_array_full.append(source)
        point_ra_dec_full.append((ra,dec))
    print('Number of scans: ' + str(len(datetime_array_srt)))
    return baseline_handles, datetime_array_srt, source_array_full, point_ra_dec_full, dt_ant, rxpos_all, duration_arr

def gen_key(rinex_files, full_data, start_date, end_date, eph_sats, iono_free, iono_freq):
    """ Generate a key file choosing a random satellite for each 60 second scan from the commonly observed SVs """
    duration_scan = 30 # seconds
    datetime_array = []
    source_array = [] 
    exp_length = end_date-start_date

    SEED=25 # change this to change the random selection
    rng = np.random.default_rng(seed=SEED)

    # find the appropriate RINEX data for each requested source
    time_idx = 0
    time_obs = start_date
    while time_obs <= end_date:
        sv_arr = []
        for ant_idx, rinex_file in enumerate(rinex_files):
            rinex_obs = full_data[rinex_file]
            try:
                obs_time = rinex_obs.sel(time=time_obs).dropna(dim='sv',how='all')
            except:
                continue
            
            # ensure L1 and L2 data present
            #obs_L1 = obs_time['C1'].dropna(dim='sv',how='all')
            pref_order = ['C1L', 'C1S', 'C1X', 'C1P', 'C1']
            obs_L1 = None
            for var in pref_order:
                if var in obs_time:                    # skip if not present
                    da = obs_time[var].dropna(dim='sv',how='all') # an xarray.DataArray
                    obs_L1 = da if obs_L1 is None else obs_L1.combine_first(da)
            
            if iono_free is True:
                if iono_freq == 'L2':
                    obs_dual = obs_time['P2'].dropna(dim='sv',how='all')
                elif iono_freq == 'L5':
                    obs_dual = obs_time['C5'].dropna(dim='sv',how='all')
                svs_obs = np.intersect1d(obs_L1.sv, obs_dual.sv)
            else:
                svs_obs = obs_L1.sv
            
            if len(sv_arr) == 0: 
                sv_arr = np.array(svs_obs)
            elif len(svs_obs.sv.values)>0:
                sv_arr = np.intersect1d(sv_arr, np.array(svs_obs))
        if len(sv_arr)>0:
            if len(sv_arr)>1:
                rand_int = rng.integers(0,len(sv_arr)-1)
            else:
                rand_int = 0
            source_array.append(sv_arr[rand_int])
            datetime_array.append(time_obs)
        
        #time_obs += duration_scan*np.timedelta64(1, 's')
        time_obs += np.timedelta64(1, 's')
        time_idx+=1   

    return datetime_array, source_array 

def thin_data(antenna_names, rinex_files, full_data, datetime_array, source_array):
    """
    Thin the RINEX data to 1 satellite at a time
    """
    thinned_data = {}
    for ant_idx, rinex_file in enumerate(rinex_files):
        rinex_obs = full_data[rinex_file]
        antenna_name = antenna_names[ant_idx]
        data_antenna = [] 

        for src_idx, time_beg in enumerate(datetime_array):
            src_name = source_array[src_idx][0:3] # Rinex source is 1 letter, 2 numbers
            try:
                obs_sv = rinex_obs.sel(sv=src_name).dropna(dim='time',how='all')
            except: continue
            if np.datetime64(time_beg) in obs_sv.time.values:
                obs_sv = obs_sv.sel(time=time_beg)
            else:
                continue
            if len(data_antenna)==0:
                data_antenna = obs_sv
            else:
                data_antenna = xr.concat((data_antenna, obs_sv), dim='time')
        thinned_data[antenna_name] = data_antenna

    return thinned_data

def average_data(antenna_names, thinned_data, dt_key, duration_key, source_key, point_key):
    """
    Average 1 Hz RINEX data to 1 point per scan
    """
    UTC2GPS_td = to_timedelta(UTC2GPS, unit='s')
    datetime_array = []
    source_array = []
    point_ra_dec_reduced = []
    obs_antennas = {}
    for antenna_name in antenna_names:
        obs_antennas[antenna_name] = []

    for time_idx, time_obs in enumerate(dt_key):
        beg_time = time_obs + UTC2GPS_td 
        end_time = time_obs + UTC2GPS_td + to_timedelta(duration_key[time_idx], unit='s')
        sat = source_key[time_idx][0:3] # Rinex source is 1 letter, 2 numbers

        # find common epochs in this scan
        ant_epoch = []
        times_common = []
        for ant_idx, antenna_name in enumerate(antenna_names):
            rinex_obs = thinned_data[antenna_name]
            obs_time = rinex_obs.sel(time=slice(beg_time,end_time))
            if len(obs_time.time.values) > 0:
                ant_epoch.append(ant_idx)
                if len(times_common) > 0:
                    times_common = np.intersect1d(times_common, obs_time.time.values)
                else:
                    times_common = obs_time.time.values
        # find the epoch to which we will interpolate
        if len(times_common) > 0:
            middle_idx = round(len(times_common)/2)
            middle_time = times_common[middle_idx]
            middle_time_pydt = to_datetime(middle_time).to_pydatetime()
            datetime_array.append(middle_time_pydt)
            source_array.append(sat)
            if len(point_key)>0:
                point_ra_dec_reduced.append(point_key[time_idx])
        
            for ant_idx, antenna_name in enumerate(antenna_names):
                if ant_idx in ant_epoch:
                    rinex_obs = thinned_data[antenna_name]
                    obs_time = rinex_obs.sel(time=slice(beg_time,end_time))
                    """
                    time0 = obs_time.time.min()
                    time_numeric = (obs_time.time - time0).values.astype('timedelta64[s]').astype(float)
                    middle_time_numeric = (middle_time - time0).values.astype('timedelta64[s]').astype(float)

                    obs_avg = obs_time.mean(dim='time', skipna=True)
                    time_as_numbers = obs_time.time.values.astype('int64').mean()
                    averaged_time_check = to_datetime(time_as_numbers)
                    averaged_time = to_datetime(time_as_numbers).round('us')
                    obs_avg = obs_avg.assign_coords(time=sampled_time)
                    obs_avg = obs_avg.assign_coords(sv=sv_value)
                    # Note -- here we're losing nanosecond precision in the averaged epoch. 
                    # This shouldn't matter, but something to keep in mind
                    if ant_idx == 0:
                        averaged_time_datetime = averaged_time.to_pydatetime()
                        datetime_array.append(averaged_time_datetime)
                        source_array.append(sat)
                    """
                    smoothed_values = {}
                    for var_name, data_array in obs_time.data_vars.items():
                        # smooth data and get desired epoch
                        valid_mask = ~np.isnan(data_array.values)
                        time_valid = obs_time.time.values[valid_mask]
                        middle_idx_ant = np.argwhere(time_valid==middle_time)
                        data_valid = data_array.values[valid_mask]
                        if len(time_valid) > 1:
                            """
                            # Create interpolation function
                            if len(time_valid)>3:
                                order = 3
                            else:
                                order = 2
                            interp_func = make_interp_spline(time_valid, data_valid, k=order)

                            # Interpolate at middle_time_numeric
                            interpolated_value = interp_func(middle_time_numeric)
                            # Store the interpolated value
                            interpolated_values[var_name] = interpolated_value
                            """
                            num_points = len(data_valid)  
                            
                            # Define a base window length for smaller datasets
                            if num_points >= 5:
                                window_length = 5
                                polyorder=2
                            elif num_points >=3:
                                window_length = 3
                                polyorder=1
                            else:
                                window_length = 1
                                polyorder=0
                            
                            # Adjust the window length for larger datasets, ensuring it's odd
                            if num_points > 30:
                                # Example: set the window length to a quarter of the dataset size, rounded up to the nearest odd number
                                window_length = int(num_points / 2)
                                window_length += 1 - window_length % 4  # Ensure it's odd
                            
                            # Ensure the window length does not exceed a maximum value, e.g., 31
                            window_length = min(window_length, 31)
                            
                            # Apply the Savitzky-Golay filter
                            smoothed_data = savgol_filter(data_valid, window_length, polyorder=polyorder)
                            smoothed_values[var_name] = smoothed_data[middle_idx_ant][0][0]

                            if ant_idx == 0 and var_name == 'C1':
                                plt.figure()
                                plt.plot(time_valid,data_valid-smoothed_data,marker='x',linestyle=None)
                                plt.xlabel('time (sec)')
                                plt.ylabel('PR-PR_smooth (m)')
                                plt.savefig('SG_test.png')
                                plt.close()
                        else:
                            # Handle cases with insufficient data by assigning NaN
                            smoothed_values[var_name] = np.nan

                    sv_value = obs_time.sv.values[0]
                    # Create the new dataset with interpolated values at middle_time
                    obs_avg = xr.Dataset(
                                 {var_name: (["time"], [value]) for var_name, value in smoothed_values.items()},
                                 coords={"time": [middle_time]})
                    obs_avg = obs_avg.assign_coords(sv=sv_value)
                    obs_avg.attrs = obs_time.attrs

                    if len(obs_antennas[antenna_name])==0:
                        obs_antennas[antenna_name] = obs_avg
                    else:
                        obs_antennas[antenna_name] = xr.concat((obs_antennas[antenna_name], obs_avg), dim='time')

    thinned_data = obs_antennas

    return thinned_data, datetime_array, source_array, point_ra_dec_reduced

def vlbi_transform_data(store_handle, antenna_handles, igs_data=False, dt_vlbi=None):
    """
    Use VLBI-like processing to transform 1 Hz RINEX data to 1 point per scan
    """
    # get all observation times in the experiment
    for idx, antenna_handle in enumerate(antenna_handles):
        if idx == 0:
            datetime_array = antenna_handle.times_gps
        else:
            datetime_array = np.union1d(datetime_array, antenna_handle.times_gps)

    source_array = np.array([store_handle.source_time_dict[time] for time in datetime_array])

    # find array of common reference times
    src_last = source_array[0] 
    datetime_ref = []
    source_ref = []
    time_src = []
    scan_beg = []
    scan_end = []
    scan_max = 20 
    for idx, src in enumerate(source_array):
        if idx > 0:
            time_diff = (datetime_array[idx]-datetime_array[idx-1])/np.timedelta64(1,'s')
        else:
            time_diff = 0

        if src == src_last and idx != len(source_array)-1 and time_diff < scan_max: 
            time_src.append(datetime_array[idx])
        else:
            # we just changed sources, re-initialize arrays and find reference for previous source
            if dt_vlbi is not None:

                lim_low = time_src[0]
                lim_high = time_src[-1]
                time_ref = None
                for time_vlbi in dt_vlbi:
                    if time_vlbi > lim_low and time_vlbi < lim_high:
                        time_ref = time_vlbi
                if time_ref is None:
                    # skip this scan -- no VLBI data
                    src_last = src
                    time_src = [datetime_array[idx]]
                    continue
            else:
                time_ref = time_src[len(time_src)//2]

            if (time_ref-time_src[0])/np.timedelta64(1,'s') < scan_max:
                scan_beg.append(time_src[0])
            else:
                scan_beg.append(time_ref-scan_max*np.timedelta64(1,'s'))

            if (time_src[-1]-time_ref)/np.timedelta64(1,'s') < scan_max:
                scan_end.append(time_src[-1])
            else:
                scan_end.append(time_ref+scan_max*np.timedelta64(1,'s'))
            
            if igs_data is True:
                # round to 30-s epoch
                time_ref = (time_ref + np.timedelta64(15, 's')).astype('datetime64[30s]').astype('datetime64[ns]')

            datetime_ref.append(time_ref)
            source_ref.append(src_last)
            time_src = [datetime_array[idx]]
        src_last = src

    f1 = 1575.42e6
    wavelength = const.c/f1
    plot = False
    # run through antenna handles and reduce the data
    for antenna_handle in antenna_handles:
        obs_data = antenna_handle.antenna_data
        obs_full = []
        for idx, time in enumerate(datetime_ref):
            src = source_ref[idx]
            obs_time = antenna_handle.antenna_data.where(antenna_handle.antenna_data.sv == src, drop=True)
            if len(obs_time.pr_data.values)==0:
                #print('no data for source ' + src +' for antenna ' + antenna_handle.antenna_name)
                continue

            obs_time = obs_time.sel(time=slice(scan_beg[idx], scan_end[idx])).dropna(dim='time',how='all')

            if len(obs_time.pr_data.values)==0:
                #print('no PR for source ' + src +' for antenna ' + antenna_handle.antenna_name)
                continue

            ref_idx = np.argwhere(obs_time.time.values==time)
            if len(ref_idx) == 0:
                #print('no time for source ' + src +' for antenna ' + antenna_handle.antenna_name)
                continue
            ref_idx = ref_idx[0]
           
            if len(obs_time.time.values)>1:
                 # pr_data, pr_model; cp_data, cp_model
                 eps_range = (obs_time.pr_data - obs_time.pr_model).values
                 eps_phase = (obs_time.cp_data - obs_time.cp_model).values

                 sig_range, use_idxs_range = find_sigmas(eps_range-np.median(eps_range[~np.isnan(eps_range)]))
                 sig_phase, use_idxs_phase = find_sigmas(eps_phase-np.median(eps_phase[~np.isnan(eps_phase)]))
                 times_sec = (obs_time.time.values-obs_time.time.values[0])/np.timedelta64(1,'s')
                 
                 ref_time = times_sec[ref_idx][0]

                 # fix cycle slips
                 ref_phase = eps_phase[ref_idx]
                 #eps_phase += wavelength*np.round((ref_phase-eps_phase)/wavelength)
                 #if np.any(np.abs(np.diff(eps_phase))>=wavelength):
                 #    eps_phase = np.unwrap(eps_phase, period=wavelength)

                 # fit pseudorange
                 res_pr = linregress(times_sec[use_idxs_range], eps_range[use_idxs_range])
                 eps_pr_ref = res_pr.slope*ref_time + res_pr.intercept
                 pr_meas = obs_time.pr_model[ref_idx] + eps_pr_ref

                 # fit carrier phase
                 #res_cp = linregress(times_sec[use_idxs_phase], eps_phase[use_idxs_phase])
                 #eps_cp_ref = res_cp.slope*ref_time + res_cp.intercept
                 if len(times_sec[use_idxs_phase])>=5:
                     res_cp = make_smoothing_spline(times_sec[use_idxs_phase], eps_phase[use_idxs_phase])
                     eps_cp_ref = res_cp(ref_time)
                 elif len(times_sec[use_idxs_phase])==0:
                     continue
                 else:
                     res_cp = linregress(times_sec[use_idxs_phase], eps_phase[use_idxs_phase])
                     eps_cp_ref = res_cp.slope*ref_time + res_cp.intercept

                 cp_meas = obs_time.cp_model[ref_idx] + eps_cp_ref
                 #if cp_meas.sv.values[0][0] == 'E':
                 #    cp_meas += 0.0*wavelength
                 #elif cp_meas.sv.values[0][0] == 'G':
                 #    #cp_meas += 0.25*wavelength
                 #    cp_meas += 0.0*wavelength
                 #elif cp_meas.sv.values[0][0] == 'C':
                 #    cp_meas += 0.0*wavelength
                 
                 if plot is True:
                    fig, [ax1, ax2] = plt.subplots(2, dpi=300)
                    ax1.scatter(times_sec, eps_range, label='data', zorder=1)
                    ax1.plot(times_sec, res_pr.slope*times_sec + res_pr.intercept, label='linear fit', zorder=1)
                    ax1.scatter(ref_time, eps_pr_ref, label='new meas', marker='*', zorder=2)
                    ax2.scatter(times_sec, eps_phase, label='data', zorder=1)
                    ax2.plot(times_sec, res_cp(times_sec), label='linear fit', zorder=1)
                    #ax2.plot(times_sec, res_cp.slope*times_sec + res_cp.intercept, label='linear fit', zorder=1)
                    ax2.scatter(ref_time, eps_cp_ref, label='new meas', marker='*', zorder=2)
                    ax1.legend()
                    ax1.set_ylabel('Range-Model (m)')
                    ax2.set_ylabel('Phase-Model (m)')
                    ax1.set_title('Source ' + src)
                    ax2.set_xlabel('time (sec)')
                    fig.savefig('./VLBI_test_' +str(idx)+'_'+antenna_handle.antenna_name +'.png')
                    plt.close()
            else:
                # no fitting to do, only 1 value
                # this happens for IGS data
                pr_meas = obs_time.pr_data[ref_idx]
                cp_meas = obs_time.cp_data[ref_idx]

            # save data to dict
            smoothed_values = {}
            smoothed_values['pr_model'] = obs_time.pr_model[ref_idx]
            smoothed_values['pr_data'] = pr_meas
            smoothed_values['cp_model'] = obs_time.cp_model[ref_idx]
            smoothed_values['cp_data'] = cp_meas

            # Create the new dataset 
            data_dict = {var_name: (["time"], value.values) for var_name, value in smoothed_values.items()}
            obs_vlbi = xr.Dataset(
                         data_dict,
                         coords={"time": [time]})
            obs_vlbi = obs_vlbi.assign_coords(sv=src)
            obs_vlbi.attrs = obs_time.attrs

            if len(obs_full)==0:
                obs_full = obs_vlbi
            else:
                obs_full = xr.concat((obs_full, obs_vlbi), dim='time')

        # re-save antenna data as downsampled xarray
        _, idxs_old, idxs_new = np.intersect1d(antenna_handle.times_gps, obs_full.time.values, return_indices=True)
        if store_handle.stochastic_clock is False:
            antenna_handle.clock_samples = antenna_handle.clock_samples[idxs_old]
            antenna_handle.phase_clock_samples = antenna_handle.phase_clock_samples[idxs_old]
            antenna_handle.clock_times = obs_full.time.values
            antenna_handle.phase_clock_times = obs_full.time.values
        else:
            _, idxs_old_clock, idxs_new_clock = np.intersect1d(antenna_handle.clock_times, obs_full.time.values, return_indices=True)
            antenna_handle.clock_samples = antenna_handle.clock_samples[idxs_old_clock]
            antenna_handle.clock_times = antenna_handle.clock_times[idxs_old_clock]
            antenna_handle.phase_clock_samples = antenna_handle.phase_clock_samples[idxs_old_clock]
            antenna_handle.phase_clock_times = antenna_handle.clock_times

        if store_handle.stochastic_trop is False and store_handle.estimate_trop is True:
            antenna_handle.trop_samples = antenna_handle.trop_samples[idxs_old]
            antenna_handle.trop_times = obs_full.time.values
        elif store_handle.stochastic_trop is True:
            _, idxs_old_trop, idxs_new_trop = np.intersect1d(antenna_handle.trop_times, obs_full.time.values, return_indices=True)
            antenna_handle.trop_samples = antenna_handle.trop_samples[idxs_old_trop]
            antenna_handle.trop_times = antenna_handle.trop_times[idxs_old_trop]

        antenna_handle.pr_errors = antenna_handle.pr_errors[idxs_old]
        antenna_handle.cp_errors = antenna_handle.cp_errors[idxs_old]
        store_handle.compute_azel(obs_full.time.values, antenna_handle)
        antenna_handle.antenna_data = obs_full
        antenna_handle.times_gps = obs_full.time.values
        if antenna_handle.ppp_clock_active is True: antenna_handle.interp_ppp_clock()
        store_handle.vlbi_like = True 
        store_handle.correct_PR_CP(antenna_handle, phase=True)

    duration_arr = (np.array(scan_end) - np.array(scan_beg))/np.timedelta64(1, 's')
    store_handle.duration_dict = {}
    for idx, time in enumerate(datetime_ref):
        store_handle.duration_dict[time] = duration_arr[idx]

def aligned_range(start, end, step_s):
    """
    Return clock-aligned epochs between [start, end] using step (e.g., '30s', '1m').
    Start is snapped up (ceil) to the grid; end is snapped down (floor) to the grid.
    """
    step = np.timedelta64(int(step_s), 's')
    # Normalize to nanoseconds for safe integer arithmetic
    start_ns = start.astype('datetime64[ns]').astype('int64')
    end_ns   = end.astype('datetime64[ns]').astype('int64')
    step_ns  = step.astype('timedelta64[ns]').astype('int64')

    # Ceil(start) to next multiple of step, floor(end) to previous multiple
    astart_ns = ((start_ns + step_ns - 1) // step_ns) * step_ns
    aend_ns   = (end_ns // step_ns) * step_ns

    n = (aend_ns - astart_ns) // step_ns + 1
    arr_ns = astart_ns + step_ns * np.arange(n, dtype='int64')
    return arr_ns.astype('datetime64[ns]')

def vlbi_transform_obs(store_handle, antenna_handle, freq, nsec_out):
    """
    Use VLBI-like processing to transform 1 Hz OBS data to 1 point per scan for one antenna
    """
    # find array of common reference times
    datetime_ref = aligned_range(antenna_handle.times_gps[0], antenna_handle.times_gps[-1], nsec_out)
    scan_beg = datetime_ref - np.timedelta64(int(nsec_out/2*1e9), 'ns')
    scan_end = datetime_ref + np.timedelta64(int(nsec_out/2*1e9), 'ns')
    wavelength = const.c/freq
    plot = False
    obs_data = antenna_handle.antenna_data
    obs_full = []
    if len(set(store_handle.source_time_dict.values())) == 1:
        source_ref = set(store_handle.source_time_dict.values()).pop()
    else:
        source_ref = np.array([store_handle.source_time_dict[time] for time in datetime_ref])

    for idx, time in enumerate(datetime_ref):
        if len(source_ref)>3:
            src = source_ref[idx]
        else:
            src = source_ref
        obs_time = antenna_handle.antenna_data.where(antenna_handle.antenna_data.sv == src, drop=True)
        obs_time = obs_time.sel(time=slice(scan_beg[idx], scan_end[idx]))

        ref_idx = np.argwhere(obs_time.time.values==time)
        if len(ref_idx) > 0: 
            ref_idx = ref_idx[0]

        if len(obs_time.time.values)>1:
             # pr_data, pr_model; cp_data, cp_model
             eps_range = (obs_time.pr_data - obs_time.pr_model).values
             eps_phase = (obs_time.cp_data - obs_time.cp_model).values

             sig_range, use_idxs_range = find_sigmas(eps_range-np.median(eps_range[~np.isnan(eps_range)]))
             sig_phase, use_idxs_phase = find_sigmas(eps_phase-np.median(eps_phase[~np.isnan(eps_phase)]))
             times_sec = (obs_time.time.values-obs_time.time.values[0])/np.timedelta64(1,'s')
             
             if len(ref_idx)>0:
                 ref_time = times_sec[ref_idx][0]
             else:
                 ref_time = (time-obs_time.time.values[0])/np.timedelta64(1,'s')

             # fit pseudorange
             res_pr = linregress(times_sec[use_idxs_range], eps_range[use_idxs_range])
             eps_pr_ref = res_pr.slope*ref_time + res_pr.intercept
             if len(ref_idx) > 0:
                 pr_meas = obs_time.pr_model[ref_idx] + eps_pr_ref
             else:
                 model_pr = linregress(times_sec[use_idxs_range], obs_time.pr_model.values[use_idxs_range])
                 pr_meas = np.array([eps_pr_ref + model_pr.slope*ref_time + model_pr.intercept])

             # fit carrier phase
             if len(times_sec[use_idxs_phase])>=5 and nsec_out > 1:
                 res_cp = make_smoothing_spline(times_sec[use_idxs_phase], eps_phase[use_idxs_phase])
                 eps_cp_ref = res_cp(ref_time)
                 res_type = 'spline'
             elif ~np.all(np.isnan(eps_phase)):
                 res_cp = linregress(times_sec[use_idxs_phase], eps_phase[use_idxs_phase])
                 eps_cp_ref = res_cp.slope*ref_time + res_cp.intercept
                 res_type = 'line'
             elif len(ref_idx) > 0:
                 eps_cp_ref = -obs_time.cp_model[ref_idx] # set phase to 0 as failure mode
             else:
                 eps_cp_ref = 0

             if len(ref_idx) > 0:
                 cp_meas = obs_time.cp_model[ref_idx] + eps_cp_ref
             elif len(times_sec[use_idxs_phase])>0:
                 # this isnt sound, but it wont matter because DMS=0 or 1 on this obs, and it wont be included in RINEX
                 model_cp = linregress(times_sec[use_idxs_phase], obs_time.cp_model.values[use_idxs_phase])
                 cp_meas = np.array([eps_cp_ref + model_cp.slope*ref_time + model_cp.intercept])
             else:
                 cp_meas = np.array([0])
             
             if plot is True and idx == 1:
                fig, [ax1, ax2] = plt.subplots(2, dpi=300)
                ax1.scatter(times_sec, eps_range, label='data', zorder=1)
                ax1.plot(times_sec, res_pr.slope*times_sec + res_pr.intercept, label='linear fit', zorder=2, c='r')
                ax1.scatter(ref_time, eps_pr_ref, label='new meas', marker='*', zorder=3)
                ax2.scatter(times_sec, eps_phase, label='data', zorder=1)
                if res_type == 'spline': 
                    ax2.plot(times_sec, res_cp(times_sec), label='linear fit', zorder=2, c='r')
                else:
                    ax2.plot(times_sec, res_cp.slope*times_sec + res_cp.intercept, label='linear fit', zorder=2, c='r')
                ax2.scatter(ref_time, eps_cp_ref, label='new meas', marker='*', zorder=3)
                ax1.legend()
                ax1.set_ylabel('Range-Model (m)')
                ax2.set_ylabel('Phase-Model (m)')
                ax1.set_title('Source ' + src)
                ax2.set_xlabel('time (sec)')
                fig.savefig('./VLBI_test_' +str(idx)+'_'+antenna_handle.antenna_name +'.png')
                plt.close()
        else:
            # no fitting to do, only 1 value
            # this happens for IGS data
            pr_meas = np.array([0])
            cp_meas = np.array([0])
            bypass=True

        # save data to dict
        smoothed_values = {}
        bypass=False
        if len(ref_idx) > 0:
            smoothed_values['pr_model'] = obs_time.pr_model[ref_idx]
            smoothed_values['cp_model'] = obs_time.cp_model[ref_idx]
        elif len(obs_time.time.values)>1 and len(times_sec[use_idxs_phase])>0:
            smoothed_values['pr_model'] = np.array([model_pr.slope*ref_time + model_pr.intercept])
            smoothed_values['cp_model'] = np.array([model_cp.slope*ref_time + model_cp.intercept])
        else:
            smoothed_values['pr_model'] = np.array([0])
            smoothed_values['cp_model'] = np.array([0])
            bypass=True
        smoothed_values['pr_data'] = pr_meas
        smoothed_values['cp_data'] = cp_meas

        # Create the new dataset 
        if len(ref_idx) > 0 and bypass == False:
            data_dict = {var_name: (["time"], value.values) for var_name, value in smoothed_values.items()}
        else:
            data_dict = {var_name: (["time"], value) for var_name, value in smoothed_values.items()}
        obs_vlbi = xr.Dataset(
                     data_dict,
                     coords={"time": [time]})
        obs_vlbi = obs_vlbi.assign_coords(sv=src)
        obs_vlbi.attrs = obs_time.attrs

        if len(obs_full)==0:
            obs_full = obs_vlbi
        else:
            obs_full = xr.concat((obs_full, obs_vlbi), dim='time')

        #print(time)
        #if time == np.datetime64('2025-04-22T00:00:00.000000000', 'ns'):
        #    breakpoint()

    antenna_handle.antenna_data = obs_full
    if len(obs_full)>0:
        antenna_handle.times_gps = obs_full.time.values
    else:
        antenna_handle.times_gps = []


#def write_SINEX(sol_type, sol_name, ls_sol, store_handle, antenna_handles, ref_antenna):
#    """
#    Write a SINEX file with:
#      - One reference station (fixed)
#      - Multiple floated stations
#      - SOLUTION/APRIORI, SOLUTION/ESTIMATE, SOLUTION/MATRIX_ESTIMATE
#      - LOCL_TIE for each floated station wrt reference
#
#    All lists (sta_codes, sta_domes, sta_xyz_apriori, sta_xyz_est, covariances)
#    must have the same length N_floated.
#    """
#    agency = 'ARL:UT' # str,
#    frame_epoch = '2015:001:00000'  # str
#    ref_frame = 'ITRF2020-u2023'
#    units= 'METERS'
#    N = len(antenna_handles)-1
#    final_state = ls_sol.x
#    unit_var = get_unit_var_full(ls_sol.fun, final_state)
#    J = ls_sol.jac  # Jacobian of the solution
#    #J = J[:len(residuals_obs),:]
#    if np.linalg.cond(J.T@ J) > 1e9:
#        cov_matrix_full = pinv(J.T @ J)
#    else:
#        cov_matrix_full = np.linalg.inv(J.T @ J)
#
#    cov_matrix_full *= unit_var
#    cov_full = cov_matrix_full[:N*3,:N*3]
#    rxpos_states = final_state[:N*3]
#
#    # get positions
#    sta_codes = []
#    sta_domes = []
#    sta_xyz_apriori = []
#    sta_xyz_est = []
#    count = 0
#    antenna_names = ''
#    for antenna_handle in antenna_handles:
#        antenna_names = antenna_names+antenna_handle.antenna_name+'_'
#        if ref_antenna == antenna_handle.antenna_name:
#            dt_epoch = antenna_handle.times_gps[0].astype("datetime64[s]").astype(datetime.datetime) # str
#            year = dt_epoch.year
#            doy  = dt_epoch.timetuple().tm_yday
#            sec  = dt_epoch.hour*3600 + dt_epoch.minute*60 + dt_epoch.second
#            est_epoch = f"{year:04d}:{doy:03d}:{sec:05d}"
#
#            sta_ref_code = antenna_handle.sta_code #: str,
#            sta_ref_domes = antenna_handle.domes_name # st,
#            sta_ref_xyz = antenna_handle.ref_pos # shape (3,)
#        else:
#            sta_codes.append('0'+antenna_handle.sta_code) # list of station codes (each length 5)
#            sta_domes.append(antenna_handle.domes_name) # list of corresponding DOMES strings
#            sta_xyz_apriori.append(antenna_handle.ref_pos)  # list of (3,) arrays
#            sta_xyz_est.append(rxpos_states[count*3:count*3+3]) # list of (3,) arrays
#            count+=1
#
#    filename =sol_type+'_'+sol_name+'_local_tie_'+antenna_names + '.SNX' 
#    expected_size = 3 * N
#
#    # Extract per-station subcovariances and deltas for LOCL_TIE
#    sigma_list = []
#    rho_list = []
#    deltas = []
#    for k in range(N):
#        idx0 = 3 * k
#        Ck = cov_full[idx0:idx0+3, idx0:idx0+3]
#        sigma_k = np.sqrt(np.diag(Ck))
#        sigma_list.append(sigma_k)
#        # correlations
#        rho_xy = Ck[0,1] / (sigma_k[0] * sigma_k[1]) if sigma_k[0]*sigma_k[1] != 0 else 0.0
#        rho_xz = Ck[0,2] / (sigma_k[0] * sigma_k[2]) if sigma_k[0]*sigma_k[2] != 0 else 0.0
#        rho_yz = Ck[1,2] / (sigma_k[1] * sigma_k[2]) if sigma_k[1]*sigma_k[2] != 0 else 0.0
#        rho_list.append((rho_xy, rho_xz, rho_yz))
#        # delta vs reference
#        deltas.append(sta_xyz_est[k] - sta_ref_xyz)
#    # Write SINEX
#    with open(filename, 'w') as f:
#        # HEADER
#        f.write("%======================\n")
#        f.write("%%=SNX 2.02\n")
#        f.write("* +SOLUTION/SINEX_HEADER\n")
#        f.write("*   SITE    DOMES       C_P  L_P   REF_FRAME         SCALE   UNITS\n")
#        f.write(f"    {agency:<4s}   {frame_epoch:<11s}   0   0    {ref_frame:<15s}   1   {units}\n")
#        f.write("* +SOLUTION/SINEX_HEADER\n\n")
#
#        # SOLUTION/APRIORI
#        f.write("+SOLUTION/APRIORI\n")
#        f.write("%   INDEX  PT  SOLN   STN    SYS   SOLN   ALIAS        IT   APPROX POSITION XYZ (m)           SIG_XY  SIG_Z\n")
#        # Reference station (fixed), now allow DOMES up to 9 chars
#        f.write(
#            f"    1      1   FIX   {sta_ref_code:<5s}  T     APR   {sta_ref_domes:<9s}   1   "
#            f"{sta_ref_xyz[0]:10.6f} {sta_ref_xyz[1]:10.6f} {sta_ref_xyz[2]:10.6f}   0.00000 0.00000\n"
#        )
#        # Floated stations start at index=2
#        for k in range(N):
#            idx = k + 2
#            placeholder_sigma = max(sigma_list[k])
#            f.write(
#                f"    {idx:<5d}  1   FLT   {sta_codes[k]:<5s}  T     APR   {sta_domes[k]:<9s}   1   "
#                f"{sta_xyz_apriori[k][0]:10.6f} {sta_xyz_apriori[k][1]:10.6f} {sta_xyz_apriori[k][2]:10.6f}   "
#                f"{placeholder_sigma:7.5f} {placeholder_sigma:7.5f}\n"
#            )
#        f.write("+SOLUTION/APRIORI\n\n")
#
#        # SOLUTION/ESTIMATE
#        f.write("+SOLUTION/ESTIMATE\n")
#        f.write("%   INDEX  PT  SOLN   STN    REF_EPOCH       PARAMETER   ESTIMATE          SIGMA\n")
#        est_index = 1
#        for k in range(N):
#            for coord_name, est_val, sigma_val in zip(("X","Y","Z"), sta_xyz_est[k], sigma_list[k]):
#                f.write(
#                    f"    {est_index:<5d}  1   FLT   {sta_codes[k]:<5s}  {est_epoch:<11s}   "
#                    f"{coord_name:<1s}      {est_val:12.6f}    {sigma_val:7.6f}\n"
#                )
#                est_index += 1
#        f.write("+SOLUTION/ESTIMATE\n\n")
#
#        # SOLUTION/MATRIX_ESTIMATE
#        f.write("+SOLUTION/MATRIX_ESTIMATE\n")
#        f.write("%   TYPE   PT  COV_PT  SOLN PT  #   INDEX  INDEX   VALUE\n")
#        total_params = expected_size
#        # Write full lower triangle
#        for i in range(total_params):
#            for j in range(i+1):
#                val = cov_full[i, j]
#                if i == j or abs(val) > 0.0:
#                    f.write(
#                        f"   X      1    1      1      {total_params:<2d}  {i+1:<5d}  {j+1:<5d}  {val: .2e}\n"
#                    )
#        f.write("+SOLUTION/MATRIX_ESTIMATE\n\n")
#
#        # LOCL_TIE/CONSTRAINTS
#        f.write("+LOCL_TIE/CONSTRAINTS\n")
#        f.write("%   ID1    ID2     DX           DY           DZ            σ_DX      σ_DY      σ_DZ    ρ_XY    ρ_XZ    ρ_YZ\n")
#        for k in range(N):
#            f.write(
#                f"    {sta_ref_code:<5s}  {sta_codes[k]:<5s}"
#                f"  {deltas[k][0]:12.6f}  {deltas[k][1]:12.6f}  {deltas[k][2]:12.6f}  "
#                f"{sigma_list[k][0]:7.6f}  {sigma_list[k][1]:7.6f}  {sigma_list[k][2]:7.6f}  "
#                f"{rho_list[k][0]:6.4f}  {rho_list[k][1]:6.4f}  {rho_list[k][2]:6.4f}\n"
#            )
#        f.write("+LOCL_TIE/CONSTRAINTS\n\n")
#        f.write("%======================\n")
#
#    return

def _ecef_to_geodetic(x, y, z):
    """WGS84 ECEF (m) -> (lat_deg, lon_deg, h_m)."""
    a  = 6378137.0
    f  = 1.0 / 298.257223563
    e2 = f * (2 - f)
    b  = a * (1 - f)
    ep2 = (a*a - b*b) / (b*b)
    p  = np.hypot(x, y)
    th = np.arctan2(a * z, b * p)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z + ep2 * b * np.sin(th)**3,
                     p - e2  * a * np.cos(th)**3)
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    h = p / np.cos(lat) - N
    return np.degrees(lat), np.degrees(lon), h

def _dms(deg, wrap_positive=False):
    if wrap_positive and deg < 0:
        deg += 360.0
    sign = -1 if deg < 0 else 1
    deg = abs(deg)
    d = int(deg)
    mf = (deg - d) * 60.0
    m = int(mf)
    s = (mf - m) * 60.0
    return sign * d, m, s

def _epoch_str(dt):
    """datetime -> 'YY:DOY:SSSSS'."""
    return (f"{dt.year % 100:02d}:"
            f"{dt.timetuple().tm_yday:03d}:"
            f"{dt.hour*3600 + dt.minute*60 + dt.second:05d}")

def _as_datetime(t):
    """np.datetime64 or astropy Time scalar -> python datetime (seconds)."""
    if hasattr(t, "to_datetime"):
        return t.to_datetime()
    return t.astype("datetime64[s]").astype(datetime.datetime)

def _unit_variance(residuals, state):
    dof = len(residuals) - len(state)
    if dof <= 0:
        return 1.0
    return float(residuals @ residuals) / dof

def write_SINEX(sol_type, sol_name, ls_sol, store_handle,
                antenna_handles, ref_antenna,
                agency="ARL", tech="C"):
    """
    Write a SINEX 2.10 local-tie solution.

    Conventions
    -----------
    - All N antennas (reference + floated) appear in SITE/ID, SOLUTION/EPOCHS,
      and SOLUTION/ESTIMATE.
    - Parameters are ordered station-by-station as [STAX, STAY, STAZ].
    - The reference station's three coordinates are held fixed (constraint
      code 0, zero sigma, zero rows/columns in the covariance).
    - Floated station coordinates use constraint code 2 (unconstrained).
    - ls_sol.x is assumed to start with the 3 * N_flt floated coordinates,
      ordered to match antenna_handles with ref_antenna removed.
    """
    # ------------------------------------------------------------------
    # 1. covariance of floated coordinates
    # ------------------------------------------------------------------
    J = ls_sol.jac
    NtN = J.T @ J
    cov_all = pinv(NtN) if np.linalg.cond(NtN) > 1e9 else np.linalg.inv(NtN)
    cov_all *= _unit_variance(ls_sol.fun, ls_sol.x)

    n_flt = len(antenna_handles) - 1
    cov_flt = cov_all[:3*n_flt, :3*n_flt]
    x_flt   = ls_sol.x[:3*n_flt]

    # ------------------------------------------------------------------
    # 2. build ordered station table (ref first, then floated in input order)
    # ------------------------------------------------------------------
    stations = []            # list of dicts with the fields we need
    c = 0
    for ah in antenna_handles:
        t0 = _as_datetime(ah.times_gps[0])
        t1 = _as_datetime(ah.times_gps[-1])
        is_ref = (ah.antenna_name == ref_antenna)
        entry = dict(
            code   = ah.sta_code[:4].upper(),
            domes  = ah.domes_name,
            desc   = ah.antenna_name,
            t0     = t0,
            t1     = t1,
            is_ref = is_ref,
        )
        if is_ref:
            entry["xyz"]    = np.asarray(ah.ref_pos,  dtype=float)
            entry["sigma"]  = np.zeros(3)
            entry["constr"] = "0"
        else:
            entry["xyz"]    = np.asarray(x_flt[3*c:3*c+3], dtype=float)
            entry["sigma"]  = np.sqrt(np.clip(
                np.diag(cov_flt[3*c:3*c+3, 3*c:3*c+3]), 0.0, None))
            entry["constr"] = "2"
            entry["flt_idx"] = c     # index into cov_flt
            c += 1
        stations.append(entry)

    # put reference first so its 3 params get indices 1..3
    stations.sort(key=lambda s: (not s["is_ref"], s.get("flt_idx", 0)))

    # rebuild the parameter-order covariance with a 3-row/col zero pad for ref
    n_par = 3 * len(stations)
    cov_par = np.zeros((n_par, n_par))
    for i, s in enumerate(stations):
        if s["is_ref"]:
            continue
        for j, t in enumerate(stations):
            if t["is_ref"]:
                continue
            cov_par[3*i:3*i+3, 3*j:3*j+3] = cov_flt[
                3*s["flt_idx"]:3*s["flt_idx"]+3,
                3*t["flt_idx"]:3*t["flt_idx"]+3,
            ]

    # ------------------------------------------------------------------
    # 3. epochs
    # ------------------------------------------------------------------
    t_start = min(s["t0"] for s in stations)
    t_end   = max(s["t1"] for s in stations)
    t_mean  = t_start + (t_end - t_start) / 2
    creation = datetime.datetime.utcnow()

    fname_tag = "_".join(s["desc"] for s in stations)
    filename  = f"{sol_type}_{sol_name}_{fname_tag}.SNX"

    # ------------------------------------------------------------------
    # 4. write file
    # ------------------------------------------------------------------
    SEP = "*" + "-" * 79 + "\n"

    with open(filename, "w") as f:
        # --- header -----------------------------------------------------
        # Constraint code of the overall solution: 2 = unconstrained
        f.write(
            f"%=SNX 2.10 {agency:<3s} {_epoch_str(creation)} "
            f"{agency:<3s} {_epoch_str(t_start)} {_epoch_str(t_end)} "
            f"{tech} {n_par:05d} 2\n"
        )
        f.write(SEP)

        # --- FILE/REFERENCE --------------------------------------------
        f.write("+FILE/REFERENCE\n")
        f.write(" DESCRIPTION       Local tie SINEX\n")
        f.write(f" OUTPUT            {sol_type} / {sol_name}\n")
        f.write(f" SOFTWARE          PyLocalTie\n")
        f.write(f" INPUT             {','.join(s['desc'] for s in stations)}\n")
        f.write("-FILE/REFERENCE\n")
        f.write(SEP)

        # --- FILE/COMMENT ----------------------------------------------
        f.write("+FILE/COMMENT\n")
        f.write(f"* Solution:        {sol_type} / {sol_name}\n")
        f.write(f"* Reference site:  {stations[0]['code']} "
                f"({stations[0]['desc']}) held fixed\n")
        f.write(f"* Floated sites:   {n_flt}\n")
        f.write("-FILE/COMMENT\n")
        f.write(SEP)

        # --- SITE/ID ----------------------------------------------------
        f.write("+SITE/ID\n")
        f.write("*CODE PT __DOMES__ T _STATION DESCRIPTION__ "
                "APPROX_LON_ APPROX_LAT_ _APP_H_\n")
        for s in stations:
            lat, lon, h = _ecef_to_geodetic(*s["xyz"])
            ld, lm, ls  = _dms(lon, wrap_positive=True)
            bd, bm, bs  = _dms(lat)
            f.write(
                f" {s['code']:<4s}  A {s['domes']:<9s} {tech} "
                f"{s['desc'][:20]:<22s}"
                f"{ld:3d} {lm:2d} {ls:4.1f} "
                f"{bd:3d} {bm:2d} {bs:4.1f} "
                f"{h:7.1f}\n"
            )
        f.write("-SITE/ID\n")
        f.write(SEP)

        # --- SOLUTION/EPOCHS -------------------------------------------
        f.write("+SOLUTION/EPOCHS\n")
        f.write("*Code PT SOLN T Data_start__ Data_end____ Mean_epoch__\n")
        for s in stations:
            m = s["t0"] + (s["t1"] - s["t0"]) / 2
            f.write(
                f" {s['code']:<4s}  A    1 {tech} "
                f"{_epoch_str(s['t0'])} {_epoch_str(s['t1'])} "
                f"{_epoch_str(m)}\n"
            )
        f.write("-SOLUTION/EPOCHS\n")
        f.write(SEP)

        # --- SOLUTION/ESTIMATE -----------------------------------------
        f.write("+SOLUTION/ESTIMATE\n")
        f.write("*INDEX TYPE__ CODE PT SOLN _REF_EPOCH__ UNIT S "
                "__ESTIMATED VALUE____ _STD_DEV___\n")
        ref_ep = _epoch_str(t_mean)
        idx = 1
        for s in stations:
            for tname, val, sig in zip(("STAX", "STAY", "STAZ"),
                                       s["xyz"], s["sigma"]):
                f.write(
                    f" {idx:5d} {tname:<6s} {s['code']:<4s}  A    1 "
                    f"{ref_ep} m    {s['constr']} "
                    f"{val: .14E} {sig:.5E}\n"
                )
                idx += 1
        f.write("-SOLUTION/ESTIMATE\n")
        f.write(SEP)

        # --- SOLUTION/MATRIX_ESTIMATE L COVA ---------------------------
        # Packed lower triangle, up to 3 consecutive elements per line,
        # starting column given by PARA2.
        f.write("+SOLUTION/MATRIX_ESTIMATE L COVA\n")
        f.write("*PARA1 PARA2 ____PARA2+0__________ "
                "____PARA2+1__________ ____PARA2+2__________\n")
        for i in range(n_par):
            j = 0
            while j <= i:
                chunk = cov_par[i, j:min(j + 3, i + 1)]
                vals  = "".join(f" {v: .14E}" for v in chunk)
                f.write(f" {i+1:5d} {j+1:5d}{vals}\n")
                j += 3
        f.write("-SOLUTION/MATRIX_ESTIMATE L COVA\n")

        f.write("%ENDSNX\n")

    return filename

def datetime64_to_dsod(datetime_ns):
    """ Convert numpy dt64 to dsod float for EphTime """
    # Convert datetime64 to nanoseconds since Unix epoch
    nanoseconds_since_epoch = datetime_ns.astype('int64')

    # Extract the date (midnight) from the datetime64
    midnight = np.datetime64(np.array(datetime_ns), 'D')

    # Convert midnight to nanoseconds since Unix epoch
    midnight_nanoseconds_since_epoch = midnight.astype('int64')

    # Calculate nanoseconds of day
    nanoseconds_of_day = nanoseconds_since_epoch - midnight_nanoseconds_since_epoch

    # Convert nanoseconds to seconds (as double precision float)
    dsod = nanoseconds_of_day / 1e9
    
    return dsod

def datetime64_to_mjd(datetime_ns):
    """ Convert numpy dt64 to dsod mjd for EphTime """
    # Convert datetime64 to nanoseconds since Unix epoch
    nanoseconds_since_epoch = datetime_ns.astype('int64')

    # find JD
    jd_unix_epoch = 2440587.5
    days_since_epoch = nanoseconds_since_epoch / (1e9 * 60 * 60 * 24)
    jd = jd_unix_epoch + days_since_epoch

    # Convert JD to MJD
    mjd = float(jd - 2400000.5)
    
    return mjd

def datetime_to_mjd(dt):
    """ Convert datetime.datetime object to mjd """
    # Define the Unix epoch start
    epoch = datetime.datetime(1970, 1, 1)
    
    # Calculate the total difference in days from the epoch
    delta = dt - epoch
    days_since_epoch = delta.total_seconds() / (24 * 60 * 60)
    
    # JD for the Unix epoch (1970-01-01T00:00:00Z) is 2440587.5
    jd_unix_epoch = 2440587.5
    jd = jd_unix_epoch + days_since_epoch
    
    # Convert JD to MJD
    mjd = float(jd - 2400000.5)
    return mjd

def datetime_to_day(dt):
    """ Convert datetime.datetime object to mjd """
    # Define the Unix epoch start
    epoch = datetime.datetime(dt.year, 1, 1)
    
    # Calculate the total difference in days from the epoch
    delta = dt - epoch
    fractional_day_of_year = delta.total_seconds() / (24 * 60 * 60)
    
    DOY = 1 + fractional_day_of_year 
    return DOY

def map_datasets(rinex_data):
    """ Map RINEX 3 carrier/ranging codes to RINEX2 """
    for rinex2_type, rinex3_list in mapping_dict.items():
        # Initialize a list to hold the data arrays for each rinex2_type
        data_arrays = []
        for rinex3_type in rinex3_list:
            # Check if the variable exists in the dataset
            if rinex3_type in rinex_data:
                data_arrays.append(rinex_data[rinex3_type])
    
        if data_arrays:
            # Combine the data arrays using 'combine_first'
            combined_data = data_arrays[0]
            for additional_data in data_arrays[1:]:
                combined_data = combined_data.combine_first(additional_data)
    
            # Assign the combined data to the new variable name in the dataset
            rinex_data[rinex2_type] = combined_data
    
    # Drop the old RINEX3 data variables from the dataset
    variables_to_drop = [var for var in sum(mapping_dict.values(), []) if var in rinex_data]
    rinex_data = rinex_data.drop_vars(variables_to_drop)
    return rinex_data

def find_common_epochs(thinned_data, antenna_names, datetime_array, source_array, point_ra_dec_arr = []):
    """ Thin to only epochs with common nonzero PR data """
    # Initialize a combined mask as False for all epochs in the first dataset

    for antenna_idx, antenna_name in enumerate(antenna_names):
        antenna_data = thinned_data[antenna_name]
        if antenna_idx == 0:
            combined_mask = xr.full_like(antenna_data['C1'], False, dtype=bool)
        else:
            mask = antenna_data['C1'].fillna(0) == 0  # Create a mask for the current file
            combined_mask = combined_mask | mask   # Update the combined mask
        
    for antenna_idx, antenna_name in enumerate(antenna_names):
        # Apply the combined mask to each dataset
        antenna_data_orig = thinned_data[antenna_name]
        antenna_data = antenna_data_orig.where(~combined_mask, drop=True)
        thinned_data[antenna_name] = antenna_data 
    
    # convert list to xarray
    datetime_xr = xr.DataArray(datetime_array, dims=["time"], coords={"time": antenna_data_orig.time})
    source_xr = xr.DataArray(source_array, dims=["time"], coords={"time": antenna_data_orig.time})

    # Apply the combined mask to filter arrays
    datetime_filtered = datetime_xr.where(~combined_mask, drop=True)
    source_filtered = source_xr.where(~combined_mask, drop=True)

    # Convert back to lists 
    datetime_array = datetime_filtered.values.tolist()
    datetime_array = to_datetime(datetime_array)
    datetime_array = datetime_array.to_pydatetime().tolist()
    source_array = source_filtered.values.tolist()

    if len(point_ra_dec_arr)>0:
        point_ra_dec_arr = np.array(point_ra_dec_arr)[~combined_mask]
        if point_ra_dec_arr.ndim==3:
            # we have an extra, unfilled dimension
            point_ra_dec_arr = point_ra_dec_arr[0]
 
    return thinned_data, datetime_array, source_array, point_ra_dec_arr

class NavStore:
    """
    Class that can read and interpret multiple navigation & ephemeris file formats.
    This is a relatively thin wrapper around the gnsstk.NavLibrary object.
    It stores information on the orbits of multiple satellites, and information on a single receiver observing them.

    Allows the user to:
    - Read and interpret multiple ephemeris file formats.
    - Extract SV XVT and elevation/azimuth information (if the receiver position is set).

    Args:
        eph_files(list(str)): File names to read into the navigation store.
        rx_pos (gnsstk.Position): Receiver location, a GNSSTk object specifying ECEF (X,Y,Z) in meters.
        sv_health (str): How to filter SV's by health: unknown, any, healthy, unhealthy, degraded.
        nav_factory (gnsstk.NavDataFactory):
            Specify a NavDataFactory for processing ephemeris files
            This allows the instantiation of multiple NavStores since
            MultiFormatNavDataFactory is bypassed
    """

    def __init__(self, eph_files, rx_pos=None, sv_health=None, *, nav_factory=None):
        self.rx_pos = rx_pos
        self.nav_lib = NavLibrary()

        # Check if user specified a factory format, else instantiate a MultiFormatNavDataFactory
        if isinstance(nav_factory, NavDataFactory):
            self.ndf = nav_factory
        else:
            self.ndf = MultiFormatNavDataFactory()

        self.nav_lib.addFactory(self.ndf)
        self.eph = None

        if sv_health is not None:
            sv_health_types = get_sv_health_dict()
            self.sv_health = sv_health_types[sv_health]
        else:
            self.sv_health = SVHealth.Any

        self.read(eph_files)

    def read(self, eph_files):
        """
        Reads ephemeris files into the NavDataFactory instance.
        This function is called by the constructor, but may be used at any time to add additional files.

        Args:
            eph_files (list(str)): Ephemeris file names to read in.
        """
        # Flatten list of lists, if needed (to support old behavior where you had to pass in a list of lists).
        if any(isinstance(sublist, list) for sublist in eph_files):
            eph_files = [item for sublist in eph_files for item in sublist]

        # Ensure all filename strings are not unicode.
        eph_files = [str(eph_file) for eph_file in eph_files]

        for eph_file in eph_files:
            loaded = self.ndf.addDataSource(eph_file)

        if self.nav_lib is not None:
            self.eph = self.nav_lib


    def add_data_source(self, eph_file):
        """
        Adds another file to the MultiFormatNavDataFactory which adds more data to the NavStore Instance

        Args:
            eph_file (str): Nav file to be read in

        Returns:

            bool: True if read was successful
        """
        return self.ndf.addDataSource(eph_file)

    def get_sat_ids(self, sat_system=None):
        """
        Determines what SatIDs are present in the NavStore and returns them.

        Args:
            sat_system (int): Enum value of the satellite system (GNSSTk integer label).

        Returns:
            (list[gnsstk.SatID]): List of SatIDs present in the NavStore.
        """
        if sat_system is None:
            sat_system = SatelliteSystem.GPS

        # Define all possible sat nums for any systems in ephemeris data.
        sat_nums = list(range(1, 999))

        # Determine what sat_ids are present in the NavStore and save them.
        sat_ids = []

        for sat_num in sat_nums:
            sat_id = SatID(sat_num, sat_system)
            nsid = NavSatelliteID(sat_id)  # must turn SatID into NavSatID for NavDataFactory.getXvt()

            #Currently limited to NavMessageType.Ephemeris
            nmid = NavMessageID(nsid, NavMessageType.Ephemeris)
            nmid.obs.mcodeMask = 0
            if self.eph.isPresent(nmid, self.eph.getInitialTime(), self.eph.getFinalTime()):
                sat_ids.append(sat_id)
        return sat_ids

    def get_xvt(self, sat_id, common_time):
        """
        Gets a GNSSTk XvtStore object for the GNSSTk SatID (or NavSatelliteID) and CommonTime specified.

        Args:
            sat_id (Union[gnsstk.SatID, gnsstk.NavSatelliteID]): GNSSTk SatID or NavSatelliteID object
            common_time (gnsstk.CommonTime): time

        Returns:
            gnsstk.Xvt: GNSSTk Xvt object
        """
        # Setup a NavMessageID of the SV for NavLibrary to search with
        # Even though the NavMessageType is set to "Ephemeris", the NavLibrary
        # `getXvt()` can still search almanac data, implicitly or explicitly
        # Currently, M code is ignored.
        # TODO: add more support for M code
        nsid = NavSatelliteID(sat_id)
        nmid = NavMessageID(nsid, NavMessageType.Ephemeris)
        nmid.obs.mcodeMask = 0

        # Set up search parameters for finding XVT source.
        # Currently, health and validity do not matter.
        # Using `NavSearchOrder.Nearest` instead of `NavSearchOrder.User`
        # since `NavSearchOrder.User` is typically for replicating what a user would see and use in real-time.
        valid = NavValidityType.Any
        order = NavSearchOrder.Nearest

        # `xvt` is the "out" parameter of `NavLibrary.getXvt()`.
        # Currently, using the `getXvt()` method that will first check for ephemeris data
        # then fall back to almanac data if there is no ephemeris for the SV.
        xvt = Xvt()
        found_xvt = self.eph.getXvt(nmid, common_time, xvt, self.sv_health, valid, order)

        return xvt if found_xvt else None

    def get_sat_record(self, sat_id, common_time, fill_value=np.nan):
        """
        Given input GNSSTk SatID/NavSatelliteID and CommonTime for a single time point,
        extracts values needed to construct a dictionary of satellite position and velocity components.

        Args:
            sat_id (gnsstk.SatID): GNSSTk satellite ID (SatID or NavSatelliteID)
            common_time (gnsstk.CommonTime): epoch of inquiry
            fill_value (scalar): default fill value for empty indices

        Returns:
            dict: satellite time and components of position & velocity
        """
        # Initialize dict
        sat_record = dict()
        keys = ['px', 'py', 'pz', 'vx', 'vy', 'vz', 'clkbias', 'clkdrift', 'relcorr', 'elevation', 'azimuth', 'range']
        for key in keys:
            sat_record[key] = fill_value

        # Retrieve xvt info
        try:
            sat_xvt = self.get_xvt(sat_id, common_time)
        except InvalidRequest as ex:
            return sat_record

        if sat_xvt is not None:
            sat_record["px"] = sat_xvt.x[0]  # SV X-position
            sat_record["py"] = sat_xvt.x[1]  # SV Y-position
            sat_record["pz"] = sat_xvt.x[2]  # SV Z-position
            sat_record["vx"] = sat_xvt.v[0]  # SV X-velocity
            sat_record["vy"] = sat_xvt.v[1]  # SV Y-velocity
            sat_record["vz"] = sat_xvt.v[2]  # SV Z-velocity
            sat_record["clkbias"]   = sat_xvt.clkbias
            sat_record["clkdrift"]  = sat_xvt.clkdrift
            sat_record["relcorr"]   = sat_xvt.relcorr

            if self.rx_pos is not None:
                sat_range = self.get_range(sat_xvt)
                sat_pos = Position(sat_xvt.x[0], sat_xvt.x[1], sat_xvt.x[2])
                # TODO: Order should be azim,elev to conform with nomenclature.
                azimuth, elevation = self.get_azel(sat_pos)
                # put contents in dictionary
                sat_record["range"]     = sat_range
                sat_record["elevation"] = elevation
                sat_record["azimuth"]   = azimuth

        return sat_record

    def get_sat_track(self, sat_id, common_times, fill_value=np.nan):
        """
        Generate a sat_track for a satellite given a GNSSTk SatID and an array of GNSSTk CommonTimes.

        A sat_track is a NumPy recarray of essential satellite track information:
          - position coordinate (x,y,z)
          - velocity vector (vx,vy,vz)
          - elev/azim
          - range
          - clock bias & drift
          - the relativity correction

        Args:
            sat_id (gnsstk.SatID): GNSSTk satellite object
            common_times (iter(gnsstk.CommonTime)): iterable of gnsstk.CommonTime objects
            fill_value (scalar): default fill value for empty indices

        Returns:
            np.rec.array: Numpy record array containing the following keys (all of which are numpy arrays):
            - 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'elevation', 'azimuth', 'range', 'clkbias', 'clkdrift', 'relcorr'
        """

        # Set up the return NumPy array.
        sat_track = np.empty(shape=np.array(common_times).shape, dtype=EPH_DTYPE)
        sat_track.fill(fill_value)

        # Need to construct a gnsstk.NavSatelliteID for NavDataFactory method calls.
        nsid = NavSatelliteID(sat_id)
        nmide = NavMessageID(nsid, NavMessageType.Ephemeris)
        nmide.obs.mcodeMask = 0
        nmida = NavMessageID(nsid, NavMessageType.Almanac)
        nmida.obs.mcodeMask = 0

        # Setting start and stop time with Any timesystem for gnss compliance
        time_start = common_times[0].setTimeSystem(TimeSystem.Any)
        time_stop = common_times[-1].setTimeSystem(TimeSystem.Any)

        # Make sure ephemeris data for the satellite exists.
        if self.eph is None:
            raise ValueError('ephemeris not found') 
        elif not (self.eph.isPresent(nmide, time_start, time_stop) or self.eph.isPresent(nmida, time_start, time_stop)):
            raise ValueError('ephemeris does not cover given time interval') 
        else:
            for ind, common_time in enumerate(common_times):
                # Skip check using isPresent() as in old class.
                sat_record = self.get_sat_record(sat_id, common_time, fill_value)
                for key in sat_record.keys():
                    sat_track[key][ind] = sat_record[key]

        return np.rec.array(sat_track)

    def get_azel(self, sat_pos, coordinate_system='geodetic'):
        """
        Compute elevation and azimuth of a satellite from the NavStore instance's receiver position, given an input
        gnsstk.Position object for the satellite.

        Args:
            sat_pos (gnsstk.Position): Satellite position
            coordinate_system (str): default='geodetic'; options= 'geocentric'

        Returns:
            tuple(float, float): Elevation and azimuth of the satellite at the receiver [degrees].

        Raises:
            RuntimeError: Raised if an unsupported coordinate system is input.
        """
        if self.rx_pos is None:
            return np.nan, np.nan
        if coordinate_system == 'geodetic':
            elev = self.rx_pos.elevationGeodetic(sat_pos)
            azim = self.rx_pos.azimuthGeodetic(sat_pos)
        elif coordinate_system == 'geocentric':
            elev = self.rx_pos.elevation(sat_pos)
            azim = self.rx_pos.azimuth(sat_pos)
        else:
            raise RuntimeError('Unsupported coordinate_system {}'.format(coordinate_system))

        return azim, elev

    def get_range(self, sat_xvt):
        """
        Compute the range from the receiver position to a satellite for the time in the input XvtStore.

        Args:
            sat_xvt (gnsstk.XvtStore): GNSSTk XvtStore object

        Returns:
            float: range from receiver to satellite, in meters
        """
        if self.rx_pos is None:
            return np.nan
        rx_pos = [self.rx_pos.X(), self.rx_pos.Y(), self.rx_pos.Z()]
        stn_xvt = Xvt()
        stn_xvt.x = Triple(*rx_pos)

        ellipsoid = WGS84Ellipsoid()
        correction = (sat_xvt.clkdrift + sat_xvt.relcorr) * ellipsoid.c()
        sat_range = sat_xvt.preciseRho(stn_xvt.x, ellipsoid, correction)

        return sat_range

def calc_ipp(rx_pos, sv_pos, iono_ht):
    """
    Computes the Ionosphere pierce Point.
    Args:
        rx_pos (list): Receiver position. ECEF [m]
        sv_pos (list): Satellite position. ECEF [m]
        iono_ht (float): VTEC grid height [km]

    Returns:
        xyz_Il (array): IPP position ECEF [m]
    """
    pos = Position(rx_pos[0], rx_pos[1], rx_pos[2])
    r_e = Position.radiusEarth(pos)
    xyz_1 = np.array(rx_pos)
    xyz_2 = np.array(sv_pos)
    iono_ht = iono_ht*1e3 + r_e
    d = xyz_2 - xyz_1
    a = np.dot(d, d)
    b = 2 * np.dot(xyz_1, d)
    c = np.dot(xyz_1, xyz_1) - iono_ht * iono_ht
    t1 = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    xyz_I1 = xyz_1 + (t1 * d.T).T
    return xyz_I1

def xyz_to_rll(xyz, in_deg=True):
    """
    Converts a cartesian point to a spherical radius/latitude/longitude
    Args:
        xyz (float, array_like) cartesian coordinates x, y, z

    Keyword Args:
        in_deg (bool): flag for angular units (default=True)

    Returns:
        r, lat, lon (float, tuple): spherical coordinates radius, latitude, longitude
    """
    x, y, z = np.asarray(xyz)
    rad = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    lat = np.arcsin(z / rad)
    lon = np.arctan2(y, x)
    if in_deg:
        lat = np.rad2deg(lat)
        lon = np.rad2deg(lon)
    return rad, lat, lon

DEG_PER_SEC = 360 / 86400
TECU_COEFF = 40.3 * 1e16
FILL_VTEC = 999.9
_FIELDS_PER_LINE = 16
_FIELD_W = 5
class IonexFile(object):
    """
    Reader/interpolator for IONEX TEC maps.

    Attributes:
        lats (array): Latitudes, ascending [deg]
        lons (array): Longitudes, ascending [deg]
        iono_hts (array): Ionosphere shell height per epoch [km]
        times (array): Map epochs, sorted, unique [UTC datetimes]
        tec_grids (array): VTEC, shape (n_time, n_lat, n_lon) [TECu]
        d_lat (float): Latitude spacing, positive [deg]
        d_lon (float): Longitude spacing, positive [deg]
        d_time (float): Median epoch spacing [s]
    """

    def __init__(self):
        self.lats = None
        self.lons = None
        self.iono_hts = None
        self.times = None
        self.tec_grids = None
        self.d_lat = None
        self.d_lon = None
        self.d_time = None
        # Internal: epoch offsets [s] for fast searchsorted
        self._t0 = None
        self._tsec = None

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    @classmethod
    def read(cls, filenames, mode='r'):
        """
        Read one or more IONEX files into a single object.

        Args:
            filenames (str or list): IONEX file path(s).
            mode (str): Open mode.

        Returns:
            IonexFile
        """
        try:
            if isinstance(filenames, str):
                filenames = [filenames]

            lats = lons = None
            times, hts, grids = [], [], []
            for filename in filenames:
                with open(filename, mode) as fid:
                    lines = fid.read().splitlines()
                f_lats, f_lons, f_times, f_hts, f_grids = cls._parse(lines)
                if lats is None:
                    lats, lons = f_lats, f_lons
                elif f_lats.shape != lats.shape or f_lons.shape != lons.shape:
                    raise Exception('Inconsistent grid definition across files')
                times.extend(f_times)
                hts.extend(f_hts)
                grids.extend(f_grids)

            if not grids:
                raise Exception('No TEC maps found')

            times = np.array(times)
            order = np.argsort(times)
            times = times[order]
            # Drop duplicate epochs (adjacent daily files share midnight)
            keep = np.concatenate(([True], times[1:] != times[:-1]))
            order = order[keep]

            obj = cls()
            obj.lats = lats
            obj.lons = lons
            obj.times = times[keep]
            obj.iono_hts = np.array(hts, dtype=np.float32)[order]
            obj.tec_grids = np.array(grids, dtype=np.float32)[order]
            obj.d_lat = abs(float(np.median(np.diff(lats))))
            obj.d_lon = abs(float(np.median(np.diff(lons))))
            obj._t0 = obj.times[0]
            obj._tsec = np.array([(t - obj._t0).total_seconds()
                                  for t in obj.times], dtype=np.float64)
            obj.d_time = float(np.median(np.diff(obj._tsec)))
            return obj
        except Exception as exc:
            raise Exception('Ionex read failed | {}'.format(exc))

    @staticmethod
    def _parse(lines):
        """
        Single-pass, fixed-width parse of an IONEX file.

        Returns:
            (lats, lons, times, iono_hts, grids) with lats ascending and each
            grid ordered (lat ascending, lon ascending).
        """
        exponent = -1
        lat_spec = lon_spec = None
        i_hdr = 0
        for i_hdr, line in enumerate(lines):
            label = line[60:].strip()
            if label == 'EXPONENT':
                exponent = int(line[:60].split()[0])
            elif label == 'LAT1 / LAT2 / DLAT':
                lat_spec = [float(line[2 + 6 * k:8 + 6 * k]) for k in range(3)]
            elif label == 'LON1 / LON2 / DLON':
                lon_spec = [float(line[2 + 6 * k:8 + 6 * k]) for k in range(3)]
            elif label == 'END OF HEADER':
                break
        if lat_spec is None or lon_spec is None:
            raise Exception('Missing LAT/LON grid definition in header')

        lat1, lat2, d_lat = lat_spec
        lon1, lon2, d_lon = lon_spec
        n_lat = int(round((lat2 - lat1) / d_lat)) + 1
        n_lon = int(round((lon2 - lon1) / d_lon)) + 1
        n_dl = -(-n_lon // _FIELDS_PER_LINE)          # data lines per latitude
        n_pad = n_dl * _FIELDS_PER_LINE * _FIELD_W    # padded chars per latitude
        flip = d_lat < 0
        lats = lat1 + d_lat * np.arange(n_lat, dtype=np.float64)
        lons = lon1 + d_lon * np.arange(n_lon, dtype=np.float64)
        if flip:
            lats = lats[::-1].copy()
        if d_lon < 0:
            lons = lons[::-1].copy()

        scale = 10.0 ** exponent
        times, hts, grids = [], [], []
        block, ht, in_map = [], None, False
        i, n = i_hdr + 1, len(lines)
        while i < n:
            line = lines[i]
            label = line[60:].strip()
            if label == 'START OF TEC MAP':
                in_map, block, ht = True, [], None
            elif in_map and label == 'END OF TEC MAP':
                raw = np.frombuffer(''.join(block).encode('latin-1'),
                                    dtype='S{}'.format(_FIELD_W))
                grid = raw.reshape(n_lat, -1)[:, :n_lon].astype(np.float32)
                grid *= scale
                if flip:
                    grid = grid[::-1]
                if d_lon < 0:
                    grid = grid[:, ::-1]
                grids.append(grid)
                hts.append(ht)
                in_map = False
            elif in_map and label == 'EPOCH OF CURRENT MAP':
                y, mo, d, hh, mm, ss = [int(v) for v in line[:60].split()[:6]]
                times.append(datetime.datetime(y, mo, d) +
                             datetime.timedelta(hours=hh, minutes=mm, seconds=ss))
            elif in_map and label == 'LAT/LON1/LON2/DLON/H':
                # Fixed width: fields run together for negative values
                ht = float(line[26:32])
                block.extend(l.ljust(n_pad)[:n_pad] for l in lines[i + 1:i + 1 + n_dl])
                i += n_dl
            elif label == 'EXPONENT':
                scale = 10.0 ** int(line[:60].split()[0])
            i += 1

        if len(times) != len(grids):
            raise Exception('Epoch/map count mismatch')
        return lats, lons, times, hts, grids

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------
    def interpolate_vtec(self, time, lat, lon):
        """
        Interpolate VTEC per the IONEX rotating-map scheme. Scalars or
        broadcastable arrays of lat/lon (and time) are accepted.

        Args:
            time (datetime or array): Desired time(s) [UTC]
            lat (float or array): Desired latitude(s) [deg]
            lon (float or array): Desired longitude(s) [deg]

        Returns:
            float or np.ndarray: Interpolated VTEC [TECu]
        """
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        if lat.min() < self.lats[0] - 1e-6 or lat.max() > self.lats[-1] + 1e-6:
            raise Exception('Provided latitude ({}) outside bounds {}-{}'.format(
                lat, self.lats[0], self.lats[-1]))
        i_0, i_1, c_0, c_1, rot_0, rot_1 = self._time_weights(time)
        v_0 = self._bilinear(i_0, lat, self._wrap_lon(lon + rot_0))
        v_1 = self._bilinear(i_1, lat, self._wrap_lon(lon + rot_1))
        vtec = c_0 * v_0 + c_1 * v_1
        return float(vtec) if np.ndim(vtec) == 0 else vtec

    def interpolate_vtec_grid(self, time, lats, lons):
        """
        Interpolated VTEC grid for a given time.

        Args:
            time (datetime): Time of grid
            lats (list): Latitudes [deg]
            lons (list): Longitudes [deg]

        Returns:
            np.ndarray: VTEC grid, shape (len(lats), len(lons)) [TECu]
        """
        lats = np.asarray(lats, dtype=np.float64).reshape(-1, 1)
        lons = np.asarray(lons, dtype=np.float64).reshape(1, -1)
        return np.asarray(self.interpolate_vtec(time, lats, lons), dtype=np.float64)

    def _bilinear(self, i_t, lat, lon):
        """Bilinear lookup in map(s) i_t at lat/lon. All args broadcastable."""
        n_lat, n_lon = self.lats.size, self.lons.size
        j_1 = np.clip(np.searchsorted(self.lats, lat, side='right'), 1, n_lat - 1)
        k_1 = np.clip(np.searchsorted(self.lons, lon, side='right'), 1, n_lon - 1)
        j_0, k_0 = j_1 - 1, k_1 - 1
        q = (lat - self.lats[j_0]) / self.d_lat
        p = (lon - self.lons[k_0]) / self.d_lon

        grids = self.tec_grids
        v_00 = grids[i_t, j_0, k_0]
        v_01 = grids[i_t, j_0, k_1]
        v_10 = grids[i_t, j_1, k_0]
        v_11 = grids[i_t, j_1, k_1]
        if np.any((v_00 == FILL_VTEC) | (v_01 == FILL_VTEC) |
                  (v_10 == FILL_VTEC) | (v_11 == FILL_VTEC)):
            raise Exception('Cannot interpolate: Desired VTEC uses Ionex file fill value.')
        return ((1 - p) * (1 - q) * v_00 + p * (1 - q) * v_01 +
                (1 - p) * q * v_10 + p * q * v_11)

    def _time_weights(self, time):
        """
        Bounding epoch indices, linear time weights, and the IONEX longitude
        rotations [deg] relative to each bounding epoch.
        """
        t = self._to_sec(time)
        i_1 = np.clip(np.searchsorted(self._tsec, t, side='right'),
                      1, self._tsec.size - 1)
        i_0 = i_1 - 1
        t_0, t_1 = self._tsec[i_0], self._tsec[i_1]
        dt = t_1 - t_0
        return i_0, i_1, (t_1 - t) / dt, (t - t_0) / dt, \
            (t - t_0) * DEG_PER_SEC, (t - t_1) * DEG_PER_SEC

    def _to_sec(self, time):
        """Datetime(s) -> seconds past the first map epoch."""
        if isinstance(time, datetime.datetime):
            return (time - self._t0).total_seconds()
        arr = np.asarray(time)
        return np.array([(t - self._t0).total_seconds()
                         for t in arr.ravel()], dtype=np.float64).reshape(arr.shape)

    @staticmethod
    def _wrap_lon(lon):
        """Wrap longitude(s) into [-180, 180)."""
        return np.mod(lon + 180.0, 360.0) - 180.0

    def _interp_iono_ht(self, time):
        """Temporally interpolated shell height [km]."""
        i_0, i_1, c_0, c_1, _, _ = self._time_weights(time)
        return c_0 * self.iono_hts[i_0] + c_1 * self.iono_hts[i_1]

    # ------------------------------------------------------------------
    # Delays
    # ------------------------------------------------------------------
    def calc_stec(self, vtec, iono_ht, elev, rx_pos):
        """
        Map VTEC to STEC with the single-layer obliquity factor.

        Args:
            vtec (float): Vertical TEC [TECu]
            iono_ht (float): Shell height [km]
            elev (float): Elevation angle rx->sv [deg]
            rx_pos (list): Receiver position, ECEF [m]

        Returns:
            float: Slant TEC [TECu]
        """
        r_e = 1e-3 * Position.radiusEarth(Position(rx_pos[0], rx_pos[1], rx_pos[2]))
        cos_e = (r_e / (r_e + iono_ht)) * np.cos(np.deg2rad(elev))
        return vtec / np.sqrt(1 - cos_e ** 2)

    def calc_stec_full(self, time, rx_pos, sv_pos, elev):
        """
        STEC along the rx->sv line of sight.

        Args:
            time (datetime): Time of measurement [UTC]
            rx_pos (list): Receiver position, ECEF [m]
            sv_pos (list): Satellite position, ECEF [m]
            elev (float): Elevation angle [deg]

        Returns:
            float: Slant TEC [TECu]
        """
        self._check_los(time, elev)
        iono_ht = self._interp_iono_ht(time)
        ipp_rll = xyz_to_rll(calc_ipp(rx_pos, sv_pos, iono_ht))
        vtec = self.interpolate_vtec(time, ipp_rll[1], ipp_rll[2])
        return self.calc_stec(vtec, iono_ht, elev, rx_pos)

    def calc_iono_correction(self, time, freq, rx_pos, sv_pos, elev):
        """
        Ionospheric group path delay.

        Args:
            time (datetime): Time of measurement [UTC]
            freq (float): Signal frequency [Hz]
            rx_pos (list): Receiver position, ECEF [m]
            sv_pos (list): Satellite position, ECEF [m]
            elev (float): Elevation angle [deg]

        Returns:
            float: Group path delay [m]
        """
        return TECU_COEFF * self.calc_stec_full(time, rx_pos, sv_pos, elev) / freq ** 2

    def _check_los(self, time, elev):
        """Validate elevation and epoch coverage."""
        if elev > 90 or elev < 0:
            raise Exception('Invalid Rx/Sv position. Elevation {} not valid'.format(elev))
        if time < self.times[0] or time > self.times[-1]:
            raise Exception('Provided time ({}) out of validity epoch: {}-{}'.format(
                time, self.times[0], self.times[-1]))

class BaselineInfo(object):
    """Class to hold information that belongs to a baseline, mostly carrier phase measurements"""
    def __init__(self, datetime_array, f1):
        """Initialize the class with L1 carrier phase samples and cycle slips
        """
        self.datetime_array = datetime_array
        self.f1 = f1
        self.wavelength = const.c/f1
        self.iono_free = False
        self.weather_data = False
        self.use_cov_kernel_range = False
        self.use_cov_kernel_phase = False
        self.q_range = 0 # additive weight in quadrature for range
        self.q_range_satellite = 0 # satellite-based weight for GNSS observations (range)
        self.q_phase = 0 # additive weight in quadrature for phase
        self.q_phase_satellite = 0 # satellite-based weight for GNSS observations (phase)

    def prepare_vlbi(self, group_delays, phase_delays, grdel_err, phdel_err, group_delays_dual=[], phase_delays_dual=[]):
        """ Set up VLBI solution with L1 group and phase delays """
        self.group_delays = group_delays
        self.phase_delays = phase_delays
        self.grdel_err = grdel_err
        self.phdel_err = phdel_err
        self.range_data_idxs = np.argwhere(np.ones(len(group_delays),dtype=bool)).flatten()
        self.phase_data_idxs = np.argwhere(np.ones(len(phase_delays),dtype=bool)).flatten()
        if len(group_delays_dual)>0 and len(phase_delays_dual)>0:
            self.group_delays_dual = group_delays_dual
            self.phase_delays_dual = phase_delays_dual

    def prepare_l1_frequency(self, pr_diff, cp_diff):
        """Set up the L1 frequency observables
        """
        self.pr_diff = pr_diff
        self.cp_diff = cp_diff

    def save_phase_slips(self, slips):
        """ Save phase cycle slip epochs"""
        self.slips = slips

    def save_range_idxs(self):
        """ Save range data idxs for later plotting against phase-only solution"""
        self.range_only_idxs = self.range_data_idxs

    def prepare_dual_frequency(self, cp_dual, f2):
        """Set up the dual frequency samples with differenced L2/L5 carrier phase
        """
        self.cp_dual = cp_dual
        self.f2 = f2
        self.iono_free = True
        self.cp_model_combination = []
        self.widelane_correction =[]

    def save_vlbi_model(self, group_delay_model, phase_delay_model=[], phase_delay_model_dual=[]):
        """Save the modeled group and phase delays for comparison to data
        """
        self.group_delay_model = group_delay_model
        if len(phase_delay_model)>0:
            self.phase_delay_model = np.array(phase_delay_model)
        if len(phase_delay_model_dual)>0:
            self.phase_delay_model_dual = np.array(phase_delay_model_dual)

    def save_weather(self, P1, H1, T1, P2, H2, T2):
        """Save pressure, humidity, and temperature values for participating antennas
        """
        self.weather_data = True
        self.P1 = P1
        self.H1 = H1
        self.T1 = T1
        self.P2 = P2
        self.H2 = H2
        self.T2 = T2

    def save_cpw(self, cpw1_arr, cpw2_arr):
        """Save the modeled group and phase delays for comparison to data
        """
        self.cpw_1 = np.array(cpw1_arr)
        self.cpw_2 = np.array(cpw2_arr)
        self.cpw_diff = self.cpw_2-self.cpw_1

    def combination_measurement(self, combination_type='WL'):
        """Prepare a combination carrier phase measurement, options are WL -- wide-lane, IF -- ionosphere-free
        """
        if combination_type == 'WL':
           self.cp_combination = 1/(self.f1-self.f2)*(self.f1*self.cp_diff-self.f2*self.cp_dual)
           self.comb_wavelength = const.c/(self.f1-self.f2)
           self.combination_type = 'WL'
        elif combination_type == 'IF':
           self.cp_combination = 1/(self.f1**2-self.f2**2)*(self.f1**2*self.cp_diff-self.f2**2*self.cp_dual) 
           # remove one ambiguity from the widelane solution
           if len(self.narrowlane_correction)>0: self.cp_combination = self.cp_combination + self.narrowlane_correction
           self.comb_wavelength = const.c/(self.f1+self.f2)
           self.combination_type = 'IF'
    
    def resolve_widelane_amb(self, fixed_amb):
        """Hold a final widelane ambiguity correction to be applied to the IF combination 
        """
        if self.combination_type == 'WL':
            self.narrowlane_correction = np.zeros_like(self.cp_combination)
            # ambiguity in IF sol. is lambda_N*(N_1 + lambda_W/lambda_2*N_W)
            # where N_1 : L1 integer amb., lambda_W : wide lane wavelength,
            # lambda_2: L2 wavelength, N_W: wide lane integer amb.
            # thus we remove lambda_N*lambda_W/lambda_2 * N_W = corr_wavelength * N_W
            corr_wavelength = const.c*self.f2/(self.f1**2-self.f2**2)  
            for idx, slip_slice in enumerate(self.slip_slices_arr):
                self.narrowlane_correction[slip_slice] = -corr_wavelength*fixed_amb[idx]
        else:
            raise ValueError('Combination type must be WL for resolve_widelane_amb')

    def combination_model(self, cp_model, cp_model_dual, combination_type='WL'):
        """Prepare a combination carrier phase model from inputs, options are WL -- wide-lane, IF -- ionosphere-free
           Store model for quicker access when calculating Jacobian
        """
        if ~np.all(self.cp_model == cp_model) or self.combination_type!=combination_type or len(self.cp_model_combination) == 0:
            if combination_type == 'WL':
               self.cp_model_combination = 1/(self.f1-self.f2)*(self.f1*cp_model-self.f2*cp_model_dual)
            elif combination_type == 'IF':
               self.cp_model_combination = 1/(self.f1**2-self.f2**2)*(self.f1**2*cp_model-self.f2**2*cp_model_dual) 

        return self.cp_model_combination

    def hold_slip_slices(self, slip_slices_arr, n_amb_state):
        """Store the array of indices of measurements to which each integer ambiguity applies
        """
        self.slip_slices_arr = slip_slices_arr
        self.n_amb_state = n_amb_state
 
    def hold_range_idxs(self, range_data_idxs):
        """Save the indices of valid pseudorange data
        """
        self.range_data_idxs = range_data_idxs

    def hold_covariance_kernels(self, kernels, phase=False):
        """Hold the covariance kernel dictionaries for use in building covariance matrix
        """
        if phase is True:
            self.kernels_phase = kernels
            self.use_cov_kernel_phase = True
        else:
            self.kernels_range = kernels
            self.use_cov_kernel_range = True

    def build_covariance_matrix(self, obs_type, X, variables):
        """
        Build the covariance matrix using kernel functions.
    
        Parameters:
        - datetime_array: np.ndarray, observation times (np.datetime64).
        - X: np.ndarray, variable values for each observation (n_variables-1 X N_obs).
        - variables: np.ndarray, names of variables ('noise' can be included LAST here without a corresponding entry in X).
    
        Returns:
        - K: np.ndarray, covariance matrix.
        """

        X_scaled = np.zeros_like(X)
        for kdx, variable in enumerate(variables):
            if obs_type == 'range':
                kernel = self.kernels_range[variable]
            elif obs_type == 'phase':
                kernel = self.kernels_phase[variable]
            if kernel['type'] != 'white':
                X_scaled[kdx,:] = X[kdx,:]/kernel['scaling_std']

        K = np.zeros((X.shape[1], X.shape[1]))
        for idx in range(X.shape[1]):
            for jdx in range(X.shape[1]):
                for kdx, variable in enumerate(variables):
                    if obs_type == 'range':
                        kernel = self.kernels_range[variable]
                    elif obs_type == 'phase':
                        kernel = self.kernels_phase[variable]
                    if kernel['type'] == 'periodic':
                        K[idx, jdx] += kernel['amplitude']**2*kernel_periodic(X_scaled[kdx,idx]-X_scaled[kdx,jdx], kernel['length_scale'])
                    elif kernel['type'] == 'sq-exp':
                        K[idx, jdx] += kernel['amplitude']**2*kernel_sqexp(X_scaled[kdx,idx]-X_scaled[kdx,jdx], kernel['length_scale'])
                    elif kernel['type'] == 'white':
                        if idx == jdx:
                            K[idx, jdx] += kernel['amplitude']**2

        return K

    def get_phase_idxs(self, iono_free):
        """Save the indices of valid phase data
        """
        if iono_free is True:
            self.phase_data_idxs = np.ndarray.flatten(np.argwhere(~np.isnan(self.cp_combination)))
        else:
            self.phase_data_idxs = np.ndarray.flatten(np.argwhere(~np.isnan(self.cp_diff)))

    def trim_phase_idxs(self, len_arr):
        """Trim the indices of valid phase data to regions without frequent cycle slipping
        """
        small_slip_idxs = []
        for slip_idx, slip_slice in enumerate(self.slip_slices_arr):
            if len_arr[slip_idx] < MIN_SLICE:
                for idx in slip_slice:
                    loc_idx = np.argwhere(self.phase_data_idxs == idx)
                    self.phase_data_idxs = np.delete(self.phase_data_idxs,loc_idx)
                small_slip_idxs.append(slip_idx)
        # remove the too-small slip indices from the slip slices, no need to process anymore
        self.slip_slices_arr = [self.slip_slices_arr[idx] for idx in range(len(self.slip_slices_arr)) if idx not in small_slip_idxs]

class AntennaInfo(object):
    """Class to hold information that belongs to a specific receiver/antenna"""
    def __init__(self, antennaName, rxposRinex, antennaType, bulkClock, clockRate=0, ditherPhase=False, tropModel=None):
        """Initialize the class with basic RINEX info
        """
        self.antenna_name = antennaName
        self.ref_pos = rxposRinex
        self.antenna_type = antennaType
        self.bulk_clock = bulkClock
        self.clock_rate = clockRate
        self.tropModel = tropModel
        self.dither_phase = ditherPhase
        self.is_VLBI = False
        self.estimate_ao = False
        self.estimate_grav_def = False
        self.estimate_trop = False
        self.cable_cal_active = False
        self.weather_cal_active = False
        self.ppp_clock_active = False
        self.offset_NEU = None
        self.l4r_name = None
        self.use_zwd_file = False

        # currently hard-code a priori PSD scalings
        #if self.antenna_name == 'DBR205':
        #    # increase by 2 orders of magnitude (rubidium)
        #    self.clock_psd_irw = 3451000 # cm^2/day^3 
        #    self.clock_psd_rw = 210000 # cm^2/day 
        #else:
        #    self.clock_psd_irw = 34.51  # cm^2/day^3 -- for VLBA H maser
        #    self.clock_psd_rw = 0.21  # cm^2/day -- for VLBA H maser
        
        self.clock_psd_irw = 0.078  # ps^2/hr^3 -- for VLBA H maser
        self.clock_psd_rw = 3.45  # ps^2/hr -- for VLBA H maser
        self.trop_psd_rw = 10 # ps^2/hr

        self.range_clock_idxs = []
        self.phase_clock_idxs = []
        self.trop_idxs = []
        self.linked_clocks = []

        if len(self.antenna_name)>=4:
            self.sta_code = self.antenna_name[:4]
        else:
            self.sta_code = self.antenna_name.ljust(4)
        if len(self.antenna_name)>=9:
            self.domes_name = self.antenna_name[:9]
        else:
            self.domes_name = self.antenna_name.ljust(9)

    def get_pr_data(self, iono_free = False, iono_freq = 'L2'):
        """Generate the pseudorange data series to be used in the least-squares estimation"""
        # currently hardcoding GPS freqs
        if iono_free is True:
            freq1 = 'G01' # GPS L1
            f1 = 1575.42
            pr_L1 = np.array(self.antenna_data.C1.values, dtype=float)

            if iono_freq == 'L2':
                freq2 = 'G02' # GPS L2
                f2 = 1227.60
                pr_dual = np.array(self.antenna_data.P2.values, dtype=float)
            elif iono_freq == 'L5':
                freq2 = 'G05'
                f2 = 1176.45
                pr_dual = np.array(self.antenna_data.C5.values, dtype=float)

            gamma = f1**2 / f2**2
            pr = (pr_dual - gamma*pr_L1)/(1 - gamma)
        else:
            pref_order = ['C1L', 'C1S', 'C1X', 'C1P', 'C1W', 'C1']
            pr = None
            for var in pref_order:
                if var in self.antenna_data:      # skip if not present
                    da = self.antenna_data[var].where(self.antenna_data[var] != 0)
                    da = da.dropna(dim='time', how='all')
                    pr = da if pr is None else pr.combine_first(da)

            #pr_vars = [var for var in self.antenna_data.data_vars if var.startswith('C1')]
            #pr_arrays = [self.antenna_data[pr_var] for pr_var in pr_vars]
            #stacked_pr = np.stack(pr_arrays)
            #pr = np.nanmax(stacked_pr, axis=0)
        
        pr_xarray = xr.DataArray(pr, coords={'time': pr.time.values}, dims='time')
        self.antenna_data = self.antenna_data.assign({'pr_data': pr_xarray})
        self.antenna_data = self.antenna_data.sel(time=pr.time.values) # avoid nan epochd
        
        # get formal errors
        SNR_vars = [var for var in self.antenna_data.data_vars if var.startswith('S')]
        SNR_arrays = [self.antenna_data[SNR_var] for SNR_var in SNR_vars]
        stacked = np.stack(SNR_arrays)
        # Use np.nanmax to pick the non-NaN values across the stacked arrays
        self.SNR_array = np.nanmax(stacked, axis=0)
        C_i = 1000 # m^2-Hz
        #C_i = C_i*20 # empirical
        self.pr_errors = C_i*10**(-self.SNR_array/10)

    def set_sinex_names(self, sta_code, domes_name):
        """ Set SINEX information for write_SINEX()
            need a DOMES ID for the station and a 4 length alphanumeric station code
        """
        self.sta_code = sta_code
        self.domes_name = domes_name

    def get_cp_data(self, iono_free = False, iono_freq = 'L2'):
        """Generate the carrier phase data series to be used in the least-squares estimation"""

        #if self.dither_phase is True:
        #    # add randomized integer wavelengths to phase
        #    SEED= int(sha256(self.antenna_name.encode('utf-8')).hexdigest(),16) % (2**32) # change this to change the random selection
        #    rng = np.random.default_rng(seed=SEED)
        #    MAX_N=4 # number of wavelengths that can be added or subtracted
        #    rand_int = rng.integers(-MAX_N, MAX_N, size=len(self.antenna_data.L1.values))

        # currently hardcoding GPS freqs
        if iono_free is True:
            f1 = 1575.42*1e6
            cp = self.antenna_data.L1.values

            if iono_freq == 'L2':
                f2 = 1227.60*1e6
                cp_dual = self.antenna_data.L2.values
            elif iono_freq == 'L5':
                f2 = 1176.45*1e6
                cp_dual = self.antenna_data.L5.values

            wavelength_1 = const.c/f1
            wavelength_2 = const.c/f2

            if self.dither_phase is True:
                cp = cp + rand_int

                # dual frequency dithering
                if iono_freq == 'L2':
                    rand_int_dual = rng.integers(-MAX_N, MAX_N, size=len(self.antenna_data.L2.values))
                elif iono_freq == 'L5':
                    rand_int_dual = rng.integers(-MAX_N, MAX_N, size=len(self.antenna_data.L5.values))
                cp_dual = cp_dual + rand_int_dual

            cp_xarray = xr.DataArray(cp*wavelength_1, coords={'time': self.antenna_data.time.values}, dims='time')
            cp_dual_xarray = xr.DataArray(cp_dual*wavelength_2, coords={'time': self.antenna_data.time.values}, dims='time')
            self.antenna_data = self.antenna_data.assign({'cp_dual': cp_dual_xarray})
        else:
            f1 = 1575.42*1e6
            wavelength = const.c/f1

            pref_order = ['L1L', 'L1S', 'L1X', 'L1P', 'L1W', 'L1']
            cp = None
            for var in pref_order:
                if var in self.antenna_data:                    # skip if not present
                    da = self.antenna_data[var].where(self.antenna_data[var] != 0)
                    da = da.dropna(dim='time', how='all')
                    cp = da if cp is None else cp.combine_first(da)

            #cp_vars = [var for var in self.antenna_data.data_vars if var.startswith('L1')]
            #cp_arrays = [self.antenna_data[cp_var] for cp_var in cp_vars]
            #stacked_cp = np.stack(cp_arrays)
            #cp = np.nanmax(stacked_cp, axis=0)
            #cp = self.antenna_data.L1.values

            if self.dither_phase is True:
                cp = cp + rand_int
            cp_xarray = xr.DataArray(cp*wavelength, coords={'time': cp.time.values}, dims='time')
        
        self.antenna_data = self.antenna_data.assign({'cp_data': cp_xarray})
        self.antenna_data = self.antenna_data.sel(time=cp.time.values) # avoid nan epochd
        C_i = 0.244 # m^2-Hz
        C_i = C_i*100 # empirical
        self.cp_errors = C_i*10**(-self.SNR_array/10)
        self.set_phase_clock()

    def set_phase_clock(self):
        """ initialize the phase clock start and end """
        self.phase_clock_start = self.times_gps[0]
        self.phase_clock_end = self.times_gps[-1]       

    def update_pos_series(self, pos_series, Rotate_obj, rxpos):
        """Hold a position series corrected for tides and the reference position it was generated with"""
        self.pos_series = pos_series
        self.rxpos_tides = rxpos
        self.Rotate_obj = Rotate_obj
        R_mat = np.zeros((3,3))
        for ldx in range(3):
            for mdx in range(3):
                R_mat[ldx,mdx] = self.Rotate_obj.get_value(ldx,mdx)
        self.R_mat = np.transpose(R_mat) # matrix from NEU to XYZ (ECEF)

    def update_rxpos(self, rxpos):
        """Update a position series for a slightly changed rxpos"""
        self.pos_series = np.add(self.pos_series, rxpos-self.rxpos_tides)
        self.rxpos_tides = rxpos

    def calc_VLBI_mount_vec(self):
        # transform VLBA antenna mount vector to XYZ from NEU
        if self.antenna_type != 'Equa':
            antenna_mount_vec = self.R_mat.dot(self.antenna_mount_NEU)
        else:
            # equatorial mount -- a=z_trf
            antenna_mount_vec = np.array([0,0,1])
        return antenna_mount_vec

    def set_VLBI(self, axis_offset, point_ra_dec_array, datetime_array_full, estimate_ao, estimate_grav_def):
        """Set the antenna as a VLBI antenna and hold the scalar axis offset.
           see vtd_alg.pdf in the VTD doc (astrogeo.org) for model details
        """
        self.axis_offset = axis_offset
        self.is_VLBI = True
      
        refPos = Position(self.ref_pos[0], self.ref_pos[1], self.ref_pos[2])
        R_obj = northEastUpGeodetic(refPos)
        R_mat = np.zeros((3,3))
        for ldx in range(3):
            for mdx in range(3):
                R_mat[ldx,mdx] = R_obj.get_value(ldx,mdx)
        self.R_mat = np.transpose(R_mat)

        if self.antenna_type == 'Az-El' or self.antenna_type == 'BWG' or self.antenna_type == 'Nasmyth':
            self.antenna_mount_NEU = np.array([0,0,1]) # azimuthal mount, see vtd_alg.pdf
        elif self.antenna_type == 'XY-N':
            self.antenna_mount_NEU = np.array([1,0,0]) # fixed axis in north dir., see vtd_alg.pdf
        elif self.antenna_type == 'XY-E':
            self.antenna_mount_NEU = np.array([0,1,0]) # fixed axis in WEST (HOBART) dir., see vtd_alg.pdf
        else:
            raise ValueError(f'unrecognized mount type {self.antenna_type}')
        self.estimate_ao = estimate_ao 
        self.estimate_grav_def = estimate_grav_def
        if self.estimate_grav_def is True:
            # initialize a*sin(e) + b*cos(e) gravitational deformation model
            self.grav_def_model = np.zeros(2) 

        if point_ra_dec_array is not None and len(point_ra_dec_array)>0:
            self.point_ra_dec_dict = {}
            for idx, time in enumerate(datetime_array_full):
                self.point_ra_dec_dict[time] = point_ra_dec_array[idx]
        else:
            self.point_ra_dec_dict = None
        
        self.thermal_model = False

    def save_thermal_coeffs(self, thermal_coeffs):
        """ Save Nothnagel (2009) thermal deformation coefficients
        """
        self.thermal_model = True
        self.T0 = thermal_coeffs[0]
        self.h_f = thermal_coeffs[6]
        self.h_p = thermal_coeffs[9]
        self.gamma_f = thermal_coeffs[8]
        self.gamma_a = thermal_coeffs[10]
        self.h_v = thermal_coeffs[13]
        self.h_s = thermal_coeffs[15]
        if thermal_coeffs[1] == 'FO_PRIM':
            self.F_a = 0.9
        elif thermal_coeffs[1] == 'FO_SECN':
            self.F_a = 1.8

    def save_offset(self, offset_NEU):
        """ Save NEU monument offset (m)
        """
        self.offset_NEU = offset_NEU

    def calculate_thermal_deformation_delay(self, fb_vec, s_vec, elev, Temp):
        """ Calculate the delay in meters from thermal deformation of the antenna structure """
        s_e = np.sin(np.radians(elev))
        dtau_therm = self.gamma_f*self.h_f*s_e*(Temp-self.T0) \
                + self.gamma_a * (self.h_p*s_e + np.dot(fb_vec,s_vec) + self.h_v - self.F_a*self.h_s) * (Temp-self.T0)
        return dtau_therm

    #def load_cable_cal_file(self, file_path):
    #    """Read cable calibration file and store datetime and values internally."""
    #    mjd_list = []
    #    cal_list = []

    #    with open(file_path, 'r') as f:
    #        for line in f:
    #            if line.startswith('#') or not line.strip():
    #                continue
    #            parts = line.split()
    #            mjd = float(parts[1])
    #            cal = float(parts[3])
    #            mjd_list.append(mjd)
    #            cal_list.append(cal)

    #    # Convert MJD to datetime64[ns] and apply UTC to GPS time correction (+18s)
    #    mjd_epoch = np.datetime64('1858-11-17T00:00:00')
    #    dt64_utc = mjd_epoch + np.array(mjd_list) * np.timedelta64(86400, 's')
    #    dt64_gps = dt64_utc + np.timedelta64(18, 's') # ASSUME UTC2GPS is +18
    #    self._cable_cal_times = dt64_gps.astype('datetime64[ns]')
    #    self._cable_cal_ref_epoch = self._cable_cal_times[0]
    #    self._cable_cal_values = np.array(cal_list)
    #    self.cable_cal_active = True

    def load_cable_cal_file(self, file_path):
        """Read cable-cal file, drop outliers, and create a smoothed series."""
        mjd_list, cal_list = [], []
    
        with open(file_path, "r") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                mjd_list.append(float(parts[1]))
                cal_list.append(float(parts[3]))
    
        mjd_arr = np.asarray(mjd_list, dtype=float)
        cal_arr = np.asarray(cal_list, dtype=float)
        mjd_epoch = np.datetime64("1858-11-17T00:00:00")

        dt64_utc_full  = mjd_epoch + mjd_arr * np.timedelta64(86400, "s")
        dt64_gps_full  = dt64_utc_full + np.timedelta64(18, "s")           # UTC-GPS shift
        times_full = dt64_gps_full.astype("datetime64[ns]")
    
        # ------------------------------------------------------------------
        # 1.  Robust outlier rejection (running median + MAD threshold)
        # ------------------------------------------------------------------
        median_local = medfilt(cal_arr, kernel_size=11)          # ~11-sample window
        residual     = cal_arr - median_local
        sigma_mad    = mad(residual) or residual.std()           # fallback if MAD=0
        inliers      = np.abs(residual) <= 4.0 * sigma_mad       # 4-σ cut (tune)
    
        mjd_arr_clean = mjd_arr[inliers]
        cal_arr_clean = cal_arr[inliers]
    
        # ------------------------------------------------------------------
        # 2.  Smooth with Savitzky–Golay 
        # ------------------------------------------------------------------
        # Window must be odd and > polyorder; cap at 51 points or (N//2)*2+1
        N          = len(cal_arr_clean)
        win_length = min(51, (N // 2) * 2 + 1)
        cal_smooth = savgol_filter(cal_arr_clean, window_length=win_length,
                                   polyorder=3, mode="interp")
    
        # ------------------------------------------------------------------
        # 3.  Convert MJD → datetime64[ns] (GPS time) and store
        # ------------------------------------------------------------------
        dt64_utc  = mjd_epoch + mjd_arr_clean * np.timedelta64(86400, "s")
        dt64_gps  = dt64_utc + np.timedelta64(18, "s")           # UTC-GPS shift
    
        #self._cable_cal_times = dt64_gps.astype("datetime64[ns]")
        self._cable_cal_times =  times_full
        #self._cable_cal_values = cal_smooth              # cleaned + smoothed
        #self._cable_cal_values = cal_arr_clean              # cleaned + smoothed
        self._cable_cal_values = cal_arr
    
        self._cable_cal_ref_epoch = self._cable_cal_times[0]
        self.cable_cal_active     = True
        
        #plt.figure()
        #times_plot = (self._cable_cal_times-self._cable_cal_times[0])/np.timedelta64(1,'s')
        #times_plot_full = (times_full-times_full[0])/np.timedelta64(1,'s')
        #plt.plot(times_plot_full, cal_arr, label='original')
        #plt.plot(times_plot, cal_arr_clean, label='outliers rejected')
        ##plt.plot(times_plot, cal_smooth, label='smoothed')
        #plt.xlabel('time (sec)')
        #plt.ylabel('cable cal (ps)')
        #plt.legend()
        #plt.savefig('cable_cal' + self.antenna_name+'.png')
        #plt.close()

    def load_zwd_file(self, logfile):
        """Read precisepos log file, get ZWD + gradients."""
        # read file
        with open(logfile) as lf:
            outlines = lf.readlines()
        # store meta-info and KFstate stuff
        Run = '-1'
        gps_epoch = np.datetime64('1980-01-06T00:00:00', 'ns')
        In4Lab = {}
        Lab4In = {}
        KFNames = []
        #  NB also add SIGCGPS SIGCGLO etc SIGRZTD SIGTRGN SIGTRGE (as clocks and trop states exist)
        extraStates = [ 'POSN', 'POSE', 'POSU', 'SIGN', 'SIGE', 'SIGU', 'SIGX', 'SIGY', 'SIGZ']
        clockStates = []
        tropStates = []
        biasStates = []
        posStates = ['POSX', 'POSY', 'POSZ']
        nextline_pos = False
        # list of selected passes (= selected bias states w/o the ^B)
        passes = []
        satsfound = []
        trop_list = []
        pos_list = []
        for line in outlines:
            line = line.rstrip("\r\n")
            F = line.split()
        
            # get options
            if re.match(r'.* \(--dump\) : .*',line):
                dumps = F[-1]
                continue
        
            # catch early failure
            if re.match(r'^procppp is terminating:',line):
                print("Early termination in file {}: {}".format(logfile.name,line.replace("procppp is terminating: ","")))
                sys.exit(1)
        
            # data rate
            if re.match(r'.* The nominal data time step .*',line):
                datarate = F[6]
                ephrate = F[13]
                continue
        
            # Begin time
            if re.match(r'.* Beg: .*',line):
                bweek = F[1]
                bsow = F[2]
                continue
        
            # End time
            if re.match(r'.* End: .*',line):
                eweek = F[1]
                esow = F[2]
                continue
        
            # residual editing limits
            if re.match(r'.* Pre-fit: .*',line):
                prefit = F[-1]
                continue
            if re.match(r'.* Post-fit: .*',line):
                postfit = F[-1]
                continue
        
            #----------------------------------
            # Kalman state namelist - break the loop if no KF or POST output
            if re.match(r'State namelist.*',line):
                line = line.replace(' /','')
                line = line.replace('is ','run '+Run)
                F = line.split()
        
                # get indexes in state vector for each element
                # "State namelist(91) is  POSX POSY POSZ CGPS CGAL RZTD TRGN TRGE B001G02 B002G06 ..."
                # Note KNL 2 0 -0.000   POSX   POSY   POSZ   CGPS   CGAL   RZTD   TRGN   TRGE   B001G07 etc
                # KFNames has this and more:
                # = "KMU" Run N time <all of state> dN dE dU sigN sigE sigU sigX sigY sigZ sigClk... sigRZTD sigGradNS sigGradEW
                # =                                |<- this much is parallel to KNL
                # add clocks/systems and trop to state vector
                KFNames = F
                for i in range(7,len(KFNames)):
                    if re.match(r'B.*',KFNames[i]):
                        biasStates.append(KFNames[i])
                    elif re.match(r'C.*',KFNames[i]):
                        extraStates.append('SIG'+KFNames[i])
                        clockStates.append(KFNames[i])
                    elif re.match(r'RZTD',KFNames[i]) or re.match(r'TRG.',KFNames[i]):
                        extraStates.append('SIG'+KFNames[i])
                        tropStates.append(KFNames[i])
                KFNames.extend(extraStates)
        
                for i in range(4,len(KFNames)):
                    In4Lab[KFNames[i]] = i
                    Lab4In[i] = KFNames[i]
               
            #----------------------------------
            # print meta-info and other info
            if re.match(r'End input Kalman configuration.',line):
                # fill passes and satsfound here
                for name in KFNames:
                    if re.match('B.*',name):
                        passes.append(name[1:])
                        #print("add pass "+name[1:])
                        s = name[4:]
                        if s not in satsfound: satsfound.append(s)
        
            #END if re.match(r'End input Kalman configuration.',line):
        
            # Kalman Filter output --------------------------------------------------------------
            if re.match(r'KNL .*',line):
                F = line.split()
                Run = F[1]                 # run number in this KNL

            if re.match(r'FINALSOL ECEF XYZ position .*',line):
                nextline_pos = True
                continue

            if nextline_pos:
                F = line.split()
                pos_final = np.array(F[1:], dtype=float)
                nextline_pos = False
        
            # NB KMU = 'KMU' N time <all of state> dN dE dU sigN sigE sigU sigX sigY sigZ sigClk... sigRZTD sigGradNS sigGradEW
            # NB                                  |<- this much is same as KNL  ... whole things parallels KFNames
            if re.match(r'KSU .*',line) and Run == '2':
                F = line.split()
                # time = seconds from start, or seconds of GPSweek week
                dt = F[3]
                dt = '{:.2f}'.format(float(F[3])+float(bsow))
        
                trop_data = [dt]
                for trop in tropStates:
                    trop_data.append(F[In4Lab[trop]])
                    #trop_data.append(F[In4Lab['SIG'+trop]])
                trop_list.append(trop_data)
                continue

        # process trop list
        trop_arr = np.array(trop_list, dtype = float)
        trop_arr = np.flipud(trop_arr)
        dt_sec = int(bweek) * 7 * 24 * 3600 + float(bsow)  # total seconds since GPS epoch
        self.zwd_ref_epoch = gps_epoch + np.timedelta64(int(dt_sec * 1e9), 'ns')
        self.zwd_times = trop_arr[:,0] - float(bsow)
        self.zwd_data = trop_arr[:,1:]
        self.trop_states = tropStates
        self.ref_pos_zwd = Position(pos_final)
        self.use_zwd_file = True

    def interp_zwd_file(self, query_times, rxpos, elevation, azimuth):
        """Interpolate precisepos ZWD + gradients to epoch"""
        query_times = np.asarray(query_times).astype('datetime64[ns]')
        query_times = np.atleast_1d(query_times)
        query_secs = (query_times - self.zwd_ref_epoch) / np.timedelta64(1, 's')
        zwd_map_factor = self._get_zwd_mapping(Position(rxpos))
        trop_corr = np.zeros(len(query_times))
        if len(self.trop_states)>1:
            e_rad = np.deg2rad(elevation)
            a_rad = np.deg2rad(azimuth)
            C_w = 0.0007 
            mfg_w = 1/(np.sin(e_rad)*np.tan(e_rad)+C_w)
        for idx, state in enumerate(self.trop_states):
            interp_vals = np.interp(query_secs, self.zwd_times, self.zwd_data[:,idx])
            if state == 'RZTD':
                trop_corr += interp_vals*zwd_map_factor
            elif state == 'TRGN':
                trop_corr += mfg_w*interp_vals*np.cos(a_rad)
            elif state == 'TRGE':
                trop_corr += mfg_w*interp_vals*np.sin(a_rad)

        return trop_corr

    def _get_zwd_mapping(self, rxpos):
        """ correct for ZWD height difference """
        lat_deg = rxpos.geodeticLatitude()
        h_ell = rxpos.height()
        # height correction for the zhd and zwd
        phi = np.deg2rad(lat_deg)
        cos2phi = np.cos(2.0*phi)
        H0 = float(self.ref_pos_zwd.height())  # meters
        return np.exp(-(h_ell - H0)/2000.0)

    def _get_grad_mapping(self, rxpos):
        """ correct for ZWD height difference """
        lat_deg = Position(rxpos[0], rxpos[1], rxpos[2]).geodeticLatitude()
        # height correction for the zhd and zwd
        phi = np.deg2rad(lat_deg)
        cos2phi = np.cos(2.0*phi)
        H0 = float(self.ref_pos.height())  # meters
        return np.exp(-(h_ell - H0)/2000.0)

    def interpolate_cable_cal(self, query_times):
        """Interpolate cable cal (in ps) for given datetime64[ns] timestamps."""
        if self.cable_cal_active is False:
            raise RuntimeError("Must call load_cable_cal_file() before interpolation.")

        query_times = np.asarray(query_times).astype('datetime64[ns]')
        query_secs = (query_times - self._cable_cal_ref_epoch) / np.timedelta64(1, 's')
        ref_secs = (self._cable_cal_times - self._cable_cal_ref_epoch) / np.timedelta64(1, 's')
        return np.interp(query_secs, ref_secs, self._cable_cal_values)

    def load_weather_file(self, file_path):
        """Read weather file and store relevant fields."""
        mjd_list = []
        temp_list = []
        pres_list = []
        dew_list = []

        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                mjd = float(parts[1])
                temp = float(parts[2])
                pres = float(parts[3])
                dew = float(parts[4])
                mjd_list.append(mjd)
                temp_list.append(temp)
                pres_list.append(pres)
                dew_list.append(dew)

        # Convert MJD to datetime64[ns], apply UTC → GPS shift
        mjd_epoch = np.datetime64('1858-11-17T00:00:00')
        dt64_utc = mjd_epoch + np.array(mjd_list) * np.timedelta64(1, 'D')
        dt64_gps = dt64_utc + np.timedelta64(18, 's')

        self._weather_times = dt64_gps.astype('datetime64[ns]')
        self._weather_ref_epoch = self._weather_times[0]
        self._temperature_C = np.array(temp_list)
        self._pressure_mbar = np.array(pres_list)
        self._dewpoint_C = np.array(dew_list)
        self.weather_cal_active = True

    def interpolate_weather(self, query_times):
        """Return temperature [C], pressure [mBar], and relative humidity [%] at the requested time(s)."""
        if self.weather_cal_active is False:
            raise RuntimeError("Must call load_weather_file() before interpolating.")

        query_times = np.asarray(query_times).astype('datetime64[ns]')
        t_seconds = (query_times - self._weather_ref_epoch) / np.timedelta64(1, 's')
        ref_seconds = (self._weather_times - self._weather_ref_epoch) / np.timedelta64(1, 's')

        temp = np.interp(t_seconds, ref_seconds, self._temperature_C)
        pres = np.interp(t_seconds, ref_seconds, self._pressure_mbar)
        dew  = np.interp(t_seconds, ref_seconds, self._dewpoint_C)
        rh   = self._compute_relative_humidity(temp, dew)

        return temp, pres, rh

    def load_ppp_clock(self, clock_file):
        """ Prepare a PPP clock function """ 
        # load clock and build interpolator function
        ppp_data = np.loadtxt(clock_file)
        time_sec_of_day = ppp_data[:,0]
        self.time_sec_of_day = time_sec_of_day
        cb_data_m = ppp_data[:,1]
        self.cb_data_m = ppp_data[:,1]
        self.ref_time_clock = time_sec_of_day[0]
        self.clock_interp_fcn = make_interp_spline(time_sec_of_day, cb_data_m)
        #self.clock_interp_fcn = interp1d(time_sec_of_day, cb_data_m)
        self.ppp_clock_active = True
        
    def interp_ppp_clock(self, times = []):
        """ Interpolate the PPP clock function  """
        # correct reference data for PPP clock
        if len(times)>0:
            sec_of_day_data = (times - times[0].astype('datetime64[D]')).astype('timedelta64[s]').astype(float)
        else:
            sec_of_day_data = (self.times_gps - self.times_gps[0].astype('datetime64[D]')).astype('timedelta64[s]').astype(float)
        if self.ref_time_clock > 86400.0/2 and sec_of_day_data[0]<86400.0/4:
            # disagreement on starting day -- add a day for agreement (FDV2 in US008B)
            sec_of_day_data += 86400.0
        self.ppp_clock_samples = self.clock_interp_fcn(sec_of_day_data)
        if True:
            plt.figure(dpi=300)
            plt.plot(sec_of_day_data, self.ppp_clock_samples)
            plt.xlabel('sec of day')
            plt.ylabel('clock bias (m)')
            plt.savefig('ppp_clock_samples'+self.antenna_name+'.png')
            plt.close()

    @staticmethod
    def _compute_relative_humidity(temp_C, dew_C):
        """Compute relative humidity (%) from temperature and dew point (both in Celsius)."""
        # August-Roche-Magnus approximation
        a, b = 17.625, 243.04  # constants for water over liquid
        alpha_td = (a * dew_C) / (b + dew_C)
        alpha_t  = (a * temp_C) / (b + temp_C)
        rh = 100 * np.exp(alpha_td - alpha_t)
        return rh

    def set_l4r_name(self, l4r_name):
        """ Set the antenna name for L4R ionosphere correction """
        self.l4r_name = l4r_name

    def hold_clock(self, clock_samples, phase_delay=False):
        """ Hold the RINEX data belonging to this antenna"""
        if phase_delay is True: 
            self.phase_clock_samples = clock_samples
        else:
            self.clock_samples = clock_samples

    def hold_disb(self, disb, phase=False):
        """ Hold the RINEX data belonging to this antenna"""
        if phase is True: 
            self.phase_disb = disb
        else:
            self.range_disb = disb

    def hold_data(self, antenna_data):
        """ Hold the RINEX data belonging to this antenna"""
        self.antenna_data = antenna_data
        #self.time_system = antenna_data.time_system
        self.times_gps = antenna_data.time.values
    
    def hold_times(self, times_gps):
        """ Directly set the relevant times (mostly useful for VLBI) """
        self.times_gps = times_gps

    def hold_trop(self, trop_samples):
        """ Hold the sampled wet troposphere delays """
        self.trop_samples = trop_samples
 
    def hold_PCO(self, PCOData):
        """ Hold the phase center offset data for the antenna"""
        self.antenna_PCO = PCOData

    def hold_elevs(self, elev_arr):
        """ Hold the source elevations for reweighting in least squares """
        self.elev_arr = elev_arr

    def hold_azims(self, azim_arr):
        """ Hold the source azimuths for plotting, etc. """
        self.azim_arr = azim_arr
 
    def hold_range_clock_params(self, range_clock_idxs, clock_times=[]):
        """ Hold the range clock state vector indices """
        self.range_clock_idxs = range_clock_idxs
        self.clock_times = np.array(clock_times)

    def hold_phase_clock_params(self, phase_clock_start, phase_clock_end, phase_clock_idxs, phase_clock_times=[]):
        """ Hold the phase clock beginning epoch and state vector indices """
        self.phase_clock_start = phase_clock_start
        self.phase_clock_end = phase_clock_end
        self.phase_clock_idxs = phase_clock_idxs
        self.phase_clock_times = np.array(phase_clock_times)

    def hold_trop_params(self, trop_idxs, trop_times=[]):
        """ Hold the wet troposphere state vector indices """
        self.trop_idxs = trop_idxs
        self.trop_times = trop_times

    def hold_range_disb_params(self, range_disb_idxs):
        """ Hold the range disb state vector indices """
        self.range_disb_idxs = range_disb_idxs

    def hold_phase_disb_params(self, phase_disb_idxs):
        """ Hold the phase disb state vector indices """
        self.phase_disb_idxs = phase_disb_idxs

class GNSSTKStores(object):
    """Class to hold and process GNSSTK stores"""
    def __init__(self, solType, srcType, SolSys, antennaStore, oceanStore, atmStore, navStore, ionoFree=True, ionoFreq='L2', analyticalDelay=False,\
            stochastic_clock=False, stochastic_trop=False, global_linear_clock=False, global_quadratic_clock=False, estimate_disb=False, estimate_phase_disb=False):
        """Initialize the GNSSTKStores class with the store objects
        """ 
        self.sol_type = solType
        self.src_type = srcType
        self.sol_sys = SolSys
        self.antenna_store = antennaStore
        self.ocean_store = oceanStore
        self.atm_store = atmStore
        self.nav_store = navStore
        self.iono_free = ionoFree
        self.iono_freq = ionoFreq
        self.stochastic_clock = stochastic_clock
        self.stochastic_trop = stochastic_trop
        if self.stochastic_trop is True:
            self.estimate_trop = True
        else:
            self.estimate_trop = False
        self.analyticalDelay = analyticalDelay
        self.iono_comp = False
        self.iono_comp_l4r = False
        self.global_linear_clock = global_linear_clock
        self.global_quadratic_clock = global_quadratic_clock
        self.estimate_disb = estimate_disb
        self.estimate_phase_disb = estimate_phase_disb
        if self.sol_type == 'VLBI' and (self.estimate_disb is True or self.estimate_phase_disb is True):
            self.estimate_disb = False
            self.estimate_phase_disb = False
            print('Cannot estimate DISB in a VLBI solution -- deactivated')

        if self.nav_store is not None:
            self.nav_lib =  self.nav_store.nav_lib
        self.ellipsoid_model = WGS84Ellipsoid()
        self.vlbi_like = False
        self.use_obx = False
        self.quat_sats = []
        self.R_T2I = []
        self.mjd_last = 0

    def build_antenna_map(self, source_array, datetime_array_full):
        """Build the satellite antenna map dictionary.
           Arguments:
               source_array: array of GNSS satellite sources included
           Returns:
               None
        """
        self.antenna_map = {}
        self.block_map = {}
        inputPRN = True
        warning_sats = []
        for source in np.unique(source_array):
            sat_antenna = AntexData()
            success, sat_name = self.antenna_store.getSatelliteAntenna(source[0], int(source[1:]), sat_antenna, inputPRN)
            self.antenna_map[source] = sat_antenna
            
    def hold_source_array(self, source_array, datetime_array_full, duration_dict):
        """ Hold source array directly in case of VLBI sources
           Arguments:
               source_array: array of GNSS satellite sources included
           Returns:
               None
        """
        self.source_time_dict = {}
        self.duration_dict = duration_dict 
        systems = []
        for idx, time in enumerate(datetime_array_full):
            self.source_time_dict[time] = source_array[idx]
            if source_array[idx][0] not in systems:
                systems.append(source_array[idx][0])
        self.systems = np.array(systems)
        self.source_array = np.unique(source_array)
        if 'G' in self.systems:
            # set GPS preferentially as reference system
            self.ref_system = 'G'
        else:
            self.ref_system = self.systems[0]

        if self.estimate_disb == True or self.estimate_phase_disb == True:
            if len(self.systems) == 1:
                print('only 1 system -- cannot estimated disb, deactivating...')
                self.estimate_disb = False
                self.estimate_phase_disb
            else:
                self.disb_systems = self.systems[self.systems!=self.ref_system]

    def build_ionex_store(self, ionex_file):
        """ Use an ionex file to compensate for ionospheric delay"""
        self.iono_comp = True
        self.ionex_store = IonexFile.read(ionex_file)

    def build_l4r_model(self, ds_iono_l4r):
        """ Use an ionex file to compensate for ionospheric delay"""
        self.iono_comp_l4r = True
        self.ds_iono_l4r = ds_iono_l4r
        self.L4R_TOL = 15*60 # must have points within 15 minutes

    def sel_l4r_baseline(self, antenna1, antenna2):
        """ Select the relevant L4R data for baseline of antenna1 and antenna2 """
        baselines = self.ds_iono_l4r.baseline.values
        for baseline in baselines:
            if antenna1 in baseline and antenna2 in baseline:
                if antenna1 == baseline.split('-')[0]:
                    return self.ds_iono_l4r.sel(baseline=baseline).dropna(dim='time', how='all')
                elif antenna1 == baseline.split('-')[1]:
                    return -self.ds_iono_l4r.sel(baseline=baseline).dropna(dim='time', how='all')

    def find_l4r_interpable(self, times, svs, antenna1, antenna2):
        """ Use an ionex file to compensate for ionospheric delay"""
        ds_baseline = self.sel_l4r_baseline(antenna1, antenna2)
        idxs_interpable = []
        for idx, time in enumerate(times):
            sv = svs[idx]
            if sv not in ds_baseline.sv.values:
                continue
            times_sv = ds_baseline.sel(sv=sv).dropna(dim='time', how='all').time.values
            if np.any(np.abs((times_sv-time)/np.timedelta64(1, 's'))<=self.L4R_TOL):
                # check if there is data to use at this measurement
                idxs_interpable.append(idx)
        return np.array(idxs_interpable)

    def interp_l4r(self, times, svs, antenna1, antenna2):
        """Interpolate/extrapolate L4R STEC values for a baseline at given times/svs."""
        if antenna1 == antenna2: return np.zeros(len(times))
        ds_baseline = self.sel_l4r_baseline(antenna1, antenna2)
        stec_vals = []
    
        for time, sv in zip(times, svs):
            # select this SV, drop all-NaN time rows
            ds_sv = ds_baseline.sel(sv=sv).dropna(dim="time", how="all")
    
            if ds_sv.time.size == 0:
                stec_vals.append(np.nan)
                continue
    
            times_sv = ds_sv.time.values
            stec_sv = ds_sv["STEC"].values
    
            # time difference in seconds (sv time - target time)
            dt = (times_sv - time) / np.timedelta64(1, "s")
    
            # only keep points within 2*L4R_TOL
            mask = np.abs(dt) <= 2*self.L4R_TOL
            if not np.any(mask):
                stec_vals.append(np.nan)
                continue
    
            dt_rel = dt[mask]
            stec_rel = stec_sv[mask]
    
            # 1) exact match
            zero_mask = (dt_rel == 0)
            if np.any(zero_mask):
                stec_vals.append(stec_rel[zero_mask][0])
                continue
    
            n_neg = np.sum(dt_rel < 0)
            n_pos = np.sum(dt_rel > 0)
    
            # 2) interpolate between a point before and after
            if n_neg > 0 and n_pos > 0:
                # nearest point before (largest negative dt)
                neg_idx_all = np.where(dt_rel < 0)[0]
                neg_idx = neg_idx_all[np.argmax(dt_rel[neg_idx_all])]  # dt closest to 0 from below
    
                # nearest point after (smallest positive dt)
                pos_idx_all = np.where(dt_rel > 0)[0]
                pos_idx = pos_idx_all[np.argmin(dt_rel[pos_idx_all])]  # dt closest to 0 from above
    
                t1, t2 = dt_rel[neg_idx], dt_rel[pos_idx]
                y1, y2 = stec_rel[neg_idx], stec_rel[pos_idx]
    
                # linear interpolation evaluated at dt=0
                stec_interp = y1 + (0 - t1) * (y2 - y1) / (t2 - t1)
                stec_vals.append(stec_interp)
    
            # 3) extrapolate forward (all points before target, at least 2)
            elif n_neg >= 2:
                neg_idx_all = np.where(dt_rel < 0)[0]
                neg_sorted = neg_idx_all[np.argsort(dt_rel[neg_idx_all])[::-1]]
                i1, i2 = neg_sorted[:2]
    
                t1, t2 = dt_rel[i1], dt_rel[i2]
                y1, y2 = stec_rel[i1], stec_rel[i2]
    
                stec_extrap = y1 + (0 - t1) * (y2 - y1) / (t2 - t1)
                stec_vals.append(stec_extrap)
    
            # 4) extrapolate backward (all points after target, at least 2)
            elif n_pos >= 2:
                pos_idx_all = np.where(dt_rel > 0)[0]
                # sort by dt ascending (closest to 0 first)
                pos_sorted = pos_idx_all[np.argsort(dt_rel[pos_idx_all])]
                i1, i2 = pos_sorted[:2]
    
                t1, t2 = dt_rel[i1], dt_rel[i2]
                y1, y2 = stec_rel[i1], stec_rel[i2]
    
                stec_extrap = y1 + (0 - t1) * (y2 - y1) / (t2 - t1)
                stec_vals.append(stec_extrap)
    
            # 5) nearest neighbor (only one relevant point on one side)
            else:
                nn_idx = np.argmin(np.abs(dt_rel))
                stec_vals.append(stec_rel[nn_idx])
    
        return np.array(stec_vals)


    def hold_state(self, state):
        """ Hold the current state during the least-squares estimation """
        self.state = state 

    def save_exp_weather(self, trop_T, trop_P, trop_H):
        """ Save experiment--wide weather parameters """
        self.trop_T = trop_T
        self.trop_P = trop_P
        self.trop_H = trop_H

    def compute_tides(self, times_gps, rxpos, antenna_name):
        """Find the XYZ tidal deformation at times_gps for receiver at rxpos.
           Arguments:
               times_gps: nanosecond-precision numpy datetime64 array in GPS timescale
               rxpos: list of XYZ floats in meters (ECEF)
           Returns:
               rxpos_series: dim length(times_gps) x 3, numpy float64 array
        """
        times_common = date_to_common(times_gps, 'GPS')
        rxpos_series = np.zeros((len(times_gps),3))
        
        # get rotation from NEU to XYZ
        refPos = Position(rxpos[0], rxpos[1], rxpos[2])
        R_obj = northEastUpGeodetic(refPos)
        R_mat = np.zeros((3,3))
        for ldx in range(3):
            for mdx in range(3):
                R_mat[ldx,mdx] = R_obj.get_value(ldx,mdx)
        R_mat = np.transpose(R_mat)

        for jdx, time_common in enumerate(times_common):
            #dsod = datetime64_to_dsod(time_utc)
            #mjd = datetime64_to_mjd(time_utc)
            #eph_time.setMJD(mjd)
            eph_time = EphTime(time_common)
            eph_time.setTimeSystem(TimeSystem.UTC) 

            # correct for solid earth tides
            dXYZ_SET_obj = self.sol_sys.computeSolidEarthTides(refPos, eph_time)
            dXYZ_SET = np.array([dXYZ_SET_obj[0], dXYZ_SET_obj[1], dXYZ_SET_obj[2]])
        
            # correct for ocean loading
            if self.ocean_store is not None:
                dNEU_ocean_obj = self.ocean_store.computeDisplacement(antenna_name, eph_time)
                dNEU_ocean = np.array([dNEU_ocean_obj[0], dNEU_ocean_obj[1], dNEU_ocean_obj[2]])
                dXYZ_OCE = R_mat.dot(dNEU_ocean)
            else:
                dXYZ_OCE = np.array([0,0,0])

            # correct for pole tides
            dXYZ_POL_obj = self.sol_sys.computePolarTides(refPos, eph_time)
            dXYZ_POL = np.array([dXYZ_POL_obj[0], dXYZ_POL_obj[1], dXYZ_POL_obj[2]])

            # correct for atmospheric loading
            if self.atm_store is not None:
                dNEU_atm_obj = self.atm_store.computeDisplacement(antenna_name, eph_time)
                dNEU_atm = np.array([dNEU_atm_obj[0], dNEU_atm_obj[1], dNEU_atm_obj[2]])
                dXYZ_ATM = R_mat.dot(dNEU_atm)
            else:
                dXYZ_ATM = np.array([0,0,0])
            
            dXYZ_TOT = dXYZ_SET + dXYZ_OCE + dXYZ_POL + dXYZ_ATM
            rxpos_series[jdx,:] = dXYZ_TOT + rxpos

        return rxpos_series, R_obj

    def compute_azel(self, times_gps, antenna_handle):
        """ Find the observed azimuth/elevation for an observing antenna """
        azim_arr = []
        elev_arr = []
        times_common = date_to_common(times_gps, 'GPS')
        if self.src_type == 'GNSS' or antenna_handle.point_ra_dec_dict is None: 
            for idx, common_time in enumerate(times_common):
                sat_id = self.source_time_dict[times_gps[idx]]
                RSID = RinexSatID(str(sat_id))
                rxpos = antenna_handle.pos_series[idx,:]
                sat_xvt = self.nav_store.get_xvt(RSID, common_time)
                rxpos_gnsstk = Position(rxpos[0], rxpos[1], rxpos[2])
                sat_pos_gnsstk = Position(sat_xvt.x[0], sat_xvt.x[1], sat_xvt.x[2])
                elevation = rxpos_gnsstk.elevation(sat_pos_gnsstk)
                azimuth = rxpos_gnsstk.azimuth(sat_pos_gnsstk)
                elev_arr.append(elevation)
                azim_arr.append(azimuth)
        elif self.src_type == 'VLBI':
            for idx, common_time in enumerate(times_common):
                eph_time = EphTime(common_time)
                eph_time.setTimeSystem(TimeSystem.UTC)
                ra, dec = antenna_handle.point_ra_dec_dict[times_gps[idx]]
                s_ecef = self.compute_ptvec(ra, dec, eph_time)
                R_NEU = antenna_handle.R_mat.T # matrix from XYZ (ECEF) to NEU 
                s_NEU = R_NEU@s_ecef
                elevation = np.degrees(np.arcsin(s_NEU[2]))
                azimuth = np.degrees(np.arctan2(s_NEU[1], s_NEU[0]))
                if azimuth < 0: azimuth += 360
                elev_arr.append(elevation)
                azim_arr.append(azimuth)

        antenna_handle.hold_elevs(np.array(elev_arr))
        antenna_handle.hold_azims(np.array(azim_arr))

    def build_obx_store(self, obx_files_path):
        """ Read an IGS OBX file to get satellite attitude data.
            Define the spherical linear interpolation (SLERP) objects from 
            quaternion representations in the OBX file 
        """
        lines = []
        for obx_file_path in obx_files_path:
            with open(obx_file_path, 'r') as obx_file:
                lines_file = obx_file.readlines()
                lines.extend(lines_file)
        self.use_obx = True
        quaternions_by_epoch = {}
        self.slerp_sats = {}
        current_epoch = None
        
        for line in lines:
            if line.startswith('##'):
                parts = line.split()
                date_line = " ".join(parts[1:6])
                date_line = date_line + " " + parts[6][:9]
                current_epoch = datetime.datetime.strptime(str(date_line), '%Y %m %d %H %M %S.%f')
                current_epoch = np.datetime64(current_epoch)
                quaternions_by_epoch[current_epoch] = {}
            elif line.startswith(' ATT'):
                parts = line.split()
                sat_id = parts[1]
                # take only satellites that we are analyzing in single diff data
                if sat_id in self.source_array:
                    # NB - scalar last format
                    q = [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[3])]
                    quaternions_by_epoch[current_epoch][sat_id] = q
                    if sat_id not in self.quat_sats:
                        self.quat_sats.append(sat_id)
        
        quat_epochs = sorted(quaternions_by_epoch.keys())
        self.quat_ref_epoch = quat_epochs[0]
        for sat in self.quat_sats:
            epochs = []
            quaternions = []
            for epoch, sat_data in quaternions_by_epoch.items():
                if sat in sat_data:
                    epochs.append((epoch-self.quat_ref_epoch)/np.timedelta64(1, 's'))
                    quaternions.append(sat_data[sat])
            if len(quaternions)>0:
                self.slerp_sats[sat] = Slerp(epochs, rot.from_quat(quaternions))

    def interpolate_quaternion(self, target_epoch, sat_id):
        """ Interpolate the quaternion to the desired epoch, 
            return satellite orientation vectors """
        targ_sec = (target_epoch-self.quat_ref_epoch)/np.timedelta64(1, 's')
        q_interp = self.slerp_sats[sat_id](targ_sec).as_matrix()
        return q_interp

    def compute_ra_dec(self, antenna_handle, times_gps):
        """ Compute the topocentric right ascension and declination of observed satellites at times_gps
        """
        isCOM = True
        # arrays for caching results of analysis
        ra_arr = []
        dec_arr = []

        f1 = 1575.42*1e6
        freq1_ant = 'G01' # antex frequency for receiver antenna
        freq2 ='0'
        f2 = 0

        if antenna_handle.is_VLBI is False:
            # RX PCO correction
            antennaPCOData = antenna_handle.antenna_PCO
            offset_L1 = antennaPCOData.getPhaseCenterOffset(freq1_ant)
            offset_L1 = np.array([offset_L1[0],offset_L1[1],offset_L1[2]])/1e3 # convert to m

        if antenna_handle.is_VLBI is True:
            a_vec = antenna_handle.calc_VLBI_mount_vec()

        times_common = date_to_common(times_gps, 'GPS')
        for idx, common_time in enumerate(times_common):
            eph_time = EphTime(common_time)
            eph_time.setTimeSystem(TimeSystem.UTC)
            sat_id = self.source_time_dict[times_gps[idx]]
            sat_antenna = self.antenna_map[str(sat_id)]  
            RSID = RinexSatID(str(sat_id))
            rxpos = antenna_handle.pos_series[idx,:]
            system = RSID.systemString()
            if system == 'GPS': # ref frequencies for precise range
                freq1 = 'G01' # GPS L1
            elif system == 'Galileo':
                freq1 = 'E01'
            elif system == 'BeiDou':
                freq1 = 'C01' 
            
            # find satellite position, pointing vector at receive time
            sat_xvt = self.nav_store.get_xvt(RSID, common_time)
            rxpos_gnsstk = Position(rxpos[0], rxpos[1], rxpos[2])
            sat_pos_gnsstk = Position(sat_xvt.x[0], sat_xvt.x[1], sat_xvt.x[2])
            elevation = rxpos_gnsstk.elevationGeodetic(sat_pos_gnsstk)
            azimuth = rxpos_gnsstk.azimuthGeodetic(sat_pos_gnsstk)
            sat_pos = np.array([sat_xvt.x[0],sat_xvt.x[1],sat_xvt.x[2]])
            sat_vel = np.array([sat_xvt.v[0],sat_xvt.v[1],sat_xvt.v[2]])
            rx2sat = sat_pos - rxpos
            rx2sat /= np.linalg.norm(rx2sat) # normalize vector
            
            if antenna_handle.is_VLBI is True: 
                # compute the geometric model for the VLBI antenna
                if antenna_handle.point_ra_dec_dict is not None:
                    ra, dec = antenna_handle.point_ra_dec_dict[times_gps[idx]]
                    #rx2sat_test = self.compute_ptvec(ra, dec, eph_time)
                point_vec = rx2sat-a_vec*np.dot(rx2sat,a_vec)
                
                if antenna_handle.estimate_ao is True:
                    fb_vec = ao_ant*point_vec/np.linalg.norm(point_vec) 
                else:
                    fb_vec = antenna_handle.axis_offset*point_vec/np.linalg.norm(point_vec) 

                # shift rxpos by offset
                rxpos_L1 = rxpos + fb_vec

            else:
                # compute antenna PCV
                dt_therm = 0
                ROT = antenna_handle.R_mat # rotation from NEU to XYZ (ECEF)
                PCV_L1 = antennaPCOData.getPhaseCenterVariation(freq1_ant, azimuth, elevation)
                rxpos_L1 = rxpos + ROT@offset_L1 - PCV_L1*1e-3*rx2sat
 
            sat_pos_L1 = self.sat_adj_PC(freq1, sat_antenna, times_gps[idx], eph_time, sat_xvt.x, rx2sat)
            if self.iono_free:
                sat_pos_dual = self.sat_adj_PC(freq2, sat_antenna, times_gps[idx], eph_time, sat_xvt.x, rx2sat)

            pr_model, sat_xvt_corr = self.get_pr_model(RSID, common_time, times_gps[idx], \
                    sat_antenna, rxpos_L1, freq1, sat_pos_L1, sat_vel)
            sat_pos_corr = np.array([sat_xvt_corr.x[0], sat_xvt_corr.x[1], sat_xvt_corr.x[2]])
            sat_vel_corr = np.array([sat_xvt_corr.v[0], sat_xvt_corr.v[1], sat_xvt_corr.v[2]])

            satpos_gcrf, satvel_gcrf = self.get_gcrf_posvel(sat_pos_corr, sat_vel_corr, eph_time)
            rxpos_gcrf, rxvel_gcrf = self.get_gcrf_posvel(rxpos_L1, [0,0,0], eph_time)

            rho_sat = satpos_gcrf - rxpos_gcrf
            ra = (np.arctan2(rho_sat[1], rho_sat[0]) + 2*np.pi) % (2*np.pi)
            dec = np.arcsin(rho_sat[2]/np.linalg.norm(rho_sat))
            ra_arr.append(ra)
            dec_arr.append(dec)

        #if np.any(np.abs(np.diff(np.array(ra_arr)))>1): breakpoint()
        
        return np.array(ra_arr), np.array(dec_arr)

    #def adj_clock_ns(self, antenna_handle, clock_samples, phase_clock_samples, phase, phase_only):
    #    """ Adjust the time stamps to the nearest ns (max precision of datetime64), remove this accounted-for clock
    #    bias from the clock_samples and phase_clock_samples arrays """
    #    times_adj = []
    #    if phase_only is False:
    #        cb_adj = []
    #    if phase is True:
    #        cb_phase_adj = []
    #    cb_test = clock_samples.copy()
    #    for idx, time in enumerate(antenna_handle.times_gps):
    #        if self.stochastic_clock is True:
    #            if phase is True and len(phase_clock_samples) > 0 and time in antenna_handle.phase_clock_times:
    #                phase_clock_idx = np.argwhere(antenna_handle.times_gps[idx]==antenna_handle.phase_clock_times)[0][0]
    #                adj = -int(round(phase_clock_samples[phase_clock_idx]*1e9/const.c))
    #                cb_phase_adj.append(adj*const.c/1e9)
    #                if phase_only is False:
    #                    cb_adj.append(adj*const.c/1e9)
    #            elif phase_only is False and time in antenna_handle.clock_times:
    #                clock_idx = np.argwhere(antenna_handle.times_gps[idx]==antenna_handle.clock_times)[0][0]
    #                adj = -int(round(clock_samples[clock_idx]*1e9/const.c))
    #                cb_adj.append(adj*const.c/1e9)
    #            else:
    #                adj = 0
    #        else:
    #            if phase is True and len(phase_clock_samples)>0:
    #                adj = -int(round(phase_clock_samples[idx]*1e9/const.c))
    #                cb_phase_adj.append(adj*const.c/1e9)
    #                if phase_only is False:
    #                    cb_adj.append(adj*const.c/1e9)
    #            else:
    #                adj = -int(round(clock_samples[idx]*1e9/const.c))
    #                cb_adj.append(adj*const.c/1e9)
    #
    #        times_adj.append(time + np.timedelta64(adj, 'ns'))
    #
    #    if phase_only is False:
    #        clock_samples += np.array(cb_adj)
    #    if phase is True and len(phase_clock_samples)>0:
    #        phase_clock_samples += np.array(cb_phase_adj)
    #    times_adj = np.array(times_adj)
    #
    #    return times_adj, clock_samples, phase_clock_samples

    def adj_clock_ns(self, antenna_handle, clock_samples, phase_clock_samples, phase, phase_only):
        """ Adjust the time stamps to the nearest ns (max precision of datetime64), remove this accounted-for clock
        bias from the clock_samples and phase_clock_samples arrays """
        times = np.asarray(antenna_handle.times_gps).astype('datetime64[ns]')
        ns_offset = np.zeros(len(times), dtype=np.int64)           
        full_offset = np.zeros(len(times), dtype=float)           

        # ------------------------------------------------------------------
        # 1.  Build the integer‑nanosecond offsets
        # ------------------------------------------------------------------
        if self.stochastic_clock is True:
            times64 = times.view('int64')

            if phase_only is False:
                clk_times64 = antenna_handle.clock_times.astype('datetime64[ns]').view('int64')
                idx_clk     = np.searchsorted(times64, clk_times64)
                idx_clk = idx_clk[idx_clk<len(ns_offset)] # investigate
                ns_offset[idx_clk] = -np.rint(
                        clock_samples * 1e9 / const.c
                    ).astype(np.int64)
                full_offset[idx_clk] = -clock_samples/const.c

            # phase last so it overwrites the range values where present
            if phase is True and len(phase_clock_samples)>0:
                ph_times64  = antenna_handle.phase_clock_times.astype('datetime64[ns]').view('int64')
                idx_ph      = np.searchsorted(times64, ph_times64)
                idx_ph = idx_ph[idx_ph<len(ns_offset)] # investigate
                ns_offset[idx_ph] = -np.rint(
                    phase_clock_samples * 1e9 / const.c
                ).astype(np.int64)
                full_offset[idx_ph] = -phase_clock_samples/const.c
        else:
            # --- dense, arrays already match times_gps ----------------------
            if phase and len(phase_clock_samples)>0:
                ns_offset = -np.rint(phase_clock_samples * 1e9 / const.c).astype(np.int64)
                full_offset = -phase_clock_samples/const.c
            else:  # use range‑clock values
                ns_offset = -np.rint(clock_samples * 1e9 / const.c).astype(np.int64)
                full_offset = -clock_samples/const.c 

        # ------------------------------------------------------------------
        # 2.  Apply the corrections
        # ------------------------------------------------------------------
        times_adj = times + ns_offset.astype('timedelta64[ns]')
        return times_adj, full_offset

    def correct_PR_CP(self, antenna_handle, phase=False, phase_only=False, sim=False, freq=None):
        """Correct pseudorange or carrier phase data using nav ephemeris and analytical delay model.
        """
        if sim == False:
            data = antenna_handle.antenna_data

        times_gps = antenna_handle.times_gps
        clock_samples = antenna_handle.clock_samples
        if self.stochastic_clock is True:
            clock_times = antenna_handle.clock_times
        if self.estimate_trop is True:
            trop_samples = antenna_handle.trop_samples
        else:
            trop_samples = []
        if self.stochastic_trop is True:
            trop_times = antenna_handle.trop_times
        if phase is True:
            phase_clock_samples = antenna_handle.phase_clock_samples
            if self.stochastic_clock is True:
                phase_clock_times = antenna_handle.phase_clock_times
        else:
            phase_clock_samples = []

        if self.estimate_disb is True and phase_only is False:
            range_disb = antenna_handle.range_disb
            disb_sys = {}
            for sdx, rinex_sys in enumerate(self.disb_systems):
                disb_sys[rinex_sys] = range_disb[sdx]
            disb_sys[self.ref_system] = 0
        if self.estimate_phase_disb is True:
            phase_disb = antenna_handle.phase_disb
            phase_disb_sys = {}
            for sdx, rinex_sys in enumerate(self.disb_systems):
                phase_disb_sys[rinex_sys] = phase_disb[sdx]
            phase_disb_sys[self.ref_system] = 0

        isCOM = True
        # arrays for caching results of analysis
        pr_model_arr = []
        if sim is True and self.iono_free is True:
            pr_dual_arr = []

        if freq is not None:
            f1 = freq
            if f1 == 1575.42e6:
                freq1 = 'G01'
            elif f1 == 1227.60e6:
                freq1 = 'G02'
            elif freq == 1176.45e6:
                freq1 = 'G05'
            elif freq == 1207.14e6:
                freq1 = 'E07'
            elif freq == 1561.098e6:
                freq1 = 'C02'
        else:
            f1 = 1575.42*1e6
            freq1 = 'G01' # antex frequency for receiver antenna
            if self.iono_free: # iono-free only for GPS, ref frequency for precise range
                if self.iono_freq == 'L2':
                    f2 = 1227.60*1e6
                    freq2 = 'G02'
                elif self.iono_freq == 'L5': 
                    f2 = 1176.45*1e6
                    freq2 = 'G05'
                if antennaPCOData.nFreq <= 4 and self.iono_freq == 'L5':
                    freq2 = 'G02' 
            else:
                freq2 ='0'
                f2 = 0

        if antenna_handle.is_VLBI is False:
            # RX PCO correction
            antennaPCOData = antenna_handle.antenna_PCO
            offset_L1 = antennaPCOData.getPhaseCenterOffset(freq1)
            offset_L1 = np.array([offset_L1[0],offset_L1[1],offset_L1[2]])/1e3 # convert to m
            if self.iono_free is True:
                offset_dual = antennaPCOData.getPhaseCenterOffset(freq2)
                offset_dual = np.array([offset_dual[0],offset_dual[1],offset_dual[2]])/1e3 # convert to m

        if antenna_handle.offset_NEU is not None:
            offset_XYZ = antenna_handle.R_mat.dot(antenna_handle.offset_NEU)

        elev_arr = []
        azim_arr = []
        offset_arr = []
        if phase is True: cp_model_arr = [] # carrier phase-based range by epoch (changing source)
        if phase is True: # and antenna_handle.is_VLBI is False: # need to account for carrier phase windup
            cpw = {} # carrrier phase windup, dictionary by source
            cpw_arr = [] # temporary -- check
            for source in self.source_array:
                cpw[source] = 0.0 # initialize cpw to 0

            wavelength_1 = const.c/f1
            if self.iono_free:
                cp_dual_model_arr = []
                wavelength_2 = const.c/f2
      
            if antenna_handle.is_VLBI is False:
                east_triple = Position(antenna_handle.Rotate_obj.get_value(1,0),\
                                     antenna_handle.Rotate_obj.get_value(1,1),
                                     antenna_handle.Rotate_obj.get_value(1,2))
  
                north_triple = Position(antenna_handle.Rotate_obj.get_value(0,0),\
                                     antenna_handle.Rotate_obj.get_value(0,1),
                                     antenna_handle.Rotate_obj.get_value(0,2))
        
        if antenna_handle.is_VLBI is True:
            a_vec = antenna_handle.calc_VLBI_mount_vec()

        times_adj, full_offset = self.adj_clock_ns(antenna_handle, clock_samples, \
                phase_clock_samples, phase, phase_only)

        times_common = date_to_common(times_gps, 'GPS')
        for idx, common_time in enumerate(times_common):
            common_time.addSeconds(full_offset[idx])
            eph_time = EphTime(common_time)
            eph_time.setTimeSystem(TimeSystem.UTC)
            sat_id = self.source_time_dict[times_gps[idx]]
            sat_antenna = self.antenna_map[str(sat_id)]  
            RSID = RinexSatID(str(sat_id))
            rxpos = antenna_handle.pos_series[idx,:].copy()

            if antenna_handle.offset_NEU is not None:
                rxpos += offset_XYZ
            system = RSID.systemString()
            if system == 'GPS': # ref frequencies for precise range
                if f1 == 1575.42e6:
                    freq1 = 'G01'
                elif f1 == 1227.60e6:
                    freq1 = 'G02'
                elif freq == 1176.45e6:
                    freq1 = 'G05'
                if self.iono_free:
                    if self.iono_freq == 'L2':
                        freq2 = 'G02'
                    elif self.iono_freq == 'L5':
                        freq2 = 'G02' # only G02 available, no G05 for satellites
            elif system == 'Galileo':
                freq1 = 'E01'
                if f1 == 1575.42e6:
                    freq1 = 'E01'
                elif f1 == 1207.14e6:
                    freq1 = 'E07'
                elif freq == 1176.45e6:
                    freq1 = 'E05'
                if self.iono_free:
                    if self.iono_freq == 'L2':
                        raise ValueError('No L2 frequency for BeiDou -- should not be here!')
                    elif self.iono_freq == 'L5':
                        freq2 = 'E05'
            elif system == 'BeiDou':
                if f1 == 1575.42e6:
                    freq1 = 'C01'
                elif f1 == 1207.14e6:
                    freq1 = 'C07'
                elif freq == 1176.45e6:
                    freq1 = 'C05'
                elif freq == 1561.098e6:
                    freq1 = 'C02'
          
                if self.iono_free:
                    if self.iono_freq == 'L2':
                        raise ValueError('No L2 frequency for BeiDou -- should not be here!')
                    elif self.iono_freq == 'L5':
                        freq2 = 'C05'
            
            # find satellite position, pointing vector at receive time
            sat_xvt = self.nav_store.get_xvt(RSID, common_time)
            rxpos_gnsstk = Position(rxpos[0], rxpos[1], rxpos[2])
            sat_pos_gnsstk = Position(sat_xvt.x[0], sat_xvt.x[1], sat_xvt.x[2])
            elevation = rxpos_gnsstk.elevationGeodetic(sat_pos_gnsstk)
            azimuth = rxpos_gnsstk.azimuthGeodetic(sat_pos_gnsstk)
            sat_pos = np.array([sat_xvt.x[0],sat_xvt.x[1],sat_xvt.x[2]])
            sat_vel = np.array([sat_xvt.v[0],sat_xvt.v[1],sat_xvt.v[2]])
            rx2sat = sat_pos - rxpos
            rx2sat /= np.linalg.norm(rx2sat) # normalize vector
            
            if antenna_handle.is_VLBI is True: 
                # compute the geometric model for the VLBI antenna
                if antenna_handle.point_ra_dec_dict is not None:
                    ra, dec = antenna_handle.point_ra_dec_dict[times_gps[idx]]
                    #rx2sat_test = self.compute_ptvec(ra, dec, eph_time)
                point_vec = rx2sat-a_vec*np.dot(rx2sat,a_vec)
                
                if antenna_handle.estimate_ao is True:
                    fb_vec = ao_ant*point_vec/np.linalg.norm(point_vec) 
                else:
                    fb_vec = antenna_handle.axis_offset*point_vec/np.linalg.norm(point_vec) 

                # shift rxpos by offset
                rxpos_L1 = rxpos + fb_vec
                if self.iono_free:
                   rxpos_dual = rxpos + fb_vec

                if antenna_handle.thermal_model is True:
                    if antenna_handle.weather_cal_active is True:
                        T, P, H = antenna_handle.interpolate_weather(times_gps[idx])
                        dt_therm = antenna_handle.calculate_thermal_deformation_delay(fb_vec, rx2sat, elevation, T)
                    elif self.trop_T is not None:
                        Temp = self.trop_T
                        dt_therm = antenna_handle.calculate_thermal_deformation_delay(fb_vec, rx2sat, elevation, Temp)
                    else:
                        dt_therm = 0
                else:
                    dt_therm = 0
            else:
                # compute antenna PCV
                dt_therm = 0
                ROT = antenna_handle.R_mat # rotation from NEU to XYZ (ECEF)
                try: PCV_L1 = antennaPCOData.getPhaseCenterVariation(freq1, azimuth, elevation)
                except: 
                     # some antennas have only GPS PCV
                     PCV_L1 = antennaPCOData.getPhaseCenterVariation('G01', azimuth, elevation)

                rxpos_L1 = rxpos + ROT@offset_L1 - PCV_L1*1e-3*rx2sat
                if self.iono_free:
                    PCV_dual = antennaPCOData.getPhaseCenterVariation(freq2, azimuth, elevation)
                    rxpos_dual = rxpos + ROT@(offset_dual) - PCV_dual*1e-3*rx2sat
 
            try: 
                sat_pos_L1 = self.sat_adj_PC(freq1, sat_antenna, times_adj[idx], eph_time, sat_xvt.x, rx2sat)
            except:
                # PRN 03 has no G05 for some reason
                freq1 = 'G02'
                sat_pos_L1 = self.sat_adj_PC(freq1, sat_antenna, times_adj[idx], eph_time, sat_xvt.x, rx2sat)
            if self.iono_free:
                sat_pos_dual = self.sat_adj_PC(freq2, sat_antenna, times_adj[idx], eph_time, sat_xvt.x, rx2sat)

            if self.analyticalDelay is True:
                pr_model, sat_xvt_corr = self.get_pr_model(RSID, common_time, times_adj[idx], \
                        sat_antenna, rxpos_L1, freq1, sat_pos_L1, sat_vel)
            else:
                pr_model, sat_xvt_corr = self.get_pr_iter(RSID, common_time, times_adj[idx], \
                        sat_antenna, rxpos_L1, freq1, sat_pos_L1, sat_vel)

            # sat_xvt_corr will be used for satellite clock bias and troposphere correction
            if self.iono_free:
                pr_dual, _ = self.get_pr_model(RSID, common_time, times_adj[idx], \
                        sat_antenna, rxpos_dual, freq2, sat_pos_dual, sat_vel)

            # get satellite az/el
            # NB: elev, azim local not geodetic
            rxpos_gnsstk = Position(rxpos_L1[0], rxpos_L1[1], rxpos_L1[2])
            sat_pos_gnsstk = Position(sat_xvt_corr.x[0], sat_xvt_corr.x[1], sat_xvt_corr.x[2])
            elevation = rxpos_gnsstk.elevationGeodetic(sat_pos_gnsstk)
            azimuth = rxpos_gnsstk.azimuthGeodetic(sat_pos_gnsstk)

            if antenna_handle.estimate_grav_def is True:
                #grav_delay = antenna_handle.grav_def_model[0]*np.sin(np.deg2rad(elevation)) + antenna_handle.grav_def_model[1]*np.cos(np.deg2rad(elevation))
                grav_delay = antenna_handle.grav_def_model[0]*np.deg2rad(elevation) + antenna_handle.grav_def_model[1]*np.deg2rad(elevation)**2
            else:
                grav_delay = 0

            elev_arr.append(elevation)
            azim_arr.append(azimuth)
           
            # clock offset correction
            if phase_only is False:
                if self.stochastic_clock is True and times_gps[idx] in clock_times:
                    clock_idx = np.argwhere(times_gps[idx]==clock_times)[0][0]
                    clock_offset = clock_samples[clock_idx]
                elif self.stochastic_clock is False and len(clock_samples)>0:
                    clock_offset = clock_samples[idx]
                else:
                    clock_offset = 0
 
            # L1 troposphere correction
            if antenna_handle.tropModel is not None:
                if antenna_handle.weather_cal_active is True:
                    T, P, H = antenna_handle.interpolate_weather(times_gps[idx])
                    antenna_handle.tropModel.setWeather(T, P, H)
                    antenna_handle.tropModel.setHumidity(H) # needed for global trop model
                #trop_delay = antenna_handle.tropModel.correction(rxpos_gnsstk, sat_pos_gnsstk, common_time)
                try: trop_delay = antenna_handle.tropModel.correction(rxpos_gnsstk, sat_pos_gnsstk, common_time)
                except:
                    #print('RX Height too high, setting trop to 0')
                    #print('exception')
                    trop_delay = 0
                if self.stochastic_trop is True and times_gps[idx] in trop_times:
                    trop_idx = np.argwhere(times_gps[idx]==trop_times)[0][0]
                    trop_delay += trop_samples[trop_idx]*antenna_handle.tropModel.wet_mapping_function(elevation)
                elif self.stochastic_trop is False and len(trop_samples)>0:
                    trop_delay += trop_samples[idx]*antenna_handle.tropModel.wet_mapping_function(elevation)
            else:
                trop_delay = 0

            # disb correction
            if phase_only is False and self.estimate_disb is True:
                disb_delay = disb_sys[str(sat_id)[0]]
            else:
                disb_delay = 0

            sat_clk = sat_xvt_corr.clkbias*const.c # satellite clock bias in m
            
            if self.iono_comp is True:
                # compensate for IONEX ionosphere delay
                rxpos_list = [rxpos_L1[0], rxpos_L1[1], rxpos_L1[2]]
                sat_pos_list = [sat_xvt_corr.x[0], sat_xvt_corr.x[1], sat_xvt_corr.x[2]]
                timestamp = (times_gps[idx] - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
                time_datetime = datetime.datetime.utcfromtimestamp(timestamp)
                iono_delay = self.ionex_store.calc_iono_correction(time_datetime, f1, \
                        rxpos_list, sat_pos_list, elevation)
            else:
                iono_delay = 0

            # cable cal 
            if antenna_handle.cable_cal_active is True:
                cc_delay = antenna_handle.interpolate_cable_cal(times_gps[idx])*const.c/1e12 # interpolate, convert from ps to m
            else:
                cc_delay = 0

            # we need to add the modeled troposphere and clock offset to mirror the real PR measurement
            if phase_only is False:
                if self.iono_free is True and sim is False:
                    pr_final = float(IonosphereFreeRange([f1, f2], [pr_model, pr_dual])) + trop_delay + clock_offset - sat_clk + cc_delay + disb_delay
                else:
                    pr_final = pr_model + trop_delay + iono_delay + clock_offset - sat_clk - dt_therm + grav_delay + cc_delay + disb_delay
                    if self.iono_free is True:
                        pr_final_dual = pr_dual + trop_delay + iono_delay + clock_offset - sat_clk - dt_therm + grav_delay + cc_delay + disb_delay
                        pr_dual_arr.append(pr_final_dual)
                pr_model_arr.append(pr_final)

            '''
            if antenna_handle.is_VLBI is False and len(phase_clock_samples)>0:
                ## temp -- check against PreciseRange
                PR_obj = PreciseRange()
                pr = float(data_idx.pr_data.values) 
                synth_pr = PR_obj.ComputeAtTransmitTime(common_time, pr, rxpos_gnsstk, RSID, sat_antenna, freq1, freq2,\
                        self.sol_sys, self.nav_lib, isCOM, self.ellipsoid_model)
                am_pc_rx = np.dot(ROT@(offset_L1 - PCV_L1*1e-3*rx2sat),rx2sat)
                pr_pc_rx = antennaPCOData.getTotalPhaseCenterOffset(freq1_ant, azimuth, elevation)/1e3
                am_tx_pos = np.array([sat_xvt_corr.x[0], sat_xvt_corr.x[1], sat_xvt_corr.x[2]])
                pr_tx_pos = np.array([PR_obj.SatR[0],PR_obj.SatR[1],PR_obj.SatR[2]])
                dt_1 = pr_model/const.c
                pr_tx_corr= ECEF2ECI(dt_1, pr_tx_pos, np.array([0,0,0]))
                tx_diff = am_tx_pos - pr_tx_pos
                # satellite position is different -- concerning. Why is z-coord identical?
                sat_pos_test = self.sat_adj_PC(freq1, sat_antenna, eph_time, sat_xvt_corr.x, rx2sat)
                am_PCO_vec_approx = sat_pos_test-am_tx_pos
                PR_PCO_vec = np.array([PR_obj.SatPCOXYZ[0],PR_obj.SatPCOXYZ[1],PR_obj.SatPCOXYZ[2]])
                pr_PR = synth_pr + trop_delay + clock_offset
                pr_diff = pr_final-pr_PR
            '''
            if phase is True: 
                if self.stochastic_clock is True and len(phase_clock_samples) > 0 and times_gps[idx] in phase_clock_times:
                    phase_clock_idx = np.argwhere(times_gps[idx]==phase_clock_times)[0][0]
                    phase_clock_offset = phase_clock_samples[phase_clock_idx]
                elif self.stochastic_clock is False and len(phase_clock_samples) > 0:
                    phase_clock_offset = phase_clock_samples[idx]
                else:
                    phase_clock_offset = 0

                # disb correction
                if self.estimate_phase_disb is True:
                    phase_disb_delay = phase_disb_sys[str(sat_id)[0]]
                else:
                    phase_disb_delay = 0
                
                cpw_source_last = cpw[sat_id]
                Rx2Tx = Position(rx2sat[0], rx2sat[1], rx2sat[2])
                #cpw_prev = PhaseWindup(cpw_source_last, common_time, sat_pos_gnsstk, \
                #        Rx2Tx, west_triple, north_triple, self.sol_sys)
                if antenna_handle.is_VLBI:
                    if False: #antenna_handle.antenna_type == 'BWG':
                        # not needed !
                        n_vec = antenna_handle.R_mat[:,0]
                        e_vec = antenna_handle.R_mat[:,1]
                        u_vec = antenna_handle.R_mat[:,2]
                        cpw_current = self.phase_windup_VLBI_BWG(antenna_handle.antenna_name, sat_id, times_adj[idx], eph_time, sat_pos_gnsstk, \
                                rx2sat, e_vec, n_vec, u_vec)
                    else:
                        cpw_current = self.phase_windup_VLBI(sat_id, times_adj[idx], eph_time, sat_pos_gnsstk, \
                                rx2sat, a_vec)
                        if antenna_handle.antenna_type == 'Nasmyth' or antenna_handle.antenna_type == 'BWG':
                        #if antenna_handle.antenna_type == 'Nasmyth':
                            cpw_current += elevation/360
                        if antenna_handle.antenna_type == 'BWG':
                            # old method of compensation
                            cpw_current -= azimuth/360
                else:
                    cpw_current = self.phase_windup_GNSS(sat_id, times_adj[idx], eph_time, sat_pos_gnsstk, \
                            rx2sat, east_triple, north_triple)
                    west_triple = Position(-antenna_handle.Rotate_obj.get_value(1,0),-antenna_handle.Rotate_obj.get_value(1,1),-antenna_handle.Rotate_obj.get_value(1,2))
                    #cpw = PhaseWindup(cpw_source_last, common_time, sat_pos_gnsstk, Rx2Tx, west_triple, north_triple, self.sol_sys)
                cpw_current = self.cycle_adj_windup(cpw_source_last, cpw_current)

                cpw[sat_id] = cpw_current
                cpw_arr.append(cpw_current)

                cp_model = pr_model + trop_delay - iono_delay + cpw_current*wavelength_1  + phase_clock_offset - sat_clk - dt_therm + grav_delay + cc_delay + phase_disb_delay
                cp_model_arr.append(cp_model)
                
                if self.iono_free is True:
                    cp_dual_model = pr_dual + trop_delay + cpw_current*wavelength_2 + phase_clock_offset - sat_clk - dt_therm + grav_delay + cc_delay + phase_disb_delay
                    cp_dual_model_arr.append(cp_dual_model)
                #if antenna_handle.antenna_name == 'HN-VLBA' and np.abs((data.sel(time=times_gps[idx]).pr_data.values-pr_final))>200 and phase is True: breakpoint()

        #if antenna_handle.antenna_name == 'HN-VLBA': print(trop_samples)
        if sim == False:
            if phase_only is False:
                pr_xarray = xr.DataArray(pr_model_arr, coords={'time': times_gps}, dims='time')
                data = data.assign({'pr_model': pr_xarray})

            if phase is True:
                cp_xarray = xr.DataArray(cp_model_arr, coords={'time': times_gps}, dims='time')
                data = data.assign({'cp_model': cp_xarray})
                if self.iono_free is True:
                    cp_dual_model_xarray = xr.DataArray(cp_dual_model_arr, coords={'time': times_gps}, dims='time')
                    data = data.assign({'cp_dual_model': cp_dual_model_xarray})

        elev_arr = np.array(elev_arr)
        antenna_handle.hold_elevs(elev_arr)
        
        # temporary -- plot correlation
        #if False and antenna_handle.antenna_name == 'FD_VLBA':
        #if antenna_handle.antenna_name == 'PIE1' or antenna_handle.antenna_name == 'PTVB': #True:# antenna_handle.antenna_name == 'FD_VLBA':
        #    azim_arr = np.array(azim_arr)
        #    corr_el_fig = plt.figure()
        #    corr_el_ax = corr_el_fig.add_subplot(111)
        #    corr_az_fig = plt.figure()
        #    corr_az_ax = corr_az_fig.add_subplot(111)
        #    azel_fig = plt.figure()
        #    azel_ax = azel_fig.add_subplot(111)
        #    for sat in self.source_array:
        #        # get epochs of satellite
        #        if sat[0] == 'G':
        #            marker='x'
        #        elif sat[0] == 'E':
        #            marker='+'
        #        else:
        #            marker='1'
        #        idxs_sat = []
        #        for idx, time in enumerate(times_gps):
        #            if sat == self.source_time_dict[time]:
        #                idxs_sat.append(idx)
        #        idx_sat = np.array(idxs_sat)
        #        corr_el_ax.plot(elev_arr[idxs_sat], data.pr_data.values[idxs_sat]-data.pr_model.values[idxs_sat], marker=marker, linestyle='None', label=sat)
        #        corr_az_ax.plot(azim_arr[idxs_sat], data.pr_data.values[idxs_sat]-data.pr_model.values[idxs_sat], marker=marker, linestyle='None', label=sat)
        #        azel_ax.plot(azim_arr[idxs_sat], elev_arr[idxs_sat], marker=marker, linestyle='None', label=sat)
        #    corr_el_ax.set_ylabel('PR error (m, data-model)')
        #    corr_el_ax.set_xlabel('elevation (deg)')
        #    corr_az_ax.set_ylabel('PR error (m, data-model)')
        #    corr_az_ax.set_xlabel('azimuth (deg)')
        #    azel_ax.set_ylabel('elevation (deg)')
        #    azel_ax.set_xlabel('azimuth (deg)')                
        #    corr_el_fig.savefig(antenna_handle.antenna_name + 'corr_fig_el.png')
        #    corr_az_fig.savefig(antenna_handle.antenna_name + 'corr_fig_az.png')
        #    azel_fig.savefig(antenna_handle.antenna_name + 'azel_fig.png')
        #    plt.close(corr_el_fig)
        #    plt.close(corr_az_fig)
        #    plt.close(azel_fig)
        #if antenna_handle.antenna_name == 'PIE1' or antenna_handle.antenna_name == 'PTVB': #True:# antenna_handle.antenna_name == 'FD_VLBA':
        #    obs_fig = plt.figure()
        #    obs_ax = obs_fig.add_subplot(111)
        #    for sat in self.source_array:
        #        # get epochs of satellite
        #        if sat[0] == 'G':
        #            marker='x'
        #        elif sat[0] == 'E':
        #            marker='+'
        #        else:
        #            marker='1'
        #        idxs_sat = []
        #        for idx, time in enumerate(times_gps):
        #            if sat == self.source_time_dict[time]:
        #                idxs_sat.append(idx)
        #        idx_sat = np.array(idxs_sat)
        #        obs_ax.plot(times_gps[idxs_sat], data.pr_data.values[idxs_sat]-data.pr_model.values[idxs_sat], marker=marker, linestyle='None', label=sat)
        #        #  plot elevation and azimuth 
        #    time_deltas_full = (times_gps - times_gps[0])/np.timedelta64(1, 's')
        #    full_time_hr = np.round(time_deltas_full[-1]/3600) 
        #    interval_hr = int(np.ceil(full_time_hr/8))       
        #    obs_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        #    obs_ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hr))  # Adjust interval as needed
        #    obs_fig.autofmt_xdate()  # Auto-rotate date labels
        #    obs_ax.set_ylabel('PR error (m, data-model)')
        #    obs_fig.savefig(antenna_handle.antenna_name + 'pr_error.png')
        #    plt.close(obs_fig)
        if sim is False:
            return data
        else:
            if self.iono_free is True:
                return pr_model_arr, pr_dual_arr, cp_model_arr, cp_dual_model_arr
            else:
                return pr_model_arr, cp_model_arr

    def model_group_phase_vlbi(self, antenna1_handle, antenna2_handle, baseline_handle, times_gps, clock_samples, \
            trop_samples1=[], trop_samples2=[], phase=False, phase_clock_samples=[], phase_only=False):
        """Model the group delay and phase delay using nav ephemeris and analytical delay model.
        """
        isCOM = True
        # arrays for caching results of analysis
        group_model_arr = []
        phase_model_arr = []
        cpw1_arr = []
        cpw2_arr = []
        f1 = 1575.42*1e6
        freq1_ant = 'G01' # antex frequency for receiver antenna
        if self.iono_free: # iono-free only for GPS, ref frequency for precise range
            if phase is True: 
                phase_model_dual_arr = []
            if self.iono_freq == 'L2':
                f2 = 1227.60*1e6
                freq2_ant = 'G02'
            elif self.iono_freq == 'L5': 
                f2 = 1176.45*1e6
                freq2_ant = 'G05'
            if antennaPCOData.nFreq <= 4 and self.iono_freq == 'L5':
                freq2_ant = 'G02' 
        else:
            freq2 ='0'
            f2 = 0

        if antenna1_handle.is_VLBI is False:
            # RX PCO correction
            antennaPCOData1 = antenna1_handle.antenna_PCO
            offset1_L1 = antennaPCOData1.getPhaseCenterOffset(freq1_ant)
            offset1_L1 = np.array([offset1_L1[0],offset1_L1[1],offset1_L1[2]])/1e3 # convert to m
            if self.iono_free is True:
                offset1_dual = antennaPCOData.getPhaseCenterOffset(freq2_ant)
                offset1_dual = np.array([offset_dual[0],offset_dual[1],offset_dual[2]])/1e3 # convert to m

        if antenna2_handle.is_VLBI is False:
            # RX PCO correction
            antennaPCOData2 = antenna2_handle.antenna_PCO
            offset2_L1 = antennaPCOData2.getPhaseCenterOffset(freq1_ant)
            offset2_L1 = np.array([offset2_L1[0],offset2_L1[1],offset2_L1[2]])/1e3 # convert to m
            if self.iono_free is True:
                offset2_dual = antennaPCOData.getPhaseCenterOffset(freq2_ant)
                offset2_dual = np.array([offset2_dual[0],offset2_dual[1],offset2_dual[2]])/1e3 # convert to m

        if phase is True: # need to account for carrier phase windup
            cpw_ant1 = {} # carrrier phase windup, dictionary by source
            cpw_ant2 = {}
            for source in self.source_array:
                cpw_ant1[source] = 0.0 # initialize cpw to 0
                cpw_ant2[source] = 0.0 

            wavelength_1 = baseline_handle.wavelength
            if self.iono_free:
                wavelength_2 = baseline_handle.wavelength_dual
      
            if antenna1_handle.is_VLBI is False:
                east_triple_ant1 = Position(antenna1_handle.Rotate_obj.get_value(1,0),\
                                     antenna1_handle.Rotate_obj.get_value(1,1),
                                     antenna1_handle.Rotate_obj.get_value(1,2))
  
                north_triple_ant1 = Position(antenna1_handle.Rotate_obj.get_value(0,0),\
                                     antenna1_handle.Rotate_obj.get_value(0,1),
                                     antenna1_handle.Rotate_obj.get_value(0,2))

            if antenna2_handle.is_VLBI is False:
                east_triple_ant2 = Position(antenna2_handle.Rotate_obj.get_value(1,0),\
                                     antenna2_handle.Rotate_obj.get_value(1,1),
                                     antenna2_handle.Rotate_obj.get_value(1,2))
  
                north_triple_ant2 = Position(antenna2_handle.Rotate_obj.get_value(0,0),\
                                     antenna2_handle.Rotate_obj.get_value(0,1),
                                     antenna2_handle.Rotate_obj.get_value(0,2))       

        if antenna1_handle.is_VLBI is True:
            a_vec1 = antenna1_handle.calc_VLBI_mount_vec()
        if antenna2_handle.is_VLBI is True:
            a_vec2 = antenna2_handle.calc_VLBI_mount_vec()

        if self.iono_comp_l4r:
            source_array = np.array([self.source_time_dict[time] for time in times_gps])
            idxs_bl = self.find_l4r_interpable(times_gps, source_array, antenna1_handle.l4r_name, antenna2_handle.l4r_name) 
            stec_vals = self.interp_l4r(times_gps[idxs_bl], source_array[idxs_bl], antenna1_handle.l4r_name, antenna2_handle.l4r_name)
        
        iono_model = []
        times_common = date_to_common(times_gps, 'GPS')
        for idx, common_time in enumerate(times_common):
            eph_time = EphTime(common_time)
            eph_time.setTimeSystem(TimeSystem.UTC)
            sat_id = self.source_time_dict[times_gps[idx]]
            sat_antenna = self.antenna_map[str(sat_id)]  
            RSID = RinexSatID(str(sat_id))
            rxpos1 = antenna1_handle.pos_series[idx,:]
            rxpos2 = antenna2_handle.pos_series[idx,:]
          
            system = RSID.systemString()
            if system == 'GPS': # ref frequencies for precise range
                freq1 = 'G01' # GPS L1
                if self.iono_free:
                    if self.iono_freq == 'L2':
                        freq2 = 'G02'
                    elif self.iono_freq == 'L5':
                        freq2 = 'G02' # only G02 available, no G05 for satellites
            elif system == 'Galileo':
                freq1 = 'E01'
                if self.iono_free:
                    if self.iono_freq == 'L2':
                        raise ValueError('No L2 frequency for BeiDou -- should not be here!')
                    elif self.iono_freq == 'L5':
                        freq2 = 'E05'
            elif system == 'BeiDou':
                freq1 = 'C01' 
                if self.iono_free:
                    if self.iono_freq == 'L2':
                        raise ValueError('No L2 frequency for BeiDou -- should not be here!')
                    elif self.iono_freq == 'L5':
                        freq2 = 'C05'
            
            # find satellite position, pointing vector at receive time
            sat_xvt = self.nav_store.get_xvt(RSID, common_time)
            rxpos1_gnsstk = Position(rxpos1[0], rxpos1[1], rxpos1[2])
            rxpos2_gnsstk = Position(rxpos2[0], rxpos2[1], rxpos2[2])
            sat_pos_gnsstk = Position(sat_xvt.x[0], sat_xvt.x[1], sat_xvt.x[2])
            sat_pos = np.array([sat_xvt.x[0],sat_xvt.x[1],sat_xvt.x[2]])
            sat_vel = np.array([sat_xvt.v[0],sat_xvt.v[1],sat_xvt.v[2]])
            rx2sat1 = sat_pos - rxpos1
            rx2sat1 /= np.linalg.norm(rx2sat1) # normalize vector
            rx2sat2 = sat_pos - rxpos2
            rx2sat2 /= np.linalg.norm(rx2sat2) # normalize vector
            
            if antenna1_handle.is_VLBI is True: 
                # compute the geometric model for the VLBI antenna
                point_vec = rx2sat1-a_vec1*np.dot(rx2sat1, a_vec1)
                fb_vec1 = antenna1_handle.axis_offset*point_vec/np.linalg.norm(point_vec) 

                # shift rxpos by offset
                rxpos1_L1 = rxpos1 + fb_vec1
                if self.iono_free:
                   rxpos1_dual = rxpos1 + fb_vec1

                if antenna1_handle.thermal_model is True:
                    if baseline_handle.weather_data is True:
                        Temp = baseline_handle.T1[idx]
                    elif antenna1_handle.weather_cal_active is True:
                        Temp, P, H = antenna1_handle.interpolate_weather(times_gps[idx])
                    elif self.trop_T is not None:
                        Temp = self.trop_T
                    else:
                        Temp = None
                    if Temp is not None:
                        elevation1 = rxpos1_gnsstk.elevationGeodetic(sat_pos_gnsstk)
                        dt_therm1 = antenna1_handle.calculate_thermal_deformation_delay(fb_vec1, rx2sat1, elevation1, Temp)
                    else:
                        dt_therm1 = 0
                else:
                    dt_therm1 = 0
            else:
                # compute antenna PCV
                dt_therm1 = 0
                elevation = rxpos1_gnsstk.elevationGeodetic(sat_pos_gnsstk)
                azimuth = rxpos1_gnsstk.azimuthGeodetic(sat_pos_gnsstk)
                ROT = antenna1_handle.R_mat # rotation from NEU to XYZ (ECEF)
                PCV_L1 = antennaPCOData1.getPhaseCenterVariation(freq1_ant, azimuth, elevation)
                rxpos1_L1 = rxpos1 + ROT@offset1_L1 - PCV_L1*1e-3*rx2sat1
                if self.iono_free:
                    PCV_dual = antennaPCOData1.getPhaseCenterVariation(freq2_ant, azimuth, elevation)
                    rxpos1_dual = rxpos1 + ROT@(offset1_dual) - PCV_dual*1e-3*rx2sat1

            if antenna2_handle.is_VLBI is True: 
                # compute the geometric model for the VLBI antenna
                point_vec = rx2sat2-a_vec2*np.dot(rx2sat2, a_vec2)
                fb_vec2 = antenna2_handle.axis_offset*point_vec/np.linalg.norm(point_vec) 

                # shift rxpos by offset
                rxpos2_L1 = rxpos2 + fb_vec2
                if self.iono_free:
                   rxpos2_dual = rxpos2 + fb_vec2

                if antenna2_handle.thermal_model is True:
                    if baseline_handle.weather_data is True:
                        Temp = baseline_handle.T1[idx]
                    elif antenna2_handle.weather_cal_active is True:
                        Temp, P, H = antenna2_handle.interpolate_weather(times_gps[idx])
                    elif self.trop_T is not None:
                        Temp = self.trop_T
                    else:
                        Temp = None
                    if Temp is not None:
                        elevation2 = rxpos2_gnsstk.elevationGeodetic(sat_pos_gnsstk)
                        dt_therm2 = antenna2_handle.calculate_thermal_deformation_delay(fb_vec2, rx2sat2, elevation2, Temp)
                    else:
                        dt_therm2 = 0
                else:
                    dt_therm2 = 0
            else:
                # compute antenna PCV
                dt_therm2 = 0
                elevation = rxpos2_gnsstk.elevationGeodetic(sat_pos_gnsstk)
                azimuth = rxpos2_gnsstk.azimuthGeodetic(sat_pos_gnsstk)
                ROT = antenna2_handle.R_mat # rotation from NEU to XYZ (ECEF)
                PCV_L1 = antennaPCOData2.getPhaseCenterVariation(freq1_ant, azimuth, elevation)
                rxpos2_L1 = rxpos2 + ROT@offset2_L1 - PCV_L1*1e-3*rx2sat2
                if self.iono_free:
                    PCV_dual = antennaPCOData2.getPhaseCenterVariation(freq2_ant, azimuth, elevation)
                    rxpos2_dual = rxpos2 + ROT@(offset2_dual) - PCV_dual*1e-3*rx2sat2
           
            sat_pos_L1 = self.sat_adj_PC(freq1, sat_antenna, times_gps[idx], eph_time, sat_xvt.x, 0, noPCV=True)
            if self.iono_free:
                sat_pos_dual = self.sat_adj_PC(freq2, sat_antenna, times_gps[idx], eph_time, sat_xvt.x, 0, noPCV=True)

            if self.analyticalDelay is True:
                delay, sat_xvt_corr = self.get_delay_model(RSID, common_time, times_gps[idx], \
                        sat_antenna, rxpos1_L1, rxpos2_L1, freq1, sat_pos_L1, sat_vel)
            else:   
                delay, sat_xvt_corr = self.get_delay_iter(RSID, common_time, times_gps[idx], \
                        sat_antenna, rxpos1_L1, rxpos2_L1, freq1, sat_pos_L1, sat_vel)
            # sat_xvt_corr will be used for satellite clock bias and troposphere correction
            if self.iono_free:
                delay_dual, sat_xvt_dual = self.get_delay_model(RSID, common_time, times_gps[idx], \
                        sat_antenna, rxpos1_dual, rxpos2_dual, freq2, sat_pos_dual, sat_vel)

            # get satellite phase center variation delay
            sat_pos = np.array([sat_xvt_corr.x[0],sat_xvt_corr.x[1],sat_xvt_corr.x[2]])
            rx2sat1 = sat_pos - rxpos1
            rx2sat1 /= np.linalg.norm(rx2sat1) # normalize vector
            rx2sat2 = sat_pos - rxpos2
            rx2sat2 /= np.linalg.norm(rx2sat2) # normalize vector
            PCV_delay1 = self.sat_PCV_delay(freq1, sat_antenna, times_gps[idx], eph_time, sat_xvt_corr.x, rx2sat1)
            PCV_delay2 = self.sat_PCV_delay(freq1, sat_antenna, times_gps[idx], eph_time, sat_xvt_corr.x, rx2sat2)
            if self.iono_free:
                PCV_dual_delay1 = self.sat_PCV_delay(freq2, sat_antenna, time_gps[idx], eph_time, sat_xvt_dual.x, rx2sat1)
                PCV_dual_delay2 = self.sat_PCV_delay(freq2, sat_antenna, time_gps[idx], eph_time, sat_xvt_dual.x, rx2sat2)

            rxpos1_gnsstk = Position(rxpos1_L1[0], rxpos1_L1[1], rxpos1_L1[2])
            rxpos2_gnsstk = Position(rxpos2_L1[0], rxpos2_L1[1], rxpos2_L1[2])
            sat_pos_gnsstk = Position(sat_xvt_corr.x[0], sat_xvt_corr.x[1], sat_xvt_corr.x[2])
            elevation1 = rxpos1_gnsstk.elevationGeodetic(sat_pos_gnsstk)
            azimuth1 = rxpos1_gnsstk.azimuthGeodetic(sat_pos_gnsstk)
            elevation2 = rxpos2_gnsstk.elevationGeodetic(sat_pos_gnsstk)
            azimuth2 = rxpos2_gnsstk.azimuthGeodetic(sat_pos_gnsstk)

            # clock offset correction
            if phase_only is False:
                clock_offset = clock_samples[idx]
 
            # L1 troposphere correction
            if antenna1_handle.tropModel is not None and antenna2_handle.tropModel is not None:
                if antenna1_handle.weather_cal_active is True:
                    T, P, H = antenna1_handle.interpolate_weather(times_gps[idx])
                    antenna1_handle.tropModel.setWeather(T, P, H)
                    antenna1_handle.tropModel.setHumidity(H)
                if antenna2_handle.weather_cal_active is True:
                    T, P, H = antenna2_handle.interpolate_weather(times_gps[idx])
                    antenna2_handle.tropModel.setWeather(T, P, H)
                    antenna2_handle.tropModel.setHumidity(H)

                try: trop_delay1 = antenna1_handle.tropModel.correction(rxpos1_gnsstk, sat_pos_gnsstk, common_time)
                except:
                    # rx height error
                    trop_delay1 = 0
                try: trop_delay2 = antenna2_handle.tropModel.correction(rxpos2_gnsstk, sat_pos_gnsstk, common_time)
                except:
                    # rx height error
                    trop_delay2 = 0

                if antenna1_handle.use_zwd_file:
                    trop_delay1 += antenna1_handle.interp_zwd_file(times_gps[idx], antenna1_handle.ref_pos, elevation1, azimuth1)[0]

                if antenna2_handle.use_zwd_file:
                    trop_delay2 += antenna2_handle.interp_zwd_file(times_gps[idx], antenna2_handle.ref_pos, elevation2, azimuth2)[0]

                if len(trop_samples1)>0:
                    trop_delay1 += trop_samples1[idx]*antenna1_handle.tropModel.wet_mapping_function(elevation1)
                if len(trop_samples2)>0:
                    trop_delay2 += trop_samples2[idx]*antenna2_handle.tropModel.wet_mapping_function(elevation2)

            else:
                trop_delay1 = 0
                trop_delay2 = 0

            if antenna1_handle.estimate_grav_def is True:
                #grav_delay1 = antenna1_handle.grav_def_model[0]*np.sin(np.deg2rad(elevation1)) + antenna1_handle.grav_def_model[1]*np.cos(np.deg2rad(elevation1))
                grav_delay1 = antenna1_handle.grav_def_model[0]*np.deg2rad(elevation1) + antenna1_handle.grav_def_model[1]*np.deg2rad(elevation1)**2
            else:
                grav_delay1 = 0

            if antenna2_handle.estimate_grav_def is True:
                #grav_delay2 = antenna2_handle.grav_def_model[0]*np.sin(np.deg2rad(elevation2)) + antenna2_handle.grav_def_model[1]*np.cos(np.deg2rad(elevation2))
                grav_delay2 = antenna2_handle.grav_def_model[0]*np.deg2rad(elevation2) + antenna2_handle.grav_def_model[1]*np.deg2rad(elevation2)**2
            else:
                grav_delay2 = 0

            if antenna1_handle.cable_cal_active is True:
                cc_delay1 = antenna1_handle.interpolate_cable_cal(times_gps[idx])*const.c/1e12 # interpolate, convert from ps to m
            else:
                cc_delay1 = 0
            if antenna2_handle.cable_cal_active is True:
                cc_delay2 = antenna2_handle.interpolate_cable_cal(times_gps[idx])*const.c/1e12 # interpolate, convert from ps to m
            else:
                cc_delay2 = 0

            if self.iono_comp_l4r is True:
                iono_delay1 = 0
                if idx in idxs_bl:
                    iono_delay2 = -ALPHA_IONO/baseline_handle.f1**2*const.c*stec_vals[idx]
                else:
                    iono_delay2 = 0
               
                # check against ionex
                #sat_pos_list = [sat_xvt_corr.x[0], sat_xvt_corr.x[1], sat_xvt_corr.x[2]]
                #rxpos1_list = [rxpos1_L1[0], rxpos1_L1[1], rxpos1_L1[2]]
                #rxpos2_list = [rxpos2_L1[0], rxpos2_L1[1], rxpos2_L1[2]]
                #timestamp = (times_gps[idx] - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
                #time_datetime = datetime.datetime.utcfromtimestamp(timestamp)
                #iono_delay1_test = self.ionex_store.calc_iono_correction(time_datetime, f1, \
                #        rxpos1_list, sat_pos_list, elevation1)
                #iono_delay2_test = self.ionex_store.calc_iono_correction(time_datetime, f1, \
                #        rxpos2_list, sat_pos_list, elevation2)
                #if antenna1_handle.l4r_name != antenna2_handle.l4r_name:
                #    breakpoint()
            elif self.iono_comp is True:
                # compensate for IONEX ionosphere delay
                sat_pos_list = [sat_xvt_corr.x[0], sat_xvt_corr.x[1], sat_xvt_corr.x[2]]
                rxpos1_list = [rxpos1_L1[0], rxpos1_L1[1], rxpos1_L1[2]]
                rxpos2_list = [rxpos2_L1[0], rxpos2_L1[1], rxpos2_L1[2]]
                timestamp = (times_gps[idx] - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
                time_datetime = datetime.datetime.utcfromtimestamp(timestamp)
                iono_delay1 = self.ionex_store.calc_iono_correction(time_datetime, f1, \
                        rxpos1_list, sat_pos_list, elevation1)
                iono_delay2 = self.ionex_store.calc_iono_correction(time_datetime, f1, \
                        rxpos2_list, sat_pos_list, elevation2)
            else:
                iono_delay1 = 0
                iono_delay2 = 0


            iono_model.append(iono_delay2-iono_delay1)

            # we need to add the modeled troposphere and clock offset to mirror the real PR measurement
            if phase_only is False:
                if self.iono_free is True:
                    group_model = float(IonosphereFreeRange([f1, f2], [delay, delay_dual])) + (trop_delay2-trop_delay1) + clock_offset \
                            - (dt_therm2-dt_therm1) + (grav_delay2-grav_delay1) + (cc_delay2-cc_delay1)
                else:
                    group_model = delay + (trop_delay2-trop_delay1) + (iono_delay2-iono_delay1) + clock_offset + PCV_delay2-PCV_delay1 \
                            - (dt_therm2-dt_therm1) + (grav_delay2-grav_delay1) + (cc_delay2-cc_delay1)

                group_model_arr.append(group_model)

            if phase is True: 
                if len(phase_clock_samples) > 0:
                    phase_clock_offset = phase_clock_samples[idx]
                else:
                    phase_clock_offset = clock_offset
                
                cpw1_source_last = cpw_ant1[sat_id]
                cpw2_source_last = cpw_ant2[sat_id]
                if antenna1_handle.is_VLBI:
                    if False: #antenna1_handle.antenna_type == 'BWG':
                        n_vec1 = antenna1_handle.R_mat[:,0]
                        e_vec1 = antenna1_handle.R_mat[:,1]
                        u_vec1 = antenna1_handle.R_mat[:,2]
                        cpw1_current = self.phase_windup_VLBI_BWG(antenna1_handle.antenna_name, sat_id, times_gps[idx], eph_time, sat_pos_gnsstk, \
                                rx2sat1, e_vec1, n_vec1, u_vec1)
                    else:
                        cpw1_current = self.phase_windup_VLBI(sat_id, times_gps[idx], eph_time, sat_pos_gnsstk, \
                                rx2sat1, a_vec1)
                        if antenna1_handle.antenna_type == 'Nasmyth' or antenna1_handle.antenna_type == 'BWG':
                        #if antenna1_handle.antenna_type == 'Nasmyth':
                            cpw1_current += elevation1/360
                        if antenna1_handle.antenna_type == 'BWG':
                            cpw1_current -= azimuth1/360
                else:
                    cpw1_current = self.phase_windup_GNSS(sat_id, times_gps[idx], eph_time, sat_pos_gnsstk, \
                            rx2sat1, east_triple_ant1, north_triple_ant1)
                cpw1_current = self.cycle_adj_windup(cpw1_source_last, cpw1_current)

                if antenna2_handle.is_VLBI:
                    if False: # antenna2_handle.antenna_type == 'BWG':
                        n_vec2 = antenna2_handle.R_mat[:,0]
                        e_vec2 = antenna2_handle.R_mat[:,1]
                        u_vec2 = antenna2_handle.R_mat[:,2]
                        cpw2_current = self.phase_windup_VLBI_BWG(antenna2_handle.antenna_name, sat_id, times_gps[idx], eph_time, sat_pos_gnsstk, \
                                rx2sat2, e_vec2, n_vec2, u_vec2)
                    else:
                        cpw2_current = self.phase_windup_VLBI(sat_id, times_gps[idx], eph_time, sat_pos_gnsstk, \
                                rx2sat2, a_vec2)
                        if antenna2_handle.antenna_type == 'Nasmyth' or antenna2_handle.antenna_type == 'BWG':
                        #if antenna2_handle.antenna_type == 'Nasmyth':
                            cpw2_current += elevation2/360
                        if antenna2_handle.antenna_type == 'BWG':
                            cpw2_current -= azimuth2/360
                else:
                    cpw2_current = self.phase_windup_GNSS(sat_id, times_gps[idx], eph_time, sat_pos_gnsstk, \
                            rx2sat2, east_triple_ant2, north_triple_ant2)
                cpw2_current = self.cycle_adj_windup(cpw2_source_last, cpw2_current)

                #if antenna1_handle.mount_type == 'XY-E' or antenna2_handle.mount_type == 'XY-E': 
                #    cpw1_current = 0 
                #    cpw2_curent = 0

                cpw_ant1[sat_id] = cpw1_current
                cpw_ant2[sat_id] = cpw2_current
                cpw1_arr.append(cpw1_current*wavelength_1)
                cpw2_arr.append(cpw2_current*wavelength_1)


                phase_model = delay + (trop_delay2-trop_delay1) - (iono_delay2-iono_delay1) + (cpw2_current-cpw1_current)*wavelength_1 \
                        + phase_clock_offset + (PCV_delay2-PCV_delay1) - (dt_therm2-dt_therm1) + (grav_delay2-grav_delay1) + (cc_delay2-cc_delay1)
                phase_model_arr.append(phase_model)
                
                if self.iono_free is True:
                    phase_model_dual = delay + (trop_delay2-trop_delay1) + (cpw2_current-cpw1_current)*wavelength_2 + phase_clock_offset\
                            + (PCV_dual_delay2-PCV_dual_delay1) - (dt_therm2-dt_therm1) + (grav_delay2-grav_delay1) + (cc_delay2-cc_delay1)
                    phase_model_dual_arr.append(phase_model_dual)

        if phase is True:
            baseline_handle.save_cpw(cpw1_arr, cpw2_arr)
            if self.iono_free is True:
                baseline_handle.save_vlbi_model(group_model_arr, phase_model_arr, phase_model_dual_arr)
            else:
                baseline_handle.save_vlbi_model(group_model_arr, phase_model_arr)
        else:
            baseline_handle.save_vlbi_model(group_model_arr)

        #if phase is True: breakpoint()

        return

    def model_group_phase_farfield(self, antenna1_handle, antenna2_handle, baseline_handle, times_gps, clock_samples, \
            trop_samples1=[], trop_samples2=[], phase=False, phase_clock_samples=[], phase_only=False):
        """Model the group delay and phase delay using nav ephemeris and analytical delay model.
        """
        isCOM = True
        # arrays for caching results of analysis
        group_model_arr = []
        phase_model_arr = []

        # temp -- plot phase windup
        cpw1_arr = []
        cpw2_arr = []
        f1 = 1575.42*1e6
        freq1_ant = 'G01' # antex frequency for receiver antenna
        if self.iono_free: # iono-free only for GPS, ref frequency for precise range
            if phase is True: 
                phase_model_dual_arr = []
            if self.iono_freq == 'L2':
                f2 = 1227.60*1e6
                freq2_ant = 'G02'
            elif self.iono_freq == 'L5': 
                f2 = 1176.45*1e6
                freq2_ant = 'G05'
            if antennaPCOData.nFreq <= 4 and self.iono_freq == 'L5':
                freq2_ant = 'G02' 
        else:
            freq2 ='0'
            f2 = 0

        if antenna1_handle.is_VLBI is False:
            # RX PCO correction
            antennaPCOData1 = antenna1_handle.antenna_PCO
            offset1_L1 = antennaPCOData1.getPhaseCenterOffset(freq1_ant)
            offset1_L1 = np.array([offset1_L1[0],offset1_L1[1],offset1_L1[2]])/1e3 # convert to m
            if self.iono_free is True:
                offset1_dual = antennaPCOData.getPhaseCenterOffset(freq2_ant)
                offset1_dual = np.array([offset_dual[0],offset_dual[1],offset_dual[2]])/1e3 # convert to m

        if antenna2_handle.is_VLBI is False:
            # RX PCO correction
            antennaPCOData2 = antenna2_handle.antenna_PCO
            offset2_L1 = antennaPCOData2.getPhaseCenterOffset(freq1_ant)
            offset2_L1 = np.array([offset2_L1[0],offset2_L1[1],offset2_L1[2]])/1e3 # convert to m
            if self.iono_free is True:
                offset2_dual = antennaPCOData.getPhaseCenterOffset(freq2_ant)
                offset2_dual = np.array([offset2_dual[0],offset2_dual[1],offset2_dual[2]])/1e3 # convert to m

        if phase is True: # need to account for carrier phase windup
            cpw_ant1 = {} # carrrier phase windup, dictionary by source
            cpw_ant2 = {}
            for source in self.source_array:
                cpw_ant1[source] = 0.0 # initialize cpw to 0
                cpw_ant2[source] = 0.0

            wavelength_1 = baseline_handle.wavelength
            if self.iono_free:
                wavelength_2 = baseline_handle.wavelength_dual

      
            if antenna1_handle.is_VLBI is False:
                east_triple_ant1 = Position(antenna1_handle.Rotate_obj.get_value(1,0),\
                                     antenna1_handle.Rotate_obj.get_value(1,1),
                                     antenna1_handle.Rotate_obj.get_value(1,2))
  
                north_triple_ant1 = Position(antenna1_handle.Rotate_obj.get_value(0,0),\
                                     antenna1_handle.Rotate_obj.get_value(0,1),
                                     antenna1_handle.Rotate_obj.get_value(0,2))

            if antenna2_handle.is_VLBI is False:
                east_triple_ant2 = Position(antenna2_handle.Rotate_obj.get_value(1,0),\
                                     antenna2_handle.Rotate_obj.get_value(1,1),
                                     antenna2_handle.Rotate_obj.get_value(1,2))
  
                north_triple_ant2 = Position(antenna2_handle.Rotate_obj.get_value(0,0),\
                                     antenna2_handle.Rotate_obj.get_value(0,1),
                                     antenna2_handle.Rotate_obj.get_value(0,2))       

        if antenna1_handle.is_VLBI is True:
            a_vec1 = antenna1_handle.calc_VLBI_mount_vec()
        if antenna2_handle.is_VLBI is True:
            a_vec2 = antenna2_handle.calc_VLBI_mount_vec()

        times_adj, full_offset = self.adj_clock_ns(antenna1_handle, antenna1_handle.clock_samples, \
                antenna1_handle.phase_clock_samples, phase, phase_only)
        times_common = date_to_common(times_adj, 'GPS')
        for idx, time in enumerate(times_gps):
            source = self.source_time_dict[time]
            eph_time = EphTime(times_common[idx])
            eph_time.setTimeSystem(TimeSystem.UTC)
            rxpos1 = antenna1_handle.pos_series[idx,:]
            rxpos2 = antenna2_handle.pos_series[idx,:]

            R_T2I = self.get_ECEF2ECI(eph_time)
            
            ra, dec = antenna1_handle.point_ra_dec_dict[time]
            s_ecef = self.compute_ptvec(ra, dec, eph_time)
            s_vec = self.compute_ptvec(ra, dec, eph_time, frame='ECI')

            R1_NEU = antenna1_handle.R_mat.T # matrix from XYZ (ECEF) to NEU 
            s1_NEU = R1_NEU@s_ecef
            R2_NEU = antenna2_handle.R_mat.T # matrix from XYZ (ECEF) to NEU 
            s2_NEU = R2_NEU@s_ecef
            elevation1 = np.degrees(np.arcsin(s1_NEU[2]))
            elevation2 = np.degrees(np.arcsin(s2_NEU[2]))

            # compute apparent source vector after refraction, aberration
            eph_time.convertSystemTo(TimeSystem.TDB)
            mjd_tdb = eph_time.dMJD() 
            set_km = True
            earth_posvel = self.sol_sys.relativeInertialPositionVelocityPyWrapper(mjd_tdb, self.sol_sys.idEarth, self.sol_sys.idSolarSystemBarycenter, set_km)
            earth_pos = np.array(earth_posvel[:3])*1e3
            earth_vel = np.array(earth_posvel[3:])*1e3/86400
            eph_time.convertSystemTo(TimeSystem.UTC) # convert time system back for other functions

            # refractivities -- Sovers et al (1998)
            refrac_1 = 3.13e-4/np.tan(np.radians(elevation1))
            refrac_2 = 3.13e-4/np.tan(np.radians(elevation2))
            rxpos1_eci = R_T2I@rxpos1
            rxpos1_eci /= np.linalg.norm(rxpos1_eci)
            rxpos2_eci = R_T2I@rxpos2
            rxpos2_eci /= np.linalg.norm(rxpos2_eci)
            s_app_1 = earth_vel/const.c - np.dot(s_vec,earth_vel)*s_vec/const.c + np.cos(refrac_1)*s_vec + np.sin(refrac_1)*(rxpos1_eci[2]-rxpos1_eci[2]*s_vec[2]*s_vec)
            s_app_2 = earth_vel/const.c - np.dot(s_vec,earth_vel)*s_vec/const.c + np.cos(refrac_2)*s_vec + np.sin(refrac_2)*(rxpos2_eci[2]-rxpos2_eci[2]*s_vec[2]*s_vec)
            s_ecef1 = R_T2I.T@s_app_1 # apparent source vector in ECEF
            s_ecef1 /= np.linalg.norm(s_ecef1)
            s_ecef2 = R_T2I.T@s_app_2
            s_ecef2 /= np.linalg.norm(s_ecef2)

            # update azimuth/elevation for apparent position
            s1_NEU = R1_NEU@s_ecef1
            s2_NEU = R2_NEU@s_ecef2
            elevation1 = np.degrees(np.arcsin(s1_NEU[2]))
            elevation2 = np.degrees(np.arcsin(s2_NEU[2]))
            azimuth1 = np.degrees(np.arctan2(s1_NEU[1], s1_NEU[0]))
            if azimuth1 < 0: azimuth1 += 360
            azimuth2 = np.degrees(np.arctan2(s2_NEU[1], s2_NEU[0]))
            if azimuth2 < 0: azimuth2 += 360

            if antenna1_handle.is_VLBI is True: 
                # compute the geometric model for the VLBI antenna
                point_vec1 = s_ecef1-a_vec1*np.dot(s_ecef1, a_vec1)
                point_vec1 /= np.linalg.norm(point_vec1)
                
                if antenna1_handle.estimate_ao is True:
                    fb_vec1 = ao_ant1*point_vec 
                else:
                    fb_vec1 = antenna1_handle.axis_offset*point_vec1
                
                # shift rxpos by offset
                rxpos1_f1 = rxpos1 + fb_vec1
                if self.iono_free:
                   rxpos1_dual = rxpos1 + fb_vec1

                if antenna1_handle.thermal_model is True:
                    if baseline_handle.weather_data is True:
                        Temp = baseline_handle.T1[idx]
                    elif antenna1_handle.weather_cal_active is True:
                        Temp, P, H = antenna1_handle.interpolate_weather(times_gps[idx])
                    elif self.trop_T is not None:
                        Temp = self.trop_T
                    else:
                        Temp = None
                    if Temp is not None:
                        dt_therm1 = antenna1_handle.calculate_thermal_deformation_delay(fb_vec1, s_ecef1, elevation1, Temp)
                    else:
                        dt_therm1 = 0
                else:
                    dt_therm1 = 0
            else:
                # compute antenna PCV
                dt_therm1 = 0
                ROT = antenna1_handle.R_mat # rotation from NEU to XYZ (ECEF)
                PCV_L1 = antennaPCOData1.getPhaseCenterVariation(freq1_ant, azimuth1, elevation1)
                rxpos1_f1 = rxpos1 + ROT@offset1_L1 - PCV_L1*1e-3*s_ecef1
                if self.iono_free:
                    PCV_dual = antennaPCOData1.getPhaseCenterVariation(freq2_ant, azimuth2, elevation2)
                    rxpos1_dual = rxpos1 + ROT@(offset1_dual) - PCV_dual*1e-3*s_ecef1

            if antenna2_handle.is_VLBI is True: 
                # compute the geometric model for the VLBI antenna
                point_vec2 = s_ecef2-a_vec2*np.dot(s_ecef2, a_vec2)
                point_vec2 /= np.linalg.norm(point_vec2) 
                
                if antenna2_handle.estimate_ao is True:
                    fb_vec2 = ao_ant2*point_vec2
                else:
                    fb_vec2 = antenna2_handle.axis_offset*point_vec2

                # shift rxpos by offset
                rxpos2_f1 = rxpos2 + fb_vec2
                if self.iono_free:
                   rxpos2_dual = rxpos2 + fb_vec2

                if antenna2_handle.thermal_model is True:
                    if baseline_handle.weather_data is True:
                        Temp = baseline_handle.T1[idx]
                    elif antenna2_handle.weather_cal_active is True:
                        Temp, P, H = antenna2_handle.interpolate_weather(times_gps[idx])
                    elif self.trop_T is not None:
                        Temp = self.trop_T
                    else:
                        Temp = None
                    if Temp is not None:
                        dt_therm2 = antenna2_handle.calculate_thermal_deformation_delay(fb_vec2, s_ecef2, elevation2, Temp)
                    else:
                        dt_therm2 = 0
                else:
                    dt_therm2 = 0
            else:
                # compute antenna PCV
                dt_therm2 = 0
                ROT = antenna2_handle.R_mat # rotation from NEU to XYZ (ECEF)
                PCV_L1 = antennaPCOData2.getPhaseCenterVariation(freq1_ant, azimuth2, elevation2)
                rxpos2_f1 = rxpos2 + ROT@offset2_L1 - PCV_L1*1e-3*s_ecef2
                if self.iono_free:
                    PCV_dual = antennaPCOData2.getPhaseCenterVariation(freq2_ant, azimuth2, elevation2)
                    rxpos2_dual = rxpos2 + ROT@(offset2_dual) - PCV_dual*1e-3*s_ecef2
           

            delay = self.get_delay_farfield(s_vec, eph_time, rxpos1_f1, rxpos2_f1)

            if self.iono_free:
                delay_dual = self.get_delay_farfield(s_vec, eph_time, rxpos1_dual, rxpos2_dual)

            # clock offset correction
            if phase_only is False:
                clock_offset = clock_samples[idx]            
 
            # L1 troposphere correction
            if antenna1_handle.tropModel is not None and antenna2_handle.tropModel is not None:
                if baseline_handle.weather_data is True:
                    if baseline_handle.P1 is not None and baseline_handle.T1 is not None and baseline_handle.H1 is not None:
                        antenna1_handle.tropModel.setWeather(baseline_handle.T1[idx], baseline_handle.P1[idx], baseline_handle.H1[idx]*100)
                    if baseline_handle.P2 is not None and baseline_handle.T2 is not None and baseline_handle.H2 is not None:
                        antenna2_handle.tropModel.setWeather(baseline_handle.T2[idx], baseline_handle.P2[idx], baseline_handle.H2[idx]*100)
                else:
                    if antenna1_handle.weather_cal_active is True:
                        T, P, H = antenna1_handle.interpolate_weather(times_gps[idx])
                        antenna1_handle.tropModel.setWeather(T, P, H)
                    if antenna2_handle.weather_cal_active is True:
                        T, P, H = antenna2_handle.interpolate_weather(times_gps[idx])
                        antenna2_handle.tropModel.setWeather(T, P, H)

                if getattr(antenna1_handle.tropModel, 'vmf3_type', None) is not None:
                    # VMF3 trop needs azimuth and time
                    rxpos1_gnsstk = Position(rxpos1[0], rxpos1[1], rxpos1[2])
                    rxpos2_gnsstk = Position(rxpos2[0], rxpos2[1], rxpos2[2])
                    common_time = times_common[idx]
                    trop_delay1 = antenna1_handle.tropModel.correction_azel(rxpos1_gnsstk, azimuth1, elevation1, common_time)
                    trop_delay2 = antenna2_handle.tropModel.correction_azel(rxpos2_gnsstk, azimuth2, elevation2, common_time)
                else:
                    trop_delay1 = antenna1_handle.tropModel.correction(elevation1)
                    trop_delay2 = antenna2_handle.tropModel.correction(elevation2)

                if antenna1_handle.use_zwd_file:
                    trop_delay1 += antenna1_handle.interp_zwd_file(times_gps[idx], antenna1_handle.ref_pos, elevation1, azimuth1)[0]

                if antenna2_handle.use_zwd_file:
                    trop_delay2 += antenna2_handle.interp_zwd_file(times_gps[idx], antenna2_handle.ref_pos, elevation2, azimuth2)[0]

                if len(trop_samples1)>0:
                    trop_delay1 += trop_samples1[idx]*antenna1_handle.tropModel.wet_mapping_function(elevation1)
                if len(trop_samples2)>0:
                    trop_delay2 += trop_samples2[idx]*antenna2_handle.tropModel.wet_mapping_function(elevation2)
            else:
                trop_delay1 = 0
                trop_delay2 = 0

            if antenna1_handle.estimate_grav_def is True:
                #grav_delay1 = antenna1_handle.grav_def_model[0]*np.sin(np.deg2rad(elevation1)) + antenna1_handle.grav_def_model[1]*np.cos(np.deg2rad(elevation1))
                grav_delay1 = antenna1_handle.grav_def_model[0]*np.deg2rad(elevation1) + antenna1_handle.grav_def_model[1]*np.deg2rad(elevation1)**2
            else:
                grav_delay1 = 0
            if antenna2_handle.estimate_grav_def is True:
                #grav_delay2 = antenna2_handle.grav_def_model[0]*np.sin(np.deg2rad(elevation2)) + antenna2_handle.grav_def_model[1]*np.cos(np.deg2rad(elevation2))
                grav_delay2 = antenna2_handle.grav_def_model[0]*np.deg2rad(elevation2) + antenna2_handle.grav_def_model[1]*np.deg2rad(elevation2)**2
            else:
                grav_delay2 = 0

            if antenna1_handle.cable_cal_active is True:
                cc_delay1 = antenna1_handle.interpolate_cable_cal(times_gps[idx])*const.c/1e12 # interpolate, convert from ps to m
            else:
                cc_delay1 = 0
            if antenna2_handle.cable_cal_active is True:
                cc_delay2 = antenna2_handle.interpolate_cable_cal(times_gps[idx])*const.c/1e12 # interpolate, convert from ps to m
            else:
                cc_delay2 = 0

            if self.iono_comp is True:
                # compensate for IONEX ionosphere delay
                # NEED TO UPDATE THIS -- CALCULATE IPP, use individual functions in ionex_store
                raise ValueError('Ionosphere correction for celestial observations not yet implemented')
                rxpos1_list = [rxpos1_L1[0], rxpos1_L1[1], rxpos1_L1[2]]
                rxpos2_list = [rxpos2_L1[0], rxpos2_L1[1], rxpos2_L1[2]]
                timestamp = (time - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
                time_datetime = datetime.datetime.utcfromtimestamp(timestamp)
                iono_delay1 = self.ionex_store.calc_iono_correction(time_datetime, f1, \
                        rxpos1_list, sat_pos_list, elevation1)
                iono_delay2 = self.ionex_store.calc_iono_correction(time_datetime, f1, \
                        rxpos2_list, sat_pos_list, elevation2)
            else:
                iono_delay1 = 0
                iono_delay2 = 0

            if False: #antenna1_handle.antenna_name=='HOBART26':
                # implement linear pointing error
                s1_NEU = R1_NEU@s_ecef1
                x_ang = np.arctan(s1_NEU[0]/s1_NEU[2]) # N = y = index 0
                y_ang = np.arctan(s1_NEU[1]/s1_NEU[2]) # E = x = index 1
                x_ang -= 0.00107*np.degrees(y_ang) # correction term
                z_trf = np.sign(s1_NEU[2])/np.sqrt(1+np.tan(x_ang)**2+np.tan(y_ang)**2) # U = z = index 3
                s1_NEU = np.array([z_trf*np.tan(x_ang),z_trf*np.tan(y_ang),z_trf])
                s_ecef1 = R1_NEU.T@s1_NEU

                #implement gravitational deformation delay correction
                #gravitational_delay = 1e-3/(4*73)*(3/73*np.degrees(x_ang)**2-7*np.degrees(x_ang)) 
                #delay = delay - gravitational_delay
            elif False: #antenna2_handle.antenna_name=='HOBART26':
                s2_NEU = R2_NEU@s_ecef1
                s2_NEU_save = s2_NEU
                x_ang = np.arctan(s2_NEU[0]/s2_NEU[2]) # N = y = index 0
                y_ang = np.arctan(s2_NEU[1]/s2_NEU[2]) # E = x = index 1
                x_ang -= 0.00107*np.degrees(y_ang) # correction term
                z_trf = np.sign(s2_NEU[2])/np.sqrt(1+np.tan(x_ang)**2+np.tan(y_ang)**2) # U = z = index 3
                s2_NEU = np.array([z_trf*np.tan(x_ang),z_trf*np.tan(y_ang),z_trf])
                s_ecef2_save = s_ecef2
                s_ecef2 = R2_NEU.T@s2_NEU
                #gravitational_delay = 1e-3/(4*73)*(3/73*np.degrees(x_ang)**2-7*np.degrees(x_ang)) 
                #delay = delay - gravitational_delay

            # we need to add the modeled troposphere and clock offset to mirror the real PR measurement
            if phase_only is False:
                if self.iono_free is True:
                    group_model = float(IonosphereFreeRange([f1, f2], [delay, delay_dual])) - (trop_delay2-trop_delay1) + clock_offset \
                            - (dt_therm2-dt_therm1) + (grav_delay2-grav_delay1) + (cc_delay2-cc_delay1)
                else:
                    group_model = delay + (trop_delay2-trop_delay1) + (iono_delay2-iono_delay1) + clock_offset - (dt_therm2-dt_therm1)\
                            + (grav_delay2-grav_delay1) + (cc_delay2-cc_delay1)
                group_model_arr.append(group_model) 

            if phase is True: 
                if len(phase_clock_samples) > 0:
                    phase_clock_offset = phase_clock_samples[idx]
                else:
                    phase_clock_offset = clock_offset
                
                cpw1_source_last = cpw_ant1[source]
                cpw2_source_last = cpw_ant2[source]
                if antenna1_handle.is_VLBI:
                    if False: #antenna1_handle.antenna_type == 'BWG':
                        n_vec1 = antenna1_handle.R_mat[:,0]
                        e_vec1 = antenna1_handle.R_mat[:,1]
                        u_vec1 = antenna1_handle.R_mat[:,2]
                        cpw1_current = self.phase_windup_VLBI_ff_BWG(antenna1_handle.antenna_name, s_ecef1, e_vec1, n_vec1, u_vec1)
                    else:
                        cpw1_current = self.phase_windup_VLBI_ff(s_ecef1, a_vec1)
                        if antenna1_handle.antenna_type == 'Nasmyth' or antenna1_handle.antenna_type == 'BWG':
                        #if antenna1_handle.antenna_type == 'Nasmyth':
                            cpw1_current += elevation1/360
                        if antenna1_handle.antenna_type == 'BWG':
                            cpw1_current -= azimuth1/360
                else:
                    cpw1_current = self.phase_windup_GNSS_ff(s_ecef1, north_triple_ant1, east_triple_ant1)
                cpw1_current = self.cycle_adj_windup(cpw1_source_last, cpw1_current)

                if antenna2_handle.is_VLBI:
                    if False: # antenna2_handle.antenna_type == 'BWG':
                        n_vec2 = antenna2_handle.R_mat[:,0]
                        e_vec2 = antenna2_handle.R_mat[:,1]
                        u_vec2 = antenna2_handle.R_mat[:,2]
                        cpw2_current = self.phase_windup_VLBI_ff_BWG(antenna2_handle.antenna_name, s_ecef2, e_vec2, n_vec2, u_vec2)
                    else:
                        cpw2_current = self.phase_windup_VLBI_ff(s_ecef2, a_vec2)
                        if antenna2_handle.antenna_type == 'Nasmyth' or antenna2_handle.antenna_type == 'BWG':
                        #if antenna2_handle.antenna_type == 'Nasmyth':
                            cpw2_current += elevation2/360
                        if antenna2_handle.antenna_type == 'BWG':
                            cpw2_current -= azimuth2/360
                else:
                    cpw2_current = self.phase_windup_GNSS_ff(s_ecef2, north_triple_ant2, east_triple_ant2)
                cpw2_current = self.cycle_adj_windup(cpw2_source_last, cpw2_current)

                cpw_ant1[source] = cpw1_current
                cpw_ant2[source] = cpw2_current
                cpw1_arr.append(cpw1_current*wavelength_1)
                cpw2_arr.append(cpw2_current*wavelength_1)

                phase_model = delay + (trop_delay2-trop_delay1) - (iono_delay2-iono_delay1) + (cpw2_current-cpw1_current)*wavelength_1 \
                        + phase_clock_offset  - (dt_therm2-dt_therm1) + (grav_delay2-grav_delay1) + (cc_delay2-cc_delay1)
                phase_model_arr.append(phase_model)
                
                if self.iono_free is True:
                    phase_model_dual = delay + (trop_delay2-trop_delay1) + (cpw2_current-cpw1_current)*wavelength_2 + phase_clock_offset \
                            - (dt_therm2-dt_therm1) + (grav_delay2-grav_delay1) + (cc_delay2-cc_delay1)
                    phase_model_dual_arr.append(phase_model_dual)

        if phase is True:
            if self.iono_free is True:
                baseline_handle.save_vlbi_model(group_model_arr, phase_model_arr, phase_model_dual_arr)
            else:
                baseline_handle.save_vlbi_model(group_model_arr, phase_model_arr)
                baseline_handle.save_cpw(cpw1_arr, cpw2_arr)
        else:
            baseline_handle.save_vlbi_model(group_model_arr)
        return 

    def sim_data(self, antenna_handle, times_gps, clock_samples, phase_clock_samples, pr_dop_only=False, source_array=[], scan_nums=[]):
        """ Produce pseudorange and carrier phase measurements from the analytical model
        """
        if antenna_handle.is_VLBI is False:
            antennaPCOData = antenna_handle.antenna_PCO
         
        isCOM = True
        # arrays for caching results of analysis
        f1 = 1575.42*1e6
        freq1_ant = 'G01' # antex frequency for receiver antenna
        if self.iono_free: # iono-free only for GPS, ref frequency for precise range
            if self.iono_freq == 'L2':
                f2 = 1227.60*1e6
                freq2_ant = 'G02'
            elif self.iono_freq == 'L5': 
                f2 = 1176.45*1e6
                freq2_ant = 'G05'
            if antennaPCOData.nFreq <= 4 and self.iono_freq == 'L5':
                freq2_ant = 'G02' 
        else:
            freq2 ='0'
            f2 = 0

        if antenna_handle.is_VLBI is False:
            # RX PCO correction
            offset_L1 = antennaPCOData.getPhaseCenterOffset(freq1_ant)
            offset_L1 = np.array([offset_L1[0],offset_L1[1],offset_L1[2]])/1e3 # convert to m
            if self.iono_free is True:
                offset_dual = antennaPCOData.getPhaseCenterOffset(freq2_ant)
                offset_dual = np.array([offset_dual[0],offset_dual[1],offset_dual[2]])/1e3 # convert to m

        if antenna_handle.offset_NEU is not None:
            offset_XYZ = antenna_handle.R_mat.dot(antenna_handle.offset_NEU)

        pr_model_arr = [] # pseudorange model
        if pr_dop_only is False:
            cp_model_arr = [] # carrier phase-based range by epoch (changing source)
            cpw = {} # carrier phase windup, dictionary by source
            for source in self.source_array:
                cpw[source] = 0.0 # initialize cpw to 0
        else:
            dop_model_arr = []
            label_arr = []

        wavelength_1 = const.c/f1
        if self.iono_free:
            cp_dual_model_arr = []
            wavelength_2 = const.c/f2
      
        east_triple = Position(antenna_handle.Rotate_obj.get_value(1,0),\
                             antenna_handle.Rotate_obj.get_value(1,1),
                             antenna_handle.Rotate_obj.get_value(1,2))
  
        north_triple = Position(antenna_handle.Rotate_obj.get_value(0,0),\
                             antenna_handle.Rotate_obj.get_value(0,1),
                             antenna_handle.Rotate_obj.get_value(0,2))
        
        if antenna_handle.is_VLBI is True:
            a_vec = antenna_handle.calc_VLBI_mount_vec()
         
        if antenna_handle.dither_phase is True:
            print('DITHERING PHASE!')
            SEED= int(sha256(antenna_handle.antenna_name.encode('utf-8')).hexdigest(),16) % (2**32)
            rng = np.random.default_rng(seed=SEED)
            MAX_N=4 # number of wavelengths that can be added or subtracted
            rand_int = rng.integers(-MAX_N, MAX_N, size=len(times_gps))
            if self.iono_free is True:
                rand_int_dual = rng.integers(-MAX_N, MAX_N, size=len(times_gps))

        # adjust the receiver time by the clock bias, use phase clock preferentially
        times_adj, full_offset = self.adj_clock_ns(antenna_handle, clock_samples, \
                phase_clock_samples, False, False)

        times_common = date_to_common(times_gps, 'GPS')

        for idx, common_time in enumerate(times_common):
            if len(source_array) > 0:
                sat_id = source_array[idx]
            else:
                sat_id = self.source_time_dict[times_gps[idx]]
            sat_antenna = self.antenna_map[str(sat_id)]  
            RSID = RinexSatID(str(sat_id))
            rxpos = antenna_handle.pos_series[idx,:]
            if antenna_handle.offset_NEU is not None:
                rxpos += offset_XYZ
            current_pos = Position(rxpos[0], rxpos[1], rxpos[2]) 
            eph_time = EphTime(common_time)
            eph_time.setTimeSystem(TimeSystem.UTC)
          
            system = RSID.systemString()
            if system == 'GPS': # ref frequencies for precise range
                freq1 = 'G01' # GPS L1
                if self.iono_free:
                    if self.iono_freq == 'L2':
                        freq2 = 'G02'
                    elif self.iono_freq == 'L5':
                        freq2 = 'G02' # only G02 available, no G05 for satellites
            elif system == 'Galileo':
                freq1 = 'E01'
                if self.iono_free:
                    if self.iono_freq == 'L2':
                        raise ValueError('No L2 frequency for BeiDou -- should not be here!')
                    elif self.iono_freq == 'L5':
                        freq2 = 'E05'
            elif system == 'BeiDou':
                freq1 = 'C01' 
                if self.iono_free:
                    if self.iono_freq == 'L2':
                        raise ValueError('No L2 frequency for BeiDou -- should not be here!')
                    elif self.iono_freq == 'L5':
                        freq2 = 'C05'
            
            # find satellite position, pointing vector at receive time
            sat_xvt = self.nav_store.get_xvt(RSID, common_time)
            rxpos_gnsstk = Position(rxpos[0], rxpos[1], rxpos[2])
            sat_pos_gnsstk = Position(sat_xvt.x[0], sat_xvt.x[1], sat_xvt.x[2])
            elevation = rxpos_gnsstk.elevationGeodetic(sat_pos_gnsstk)
            azimuth = rxpos_gnsstk.azimuthGeodetic(sat_pos_gnsstk)
            sat_pos = np.array([sat_xvt.x[0],sat_xvt.x[1],sat_xvt.x[2]])
            sat_vel = np.array([sat_xvt.v[0],sat_xvt.v[1],sat_xvt.v[2]])
            rx2sat = sat_pos - rxpos
            rx2sat /= np.linalg.norm(rx2sat) # normalize vector
            rx2sat_hold = rx2sat
            
            if antenna_handle.is_VLBI is True: 
                # compute the geometric model for the VLBI antenna
                if antenna_handle.point_ra_dec_dict is not None:
                    ra, dec = antenna_handle.point_ra_dec_dict[times_gps[idx]]
                    rx2sat = self.compute_ptvec(ra, dec, eph_time)
                point_vec = rx2sat-a_vec*np.dot(rx2sat,a_vec)
                fb_vec = antenna_handle.axis_offset*point_vec/np.linalg.norm(point_vec)
                # shift rxpos by offset
                rxpos_L1 = rxpos + fb_vec
                if self.iono_free:
                   rxpos_dual = rxpos + fb_vec

            # compute antenna PCV
            if antenna_handle.is_VLBI is False:
                ROT = antenna_handle.R_mat # rotation from NEU to XYZ (ECEF) 
                try: PCV_L1 = antennaPCOData.getPhaseCenterVariation(freq1_ant, azimuth, elevation)
                except:
                    # likely below horizon, just set to 0
                    PCV_L1 = 0
                rxpos_L1 = rxpos + ROT@(offset_L1) - PCV_L1*1e-3*rx2sat
                if self.iono_free:
                    PCV_dual = antennaPCOData.getPhaseCenterVariation(freq2_ant, azimuth, elevation)
                    rxpos_dual = rxpos + ROT@(offset_dual) - PCV_dual*1e-3*rx2sat
           
            sat_pos_L1 = self.sat_adj_PC(freq1, sat_antenna, times_adj[idx], eph_time, sat_xvt.x, rx2sat)
            if self.iono_free:
                sat_pos_dual = self.sat_adj_PC(freq2, sat_antenna, times_adj[idx], eph_time, sat_xvt.x, rx2sat)

            pr_model, sat_xvt_corr = self.get_pr_model(RSID, common_time, times_adj[idx], \
                    sat_antenna, rxpos_L1, freq1, sat_pos_L1, sat_vel)
            # sat_xvt_corr will be used for satellite clock bias and troposphere correction
            if self.iono_free:
                pr_dual, _ = self.get_pr_model(RSID, common_time, times_adj[idx], \
                        sat_antenna, rxpos_dual, freq2, sat_pos_dual, sat_vel)

            # get satellite az/el
            # NB: elev, azim local not geodetic
            rxpos_gnsstk = Position(rxpos_L1[0], rxpos_L1[1], rxpos_L1[2])
            sat_pos_gnsstk = Position(sat_xvt_corr.x[0], sat_xvt_corr.x[1], sat_xvt_corr.x[2])
            elevation = rxpos_gnsstk.elevationGeodetic(sat_pos_gnsstk)
            azimuth = rxpos_gnsstk.azimuthGeodetic(sat_pos_gnsstk)

            # clock offset correction
            clock_offset = clock_samples[idx]            
 
            # L1 troposphere correction
            if antenna_handle.tropModel is not None:
                trop_delay = antenna_handle.tropModel.correction(rxpos_gnsstk, sat_pos_gnsstk, common_time)
            else:
                trop_delay = 0

            if antenna_handle.use_zwd_file:
                trop_delay += antenna_handle.interp_zwd_file(times_gps[idx], antenna_handle.ref_pos, elevation, azimuth)[0]

            sat_clk = sat_xvt_corr.clkbias*const.c # satellite clock bias in m
            
            if self.iono_comp is True and not self.iono_comp_l4r:
                # compensate for IONEX ionosphere delay
                rxpos_list = [rxpos_L1[0], rxpos_L1[1], rxpos_L1[2]]
                sat_pos_list = [sat_xvt_corr.x[0], sat_xvt_corr.x[1], sat_xvt_corr.x[2]]
                timestamp = (times_gps[idx] - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
                time_datetime = datetime.datetime.utcfromtimestamp(timestamp)
                iono_delay = self.ionex_store.calc_iono_correction(time_datetime, f1, \
                        rxpos_list, sat_pos_list, elevation)
                if self.iono_free is True:
                    iono_delay_dual = self.ionex_store.calc_iono_correction(time_datetime, f1, \
                            rxpos_list, sat_pos_list, elevation)
            else:
                iono_delay = 0
                if self.iono_free is True:
                    iono_delay_dual = 0
            # we need to add the modeled troposphere and clock offset to mirror the real PR measurement
            pr_final = pr_model + trop_delay + iono_delay + clock_offset - sat_clk
            pr_model_arr.append(pr_final)
            
            if pr_dop_only is True:
                # we want the pseudorange and doppler information for software correlation only
                sat_vel = np.array([sat_xvt_corr.v[0],sat_xvt_corr.v[1],sat_xvt_corr.v[2]])
                sat_pos_dop = np.array([sat_xvt_corr.x[0],sat_xvt_corr.x[1],sat_xvt_corr.x[2]])
                rx2sat_dop =  sat_pos_dop - rxpos
                rx2sat_dop /= np.linalg.norm(rx2sat_dop) # normalize vector
                doppler = -np.dot(sat_vel,rx2sat_dop)*f1/const.c 
                dop_model_arr.append(doppler)

                if len(sat_id) == 2:
                    sat_print = str(sat_id[0] + '0' + sat_id[1])
                else: 
                    sat_print = str(sat_id)

                if len(scan_nums) > 0:
                    idx_scan = int(scan_nums[idx])
                else:
                    idx_scan = idx+1

                if idx_scan < 100:
                    if idx_scan < 10:
                        idx_print = '00'+str(idx_scan)
                    else:
                        idx_print = '0'+str(idx_scan)
                else:
                    idx_print = str(idx_scan)
                label_arr.append(sat_print+idx_print)
            else:

                if self.iono_free is True:
                    pr_dual_final = pr_model + trop_delay + iono_delay_dual + clock_offset - sat_clk

                if len(phase_clock_samples) > 0:
                    phase_clock_offset = phase_clock_samples[idx]
                else:
                    phase_clock_offset = clock_offset
                
                cpw_source_last = cpw[sat_id]
                Rx2Tx = Position(rx2sat[0], rx2sat[1], rx2sat[2])
                #cpw_current = PhaseWindup(cpw_source_last, common_time, sat_pos_gnsstk, \
                #        Rx2Tx, west_triple, north_triple, self.sol_sys)
                if antenna_handle.is_VLBI:
                    cpw_current = self.phase_windup_VLBI(sat_id, times_gps[idx], eph_time, sat_pos_gnsstk, \
                            rx2sat, a_vec)
                else:
                    cpw_current = self.phase_windup_GNSS(sat_id, times_gps[idx], eph_time, sat_pos_gnsstk, \
                            rx2sat, east_triple, north_triple)
                cpw[sat_id] = cpw_current

                cp_model = pr_model + trop_delay - iono_delay + cpw_current*wavelength_1  + phase_clock_offset - sat_clk # satellite clock bias in mk
                cp_model_arr.append(cp_model)
                
                if self.iono_free is True:
                    cp_dual_model = pr_dual - iono_delay_dual + trop_delay + cpw_current*wavelength_2 + phase_clock_offset - sat_clk
                    cp_dual_model_arr.append(cp_dual_model)

        if antenna_handle.dither_phase is True and pr_dop_only is False:
            cp_model_arr = np.array(cp_model_arr) + wavelength_1*rand_int
            if self.iono_free is True:
                cp_dual_model = np.array(cp_dual_model) + wavelength_2*rand_int_dual

        if pr_dop_only is True:
            return pr_model_arr, dop_model_arr, label_arr 
        else:
            satellites = self.source_array.tolist()

            # Create 2D arrays for pr and cp with nan values
            pr_2d = np.full((len(times_gps), len(satellites)), np.nan)
            cp_2d = np.full((len(times_gps), len(satellites)), np.nan)
            if self.iono_free is True:
                pr_dual_2d = np.full((len(times_gps), len(satellites)), np.nan)
                cp_dual_2d = np.full((len(times_gps), len(satellites)), np.nan)
            
            # Assign values to the source satellite at each epoch
            for jdx in range(len(times_gps)):
                sv_index = satellites.index(self.source_time_dict[times_gps[jdx]])
                pr_2d[jdx, sv_index] = pr_model_arr[jdx]
                cp_2d[jdx, sv_index] = cp_model_arr[jdx]
                if self.iono_free is True:
                    pr_2d[jdx, sv_index] = pr_model_arr[jdx]
                    cp_2d[jdx, sv_index] = cp_model_arr[jdx]

            data = xr.Dataset({}, coords={"time": times_gps, "sv": satellites})
            pr_xarray = xr.DataArray(pr_2d, coords={'time': times_gps, 'sv': satellites})
            data = data.assign({'C1': pr_xarray})

            cp_xarray = xr.DataArray(cp_2d, coords={'time': times_gps, 'sv': satellites})
            data = data.assign({'L1': cp_xarray})

            if self.iono_free is True:
                if self.iono_freq == 'L2':
                    cp_dual_model_xarray = xr.DataArray(cp_dual_model_arr, coords={'time_sv': multi_index}, dims='time_sv')
                    data = data.assign({'L2': cp_dual_model_xarray})
                    pr_xarray = xr.DataArray(pr_dual_model_arr, coords={'time_sv': multi_index}, dims='time_sv')
                    data = data.assign({'P2': pr_xarray})
                elif self.iono_freq == 'L5':
                    cp_dual_model_xarray = xr.DataArray(cp_dual_model_arr, coords={'time_sv': multi_index}, dims='time_sv')
                    data = data.assign({'L5': cp_dual_model_xarray})
                    pr_xarray = xr.DataArray(pr_dual_model_arr, coords={'time_sv': multi_index}, dims='time_sv')
                    data = data.assign({'C5': pr_xarray})

            return data

    def sim_pr_simple(self, antenna_handle, times_gps, clock_samples, freq, source_array=[]):
        """ Produce pseudorange from a simplified analytical model
        """
        pr_model_arr = [] # pseudorange model
        dop_model_arr = [] # Doppler model
        if antenna_handle.is_VLBI is True:
            a_vec = antenna_handle.calc_VLBI_mount_vec()
         
        # adjust the receiver time by the clock bias
        times_common = date_to_common(times_gps, 'GPS')
        times_adj, full_offset = self.adj_clock_ns(antenna_handle, clock_samples, None, False, False)

        for idx, common_time in enumerate(times_common):
            if len(source_array) > 0:
                sat_id = source_array[idx]
            else:
                sat_id = self.source_time_dict[times_gps[idx]]
            RSID = RinexSatID(str(sat_id))
            rxpos = antenna_handle.ref_pos
            current_pos = Position(rxpos[0], rxpos[1], rxpos[2]) 
            eph_time = EphTime(common_time)
            eph_time.setTimeSystem(TimeSystem.UTC)
          
            system = RSID.systemString()
            
            # find satellite position, pointing vector at receive time
            sat_xvt = self.nav_store.get_xvt(RSID, common_time)
            rxpos_gnsstk = Position(rxpos[0], rxpos[1], rxpos[2])
            sat_pos_gnsstk = Position(sat_xvt.x[0], sat_xvt.x[1], sat_xvt.x[2])
            elevation = rxpos_gnsstk.elevationGeodetic(sat_pos_gnsstk)
            azimuth = rxpos_gnsstk.azimuthGeodetic(sat_pos_gnsstk)
            sat_pos = np.array([sat_xvt.x[0],sat_xvt.x[1],sat_xvt.x[2]])
            sat_vel = np.array([sat_xvt.v[0],sat_xvt.v[1],sat_xvt.v[2]])
            rx2sat = sat_pos - rxpos
            rx2sat /= np.linalg.norm(rx2sat) # normalize vector
            rx2sat_hold = rx2sat
            
            if antenna_handle.is_VLBI is True: 
                # compute the geometric model for the VLBI antenna
                if antenna_handle.point_ra_dec_dict is not None:
                    ra, dec = antenna_handle.point_ra_dec_dict[times_adj[idx]]
                    rx2sat = self.compute_ptvec(ra, dec, eph_time)
                point_vec = rx2sat-a_vec*np.dot(rx2sat,a_vec)
                fb_vec = antenna_handle.axis_offset*point_vec/np.linalg.norm(point_vec)
                # shift rxpos by offset
                rxpos += fb_vec

            sat_pos = np.array([sat_xvt.x[0],sat_xvt.x[1],sat_xvt.x[2]])
            pr_model, sat_xvt_corr = self.get_pr_model(RSID, common_time, times_adj[idx], \
                    None, rxpos, None, sat_pos, sat_vel, False, False)

            # get satellite az/el
            # NB: elev, azim local not geodetic
            rxpos_gnsstk = Position(rxpos[0], rxpos[1], rxpos[2])
            sat_pos_gnsstk = Position(sat_xvt_corr.x[0], sat_xvt_corr.x[1], sat_xvt_corr.x[2])
            elevation = rxpos_gnsstk.elevationGeodetic(sat_pos_gnsstk)
            azimuth = rxpos_gnsstk.azimuthGeodetic(sat_pos_gnsstk)

            # clock offset correction
            clock_offset = clock_samples[idx]            
 
            # L1 troposphere correction
            if antenna_handle.tropModel is not None:
                trop_delay = antenna_handle.tropModel.correction(rxpos_gnsstk, sat_pos_gnsstk, common_time)
            else:
                trop_delay = 0

            sat_clk = sat_xvt_corr.clkbias*const.c # satellite clock bias in m
            
            # we need to add the modeled troposphere and clock offset to mirror the real PR measurement
            pr_final = pr_model + trop_delay + clock_offset - sat_clk
            pr_model_arr.append(pr_final)

            # find doppler shift
            doppler = -np.dot(sat_vel,rx2sat)*freq/const.c 
            dop_model_arr.append(doppler)

        return pr_model_arr, dop_model_arr

    def sat_adj_PC(self, freq, sat_antenna, time_gps, eph_time, sat_x, rx2sat, noPCV=False):
        """ Compute the satellite position including PCO/PCV
        """
        sat_pos = np.array([sat_x[0],sat_x[1],sat_x[2]])
        sat_id=sat_antenna.systemChar+str(sat_antenna.PRN)
        if self.use_obx is True and sat_id in self.quat_sats:
            rot = self.interpolate_quaternion(time_gps, sat_id)
        else:
            sv_attitude = self.sol_sys.satelliteAttitude(eph_time, Position(sat_x[0],sat_x[1],sat_x[2]))
            rot = np.zeros((3,3))
            for ldx in range(3):
                for mdx in range(3):
                    rot[ldx,mdx] = sv_attitude.get_value(ldx,mdx)
        sat_PCO = sat_antenna.getPhaseCenterOffset(freq)
        sat_PCO = np.array([sat_PCO[0],sat_PCO[1],sat_PCO[2]])
        if noPCV is False:
            # transform pointing vector to body-fixed frame
            body_frame_xyz = rot@(-rx2sat)
            nadir_sv = np.rad2deg(np.arccos(body_frame_xyz[2]))
            azim_sv = np.rad2deg(np.arctan2(body_frame_xyz[1], body_frame_xyz[0]))
            sat_PCV = sat_antenna.getPhaseCenterVariation(freq, azim_sv, nadir_sv)
            PCO_body_fixed = (sat_PCO + sat_PCV*rx2sat)/1e3 # +rx2sat instead of -sat2rx
        else:
            PCO_body_fixed = sat_PCO/1e3 # +rx2sat instead of -sat2rx

        sat_pos = sat_pos + rot.T@PCO_body_fixed # transform PCO/V to ECEF frame and add 
        return sat_pos

    def sat_PCV_delay(self, freq, sat_antenna, time_gps, eph_time, sat_x, rx2sat):
        """ Compute the delay caused by satellite PCV
        """
        sat_pos = np.array([sat_x[0],sat_x[1],sat_x[2]])
        sat_id=sat_antenna.systemChar+str(sat_antenna.PRN)
        if self.use_obx is True and sat_id in self.quat_sats:
            rot = self.interpolate_quaternion(time_gps, sat_id)
        else:
            sv_attitude = self.sol_sys.satelliteAttitude(eph_time, Position(sat_x[0],sat_x[1],sat_x[2]))
            rot = np.zeros((3,3))
            for ldx in range(3):
                for mdx in range(3):
                    rot[ldx,mdx] = sv_attitude.get_value(ldx,mdx)

        # transform pointing vector to body-fixed frame
        body_frame_xyz = rot@(-rx2sat)
        nadir_sv = np.rad2deg(np.arccos(body_frame_xyz[2]))
        azim_sv = np.rad2deg(np.arctan2(body_frame_xyz[1], body_frame_xyz[0]))
        sat_PCV = sat_antenna.getPhaseCenterVariation(freq, azim_sv, nadir_sv)
        pcv_delay = -sat_PCV*1e-3
        return pcv_delay

    def compute_ptvec(self, ra, dec, eph_time, frame='ECEF'):
        """ Compute the ECEF pointing vector to the source from right ascension and declination """
        ra_rad = np.deg2rad(ra)
        dec_rad = np.deg2rad(dec)
        eci_ptvec = np.array([np.cos(dec_rad)*np.cos(ra_rad), np.cos(dec_rad)*np.sin(ra_rad), np.sin(dec_rad)])
        if frame=='ECEF':
            R_T2I = self.get_ECEF2ECI(eph_time)
            ecef_ptvec = R_T2I.T @ eci_ptvec
            return ecef_ptvec
        elif frame=='ECI':
            return eci_ptvec

    def get_ECEF2ECI(self, eph_time):
        """ Get transformation ECEF to inertial """
        if eph_time.dMJD() != self.mjd_last:
            self.EO_obj = self.sol_sys.getEOP(eph_time.dMJD())
            ECEF2ECI_mat = self.EO_obj.ECEFtoInertial(eph_time)
            R_T2I = np.zeros((3,3))
            for ldx in range(3):
                for mdx in range(3):
                    R_T2I[ldx,mdx] = ECEF2ECI_mat.get_value(ldx,mdx)
            self.R_T2I = R_T2I
            self.mjd_last = eph_time.dMJD()
        else:
            R_T2I = self.R_T2I

        return R_T2I

    def get_gcrf_posvel(self, pos_itrf, vel_itrf, eph_time, only_pos=False):
        """ Get transformation ECEF to inertial """
        R_T2I = self.get_ECEF2ECI(eph_time)
        pos_gcrf = R_T2I@pos_itrf
        if only_pos is True:
            return pos_gcrf
        else:
            # get Earth rotation accounting for polar motion, dUT1
            xp = self.EO_obj.xp*np.pi/648000
            yp = self.EO_obj.yp*np.pi/648000
            w_earth = 7.2921150e-5
            w_vec = w_earth * np.array([-yp, xp, np.sqrt(1-xp**2 - yp**2)])
            # compute GCRF velocity
            vel_gcrf = R_T2I@vel_itrf-np.cross(w_vec,pos_gcrf)

            return pos_gcrf, vel_gcrf

    def get_gcrf_posvel_astro(self, pos, vel, eph_time):
        """ Transform an ITRF position and velocity to GCRF position and velocity """
        from astropy.coordinates import CartesianDifferential, CartesianRepresentation, GCRS, ITRS
        from astropy.time import Time
        from astropy.utils import iers 
        from os import path
        leap_sec_path = path.join('../sim_vis/','Leap_Second.dat')
        iers_a_file = path.join('../sim_vis/','finals2000A.all')
        iers.LeapSeconds.from_iers_leap_seconds(leap_sec_path)
        iers_a = iers.IERS_A.open(iers_a_file)
        iers.earth_orientation_table.set(iers_a)
        time = Time(eph_time.dMJD(), format='mjd', scale='utc')

        pos_cart = CartesianRepresentation(pos[0], pos[1], pos[2], unit='m')
        vel_cart = CartesianDifferential(vel[0], vel[1], vel[2], unit='m/s')
        coords_itrf = ITRS(pos_cart.with_differentials(vel_cart), obstime=time)
        coords_gcrf = coords_itrf.transform_to(GCRS(obstime=time))
        pos_gcrf = coords_gcrf.cartesian.xyz.to('m').value
        vel_gcrf = coords_gcrf.cartesian.differentials['s'].d_xyz.to('m/s').value

        # get rotation matrix 
        unit_vectors = [
            CartesianRepresentation(1, 0, 0),
            CartesianRepresentation(0, 1, 0),
            CartesianRepresentation(0, 0, 1)
        ]
        
        # Initialize the transformation matrices
        transformation_matrix = np.zeros((3, 3))
        
        # Transform unit vectors from ITRS to GCRS
        for i, unit_vector in enumerate(unit_vectors):
            itrs = ITRS(unit_vector, obstime=time)
            gcrs = itrs.transform_to(GCRS(obstime=time))
            cartesian_gcrs = gcrs.cartesian
            transformation_matrix[:, i] = [cartesian_gcrs.x.value, cartesian_gcrs.y.value, cartesian_gcrs.z.value]

        return pos_gcrf, vel_gcrf, transformation_matrix
  
    def get_sat_vectors(self, sat_id, datetime, eph_time, sat_pos):
        """ Get the satellite attitude. If we have an OBX file, use this. 
            Otherwise, use the simple GNSSTk satellite attitude. """
        if self.use_obx is True and sat_id in self.quat_sats:
            q_interp = self.interpolate_quaternion(datetime, sat_id)
            t_a = q_interp[0,:]
            t_t = q_interp[1,:]
        else:
            Att = self.sol_sys.satelliteAttitude(eph_time, sat_pos)
            t_a = np.array([Att.get_value(0,0),Att.get_value(0,1),Att.get_value(0,2)])
            t_t = np.array([Att.get_value(1,0),Att.get_value(1,1),Att.get_value(1,2)])
        
        return t_a, t_t

    def phase_windup_VLBI(self, sat_id, datetime, eph_time, sat_pos, rx2sat, a_vec):
        """ Compute the carrier phase windup correction for a VLBI antenna
            using the expression from https://link.springer.com/article/10.1007/s10291-008-0112-1
            This does not assume perfect RHCP signals
        """
        t_a, t_t = self.get_sat_vectors(sat_id, datetime, eph_time, sat_pos)
        S_x = np.array([[0, -rx2sat[2], rx2sat[1]], [rx2sat[2], 0, -rx2sat[0]], [-rx2sat[1], rx2sat[0], 0]])
        P = S_x @ S_x.T

        r = P@a_vec
        t = P@t_a + S_x@t_t
        phase_windup = np.arctan2(r.T@S_x.T@t, r.T@t)/(2*np.pi)

        return phase_windup

    def phase_windup_VLBI_BWG(self, antenna_name, sat_id, datetime, eph_time, sat_pos, rx2sat, e_vec, n_vec, u_vec):
        """ Compute the carrier phase windup correction for a VLBI antenna
            using the expression from https://link.springer.com/article/10.1007/s10291-008-0112-1
            This does not assume perfect RHCP signals
        """
        t_a, t_t = self.get_sat_vectors(sat_id, datetime, eph_time, sat_pos)
        phase_windup = pw_reflection_bwg(antenna_name, t_a, t_t, rx2sat, e_vec, n_vec, u_vec)

        return phase_windup

    def phase_windup_GNSS(self, sat_id, datetime, eph_time, sat_pos, \
            rx2sat, east_triple, north_triple):
        """ Compute the carrier phase windup correction for a GNSS antenna """
        t_a, t_t = self.get_sat_vectors(sat_id, datetime, eph_time, sat_pos)
        t_a_proj = np.cross(np.cross(rx2sat,t_a),rx2sat)
        t_t_proj = np.cross(np.cross(rx2sat,t_t),rx2sat)

        r_a = np.array([east_triple[0], east_triple[1], east_triple[2]])
        r_t = np.array([north_triple[0], north_triple[1], north_triple[2]])
        phase_windup = np.arctan2(np.dot(t_t_proj,r_a)+np.dot(t_a_proj,r_t),np.dot(t_a_proj,r_a)-np.dot(t_t_proj,r_t))/(2*np.pi)

        return phase_windup

    def phase_windup_VLBI_ff(self, s_ecef, a_vec):
        """ Compute the carrier phase windup correction for a VLBI antenna
            using the expression from https://link.springer.com/article/10.1007/s10291-008-0112-1
            This does not assume perfect RHCP signals
        """
        z_trf = np.array([0,0,1]) # NCP by definition (TRF)
        S_x = np.array([[0, -s_ecef[2], s_ecef[1]], [s_ecef[2], 0, -s_ecef[0]], [-s_ecef[1], s_ecef[0], 0]])
        P = S_x @ S_x.T

        r = P@a_vec
        t = P@z_trf
        phase_windup = np.arctan2(r.T@S_x.T@t, r.T@t)/(2*np.pi)

        return phase_windup

    def phase_windup_VLBI_ff_BWG(self, antenna_name, s_ecef, e_vec, n_vec, u_vec):
        """ Compute the carrier phase windup correction for a BWG VLBI antenna
            using the expression from https://link.springer.com/article/10.1007/s10291-008-0112-1
            This does not assume perfect RHCP signals
        """
        z_trf = np.array([0,0,1]) # NCP by definition (TRF)
        t_t = np.cross(z_trf, s_vec)
        t_a = np.cross(s_vec, t_t) 
        # for celestial sources, (s x t_a) x s = t_a; (s x t_t) x s = t_t

        phase_windup = pw_reflection_bwg(antenna_name, t_a, t_t, s_ecef, e_vec, n_vec, u_vec)

        return phase_windup

    def phase_windup_GNSS_ff(self, s_ecef, east_triple, north_triple):
        """ Compute the carrier phase windup correction for a GNSS antenna """
        z_trf = np.array([0,0,1]) # NCP by definition (TRF)
        r_a = np.array([east_triple[0], east__triple[1], east_triple[2]])
        r_t = np.array([north_triple[0], north_triple[1], north_triple[2]])

        S_x = np.array([[0, -s_ecef[2], s_ecef[1]], [s_ecef[2], 0, -s_ecef[0]], [-s_ecef[1], s_ecef[0], 0]])
        P = S_x @ S_x.T

        r = P@r_a - S_x@r_t
        t = P@z_trf
        phase_windup = np.arctan2(r.T@S_x.T@t, r.T@t)/(2*np.pi)

        return phase_windup

    def cycle_adj_windup(self, cpw_source_last, phase_windup):
        """ adjust the phase windup to maintain cycle continuity """
        pw_diff = cpw_source_last - phase_windup
        cycle_adj = np.rint(pw_diff)
        phase_windup = phase_windup + cycle_adj
        return phase_windup

    def get_delay_model(self, RSID, common_time, time_gps, sat_antenna, \
                        rxpos1, rxpos2, freq, sat_pos, sat_vel, derivs=False, ant_deriv=1):
        """ Use the modified Jaron and Nothnagel (2019) analytical delay model to compute the VLBI delay.
            Input arguments are in ECEF
            ant_deriv: which antenna to retrieve partials for, 1, 2, or 3 (both)
        """
        eph_time = EphTime(common_time)
        eph_time.setTimeSystem(TimeSystem.UTC)
        # Define an ECI frame of convenience -- correct only for Earth rotation in ECEF
        x0, v0 = ECEF2ECI(0, sat_pos, sat_vel) 
        x1 = rxpos1
        x2, v2 = ECEF2ECI(0, rxpos2, np.array([0,0,0]))
       
        # Compute first-order correction for light-time
        delta_1 = np.linalg.norm(x1-x0)/const.c
        delta_2 = np.linalg.norm(x2-x0)/const.c
        tau = delta_2-delta_1
        
        # adjust time object and find satellite position at this time 
        common_time.addSeconds(-delta_1)
        sat_xvt_mdt1 = self.nav_store.get_xvt(RSID, common_time)
        common_time.addSeconds(delta_1) # reset common time to epoch
        sat_pos_mdt1 = np.array([sat_xvt_mdt1.x[0],sat_xvt_mdt1.x[1],sat_xvt_mdt1.x[2]])
        sat_vel_mdt1 = np.array([sat_xvt_mdt1.v[0],sat_xvt_mdt1.v[1],sat_xvt_mdt1.v[2]])
        sat_pos_mdt1 = self.sat_adj_PC(freq, sat_antenna, time_gps, eph_time, sat_xvt_mdt1.x, 0, noPCV=True)

        # adjust new satellite position for PCO/PCV
        x0_dt1, v0_dt1 = ECEF2ECI(-delta_1, sat_pos_mdt1, sat_vel_mdt1)
        x2_dtau, v2_dtau = ECEF2ECI(tau, rxpos2, np.array([0,0,0]))
        
        # compute adjusted pointing vector
        x_01 = v0_dt1*delta_1 + x0_dt1 - x1
        x0_mag =  np.linalg.norm(x0)
        x1_mag =  np.linalg.norm(x1)
        x01_mag =  np.linalg.norm(x_01)

        # compute gravitational delay
        tg_1 = 2*MU_EARTH/const.c**3*np.log((x0_mag+x1_mag+x01_mag)/(x0_mag+x1_mag-x01_mag)) 
        
        # find modeled pseudorange
        v0_mag = np.linalg.norm(v0)
        g0 = 1/np.sqrt(1-v0_mag**2/const.c**2)
        v2_mag = np.linalg.norm(v2)
        g2 = 1/np.sqrt(1-v2_mag**2/const.c**2)
        
        dt_0 = np.dot(x_01,v0)/const.c**2 - g0/const.c*np.sqrt(x01_mag**2 + np.dot(x_01,v0)**2/const.c**2) - tg_1
        
        # compute station 2 pointing vector
        x_02 = v0_dt1*delta_1 + x0_dt1 + v2_dtau*tau - x2_dtau + (v0-v2)*dt_0
        x2_mag =  np.linalg.norm(x2)
        x02_mag =  np.linalg.norm(x_02)
        if derivs is False:
            tg_2 = 2*MU_EARTH/const.c**3*np.log((x0_mag+x2_mag+x02_mag)/(x0_mag+x2_mag-x02_mag)) 
            dt_2 = -np.dot(x_02,v2)/const.c**2 + g2/const.c*np.sqrt(x02_mag**2 + np.dot(x_02,v2)**2/const.c**2) + tg_2
            delay = dt_2 + dt_0

            # get location of satellite at true transmit time
            common_time.addSeconds(dt_0)
            sat_xvt_corr = self.nav_store.get_xvt(RSID, common_time)
            common_time.addSeconds(-dt_0)
            
            # correct for LOS change due to Earth rotation -- this is where the rx 'sees' the satellite
            # NB: neglecting change in rx2sat vector 
            eph_time.addSeconds(dt_0)
            sat_pos_final = self.sat_adj_PC(freq, sat_antenna, time_gps, eph_time, sat_xvt_corr.x, 0, noPCV=True)
            sat_vel_final = np.array([sat_xvt_corr.v[0],sat_xvt_corr.v[1],sat_xvt_corr.v[2]])
            x0_final, v0_final = ECEF2ECI(dt_0, sat_pos_mdt1, sat_vel_mdt1, no_velcorr=True)
            for idx in range(3):
                sat_xvt_corr.x[idx] = x0_final[idx]
                sat_xvt_corr.v[idx] = v0_final[idx]
            return delay*const.c, sat_xvt_corr
        else: 
            # get partial derivatives
            # NB: neglecting partial for gravitational delay
            if ant_deriv == 1:
                d_delta1_dx1 = (x1-x0)/(const.c*np.linalg.norm(x1-x0))
                d_tau_dx1 = -d_delta1_dx1
                d_dt0_dx1 = np.zeros(3)
                d_dt2_dx1 = np.zeros(3)
                eye_mat = np.eye(3)
                for idx in range(3):
                    dx01_dx1_idx = v0_dt1*d_delta1_dx1[idx] - eye_mat[idx,:]
                    d_dt0_dx1[idx] = np.dot(dx01_dx1_idx,v0)/const.c**2 - \
                            g0*(np.dot(dx01_dx1_idx,x_01) + np.dot(x_01,v0)/const.c**2*np.dot(dx01_dx1_idx,v0))\
                            /(const.c*np.sqrt(x01_mag**2 + np.dot(x_01,v0)**2/const.c**2))
                for idx in range(3):
                    dx02_dx1_idx = v0_dt1*d_delta1_dx1[idx] + v2_dtau*d_tau_dx1[idx] + (v0-v2)*d_dt0_dx1[idx]
                    d_dt2_dx1[idx] = -np.dot(dx02_dx1_idx,v2)/const.c**2 + \
                            g2*(np.dot(dx02_dx1_idx,x_02) + np.dot(x_02,v2)/const.c**2*np.dot(dx02_dx1_idx,v2))\
                            /(const.c*np.sqrt(x02_mag**2 + np.dot(x_02,v2)**2/const.c**2))
                dtau_dx1 = d_dt0_dx1 + d_dt2_dx1
                return dtau_dx1*const.c

            elif ant_deriv == 2:
                d_delta2_dx2 = (x2-x0)/(const.c*np.linalg.norm(x2-x0))
                d_tau_dx2 = d_delta2_dx2
                d_dt2_dx2 = np.zeros(3)
                eye_mat = np.eye(3)
                for idx in range(3):
                    dx02_dx2_idx = v2_dtau*d_tau_dx2[idx] - eye_mat[idx,:]
                    d_dt2_dx2[idx] = -np.dot(dx02_dx2_idx,v2)/const.c**2 + \
                            g2*(np.dot(dx02_dx2_idx,x_02) + np.dot(x_02,v2)/const.c**2*np.dot(dx02_dx2_idx,v2))\
                            /(const.c*np.sqrt(x02_mag**2 + np.dot(x_02,v2)**2/const.c**2))
                dtau_dx2 = d_dt2_dx2
                return dtau_dx2*const.c
            else:
                raise ValueError('Argument ant_deriv must be 1 or 2')

    def get_delay_farfield(self, s_vec, eph_time, rxpos1, rxpos2, derivs=False, ant_deriv=1):
        """ Use the Petrov/Kopeikin (2001) relativistic delay model to compute the VLBI delay.
            ant_deriv: which antenna to retrieve partials for, 1 or 2
        """
        # Initialize variables

        planets = [1,2,4,5,6,7,8] # Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune
        rxpos1_crs, rxvel1_crs = self.get_gcrf_posvel(rxpos1, [0,0,0], eph_time)
        rxpos2_crs, rxvel2_crs = self.get_gcrf_posvel(rxpos2, [0,0,0], eph_time)
        eph_time.convertSystemTo(TimeSystem.TDB)
        mjd_tdb = eph_time.dMJD() 
        set_km = True
        #self.sol_sys.relativeInertialPositionVelocity(mjd_tdb, self.sol_sys.idEarth, self.sol_sys.idSolarSystemBarycenter, earth_posvel, set_km)
        earth_posvel = self.sol_sys.relativeInertialPositionVelocityPyWrapper(mjd_tdb, self.sol_sys.idEarth, self.sol_sys.idSolarSystemBarycenter, set_km)
        earth_pos = np.array(earth_posvel[:3])*1e3
        earth_vel = np.array(earth_posvel[3:])*1e3/86400

        if derivs is False:
            # Loop over planets
            rxpos1_brs = rxpos1_crs + earth_pos
            rxpos2_brs = rxpos2_crs + earth_pos
            tau_grav = 0.0
            for j1 in planets:
                mu = planet_mu[j1]
                #mjd_tdb = MJD(eph_time.dMJD(), TimeSystem.TDB)
                mjd_tdb = eph_time.dMJD() 
                
                # Get coordinates of the planet at the moment of time of arrival
                planet_posvel = [0,0,0,0,0,0]
                #self.sol_sys.relativeInertialPositionVelocity(mjd_tdb, j1, self.sol_sys.idSolarSystemBarycenter, planet_posvel, set_km)
                planet_posvel = self.sol_sys.relativeInertialPositionVelocityPyWrapper(mjd_tdb, j1, self.sol_sys.idSolarSystemBarycenter, set_km)
                planet_pos = np.array(planet_posvel[:3])*1e3
                planet_vel = np.array(planet_posvel[3:])*1e3/86400
            
                # Solve light cone equation (one iteration)
                r1a = rxpos1_brs - planet_pos 
                r2a = rxpos2_brs - planet_pos
                tau_retr1 = -np.linalg.norm(r1a) / const.c
                tau_retr2 = -np.linalg.norm(r2a) / const.c
            
                # Compute position of the gravitating body at retarded moment of time
                planet_pos1 = planet_pos + tau_retr1*planet_vel
                r1a = rxpos1_brs - planet_pos1
                r1a_len = np.linalg.norm(r1a)
            
                planet_pos2 = planet_pos + tau_retr2*planet_vel
                r2a = rxpos2_brs - planet_pos2
                r2a_len = np.linalg.norm(r2a)
            
                tau_grav += (2.0 * mu / const.c**3 * (1.0 + np.dot(s_vec, planet_vel) / const.c) *
                            np.log((r1a_len * (1.0 + np.dot(s_vec, r1a))) /
                                   (r2a_len * (1.0 + np.dot(s_vec, r2a)))))
            
            # Final computations
            b_crs = rxpos2_crs-rxpos1_crs
            #self.sol_sys.relativeInertialPositionVelocity(mjd_tdb, self.sol_sys.idSun, self.sol_sys.idSolarSystemBarycenter, sun_posvel, set_km)
            sun_posvel = [0,0,0,0,0,0]
            sun_posvel = self.sol_sys.relativeInertialPositionVelocityPyWrapper(mjd_tdb, self.sol_sys.idSun, self.sol_sys.idSolarSystemBarycenter, set_km)
            sun_pos = np.array(sun_posvel[:3])*1e3
            dist_sun_earth = np.linalg.norm(sun_pos-earth_pos) 
            
            # Compute geometric path delay
            tau_geom = (-np.dot(s_vec, b_crs) / const.c * \
                       (1.0 - 2.0 * MU_SUN / (dist_sun_earth * const.c**2) - np.linalg.norm(earth_vel) / (2.0 * const.c**2) - \
                        np.dot(earth_vel, rxvel2_crs) / const.c**2) - \
                       np.dot(earth_vel, b_crs) / const.c**2 * (1.0 + np.dot(s_vec, earth_vel) / (2.0 * const.c) ) + tau_grav) \
                       / (1.0 + np.dot(s_vec, earth_vel + rxvel2_crs)  / const.c)
            #tau_test = -np.dot(s_vec, b_crs)/const.c
            eph_time.convertSystemTo(TimeSystem.UTC) # convert time system back for other functions
            return tau_geom*const.c
        else: 
            # get partial derivatives
            eph_time.convertSystemTo(TimeSystem.UTC) 
            tau_der_sta1 = np.zeros(3)
            C_FACTOR = 1 / (np.dot(s_vec, earth_vel) + np.dot(s_vec, rxvel2_crs) + const.c)
            R_mat = self.get_ECEF2ECI(eph_time)
            for j2 in range(3):
                tau_der_sta1[j2] = (C_FACTOR * np.dot(R_mat[:, j2], s_vec) +
                                    C_FACTOR / const.c * np.dot(R_mat[:, j2], earth_vel))
            if ant_deriv == 1:
                return tau_der_sta1*const.c

            elif ant_deriv == 2:
                return -tau_der_sta1*const.c
            else:
                raise ValueError('Argument ant_deriv must be 1 or 2')

    def get_delay_iter(self, RSID, common_time, time_gps, sat_antenna, \
                        rxpos1, rxpos2, freq, sat_pos, sat_vel):
        """ Use an iterative approach to get a high-accuracy group delay model 
        """
        eph_time = EphTime(common_time)
        eph_time.setTimeSystem(TimeSystem.UTC)
        pr_rx1, _ = self.get_pr_iter(RSID, common_time, time_gps, sat_antenna, rxpos1, freq, sat_pos, sat_vel)
        t0 = pr_rx1/const.c
        common_time.addSeconds(-t0)
        sat_xvt_mt0 = self.nav_store.get_xvt(RSID, common_time)
        common_time.addSeconds(t0) # reset common time to epoch
        sat_vel_mt0 = np.array([sat_xvt_mt0.v[0],sat_xvt_mt0.v[1],sat_xvt_mt0.v[2]])
        time_mt0 = time_gps - np.timedelta64(int(t0*1e9),'ns')
        eph_time.addSeconds(-t0)
        sat_pos_mt0 = self.sat_adj_PC(freq, sat_antenna, time_mt0, eph_time, sat_xvt_mt0.x, 0, noPCV=True)
        satpos_gcrf, satvel_gcrf = self.get_gcrf_posvel(sat_pos_mt0, sat_vel_mt0, eph_time)
        #satpos_gcrf, satvel_gcrf = ECEF2ECI(-t0, sat_pos_mt0, sat_vel_mt0)

        # now we solve for the down leg -- t0 to t2
        rxpos_gcrf, rxvel_gcrf = self.get_gcrf_posvel(rxpos2, [0,0,0], eph_time)
        #rxpos_gcrf, rxvel_gcrf = ECEF2ECI(-t0, rxpos2, [0,0,0])
        t2_init = np.linalg.norm(rxpos_gcrf-satpos_gcrf)/const.c
        r_0e_mag = np.linalg.norm(satpos_gcrf)

        if USE_SUN is True:
            pos_sun = self.sol_sys.solarPosition(eph_time)
            sun_pos_itrf = np.array([pos_sun[0], pos_sun[1], pos_sun[2]])
            sun_gcrf = self.get_gcrf_posvel(sun_pos_itrf, 0, eph_time, only_pos=True)
            r_0s = satpos_gcrf-sun_gcrf
            r_0s_mag = np.linalg.norm(r_0s)

        dT_2 = np.inf
        t2 = t2_init
        stop_crit = 1e-14 # .01 picoseconds
        while abs(dT_2) > stop_crit:
            eph_time.addSeconds(t2)
            rxpos_gcrf, rxvel_gcrf = self.get_gcrf_posvel(rxpos2, [0,0,0], eph_time)
            #rxpos_gcrf, rxvel_gcrf = ECEF2ECI(t2-t0, rxpos2, [0,0,0])
            x_02 = rxpos_gcrf - satpos_gcrf
            p_02 = np.dot(x_02/np.linalg.norm(x_02), rxvel_gcrf)
            r_2e_mag = np.linalg.norm(rxpos_gcrf)
            r_02e_mag = np.linalg.norm(satpos_gcrf-rxpos_gcrf)
            if USE_SUN is True:
                pos_sun = self.sol_sys.solarPosition(eph_time)
                sun_pos_itrf = np.array([pos_sun[0], pos_sun[1], pos_sun[2]])
                sun_gcrf = self.get_gcrf_posvel(sun_pos_itrf, 0, eph_time, only_pos=True)
                r_2s = rxpos_gcrf-sun_gcrf
                r_2s_mag = np.linalg.norm(r_2s)
                r_02s_mag = np.linalg.norm(r_2s-r_0s)
                rlt_02 = 2*MU_EARTH/const.c**3*np.log((r_0e_mag+r_2e_mag+r_02e_mag)/(r_0e_mag+r_2e_mag-r_02e_mag))\
                        +2*MU_SUN/const.c**3*np.log((r_0s_mag+r_2s_mag+r_02s_mag+2*MU_SUN/const.c**2)/(r_0s_mag+r_2s_mag-r_02s_mag+2*MU_SUN/const.c**2))
            else:
                rlt_02 = 2*MU_EARTH/const.c**3*np.log((r_0e_mag+r_2e_mag+r_02e_mag)/(r_0e_mag+r_2e_mag-r_02e_mag))
            dT_2 = -(t2 - np.linalg.norm(x_02)/const.c - rlt_02)/(1-p_02/const.c)
            eph_time.addSeconds(-t2)
            t2 = t2 + dT_2
        delay = (t2*const.c - pr_rx1)*(1-L_G)
  
        # correct for LOS change due to Earth rotation -- this is where the rx 'sees' the satellite
        # NB: neglecting change in rx2sat vector 
        x0_final, v0_final = ECEF2ECI(-pr_rx1/const.c, sat_pos_mt0, sat_vel_mt0, no_velcorr=True)
        sat_xvt_corr = sat_xvt_mt0
        for idx in range(3):
            sat_xvt_corr.x[idx] = x0_final[idx]
            sat_xvt_corr.v[idx] = v0_final[idx]
        return delay, sat_xvt_corr

    def get_pr_model(self, RSID, common_time, time_gps, sat_antenna, rxpos, freq, sat_pos, sat_vel, derivs=False, PCV=True):
        """ Use the modified Jaron and Nothnagel (2019) analytical delay model to compute the pseudorange.
            Input arguments are in ECEF
        """
        eph_time = EphTime(common_time)
        eph_time.setTimeSystem(TimeSystem.UTC)
        # Define an ECI frame of convenience -- correct only for Earth rotation in ECEF
        dt = 0
        x0, v0 = ECEF2ECI(dt, sat_pos, sat_vel)
        x1 = rxpos
    
        # Compute first-order correction for light-time
        delta_1 = np.linalg.norm(x1-x0)/const.c
    
        # adjust time object and find satellite position at this time
        common_time.addSeconds(-delta_1)
        sat_xvt_mdt1 = self.nav_store.get_xvt(RSID, common_time)
        common_time.addSeconds(delta_1) # reset common time to epoch
        sat_pos_mdt1 = np.array([sat_xvt_mdt1.x[0],sat_xvt_mdt1.x[1],sat_xvt_mdt1.x[2]])
        sat_vel_mdt1 = np.array([sat_xvt_mdt1.v[0],sat_xvt_mdt1.v[1],sat_xvt_mdt1.v[2]])
    
        # receiver to satellite vector
        rx2sat_mdt1 = sat_pos_mdt1 - rxpos
        rx2sat_mdt1 /= np.linalg.norm(rx2sat_mdt1) # normalize vector
        
        # adjust new satellite position for PCO/PCV
        #eph_time.addSeconds(-delta_1)
        if PCV:
            sat_pos_mdt1 = self.sat_adj_PC(freq, sat_antenna, time_gps, eph_time, sat_xvt_mdt1.x, rx2sat_mdt1)
        x0_dt1, v0_dt1 = ECEF2ECI(-delta_1, sat_pos_mdt1, sat_vel_mdt1)
        #eph_time.addSeconds(delta_1)
        
        # compute adjusted pointing vector
        x_01 = v0_dt1*delta_1 + x0_dt1 - x1
        x0_mag =  np.linalg.norm(x0)
        x1_mag =  np.linalg.norm(x1)
        x01_mag =  np.linalg.norm(x_01)

        # compute gravitational delay
        tg_1 = 2*MU_EARTH/const.c**3*np.log((x0_mag+x1_mag+x01_mag)/(x0_mag+x1_mag-x01_mag)) 
    
        # find modeled pseudorange
        v0_mag = np.linalg.norm(v0)
        g0 = 1/np.sqrt(1-v0_mag**2/const.c**2)

        if derivs is False:
            # get delay model
            dt_1 = -np.dot(x_01,v0)/const.c**2 + g0/const.c*np.sqrt(x01_mag**2 + np.dot(x_01,v0)**2/const.c**2) + tg_1
            pr_model = dt_1*const.c

            # get location of satellite at true transmit time
            common_time.addSeconds(-dt_1)
            sat_xvt_corr = self.nav_store.get_xvt(RSID, common_time)
            common_time.addSeconds(dt_1)
            
            # correct for LOS change due to Earth rotation -- this is where the rx 'sees' the satellite
            # NB: neglecting change in rx2sat vector 
            if PCV:
                sat_pos_final = self.sat_adj_PC(freq, sat_antenna, time_gps, eph_time, sat_xvt_corr.x, rx2sat_mdt1)
            else:
                sat_pos_final = np.array([sat_xvt_corr.x[0],sat_xvt_corr.x[1],sat_xvt_corr.x[2]])

            sat_vel_final = np.array([sat_xvt_corr.v[0],sat_xvt_corr.v[1],sat_xvt_corr.v[2]])
            x0_final, v0_final = ECEF2ECI(-dt_1, sat_pos_final, sat_vel_final, no_velcorr=True)
            for idx in range(3):
                sat_xvt_corr.x[idx] = x0_final[idx]
                sat_xvt_corr.v[idx] = v0_final[idx]
            
            return pr_model, sat_xvt_corr
        else: 
            # get partial derivatives
            # NB: neglecting partial for gravitational delay
            d_delta1_dx1 = (x1-x0)/(const.c*np.linalg.norm(x1-x0))
            d_dt_dx1 = np.zeros(3)
            eye_mat = np.eye(3)
            for idx in range(3):
                dx01_dx1_idx = v0_dt1*d_delta1_dx1[idx] - eye_mat[idx,:]
                d_dt_dx1[idx] = -np.dot(dx01_dx1_idx,v0)/const.c**2 + \
                        g0*(np.dot(dx01_dx1_idx,x_01) + np.dot(x_01,v0)/const.c**2*np.dot(dx01_dx1_idx,v0))\
                        /(const.c*np.sqrt(x01_mag**2 + np.dot(x_01,v0)**2/const.c**2))
            pr_deriv = d_dt_dx1*const.c

            return pr_deriv

    def get_pr_iter(self, RSID, common_time, time_gps, sat_antenna, rxpos, freq, sat_pos, sat_vel):
        """ Use the iterative model from Moyer (2003), chapter 8 to compute the pseudorange.
            Input arguments are in ECEF
        """
        eph_time = EphTime(common_time)
        eph_time.setTimeSystem(TimeSystem.UTC)
        rxpos_gcrf, rxvel_gcrf = self.get_gcrf_posvel(rxpos, [0,0,0], eph_time)
        satpos_gcrf, satvel_gcrf = self.get_gcrf_posvel(sat_pos, sat_vel, eph_time)
        #rxpos_gcrf, rxvel_gcrf = ECEF2ECI(0, rxpos, [0,0,0])
        #satpos_gcrf, satvel_gcrf = ECEF2ECI(0, sat_pos, sat_vel)
        t0_init = np.linalg.norm(rxpos_gcrf-satpos_gcrf)/const.c

        #set_km = True
        if USE_SUN is True:
            eph_time.convertSystemTo(TimeSystem.TDB)
            mjd_tdb = eph_time.dMJD() 
            set_km = True
            sun_posvel = self.sol_sys.relativeInertialPositionVelocityPyWrapper(mjd_tdb, self.sol_sys.idEarth, self.sol_sys.idSolarSystemBarycenter, set_km)
            sun_gcrf = np.array(sun_posvel[:3])*1e3
            eph_time.convertSystemTo(TimeSystem.UTC)
            r_1s = rxpos_gcrf-sun_gcrf
            r_1s_mag = np.linalg.norm(r_1s)
        r_1e_mag = np.linalg.norm(rxpos_gcrf)
        
        dT_0 = np.inf
        t0 = t0_init
        stop_crit = 1e-14 # .01 picoseconds
        while abs(dT_0) > stop_crit: 
            common_time.addSeconds(-t0)
            eph_time.addSeconds(-t0)
            sat_xvt_mt0 = self.nav_store.get_xvt(RSID, common_time)
            sat_x_mt0 = np.array([sat_xvt_mt0.x[0],sat_xvt_mt0.x[1],sat_xvt_mt0.x[2]])

            # get PCO
            time_t0 = time_gps - np.timedelta64(int(t0*1e9),'ns')
            rx2sat_mt0 = sat_x_mt0 - rxpos
            rx2sat_mt0 /= np.linalg.norm(rx2sat_mt0) # normalize vector
            sat_pos_itrf = self.sat_adj_PC(freq, sat_antenna, time_t0, eph_time, sat_xvt_mt0.x, rx2sat_mt0)
            sat_vel_itrf = np.array([sat_xvt_mt0.v[0],sat_xvt_mt0.v[1],sat_xvt_mt0.v[2]])

            # transform to GCRF
            satpos_gcrf, satvel_gcrf = self.get_gcrf_posvel(sat_pos_itrf, sat_vel_itrf, eph_time)
            #satpos_gcrf, satvel_gcrf = ECEF2ECI(-t0, sat_pos_itrf, sat_vel_itrf)
            x_01 = rxpos_gcrf - satpos_gcrf
            p_01 = np.dot(x_01/np.linalg.norm(x_01), satvel_gcrf)

            # prepare relativistic correction
            r_0e_mag = np.linalg.norm(satpos_gcrf)
            r_01e_mag = np.linalg.norm(satpos_gcrf-rxpos_gcrf)
            if USE_SUN is True:
                eph_time.convertSystemTo(TimeSystem.TDB)
                mjd_tdb = eph_time.dMJD() 
                sun_posvel = self.sol_sys.relativeInertialPositionVelocityPyWrapper(mjd_tdb, self.sol_sys.idEarth, self.sol_sys.idSolarSystemBarycenter, set_km)
                sun_gcrf = np.array(sun_posvel[:3])*1e3
                eph_time.convertSystemTo(TimeSystem.UTC)
                r_0s = satpos_gcrf-sun_gcrf
                r_0s_mag = np.linalg.norm(r_0s)
                r_01s_mag = np.linalg.norm(r_1s-r_0s)
                rlt_01 = 2*MU_EARTH/const.c**3*np.log((r_0e_mag+r_1e_mag+r_01e_mag)/(r_0e_mag+r_1e_mag-r_01e_mag))\
                        +2*MU_SUN/const.c**3*np.log((r_0s_mag+r_1s_mag+r_01s_mag+2*MU_SUN/const.c**2)/(r_0s_mag+r_1s_mag-r_01s_mag+2*MU_SUN/const.c**2))
            else:
                rlt_01 = 2*MU_EARTH/const.c**3*np.log((r_0e_mag+r_1e_mag+r_01e_mag)/(r_0e_mag+r_1e_mag-r_01e_mag))

            # Newton-Raphson step
            dT_0 = -(t0 - np.linalg.norm(x_01)/const.c - rlt_01)/(1-p_01/const.c)
            common_time.addSeconds(t0) # restore common time to epoch
            eph_time.addSeconds(t0)
            t0 = t0 + dT_0
        
        sat_xvt_apparent = sat_xvt_mt0
        x0_final, v0_final = ECEF2ECI(-t0, sat_pos_itrf, sat_vel_itrf, no_velcorr=True)
        for idx in range(3):
            sat_xvt_apparent.x[idx] = x0_final[idx]
            sat_xvt_apparent.v[idx] = v0_final[idx]
        pr = t0*const.c
        return pr, sat_xvt_apparent

    def compute_analytical_jac(self, antenna_handle, clock_state, clock_poly_length, \
            trop_state, trop_poly_length, disb_state, ao_ant = None, phase=False, phase_clock_state=[], phase_disb_state=[], phase_only=False):
        """ Compute an analytical Jacobian using nav ephemeris and analytical delay model.
        """
        isCOM = True
        times_gps = antenna_handle.times_gps
        f1 = 1575.42*1e6
        freq1_ant = 'G01' # antex frequency for receiver antenna
        if self.iono_free: # iono-free only for GPS, ref frequency for precise range
            if self.iono_freq == 'L2':
                f2 = 1227.60*1e6
                freq2_ant = 'G02'
            elif self.iono_freq == 'L5': 
                f2 = 1176.45*1e6
                freq2_ant = 'G05'
            if antennaPCOData.nFreq <= 4 and self.iono_freq == 'L5':
                freq2_ant = 'G02' 
        else:
            freq2 ='0'
            f2 = 0

        if self.global_linear_clock is True:
            idx_start = 1
        elif self.global_quadratic_clock is True:
            idx_start = 2
        else:
            idx_start = 0

        if antenna_handle.is_VLBI is False:
            # RX PCO correction
            antennaPCOData = antenna_handle.antenna_PCO
            offset_L1 = antennaPCOData.getPhaseCenterOffset(freq1_ant)
            offset_L1 = np.array([offset_L1[0],offset_L1[1],offset_L1[2]])/1e3 # convert to m
            if self.iono_free is True:
                offset_dual = antennaPCOData.getPhaseCenterOffset(freq2_ant)
                offset_dual = np.array([offset_dual[0],offset_dual[1],offset_dual[2]])/1e3 # convert to m

        if phase is True: # and antenna_handle.is_VLBI is False: # need to account for carrier phase windup
            wavelength_1 = const.c/f1
            if self.iono_free:
                wavelength_2 = const.c/f2
        
        if antenna_handle.is_VLBI is True:
            a_vec = antenna_handle.calc_VLBI_mount_vec()

        #if phase is True and len(phase_clock_samples)>0:
        #    # adjust the receiver time by the clock bias, use phase clock preferentially
        #    times_adj = [dt + datetime.timedelta(seconds=-adj/const.c) for dt, adj in zip(times_gps, phase_clock_samples)]
        #else:
        #    times_adj = [dt + datetime.timedelta(seconds=-adj/const.c) for dt, adj in zip(times_gps, clock_samples)]
        
        times_common = date_to_common(times_gps, 'GPS')
        if antenna_handle.estimate_ao is True:
            len_ao = 1
        else:
            len_ao = 0

        if antenna_handle.estimate_grav_def is True:
            len_grav = 2
        else:
            len_grav = 0

        epoch0 = times_gps[0]
        if phase_only is False:
            analytical_jacobian = np.zeros((len(times_gps), 3+len(clock_state)+len(trop_state)+len(phase_clock_state)+len_ao+len_grav+len(disb_state)+len(phase_disb_state)))
        else:
            analytical_jacobian = np.zeros((len(times_gps), 3+len(phase_clock_state)+len(trop_state)+len_ao+len_grav+len(phase_disb_state)))

        for idx, common_time in enumerate(times_common):
            eph_time = EphTime(common_time)
            eph_time.setTimeSystem(TimeSystem.UTC)
            sat_id = self.source_time_dict[times_gps[idx]]
            sat_antenna = self.antenna_map[str(sat_id)]  
            RSID = RinexSatID(str(sat_id))
            rxpos = antenna_handle.pos_series[idx,:]
          
            system = RSID.systemString()
            if system == 'GPS': # ref frequencies for precise range
                freq1 = 'G01' # GPS L1
                if self.iono_free:
                    if self.iono_freq == 'L2':
                        freq2 = 'G02'
                    elif self.iono_freq == 'L5':
                        freq2 = 'G02' # only G02 available, no G05 for satellites
            elif system == 'Galileo':
                freq1 = 'E01'
                if self.iono_free:
                    if self.iono_freq == 'L2':
                        raise ValueError('No L2 frequency for BeiDou -- should not be here!')
                    elif self.iono_freq == 'L5':
                        freq2 = 'E05'
            elif system == 'BeiDou':
                freq1 = 'C01' 
                if self.iono_free:
                    if self.iono_freq == 'L2':
                        raise ValueError('No L2 frequency for BeiDou -- should not be here!')
                    elif self.iono_freq == 'L5':
                        freq2 = 'C05'
            
            # find satellite position, pointing vector at receive time
            sat_xvt = self.nav_store.get_xvt(RSID, common_time)
            rxpos_gnsstk = Position(rxpos[0], rxpos[1], rxpos[2])
            sat_pos_gnsstk = Position(sat_xvt.x[0], sat_xvt.x[1], sat_xvt.x[2])
            elevation = rxpos_gnsstk.elevationGeodetic(sat_pos_gnsstk)
            azimuth = rxpos_gnsstk.azimuthGeodetic(sat_pos_gnsstk)
            sat_pos = np.array([sat_xvt.x[0],sat_xvt.x[1],sat_xvt.x[2]])
            sat_vel = np.array([sat_xvt.v[0],sat_xvt.v[1],sat_xvt.v[2]])
            rx2sat = sat_pos - rxpos
            rx2sat /= np.linalg.norm(rx2sat) # normalize vector
            
            if antenna_handle.is_VLBI is True: 
                # compute the geometric model for the VLBI antenna
                if antenna_handle.point_ra_dec_dict is not None:
                    ra, dec = antenna_handle.point_ra_dec_dict[times_gps[idx]]
                    rx2sat = self.compute_ptvec(ra, dec, eph_time)
                point_vec = rx2sat-a_vec*np.dot(rx2sat,a_vec)

                if antenna_handle.estimate_ao is True:
                    fb_vec = ao_ant*point_vec/np.linalg.norm(point_vec)
                    ao_jac = -np.sqrt(1-np.dot(a_vec,rx2sat)**2)
                else:
                    fb_vec = antenna_handle.axis_offset*point_vec/np.linalg.norm(point_vec)

                # shift rxpos by offset
                rxpos_L1 = rxpos + fb_vec
                if self.iono_free:
                   rxpos_dual = rxpos + fb_vec
            else:
                # compute antenna PCV
                ROT = antenna_handle.R_mat # rotation from NEU to XYZ (ECEF) 
                try: PCV_L1 = antennaPCOData.getPhaseCenterVariation(freq1_ant, azimuth, elevation)
                except: PCV_L1 = np.zeros(3)
                rxpos_L1 = rxpos + ROT@(offset_L1) - PCV_L1*1e-3*rx2sat
                if self.iono_free:
                    PCV_dual = antennaPCOData.getPhaseCenterVariation(freq2_ant, azimuth, elevation)
                    rxpos_dual = rxpos + ROT@(offset_dual) - PCV_dual*1e-3*rx2sat

            sat_pos_L1 = self.sat_adj_PC(freq1, sat_antenna, times_gps[idx], eph_time, sat_xvt.x, rx2sat)
            if self.iono_free:
                sat_pos_dual = self.sat_adj_PC(freq2, sat_antenna, times_gps[idx], eph_time, sat_xvt.x, rx2sat)

            pr_derivs = self.get_pr_model(RSID, common_time, times_gps[idx], \
                    sat_antenna, rxpos_L1, freq1, sat_pos_L1, sat_vel, derivs=True)
            if self.iono_free:
                pr_derivs_dual = self.get_pr_model(RSID, common_time, times_gps[idx], \
                        sat_antenna, rxpos_dual, freq2, sat_pos_dual, sat_vel, derivs=True)
                for ndx in range(3):
                    pr_derivs[ndx] = float(IonosphereFreeRange([f1, f2], [pr_derivs[ndx], pr_derivs_dual[ndx]])) 
            
            analytical_jacobian[idx, :3] = pr_derivs 
          
            # get clock derivatives
            if phase_only is False and len(clock_state)>0:
                clock_jac = np.zeros(len(clock_state))
                if self.global_linear_clock is True or self.global_quadratic_clock is True:
                    clock_jac[:idx_start] = global_poly_jac_at_epoch(clock_state[:idx_start], \
                            times_gps[idx], antenna_handle.times_gps[0], antenna_handle.times_gps[-1])
                    analytical_jacobian[idx, 3:3+idx_start] = clock_jac[:idx_start] 
                if self.stochastic_clock is False and clock_poly_length > 0:
                    clock_jac[idx_start:] = poly_jac_at_epoch(clock_state[idx_start:], clock_poly_length, times_gps[idx], epoch0)
                    analytical_jacobian[idx, 3+idx_start:3+len(clock_state)] = clock_jac[idx_start:] 
                elif self.stochastic_clock is True:
                    clock_idx = np.argwhere(antenna_handle.clock_times == times_gps[idx])
                    analytical_jacobian[idx, 3+idx_start+clock_idx] = 1
                else:
                    analytical_jacobian[idx, 3+idx_start] = 1 # bulk clock offset stored after global clock coefficients

            if antenna_handle.estimate_ao is True:
                analytical_jacobian[idx, 3+len(clock_state)] = ao_jac

            if antenna_handle.estimate_grav_def is True:
                #analytical_jacobian[idx, 3+len(clock_state)+len_ao] = np.sin(np.deg2rad(elevation))
                analytical_jacobian[idx, 3+len(clock_state)+len_ao] = np.deg2rad(elevation)
                #analytical_jacobian[idx, 3+len(clock_state)+len_ao+1] = np.cos(np.deg2rad(elevation))
                analytical_jacobian[idx, 3+len(clock_state)+len_ao+1] = np.deg2rad(elevation)**2

            if len(trop_state)>0:
                trop_delay = antenna_handle.tropModel.correction(rxpos_gnsstk, sat_pos_gnsstk, common_time)
                map_fcn = antenna_handle.tropModel.wet_mapping_function(elevation)
                if self.stochastic_trop is False:
                    trop_jac = poly_jac_at_epoch(trop_state, trop_poly_length, times_gps[idx], epoch0)
                    analytical_jacobian[idx, 3+len_ao+len_grav+len(clock_state):3+len_ao+len_grav+len(clock_state)+len(trop_state)] = trop_jac*map_fcn 
                else:
                    trop_idx = np.argwhere(antenna_handle.trop_times == times_gps[idx])
                    analytical_jacobian[idx, 3+len_ao+len_grav+len(clock_state)+trop_idx] = map_fcn

            if self.estimate_disb is True or self.estimate_phase_disb is True:
                idx_disb = np.argwhere(self.disb_systems==str(sat_id)[0])
                if len(idx_disb)>0:
                    idx_disb = idx_disb[0][0]
                else:
                    idx_disb = None

            if len(disb_state)>0 and idx_disb is not None:
                analytical_jacobian[idx, 3+len_ao+len_grav+len(clock_state)+len(trop_state)+idx_disb] = 1

            if phase is True and len(phase_clock_state)>0:
                phase_clock_jac = np.zeros(len(phase_clock_state))
                if self.global_linear_clock is True or self.global_quadratic_clock is True:
                    phase_clock_jac[:idx_start] = global_poly_jac_at_epoch(phase_clock_state[:idx_start], \
                            times_gps[idx], antenna_handle.times_gps[0], antenna_handle.times_gps[-1])
                    analytical_jacobian[idx, 3+len(clock_state)+len_ao+len_grav+len(trop_state)+len(disb_state):\
                            3+len(clock_state)+len_ao+len_grav+len(trop_state)+len(disb_state)+idx_start] = phase_clock_jac[:idx_start]
                if self.stochastic_clock is False and clock_poly_length>0: 
                    phase_clock_jac[idx_start:] = poly_jac_at_epoch(phase_clock_state[idx_start:], clock_poly_length, \
                            times_gps[idx], antenna_handle.phase_clock_start)
                    analytical_jacobian[idx, 3+len(clock_state)+len_ao+len_grav+len(trop_state)+len(disb_state)+idx_start:\
                            3+len(clock_state)+len_ao+len_grav+len(trop_state)+len(disb_state)+len(phase_clock_state)] = phase_clock_jac[idx_start:]
                elif self.stochastic_clock is True:
                    phase_clock_idx = np.argwhere(antenna_handle.phase_clock_times == times_gps[idx])
                    analytical_jacobian[idx, 3+len(clock_state)+len_ao+len_grav+len(trop_state)+len(disb_state)+idx_start+phase_clock_idx] = 1
                else:
                    # bulk clock offset stored after global clock coefficients
                    analytical_jacobian[idx, 3+len(clock_state)+len_ao+len_grav+len(trop_state)+len(disb_state)+idx_start] = 1 

                if len(phase_disb_state)>0 and idx_disb is not None:
                    analytical_jacobian[idx, 3+len_ao+len_grav+len(clock_state)+len(trop_state)+len(disb_state)+len(phase_clock_state)+idx_disb] = 1

        return analytical_jacobian

    def compute_analytical_jac_vlbi(self, antenna1_handle, antenna2_handle, baseline_handle, ant_deriv, times_gps, clock_state, clock_poly_length, \
            trop_state, trop_poly_length, ao_ant1=None, ao_ant2=None, phase=False, phase_clock_state=[], phase_only=False):
        """ Compute an analytical Jacobian using nav ephemeris and analytical delay model.
        """
        isCOM = True
        f1 = const.c/baseline_handle.wavelength
        freq1_ant = 'G01' # antex frequency for receiver antenna
        if self.iono_free: # iono-free only for GPS, ref frequency for precise range
            f2 = const.c/baseline_handle.wavelength_dual
            if self.iono_freq == 'L2':
                freq2_ant = 'G02'
            elif self.iono_freq == 'L5': 
                freq2_ant = 'G05'
            if antennaPCOData.nFreq <= 4 and self.iono_freq == 'L5':
                freq2_ant = 'G02' 
        else:
            freq2 ='0'
            f2 = 0

        if self.global_linear_clock is True:
            idx_start = 1
        elif self.global_quadratic_clock is True:
            idx_start = 2
        else:
            idx_start = 0

        if antenna1_handle.is_VLBI is False:
            # RX PCO correction
            antennaPCOData1 = antenna1_handle.antenna_PCO
            offset1_L1 = antennaPCOData1.getPhaseCenterOffset(freq1_ant)
            offset1_L1 = np.array([offset1_L1[0],offset1_L1[1],offset1_L1[2]])/1e3 # convert to m
            if self.iono_free is True:
                offset1_dual = antennaPCOData1.getPhaseCenterOffset(freq2_ant)
                offset1_dual = np.array([offset1_dual[0],offset1_dual[1],offset1_dual[2]])/1e3 # convert to m
        if antenna2_handle.is_VLBI is False:
            # RX PCO correction
            antennaPCOData2 = antenna2_handle.antenna_PCO
            offset2_L1 = antennaPCOData2.getPhaseCenterOffset(freq1_ant)
            offset2_L1 = np.array([offset2_L1[0],offset2_L1[1],offset2_L1[2]])/1e3 # convert to m
            if self.iono_free is True:
                offset2_dual = antennaPCOData2.getPhaseCenterOffset(freq2_ant)
                offset2_dual = np.array([offset2_dual[0],offset2_dual[1],offset2_dual[2]])/1e3 # convert to m

        if phase is True: # and antenna_handle.is_VLBI is False: # need to account for carrier phase windup
            wavelength_1 = baseline_handle.wavelength
            if self.iono_free:
                wavelength_2 = baseline_handle.wavelength_dual
        
        if antenna1_handle.is_VLBI is True:
            a_vec1 = antenna1_handle.calc_VLBI_mount_vec()
        if antenna2_handle.is_VLBI is True:
            a_vec2 = antenna2_handle.calc_VLBI_mount_vec()

        times_common = date_to_common(times_gps, 'GPS')
        len_ao = 0
        if antenna1_handle.estimate_ao is True and ant_deriv == 1:
           len_ao = 1
        if antenna2_handle.estimate_ao is True and ant_deriv == 2:
           len_ao = 1

        len_grav = 0
        if antenna1_handle.estimate_grav_def is True and ant_deriv == 1:
            len_grav = 2
        if antenna2_handle.estimate_grav_def is True and ant_deriv == 2:
            len_grav = 2

        if ant_deriv == 1 or ant_deriv ==2:
            len_rxpos = 3
        #elif ant_deriv == 3:
        #    len_rxpos = 6
        else:
            raise ValueError('ant_deriv must be 1, 2, or 3')

        if ant_deriv == 1:
            epoch0 = antenna1_handle.times_gps[0]
        else:
            epoch0 = antenna2_handle.times_gps[0]
        if phase_only is False:
            analytical_jacobian = np.zeros((len(times_gps), len_rxpos+len(clock_state)+len(trop_state)+len(phase_clock_state)+len_ao+len_grav))
        else:
            analytical_jacobian = np.zeros((len(times_gps), len_rxpos+len(trop_state)+len(phase_clock_state)+len_ao+len_grav))

        if phase is True:
            if ant_deriv == 1:
                phase_clock_start = antenna1_handle.phase_clock_start
            else:
                phase_clock_start = antenna2_handle.phase_clock_start

        for idx, common_time in enumerate(times_common):
            eph_time = EphTime(common_time)
            eph_time.setTimeSystem(TimeSystem.UTC)
            sat_id = self.source_time_dict[times_gps[idx]]
            sat_antenna = self.antenna_map[str(sat_id)]  
            RSID = RinexSatID(str(sat_id))
            rxpos1 = antenna1_handle.pos_series[idx,:]
            rxpos2 = antenna2_handle.pos_series[idx,:]
          
            system = RSID.systemString()
            if system == 'GPS': # ref frequencies for precise range
                freq1 = 'G01' # GPS L1
                if self.iono_free:
                    if self.iono_freq == 'L2':
                        freq2 = 'G02'
                    elif self.iono_freq == 'L5':
                        freq2 = 'G02' # only G02 available, no G05 for satellites
            elif system == 'Galileo':
                freq1 = 'E01'
                if self.iono_free:
                    if self.iono_freq == 'L2':
                        raise ValueError('No L2 frequency for BeiDou -- should not be here!')
                    elif self.iono_freq == 'L5':
                        freq2 = 'E05'
            elif system == 'BeiDou':
                freq1 = 'C01' 
                if self.iono_free:
                    if self.iono_freq == 'L2':
                        raise ValueError('No L2 frequency for BeiDou -- should not be here!')
                    elif self.iono_freq == 'L5':
                        freq2 = 'C05'
            
            # find satellite position, pointing vector at receive time
            sat_xvt = self.nav_store.get_xvt(RSID, common_time)
            rxpos1_gnsstk = Position(rxpos1[0], rxpos1[1], rxpos1[2])
            rxpos2_gnsstk = Position(rxpos2[0], rxpos2[1], rxpos2[2])
            sat_pos_gnsstk = Position(sat_xvt.x[0], sat_xvt.x[1], sat_xvt.x[2])
            sat_pos = np.array([sat_xvt.x[0],sat_xvt.x[1],sat_xvt.x[2]])
            sat_vel = np.array([sat_xvt.v[0],sat_xvt.v[1],sat_xvt.v[2]])

            rx2sat1 = sat_pos - rxpos1
            rx2sat1 /= np.linalg.norm(rx2sat1) # normalize vector
            rx2sat2 = sat_pos - rxpos2
            rx2sat2 /= np.linalg.norm(rx2sat2) # normalize vector
            
            elevation1 = rxpos1_gnsstk.elevationGeodetic(sat_pos_gnsstk)
            if antenna1_handle.is_VLBI is True: 
                # compute the geometric model for the VLBI antenna
                point_vec = rx2sat1-a_vec1*np.dot(rx2sat1,a_vec1)

                if antenna1_handle.estimate_ao is True:
                    fb_vec = ao_ant1*point_vec/np.linalg.norm(point_vec)
                    ao_jac1 = -np.sqrt(1-np.dot(a_vec1,rx2sat1)**2)
                else:
                    fb_vec = antenna1_handle.axis_offset*point_vec/np.linalg.norm(point_vec)

                # shift rxpos by offset
                rxpos1_L1 = rxpos1 + fb_vec
                if self.iono_free:
                   rxpos1_dual = rxpos1 + fb_vec
            else:
                # compute antenna PCV
                ROT = antenna1_handle.R_mat # rotation from NEU to XYZ (ECEF) 
                azimuth1 = rxpos1_gnsstk.azimuthGeodetic(sat_pos_gnsstk)
                PCV_L1 = antennaPCOData1.getPhaseCenterVariation(freq1_ant, azimuth1, elevation1)
                rxpos1_L1 = rxpos1 + ROT@(offset1_L1) - PCV_L1*1e-3*rx2sat1
                if self.iono_free:
                    PCV_dual = antennaPCOData1.getPhaseCenterVariation(freq2_ant, azimuth1, elevation1)
                    rxpos1_dual = rxpos1 + ROT@(offset1_dual) - PCV_dual*1e-3*rx2sat1

            elevation2 = rxpos2_gnsstk.elevationGeodetic(sat_pos_gnsstk)
            if antenna2_handle.is_VLBI is True: 
                # compute the geometric model for the VLBI antenna
                point_vec = rx2sat2-a_vec2*np.dot(rx2sat2,a_vec2)

                if antenna2_handle.estimate_ao is True:
                    fb_vec = ao_ant2*point_vec/np.linalg.norm(point_vec)
                    ao_jac2 = -np.sqrt(1-np.dot(a_vec2,rx2sat2)**2)
                else:
                    fb_vec = antenna2_handle.axis_offset*point_vec/np.linalg.norm(point_vec)

                # shift rxpos by offset
                rxpos2_L1 = rxpos2 + fb_vec
                if self.iono_free:
                   rxpos2_dual = rxpos2 + fb_vec
            else:
                # compute antenna PCV
                ROT = antenna1_handle.R_mat # rotation from NEU to XYZ (ECEF) 
                azimuth2 = rxpos2_gnsstk.azimuthGeodetic(sat_pos_gnsstk)
                PCV_L1 = antennaPCOData2.getPhaseCenterVariation(freq1_ant, azimuth2, elevation2)
                rxpos2_L1 = rxpos2 + ROT@(offset2_L1) - PCV_L1*1e-3*rx2sat2
                if self.iono_free:
                    PCV_dual = antennaPCOData1.getPhaseCenterVariation(freq2_ant, azimuth2, elevation2)
                    rxpos2_dual = rxpos2 + ROT@(offset2_dual) - PCV_dual*1e-3*rx2sat2

            sat_pos_L1 = self.sat_adj_PC(freq1, sat_antenna, times_gps[idx], eph_time, sat_xvt.x, 0, noPCV=True)
            if self.iono_free:
                sat_pos_dual = self.sat_adj_PC(freq2, sat_antenna, times_gps[idx], eph_time, sat_xvt.x, 0, noPCV=True)
            
            if ant_deriv == 1:
                dtau_dx1 = self.get_delay_model(RSID, common_time, times_gps[idx], \
                        sat_antenna, rxpos1_L1, rxpos2_L1, freq1, sat_pos_L1, sat_vel, derivs=True, ant_deriv=1)
                if self.iono_free:
                    dtau_dx1_dual = self.get_delay_model(RSID, common_time, times_gps[idx], \
                            sat_antenna, rxpos1_dual, rxpos2_dual, freq2, sat_pos_dual, sat_vel, derivs=True, ant_deriv=1)
                    for ndx in range(3):
                        dtau_dx1[ndx] = float(IonosphereFreeRange([f1, f2], [dtau_dx1[ndx], dtau_dx1_dual[ndx]]))
                analytical_jacobian[idx, :3] = dtau_dx1

            elif ant_deriv == 2:
                dtau_dx2 = self.get_delay_model(RSID, common_time, times_gps[idx], \
                        sat_antenna, rxpos1_L1, rxpos2_L1, freq1, sat_pos_L1, sat_vel, derivs=True, ant_deriv=2)
                if self.iono_free:
                    dtau_dx2_dual = self.get_delay_model(RSID, common_time, times_gps[idx], \
                            sat_antenna, rxpos1_dual, rxpos2_dual, freq2, sat_pos_dual, sat_vel, derivs=True, ant_deriv=2)
                    for ndx in range(3):
                        dtau_dx2[ndx] = float(IonosphereFreeRange([f1, f2], [dtau_dx2[ndx], dtau_dx2_dual[ndx]]))
                analytical_jacobian[idx, :3] = dtau_dx2
            #else:
            #    dtau_dx1, dtau_dx2 = self.get_delay_model(RSID, common_time, times_gps[idx], \
            #            sat_antenna, rxpos1_L1, rxpos2_L1, freq1, sat_pos_L1, sat_vel, derivs=True, ant_deriv=3)
            #    if self.iono_free:
            #        dtau_dx1_dual, dtau_dx2_dual = self.get_delay_model(RSID, common_time, times_gps[idx], \
            #                sat_antenna, rxpos1_dual, rxpos2_dual, freq2, sat_pos_dual, sat_vel, derivs=True, ant_deriv=3)
            #        for ndx in range(3):
            #            dtau_dx1[ndx] = float(IonosphereFreeRange([f1, f2], [dtau_dx1[ndx], dtau_dx1_dual[ndx]]))
            #            dtau_dx2[ndx] = float(IonosphereFreeRange([f1, f2], [dtau_dx2[ndx], dtau_dx2_dual[ndx]]))
            #    analytical_jacobian[idx, :3] = dtau_dx1
            #    analytical_jacobian[idx, 3:6] = dtau_dx2
          
            # get clock derivatives
            if phase_only is False and len(clock_state)>0:
                clock_jac = np.zeros(len(clock_state))
                if self.global_linear_clock is True or self.global_quadratic_clock is True:
                    if ant_deriv==1:
                        clock_jac[:idx_start] = global_poly_jac_at_epoch(clock_state[:idx_start], \
                                times_gps[idx], antenna1_handle.times_gps[0], antenna1_handle.times_gps[-1])
                    elif ant_deriv==2:
                        clock_jac[:idx_start] = global_poly_jac_at_epoch(clock_state[:idx_start], \
                                times_gps[idx], antenna2_handle.times_gps[0], antenna2_handle.times_gps[-1])
                    analytical_jacobian[idx, 3:3+idx_start] = clock_jac[:idx_start]

                if self.stochastic_clock is False and clock_poly_length>0:
                    clock_jac[idx_start:] = poly_jac_at_epoch(clock_state[idx_start:], clock_poly_length, times_gps[idx], epoch0)
                    analytical_jacobian[idx, 3+idx_start:3+len(clock_state)] = clock_jac[idx_start:]           
                elif self.stochastic_clock is True:
                    if ant_deriv == 1:
                        clock_idx = np.argwhere(antenna1_handle.clock_times == times_gps[idx])
                    else:
                        clock_idx = np.argwhere(antenna2_handle.clock_times == times_gps[idx])
                    analytical_jacobian[idx, 3+idx_start+clock_idx] = 1
                else:
                    # only global clock model -- bulk clock offset stored after linear/quadratic coeffs
                    analytical_jacobian[idx, 3+idx_start] = 1

            if antenna1_handle.estimate_ao is True and ant_deriv==1:
                analytical_jacobian[idx, 3+len(clock_state)] = ao_jac1
            if antenna2_handle.estimate_ao is True and ant_deriv==2:
                analytical_jacobian[idx, 3+len(clock_state)] = ao_jac2

            if antenna1_handle.estimate_grav_def is True and ant_deriv==1:
                #analytical_jacobian[idx, 3+len(clock_state)+len_ao] = np.sin(np.deg2rad(elevation1))
                analytical_jacobian[idx, 3+len(clock_state)+len_ao] = np.deg2rad(elevation1)
                #analytical_jacobian[idx, 3+len(clock_state)+len_ao+1] = np.cos(np.deg2rad(elevation1))
                analytical_jacobian[idx, 3+len(clock_state)+len_ao+1] = np.deg2rad(elevation1)**2
            if antenna2_handle.estimate_grav_def is True and ant_deriv==2:
                #analytical_jacobian[idx, 3+len(clock_state)+len_ao] = np.sin(np.deg2rad(elevation2))
                analytical_jacobian[idx, 3+len(clock_state)+len_ao] = np.deg2rad(elevation2)
                #analytical_jacobian[idx, 3+len(clock_state)+len_ao+1] = np.cos(np.deg2rad(elevation2))
                analytical_jacobian[idx, 3+len(clock_state)+len_ao+1] = np.deg2rad(elevation2)**2

            if len(trop_state)>0:
                if ant_deriv == 1:
                    trop_delay1 = antenna1_handle.tropModel.correction(rxpos1_gnsstk, sat_pos_gnsstk, common_time)
                    map_fcn = antenna1_handle.tropModel.wet_mapping_function(elevation1)
                elif ant_deriv == 2:
                    trop_delay2 = antenna2_handle.tropModel.correction(rxpos2_gnsstk, sat_pos_gnsstk, common_time)
                    map_fcn = antenna2_handle.tropModel.wet_mapping_function(elevation2)
                if self.stochastic_trop is False:
                    trop_jac = poly_jac_at_epoch(trop_state, trop_poly_length, times_gps[idx], epoch0)
                    analytical_jacobian[idx, 3+len_ao+len(clock_state):3+len_ao+len(clock_state)+len(trop_state)] = trop_jac*map_fcn 
                else:
                    if ant_deriv == 1:
                        trop_idx = np.argwhere(antenna1_handle.trop_times == times_gps[idx])
                    else:
                        trop_idx = np.argwhere(antenna2_handle.trop_times == times_gps[idx])
                    analytical_jacobian[idx, 3+len_ao+len(clock_state)+trop_idx] = map_fcn

            if phase is True and len(phase_clock_state)>0:
                phase_clock_jac = np.zeros(len(phase_clock_state))
                if self.global_linear_clock is True or self.global_quadratic_clock is True:
                    if ant_deriv==1:
                        phase_clock_jac[:idx_start] = global_poly_jac_at_epoch(phase_clock_state[:idx_start], \
                                times_gps[idx], antenna1_handle.times_gps[0], antenna1_handle.times_gps[-1])
                    elif ant_deriv==2:
                        phase_clock_jac[:idx_start] = global_poly_jac_at_epoch(phase_clock_state[:idx_start], \
                                times_gps[idx], antenna2_handle.times_gps[0], antenna2_handle.times_gps[-1])
                    analytical_jacobian[idx, 3+len(clock_state)+len_ao+len_grav+len(trop_state):\
                            3+len(clock_state)+len_ao+len_grav+len(trop_state)+idx_start] = phase_clock_jac[:idx_start]
                if self.stochastic_clock is False and clock_poly_length>0: 
                    phase_clock_jac[idx_start:] = poly_jac_at_epoch(phase_clock_state[idx_start:], clock_poly_length, \
                            times_gps[idx], phase_clock_start)
                    analytical_jacobian[idx, 3+len(clock_state)+len_ao+len_grav+len(trop_state)+idx_start:\
                            3+len(clock_state)+len_ao+len_grav+len(trop_state)+len(phase_clock_state)] = phase_clock_jac[idx_start:]
                elif self.stochastic_clock is True:
                    if ant_deriv == 1:
                        phase_clock_idx = np.argwhere(antenna1_handle.phase_clock_times == times_gps[idx])
                    else:
                        phase_clock_idx = np.argwhere(antenna2_handle.phase_clock_times == times_gps[idx])
                    analytical_jacobian[idx, 3+len(clock_state)+len_ao+len_grav+len(trop_state)+idx_start+phase_clock_idx] = 1
                else:
                    # only global clock model -- bulk clock offset stored after linear/quadratic coeffs
                    analytical_jacobian[idx, 3+len(clock_state)+len_ao+len_grav+len(trop_state)+idx_start ] =1

        return analytical_jacobian

    def compute_analytical_jac_farfield(self, antenna1_handle, antenna2_handle, baseline_handle, ant_deriv, times_gps, clock_state, clock_poly_length, \
            trop_state, trop_poly_length, ao_ant1=None, ao_ant2=None, phase=False, phase_clock_state=[], phase_only=False):
        """ Compute an analytical Jacobian using nav ephemeris and analytical delay model.
        """
        len_ao = 0
        if antenna1_handle.estimate_ao is True and ant_deriv==1:
           len_ao = 1
        if antenna2_handle.estimate_ao is True and ant_deriv==2:
           len_ao = 1

        len_grav=0
        if antenna1_handle.estimate_grav_def is True and ant_deriv==1:
            len_grav = 2
        if antenna2_handle.estimate_grav_def is True and ant_deriv==2:
            len_grav = 2

        if ant_deriv == 1 or ant_deriv ==2:
            len_rxpos = 3
        else:
            raise ValueError('ant_deriv must be 1 or 2')

        if ant_deriv == 1:
            epoch0 = antenna1_handle.times_gps[0]
        else:
            epoch0 = antenna2_handle.times_gps[0]
        if phase_only is False:
            analytical_jacobian = np.zeros((len(times_gps), len_rxpos+len(clock_state)+len(trop_state)+len(phase_clock_state)+len_ao+len_grav))
        else:
            analytical_jacobian = np.zeros((len(times_gps), len_rxpos+len(trop_state)+len(phase_clock_state)+len_ao+len_grav))

        if phase is True:
            if ant_deriv == 1:
                phase_clock_start = antenna1_handle.phase_clock_start
            else:
                phase_clock_start = antenna2_handle.phase_clock_start
        f1 = const.c/baseline_handle.wavelength
        freq1_ant = 'G01' # antex frequency for receiver antenna
        if self.iono_free: # iono-free only for GPS, ref frequency for precise range
            f1 = const.c/baseline_handle.wavelength_dual
            if phase is True: 
                phase_model_dual_arr = []
            if self.iono_freq == 'L2':
                freq2_ant = 'G02'
            elif self.iono_freq == 'L5': 
                freq2_ant = 'G05'
            if antennaPCOData.nFreq <= 4 and self.iono_freq == 'L5':
                freq2_ant = 'G02' 
        else:
            freq2 ='0'
            f2 = 0

        if self.global_linear_clock is True:
            idx_start = 1
        elif self.global_quadratic_clock is True:
            idx_start = 2
        else:
            idx_start = 0

        if antenna1_handle.is_VLBI is False:
            # RX PCO correction
            antennaPCOData1 = antenna1_handle.antenna_PCO
            offset1_L1 = antennaPCOData1.getPhaseCenterOffset(freq1_ant)
            offset1_L1 = np.array([offset1_L1[0],offset1_L1[1],offset1_L1[2]])/1e3 # convert to m
            if self.iono_free is True:
                offset1_dual = antennaPCOData.getPhaseCenterOffset(freq2_ant)
                offset1_dual = np.array([offset_dual[0],offset_dual[1],offset_dual[2]])/1e3 # convert to m

        if antenna2_handle.is_VLBI is False:
            # RX PCO correction
            antennaPCOData2 = antenna2_handle.antenna_PCO
            offset2_L1 = antennaPCOData2.getPhaseCenterOffset(freq1_ant)
            offset2_L1 = np.array([offset2_L1[0],offset2_L1[1],offset2_L1[2]])/1e3 # convert to m
            if self.iono_free is True:
                offset2_dual = antennaPCOData.getPhaseCenterOffset(freq2_ant)
                offset2_dual = np.array([offset2_dual[0],offset2_dual[1],offset2_dual[2]])/1e3 # convert to m

        if phase is True: # need to account for carrier phase windup
            cpw_ant1 = {} # carrrier phase windup, dictionary by source
            cpw_ant2 = {}
            for source in self.source_array:
                cpw_ant1[source] = 0.0 # initialize cpw to 0
                cpw_ant2[source] = 0.0 

            wavelength_1 = baseline_handle.wavelength
            if self.iono_free:
                wavelength_2 = baseline_handle.wavelength_dual
      
            if antenna1_handle.is_VLBI is False:
                east_triple_ant1 = Position(antenna1_handle.Rotate_obj.get_value(1,0),\
                                     antenna1_handle.Rotate_obj.get_value(1,1),
                                     antenna1_handle.Rotate_obj.get_value(1,2))
  
                north_triple_ant1 = Position(antenna1_handle.Rotate_obj.get_value(0,0),\
                                     antenna1_handle.Rotate_obj.get_value(0,1),
                                     antenna1_handle.Rotate_obj.get_value(0,2))

            if antenna2_handle.is_VLBI is False:
                east_triple_ant2 = Position(antenna2_handle.Rotate_obj.get_value(1,0),\
                                     antenna2_handle.Rotate_obj.get_value(1,1),
                                     antenna2_handle.Rotate_obj.get_value(1,2))
  
                north_triple_ant2 = Position(antenna2_handle.Rotate_obj.get_value(0,0),\
                                     antenna2_handle.Rotate_obj.get_value(0,1),
                                     antenna2_handle.Rotate_obj.get_value(0,2))       

        if antenna1_handle.is_VLBI is True:
            a_vec1 = antenna1_handle.calc_VLBI_mount_vec()
        if antenna2_handle.is_VLBI is True:
            a_vec2 = antenna2_handle.calc_VLBI_mount_vec()

        times_common = date_to_common(times_gps, 'GPS')
        for idx, time in enumerate(times_gps):
            eph_time = EphTime(times_common[idx])
            eph_time.setTimeSystem(TimeSystem.UTC)
            source = self.source_time_dict[time]
            rxpos1 = antenna1_handle.pos_series[idx,:]
            rxpos2 = antenna2_handle.pos_series[idx,:]

            R_T2I = self.get_ECEF2ECI(eph_time)

            ra, dec = antenna1_handle.point_ra_dec_dict[time]
            s_ecef = self.compute_ptvec(ra, dec, eph_time)
            s_vec = self.compute_ptvec(ra, dec, eph_time, frame='ECI')

            R1_NEU = antenna1_handle.R_mat.T # matrix from XYZ (ECEF) to NEU 
            s1_NEU = R1_NEU@s_ecef
            R2_NEU = antenna2_handle.R_mat.T # matrix from XYZ (ECEF) to NEU 
            s2_NEU = R2_NEU@s_ecef
            elevation1 = np.degrees(np.arcsin(s1_NEU[2]))
            elevation2 = np.degrees(np.arcsin(s2_NEU[2]))

            # compute apparent source vector after refraction, aberration
            eph_time.convertSystemTo(TimeSystem.TDB)
            mjd_tdb = eph_time.dMJD() 
            set_km = True
            earth_posvel = self.sol_sys.relativeInertialPositionVelocityPyWrapper(mjd_tdb, self.sol_sys.idEarth, self.sol_sys.idSolarSystemBarycenter, set_km)
            earth_pos = np.array(earth_posvel[:3])*1e3
            earth_vel = np.array(earth_posvel[3:])*1e3
            eph_time.convertSystemTo(TimeSystem.UTC) # convert time system back for other functions
            # refractivities -- Sovers et al (1998)
            refrac_1 = 3.13e-4/np.tan(np.radians(elevation1))
            refrac_2 = 3.13e-4/np.tan(np.radians(elevation2))
            rxpos1_eci = R_T2I@rxpos1
            rxpos1_eci /= np.linalg.norm(rxpos1_eci)
            rxpos2_eci = R_T2I@rxpos1
            rxpos2_eci /= np.linalg.norm(rxpos2_eci)
            s_app_1 = earth_vel/const.c - np.dot(s_vec,earth_vel)*s_vec/const.c + np.cos(refrac_1)*s_vec + np.sin(refrac_1)*(rxpos1_eci[2]-rxpos1_eci[2]*s_vec[2]*s_vec)
            s_app_2 = earth_vel/const.c - np.dot(s_vec,earth_vel)*s_vec/const.c + np.cos(refrac_2)*s_vec + np.sin(refrac_2)*(rxpos2_eci[2]-rxpos2_eci[2]*s_vec[2]*s_vec)
            s_ecef1 = R_T2I.T@s_app_1 # apparent source vector in ECEF
            s_ecef2 = R_T2I.T@s_app_2

            rxpos1_gnsstk = Position(rxpos1[0], rxpos1[1], rxpos1[2])
            rxpos2_gnsstk = Position(rxpos2[0], rxpos2[1], rxpos2[2])

            if antenna1_handle.is_VLBI is True: 
                # compute the geometric model for the VLBI antenna
                point_vec = s_ecef1-a_vec1*np.dot(s_ecef1, a_vec1)
                
                if antenna1_handle.estimate_ao is True:
                    fb_vec = ao_ant1*point_vec/np.linalg.norm(point_vec) 
                else:
                    fb_vec = antenna1_handle.axis_offset*point_vec/np.linalg.norm(point_vec) 

                # shift rxpos by offset
                rxpos1_L1 = rxpos1 + fb_vec
                if self.iono_free:
                   rxpos1_dual = rxpos1 + fb_vec
            else:
                # compute antenna PCV
                azimuth1 = np.degrees(np.arctan2(s1_NEU[1], s1_NEU[0]))
                if azimuth1 < 0: azimuth1 += 360
                ROT = antenna1_handle.R_mat # rotation from NEU to XYZ (ECEF)
                PCV_L1 = antennaPCOData1.getPhaseCenterVariation(freq1_ant, azimuth1, elevation1)
                rxpos1_L1 = rxpos1 + ROT@offset1_L1 - PCV_L1*1e-3*s_ecef1
                if self.iono_free:
                    PCV_dual = antennaPCOData1.getPhaseCenterVariation(freq2_ant, azimuth1, elevation1)
                    rxpos1_dual = rxpos1 + ROT@(offset1_dual) - PCV_dual*1e-3*s_ecef1

            if antenna2_handle.is_VLBI is True: 
                # compute the geometric model for the VLBI antenna
                point_vec = s_ecef2-a_vec2*np.dot(s_ecef2, a_vec2)
                
                if antenna2_handle.estimate_ao is True:
                    fb_vec = ao_ant2*point_vec/np.linalg.norm(point_vec) 
                else:
                    fb_vec = antenna2_handle.axis_offset*point_vec/np.linalg.norm(point_vec) 

                # shift rxpos by offset
                rxpos2_L1 = rxpos2 + fb_vec
                if self.iono_free:
                   rxpos2_dual = rxpos2 + fb_vec
            else:
                # compute antenna PCV
                azimuth2 = np.degrees(np.arctan2(s1_NEU[1], s1_NEU[0]))
                if azimuth2 < 0: azimuth2 += 360
                ROT = antenna2_handle.R_mat # rotation from NEU to XYZ (ECEF)
                PCV_L1 = antennaPCOData2.getPhaseCenterVariation(freq1_ant, azimuth2, elevation2)
                rxpos2_L1 = rxpos2 + ROT@offset2_L1 - PCV_L1*1e-3*s_ecef2
                if self.iono_free:
                    PCV_dual = antennaPCOData2.getPhaseCenterVariation(freq2_ant, azimuth2, elevation2)
                    rxpos2_dual = rxpos2 + ROT@(offset2_dual) - PCV_dual*1e-3*s_ecef2

            delay = self.get_delay_farfield(s_vec, eph_time, rxpos1_L1, rxpos2_L1)

            if self.iono_free:
                delay_dual = self.get_delay_farfield(s_vec, eph_time, rxpos1_dual, rxpos2_dual)

            if ant_deriv == 1:
                dtau_dx1 = self.get_delay_farfield(s_vec, eph_time, rxpos1_L1, rxpos2_L1, derivs=True, ant_deriv=1)
                if self.iono_free:
                    dtau_dx1_dual = self.get_delay_farfield(s_vec, eph_time, rxpos1_dual, rxpos2_dual, derivs=True, ant_deriv=1)
                    for ndx in range(3):
                        dtau_dx1[ndx] = float(IonosphereFreeRange([f1, f2], [dtau_dx1[ndx], dtau_dx1_dual[ndx]]))
                analytical_jacobian[idx, :3] = dtau_dx1

            elif ant_deriv == 2:
                dtau_dx2 = self.get_delay_farfield(s_vec, eph_time, rxpos1_L1, rxpos2_L1, derivs=True, ant_deriv=2)
                if self.iono_free:
                    dtau_dx2_dual = self.get_delay_farfield(s_vec, eph_time, rxpos1_dual, rxpos2_dual, derivs=True, ant_deriv=2)
                    for ndx in range(3):
                        dtau_dx2[ndx] = float(IonosphereFreeRange([f1, f2], [dtau_dx2[ndx], dtau_dx2_dual[ndx]]))
                analytical_jacobian[idx, :3] = dtau_dx2
          
            # get clock derivatives
            if phase_only is False and len(clock_state)>0:
                clock_jac = np.zeros(len(clock_state))
                if self.global_linear_clock is True or self.global_quadratic_clock is True:
                    if ant_deriv==1:
                        clock_jac[:idx_start] = global_poly_jac_at_epoch(clock_state[:idx_start], \
                                times_gps[idx], antenna1_handle.times_gps[0], antenna1_handle.times_gps[-1])
                    elif ant_deriv==2:
                        clock_jac[:idx_start] = global_poly_jac_at_epoch(clock_state[:idx_start], \
                                times_gps[idx], antenna2_handle.times_gps[0], antenna2_handle.times_gps[-1])
                    analytical_jacobian[idx, 3:3+idx_start] = clock_jac[:idx_start] 
                if self.stochastic_clock is False and clock_poly_length>0:
                    clock_jac[idx_start:] = poly_jac_at_epoch(clock_state[idx_start:], clock_poly_length, times_gps[idx], epoch0)
                    analytical_jacobian[idx, 3+idx_start:3+len(clock_state)] = clock_jac[idx_start:]
                elif self.stochastic_clock is True:
                    if ant_deriv == 1:
                        clock_idx = np.argwhere(antenna1_handle.clock_times == times_gps[idx])
                    else:
                        clock_idx = np.argwhere(antenna2_handle.clock_times == times_gps[idx])
                    analytical_jacobian[idx, 3+idx_start+clock_idx] = 1
                else:
                    # only global clock model -- bulk clock offset stored after linear/quadratic coeffs
                    analytical_jacobian[idx, 3+idx_start] = 1

            if antenna1_handle.estimate_ao is True and ant_deriv==1:
                analytical_jacobian[idx, 3+len(clock_state)] = ao_jac1
            if antenna2_handle.estimate_ao is True and ant_deriv==2:
                analytical_jacobian[idx, 3+len(clock_state)] = ao_jac2

            if antenna1_handle.estimate_grav_def is True and ant_deriv==1:
                #analytical_jacobian[idx, 3+len(clock_state)+len_ao] = np.sin(np.deg2rad(elevation1))
                analytical_jacobian[idx, 3+len(clock_state)+len_ao] = np.deg2rad(elevation1)
                #analytical_jacobian[idx, 3+len(clock_state)+len_ao+1] = np.cos(np.deg2rad(elevation1))
                analytical_jacobian[idx, 3+len(clock_state)+len_ao+1] = np.deg2rad(elevation1)**2
            if antenna2_handle.estimate_grav_def is True and ant_deriv==2:
                #analytical_jacobian[idx, 3+len(clock_state)+len_ao] = np.sin(np.deg2rad(elevation2))
                analytical_jacobian[idx, 3+len(clock_state)+len_ao] = np.deg2rad(elevation2)
                #analytical_jacobian[idx, 3+len(clock_state)+len_ao+1] = np.cos(np.deg2rad(elevation2))
                analytical_jacobian[idx, 3+len(clock_state)+len_ao+1] = np.deg2rad(elevation2)**2

            if len(trop_state)>0:
                if self.src_type == 'VLBI' and getattr(antenna1_handle.tropModel, 'vmf3_type', None) is not None:
                    vmf3_type = True
                    # VMF3 trop needs azimuth and time
                    rxpos1_gnsstk = Position(rxpos1[0], rxpos1[1], rxpos1[2])
                    rxpos2_gnsstk = Position(rxpos2[0], rxpos2[1], rxpos2[2])
                    azimuth1 = np.degrees(np.arctan2(s1_NEU[1], s1_NEU[0]))
                    if azimuth1 < 0: azimuth1 += 360
                    azimuth2 = np.degrees(np.arctan2(s1_NEU[1], s1_NEU[0]))
                    if azimuth2 < 0: azimuth2 += 360
                    common_time = times_common[idx]
                else:
                    vmf3_type = False

                if ant_deriv == 1:
                    if self.src_type == 'GNSS':
                        trop_delay1 = antenna1_handle.tropModel.correction(rxpos1_gnsstk, sat_pos_gnsstk, common_time)
                    elif vmf3_type:
                        trop_delay1 = antenna1_handle.tropModel.correction_azel(rxpos1_gnsstk, azimuth1, elevation1, common_time)
                    else:
                        trop_delay1 = antenna1_handle.tropModel.correction(elevation1)
                    map_fcn = antenna1_handle.tropModel.wet_mapping_function(elevation1)
                elif ant_deriv == 2:
                    if self.src_type == 'GNSS':
                        trop_delay2 = antenna2_handle.tropModel.correction(rxpos2_gnsstk, sat_pos_gnsstk, common_time)
                    elif vmf3_type:
                        trop_delay2 = antenna2_handle.tropModel.correction_azel(rxpos2_gnsstk, azimuth2, elevation2, common_time)
                    else:
                        trop_delay2 = antenna2_handle.tropModel.correction(elevation2)
                    map_fcn = antenna2_handle.tropModel.wet_mapping_function(elevation2)
                if self.stochastic_trop is False:
                    trop_jac = poly_jac_at_epoch(trop_state, trop_poly_length, times_gps[idx], epoch0)
                    analytical_jacobian[idx, 3+len_ao+len_grav+len(clock_state):3+len_ao+len_grav+len(clock_state)+len(trop_state)] = trop_jac*map_fcn 
                else:
                    if ant_deriv == 1:
                        trop_idx = np.argwhere(antenna1_handle.trop_times == times_gps[idx])
                    else:
                        trop_idx = np.argwhere(antenna2_handle.trop_times == times_gps[idx])
                    analytical_jacobian[idx, 3+len_ao+len_grav+len(clock_state)+trop_idx] = map_fcn

            if phase is True and len(phase_clock_state)>0:
                phase_clock_jac = np.zeros(len(phase_clock_state))
                if self.global_linear_clock is True or self.global_quadratic_clock is True:
                    if ant_deriv==1:
                        phase_clock_jac[:idx_start] = global_poly_jac_at_epoch(phase_clock_state[:idx_start], \
                                times_gps[idx], antenna1_handle.times_gps[0], antenna1_handle.times_gps[-1])
                    elif ant_deriv==2:
                        phase_clock_jac[:idx_start] = global_poly_jac_at_epoch(phase_clock_state[:idx_start], \
                                times_gps[idx], antenna2_handle.times_gps[0], antenna2_handle.times_gps[-1])
                    analytical_jacobian[idx, 3+len(clock_state)+len_ao+len_grav+len(trop_state):\
                            3+len(clock_state)+len_ao+len_grav+len(trop_state)+idx_start] = phase_clock_jac[:idx_start]
                if self.stochastic_clock is False and clock_poly_length>0: 
                    phase_clock_jac[idx_start:] = poly_jac_at_epoch(phase_clock_state[idx_start:], clock_poly_length, \
                            times_gps[idx], phase_clock_start)
                    analytical_jacobian[idx, 3+len(clock_state)+len_ao+len_grav+len(trop_state)+idx_start:\
                            3+len(clock_state)+len_ao+len_grav+len(trop_state)+len(phase_clock_state)] = phase_clock_jac[idx_start:]
                elif self.stochastic_clock is True:
                    if ant_deriv == 1:
                        phase_clock_idx = np.argwhere(antenna1_handle.phase_clock_times == times_gps[idx])
                    else:
                        phase_clock_idx = np.argwhere(antenna2_handle.phase_clock_times == times_gps[idx])
                    analytical_jacobian[idx, 3+len(clock_state)+len_ao+len_grav+len(trop_state)+idx_start+phase_clock_idx] = 1
                else:
                    # only global clock model -- bulk clock offset stored after linear/quadratic coeffs
                    analytical_jacobian[idx, 3+len(clock_state)+len_ao+len_grav+len(trop_state)+idx_start] = 1

        return analytical_jacobian

    def replace_meas_hack(self, datetime_array, source_array):
        """Replace GNSS measurements with VLBI measurements to test differences"""
        self.vlbi_hack = True
        from plot_observable_diff import read_psolve_file
        psolve_data_file = '/home/jskeens/mdo_rinex/observables_final_pared.txt'
        key_file = '/home/jskeens/gnssvlbi/software/sim_vis/uy001d_prn.key'
        #baseline = 'DBR205   / FD'
        baseline = 'DBR231   / FD'
        data_by_sta, baselines = read_psolve_file(psolve_data_file)
        data = data_by_sta[baseline]

        self.group_delays = np.array(data['Tot_gr_1'])
        self.phase_delays = np.array(data['Tot_phs_1'])
        dt_psolve = np.array(data['SRT'])
        SNR_obs = np.array(data['SNR'])
        FRT = np.array(data['FRT_off_1'])
        ELEV = np.array(data['Elev_1'])
        frq = np.array(data['Ref_frq_1'])
        # take only good data
        use_idxs = SNR_obs > 500 # limit to 500 SNR
        self.group_delays = self.group_delays[use_idxs] 
        self.phase_delays = self.phase_delays[use_idxs] 
        dt_psolve = dt_psolve[use_idxs]
        FRT = FRT[use_idxs]
        ELEV = ELEV[use_idxs]
        frq = frq[use_idxs]
        
        # correct datetime from scan reference time (TAI) to GPS time
        for idx in range(len(dt_psolve)):
            dt_psolve[idx] = dt_psolve[idx] + datetime.timedelta(seconds=TAI2GPS)

        source_array_vlbi = []
        idxs_keep = []
        idxs_orig = []
        source_array_np = np.array(source_array)
        
        # take only data points that where we know what satellite was observed
        for idx, time in enumerate(dt_psolve):
            diff_time = np.array([(time_single_diff-time).total_seconds() for time_single_diff in datetime_array])
            check_idxs = np.abs(diff_time)<30
            if np.any(check_idxs) and np.any(diff_time==0):
                # we have data to compare
                # ensure that all epochs correspond to the same satellite
                sats = source_array_np[check_idxs]
                unique,pos = np.unique(sats,return_inverse=True) #Find all unique elements and their positions
                counts = np.bincount(pos) #Count the number of each unique element
                maxpos = counts.argmax() #Find the positions of the maximum count
                sat = sats[maxpos] # the most common satellite in the set i.e. the satellite we want to consider
                source_array_vlbi.append(sat)
                idxs_keep.append(idx)
                idxs_orig.append(np.argwhere(diff_time==0)[0][0])
        
        self.group_delays = self.group_delays[idxs_keep]*const.c # group delay in meters
        # get correlation between ambiguities by adding nearest integer from group delay -- this will cause linked measurements hopefully
        wavelength = const.c/frq[idxs_keep]
        amb_int = np.rint(self.group_delays/wavelength)
        self.phase_delays = (self.phase_delays[idxs_keep]/(2*np.pi) + amb_int) * wavelength

        return dt_psolve[idxs_keep], source_array_vlbi, idxs_orig, frq[idxs_keep]

def to_mjd(t):
    """
    Convert to Modified Julian Date (MJD).

    Accepted types:
      - float / numpy.floating (assumed already MJD)
      - numpy.datetime64 (any unit, e.g. 'ns', 'us', 'ms', 's', 'm', 'h', 'D')
      - numpy arrays thereof (vectorized)

    Returns:
      - float for scalar input; np.ndarray(float64) for array input.
        NaT -> np.nan.
    """
    # Pass through floats as already-MJD
    if isinstance(t, (float, np.floating)):
        return float(t)

    arr = np.asarray(t)

    if np.issubdtype(arr.dtype, np.datetime64):
        # Normalize to ns since epoch, handle NaT separately
        ns = arr.astype('datetime64[ns]').astype('int64')
        # NaT becomes the minimum int64; map to NaN
        is_nat = ns == np.iinfo(np.int64).min

        # Convert: seconds since Unix epoch -> days -> add MJD at Unix epoch
        mjd = (ns / 1e9) / SEC_PER_DAY + MJD_UNIX
        mjd = mjd.astype(np.float64, copy=False)
        if np.any(is_nat):
            mjd = mjd.copy()
            mjd[is_nat] = np.nan

        return mjd.item() if np.isscalar(t) else mjd

    raise TypeError(f"Unsupported type for to_mjd: {type(t)}")

def _norm_lon_deg(lon_deg: float) -> float:
    # Normalize to [0, 360)
    x = lon_deg % 360.0
    # Avoid 360 exactly
    if x == 360.0:
        x = 0.0
    return x


class VMF3Model:
    """
    Class handling the VMF3 delay/mapping functions and gradients 
    ref: Landskron, D. & Böhm, J. J Geod (2018). https://doi.org/10.1007/s00190-018-1127-1
    """
    def __init__(self, vmf3_type, station=None):
        # Define Legendre functions for bh, bw, ch and cw
        self.vmf3_type = vmf3_type
        self.anm_bh = np.array([[0.00271285863109945,- 1.39197786008938e-06,1.34955672002719e-06,2.71686279717968e-07,1.56659301773925e-06],[9.80476624811974e-06,- 5.83922611260673e-05,- 2.07307023860417e-05,1.14628726961148e-06,4.93610283608719e-06],[- 1.03443106534268e-05,- 2.05536138785961e-06,2.09692641914244e-06,- 1.55491034130965e-08,- 1.89706404675801e-07],[- 3.00353961749658e-05,2.37284447073503e-05,2.02236885378918e-05,1.69276006349609e-06,8.72156681243892e-07],[- 7.99121077044035e-07,- 5.39048313389504e-06,- 4.21234502039861e-06,- 2.70944149806894e-06,- 6.80894455531746e-07],[7.51439609883296e-07,3.8550970886552e-07,4.41508016098164e-08,- 2.07507808307757e-08,4.95354985050743e-08],[2.21790962160087e-05,- 5.56986238775212e-05,- 1.81287885563308e-05,- 4.41076013532589e-06,4.93573223917278e-06],[- 4.47639989737328e-06,- 2.6045289307212e-06,2.56376320011189e-06,4.41600992220479e-07,2.93437730332869e-07],[8.14992682244945e-07,2.03945571424434e-07,1.11832498659806e-08,3.25756664234497e-08,3.01029040414968e-08],[- 7.96927680907488e-08,- 3.66953150925865e-08,- 6.74742632186619e-09,- 1.30315731273651e-08,- 2.00748924306947e-09],[- 2.16138375166934e-05,1.67350317962556e-05,1.93768260076821e-05,1.9959512016185e-06,- 2.42463528222014e-06],[5.34360283708044e-07,- 3.641890220406e-06,- 2.99935375194279e-06,- 2.06880962903922e-06,- 9.40815692626002e-07],[6.80235884441822e-07,1.33023436079845e-07,- 1.80349593705226e-08,2.51276252565192e-08,- 1.43240592002794e-09],[- 7.13790897253802e-08,7.81998506267559e-09,1.13826909570178e-09,- 5.89629600214654e-09,- 4.20760865522804e-09],[- 5.80109372399116e-09,1.13702284491976e-09,7.29046067602764e-10,- 9.10468988754012e-10,- 2.58814364808642e-10],[1.75558618192965e-05,- 2.85579168876063e-05,- 1.47442190284602e-05,- 6.29300414335248e-06,- 5.1220453891346e-07],[- 1.9078855829131e-06,- 1.62144845155361e-06,7.57239241641566e-07,6.93365788711348e-07,6.88855644570695e-07],[2.27050351488552e-07,1.0392579127766e-07,- 3.31105076632079e-09,2.88065761026675e-08,- 8.00256848229136e-09],[- 2.77028851807614e-08,- 5.9625113220693e-09,2.95987495527251e-10,- 5.87644249625625e-09,- 3.28803981542337e-09],[- 1.89918479865558e-08,3.54083436578857e-09,8.10617835854935e-10,4.99207055948336e-10,- 1.52691648387663e-10],[1.04022499586096e-09,- 2.36437143845013e-10,- 2.25110813484842e-10,- 7.39850069252329e-11,7.95929405440911e-11],[- 3.1157942126763e-05,- 3.43576336877494e-06,5.81663608263384e-06,8.31534700351802e-07,4.02619520312154e-06],[6.00037066879001e-07,- 1.12538760056168e-07,- 3.8674533211559e-07,- 3.88218746020826e-07,- 6.83764967176388e-07],[- 9.79583981249316e-08,9.14964449851003e-08,4.77779838549237e-09,2.44283811750703e-09,- 6.26361079345158e-09],[- 2.37742207548109e-08,- 5.53336301671633e-09,- 3.73625445257115e-09,- 1.92304189572886e-09,- 7.18681390197449e-09],[- 6.58203463929583e-09,9.28456148541896e-10,2.47218904311077e-10,1.10664919110218e-10,- 4.20390976974043e-11],[9.45857603373426e-10,- 3.29683402990254e-11,- 8.15440375865127e-11,- 1.21615589356628e-12,- 9.70713008848085e-12],[1.61377382316176e-10,6.84326027598147e-12,- 4.66898885683671e-12,2.31211355085535e-12,2.39195112937346e-12],[2.99634365075821e-07,8.14391615472128e-06,6.70458490942443e-06,- 9.92542646762e-07,- 3.0407806499275e-06],[- 6.52697933801393e-07,2.87255329776428e-07,- 1.78227609772085e-08,2.65525429849935e-07,8.60650570551813e-08],[- 1.6272716401171e-07,1.09102479325892e-07,4.97827431850001e-09,7.86649963082937e-11,- 6.67193813407656e-09],[- 2.9637000098776e-09,1.20008401576557e-09,1.75885448022883e-09,- 1.74756709684384e-09,3.21963061454248e-09],[- 9.9110169777856e-10,7.54541713140752e-10,- 2.95880967800875e-10,1.81009160501278e-10,8.31547411640954e-11],[1.21268051949609e-10,- 5.93572774509587e-11,- 5.03295034994351e-11,3.05383430975252e-11,3.56280438509939e-11],[6.92012970333794e-11,- 9.02885345797597e-12,- 3.4415183274488e-12,2.03164894681921e-12,- 5.44852265137606e-12],[5.567312636728e-12,3.57272150106101e-12,2.25885622368678e-12,- 2.44508240047675e-13,- 6.83314378535235e-13],[3.96883487797254e-06,- 4.57100506169608e-06,- 3.30208117813256e-06,3.32599719134845e-06,4.26539325549339e-06],[1.10123151770973e-06,4.58046760144882e-07,1.86831972581926e-07,- 1.60092770735081e-07,- 5.58956114867062e-07],[- 3.40344900506653e-08,2.87649741373047e-08,- 1.83929753066251e-08,- 9.74179203885847e-09,- 2.42064137485043e-09],[- 6.49731596932566e-09,- 3.07048108404447e-09,- 2.84380614669848e-09,1.55123146524283e-09,4.53694984588346e-10],[5.45175793803325e-10,- 3.73287624700125e-10,- 1.16293122618336e-10,7.2584561860269e-11,- 4.34112440021627e-11],[1.89481447552805e-10,3.67431482211078e-12,- 1.72180065021194e-11,1.47046319023226e-11,1.31920481414062e-11],[2.10125915737167e-12,- 3.08420783495975e-12,- 4.8774871236302e-12,1.1636359990249e-14,1.26698255558605e-13],[- 8.07894928696254e-12,9.19344620512607e-13,3.26929173307443e-13,2.00438149416495e-13,- 9.57035765212079e-15],[1.38737151773284e-12,1.0934017837142e-13,5.15714202449053e-14,- 5.92156438588931e-14,- 3.29586752336143e-14],[6.38137197198254e-06,4.62426300749908e-06,4.42334454191034e-06,1.15374736092349e-06,- 2.61859702227253e-06],[- 2.25320619636149e-07,3.21907705479353e-07,- 3.34834530764823e-07,- 4.8213275360181e-07,- 3.22410936343355e-07],[3.48894515496995e-09,3.49951261408458e-08,- 6.01128959281142e-09,4.78213900943443e-09,1.46012816168576e-08],[- 9.66682871952083e-11,3.75806627535317e-09,2.38984004956705e-09,2.07545049877203e-09,1.58573595632766e-09],[1.06834370693917e-09,- 4.07975055112153e-10,- 2.37598937943957e-10,5.89327007480137e-11,1.18891820437634e-10],[5.22433722695807e-11,6.02011995016293e-12,- 7.80605402956048e-12,1.50873145627341e-11,- 1.40550093106311e-12],[2.13396242187279e-13,- 1.71939313965536e-12,- 3.57625378660975e-14,- 5.01675184988446e-14,- 1.07805487368797e-12],[- 1.24352330043311e-12,8.26105883301606e-13,4.63606970128517e-13,6.39517888984486e-14,- 7.35135439920086e-14],[- 5.39023859065631e-13,2.54188315588243e-14,1.30933833278664e-14,6.06153473304781e-15,- 4.24722717533726e-14],[3.12767756884813e-14,- 2.29517847871632e-15,2.53117304424948e-16,7.07504914138118e-16,- 1.20089065310688e-15],[2.08311178819214e-06,- 1.22179185044174e-06,- 2.98842190131044e-06,3.07310218974299e-06,2.27100346036619e-06],[- 3.94601643855452e-07,- 5.44014825116083e-07,- 6.16955333162507e-08,- 2.3195482158067e-07,1.1401081300531e-07],[6.11067575043044e-08,- 3.93240193194272e-08,- 1.62979132528933e-08,1.01339204652581e-08,1.97319601566071e-08],[2.57770508710055e-09,1.87799543582899e-09,1.95407654714372e-09,1.1527641928127e-09,2.2539700540212e-09],[7.16926338026236e-10,- 3.65857693313858e-10,- 1.54864067050915e-11,6.50770211276549e-11,- 7.85160007413546e-12],[4.90007693914221e-12,3.3164939653634e-12,4.8166487116564e-13,7.26080745617085e-12,2.30960953372164e-12],[9.75489202240545e-13,- 1.68967954531421e-13,7.3838339133411e-13,- 3.58435515913239e-13,- 3.0156471002745e-13],[- 3.79533601922805e-13,2.76681830946617e-13,1.21480375553803e-13,- 1.5772907764485e-14,- 8.876649778187e-14],[- 3.96462845480288e-14,2.9415569093461e-14,6.78413205760717e-15,- 4.12135802787361e-15,- 1.46373307795619e-14],[- 8.64941937408121e-15,- 1.91822620970386e-15,- 8.01725413560744e-16,5.02941051180784e-16,- 1.07572628474344e-15],[- 4.13816294742758e-15,- 7.4360201978588e-17,- 5.54248556346072e-17,- 4.83999456005158e-17,- 1.19622559730466e-16],[- 8.34852132750364e-07,- 7.45794677612056e-06,- 6.58132648865533e-06,- 1.38608110346732e-06,5.32326534882584e-07],[- 2.7551380241415e-07,3.64713745106279e-08,- 7.12385417940442e-08,- 7.86206067228882e-08,2.28048393207161e-08],[- 4.26696415431918e-08,- 4.65599668635087e-09,7.35037936327566e-09,1.17098354115804e-08,1.44594777658035e-08],[1.12407689274199e-09,7.62142529563709e-10,- 6.72563708415472e-10,- 1.18094592485992e-10,- 1.17043815733292e-09],[1.76612225246125e-10,- 1.01188552503192e-10,7.32546072616968e-11,1.7954282180161e-11,- 2.23264859965402e-11],[- 9.35960722512375e-12,1.90894283812231e-12,- 6.3479282452576e-13,3.98597963877826e-12,- 4.47591409078971e-12],[- 3.34623858556099e-12,4.56384903915853e-14,2.72561108521416e-13,- 3.57942733300468e-15,1.99794810657713e-13],[- 6.16775522568954e-14,8.25316968328823e-14,7.19845814260518e-14,- 2.92415710855106e-14,- 5.49570017444031e-15],[- 8.50728802453217e-15,8.38161600916267e-15,3.43651657459983e-15,- 8.1942943411591e-16,- 4.089057464611e-15],[4.39042894275548e-15,- 3.69440485320477e-16,1.22249256876779e-16,- 2.09359444520984e-16,- 3.34211740264257e-16],[- 5.36054548134225e-16,3.29794204041989e-17,2.13564354374585e-17,- 1.37838993720865e-18,- 1.29188342867753e-17],[- 3.26421841529845e-17,7.38235405234126e-18,2.4929165967621e-18,8.18252735459593e-19,1.7382495227923e-20],[4.67237509268208e-06,1.93611283787239e-06,9.39035455627622e-07,- 5.84565118072823e-07,- 1.76198705802101e-07],[- 3.33739157421993e-07,4.12139555299163e-07,1.58754695700856e-07,1.37448753329669e-07,1.04722936936873e-07],[6.64200603076386e-09,1.45412222625734e-08,1.8249879611803e-08,2.86633517581614e-09,1.060669845481e-09],[5.25549696746655e-09,- 1.33677183394083e-09,7.60804375937931e-11,- 1.07918624219037e-10,8.09178898247941e-10],[1.89318454110039e-10,9.23092164791765e-11,5.5143457313118e-11,3.8669639228924e-11,- 1.15208165047149e-11],[- 1.02252706006226e-12,- 7.25921015411136e-13,- 1.9811012688762e-12,- 2.18964868282672e-13,- 7.18834476685625e-13],[- 2.69770025318548e-12,- 2.17850340796321e-14,4.73040820865871e-13,1.57947421572149e-13,1.86925164972766e-13],[1.07831718354771e-13,2.26681841611017e-14,2.56046087047783e-14,- 1.14995851659554e-14,- 2.27056907624485e-14],[6.29825154734712e-15,8.04458225889001e-16,9.53173540411138e-16,1.16892301877735e-15,- 1.04324684545047e-15],[- 5.57345639727027e-16,- 2.93949227634932e-16,7.47621406284534e-18,- 5.36416885470756e-17,- 2.87213280230513e-16],[1.73219775047208e-16,2.05017387523061e-17,9.08873886345587e-18,- 2.86881547225742e-18,- 1.25303645304992e-17],[- 7.30829109684568e-18,2.03711261415353e-18,7.62162636124024e-19,- 7.54847922012517e-19,- 8.8510509819503e-19],[5.62039968280587e-18,- 1.38144206573507e-19,1.68028711767211e-20,1.81223858251981e-19,- 8.50245194985878e-20]])
        self.anm_bw = np.array([[0.00136127467401223,- 6.83476317823061e-07,- 1.37211986707674e-06,7.02561866200582e-07,- 2.16342338010651e-07],[- 9.53197486400299e-06,6.58703762338336e-06,2.42000663952044e-06,- 6.04283463108935e-07,2.0214442467699e-07],[- 6.76728911259359e-06,6.03830755085583e-07,- 8.72568628835897e-08,2.21750344140938e-06,1.0514603293102e-06],[- 3.21102832397338e-05,- 7.88685357568093e-06,- 2.55495673641049e-06,- 1.99601934456719e-06,- 4.62005252198027e-07],[- 7.8463926352325e-07,3.11624739733849e-06,9.02170019697389e-07,6.37066632506008e-07,- 9.44485038780872e-09],[2.19476873575507e-06,- 2.20580510638233e-07,6.94761415598378e-07,4.80770865279717e-07,- 1.34357837196401e-07],[2.18469215148328e-05,- 1.80674174262038e-06,- 1.5275428560506e-06,- 3.51212288219241e-07,2.73741237656351e-06],[2.85579058479116e-06,1.57201369332361e-07,- 2.80599072875081e-07,- 4.91267304946072e-07,- 2.11648188821805e-07],[2.8172925559477e-06,3.02487362536122e-07,- 1.64836481475431e-07,- 2.11607615408593e-07,- 6.47817762225366e-08],[1.31809947620223e-07,- 1.58289524114549e-07,- 7.05580919885505e-08,5.56781440550867e-08,1.23403290710365e-08],[- 1.29252282695869e-05,- 1.0724707203759e-05,- 3.31109519638196e-06,2.13776673779736e-06,- 1.49519398373391e-07],[1.81685152305722e-06,- 1.17362204417861e-06,- 3.1920527713637e-08,4.09166457255416e-07,1.53286667406152e-07],[1.63477723125362e-06,- 2.68584775517243e-08,4.94662064805191e-09,- 7.09027987928288e-08,4.44353430574937e-08],[- 2.13090618917978e-07,4.05836983493219e-08,2.94495876336549e-08,- 1.75005469063176e-08,- 3.03015988647002e-09],[- 2.16074435298006e-09,9.37631708987675e-09,- 2.05996036369828e-08,6.97068002894092e-09,- 8.90988987979604e-09],[1.38047798906967e-05,2.05528261553901e-05,1.59072148872708e-05,7.34088731264443e-07,1.2822671038358e-06],[7.08175753966264e-07,- 9.27988276636505e-07,1.60535820026081e-07,- 3.27296675122065e-07,- 2.20518321170684e-07],[1.90932483086199e-07,- 7.44215272759193e-08,1.81330673333187e-08,4.37149649043616e-08,4.18884335594172e-08],[- 5.37009063880924e-08,2.22870057779431e-08,1.73740123037651e-08,- 4.45137302235032e-09,9.44721910524571e-09],[- 6.83406949047909e-08,- 1.95046676795923e-10,2.57535903049686e-09,4.8264316408302e-09,3.37657333705158e-09],[3.96128688448981e-09,- 6.63809403270686e-10,2.44781464212534e-10,5.92280853590699e-11,- 4.78502591970721e-10],[1.75859399041414e-05,- 2.81238050668481e-06,- 2.43670534594848e-06,3.58244562699714e-06,- 1.76547446732691e-06],[- 1.06451311473304e-07,1.54336689617184e-06,- 2.00690000442673e-07,1.3879004791188e-09,- 1.62490619890017e-07],[- 2.72757421686155e-07,1.71139266205398e-07,- 2.55080309401917e-08,- 8.40793079489831e-09,- 1.01129447760167e-08],[2.92966025844079e-08,- 2.07556718857313e-08,5.45985315647905e-09,8.7685769027415e-09,1.06785510440474e-08],[- 1.22059608941331e-08,6.52491630264276e-09,- 1.79332492326928e-10,3.75921793745396e-10,- 7.06416506254786e-10],[1.63224355776652e-09,4.95586028736232e-10,- 3.0787901175904e-10,- 7.78354087544277e-11,1.4395904706725e-10],[3.86319414653663e-10,- 2.06467134617933e-10,4.37330971382694e-11,- 5.00421056263711e-11,- 9.40237773015723e-12],[- 1.23856142706451e-05,7.61047394008415e-06,- 1.99104114578138e-07,6.86177748886858e-07,- 1.09466747592827e-07],[2.99866062403128e-07,1.8752556139739e-07,4.99374806994715e-08,4.86229763781404e-07,4.46570575517658e-07],[- 5.0574833236843e-07,1.95523624722285e-08,- 9.17535435911345e-08,- 2.56671607433547e-08,- 7.11896201616653e-08],[- 2.66062200406494e-08,- 5.40470019739274e-09,- 2.29718660244954e-09,- 3.73328592264404e-09,3.38748313712376e-09],[5.30855327954894e-10,5.28851845648032e-10,- 2.22278913745418e-10,- 5.52628653064771e-11,- 9.24825145219684e-10],[6.03737227573716e-10,- 3.52190673510919e-12,- 1.30371720641414e-10,- 9.12787239944822e-12,6.42187285537238e-12],[1.78081862458539e-10,2.93772078656037e-12,- 1.04698379945322e-11,- 2.82260024833024e-11,- 5.61810459067525e-12],[9.3500309229958e-12,- 8.23133834521577e-13,5.54878414224198e-13,- 3.62943215777181e-13,2.38858933771653e-12],[- 1.31216096107331e-05,- 5.70451670731759e-06,- 5.11598683573971e-06,- 4.99990779887599e-06,1.27389320221511e-07],[- 1.23108260369048e-06,5.53093245213587e-07,8.60093183929302e-07,2.65569700925696e-07,1.95485134805575e-07],[- 2.29647072638049e-07,- 5.45266515081825e-08,2.85298129762263e-08,1.98167939680185e-08,5.52227340898335e-09],[- 2.73844745019857e-08,- 4.48345173291362e-10,- 1.93967347049382e-09,- 1.41508853776629e-09,- 1.75456962391145e-09],[- 2.68863184376108e-11,- 2.20546981683293e-09,6.56116990576877e-10,1.27129855674922e-10,- 2.32334506413213e-10],[1.98303136881156e-10,6.04782006047075e-11,2.9129111543157e-11,6.18098615782757e-11,- 3.82682292530379e-11],[9.48294455071158e-12,- 3.05873596453015e-13,5.31539408055057e-13,- 7.310164386656e-12,- 1.19921002209198e-11],[- 2.25188050845725e-11,- 3.91627574966393e-13,- 6.80217235976769e-13,5.91033607278405e-13,5.02991534452191e-13],[1.29532063896247e-12,1.66337285851564e-13,3.25543028344555e-13,1.89143357962363e-13,3.32288378169726e-13],[- 2.45864358781728e-06,4.4946052489826e-06,1.03890496648813e-06,- 2.73783420376785e-06,7.12695730642593e-07],[- 9.27805078535168e-07,- 4.97733876686731e-07,9.1868029890651e-08,- 2.4720061742398e-07,6.16163630140379e-08],[- 1.39623661883136e-08,- 1.12580495666505e-07,2.61821435950379e-08,- 2.31875562002885e-08,5.72679835033659e-08],[- 9.52538983318497e-09,- 5.40909215302433e-09,1.88698793952475e-09,- 4.08127746406372e-09,1.09534895853812e-10],[3.79767457525741e-09,1.11549801373366e-10,- 6.45504957274111e-10,3.05477141010356e-10,1.26261210565856e-10],[5.088135779453e-11,1.43250547678637e-11,8.81616572082448e-12,2.58968878880804e-11,3.83421818249954e-11],[8.95094368142044e-12,- 3.26220304555971e-12,- 1.28047847191896e-12,2.67562170258942e-12,2.7219503157667e-12],[- 6.47181697409757e-12,1.13776457455685e-12,2.84856274334969e-13,- 7.63667272085395e-14,- 1.34451657758826e-13],[- 1.25291265888343e-12,8.63500441050317e-14,- 1.21307856635548e-13,5.12570529540511e-14,3.32389276976573e-14],[3.73573418085813e-14,- 5.37808783042784e-16,- 4.2343040827085e-16,- 4.75110565740493e-15,6.02553212780166e-15],[8.95483987262751e-06,- 3.90778212666235e-06,- 1.12115019808259e-06,1.78678942093383e-06,1.46806344157962e-06],[- 4.59185232678613e-07,1.09497995905419e-07,1.31663977640045e-07,4.20525791073626e-08,- 9.71470741607431e-08],[1.63399802579572e-07,1.50909360648645e-08,- 1.11480472593347e-08,- 1.84000857674573e-08,7.82124614794256e-09],[1.22887452385094e-08,- 4.06647399822746e-10,- 6.49120327585597e-10,8.63651225791194e-10,- 2.73440085913102e-09],[2.51748630889583e-09,4.79895880425564e-10,- 2.44908073860844e-10,2.56735882664876e-10,- 1.64815306286912e-10],[4.85671381736718e-11,- 2.51742732115131e-11,- 2.60819437993179e-11,6.12728324086123e-12,2.16833310896138e-11],[4.11389702320298e-12,- 8.09433180989935e-13,- 1.19812498226024e-12,1.4688573788852e-12,3.15807685137836e-12],[- 1.47614580597013e-12,4.6672641390932e-13,1.72089709006255e-13,1.13854935381418e-13,2.77741161317003e-13],[- 1.02257724967727e-13,1.10394382923502e-13,- 3.14153505370805e-15,2.41103099110106e-14,2.13853053149771e-14],[- 3.19080885842786e-14,- 9.53904307973447e-15,2.74542788156379e-15,2.33797859107844e-15,- 2.53192474907304e-15],[- 5.87702222126367e-15,- 1.80133850930249e-15,- 3.09793125614454e-16,- 1.04197538975295e-16,3.72781664701327e-16],[1.86187054729085e-06,8.33098045333428e-06,3.18277735484232e-06,- 7.68273797022231e-07,- 1.52337222261696e-06],[- 5.07076646593648e-07,- 8.61959553442156e-07,- 3.51690005432816e-07,- 4.20797082902431e-07,- 3.07652993252673e-07],[- 7.38992472164147e-08,- 8.3947308308028e-08,- 2.51587083298935e-08,7.30691259725451e-09,- 3.19457155958983e-08],[- 1.99777182012924e-09,- 3.21265085916022e-09,- 4.84477421865675e-10,- 1.82924814205799e-09,- 3.46664344655997e-10],[- 7.05788559634927e-11,1.21840735569025e-10,7.97347726425926e-11,1.08275679614409e-10,- 1.17891254809785e-10],[1.10299718947774e-11,- 3.22958261390263e-11,- 1.43535798209229e-11,6.87096504209595e-12,- 6.64963212272352e-12],[- 6.47393639740084e-12,1.0315697832512e-12,- 9.20099775082358e-14,- 2.40150316641949e-13,1.14008812047857e-12],[- 1.2395784639725e-13,2.85996703969692e-13,1.91579874982553e-13,5.20597174693064e-14,- 4.0674143488337e-14],[- 2.35479068911236e-14,1.97847338186993e-14,1.58935977518516e-15,- 2.32217195254742e-15,- 8.48611789490575e-15],[1.03992320391626e-14,1.54017082092642e-15,1.05950035082788e-16,- 1.17870898461353e-15,- 1.10937420707372e-15],[- 1.0901194837452e-15,- 6.04168007633584e-16,- 9.10901998157436e-17,1.98379116989461e-16,- 1.03715496658498e-16],[- 1.38171942108278e-16,- 6.33037999097522e-17,- 1.3877769501147e-17,1.94191397045401e-17,5.70055906754485e-18],[1.92989406002085e-06,- 3.82662130483128e-06,- 4.60189561036048e-07,2.24290587856309e-06,1.4054437945155e-06],[6.49033717633394e-08,2.41396114435326e-07,2.73948898223321e-07,1.10633664439332e-07,- 3.19555270171075e-08],[- 2.91988966963297e-08,- 6.03828192816571e-09,1.1846238644484e-08,1.32095545004128e-08,- 5.06572721528914e-09],[7.31079058474148e-09,- 8.42775299751834e-10,1.10190810090667e-09,1.96592273424306e-09,- 2.13135932785688e-09],[7.06656405314388e-11,1.43441125783756e-10,1.46962246686924e-10,7.44592776425197e-11,- 3.64331892799173e-11],[- 2.52393942119372e-11,1.07520964869263e-11,5.84669886072094e-12,6.52029744217103e-12,1.82947123132059e-12],[- 4.15669940115121e-12,- 1.95963254053648e-13,2.16977822834301e-13,- 2.84701408462031e-13,4.27194601040231e-13],[3.07891105454129e-13,1.91523190672955e-13,1.05367297580989e-13,- 5.28136363920236e-14,- 3.53364110005917e-14],[7.02156663274738e-15,9.52230536780849e-15,- 3.41019408682733e-15,- 3.59825303352899e-15,- 2.6257641163615e-15],[- 1.75110277413804e-15,5.29265220719483e-16,4.45015980897919e-16,- 3.80179856341347e-16,- 4.32917763829695e-16],[1.16038609651443e-16,- 6.69643574373352e-17,2.65667154817303e-17,- 9.76010333683956e-17,4.07312981076655e-17],[5.72659246346386e-18,1.30357528108671e-18,2.49193258417535e-18,1.76247014075584e-18,7.59614374197688e-19],[1.03352170833303e-17,- 2.30633516638829e-18,2.84777940620193e-18,- 7.72161347944693e-19,6.0702803450638e-19]])
        self.anm_ch = np.array([[0.0571481238161787,3.35402081801137e-05,3.15988141788728e-05,- 1.34477341887086e-05,- 2.61831023577773e-07],[5.77367395845715e-05,- 0.000669057185209558,- 6.51057691648904e-05,- 1.61830149147091e-06,8.96771209464758e-05],[- 8.50773002452907e-05,- 4.87106614880272e-05,4.03431160775277e-05,2.54090162741464e-06,- 5.59109319864264e-06],[0.00150536423187709,0.000611682258892697,0.000369730024614855,- 1.95658439780282e-05,- 3.462467265537e-05],[- 2.32168718433966e-05,- 0.000127478686553809,- 9.00292451740728e-05,- 6.0783431590183e-05,- 1.04628419422714e-05],[- 1.38607250922551e-06,- 3.97271603842309e-06,- 8.16155320152118e-07,5.73266706046665e-07,2.00366060212696e-07],[6.52491559188663e-05,- 0.00112224323460183,- 0.000344967958304075,- 7.672826409473e-05,0.000107907110551939],[- 0.000138870461448036,- 7.29995695401936e-05,5.35986591445824e-05,9.0380486970389e-06,8.61370129482732e-06],[- 9.98524443968768e-07,- 6.84966792665998e-08,1.47478021860771e-07,1.94857794008064e-06,7.1717685273291e-07],[1.2706636791172e-06,1.12113289164288e-06,2.71525688515375e-07,- 2.76125723009239e-07,- 1.05429690305013e-07],[- 0.000377264999981652,0.000262691217024294,0.00018363978583759,3.93177048515576e-06,- 6.66187081899168e-06],[- 4.93720951871921e-05,- 0.000102820030405771,- 5.69904376301748e-05,- 3.79603438055116e-05,- 3.9672601783493e-06],[- 2.21881958961135e-06,- 1.40207117987894e-06,1.60956630798516e-07,2.06121145135022e-06,6.50944708093149e-07],[2.21876332411271e-07,1.92272880430386e-07,- 6.44016558013941e-09,- 1.4095492133241e-07,- 4.26742169137667e-07],[- 3.51738525149881e-08,2.89616194332516e-08,- 3.40343352397886e-08,- 2.89763392721812e-08,- 6.40980581663785e-10],[3.51240856823468e-05,- 0.000725895015345786,- 0.000322514037108045,- 0.000106143759981636,4.08153152459337e-05],[- 2.36269716929413e-05,- 4.20691836557932e-05,1.43926743222922e-05,2.61811210631784e-05,2.09610762194903e-05],[- 7.9176575667389e-07,1.64556789159745e-06,- 9.43930166276555e-07,6.46641738736139e-07,- 5.91509547299176e-07],[3.92768838766879e-07,- 1.9802773170369e-07,- 5.41303590057253e-08,- 4.21705797874207e-07,- 6.06042329660681e-08],[- 1.56650141024305e-08,7.61808165752027e-08,- 1.81900460250934e-08,1.30196216971675e-08,1.08616031342379e-08],[- 2.80964779829242e-08,- 7.25951488826103e-09,- 2.59789823306225e-09,- 2.79271942407154e-09,4.10558774868586e-09],[- 0.000638227857648286,- 0.000154814045363391,7.78518327501759e-05,- 2.95961469342381e-05,1.15965225055757e-06],[4.47833146915112e-06,1.33712284237555e-05,3.61048816552123e-06,- 2.50717844073547e-06,- 1.28100822021734e-05],[- 2.26958070007455e-06,2.57779960912242e-06,1.08395653197976e-06,1.29403393862805e-07,- 1.04854652812567e-06],[- 3.98954043463392e-07,- 2.26931182815454e-07,- 1.09169545045028e-07,- 1.49509536031939e-07,- 3.98376793949903e-07],[2.3041891107111e-08,1.23098508481555e-08,- 1.71161401463708e-08,2.35829696577657e-09,1.3113616416204e-08],[3.69423793101582e-09,3.49231027561927e-10,- 1.18581468768647e-09,5.4318073582882e-10,5.43192337651588e-10],[- 1.38608847117992e-09,- 1.86719145546559e-10,- 8.13477384765498e-10,2.01919878240491e-10,1.00067892622287e-10],[- 4.35499078415956e-05,0.000450727967957804,0.00032897849426885,- 3.05249478582848e-05,- 3.2191483454431e-05],[1.24887940973241e-05,1.34275239548403e-05,1.11275518344713e-06,7.46733554562851e-06,- 2.12458664760353e-06],[9.50250784948476e-07,2.34367372695203e-06,- 5.4309924479898e-07,- 4.35196904508734e-07,- 8.31852234345897e-07],[5.91775478636535e-09,- 1.48970922508592e-07,2.9984006117384e-08,- 1.30595933407792e-07,1.27136765045597e-07],[- 1.78491083554475e-08,1.76864919393085e-08,- 1.96740493482011e-08,1.21096708004261e-08,2.95518703155064e-10],[1.75053510088658e-09,- 1.31414287871615e-09,- 1.44689439791928e-09,1.1468248366846e-09,1.74488616540169e-09],[1.08152964586251e-09,- 3.85678162063266e-10,- 2.77851016629979e-10,3.8989057862559e-11,- 2.54627365853495e-10],[- 1.88340955578221e-10,5.19645384002867e-11,2.14131326027631e-11,1.24027770392728e-11,- 9.42818962431967e-12],[0.000359777729843898,- 0.000111692619996219,- 6.87103418744904e-05,0.000115128973879551,7.59796247722486e-05],[5.23717968000879e-05,1.32279078116467e-05,- 5.72277317139479e-07,- 7.56326558610214e-06,- 1.95749622214651e-05],[1.00109213210139e-06,- 2.75515216592735e-07,- 1.13393194050846e-06,- 4.75049734870663e-07,- 3.21499480530932e-07],[- 2.0701371659889e-07,- 7.31392258077707e-08,- 3.9644571408416e-08,3.21390452929387e-08,- 1.43738764991525e-08],[2.03081434931767e-09,- 1.35423687136122e-08,- 4.47637454261816e-09,2.18409121726643e-09,- 3.74845286805217e-09],[3.17469255318367e-09,2.44221027314129e-10,- 2.46820614760019e-10,7.55851003884434e-10,6.98980592550891e-10],[9.89541493531067e-11,- 2.78762878057315e-11,- 2.10947962916771e-10,3.77882267360636e-11,- 1.20009542671532e-12],[5.0172057573094e-11,1.66470417102135e-11,- 7.50624817938091e-12,9.97880221482238e-12,4.87141864438892e-12],[2.53137945301589e-11,1.93030083090772e-12,- 1.4470880423129e-12,- 1.77837100743423e-12,- 8.10068935490951e-13],[0.000115735341520738,0.00011691059104835,8.36315620479475e-05,1.61095702669207e-05,- 7.53084853489862e-05],[- 9.76879433427199e-06,9.16968438003335e-06,- 8.7275512728883e-06,- 1.30077933880053e-05,- 9.7884193799332e-06],[1.04902782517565e-07,2.14036988364936e-07,- 7.19358686652888e-07,1.12529592946332e-07,7.07316352860448e-07],[7.6317726528508e-08,1.2278197443429e-07,8.99971272969286e-08,5.6348223935299e-08,4.31054352285547e-08],[3.29855763107355e-09,- 6.95004336734441e-09,- 6.52491370576354e-09,1.97749180391742e-09,3.51941791940498e-09],[3.85373745846559e-10,1.65754130924183e-10,- 3.31326088103057e-10,5.93256024580436e-10,1.27725220636915e-10],[- 1.08840956376565e-10,- 4.56042860268189e-11,- 4.77254322645633e-12,- 2.94405398621875e-12,- 3.07199979999475e-11],[2.0738987909501e-11,1.51186798732451e-11,9.28139802941848e-12,5.92738269687687e-12,9.70337402306505e-13],[- 2.85879708060306e-12,1.92164314717053e-13,4.0266467896789e-14,5.18246319204277e-13,- 7.91438726419423e-13],[6.91890667590734e-13,- 8.49442290988352e-14,- 5.54404947212402e-15,9.7109337753879e-15,- 5.33714333415971e-14],[- 5.06132972789792e-05,- 4.28348772058883e-05,- 6.90746551020305e-05,8.48380415176836e-05,7.04135614675053e-05],[- 1.27945598849788e-05,- 1.92362865537803e-05,- 2.30971771867138e-06,- 8.98515975724166e-06,5.25675205004752e-06],[- 8.71907027470177e-07,- 1.02091512861164e-06,- 1.69548051683864e-07,4.87239045855761e-07,9.13163249899837e-07],[- 6.23651943425918e-08,6.98993315829649e-08,5.9159776673339e-08,4.36227124230661e-08,6.45321798431575e-08],[- 1.46315079552637e-10,- 7.85142670184337e-09,1.48788168857903e-09,2.1687049991216e-09,- 1.16723047065545e-09],[3.31888494450352e-10,1.90931898336457e-10,- 3.13671901557599e-11,2.60711798190524e-10,8.45240112207997e-11],[1.36645682588537e-11,- 5.68830303783976e-12,1.5751892384814e-11,- 1.61935794656758e-11,- 4.16568077748351e-12],[9.44684950971905e-13,7.30313977131995e-12,3.14451447892684e-12,6.49029875639842e-13,- 9.66911019905919e-13],[- 8.13097374090024e-13,5.23351897822186e-13,8.94349188113951e-14,- 1.3332775967327e-13,- 4.04549450989029e-13],[- 3.76176467005839e-14,- 6.19953702289713e-14,- 3.74537190139726e-14,1.71275486301958e-14,- 3.81946773167132e-14],[- 4.8139338554416e-14,3.66084990006325e-15,3.10432030972253e-15,- 4.10964475657416e-15,- 6.586442442429e-15],[- 7.81077363746945e-05,- 0.000254773632197303,- 0.000214538508009518,- 3.80780934346726e-05,1.8349535919399e-05],[5.89140224113144e-06,- 3.17312632433258e-06,- 3.81872516710791e-06,- 2.27592226861647e-06,1.57044619888023e-06],[- 1.4427250508869e-06,- 1.10236588903758e-07,2.64336813084693e-07,4.7607416333246e-07,4.2862358769457e-07],[3.98889120733904e-08,- 1.29638005554027e-08,- 4.13668481273828e-08,1.27686793719542e-09,- 3.54202962042383e-08],[1.6072683755175e-09,- 2.70750776726156e-09,2.7938709268107e-09,- 3.01419734793998e-10,- 1.29101669438296e-10],[- 2.55708290234943e-10,2.27878015173471e-11,- 6.43063443462716e-12,1.26531554846856e-10,- 1.6582214743722e-10],[- 3.35886470557484e-11,- 3.51895009091595e-12,5.80698399963198e-12,- 2.84881487149207e-12,8.91708061745902e-12],[- 3.12788523950588e-12,3.35366912964637e-12,2.52236848033838e-12,- 8.12801050709184e-13,- 2.63510394773892e-13],[6.83791881183142e-14,2.41583263270381e-13,8.58807794189356e-14,- 5.12528492761045e-14,- 1.40961725631276e-13],[- 1.28585349115321e-14,- 2.11049721804969e-14,5.26409596614749e-15,- 4.31736582588616e-15,- 1.60991602619068e-14],[- 9.35623261461309e-15,- 3.94384886372442e-16,5.04633016896942e-16,- 5.40268998456055e-16,- 1.07857944298104e-15],[8.79756791888023e-16,4.5252993567533e-16,1.36886341163227e-16,- 1.12984402980452e-16,6.30354561057224e-18],[0.000117829256884757,2.67013591698442e-05,2.5791344677525e-05,- 4.40766244878807e-05,- 1.60651761172523e-06],[- 1.87058092029105e-05,1.34371169060024e-05,5.59131416451555e-06,4.50960364635647e-06,2.87612873904633e-06],[2.79835536517287e-07,8.93092708148293e-07,8.37294601021795e-07,- 1.99029785860896e-08,- 8.87240405168977e-08],[4.95854313394905e-08,- 1.44694570735912e-08,2.51662229339375e-08,- 3.87086600452258e-09,2.2974191907127e-08],[4.71497840986162e-09,2.47509999454076e-09,1.67323845102824e-09,8.1419676828353e-10,- 3.71467396944165e-10],[- 1.07340743907054e-10,- 8.07691657949326e-11,- 5.99381660248133e-11,2.33173929639378e-12,- 2.26994195544563e-11],[- 3.83130441984224e-11,- 5.82499946138714e-12,1.43286311435124e-11,3.15150503353387e-12,5.97891025146774e-12],[- 5.6438919107223e-13,9.57258316335954e-13,1.12055192185939e-12,- 4.4241770677542e-13,- 9.93190361616481e-13],[1.78188860269677e-13,7.8258202490495e-14,5.18061650118009e-14,2.13456507353387e-14,- 5.2620211377951e-14],[- 8.18481324740893e-15,- 3.71256746886786e-15,4.23508855164371e-16,- 2.91292502923102e-15,- 1.1545420538935e-14],[6.1657869169681e-15,6.74087154080877e-16,5.71628946437034e-16,- 2.05251213979975e-16,- 7.25999138903781e-16],[9.35481959699383e-17,6.23535830498083e-17,3.1807672880206e-18,- 2.92353209354587e-17,7.65216088665263e-19],[2.34173078531701e-17,- 8.30342420281772e-18,- 4.33602329912952e-18,1.90226281379981e-18,- 7.85507922718903e-19]])
        self.anm_cw = np.array([[0.0395329695826997,- 0.000131114380761895,- 0.000116331009006233,6.23548420410646e-05,5.72641113425116e-05],[- 0.00044183764088065,0.000701288648654908,0.00033848980285827,3.76700309908602e-05,- 8.70889013574699e-06],[1.30418530496887e-05,- 0.000185046547597376,4.31032103066723e-05,0.000105583334124319,3.23045436993589e-05],[3.68918433448519e-05,- 0.000219433014681503,3.46768613485e-06,- 9.17185187163528e-05,- 3.69243242456081e-05],[- 6.50227201116778e-06,2.07614874282187e-05,- 5.09131314798362e-05,- 3.08053225174359e-05,- 4.18483655873918e-05],[2.67879176459056e-05,- 6.89303730743691e-05,2.11046783217168e-06,1.93163912538178e-05,- 1.97877143887704e-06],[0.000393937595007422,- 0.000452948381236406,- 0.000136517846073846,0.000138239247989489,0.000133175232977863],[5.00214539435002e-05,3.57229726719727e-05,- 9.38010547535432e-07,- 3.52586798317563e-05,- 7.01218677681254e-06],[3.91965314099929e-05,1.02236686806489e-05,- 1.95710695226022e-05,- 5.93904795230695e-06,3.24339769876093e-06],[6.68158778290653e-06,- 8.10468752307024e-06,- 9.91192994096109e-06,- 1.89755520007723e-07,- 3.26799467595579e-06],[0.000314196817753895,- 0.000296548447162009,- 0.000218410153263575,- 1.57318389871e-05,4.69789570185785e-05],[0.000104597721123977,- 3.31000119089319e-05,5.60326793626348e-05,4.71895007710715e-05,3.57432326236664e-05],[8.95483021572039e-06,1.44019305383365e-05,4.87912790492931e-06,- 3.45826387853503e-06,3.23960320438157e-06],[- 1.3524965100993e-05,- 2.49349762695977e-06,- 2.51509483521132e-06,- 9.14254874104858e-07,- 8.5789740610089e-07],[- 1.68143325235195e-06,1.72073417594235e-06,1.38765993969565e-06,4.0977098213753e-07,- 6.60908742097123e-07],[- 0.000639889366487161,0.00120194042474696,0.000753258598887703,3.87356377414663e-05,1.31231811175345e-05],[2.77062763606783e-05,- 9.51425270178477e-06,- 6.61068056107547e-06,- 1.38713669012109e-05,9.84662092961671e-06],[- 2.69398078539471e-06,6.50860676783123e-06,3.8085592698809e-06,- 1.98076068364785e-06,1.17187335666772e-06],[- 2.63719028151905e-06,5.03149473656743e-07,7.38964893399716e-07,- 8.38892485369078e-07,1.30943917775613e-06],[- 1.56634992245479e-06,- 2.97026487417045e-08,5.06602801102463e-08,- 4.60436007958792e-08,- 1.62536449440997e-07],[- 2.37493912770935e-07,1.69781593069938e-08,8.35178275224265e-08,- 4.83564044549811e-08,- 4.96448864199318e-08],[0.00134012259587597,- 0.000250989369253194,- 2.97647945512547e-05,- 6.47889968094926e-05,8.41302130716859e-05],[- 0.000113287184900929,4.78918993866293e-05,- 3.14572113583139e-05,- 2.10518256626847e-05,- 2.03933633847417e-05],[- 4.97413321312139e-07,3.72599822034753e-06,- 3.53221588399266e-06,- 1.05232048036416e-06,- 2.74821498198519e-06],[4.81988542428155e-06,4.21400219782474e-07,1.02814808667637e-06,4.40299068486188e-09,3.37103399036634e-09],[1.10140301678818e-08,1.90257670180182e-07,- 1.00831353341885e-08,1.44860642389714e-08,- 5.29882089987747e-08],[6.12420414245775e-08,- 4.48953461152996e-09,- 1.38837603709003e-08,- 2.05533675904779e-08,1.49517908802329e-09],[9.17090243673643e-10,- 9.24878857867367e-09,- 2.30856560363943e-09,- 4.36348789716735e-09,- 4.45808881183025e-10],[- 0.000424912699609112,- 0.000114365438471564,- 0.000403200981827193,4.19949560550194e-05,- 3.02068483713739e-05],[3.85435472851225e-05,- 5.70726887668306e-05,4.96313706308613e-07,1.02395703617082e-05,5.85550000567006e-06],[- 7.38204470183331e-06,- 4.56638770109511e-06,- 3.94007992121367e-06,- 2.16666812189101e-06,- 4.55694264113194e-06],[5.89841165408527e-07,1.40862905173449e-08,1.08149086563211e-07,- 2.18592601537944e-07,- 3.78927431428119e-07],[4.85164687450468e-08,8.34273921293655e-08,1.47489605513673e-08,6.01494125001291e-08,6.43812884159484e-09],[1.13055580655363e-08,3.50568765400469e-09,- 5.0939616250175e-09,- 1.83362063152411e-09,- 4.11227251553035e-09],[3.16454132867156e-09,- 1.39634794131087e-09,- 7.34085003895929e-10,- 7.55541371271796e-10,- 1.57568747643705e-10],[1.27572900992112e-09,- 3.51625955080441e-10,- 4.84132020565098e-10,1.52427274930711e-10,1.27466120431317e-10],[- 0.000481655666236529,- 0.000245423313903835,- 0.000239499902816719,- 0.000157132947351028,5.54583099258017e-05],[- 1.52987254785589e-05,2.78383892116245e-05,4.3229912399186e-05,1.70981319744327e-05,- 1.35090841769225e-06],[- 8.65400907717798e-06,- 6.51882656990376e-06,- 2.43810171017369e-07,8.54348785752623e-07,2.98371863248143e-07],[- 1.68155571776752e-06,- 3.53602587563318e-07,- 1.00404435881759e-07,- 2.14162249012859e-08,- 2.42131535531526e-07],[- 1.08048603277187e-08,- 9.7885078576303e-08,- 2.32906554437417e-08,2.22003630858805e-08,- 2.27230368089683e-09],[- 5.98864391551041e-09,7.38970926486848e-09,3.61322835311957e-09,3.70037329172919e-09,- 3.41121137081362e-09],[- 7.33113754909726e-10,- 9.0837424933522e-11,- 1.78204392133739e-10,8.28618491929026e-11,- 1.32966817912373e-10],[- 5.23340481314676e-10,1.36403528233346e-10,- 7.04478837151279e-11,- 6.83175201536443e-12,- 2.86040864071134e-12],[3.75347503578356e-11,- 1.08518134138781e-11,- 2.53583751744508e-12,1.00168232812303e-11,1.74929602713312e-11],[- 0.00068680533637057,0.000591849814585706,0.000475117378328026,- 2.59339398048415e-05,3.74825110514968e-05],[3.35231363034093e-05,2.38331521146909e-05,7.43545963794093e-06,- 3.41430817541849e-06,7.20180957675353e-06],[3.60564374432978e-07,- 3.13300039589662e-06,- 6.3897474610802e-07,- 8.63985524672024e-07,2.43367665208655e-06],[- 4.09605238516094e-07,- 2.51158699554904e-07,- 1.29359217235188e-07,- 2.27744642483133e-07,7.04065989970205e-08],[6.74886341820129e-08,- 1.02009407061935e-08,- 3.30790296448812e-08,1.64959795655031e-08,1.40641779998855e-08],[1.31706886235108e-09,- 1.06243701278671e-09,- 2.85573799673944e-09,3.72566568681289e-09,2.48402582003925e-09],[- 3.68427463251097e-11,- 1.90028122983781e-10,- 3.98586561768697e-11,1.14458831693287e-11,- 2.27722300377854e-12],[- 7.90029729611056e-11,3.81213646526419e-11,4.63303426711788e-11,1.52294835905903e-11,- 2.99094751490726e-12],[- 2.36146602045017e-11,1.03852674709985e-11,- 4.472421263071e-12,5.30884113537806e-12,1.68499023262969e-12],[- 3.30107358134527e-13,- 4.73989085379655e-13,5.17199549822684e-13,2.34951744478255e-13,2.05931351608192e-13],[0.00043021568751178,- 0.000132831373000014,- 3.41830835017045e-05,4.70312161436033e-06,- 3.84807179340006e-05],[1.66861163032403e-05,- 8.1009290852355e-06,8.20658107437905e-06,6.12399025026683e-06,- 1.85536495631911e-06],[1.53552093641337e-06,2.19486495660361e-06,- 1.07253805120137e-06,- 4.72141767909137e-07,4.00744581573216e-07],[2.56647305130757e-07,- 8.07492046592274e-08,- 2.05858469296168e-07,1.09784168930599e-07,- 7.76823030181225e-08],[1.77744008115031e-08,1.6413467781742e-08,4.8616304487902e-09,1.13334251800856e-08,- 7.17260621115426e-09],[1.61133063219326e-09,- 1.85414677057024e-09,- 2.13798537812651e-09,1.15255123229679e-09,2.24504700129464e-09],[1.23344223096739e-10,- 1.20385012169848e-10,- 2.18038256346433e-12,3.23033120628279e-11,8.011795682134e-11],[- 6.55745274387847e-12,1.22127104697198e-11,5.83805016355883e-12,- 8.31201582509817e-12,1.90985373872656e-12],[- 2.89199983667265e-12,5.05962500506667e-12,1.28092925110279e-12,5.60353813743813e-13,1.7675373196877e-12],[- 1.61678729774956e-13,- 3.92206170988615e-13,- 9.04941327579237e-14,1.89847694200763e-13,4.10008676756463e-14],[- 1.16808369005656e-13,- 9.9746459143051e-14,7.46366550245722e-15,2.53398578153179e-14,1.06510689748906e-14],[- 0.00011371692138479,- 0.000131902722651488,- 0.000162844886485788,7.90171538739454e-06,- 0.000178768066961413],[- 2.131465353665e-06,- 3.57818705543597e-05,- 1.50825855069298e-05,- 2.17909259570022e-05,- 8.19332236308581e-06],[- 2.88001138617357e-06,- 2.09957465440793e-06,6.81466526687552e-08,3.58308906974448e-07,- 4.18502067223724e-07],[- 1.10761444317605e-07,6.91773860777929e-08,8.17125372450372e-08,- 2.16476237959181e-08,7.59221970502074e-08],[- 9.56994224818941e-09,6.64104921728432e-09,6.33077902928348e-09,2.85721181743727e-09,- 6.39666681678123e-09],[4.62558627839842e-10,- 1.69014863754621e-09,- 2.80260429599733e-10,4.27558937623863e-11,- 1.66926133269027e-10],[- 7.23385132663753e-11,5.5196119354528e-11,3.04070791942335e-11,3.23227055919062e-12,8.47312431934829e-11],[- 1.61189613765486e-11,1.66868155925172e-11,1.05370341694715e-11,- 4.41495859079592e-12,- 2.2493905140175e-12],[- 8.72229568056267e-13,1.88613726203286e-12,1.2171113753439e-14,- 1.13342372297867e-12,- 6.87151975256052e-13],[7.9931198854409e-15,4.46150979586709e-14,7.50406779454998e-14,- 3.20385428942275e-14,- 1.26543636054393e-14],[4.80503817699514e-14,- 3.35545623603729e-14,- 1.18546423610485e-14,4.1941920998598e-15,- 1.7352561443688e-14],[- 1.20464898830163e-15,- 8.80752065000456e-16,- 1.22214298993313e-15,1.69928513019657e-15,1.93593051311405e-16],[1.68528879784841e-05,3.57144412031081e-05,- 1.65999910125077e-05,5.40370336805755e-05,0.000118138122851376],[- 3.28151779115881e-05,1.04231790790798e-05,- 2.8076186289064e-06,2.98996152515593e-06,- 2.67641158709985e-06],[- 2.08664816151978e-06,- 1.64463884697475e-06,6.79099429284834e-08,7.23955842946495e-07,- 6.86378427465657e-07],[- 2.88205823027255e-09,2.38319699493291e-09,1.14169347509045e-07,8.12981074994402e-08,- 1.56957943666988e-07],[- 7.09711403570189e-09,6.29470515502988e-09,3.50833306577579e-09,8.31289199649054e-09,- 2.14221463168338e-09],[- 8.11910123910038e-10,3.34047829618955e-10,3.7061937744649e-10,3.30426088213373e-10,4.86297305597865e-11],[1.98628160424161e-11,- 4.98557831380098e-12,- 5.90523187802174e-12,- 1.27027116925122e-12,1.49982368570355e-11],[2.62289263262748e-12,3.91242360693861e-12,6.56035499387192e-12,- 1.17412941089401e-12,- 9.40878197853394e-13],[- 3.37805010124487e-13,5.39454874299593e-13,- 2.41569839991525e-13,- 2.41572016820792e-13,- 3.01983673057198e-13],[- 1.85034053857964e-13,4.31132161871815e-14,4.13497222026824e-15,- 4.6007551459598e-14,- 1.92454846400146e-14],[2.96113888929854e-15,- 1.11688534391626e-14,3.76275373238932e-15,- 3.72593295948136e-15,1.98205490249604e-16],[1.40074667864629e-15,- 5.15564234798333e-16,3.56287382196512e-16,5.07242777691587e-16,- 2.30405782826134e-17],[2.96822530176851e-16,- 4.77029898301223e-17,1.12782285532775e-16,1.58443229778573e-18,8.22141904662969e-17]])
        self.bnm_bh = np.array([[0,0,0,0,0],[0,0,0,0,0],[- 2.29210587053658e-06,- 2.33805004374529e-06,- 7.49312880102168e-07,- 5.12022747852006e-07,5.88926055066172e-07],[0,0,0,0,0],[- 4.6338275484369e-06,- 2.23853015662938e-06,8.14830531656518e-07,1.15453269407116e-06,- 4.53555450927571e-07],[- 6.92432096320778e-07,- 2.98734455136141e-07,1.48085153955641e-08,1.37881746148773e-07,- 6.92492118460215e-09],[0,0,0,0,0],[- 1.9150797985031e-06,- 1.83614825459598e-06,- 7.46807436870647e-07,- 1.28329122348007e-06,5.04937180063059e-07],[- 8.07527103916713e-07,2.8399784057457e-08,- 6.01890498063025e-08,- 2.48339507554546e-08,2.46284627824308e-08],[- 2.82995069303093e-07,1.38818274596408e-09,3.22731214161408e-09,2.87731153972404e-10,1.53895537278496e-08],[0,0,0,0,0],[- 6.682102709568e-07,- 2.19104833297845e-06,1.30116691657253e-07,4.7844573043345e-07,- 4.40344300914051e-07],[- 2.36946755740436e-07,- 1.32730991878204e-07,1.8366959369386e-08,7.90218931983569e-08,- 4.70161979232584e-08],[1.07746083292179e-07,- 4.1708863776033e-09,- 1.83296035841109e-09,- 5.80243971371211e-09,- 2.11682361167439e-09],[- 5.44712355496109e-08,1.89717032256923e-09,2.27327316287804e-10,7.78400728280038e-10,8.82380487618991e-12],[0,0,0,0,0],[- 5.61707049615673e-08,- 1.09066447089585e-06,- 2.25742250174119e-07,- 8.64367795924377e-07,1.0641127524068e-08],[2.41782935157918e-08,- 3.65762298303819e-08,- 6.93420659586875e-08,- 3.97316214341991e-08,- 2.0876781648639e-08],[6.38293030383436e-08,1.1137793633447e-08,6.91424941454782e-09,1.39887159955004e-09,5.25428749022906e-09],[1.09291268489958e-08,1.23935926756516e-10,3.92917259954515e-10,- 1.79144682483562e-10,- 9.11802874917597e-10],[- 4.40957607823325e-09,1.45751390560667e-10,1.24641258165301e-10,- 6.45810339804674e-11,- 8.92894658893326e-12],[0,0,0,0,0],[1.54754294162102e-08,- 1.60154742388847e-06,- 4.08425188394881e-07,6.18170290113531e-09,- 2.58919765162122e-07],[1.37130642286873e-08,- 6.67813955828458e-08,- 7.01410996605609e-09,3.82732572660461e-08,- 2.73381870915135e-08],[2.19113155379218e-08,4.11027496396868e-09,6.33816020485226e-09,- 1.49242411327524e-09,- 6.14224941851705e-10],[6.26573021218961e-09,5.17137416480052e-10,- 3.49784328298676e-10,1.13578756343208e-10,2.80414613398411e-10],[1.65048133258794e-11,1.00047239417239e-10,1.05124654878499e-10,- 3.03826002621926e-11,4.57155388334682e-11],[6.20221691418381e-11,9.75852610098156e-12,- 5.46716005756984e-12,1.31643349569537e-11,3.6161877571547e-12],[0,0,0,0,0],[- 1.03938913012708e-06,- 1.78417431315664e-07,2.86040141364439e-07,1.83508599345952e-08,- 1.34452220464346e-07],[- 4.36557481393662e-08,7.49780206868834e-09,- 8.62829428674082e-09,5.50577793039009e-09,- 9.46897502333254e-09],[3.43193738406672e-10,1.13545447306468e-08,1.25242388852214e-09,6.0322150195962e-10,1.5717207036118e-09],[- 4.73307591021391e-10,1.70855824051391e-10,- 2.62470421477037e-11,2.04525835988874e-10,- 1.17859695928164e-10],[- 3.36185995299839e-10,3.19243054562183e-11,1.17589412418126e-10,- 1.35478747434514e-12,5.11192214558542e-11],[3.19640547592136e-11,2.94297823804643e-12,- 1.0065152627699e-11,- 1.67028733953153e-12,3.03938833625503e-12],[1.68928641118173e-11,- 7.90032886682002e-13,- 1.40899773539137e-12,7.76937592393354e-13,7.32539820298651e-13],[0,0,0,0,0],[2.32949756055277e-07,1.46237594908093e-07,- 1.07770884952484e-07,1.26824870644476e-07,- 2.36345735961108e-08],[8.89572676497766e-08,7.24810004121931e-08,2.67583556180119e-08,2.48434796111361e-08,- 3.55004782858686e-09],[- 1.00823909773603e-08,8.84433929029076e-10,- 2.55502517594511e-10,- 5.48034274059119e-10,- 8.50241938494079e-10],[1.13259819566467e-09,5.55186945221216e-10,7.63679807785295e-11,- 1.70067998092043e-11,1.57081965572493e-10],[- 2.37748192185353e-10,2.45463764948e-11,3.2320841480286e-11,- 2.72624834520723e-12,8.144491836665e-12],[- 1.54977633126025e-11,4.58754903157884e-12,- 1.25864665839074e-12,2.44139868157872e-12,- 1.82827441958193e-12],[3.28285563794513e-12,- 1.10072329225465e-12,- 7.23470501810935e-13,5.85309745620389e-13,4.11317589687125e-13],[4.5759697438417e-13,9.84198128213558e-14,3.3450381770283e-14,7.08431086558307e-15,2.79891177268807e-14],[0,0,0,0,0],[- 3.6782071915558e-07,6.98497901205902e-07,1.833973887503e-07,2.39730262495372e-07,- 2.58441984368194e-07],[5.17793954077994e-08,5.54614175977835e-08,1.75026214305232e-09,- 2.55518450411346e-09,- 6.12272723006537e-09],[- 7.94292648157198e-09,- 1.01709107852895e-09,- 1.4925124181231e-09,9.32827213605682e-10,- 8.24490722043118e-10],[1.36410408475679e-11,2.16390220454971e-10,1.24934806872235e-10,- 6.82507825145903e-11,- 4.01575177719668e-11],[- 1.41619917600555e-11,- 1.54733230409082e-11,1.36792829351538e-11,1.11157862104733e-12,2.08548465892268e-11],[- 3.56521723755846e-12,4.47877185884557e-12,- 6.34096209274637e-16,- 1.13010624512348e-12,- 2.82018136861041e-13],[2.22758955943441e-12,- 4.6387646555938e-13,- 5.80688019272507e-13,2.45878690598655e-13,1.49997666808106e-13],[- 6.26833903786958e-14,2.73416335780807e-14,1.91842340758425e-14,1.6740506112901e-14,- 2.45268543953704e-17],[1.81972870222228e-14,5.43036245069085e-15,1.92476637107321e-15,8.78498602508626e-17,- 1.42581647227657e-15],[0,0,0,0,0],[9.74322164613392e-07,- 5.23101820582724e-07,- 2.81997898176227e-07,4.54762451707384e-08,- 3.34645078118827e-08],[- 6.75813194549663e-09,3.49744702199583e-08,- 5.09170419895883e-09,5.24359476874755e-09,4.96664262534662e-09],[4.53858847892396e-10,- 1.49347392165963e-09,- 2.00939511362154e-09,9.30987163387955e-10,9.74450200826854e-11],[- 4.92900885858693e-10,5.34223033225688e-12,1.08501839729368e-10,- 6.43526142089173e-11,- 3.11063319142619e-11],[1.3846924638669e-11,- 7.91180584906922e-12,2.26641656746936e-13,4.55251515177956e-12,6.05270575117769e-12],[4.02247935664225e-12,1.82776657951829e-12,- 1.28348801405445e-13,- 2.1625730130035e-13,- 5.54363979435025e-14],[4.15005914461687e-13,- 2.00647573581168e-13,- 1.67278251942946e-13,1.30332398257985e-13,1.52742363652434e-13],[6.36376500056974e-14,1.65794532815776e-14,- 3.80832559052662e-15,- 6.40262894005341e-16,2.42577181848072e-15],[- 5.55273521249151e-15,3.69725182221479e-15,2.02114207545759e-15,- 4.50870833392161e-16,9.62950493696677e-17],[1.00935904205024e-17,6.54751873609395e-17,- 1.09138810997186e-16,- 8.62396750098759e-17,- 3.82788257844306e-17],[0,0,0,0,0],[4.21958510903678e-07,- 8.30678271007705e-08,- 3.47006439555247e-07,- 3.36442823712421e-08,9.90739768222027e-08],[2.64389033612742e-08,2.65825090066479e-09,- 1.28895513428522e-08,- 7.07182694980098e-10,7.1090716530118e-09],[6.31203524153492e-09,- 1.67038260990134e-09,1.33104703539822e-09,8.34376495185149e-10,- 2.52478613522612e-10],[1.18414896299279e-10,- 2.57745052288455e-11,2.88295935685818e-11,- 3.27782977418354e-11,- 1.05705000036156e-11],[- 4.20826459055091e-12,- 6.97430607432268e-12,- 3.90660545970607e-12,- 3.90449239948755e-13,- 4.60384797517466e-13],[- 9.476683565582e-13,6.53305025354881e-13,2.6324018543496e-13,1.40129115015734e-13,3.85788887132074e-14],[2.23947810407291e-13,7.35262771548253e-15,- 3.83348211931292e-14,4.20376514344176e-14,4.26445836468461e-14],[- 3.88008154470596e-16,2.2856142466775e-15,- 8.73599966653373e-16,2.14321147947665e-15,6.3863182507192e-16],[- 8.62165565535721e-15,1.7974291214981e-15,1.01541125038661e-15,- 7.91027655831866e-17,- 4.0650513282523e-16],[- 2.35355054392189e-16,- 6.13997759731013e-17,- 2.73490528665965e-17,2.63895177155121e-17,- 4.47531057245187e-18],[6.0190970682353e-17,5.35520010856833e-18,- 2.15530106132531e-18,- 2.46778496746231e-18,- 7.09947296442799e-19],[0,0,0,0,0],[- 3.75005956318736e-07,- 5.39872297906819e-07,- 1.19929654883034e-07,4.52771083775007e-08,1.82790552943564e-07],[7.82606642505646e-09,- 1.68890832383153e-08,- 8.45995188378997e-09,1.42958730598502e-09,3.21075754133531e-09],[4.28818421913782e-09,- 1.07501469928219e-09,8.84086350297418e-10,9.74171228764155e-10,8.59877149602304e-12],[1.28983712172521e-10,- 6.96375160373676e-11,- 2.13481436408896e-11,1.33516375568179e-11,- 1.65864626508258e-11],[- 4.48914384622368e-12,9.68953616831263e-13,- 1.61372463422897e-12,- 2.09683563440448e-12,- 1.90096826314068e-12],[- 1.12626619779175e-13,3.34903159106509e-14,- 1.21721528343657e-13,7.46246339290354e-14,3.68424909859186e-13],[5.0829427436779e-14,2.8303615997709e-14,1.48074873486387e-14,- 9.59633528834945e-15,- 1.262310609511e-14],[- 4.01464098583541e-16,1.97047929526674e-15,- 5.29967950447497e-16,- 3.59120406619931e-16,1.69690933982683e-16],[- 1.73919209873841e-15,7.52792462841274e-16,3.65589287101147e-16,- 7.79247612043812e-17,- 8.24599670368999e-17],[- 4.61555616150128e-17,4.94529746019753e-19,- 1.0985815721227e-17,3.95550811124928e-18,3.239723998841e-18],[- 2.27040686655766e-17,- 3.27855689001215e-18,- 3.30649011116861e-19,9.08748546536849e-19,8.92197599890994e-19],[5.67241944733762e-18,3.84449400209976e-19,1.77668058015537e-19,2.00432838283455e-20,- 2.00801461564767e-19]])
        self.bnm_bw = np.array([[0,0,0,0,0],[0,0,0,0,0],[- 9.56715196386889e-06,- 3.6804063302042e-08,1.27846786489883e-07,1.32525487755973e-06,1.53075361125066e-06],[0,0,0,0,0],[- 7.17682617983607e-06,2.89994188119445e-06,- 2.97763578173405e-07,8.95742089134942e-07,3.44416325304006e-07],[- 8.0266113228521e-07,3.66738692077244e-07,- 3.0288096572328e-07,3.54144282036103e-07,- 1.68873066391463e-07],[0,0,0,0,0],[- 2.89640569283461e-06,- 7.83566373343614e-07,- 8.36667214682577e-07,- 7.41891843549121e-07,- 9.23922655636489e-08],[- 1.06144662284862e-06,1.57709930505924e-07,1.04203025714319e-07,1.20783300488461e-07,- 1.38726055821134e-07],[- 4.16549018672265e-07,- 1.35220897698872e-07,- 6.40269964829901e-08,1.63258283210837e-08,- 2.57958025095959e-08],[0,0,0,0,0],[3.52324885892419e-06,- 2.26705543513814e-07,1.53835589488292e-06,- 3.75263061267433e-07,3.69384057396017e-07],[- 2.06569149157664e-07,- 9.36260183227175e-08,- 3.55985284353048e-08,- 9.13671163891094e-08,6.931562565626e-09],[1.32437594740782e-07,4.44349887272663e-08,- 3.38192451721674e-08,- 3.97263855781102e-08,- 1.930878229958e-09],[- 1.29595244818942e-07,- 1.40852985547683e-08,1.4258759293976e-09,7.05779876554001e-09,- 1.00996269264535e-08],[0,0,0,0,0],[4.06960756215938e-06,- 1.97898540226986e-06,7.21905857553588e-08,- 1.19908881538755e-06,- 5.67561861536903e-08],[6.53369660286999e-08,- 2.42818687866392e-07,- 1.66203004559493e-08,- 2.41512414151897e-08,4.45426333411018e-08],[1.44650670663281e-07,8.50666367433859e-09,- 4.61165612004307e-09,4.88527987491045e-09,1.06277326713172e-08],[1.86770937103513e-08,- 6.4419794028893e-10,- 7.60456736846174e-09,- 9.97186468682689e-10,8.73229752697716e-10],[- 1.00206566229113e-08,1.33934372663121e-09,1.4169150343922e-09,8.72352590578753e-10,- 8.04561626629829e-10],[0,0,0,0,0],[3.07161843116618e-06,1.8296208565647e-06,1.87728623016069e-07,7.10611617623261e-07,2.26499092250481e-07],[4.50766403064905e-08,- 1.67752393078256e-07,2.4784472363907e-08,- 3.56484348424869e-09,- 1.56634836636584e-08],[3.7701165188109e-08,- 7.23045828480496e-09,5.22995988863761e-09,- 1.03740320341306e-09,4.57839777217789e-09],[8.09495635883121e-09,- 3.01977244420529e-10,- 2.30104544933093e-09,3.63658580939428e-10,4.39320811714867e-10],[9.37087629961269e-11,1.00780920426635e-09,1.2814053991335e-10,- 6.65795285522138e-12,4.71732796198631e-11],[- 8.88504487069155e-11,- 1.63253810435461e-10,7.22669710644299e-11,5.64715132584527e-11,- 1.08949308197617e-12],[0,0,0,0,0],[- 2.64054293284174e-07,- 2.37611606117256e-06,- 1.83671059706264e-06,- 3.12199354841993e-07,- 1.05598289276114e-07],[7.41706968747147e-08,- 1.64359098062646e-08,- 3.09750224040234e-08,- 9.68640079410317e-09,- 7.90399057863403e-08],[- 1.00254376564271e-08,1.12528248631191e-08,- 2.678415491741e-09,- 2.69481819323647e-09,1.56550607475331e-09],[- 2.18568129350729e-09,6.2642205697745e-10,1.95007291427316e-09,3.14226463591125e-10,- 3.62000388344482e-10],[- 9.30451291747549e-10,5.62175549482704e-11,1.01022849902012e-10,5.18675856498499e-11,5.37561696283235e-11],[5.33151334468794e-11,1.07571307336725e-10,- 1.31714567944652e-11,- 4.17524405900018e-11,- 2.16737797893502e-12],[4.69916869001309e-11,- 4.34516364859583e-12,- 6.61054225868897e-12,- 5.75845818545368e-12,- 2.32180293529175e-12],[0,0,0,0,0],[- 3.50305843086926e-06,1.76085131953403e-06,8.16661224478572e-07,4.09111042640801e-07,- 9.85414469804995e-08],[1.44670876127274e-07,- 1.41331228923029e-08,- 3.06530152369269e-08,- 1.46732098927996e-08,- 2.30660839364244e-08],[- 2.00043052422933e-08,1.72145861031776e-09,2.13714615094209e-09,1.02982676689194e-09,- 1.64945224692217e-10],[1.23552540016991e-09,1.42028470911613e-09,8.79622616627508e-10,- 7.44465600265154e-10,- 7.17124672589442e-11],[- 6.67749524914644e-10,- 5.7772287493405e-11,3.40077806879472e-11,4.2617607654184e-11,8.23189659748212e-11],[- 4.62771648935992e-11,- 7.24005305716782e-13,1.18233730497485e-12,5.18156973532267e-12,- 1.53329687155297e-12],[4.75581699468619e-12,- 3.79782291469732e-12,1.33077109836853e-12,- 1.0242602010712e-12,3.1038501924913e-13],[1.66486090578792e-12,1.08573672403649e-12,1.26268044166279e-13,- 1.23509297742757e-13,- 1.81842007284038e-13],[0,0,0,0,0],[9.93870680202303e-08,- 1.85264736035628e-06,- 5.58942734710854e-07,- 5.5418344831627e-07,- 3.95581289689398e-08],[7.88329069002365e-08,2.04810091451078e-08,3.74588851000076e-09,3.42429296613803e-08,- 2.00840228416712e-08],[- 5.93700447329696e-10,- 6.57499436973459e-10,- 6.90560448220751e-09,3.56586371051089e-09,7.33310245621566e-11],[- 6.38101662363634e-11,4.23668020216529e-10,- 2.43764895979202e-10,- 9.31466610703172e-11,- 3.17491457845975e-10],[1.5094372538247e-11,- 6.11641188685078e-11,- 4.37018785685645e-11,- 2.32871158949602e-11,4.19757251950526e-11],[- 1.18165328825853e-11,- 9.91299557532438e-13,6.40908678055865e-14,2.41049422936434e-12,- 8.20746054454953e-14],[6.01892101914838e-12,- 8.7848712287345e-13,- 1.58887481332294e-12,- 3.13556902469604e-13,5.14523727801645e-14],[- 1.50791729401891e-13,- 1.45234807159695e-13,1.65302377570887e-13,- 5.77094211651483e-15,9.22218953528393e-14],[- 1.85618902787381e-14,5.64333811864051e-14,- 9.9431137794557e-15,- 2.40992156199999e-15,- 2.19196760659665e-14],[0,0,0,0,0],[- 8.16252352075899e-08,1.61725487723444e-06,9.55522506715921e-07,4.02436267433511e-07,- 2.80682052597712e-07],[7.6868479032863e-09,- 5.00940723761353e-09,- 2.43640127974386e-08,- 2.59119930503129e-08,3.35015169182094e-08],[7.97903115186673e-09,3.73803883416618e-09,3.27888334636662e-09,1.37481300578804e-09,- 1.10677168734482e-10],[- 1.67853012769912e-09,- 1.61405252173139e-10,- 1.98841576520056e-10,- 1.46591506832192e-11,9.3571048780466e-11],[4.08807084343221e-11,- 3.74514169689568e-11,- 3.0363849332391e-11,- 5.02332555734577e-12,- 8.03417498408344e-12],[6.48922619024579e-12,1.96166891023817e-12,- 1.96968755122868e-12,- 5.20970156382361e-12,- 1.62656885103402e-12],[1.28603518902875e-12,- 4.88146958435109e-13,- 3.3703488699184e-13,1.37393696103e-14,4.41398325716943e-14],[1.48670014793021e-13,4.41636026364555e-14,2.06210477976005e-14,- 3.4371758358539e-14,- 1.21693704024213e-14],[- 1.67624180330244e-14,6.59317111144238e-15,2.57238525440646e-15,- 3.21568425020512e-17,5.29659568026553e-15],[7.85453466393227e-16,6.91252183915939e-16,- 1.20540764178454e-15,- 3.85803892583301e-16,3.46606994632006e-16],[0,0,0,0,0],[2.86710087625579e-06,- 1.68179842305865e-06,- 8.4830677201687e-07,- 7.08798062479598e-07,- 1.27469453733635e-07],[2.11824305734993e-09,2.02274279084379e-08,1.61862253091554e-08,3.25597167111807e-08,3.40868964045822e-09],[1.21757111431438e-08,1.68405530472906e-09,1.55379338018638e-09,- 3.81467795805531e-10,2.53316405545058e-09],[- 9.98413758659768e-11,5.38382145421318e-10,3.92629628330704e-10,- 1.43067134097778e-10,3.74959329667113e-12],[- 1.57270407028909e-11,- 9.02797202317592e-12,8.4599705988769e-12,4.71474382524218e-12,5.41880986596427e-12],[- 1.20658618702054e-12,7.12940685593433e-13,1.02148613026937e-12,1.63063852348169e-13,1.74048793197708e-13],[3.80559390991789e-13,1.19678271353485e-13,9.72859455604188e-14,5.42642400031729e-14,8.18796710714586e-14],[- 4.69629218656902e-14,5.59889038686206e-15,2.05363292795059e-15,5.38599403288686e-15,- 2.68929559474202e-15],[- 1.88759348081742e-14,5.20975954705924e-15,- 4.43585653096395e-16,5.57436617793556e-16,- 3.95922805817677e-16],[- 9.80871456373282e-16,2.50857658461759e-17,- 1.24253000050963e-16,6.00857065211394e-17,3.537996353115e-18],[2.49370713054872e-16,- 1.49119714269816e-17,- 3.12276052640583e-17,- 2.42001662334001e-17,- 1.69766504318143e-17],[0,0,0,0,0],[- 1.69222102455713e-06,1.64277906173064e-06,5.28855114364096e-07,4.2815985326865e-07,- 1.57362445882665e-07],[1.67656782413678e-08,- 3.77746114074055e-08,- 2.21564555842165e-08,- 3.37071806992217e-08,1.474540087398e-08],[1.06080499491408e-08,3.21990403709678e-09,3.87301757435359e-09,2.92241827834347e-10,- 1.86619473655742e-11],[1.62399669665839e-10,3.51322865845172e-10,2.67086377702958e-11,- 1.31596563625491e-10,3.14164569507034e-11],[- 2.02180016657259e-11,2.03305178342732e-11,6.34969032565839e-12,5.99522296668787e-12,- 4.46275273451008e-12],[- 9.88409290158885e-13,- 1.47692750858224e-13,3.1465555073053e-13,- 2.41857189187879e-13,4.47727504501486e-13],[1.71430777754854e-13,1.73950835042486e-13,5.92323956541558e-14,8.06625710171825e-15,2.33252485755634e-14],[- 1.74184545690134e-15,- 8.18003353124179e-16,- 6.62369006497819e-16,4.16303374396147e-15,7.06513748014024e-15],[- 6.02936238677014e-15,1.89241084885229e-15,1.9909788194427e-17,- 6.9997429069664e-16,- 2.69504942597709e-17],[- 4.65632962602379e-16,3.70281995445114e-18,- 9.04232973763345e-17,2.20847370761932e-17,7.62909453726566e-17],[- 6.25921477907943e-17,- 2.10532795609842e-17,- 1.03808073867183e-17,1.15091380049019e-18,4.66794445408388e-19],[9.39427013576903e-18,9.17044662931859e-19,2.04132745117549e-18,- 1.72364063154625e-19,- 1.18098896532163e-18]])
        self.bnm_ch = np.array([[0,0,0,0,0],[0,0,0,0,0],[3.44092035729033e-05,- 1.21876825440561e-05,- 1.87490665238967e-05,- 2.60980336247863e-05,4.31639313264615e-06],[0,0,0,0,0],[- 2.60125613000133e-05,1.70570295762269e-05,3.08331896996832e-05,1.66256596588688e-05,- 1.07841055501996e-05],[8.74011641844073e-06,- 2.25874169896607e-06,6.50985196673747e-07,1.30424765493752e-06,- 1.85081244549542e-07],[0,0,0,0,0],[3.77496505484964e-05,- 1.08198973553337e-05,- 1.67717574544937e-05,- 3.22476096673598e-05,1.12281888201134e-05],[- 7.68623378647958e-07,- 4.01400837153063e-06,- 2.16390246700835e-06,- 1.76912959937924e-06,- 1.12740084951955e-06],[- 2.37092815818895e-06,- 9.52317223759653e-07,- 2.22722065579131e-07,- 6.2515761977253e-08,1.86582003894639e-08],[0,0,0,0,0],[- 6.10254317785872e-05,- 2.51815503068494e-05,2.01046207874667e-05,7.21107723367308e-06,- 1.30692058660457e-05],[- 9.60655417241537e-06,- 7.31381721742373e-06,- 2.52767927589636e-06,9.09039973214621e-07,- 6.76454911344246e-07],[- 2.25743206384908e-08,2.33058746737575e-07,2.24746779293445e-07,6.78551351968876e-08,1.25076011387284e-07],[- 2.25744112770133e-07,- 1.44429560891636e-07,- 2.96810417448652e-08,- 5.93858519742856e-08,- 2.4321022945542e-08],[0,0,0,0,0],[7.45721015256308e-06,- 3.8139682167641e-05,- 1.41086198468687e-05,- 2.28514517574713e-05,7.28638705683277e-06],[- 5.77517778169692e-06,- 3.93061211403839e-06,- 2.17369763310752e-06,- 1.48060935583664e-07,- 2.74200485662814e-07],[4.52962035878238e-07,9.80990375495214e-07,4.67492045269286e-07,- 8.31032252212116e-09,1.6942602342774e-07],[7.20536791795515e-10,2.75612253452141e-09,2.47772119382536e-09,4.30621825021233e-09,- 2.86498479499428e-08],[- 2.46253956492716e-08,- 3.10300833499669e-09,8.06559148724445e-09,2.98197408430123e-10,6.32503656532846e-09],[0,0,0,0,0],[- 6.01147094179306e-05,- 3.16631758509869e-05,4.1003811510001e-06,3.55215057231403e-07,- 2.23606515237408e-06],[- 2.85937516921923e-06,- 3.6777570661063e-06,- 5.06445540401637e-07,8.21776759711184e-07,- 5.98690271725558e-07],[7.77122595418965e-07,3.60896376754085e-07,3.88610487893381e-07,- 4.39533892679537e-08,- 6.26882227849174e-08],[1.05759993661891e-07,2.58009912408833e-08,- 1.51356049060972e-08,- 1.13335813107412e-09,5.3747085785037e-10],[7.99831506181984e-09,1.67423735327465e-09,2.94736760548677e-09,- 1.56727133704788e-09,8.46186800849124e-10],[3.07727104043851e-09,3.93584215798484e-10,3.86721562770643e-11,1.72181091277391e-10,- 2.16915737920145e-10],[0,0,0,0,0],[- 1.16335389078126e-05,- 1.39864676661484e-05,2.52546278407717e-06,- 8.79152625440188e-06,- 8.97665132187974e-06],[- 3.95874550504316e-06,- 1.1797626252873e-07,7.031899263693e-07,3.38907065351535e-07,- 3.67714052493558e-07],[2.2908244937044e-07,5.72961531093329e-07,4.21969662578894e-08,1.24112958141431e-08,9.56404486571888e-08],[1.44631865298671e-09,6.19368473895584e-09,1.67110424041236e-09,2.57979463602951e-09,- 6.90806907510366e-09],[1.77235802019153e-09,- 8.1438884622897e-10,4.50421956523579e-09,5.67452314909707e-10,2.4761044367556e-09],[4.85932343880617e-10,2.24864117422804e-10,- 2.22534534468511e-10,- 7.96395824973477e-11,3.12587399902493e-12],[- 3.20173937255409e-11,- 1.29872402028088e-11,- 4.24092901203818e-11,2.66570185704416e-11,- 5.25164954403909e-12],[0,0,0,0,0],[- 1.36010179191872e-05,1.77873053642413e-05,4.80988546657119e-06,3.46859608161212e-06,- 1.73247520896541e-06],[2.00020483116258e-06,2.43393064079673e-06,1.21478843695862e-06,1.95582820041644e-07,- 3.11847995109088e-07],[- 8.1328721897931e-09,1.05206830238665e-08,6.54040136224164e-09,- 1.9640266057599e-08,- 1.40379796070732e-08],[4.0129102031074e-08,2.92634301047947e-08,6.04179709273169e-09,8.61849065020545e-10,5.98065429697245e-09],[- 1.06149335032911e-09,- 4.39748495862323e-10,8.83040310269353e-10,3.49392227277679e-10,8.57722299002622e-10],[- 1.2504988890939e-11,2.05203288281631e-10,1.37817670505319e-11,6.82057794430145e-11,- 9.41515631694254e-11],[7.4719602264413e-12,- 2.51369898528782e-11,- 2.121966878092e-11,1.55282119505201e-11,9.99224438231805e-12],[- 7.90534019004874e-13,3.55824506982589e-12,8.00835777767281e-13,8.73460019069655e-13,1.34176126600106e-12],[0,0,0,0,0],[3.12855262465316e-05,1.31629386003608e-05,2.65598119437581e-06,8.68923340949135e-06,- 7.51164082949678e-06],[1.56870792650533e-06,1.8922730168537e-06,4.15620385341985e-07,- 2.74253787880603e-07,- 4.288262101192e-07],[- 9.99176994565587e-08,- 1.10785129426286e-07,- 1.10318125091182e-07,6.22726507350764e-09,- 3.3921456638625e-08],[1.24872975018433e-08,1.10663206077249e-08,5.40658975901469e-09,- 2.79119137105115e-09,- 2.47500096192502e-09],[1.1151891715406e-10,- 4.21965763244849e-10,3.26786005211229e-10,1.93488254914545e-10,7.00774679999972e-10],[1.50889220040757e-10,1.03130002661366e-10,- 3.09481760816903e-11,- 4.47656630703759e-11,- 7.362450218038e-12],[- 1.91144562110285e-12,- 1.11355583995978e-11,- 1.76207323352556e-11,8.15289793192265e-12,3.45078925412654e-12],[- 2.73248710476019e-12,- 1.65089342283056e-13,- 2.20125355220819e-13,5.32589191504356e-13,5.70008982140874e-13],[8.06636928368811e-13,1.30893069976672e-13,9.72079137767479e-14,3.87410156264322e-14,- 5.56410013263563e-14],[0,0,0,0,0],[2.02454485403216e-05,- 9.77720471118669e-06,- 4.35467548126223e-06,2.19599868869063e-06,- 3.2667081904369e-06],[- 3.2183925631054e-08,8.38760368015005e-07,- 5.0805883572406e-07,4.16177282491396e-08,1.5384259276212e-07],[- 1.57377633165313e-07,- 7.86803586842404e-08,- 7.40444711426898e-08,3.15259864117954e-08,5.60536231567172e-09],[- 3.26080428920229e-10,- 3.14576780695439e-09,8.46796096612981e-10,- 2.59329379174262e-09,- 8.01054756588382e-10],[- 4.58725236153576e-11,- 6.87847958546571e-11,8.18226480126754e-12,1.81082075625897e-10,1.74510532938256e-10],[7.60233505328792e-11,4.76463939581321e-11,- 2.47198455442033e-11,- 8.83439688929965e-12,5.93967446277316e-13],[- 8.92919292558887e-12,- 4.38524572312029e-12,- 4.02709146060896e-12,4.84344426425295e-12,5.1286904278152e-12],[1.91518361809952e-12,3.06846255371817e-13,- 2.44830265306345e-13,7.86297493099244e-14,2.7234780580198e-13],[9.09936624159538e-14,7.20650818861447e-15,2.45383991578283e-14,- 4.79580974186462e-15,3.64604724046944e-14],[- 4.63611142770709e-14,1.73908246420636e-15,- 4.41651410674801e-15,- 6.61409045306922e-16,- 1.60016049099639e-15],[0,0,0,0,0],[6.17105245892845e-06,- 1.04342983738457e-05,- 1.72711741097994e-05,- 8.16815967888426e-07,3.42789959967593e-06],[- 2.44014060833825e-07,2.06991837444652e-07,- 3.85805819475679e-07,1.67162359832166e-08,4.15139610402483e-07],[8.1819900680402e-08,- 3.20013409049159e-08,5.94000906771151e-08,2.24122167188946e-08,- 1.33796186160409e-08],[7.66269294674338e-11,- 6.07862178874828e-10,4.95795757186248e-10,- 3.07589245481422e-10,3.44456287710689e-10],[- 1.84076250254929e-10,- 1.30985312312781e-10,- 1.52547325533276e-10,- 2.51000125929512e-11,- 1.93924012590455e-11],[- 2.93307452197665e-11,2.88627386757582e-11,5.58812021182217e-12,- 1.68692874069187e-13,1.80464313900575e-12],[- 9.59053874473003e-13,6.04803122874761e-13,- 9.80015608958536e-13,1.70530372034214e-12,1.70458664160775e-12],[2.80169588226043e-13,9.09573148053551e-14,2.16449186617004e-14,1.15550091496353e-13,4.97772796761321e-14],[- 3.04524400761371e-14,3.42845631349694e-14,2.44230630602064e-14,5.76017546103056e-16,- 9.74409465961093e-15],[5.98765340844291e-15,- 2.63942474859535e-15,- 1.80204805804437e-15,- 1.84981819321183e-16,- 5.8507339216366e-16],[- 2.37069441910133e-15,2.87429226086856e-16,- 1.67055963193389e-16,2.7211068491409e-18,8.46646962667892e-17],[0,0,0,0,0],[- 2.71386164105722e-05,- 1.41834938338454e-05,- 2.00777928859929e-07,5.94329804681196e-07,8.61856994375586e-06],[- 3.93656495458664e-08,- 6.36432821807576e-07,- 2.47887475106438e-07,- 2.64906446204966e-08,1.10689794197004e-07],[5.25319489188562e-08,9.00866357158695e-09,5.00693379572512e-08,2.47269011056404e-08,- 7.27648556194598e-09],[1.87207107149043e-09,- 1.46428282396138e-09,- 2.71812237167257e-10,8.44902265891466e-10,- 5.62683870906027e-10],[- 1.08295119666184e-10,4.75553388543793e-11,- 5.49429386495686e-11,- 6.60907871731611e-11,- 5.97347322824822e-11],[- 4.95118306815571e-12,5.3108373523497e-13,- 1.93679746327378e-12,- 1.6177052184051e-12,1.2327672720251e-11],[6.685826829099e-13,7.38288575160449e-13,5.47630483499201e-13,- 1.00770258118914e-13,- 1.65564928475981e-13],[5.80963409268471e-14,6.93474288078737e-14,6.60728092794315e-15,- 5.21029056725202e-15,- 1.11283532854883e-16],[- 4.10567742688903e-15,1.62252646805882e-14,1.00774699865989e-14,- 2.44793214897877e-16,- 1.59283906414563e-15],[1.84669506619904e-17,8.28473337813919e-17,- 1.53400662078899e-16,- 5.01060672199689e-17,- 2.20727935766132e-16],[2.65355116203636e-16,- 3.70233146147684e-17,3.52689394451586e-18,- 8.62215942516328e-18,9.26909361974526e-18],[9.94266950643135e-17,4.17028699663441e-18,- 7.65153491125819e-21,- 5.62131270981041e-18,- 3.03732817297438e-18]])        
        self.bnm_cw = np.array([[0,0,0,0,0],[0,0,0,0,0],[- 0.000209104872912563,- 1.4153027497354e-05,3.00318745764815e-05,- 1.82864291318284e-05,- 7.62965409959238e-06],[0,0,0,0,0],[- 0.000186336519900275,0.000191256553935638,7.28356195304996e-05,3.59637869639906e-05,- 2.53927226167388e-05],[0.000108195343799485,- 6.97050977217619e-05,- 6.68037133871099e-05,2.30387653190503e-05,- 1.22735483925784e-05],[0,0,0,0,0],[0.000119941091277039,- 7.70547844186875e-05,- 8.15376297964528e-05,1.06005789545203e-05,2.3117723226872e-05],[- 1.77494760217164e-05,- 1.37061385686605e-05,- 1.74805936475816e-05,- 6.91745900867532e-07,- 7.10231790947787e-06],[- 1.47564103733219e-05,2.0889078548526e-06,3.19876879447867e-06,9.43984664503715e-07,- 4.90480527577521e-06],[0,0,0,0,0],[4.93300138389457e-05,- 6.77641298460617e-05,- 3.25043347246397e-05,8.33226714911921e-06,8.11499972792905e-06],[- 2.80449863471272e-05,- 1.04367606414606e-05,1.64473584641163e-07,- 3.57420965807816e-06,2.95887156564038e-06],[1.88835280111533e-06,5.69125761193702e-07,- 2.22757382799409e-06,- 1.96699131032252e-07,- 2.91861219283659e-07],[- 4.6991897143668e-06,- 7.00778948636735e-07,2.97544157334673e-09,3.8610051254441e-07,2.30939653701027e-07],[0,0,0,0,0],[1.77050610394149e-05,- 3.18353071311574e-05,3.04232260950316e-05,- 6.26821316488169e-05,- 1.75094810002378e-06],[9.25605901565775e-06,- 8.25179123302247e-06,6.74032752408358e-06,3.22192289084524e-06,6.09414500075259e-06],[4.282338252422e-06,2.10470570087927e-07,- 4.75050074985668e-07,- 4.89382663470592e-07,8.75232347469207e-07],[8.50393520366934e-07,1.58764911467186e-07,- 2.1626763832121e-07,- 7.43341300487416e-10,1.7513172981323e-07],[- 2.87064111623119e-07,4.5039389310283e-08,6.6331504441669e-08,7.61199387418853e-08,- 6.05694385243652e-09],[0,0,0,0,0],[- 1.95692079507947e-05,5.15486098887851e-05,3.00852761598173e-05,1.21485028343416e-05,- 6.72450521493428e-06],[5.34496867088158e-06,3.90973451680699e-06,3.70148924718425e-06,5.73731499938212e-08,5.5225822028878e-07],[3.39950838185315e-07,- 5.63443976772634e-07,4.52082211980595e-07,- 2.57094645806243e-07,- 6.84885762924729e-08],[2.15793276880684e-07,2.05911354090873e-07,1.33747872341142e-08,- 2.07997626478952e-08,- 3.69812938736019e-08],[2.11952749403224e-09,4.04317822544732e-08,2.4097202488365e-09,8.56289126938059e-09,2.310352834902e-08],[- 2.08402298813248e-09,- 8.50243600879112e-09,2.60895410117768e-09,- 6.69156841738591e-10,- 5.16280278087006e-09],[0,0,0,0,0],[0.000124901291436683,- 5.70770326719086e-05,- 8.44887248105015e-05,- 3.11442665354698e-05,- 1.12982893252046e-05],[- 8.38934444233944e-06,1.56860091415414e-06,- 1.77704563531825e-06,- 5.70219068898717e-08,- 4.30377735031244e-06],[3.72965318017681e-07,6.98175439446187e-07,1.75760544807919e-08,1.59731284857151e-07,3.62363848767891e-07],[- 2.32148850787091e-07,- 4.21888751852973e-08,8.35926113952108e-08,- 2.24572480575674e-08,- 6.92114100904503e-08],[- 2.92635642210745e-09,3.38086229163415e-09,4.72186694662901e-09,- 8.32354437305758e-11,4.19673890995627e-09],[- 1.264528876929e-09,1.91309690886864e-09,1.54755631983655e-09,- 1.09865169400249e-09,1.83645326319994e-10],[9.92539437011905e-10,- 2.963182034883e-10,1.17466020823486e-10,- 5.00185957995526e-10,- 8.54777591408537e-11],[0,0,0,0,0],[- 0.000182885335404854,7.27424724520089e-05,3.05286278023427e-05,2.55324463432562e-05,- 6.39859510763234e-06],[- 5.21449265232557e-06,- 6.70572386081398e-06,- 3.95473351292738e-06,- 6.41023334372861e-07,- 3.11616331059009e-06],[2.37090789071727e-07,3.58427517014705e-07,2.55709192777007e-07,8.44593804408541e-08,9.27243162355359e-09],[7.24370898432057e-08,- 7.4394512033771e-09,8.61751911975683e-10,- 2.34651212610623e-08,2.94052921681456e-09],[- 1.22127317934425e-08,- 3.89758984276768e-09,4.12890383904924e-11,2.06528068002723e-09,1.7348869697227e-09],[- 5.4413740690762e-10,- 4.81034553189921e-10,- 2.56101759039694e-11,3.21880564410154e-10,- 2.7019534316525e-11],[1.08394225300546e-10,- 7.99525492688661e-11,1.73850287030654e-10,- 8.06390014426271e-11,- 7.6314336429116e-13],[- 3.41446959267441e-11,2.72675729042792e-11,5.69674704865345e-12,- 3.38402998344892e-12,- 2.96732381931007e-12],[0,0,0,0,0],[2.9116131598725e-05,- 7.24641166590735e-05,- 8.58323519857884e-06,- 1.1403744425582e-05,1.32244819451517e-05],[1.24266748259826e-06,- 4.13127038469802e-06,- 8.47496394492885e-07,5.48722958754267e-07,- 1.98288551821205e-06],[- 1.70671245196917e-08,1.3689112708354e-08,- 2.8090197224987e-07,- 5.45369793946222e-09,- 9.58796303763498e-08],[1.14115335901746e-08,2.79308166429178e-08,- 1.71144803132413e-08,4.8611624356538e-09,- 8.1306145995228e-09],[- 1.19144311035824e-09,- 1.28197815211763e-09,- 1.22313592972373e-09,6.23116336753674e-10,2.11527825898689e-09],[4.94618645030426e-10,- 1.01554483531252e-10,- 3.58808808952276e-10,1.23499783028794e-10,- 1.21017599361833e-10],[1.33959569836451e-10,- 1.87140898812283e-11,- 3.04265350158941e-11,- 1.42907553051431e-11,- 1.09873858099638e-11],[1.30277419203512e-11,- 4.95312627777245e-12,2.23070215544358e-12,1.66450226016423e-12,6.26222944728474e-12],[- 4.40721204874728e-12,2.99575133064885e-12,- 1.54917262009097e-12,8.9001566452706e-14,- 1.59135267012937e-12],[0,0,0,0,0],[- 4.1766721132316e-05,1.39005215116294e-05,1.46521361817829e-05,3.23485458024416e-05,- 8.57936261085263e-06],[9.4849102652445e-07,1.67749735481991e-06,6.80159475477603e-07,- 1.34558044496631e-06,1.62108231492249e-06],[- 2.67545753355631e-07,- 3.31848493018159e-08,1.05837219557465e-07,1.555876554794e-07,- 2.84996014386667e-08],[- 5.15113778734878e-08,8.83630725241303e-09,3.36579455982772e-09,- 6.22350102096402e-09,5.03959133095369e-09],[2.04635880823035e-11,- 1.07923589059151e-09,- 6.96482137669712e-10,- 4.70238500452793e-10,- 6.60277903598297e-10],[- 2.41897168749189e-11,1.33547763615216e-10,- 5.13534673658908e-11,- 8.32767177662817e-11,5.72614717082428e-11],[7.5517056235994e-12,- 1.57123461699055e-11,- 1.48874069619124e-11,- 7.10529462981252e-13,- 7.99006335025107e-12],[2.4188315673896e-12,2.97346980183361e-12,1.2871997773145e-12,- 2.49240876894143e-12,6.71155595793198e-13],[4.16995565336914e-13,- 1.71584521275288e-13,- 7.23064067359978e-14,2.45405880599037e-13,4.4353293490583e-13],[3.56937508828997e-14,2.430125112603e-14,- 7.96090778289326e-14,- 1.59548529636358e-14,8.99103763000507e-15],[0,0,0,0,0],[0.000117579258399489,- 4.52648448635772e-05,- 2.69130037097862e-05,- 3.82266335794366e-05,- 4.36549257701084e-06],[- 1.43270371215502e-06,1.21565440183855e-06,8.53701136074284e-07,1.52709810023665e-06,1.22382663462904e-06],[3.06089147519664e-07,9.79084123751975e-08,7.96524661441178e-08,4.54770947973458e-08,2.22842369458882e-07],[- 9.94254707745127e-09,1.43251376378012e-08,1.9391175368516e-08,- 6.52214645690987e-09,- 1.97114016452408e-09],[- 9.20751919828404e-10,- 9.44312829629076e-10,7.24196738163952e-11,- 6.71801072324561e-11,2.33146774065873e-10],[- 1.4354429895641e-11,1.78464235318769e-10,7.69950023012326e-11,- 4.22390057304453e-12,3.05176324574816e-11],[- 7.8805375397399e-12,- 3.20207793051003e-12,1.01527407317625e-12,6.02788185858449e-12,1.14919530900453e-11],[- 1.21558899266069e-12,5.31300597882986e-13,3.44023865079264e-13,- 6.22598216726224e-14,- 5.47031650765402e-14],[- 4.15627948750943e-13,2.77620907292721e-13,- 8.99784134364011e-14,1.07254247320864e-13,6.85990080564196e-14],[- 3.91837863922901e-14,9.7471497681618e-15,6.79982450963903e-15,- 2.41420876658572e-15,- 2.20889384455344e-15],[9.25912068402776e-15,- 4.02621719248224e-15,- 2.43952036351187e-15,- 1.97006876049866e-15,1.03065621527869e-16],[0,0,0,0,0],[- 0.000103762036940193,4.38145356960292e-05,2.43406920349913e-05,7.89103527673736e-06,- 1.6684146533916e-05],[- 1.18428449371744e-06,- 1.30188721737259e-06,- 1.8801355711665e-06,- 1.01342046295303e-06,9.21813037802502e-07],[1.5183606871246e-07,1.11362553803933e-07,1.55375052233052e-07,1.94450910788747e-09,- 1.73093755828342e-08],[- 3.77758211813121e-09,1.2332396958361e-08,1.72510045250302e-09,- 1.88609789458597e-09,1.28937597985937e-09],[- 1.07947760393523e-09,5.26051570105365e-10,- 3.67657536332496e-11,3.1611012352384e-10,- 3.2427319824217e-10],[- 2.0038564920982e-12,2.5470386968239e-11,4.08563622440851e-12,- 4.83350348928636e-11,- 3.98153443845079e-13],[2.73094467727215e-12,5.08900664114903e-12,- 7.66669089075134e-13,2.50015592643012e-12,4.29763262853853e-12],[6.5394648753789e-13,- 2.24958413781008e-13,6.74638861781238e-15,3.28537647613903e-14,2.54199700290116e-13],[- 1.09122051193505e-13,8.36362392931501e-14,- 3.907501539123e-14,- 5.4491591074195e-14,2.43816947219217e-14],[- 1.41882561550134e-14,1.00455397812713e-14,2.63347255121581e-15,1.53043256823601e-15,2.49081021428095e-15],[- 1.17256193152654e-15,1.05648985031971e-16,1.31778372453016e-16,1.44815198666577e-16,- 3.7253276861848e-16],[2.66203457773766e-16,- 7.67224608659658e-17,3.51487351031864e-18,4.10287131339291e-17,- 6.72171711728514e-17]])
        if vmf3_type == 'station':
            if station is None:
                raise ValueError('Need to supply a station name to use station-based VMF3')
            self.station = station

    def correction(self, rx_gnsstk, tx_gnsstk, common_time):
        """ get full correction for time, receiver position, transmitter position (matches GNSSTk function signature) """
        ell_model = WGS84Ellipsoid() 
        geo_pos = rx_gnsstk.asGeodetic(ell_model)
        lat = geo_pos[0]
        lon = geo_pos[1]
        ht_el = geo_pos[2]
        MJD = common_time.getDays() - 2400001.0 + GPS2UTC/SECONDS_PER_DAY

        elev = rx_gnsstk.elevationGeodetic(tx_gnsstk)
        azim = rx_gnsstk.azimuthGeodetic(tx_gnsstk)
        e_rad = np.deg2rad(elev)
        a_rad = np.deg2rad(azim)

        if self.vmf3_type == 'V3GR':
            Gn_h, Ge_h, Gn_w, Ge_w = self.interpolate_grad(lat, lon, MJD)
            mfh, mfw, zhd, zwd = self.interpolate_vmf3(lat, lon, ht_el, MJD, elev)
        elif self.vmf3_type == 'station':
            Gn_h, Ge_h, Gn_w, Ge_w = self.interpolate_station_grad(MJD)
            mfh, mfw, zhd, zwd = self.interpolate_station_vmf3(lat, lon, ht_el, MJD, elev)

        self.mfw = mfw # save for dZWD
        self.elev = elev
        self.mfh = mfh
        self.zhd = zhd
        self.zwd = zwd
        C_h = 0.0031 # Chen, Herring (1997)
        C_w = 0.0007 
        mfg_h = 1/(np.sin(e_rad)*np.tan(e_rad)+C_h)
        mfg_w = 1/(np.sin(e_rad)*np.tan(e_rad)+C_w)
        trop_delay = mfh*zhd + mfw*zwd
        trop_delay +=  mfg_h*(Gn_h*np.cos(a_rad)+Ge_h*np.sin(a_rad)) * 1e-3 \
                + mfg_w*(Gn_w*np.cos(a_rad)+Ge_w*np.sin(a_rad)) * 1e-3 # grad vals are in mm

        return trop_delay

    def correction_azel(self, rx_gnsstk, azim, elev, common_time):
        """ get full correction for time, receiver position, transmitter position (matches GNSSTk function signature) """
        ell_model = WGS84Ellipsoid() 
        geo_pos = rx_gnsstk.asGeodetic(ell_model)
        lat = geo_pos[0]
        lon = geo_pos[1]
        ht_el = geo_pos[2]
        MJD = common_time.getDays() - 2400001.0 + GPS2UTC/SECONDS_PER_DAY

        e_rad = np.deg2rad(elev)
        a_rad = np.deg2rad(azim)

        if self.vmf3_type == 'V3GR':
            Gn_h, Ge_h, Gn_w, Ge_w = self.interpolate_grad(lat, lon, MJD)
            mfh, mfw, zhd, zwd = self.interpolate_vmf3(lat, lon, ht_el, MJD, elev)
        elif self.vmf3_type == 'station':
            Gn_h, Ge_h, Gn_w, Ge_w = self.interpolate_station_grad(MJD)
            mfh, mfw, zhd, zwd = self.interpolate_station_vmf3(lat, lon, ht_el, MJD, elev)

        self.mfw = mfw # save for dZWD
        self.elev = elev
        self.mfh = mfh
        self.zhd = zhd
        self.zwd = zwd
        C_h = 0.0031 # Chen, Herring (1997)
        C_w = 0.0007 
        mfg_h = 1/(np.sin(e_rad)*np.tan(e_rad)+C_h)
        mfg_w = 1/(np.sin(e_rad)*np.tan(e_rad)+C_w)
        trop_delay = mfh*zhd + mfw*zwd
        trop_delay +=  mfg_h*(Gn_h*np.cos(a_rad)+Ge_h*np.sin(a_rad)) * 1e-3 \
                + mfg_w*(Gn_w*np.cos(a_rad)+Ge_w*np.sin(a_rad)) * 1e-3 # grad vals are in mm

        return trop_delay

    def wet_mapping_function(self, elevation):
        """ Get the wet mapping function based on last correction() execution """
        if np.isclose(self.elev,elevation):
            return self.mfw
        else:
            raise ValueError('Elevation does not match last correction() elevation')

    def dry_mapping_function(self, elevation):
        """ Get the hydrostatic mapping function based on last correction() execution """
        if np.isclose(self.elev, elevation):
            return self.mfh
        else:
            raise ValueError('Elevation does not match last correction() elevation')

    def dry_zenith_delay(self):
        """ Get the dry zenith delay based on last correction() execution """
        return self.zhd

    def wet_zenith_delay(self):
        """ Get the wet zenith delay based on last correction() execution """
        return self.zwd

    def load_v3gr_files(self, files, orography_file):
        self._epochs_grad = {}
        self._epochs_vmf3 = {}

        # Load orography (meters) as (N,) array
        orography = np.loadtxt(orography_file)
        self.orography = np.asarray(orography, dtype=float)

        for f in files:
            self._load_file(f)

        sample = next(iter(self._epochs_grad.values()))
        self._lat_vals = sample['lat_deg_desc']
        self._lon_vals = sample['lon_deg_asc']

        # infer resolution (just for reference)
        if len(self._lat_vals) > 1:
            # lat is descending: res ~ |lat[i] - lat[i+1]|
            self.grid_res = abs(self._lat_vals[0] - self._lat_vals[1])

        # Precompute grid lat/lon arrays (centered on .5*grid_res)
        lat_all = np.arange(90 - self.grid_res/2, -90 - 1e-9, -self.grid_res)  # N_lat
        lon_all = np.arange(0 + self.grid_res/2, 360 + 1e-9,  self.grid_res)   # N_lon+1 (wrap)
        self.lat_all = lat_all
        self.lon_all = lon_all
        self.nlat = len(lat_all)
        self.nlon = len(lon_all)
        self.N = self.nlat * self.nlon
        if self.grid_res not in (1, 5):
            raise ValueError("grid_res must be 1 or 5 degrees")

        #if self.orography.shape[0] != self.N:
        #    raise ValueError(f"orography length {self.orography.shape[0]} != grid size {self.N}")

    # -------------------------- file loading --------------------------------
    def _load_file(self, file_path):
        path_file = Path(file_path) 
        epoch_mjd = self._mjd_from_filename(path_file.name)
        data = np.loadtxt(file_path, comments='!')
        self._epochs_vmf3[epoch_mjd] = data[:, :6].astype(float, copy=False)

        # Robust ASCII parse: skip comment lines starting with '!'
        lat_list, lon_list = [], []
        ah_list, aw_list, zhd_list, zwd_list = [], [], [], []
        gnh_list, geh_list, gnw_list, gew_list = [], [], [], []
        with path_file.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                s = line.strip()
                if not s or s.startswith("!"):
                    continue
                parts = s.split()
                if len(parts) < 6:
                    continue
                try:
                    la, lo, ah, aw, zhd, zwd, gnh, geh, gnw, gew = map(float, parts[:10])
                except ValueError:
                    continue
                lat_list.append(la)
                lon_list.append(lo)
                gnh_list.append(gnh)
                geh_list.append(geh)
                gnw_list.append(gnw)
                gew_list.append(gew)

        if not lat_list:
            raise ValueError(f"No numeric data parsed from {file_path}")

        lat_arr = np.asarray(lat_list, dtype=float)
        lon_arr = np.asarray(lon_list, dtype=float)
        gnh_arr = np.asarray(gnh_list, dtype=float)
        geh_arr = np.asarray(geh_list, dtype=float)
        gnw_arr = np.asarray(gnw_list, dtype=float)
        gew_arr = np.asarray(gew_list, dtype=float)

        # Build unique coordinate axes.
        # Files use cell centers: lat ~ 90 - res/2 ... -90; lon ~ res/2 ... 360-res/2
        # We'll construct lat descending, lon ascending in [0,360).
        lat_unique = np.unique(lat_arr)
        lon_unique = np.unique(lon_arr)

        # Some files may list lon as 360 instead of 0; normalize to [0,360)
        lon_unique = np.mod(lon_unique, 360.0)
        lon_unique.sort()
        # For lat, enforce descending order:
        lat_unique.sort()
        lat_deg_desc = lat_unique[::-1]  # high to low
        lon_deg_asc = lon_unique

        nlat = lat_deg_desc.size
        nlon = lon_deg_asc.size

        # map each row to (ilat, ilon)
        # prepare indexers
        lat_to_idx = {val: i for i, val in enumerate(lat_deg_desc)}
        lon_to_idx = {val: i for i, val in enumerate(lon_deg_asc)}
        ilat = np.fromiter((lat_to_idx[v] for v in lat_arr), dtype=int)
        ilon = np.fromiter((lon_to_idx[(v % 360.0)] for v in lon_arr), dtype=int)

        # allocate 2D fields
        def gridfill(values: np.ndarray) -> np.ndarray:
            arr = np.full((nlat, nlon), np.nan, dtype=float)
            arr[ilat, ilon] = values
            return arr

        data_grad = {
            'lat_deg_desc': lat_deg_desc,
            'lon_deg_asc': lon_deg_asc,
            'Gn_h': gridfill(gnh_arr),
            'Ge_h': gridfill(geh_arr),
            'Gn_w': gridfill(gnw_arr),
            'Ge_w': gridfill(gew_arr),
        }

        self._epochs_grad[epoch_mjd] = data_grad

    def interpolate_grad(
        self,
        lat_deg: float,
        lon_deg: float,
        time
    ) -> Tuple[float, float, float, float]:
        """
        Return (Gn_h, Ge_h, Gn_w, Ge_w) for given (lat, lon, time).
        - lat, lon: geographic coords (radians if radians=True; else degrees)
        - time: float (MJD) or datetime
        """
        mjd = to_mjd(time)
        e0, e1 = self._epoch_pair(mjd)

        # pull grids
        if e1 is None:
            d0 = self._epochs_grad.get(e0)
            if d0 is None:
                raise KeyError(self._missing_epoch_msg(e0))
            Gn_h, Ge_h, Gn_w, Ge_w = self._bilinear_grad(d0, lat_deg, lon_deg)
            return Gn_h, Ge_h, Gn_w, Ge_w

        d0 = self._epochs_grad.get(e0)
        d1 = self._epochs_grad.get(e1)
        if d0 is None or d1 is None:
            missing = []
            if d0 is None: missing.append(e0)
            if d1 is None: missing.append(e1)
            raise KeyError(self._missing_epoch_msg(*missing))

        # time interpolation
        Gn_h0, Ge_h0, Gn_w0, Ge_w0 = self._bilinear_grad(d0, lat_deg, lon_deg)
        Gn_h1, Ge_h1, Gn_w1, Ge_w1 = self._bilinear_grad(d1, lat_deg, lon_deg)
        w = (mjd - e0) / (e1 - e0)  # in (0,1)

        Gn_h = Gn_h0 + (Gn_h1 - Gn_h0) * w
        Ge_h = Ge_h0 + (Ge_h1 - Ge_h0) * w
        Gn_w = Gn_w0 + (Gn_w1 - Gn_w0) * w
        Ge_w = Ge_w0 + (Ge_w1 - Ge_w0) * w
        return Gn_h, Ge_h, Gn_w, Ge_w

    def _bilinear_grad(self, data: dict, lat_deg: float, lon_deg: float) -> Tuple[float, float, float, float]:
        lat_axis = data['lat_deg_desc']  # descending
        lon_axis = data['lon_deg_asc']   # ascending in [0,360)
        Gn_h = data['Gn_h']; Ge_h = data['Ge_h']; Gn_w = data['Gn_w']; Ge_w = data['Ge_w']

        # Fast paths if on-grid:
        ilat_exact = np.where(np.isclose(lat_axis, lat_deg, atol=1e-10))[0]
        ilon_exact = np.where(np.isclose(lon_axis, lon_deg, atol=1e-10))[0]

        if ilat_exact.size and ilon_exact.size:
            i = ilat_exact[0]; j = ilon_exact[0]
            return Gn_h[i, j], Ge_h[i, j], Gn_w[i, j], Ge_w[i, j]

        # Find bracketing indices for latitude (lat_axis is descending)
        lat_asc = lat_axis[::-1]
        pos = np.searchsorted(lat_asc, lat_deg)  # in ascending axis
        # clamp to interior
        pos = max(1, min(pos, lat_asc.size-1))
        # map back to descending indices
        ilat_lo_desc = (lat_asc.size - pos)      # higher latitude (numerically larger)
        ilat_hi_desc = ilat_lo_desc - 1          # lower latitude
        lat_lo = lat_axis[ilat_lo_desc]
        lat_hi = lat_axis[ilat_hi_desc]
        # fraction in latitude (0..1)
        # note: lat_axis is descending, so lat_hi < lat_lo
        fy = (lat_deg - lat_lo) / (lat_hi - lat_lo) if not np.isclose(lat_lo, lat_hi) else 0.0

        # Find bracketing indices for longitude with wrap
        nlon = lon_axis.size
        j = np.searchsorted(lon_axis, lon_deg)
        j0 = (j - 1) % nlon
        j1 = j % nlon
        lon0 = lon_axis[j0]
        lon1 = lon_axis[j1]
        # handle wrap-around distance
        dlon = (lon1 - lon0) % 360.0
        tlon = (lon_deg - lon0) % 360.0
        fx = (tlon / dlon) if dlon != 0 else 0.0

        # Corner values:
        # lat_lo (ilat_lo_desc), lat_hi (ilat_hi_desc)
        # lon0 (j0), lon1 (j1)
        def interp_field(F):
            f00 = F[ilat_lo_desc, j0]
            f10 = F[ilat_hi_desc, j0]
            f01 = F[ilat_lo_desc, j1]
            f11 = F[ilat_hi_desc, j1]
            # linear along lon at each latitude
            g0 = f00 + (f01 - f00) * fx
            g1 = f10 + (f11 - f10) * fx
            # then along lat
            return g0 + (g1 - g0) * fy

        return (
            interp_field(Gn_h),
            interp_field(Ge_h),
            interp_field(Gn_w),
            interp_field(Ge_w),
        )

    # -------------------------- messaging -----------------------------------

    def _missing_epoch_msg(self, *epochs_mjd: float) -> str:
        want = ", ".join(f"{e:.5f}" for e in epochs_mjd)
        have = ", ".join(f"{e:.5f}" for e in sorted(self._epochs_grad.keys()))
        return (
            f"Required VMF3 GRAD epoch(s) not loaded: {want}. "
            f"Loaded epochs: {have}. "
            f"Hint: provide the GRAD_YYYYMMDD.HHH files for the two surrounding 6h epochs."
        )
    def setWeather(self, T, P, H):
        """ VMF3 uses numerical weather model, nothing to do here """
        return
    def setHumidity(self, H):
        """ VMF3 uses numerical weather model, nothing to do here """
        return

    def interpolate_vmf3(
        self,
        lat_deg,
        lon_deg,
        h_ell,
        time,                # float MJD or numpy.datetime64[*]
        el_deg: float,       # elevation (deg)
    ) -> Tuple[float, float, float, float]:
        """
        Returns (mfh, mfw, zhd, zwd) at station height and given zenith distance.
        """
        mjd = to_mjd(time)
        e0, e1 = self._epoch_pair(mjd)  # e1 may be None if exact
        lat = np.deg2rad(lat_deg)
        lon = np.deg2rad(lon_deg)

        # Gather surrounding 4 indices
        idx4, lat4_deg, lon4_deg = self._surrounding_indices(lat_deg, lon_deg)

        # Extract data at epochs and time-interpolate a_h, a_w, zhd@grid, zwd@grid
        a_h4, a_w4, zhd4_grid, zwd4_grid = self._time_interp_values(e0, e1, mjd, idx4)

        # Lift zenith delays from grid height to station height (Kouba 2008)
        # (a) hydrostatic: convert zhd@grid -> pressure, scale to station, back to zhd
        #     p_grid = zhd_grid / 0.0022768 / (1 - 0.00266 cos(2φ) - 0.28e-6 * H_grid)
        #     p_stat = p_grid * (1 - 2.26e-5 * (h - H_grid))^5.225
        #     zhd_stat = 0.0022768 * p_stat * (1 - 0.00266 cos(2φ) - 0.28e-6 * h)
        cos2phi = np.cos(2.0 * lat)
        H_grid = self.orography[idx4]  # meters
        p_grid = (zhd4_grid / 0.0022768) * (1.0 - 0.00266*cos2phi - 0.28e-6*H_grid)
        p_stat = p_grid * (1.0 - 0.0000226*(h_ell - H_grid))**5.225
        zhd4 = 0.0022768 * p_stat / (1.0 - 0.00266*cos2phi - 0.28e-6*h_ell)

        # (b) wet: simple exponential decay with 2 km scale height
        zwd4 = zwd4_grid * np.exp(-(h_ell - H_grid)/2000.0)

        # Day-of-year with fraction from MJD
        y, m, d = self._ymd_from_mjd(mjd)
        doy = self._mjd_to_doy(mjd)

        # Build V/W “legendre-like” arrays up to degree/order 12 
        nmax = 12
        el = np.deg2rad(el_deg)
        mfh4_h0 = np.zeros(4)
        mfw4_h0 = np.zeros(4)
        x_save = np.zeros(4)
        y_save = np.zeros(4)
        z_save = np.zeros(4)
        for i_grid in range(4):
            polDist = np.pi/2.0 - np.deg2rad(lat4_deg[i_grid])
            x = np.sin(polDist) * np.cos(np.deg2rad(lon4_deg[i_grid]))
            y = np.sin(polDist) * np.sin(np.deg2rad(lon4_deg[i_grid]))
            z = np.cos(polDist)
            V, W = self._compute_VW(x, y, z, nmax=nmax)

            # Extract V_nm/W_nm for the triangular (n,m) stacking order: (0,0),(1,0),(1,1),(2,0)...(12,12)
            pairs = [(n, m) for n in range(nmax + 1) for m in range(n + 1)]
            Vnm = np.array([V[n, m] for (n, m) in pairs])
            Wnm = np.array([W[n, m] for (n, m) in pairs])

            # 3) Evaluate spherical-harmonic sums for the five time columns: A0, A1, B1, A2, B2
            #    self.anm_* / self.bnm_* must be shaped (N, 5) with N = (nmax+1)(nmax+2)/2 = 91.
            def eval_cols(anm: np.ndarray, bnm: np.ndarray) -> np.ndarray:
                anm = np.asarray(anm)
                bnm = np.asarray(bnm)
                # shape: (N,5) * (N,1) -> (N,5) then sum over N -> (5,)
                return np.sum(anm * Vnm[:, None] + bnm * Wnm[:, None], axis=0)

            bh_cols = eval_cols(self.anm_bh, self.bnm_bh)  # -> [A0, A1, B1, A2, B2]
            bw_cols = eval_cols(self.anm_bw, self.bnm_bw)
            ch_cols = eval_cols(self.anm_ch, self.bnm_ch)
            cw_cols = eval_cols(self.anm_cw, self.bnm_cw)

            # 4) Add seasonal terms (annual + semi-annual)
            w1 = 2.0 * np.pi * (doy / 365.25)
            cos1, sin1 = np.cos(w1), np.sin(w1)
            cos2, sin2 = np.cos(2.0 * w1), np.sin(2.0 * w1)

            def assemble(cols: np.ndarray) -> float:
                A0, A1, B1, A2, B2 = cols
                return float(A0 + A1 * cos1 + B1 * sin1 + A2 * cos2 + B2 * sin2)

            b_h = assemble(bh_cols)
            b_w = assemble(bw_cols)
            c_h = assemble(ch_cols)
            c_w = assemble(cw_cols)

            # Mapping functions at ZERO height nodes (VMF3 a_h/a_w are at h=0)
            mfh4_h0[i_grid] = mapping_from_abc(a_h4[i_grid], b_h, c_h, el)
            mfw4_h0[i_grid] = mapping_from_abc(a_w4[i_grid], b_w, c_w, el)

        # height correction for the hydrostatic part [Niell, 1996]
        a_ht = 2.53e-5
        b_ht = 5.49e-3
        c_ht = 1.14e-3
        h_ell_km     = h_ell/1000
        ht_corr_coef = 1/np.sin(el) - (1+(a_ht/(1+b_ht/(1+c_ht)))) / (np.sin(el)+(a_ht/(np.sin(el)+b_ht/(np.sin(el)+c_ht))))
        ht_corr      = ht_corr_coef * h_ell_km
        mfh4 = mfh4_h0 + ht_corr
        mfw4 = mfw4_h0  # wet height dependence is weak; usually neglected for mapping itself

        # Bilinear interpolation in the horizontal for all four outputs
        mfh = self._bilinear_vmf3(lat4_deg, lon4_deg, lat_deg, lon_deg, mfh4)
        mfw = self._bilinear_vmf3(lat4_deg, lon4_deg, lat_deg, lon_deg, mfw4)
        zhd = self._bilinear_vmf3(lat4_deg, lon4_deg, lat_deg, lon_deg, zhd4)
        zwd = self._bilinear_vmf3(lat4_deg, lon4_deg, lat_deg, lon_deg, zwd4)

        return float(mfh), float(mfw), float(zhd), float(zwd)

    def _bilinear_vmf3(
        self,
        lat4: np.ndarray, lon4: np.ndarray,
        lat_deg: float, lon_deg: float,
        vals4: np.ndarray,
    ) -> float:
        # four corners in order: (i1,j1),(i1,j2),(i2,j1),(i2,j2)
        v11, v12, v21, v22 = vals4
        x1, x2 = lon4[0], lon4[1]
        y1, y2 = lat4[0], lat4[2]
        x, y = lon_deg, lat_deg
        # Handle degenerate edges
        if x2 == x1:
            f1 = v11
            f2 = v21
        else:
            f1 = v11 + (v12 - v11) * (x - x1)/(x2 - x1)
            f2 = v21 + (v22 - v21) * (x - x1)/(x2 - x1)
        if y2 == y1:
            return float(f1)
        return float(f1 + (f2 - f1) * (y - y1)/(y2 - y1))


    # --------- internals ---------
    _fname_re = re.compile(r"V3GR_(\d{4})(\d{2})(\d{2})\.H(\d{2})$")
    @classmethod
    def _mjd_from_filename(cls, name: str) -> float:
        m = cls._fname_re.match(name)
        if not m:
            raise ValueError(f"Filename does not match 'VMF3_YYYYMMDD.HHH': {name}")
        y, mo, d, hh = map(int, m.groups())
        # MJD for 00:00 UTC of that date
        mjd0 = _ymd_to_mjd(y, mo, d)
        return mjd0 + (hh / 24.0)

    def _epoch_pair(self, mjd: float) -> Tuple[float, Optional[float]]:
        # 6-hour grid: epochs at ... .00, .25, .50, .75 MJD
        q = 0.25
        if abs((mjd / q) - round(mjd / q)) < 1e-12:
            e0 = round(mjd / q) * q
            return e0, None
        lower = np.floor(mjd / q) * q
        upper = lower + q
        return lower, upper

    def _surrounding_indices(self, lat_deg: float, lon_deg: float):
        # nearest two latitudes
        lat_diff = lat_deg - self.lat_all
        i1 = int(np.argmin(np.abs(lat_diff)))
        i2 = i1 - int(np.sign(lat_diff[i1]))
        i1 = max(0, min(self.nlat-1, i1))
        i2 = max(0, min(self.nlat-1, i2))

        lon_diff = lon_deg - self.lon_all
        j1 = int(np.argmin(np.abs(lon_diff)))
        j2 = j1 + int(np.sign(lon_diff[j1]))
        j1 = (j1 % self.nlon)
        j2 = (j2 % self.nlon)

        idx = np.array([
            i1*self.nlon + j1,
            i1*self.nlon + j2,
            i2*self.nlon + j1,
            i2*self.nlon + j2,
        ], dtype=int)

        lat4 = np.array([self.lat_all[i1], self.lat_all[i1], self.lat_all[i2], self.lat_all[i2]])
        lon4 = np.array([self.lon_all[j1], self.lon_all[j2], self.lon_all[j1], self.lon_all[j2]])
        return idx, lat4, lon4

    def _time_interp_values(self, e0: float, e1: Optional[float], mjd: float, idx4: np.ndarray):
        d0 = self._epochs_vmf3.get(e0)
        if d0 is None:
            raise KeyError(f"Missing VMF3 epoch {e0:.5f}")
        v0 = d0[idx4, 2:6]  # [a_h, a_w, zhd@grid, zwd@grid]

        if e1 is None:
            a_h4, a_w4, zhd4, zwd4 = v0.T
            return a_h4, a_w4, zhd4, zwd4

        d1 = self._epochs_vmf3.get(e1)
        if d1 is None:
            raise KeyError(f"Missing VMF3 epoch {e1:.5f}")
        v1 = d1[idx4, 2:6]

        w = (mjd - e0) / (e1 - e0)
        v = v0 + (v1 - v0) * w
        a_h4, a_w4, zhd4, zwd4 = v.T
        return a_h4, a_w4, zhd4, zwd4

    @staticmethod
    def _ymd_from_mjd(mjd: float) -> Tuple[int, int, int]:
        # Inverse of civilian calendar conversion used in your MATLAB
        jd = mjd + 2400000.5
        j = int(np.floor(jd + 0.5))
        a = j + 32044
        b = (4*a + 3)//146097
        c = a - (b*146097)//4
        d = (4*c + 3)//1461
        e = c - (1461*d)//4
        m = (5*e + 2)//153
        day = e - (153*m + 2)//5 + 1
        month = m + 3 - 12*(m//10)
        year = b*100 + d - 4800 + (m//10)
        return int(year), int(month), int(day)

    @staticmethod
    def _mjd_to_doy(mjd: float) -> float:
        """
        Returns fractional day-of-year in the Gregorian calendar at the site’s UTC.
        """
        mjd_floor = np.floor(mjd)
        frac = mjd - mjd_floor

        # hours, minutes, seconds with carry to avoid 60
        hour = np.floor(frac * 24.0)
        minu = np.floor((frac * 24.0 - hour) * 60.0)
        sec = ((frac * 24.0 - hour) * 60.0 - minu) * 60.0

        if sec >= 60.0 - 1e-12:
            sec = 0.0
            minu += 1
        if minu >= 60:
            minu = 0
            hour += 1

        # JD and integer JD
        jd = mjd + 2400000.5
        if hour >= 24:
            jd += 1.0
            hour = 0
        jd_int = np.floor(jd + 0.5)

        # Gregorian calendar from integer JD (same constants as the MATLAB code)
        aa = jd_int + 32044
        bb = (4 * aa + 3) // 146097
        cc = aa - (bb * 146097) // 4
        dd = (4 * cc + 3) // 1461
        ee = cc - (1461 * dd) // 4
        mm = (5 * ee + 2) // 153

        day = ee - (153 * mm + 2) // 5 + 1
        month = mm + 3 - 12 * (mm // 10)
        year = bb * 100 + dd - 4800 + (mm // 10)

        # leap year
        leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        doy = sum(days_in_month[:int(month - 1)]) + day
        if leap and month > 2:
            doy += 1

        # add the fractional part of the MJD day (matches MATLAB's final line)
        return float(doy + (mjd - mjd_floor))

    @staticmethod
    def _compute_VW(x: float, y: float, z: float, nmax: int = 12) -> tuple[np.ndarray, np.ndarray]:
        """
        Recurrence for real spherical-harmonic V/W arrays following the provided MATLAB code.
        Returns V, W with shape (nmax+1, nmax+1) and zero-based indexing V[n, m].
        """
        V = np.zeros((nmax + 1, nmax + 1), dtype=float)
        W = np.zeros_like(V)

        V[0, 0] = 1.0
        W[0, 0] = 0.0
        V[1, 0] = z * V[0, 0]
        W[1, 0] = 0.0

        for n in range(2, nmax + 1):
            V[n, 0] = ((2 * n - 1) * z * V[n - 1, 0] - (n - 1) * V[n - 2, 0]) / n
            W[n, 0] = 0.0

        for m in range(1, nmax + 1):
            V[m, m] = (2 * m - 1) * (x * V[m - 1, m - 1] - y * W[m - 1, m - 1])
            W[m, m] = (2 * m - 1) * (x * W[m - 1, m - 1] + y * V[m - 1, m - 1])

            if m < nmax:
                V[m + 1, m] = (2 * m + 1) * z * V[m, m]
                W[m + 1, m] = (2 * m + 1) * z * W[m, m]

            for n in range(m + 2, nmax + 1):
                denom = (n - m)
                V[n, m] = ((2 * n - 1) * z * V[n - 1, m] - (n + m - 1) * V[n - 2, m]) / denom
                W[n, m] = ((2 * n - 1) * z * W[n - 1, m] - (n + m - 1) * W[n - 2, m]) / denom

        return V, W

    def load_station_vmf3_files(self, vmf3_files, sta_pos_file):
        """ Load station-based VMF3 zenith delays and mapping function parameters for self.station """
        self.vmf3_epochs = {}
        stations = []
        for vmf3_file in vmf3_files:
            with open(vmf3_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    splt_line = line.split()
                    station = splt_line[0]
                    stations.append(station)
                    epoch = float(splt_line[1])
                    if station == self.station:
                        self.vmf3_epochs[epoch] = np.array(splt_line[2:6], dtype=float)

        if self.station not in stations:
            raise ValueError("Station " + self.station + " not in station file " + sta_pos_file)

        with open(sta_pos_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                splt_line = line.split()
                station = splt_line[0]
                if station == self.station:
                    self.ref_ht = float(splt_line[3])

    def load_station_grad_files(self, grad_files):
        """ Load station-based VMF3 gradient parameters for self.station """
        self.vmf3_grad_epochs = {}
        for grad_file in grad_files:
            with open(grad_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    splt_line = line.split()
                    station = splt_line[0]
                    epoch = float(splt_line[1])
                    if station == self.station:
                        self.vmf3_grad_epochs[epoch] = np.array(splt_line[2:6],dtype=float)

    def interpolate_station_vmf3(self, lat_deg, lon_deg, h_ell, mjd, el_deg):
        """ Find the VMF3 zenith delays and mapping functions at MJD for self.station """
        e1, e2 = self._epoch_pair(mjd)
        vmf3_data_e1 = self.vmf3_epochs[e1]
        if e2 is None:
            vmf3_epoch = vmf3_data_e1
        else:
            vmf3_data_e2 = self.vmf3_epochs[e2]
            w = (mjd - e1) / (e2 - e1)
            vmf3_epoch = vmf3_data_e1 + (vmf3_data_e2 - vmf3_data_e1) * w
        a_h = vmf3_epoch[0]
        a_w = vmf3_epoch[1]
        zhd = vmf3_epoch[2] 
        zwd = vmf3_epoch[3]

        # compute mapping functions
        el = np.deg2rad(el_deg)
        polDist = np.pi/2.0 - np.deg2rad(lat_deg)
        x = np.sin(polDist) * np.cos(np.deg2rad(lon_deg))
        y = np.sin(polDist) * np.sin(np.deg2rad(lon_deg))
        z = np.cos(polDist)
        nmax = 12
        V, W = self._compute_VW(x, y, z, nmax=nmax)

        # Extract V_nm/W_nm for the triangular (n,m) stacking order: (0,0),(1,0),(1,1),(2,0)...(12,12)
        pairs = [(n, m) for n in range(nmax + 1) for m in range(n + 1)]
        Vnm = np.array([V[n, m] for (n, m) in pairs])
        Wnm = np.array([W[n, m] for (n, m) in pairs])

        # Evaluate spherical-harmonic sums for the five time columns: A0, A1, B1, A2, B2
        #    self.anm_* / self.bnm_* must be shaped (N, 5) with N = (nmax+1)(nmax+2)/2 = 91.
        def eval_cols(anm: np.ndarray, bnm: np.ndarray) -> np.ndarray:
            anm = np.asarray(anm)
            bnm = np.asarray(bnm)
            # shape: (N,5) * (N,1) -> (N,5) then sum over N -> (5,)
            return np.sum(anm * Vnm[:, None] + bnm * Wnm[:, None], axis=0)

        bh_cols = eval_cols(self.anm_bh, self.bnm_bh)  # -> [A0, A1, B1, A2, B2]
        bw_cols = eval_cols(self.anm_bw, self.bnm_bw)
        ch_cols = eval_cols(self.anm_ch, self.bnm_ch)
        cw_cols = eval_cols(self.anm_cw, self.bnm_cw)
        # Add seasonal terms (annual + semi-annual)
        doy = self._mjd_to_doy(mjd)
        w1 = 2.0 * np.pi * (doy / 365.25)
        cos1, sin1 = np.cos(w1), np.sin(w1)
        cos2, sin2 = np.cos(2.0 * w1), np.sin(2.0 * w1)

        def assemble(cols: np.ndarray) -> float:
            A0, A1, B1, A2, B2 = cols
            return float(A0 + A1 * cos1 + B1 * sin1 + A2 * cos2 + B2 * sin2)

        b_h = assemble(bh_cols)
        b_w = assemble(bw_cols)
        c_h = assemble(ch_cols)
        c_w = assemble(cw_cols)

        mfh_sta = mapping_from_abc(a_h, b_h, c_h, el)
        mfw = mapping_from_abc(a_w, b_w, c_w, el) # wet height dependence is weak; usually neglected for mapping itself

        # height correction for the hydrostatic part of the mapping function [Niell, 1996]
        a_ht = 2.53e-5
        b_ht = 5.49e-3
        c_ht = 1.14e-3
        h_ell_km     = (h_ell-self.ref_ht)/1000
        ht_corr_coef = 1/np.sin(el) - (1+(a_ht/(1+b_ht/(1+c_ht)))) / (np.sin(el)+(a_ht/(np.sin(el)+b_ht/(np.sin(el)+c_ht))))
        ht_corr      = ht_corr_coef * h_ell_km
        mfh = mfh_sta + ht_corr

        # height correction for the zhd and zwd
        phi = np.deg2rad(lat_deg)
        cos2phi = np.cos(2.0*phi)
        H0 = float(self.ref_ht)  # meters
        # Back out p at ref height from ZHD
        p0 = zhd * (1.0 - 0.00266*cos2phi - 0.28e-6*H0) / 0.0022768
        # Barometric law to station
        p  = p0 * (1.0 - 2.26e-5*(h_ell - H0))**5.225
        zhd = 0.0022768 * p / (1.0 - 0.00266*cos2phi - 0.28e-6*h_ell)
        # Wet scaling (~2 km)
        zwd = zwd * np.exp(-(h_ell - H0)/2000.0)

        return float(mfh), float(mfw), float(zhd), float(zwd)

    def interpolate_station_grad(self, mjd):
        """ Find the VMF3 gradient paramters at MJD for self.station """
        e1, e2 = self._epoch_pair(mjd)
        vmf3_grad_e1 = self.vmf3_grad_epochs[e1]
        if e2 is None:
            vmf3_grad_epoch = vmf3_grad_e1
        else:
            w = (mjd - e1) / (e2 - e1)
            vmf3_grad_e2 = self.vmf3_grad_epochs[e2]
            vmf3_grad_epoch = vmf3_grad_e1 + (vmf3_grad_e2 - vmf3_grad_e1) * w
        Gn_h = vmf3_grad_epoch[0]
        Ge_h = vmf3_grad_epoch[1]
        Gn_w = vmf3_grad_epoch[2]
        Ge_w = vmf3_grad_epoch[3]
        return Gn_h, Ge_h, Gn_w, Ge_w


def mapping_from_abc(a: float, b: float, c: float, el_rad: float) -> float:
    s = np.sin(el_rad)
    return (1.0 + a/(1.0 + b/(1.0 + c))) / (s + a/(s + b/(s + c)))

def _ymd_to_mjd(year: int, month: int, day: int) -> float:
    # Fliegel-Van Flandern-like
    a = (14 - month)//12
    y = year + 4800 - a
    m = month + 12*a - 3
    jdn = day + (153*m + 2)//5 + 365*y + y//4 - y//100 + y//400 - 32045
    jd = float(jdn) - 0.5
    return jd - 2400000.5

def pw_reflection_bwg(antenna_name, t_a, t_t, s_vec, r_a, r_t, u_vec):
    """ Account for the BWG mounted telescope WARK30M """
    # IN NEU
    # s = np.array([np.cos(e)*np.sin(a), np.cos(e)*np.cos(a), np.sin(e)]) 
    # ksi = np.array([np.sin(a), np.cos(a), 0])
    # eta = np.array([-np.cos(a), np.sin(a), 0])
    # u = np.array([0, 0, 1])

    ksi = s_vec - np.dot(u_vec,s_vec)*u_vec
    ksi /= np.linalg.norm(ksi)
    eta = np.cross(u_vec, ksi)
    # new formalism
    S_x = np.array([[0, -s_vec[2], s_vec[1]], [s_vec[2], 0, -s_vec[0]], [-s_vec[1], s_vec[0], 0]])
    P = S_x @ S_x.T
    t = P@t_a + S_x@t_t
    reflect = -1

    # Mirror 3
    k1 = -s_vec
    k2 = -eta 
    n_3 = -k1 + k2
    n_3 = n_3 / np.linalg.norm(n_3)
    q_3 = reflect*(t - 2*np.dot(t,n_3)*n_3)

    # Mirror 4
    if antenna_name == 'WARK30M': 
        k3 = -2500*ksi - 9500*u_vec
        k3 = k3 / np.linalg.norm(k3)
    else:
        k3 = -u_vec

    n_4 = -k2 + k3
    n_4 = n_4 / np.linalg.norm(n_4)
    q_4 = reflect*(q_3 - 2*np.dot(q_3,n_4)*n_4)

    # Mirror 5
    k4 = eta
    n_5 = -k3 + k4
    n_5 = n_5 / np.linalg.norm(n_5)
    q_5 = reflect*(q_4 - 2*np.dot(q_4,n_5)*n_5)

    # Mirror 6
    k5 = -u_vec
    n_6 = -k4 + k5
    n_6 = n_6 / np.linalg.norm(n_6)
    q_6 = reflect*(q_5 - 2*np.dot(q_5,n_6)*n_6)
    kxq = np.cross(k5, q_6)
    pw = np.arctan2(np.dot(r_a, kxq) + np.dot(r_t, q_6), np.dot(r_a, q_6) - np.dot(r_t, kxq))/(2*np.pi)

    return pw


def ECEF2ECI(dt, pos, vel, no_velcorr=False):
    """ Convert an ECEF position and velocity to ECI with a rotation matrix 
    """
    w_earth = 7.2921150e-5
    w_vec = np.array([0,0,w_earth])
    c_w = np.cos(w_earth*dt)
    s_w = np.sin(w_earth*dt)
    ROT = np.array([[c_w, -s_w, 0],\
                    [s_w, c_w,0],\
                    [0, 0, 1]])
    pos_ECI = ROT@pos
    if no_velcorr is True:
        vel_ECI = ROT@vel
    else:
        vel_ECI = ROT@vel - np.cross(w_vec, pos_ECI)

    return pos_ECI, vel_ECI

def slip_detect_MW(freq_1, freq_2, phase1, phase2, range1, range2, times_arr, source_arr, plot=False):
    """
    Use Melbourne-Wubbena combination cycle slip detection if wide-lane available,
    https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/GL017i003p00199
    """
    L_del = (freq_1*phase1-freq_2*phase2)/(freq_1-freq_2) # carrier-phase combination
    P_del = (freq_1*range1+freq_2*range2)/(freq_1+freq_2) # pseudorange combination
    b_del = L_del-P_del
    K_factor = 4 # 4?
    time_limit = 3600 # 1 hr

    #lambda_del = const.c/(freq_1-freq_2)
    #S_Bw = (lambda_del/2)**2  # initialize sigma value

    # assess variance using longest segment of continuous phase
    b_del_init=b_del-np.median(b_del)
    b_del_init=b_del_init[np.abs(b_del_init)<20]
    sigma_phase, _ = find_sigmas(b_del_init)
    S_Bw = sigma_phase**2

    threshold = K_factor*S_Bw
    slips = []
    k=2

    m_Bw_dict = {}
    time_dict = {}
    threshold_dict = {}
    m_Bw_arr = []
    threshold_arr = []
    m_Bw_arr.append(b_del[0])
    threshold_arr.append(threshold)
    for src in np.unique(source_arr): # initialize src dicts
        epochs_src = np.array(source_arr) == src
        m_Bw_candidate = b_del[epochs_src]
        # first value needs to be a non-NaN
        m_Bw_good = m_Bw_candidate[~np.isnan(m_Bw_candidate)] 
        m_Bw_dict[src] = m_Bw_good[0] # initialize m_Bw
        threshold_dict[src] = threshold
        time_dict[src] = times_arr[epochs_src][0] 
    
    for epoch in range(1,len(phase1)):
        src = source_arr[epoch]
        m_Bw = m_Bw_dict[src]
        threshold = threshold_dict[src]
        time = times_arr[epoch]
        time_last = time_dict[src]
        # avoid bad data -- not a slip
        if np.isnan(b_del[epoch]): 
            m_Bw_arr.append(m_Bw)
            threshold_arr.append(threshold)
            continue
        if np.abs(b_del[epoch]-m_Bw)>threshold\
           or time-time_last > time_limit:
            threshold_arr.append(threshold)
            m_Bw_arr.append(m_Bw)

            slips.append(epoch)
            m_Bw = b_del[epoch]
            #S_Bw = (lambda_del/2)**2 
            S_Bw = sigma_phase**2
            threshold = K_factor*np.sqrt(S_Bw)
            k = 2
            m_Bw_dict[src] = m_Bw
            threshold_dict[src] = threshold
            time_dict[src] = time
            continue

        S_Bw = (k-1)*1./k*S_Bw + 1./k*(b_del[epoch]-m_Bw)**2
        m_Bw = (k-1)*1./k*m_Bw + 1./k*b_del[epoch]
        threshold = K_factor*np.sqrt(S_Bw)
        m_Bw_dict[src] = m_Bw
        threshold_dict[src] = threshold
        time_dict[src] = time
        k = k + 1

        threshold_arr.append(threshold)
        m_Bw_arr.append(m_Bw)
    
    if plot is True:
        m_Bw_arr = np.array(m_Bw_arr)
        threshold_arr = np.array(threshold_arr)
        slip_fig = plt.figure()
        slip_ax = slip_fig.add_subplot(111)
        slip_ax.plot(range(len(m_Bw_arr[~np.isnan(b_del)])),m_Bw_arr[~np.isnan(b_del)]-b_del[~np.isnan(b_del)],label='weighted mean')
        slip_ax.plot(range(len(m_Bw_arr[~np.isnan(b_del)])),threshold_arr[~np.isnan(b_del)],label='upper bound')
        slip_ax.plot(range(len(m_Bw_arr[~np.isnan(b_del)])),-threshold_arr[~np.isnan(b_del)],label='lower bound')
        slip_ax.legend()
        slip_ax.set_xlabel('epoch')
        slip_ax.set_ylabel('wide-lane range difference (m)')
        slip_fig.tight_layout()
        slip_fig.savefig('slip_fig.png')
        plt.close(slip_fig)
        breakpoint()

    return slips, m_Bw_arr, threshold_arr

def slip_detect_single_freq(f1, diff_phase_data, diff_phase_model, times_arr, source_array, plot=True):
    """
    Detect cycle slips using single frequency data
    """
    '''
    wavelength1 = const.c/f1
    N_float = construct_float_amb(diff_phase_data, diff_phase_model, wavelength1)

    slips = []
    for src in source_array: # get slips for each source data series
        idxs_src = np.ndarray.flatten(np.argwhere(np.array(source_array) == src))
        N_src = N_float[idxs_src] 
        diff_N = np.diff(N_src,axis=0)
        
        # sudden difference of >1 cycle, + 1 b/c assigning new ambiguity at epoch [slip]
        slips_src = np.ndarray.flatten(np.argwhere(np.abs(diff_N) > 1)) + 1 
        slips.extend(idxs_src[slips_src])
    slips.sort() # sort the list to get slips in order of ascending epoch
    '''
    b_del = diff_phase_data-diff_phase_model
    K_factor = 4 # 4?
    time_limit = 3600 # 1 hr
    b_del_init=b_del-b_del[0]
    b_del_init=b_del_init[np.abs(b_del_init<20)]
    sigma_phase, _ = find_sigmas(b_del_init-np.median(b_del_init))

    S_Bw_0 = sigma_phase**2
    threshold = K_factor*np.sqrt(S_Bw_0)
    if threshold > const.c/(f1*3): 
        S_Bw_0 = (const.c/(f1*3*K_factor))**2
        threshold = const.c/(f1*3)
    S_Bw = S_Bw_0

    slips = []
    k=2

    time_dict = {}
    m_Bw_dict = {}
    threshold_dict = {}
    m_Bw_arr = []
    threshold_arr = []
    m_Bw_arr.append(b_del[0])
    threshold_arr.append(threshold)
    for src in np.unique(source_array): # initialize src dicts
        epochs_src = np.array(source_array) == src
        m_Bw_candidate = b_del[epochs_src]
        
        # first value needs to be a non-NaN
        m_Bw_good = m_Bw_candidate[~np.isnan(m_Bw_candidate)]
        m_Bw_dict[src] = m_Bw_good[0] # initialize m_Bw
        threshold_dict[src] = threshold
        time_dict[src] = times_arr[epochs_src][0] 

    for epoch in range(1,len(b_del)):
        src = source_array[epoch]
        epochs_src = np.array(source_array) == src
        m_Bw = m_Bw_dict[src]
        threshold = threshold_dict[src]
        time = times_arr[epoch]
        time_last = time_dict[src]
        # avoid bad data -- not a slip
        if np.isnan(b_del[epoch]): 
            m_Bw_arr.append(m_Bw)
            threshold_arr.append(threshold)
            continue

        if np.abs(b_del[epoch]-m_Bw)>threshold\
           or time-time_last > time_limit:
            threshold_arr.append(threshold)
            m_Bw_arr.append(m_Bw)
            slips.append(epoch)
            m_Bw = b_del[epoch]
            #S_Bw = (const.c/(f1*2))**2  # initialize sigma value
            #threshold = K_factor*np.sqrt(S_Bw)
            S_Bw = S_Bw_0
            threshold = K_factor*np.sqrt(S_Bw)
            k = 2
            m_Bw_dict[src] = m_Bw
            threshold_dict[src] = threshold
            time_dict[src] = time
            continue

        S_Bw = (k-1)*1./k*S_Bw + 1./k*(b_del[epoch]-m_Bw)**2
        threshold = K_factor*np.sqrt(S_Bw)
        if threshold > const.c/(f1*3): 
            S_Bw = (const.c/(f1*3*K_factor))**2
            threshold = const.c/(f1*3)

        m_Bw = (k-1)*1./k*m_Bw + 1./k*b_del[epoch]
        m_Bw_dict[src] = m_Bw
        threshold_dict[src] = threshold
        time_dict[src] = time
        k = k + 1

        threshold_arr.append(threshold)
        m_Bw_arr.append(m_Bw)

    if plot is True:
        m_Bw_arr = np.array(m_Bw_arr)
        threshold_arr = np.array(threshold_arr)
        slip_fig = plt.figure()
        slip_ax = slip_fig.add_subplot(111)
        slip_ax.plot(range(len(m_Bw_arr[~np.isnan(b_del)])),b_del[~np.isnan(b_del)]-m_Bw_arr[~np.isnan(b_del)],label='$d-\mu$')
        slip_ax.plot(range(len(m_Bw_arr[~np.isnan(b_del)])),threshold_arr[~np.isnan(b_del)],label='upper bound')
        slip_ax.plot(range(len(m_Bw_arr[~np.isnan(b_del)])),-threshold_arr[~np.isnan(b_del)],label='lower bound')
        slip_ax.legend()
        slip_ax.set_xlabel('epoch')
        slip_ax.set_ylabel('single-freq range difference (m)')
        slip_fig.tight_layout()
        slip_fig.savefig('slip_fig_source.png')
        plt.close(slip_fig)

    return slips


def slip_detect_full(f1, diff_phase_data, diff_phase_model, times_arr, plot=False):
    """
    Detect cycle slips using single frequency data, no source differentiation
    """
    b_del = diff_phase_data-diff_phase_model
    K_factor = 4 
    b_del_init=b_del-b_del[0]
    b_del_init=b_del_init[np.abs(b_del_init<20)]
    sigma_phase, _ = find_sigmas(b_del_init-np.median(b_del_init))

    threshold = const.c/(f1*3)

    slips = []
    k=2
    k_max = int(5*60/np.mean(np.diff(times_arr)))
    if k_max < 5: k_max = 5

    m_Bw_arr = []
    threshold_arr = []
    rate_Bw_arr =[]
    m_Bw = b_del[~np.isnan(b_del)][0]
    m_Bw_arr.append(m_Bw)
    threshold_arr.append(threshold)

    for epoch in range(1,len(b_del)):
        time = times_arr[epoch]
        time_last = times_arr[epoch-1]
        # avoid bad data -- not a slip
        if np.isnan(b_del[epoch]): 
            m_Bw_arr.append(m_Bw)
            threshold_arr.append(threshold)
            continue

        if np.abs(b_del[epoch]-m_Bw)>threshold:
            threshold_arr.append(threshold)
            m_Bw_arr.append(m_Bw)
            slips.append(epoch)
            m_Bw = b_del[epoch]
            k = 2
            continue

        threshold_arr.append(threshold)
        m_Bw_arr.append(m_Bw)
        m_Bw = (k-1)*1./k*m_Bw + 1./k*b_del[epoch]
        if k < k_max:
            k = k + 1
        else:
            k = k_max

    if plot is True:
        good_idxs = np.bitwise_and(np.abs(m_Bw_arr)<100, np.abs(b_del)<100)
        m_Bw_arr = np.array(m_Bw_arr)
        threshold_arr = np.array(threshold_arr)
        slip_fig = plt.figure()
        slip_ax = slip_fig.add_subplot(111)
        slip_ax.plot(np.array(range(len(m_Bw_arr[~np.isnan(b_del)])))[good_idxs],b_del[~np.isnan(b_del)][good_idxs]-m_Bw_arr[~np.isnan(b_del)][good_idxs],label='$d-\mu$')
        slip_ax.plot(np.array(range(len(m_Bw_arr[~np.isnan(b_del)])))[good_idxs],threshold_arr[~np.isnan(b_del)][good_idxs],label='upper bound')
        slip_ax.plot(np.array(range(len(m_Bw_arr[~np.isnan(b_del)])))[good_idxs],-threshold_arr[~np.isnan(b_del)][good_idxs],label='lower bound')
        slip_ax.legend()
        slip_ax.set_xlabel('epoch')
        slip_ax.set_ylabel('single-freq range difference (m)')
        slip_fig.tight_layout()
        slip_fig.savefig('slip_fig.png')
        plt.close(slip_fig)

    return slips

def slip_detect_full_old(f1, diff_phase_data, diff_phase_model, times_arr, plot=True):
    """
    Detect cycle slips using single frequency data, no source differentiation
    """
    b_del = diff_phase_data-diff_phase_model
    K_factor = 4 # 4?
    time_limit = 3600 # 1 hr
    b_del_init=b_del-b_del[0]
    b_del_init=b_del_init[np.abs(b_del_init<20)]
    sigma_phase, _ = find_sigmas(b_del_init-np.median(b_del_init))

    S_Bw_0 = sigma_phase**2
    threshold = K_factor*np.sqrt(S_Bw_0)
    if True: #threshold > const.c/(f1*4): 
        S_Bw_0 = (const.c/(f1*3*K_factor))**2
        threshold = const.c/(f1*3)
    S_Bw = S_Bw_0
    rate = np.diff(b_del)/np.diff(times_arr)

    slips = []
    k=2
    k_max = int(5*60/np.mean(np.diff(times_arr)))
    if k_max < 5: k_max = 5
    rate_moving = np.convolve(rate, np.ones(k_max)/k_max, mode='same')

    m_Bw_arr = []
    threshold_arr = []
    rate_Bw_arr =[]
    m_Bw = b_del[~np.isnan(b_del)][0]
    rate_Bw = 0
    m_Bw_arr.append(m_Bw)
    threshold_arr.append(threshold)

    for epoch in range(1,len(b_del)):
        time = times_arr[epoch]
        time_last = times_arr[epoch-1]
        rate_Bw = rate_moving[epoch-1]
        # avoid bad data -- not a slip
        if np.isnan(b_del[epoch]): 
            m_Bw_arr.append(m_Bw)
            threshold_arr.append(threshold)
            continue

        if np.abs(b_del[epoch]-m_Bw)>threshold:# and \
           #np.abs(b_del[epoch]-m_Bw-rate_Bw*(time-time_last))>threshold:
           #or time-time_last > time_limit:
            threshold_arr.append(threshold)
            m_Bw_arr.append(m_Bw)
            slips.append(epoch)
            rate_Bw_arr.append(rate_Bw)
            m_Bw = b_del[epoch]
            rate_Bw = 0
            #S_Bw = S_Bw_0
            #threshold = K_factor*np.sqrt(S_Bw)
            k = 2
            continue

        S_Bw = (k-1)*1./k*S_Bw + 1./k*(b_del[epoch]-m_Bw)**2
        threshold = K_factor*np.sqrt(S_Bw)
        if False: #threshold > const.c/(f1*2): 
            S_Bw = (const.c/(f1*2*K_factor))**2
            threshold = const.c/(f1*2)
        elif True: # threshold < const.c/(f1*4):
            S_Bw = (const.c/(f1*3*K_factor))**2
            threshold = const.c/(f1*3)

        threshold_arr.append(threshold)
        m_Bw_arr.append(m_Bw)
        rate_Bw_arr.append(rate_Bw)
        m_Bw = (k-1)*1./k*m_Bw + 1./k*b_del[epoch]
        if k < k_max:
            k = k + 1
        else:
            k = k_max

    if plot is True:
        good_idxs = np.bitwise_and(np.abs(m_Bw_arr)<100, np.abs(b_del)<100)
        m_Bw_arr = np.array(m_Bw_arr)
        threshold_arr = np.array(threshold_arr)
        slip_fig = plt.figure()
        slip_ax = slip_fig.add_subplot(111)
        slip_ax.plot(np.array(range(len(m_Bw_arr[~np.isnan(b_del)])))[good_idxs],b_del[~np.isnan(b_del)][good_idxs]-m_Bw_arr[~np.isnan(b_del)][good_idxs],label='$d-\mu$')
        slip_ax.plot(np.array(range(len(m_Bw_arr[~np.isnan(b_del)])))[good_idxs],threshold_arr[~np.isnan(b_del)][good_idxs],label='upper bound')
        slip_ax.plot(np.array(range(len(m_Bw_arr[~np.isnan(b_del)])))[good_idxs],-threshold_arr[~np.isnan(b_del)][good_idxs],label='lower bound')
        slip_ax.legend()
        slip_ax.set_xlabel('epoch')
        slip_ax.set_ylabel('single-freq range difference (m)')
        slip_fig.tight_layout()
        slip_fig.savefig('slip_fig.png')
        plt.close(slip_fig)

    return slips




def slip_detect_phase_delay(f1, diff_phase_data, diff_phase_model, times_arr, source_array, plot=True):
    """
    Detect cycle slips using only L1 data (phase delay--dont adjust threshold)
    """
    b_del = diff_phase_data-diff_phase_model
    K_factor = 4
    time_limit = np.inf
    #S_Bw = (const.c/(f1*2))**2
    #threshold = K_factor*S_Bw
    threshold = const.c/(f1*2)
    slips = []
    k=2

    time_dict = {}
    m_Bw_dict = {}
    m_Bw_arr = []
    threshold_arr = []
    m_Bw_arr.append(b_del[0])
    threshold_arr.append(threshold)
    for src in np.unique(source_array): # initialize src dicts
        epochs_src = np.array(source_array) == src
        m_Bw_candidate = b_del[epochs_src]
        
        # first value needs to be a non-NaN
        m_Bw_good = m_Bw_candidate[~np.isnan(m_Bw_candidate)]
        m_Bw_dict[src] = m_Bw_good[0] # initialize m_Bw
        time_dict[src] = times_arr[epochs_src][0] 

    for epoch in range(1,len(b_del)):
        src = source_array[epoch]
        epochs_src = np.array(source_array) == src
        m_Bw = m_Bw_dict[src]
        time = times_arr[epoch]
        time_last = time_dict[src]
        # avoid bad data -- not a slip
        if np.isnan(b_del[epoch]): 
            m_Bw_arr.append(m_Bw)
            threshold_arr.append(threshold)
            continue

        if np.abs(b_del[epoch]-m_Bw)>threshold\
           or time-time_last > time_limit:
            threshold_arr.append(threshold)
            m_Bw_arr.append(m_Bw)
            slips.append(epoch)
            m_Bw = b_del[epoch]
            k = 2
            m_Bw_dict[src] = m_Bw
            time_dict[src] = time
            continue

        m_Bw = (k-1)*1./k*m_Bw + 1./k*b_del[epoch]
        m_Bw_dict[src] = m_Bw
        time_dict[src] = time
        k = k + 1

        threshold_arr.append(threshold)
        m_Bw_arr.append(m_Bw)

    if plot is True:
        m_Bw_arr = np.array(m_Bw_arr)
        threshold_arr = np.array(threshold_arr)
        slip_fig = plt.figure()
        slip_ax = slip_fig.add_subplot(111)
        slip_ax.plot(range(len(m_Bw_arr[~np.isnan(b_del)])),b_del[~np.isnan(b_del)]-m_Bw_arr[~np.isnan(b_del)],label='weighted mean')
        slip_ax.plot(range(len(m_Bw_arr[~np.isnan(b_del)])),threshold_arr[~np.isnan(b_del)],label='upper bound')
        slip_ax.plot(range(len(m_Bw_arr[~np.isnan(b_del)])),-threshold_arr[~np.isnan(b_del)],label='lower bound')
        slip_ax.legend()
        slip_ax.set_xlabel('epoch')
        slip_ax.set_ylabel('single-freq range difference (m)')
        slip_fig.tight_layout()
        slip_fig.savefig('slip_fig.png')
        plt.close(slip_fig)

    return slips

def get_residuals(residuals, baseline_handles, phase_delay=True, phase_only=False, unweighted=False):
    """ Retrieve the separated pseudorange and carrier phase residuals """
    num_samples = 0
    residuals_phase = np.array([])
    residuals_range = np.array([])
    for jdx, baseline_handle in enumerate(baseline_handles): 
        if phase_only is False:
            residuals_range_baseline = residuals[num_samples:\
                                       num_samples+len(baseline_handle.range_data_idxs)]
            num_samples = num_samples + len(baseline_handle.range_data_idxs) 
            residuals_range = np.concatenate((residuals_range, residuals_range_baseline))

        if phase_delay is True:
            residuals_phase_baseline = residuals[num_samples:\
                                    num_samples + len(baseline_handle.phase_data_idxs)]
            num_samples = num_samples + len(baseline_handle.phase_data_idxs) 
            residuals_phase = np.concatenate((residuals_phase, residuals_phase_baseline))
    if phase_delay is True and phase_only is False:
        return residuals_range, residuals_phase
    elif phase_delay is False:
        return residuals_range
    else:
        return residuals_phase

def get_residuals_unweighted(residuals, store_handle, baseline_handles, baselines, antenna_handles, phase_delay=True, phase_only=False):
    """ Retrieve the separated pseudorange and carrier phase residuals without measurement weights """
    num_samples = 0
    residuals_phase = np.array([])
    residuals_range = np.array([])
    for jdx, baseline in enumerate(baselines):
        antenna1_handle = antenna_handles[baseline[0]]
        antenna2_handle = antenna_handles[baseline[1]]     
        baseline_handle = baseline_handles[jdx]
        if phase_only is False:
            residuals_range_baseline = residuals[num_samples:\
                                       num_samples+len(baseline_handle.range_data_idxs)]
            range_weight_mat = get_obs_weights(store_handle, antenna1_handle, antenna2_handle, 'range', \
                    baseline_handle.range_data_idxs, baseline_handle)
            residuals_range_baseline = np.linalg.inv(range_weight_mat)@residuals_range_baseline
            num_samples = num_samples + len(baseline_handle.range_data_idxs) 
            residuals_range = np.concatenate((residuals_range, residuals_range_baseline))

        if phase_delay is True:
            residuals_phase_baseline = residuals[num_samples:\
                                    num_samples + len(baseline_handle.phase_data_idxs)]
            phase_weight_mat = get_obs_weights(store_handle, antenna1_handle, antenna2_handle, 'phase', \
                    baseline_handle.phase_data_idxs, baseline_handle, True)
            residuals_phase_baseline = np.linalg.inv(phase_weight_mat)@residuals_phase_baseline
            num_samples = num_samples + len(baseline_handle.phase_data_idxs) 
            residuals_phase = np.concatenate((residuals_phase, residuals_phase_baseline))
    if phase_delay is True and phase_only is False:
        return residuals_range, residuals_phase
    elif phase_delay is False:
        return residuals_range
    else:
        return residuals_phase

#def get_weights(store_handle, antenna_handles, baselines, baseline_handles=[], phase_delay=True, phase_only=False, use_phase_weights=False):
#    """ Retrieve the observation weights for all baselines"""
#    weights = []
#    for jdx, baseline in enumerate(baselines): # generate differential measurements on the baselines
#        antenna1_handle = antenna_handles[baseline[0]]
#        antenna2_handle = antenna_handles[baseline[1]]      
#        if len(baseline_handles) >0:
#            baseline_handle = baseline_handles[jdx]
#
#        if phase_only is False: # part of function residuals belong to phase residuals
#            if len(baseline_handles) == 0:
#                range_weights = get_obs_weights(store_handle, antenna1_handle, antenna2_handle, 'range')
#            else:
#                range_weights = get_obs_weights(store_handle, antenna1_handle, antenna2_handle, 'range',\
#                            baseline_handle.range_data_idxs, baseline_handle)
#            weights.extend(range_weights)
#        if phase_delay is True:
#            phase_weights = get_obs_weights(store_handle, antenna1_handle, antenna2_handle, 'phase', baseline_handle.phase_data_idxs, baseline_handle, use_phase_weights)
#            weights.extend(phase_weights)
#        
#    return np.array(weights)

def get_unit_var_meas(idx, residuals, final_state, antenna_handles, baselines, baseline_handles, phase_delay, phase_only, phase_clock_idxs):
    """ Get the variance of unit weight for the given antenna from the postfit residuals"""
    num_samples = 0
    if phase_only is False:
        residuals_range_full = np.array([])
    if phase_delay is True:
        residuals_phase_full = np.array([]) 
    # collect all of the residuals that were used in the estimation of this antenna's state variables      
    for jdx, baseline in enumerate(baselines):
        if idx in baseline:
            if phase_delay is False:
                antenna1_handle = antenna_handles[baseline[0]]
                antenna2_handle = antenna_handles[baseline[1]]
                datetime_array = np.intersect1d(antenna1_handle.times_gps, antenna2_handle.times_gps)
                residuals_range = residuals[jdx*len(datetime_array):(jdx+1)*len(datetime_array)]
                residuals_range_full = np.concatenate((residuals_range_full, residuals_range))
            elif phase_only is False:
                baseline_handle = baseline_handles[jdx]
                residuals_range = residuals[num_samples:\
                                         num_samples + len(baseline_handle.range_data_idxs)]
                num_samples = num_samples + len(baseline_handle.range_data_idxs) 
                residuals_phase = residuals[num_samples:\
                                         num_samples + len(baseline_handle.phase_data_idxs)]
                num_samples = num_samples + len(baseline_handle.phase_data_idxs) 
               
                residuals_range_full = np.concatenate((residuals_range_full,residuals_range))
                residuals_phase_full = np.concatenate((residuals_phase_full,residuals_phase))
            else:
                baseline_handle = baseline_handles[jdx]
                residuals_phase = residuals[num_samples:\
                                         num_samples + len(baseline_handle.phase_data_idxs)]
                num_samples = num_samples + len(baseline_handle.phase_data_idxs) 
               
                residuals_phase_full = np.concatenate((residuals_phase_full,residuals_phase))
        else:
            continue
    if phase_delay is False: 
        # Variance estimate of pseudorange
        s_sq_range = np.sum(residuals_range_full**2) / (len(residuals_range_full) - len(final_state))  # Variance estimate
        s_sq = s_sq_range
        s_sq_phase = np.nan
    elif phase_only is False:
        residuals_full = np.concatenate((residuals_range_full, residuals_phase_full))
        # Variance estimate of pseudorange
        s_sq_range = np.sum(residuals_range_full**2) / (len(residuals_range_full) - len(final_state[:phase_clock_idxs.start])+1)  
        # Variance estimate of carrier phase
        s_sq_phase = np.sum(residuals_phase_full**2) / (len(residuals_phase_full) - len(final_state[phase_clock_idxs.start:])+1)
        if s_sq_phase < 0: s_sq_phase = np.sum(residuals_phase_full**2) 
        s_sq = np.sum(residuals_full**2) / (len(residuals_full - len(final_state)))
    else:
        # Variance estimate of carrier phase
        s_sq_phase = np.sum(residuals_phase_full**2) / (len(residuals_phase_full) - len(final_state))  # Variance estimate of carrier phase
        if s_sq_phase < 0: s_sq_phase = np.sum(residuals_phase_full**2) 
        s_sq = s_sq_phase
        s_sq_range = np.nan

    return s_sq, s_sq_range, s_sq_phase

def get_unit_var(idx, residuals, final_state, antenna_handles, baselines, baseline_handles, phase_delay, phase_only, phase_clock_idxs):
    """ Get the variance of unit weight for the given antenna from the postfit residuals"""
    num_samples = 0
    if phase_only is False:
        residuals_range_full = np.array([])
    if phase_delay is True:
        residuals_phase_full = np.array([]) 
    # collect all of the residuals that were used in the estimation of this antenna's state variables      
    for jdx, baseline in enumerate(baselines):
        if idx in baseline:
            if phase_delay is False:
                antenna1_handle = antenna_handles[baseline[0]]
                antenna2_handle = antenna_handles[baseline[1]]
                datetime_array = np.intersect1d(antenna1_handle.times_gps, antenna2_handle.times_gps)
                residuals_range = residuals[jdx*len(datetime_array):(jdx+1)*len(datetime_array)]
                residuals_range_full = np.concatenate((residuals_range_full, residuals_range))
            elif phase_only is False:
                baseline_handle = baseline_handles[jdx]
                residuals_range = residuals[num_samples:\
                                         num_samples + len(baseline_handle.range_data_idxs)]
                num_samples = num_samples + len(baseline_handle.range_data_idxs) 
                residuals_phase = residuals[num_samples:\
                                         num_samples + len(baseline_handle.phase_data_idxs)]
                num_samples = num_samples + len(baseline_handle.phase_data_idxs) 
               
                residuals_range_full = np.concatenate((residuals_range_full,residuals_range))
                residuals_phase_full = np.concatenate((residuals_phase_full,residuals_phase))
            else:
                baseline_handle = baseline_handles[jdx]
                residuals_phase = residuals[num_samples:\
                                         num_samples + len(baseline_handle.phase_data_idxs)]
                num_samples = num_samples + len(baseline_handle.phase_data_idxs) 
               
                residuals_phase_full = np.concatenate((residuals_phase_full,residuals_phase))
        else:
            continue
    if phase_delay is False: 
        # Variance estimate of pseudorange
        s_sq_range = np.sum(residuals_range_full**2) / (len(residuals_range_full) - len(final_state))  # Variance estimate
        s_sq = s_sq_range
        s_sq_phase = np.nan
    elif phase_only is False:
        residuals_full = np.concatenate((residuals_range_full, residuals_phase_full))
        # Variance estimate of pseudorange
        s_sq_range = np.sum(residuals_range_full**2) / (len(residuals_range_full) - len(final_state[:phase_clock_idxs.start])+1)  
        # Variance estimate of carrier phase
        s_sq_phase = np.sum(residuals_phase_full**2) / (len(residuals_phase_full) - len(final_state[phase_clock_idxs.start:])+1)
        if s_sq_phase < 0: s_sq_phase = np.sum(residuals_phase_full**2) 
        s_sq = np.sum(residuals_full**2) / (len(residuals_full - len(final_state)))
    else:
        # Variance estimate of carrier phase
        s_sq_phase = np.sum(residuals_phase_full**2) / (len(residuals_phase_full) - len(final_state))  # Variance estimate of carrier phase
        if s_sq_phase < 0: s_sq_phase = np.sum(residuals_phase_full**2) 
        s_sq = s_sq_phase
        s_sq_range = np.nan

    return s_sq, s_sq_range, s_sq_phase

def get_unit_var_full(residuals, final_state):
    """ Get the variance of unit weight for the given antenna from the postfit residuals"""
    return np.sum(residuals**2) / (len(residuals) - len(final_state))   

def update_cb(antenna_handle, times_gps, clock_samples):
    """ Use existing raw range data but update clock bias correction 
    """
    data = antenna_handle.antenna_data
    clock_samples_prev = antenna_handle.clock_samples
    
    pr_vals = data.pr_model.values[~np.isnan(data.pr_model.values)]
    pr_vals = pr_vals + np.array(clock_samples) - np.array(clock_samples_prev)
    pr_xarray = xr.DataArray(pr_vals, coords={'time': times_gps}, dims='time')
    data = data.assign({'pr_model': pr_xarray})

    return data

def update_cb_phase(antenna_handle, times_gps, phase_clock_samples, iono_free):
    """ Use existing raw range data but update phase clock bias correction 
    """
    data = antenna_handle.antenna_data
    phase_clock_prev = antenna_handle.phase_clock_samples

    cp_vals = data.cp_model.values[~np.isnan(data.cp_model.values)]
    cp_vals = cp_vals + np.array(phase_clock_samples) - np.array(phase_clock_prev)
    cp_xarray = xr.DataArray(cp_vals, coords={'time': times_gps}, dims='time')
    data = data.assign({'cp_model': cp_xarray}) 
    if iono_free is True:
        cp_dual_model_vals = data.cp_dual_model.values[~np.isnan(data.cp_model.values)]
        cp_dual_model_vals = cp_dual_model_vals +\
            np.array(phase_clock_samples) - np.array(phase_clock_prev)
        cp_dual_model_xarray = xr.DataArray(cp_dual_model_vals, coords={'time': times_gps}, dims='time')
        data = data.assign({'cp_dual_model': cp_dual_model_xarray}) 

    return data

def sample_poly_at_interval(poly_state, poly_length, times_gps, epoch0=None):
    """ Use a piecewise linear state variable to sample at each measurement epoch 
        NB: clock state is in m
    """
    if epoch0 is None:
        epoch0 = times_gps[0]
    poly_samples = []
    t_deltas = (times_gps - epoch0)/np.timedelta64(1, 's')
    exp_len = t_deltas[-1]
    for sec_past_epoch in t_deltas:
        # determine the current interval
        n_interval = int(np.floor(sec_past_epoch / poly_length))
        if n_interval < 0: 
            # if sampling before epoch0, need to reset n_interval to avoid clock state indexing issue
            n_interval = 0
        
        if n_interval < len(poly_state)-1:
            # starting clock value of the interval
            begin_value = sum(poly_state[:n_interval+1])
            # starting time of the interval
            interval_start = n_interval*poly_length
            # slope of the clock function at the interval
            slope_interval = poly_state[n_interval+1]/poly_length
        else:
            begin_value = sum(poly_state)
            interval_start = (len(poly_state)-1)*poly_length
            slope_interval = poly_state[-1]/poly_length
 
        clock_sample =  slope_interval*(sec_past_epoch-interval_start) + begin_value
        poly_samples.append(clock_sample)
    
    return np.array(poly_samples)

def poly_jac_at_epoch(poly_state, poly_length, time_gps, epoch0):
    """ Compute the Jacobian of a piecewise linear state vector at a specified epoch 
    """
    analytical_jacobian = np.zeros(len(poly_state))

    # determine the current interval
    sec_past_epoch = (time_gps - epoch0)/np.timedelta64(1, 's')
    n_interval = int(np.floor(sec_past_epoch / poly_length))
    if n_interval < 0: n_interval = 0
    
    if n_interval < len(poly_state)-1:
        # starting clock value of the interval
        interval_start = n_interval*poly_length
        analytical_jacobian[:n_interval+1] = 1
        # linear interpolation
        analytical_jacobian[n_interval+1] = (sec_past_epoch-interval_start)/poly_length
    else:
        interval_start = (len(poly_state)-1)*poly_length
        analytical_jacobian[:-1] = 1
        # linear extrapolation
        analytical_jacobian[-1] = (sec_past_epoch-interval_start)/poly_length
    
    return analytical_jacobian

def sample_global_poly_at_interval(poly_state, times_gps, epoch0=None, epochN=None):
    """ Use a global linear or quadratic state variable to sample at each measurement epoch 
        NB: clock state is in m
    """
    if epoch0 is None:
        epoch0 = times_gps[0]
    if epochN is None:
        epochN = times_gps[-1]
    exp_len = (epochN - epoch0)/np.timedelta64(1, 's')

    t_deltas = (times_gps - epoch0)/np.timedelta64(1, 's')
    poly_samples = t_deltas/exp_len*poly_state[0]
    if len(poly_state)==2:
        poly_samples += t_deltas**2/exp_len**2*poly_state[1]
    
    return poly_samples

def global_poly_jac_at_epoch(poly_state, time_gps, epoch0=None, epochN=None):
    """ Compute the Jacobian of a global linear or quadratic model at a specified epoch 
    """
    if epoch0 is None:
        epoch0 = times_gps[0]
    if epochN is None:
        epochN = times_gps[-1]
    exp_len = (epochN - epoch0)/np.timedelta64(1, 's')
    t_delta = (time_gps - epoch0)/np.timedelta64(1, 's')
    analytical_jacobian = np.ones(len(poly_state))
    analytical_jacobian[0] = t_delta/exp_len
    if len(poly_state) == 2:
        analytical_jacobian[1] = t_delta**2/exp_len**2
    
    return analytical_jacobian

def sample_stoch_params_at_times(vals, prev_times, new_times):
    """ Sample a stochastic model (one value per epoch) at a set of new epochs 
        args:
            prev_times: list of numpy.datetime64 objects
            vals: numpy.ndarray of values at prev_times
            new_times: list of numpy.datetime64 objects
        returns:
            new_vals: numpy.ndarray of values at new_times
    """
    times_sec = (prev_times-prev_times[0])/np.timedelta64(1,'s')
    interp_fcn = interp1d(times_sec, vals, fill_value='extrapolate')

    times_sec_new = (new_times-new_times[0])/np.timedelta64(1,'s')
    new_vals = interp_fcn(times_sec_new)
    return new_vals

def clock_best_fit_line(clock_state, clock_poly_length):
    """ Find the analytical least squares best fit line from the clock state 
    """
    n = len(clock_state)
    clock_times = np.arange(n)*clock_poly_length
    clock_sum = np.sum(clock_times)
    clock_function = np.cumsum(clock_state)
    fcn_sum = np.sum(clock_function)
    slope = (n*np.sum(clock_times*clock_function)-clock_sum*fcn_sum)\
            /(n*np.sum(clock_times**2)-clock_sum**2)
    intercept = (fcn_sum-slope*clock_sum)/n
    return slope, intercept

def clock_best_fit_derivs(clock_state, clock_poly_length):
    """ Find the analytical least squares best fit line from the clock state 
    """
    n = len(clock_state)
    clock_times = np.arange(n)*clock_poly_length
    clock_sum = np.sum(clock_times)
    df_dc = np.tril(np.ones((n,n)))
    df_sum = np.sum(df_dc,axis=0)
    sum_ct_df = np.zeros_like(df_sum)
    for idx, time in enumerate(clock_times):
        sum_ct_df += time*df_dc[idx,:]
    dm_dc = (n*sum_ct_df-clock_sum*df_sum)\
            /(n*np.sum(clock_times**2)-clock_sum**2)
    db_dc = (df_sum-dm_dc*clock_sum)/n
    return dm_dc, db_dc

def get_clock_variation_baseline(ref_antenna, clock_poly_len, antenna1_handle, antenna2_handle, baseline_handle, phase_clock_states):
    """ Get a clock state array for the diff. clock function minus a global linear clock """
    _, ant1_idxs, _ = np.intersect1d(antenna1_handle.times_gps, \
            baseline_handle.datetime_array[baseline_handle.phase_data_idxs], return_indices=True)
    _, ant2_idxs, _ = np.intersect1d(antenna2_handle.times_gps, \
            baseline_handle.datetime_array[baseline_handle.phase_data_idxs], return_indices=True)

    times_sec_ant1 = (antenna1_handle.times_gps[ant1_idxs]-antenna1_handle.times_gps[0])/np.timedelta64(1,'s')
    if antenna1_handle.antenna_name == ref_antenna:
        # get bulk offset -- treat as scalar modifier of differential clock
        res = linregress(times_sec_ant1,antenna1_handle.phase_clock_samples[ant1_idxs])
        clock_bf_ant1 = res.slope*times_sec_ant1 + res.intercept
    else:
        phase_clock_state_ant1 = phase_clock_states[antenna1_handle.phase_clock_idxs]
        m1, b1 = clock_best_fit_line(phase_clock_state_ant1, clock_poly_len)
        clock_bf_ant1 = b1 + m1*times_sec_ant1 

    clock_vals_ant1 = antenna1_handle.phase_clock_samples[ant1_idxs]

    times_sec_ant2 = (antenna2_handle.times_gps[ant2_idxs]-antenna2_handle.times_gps[0])/np.timedelta64(1,'s')
    if antenna2_handle.antenna_name == ref_antenna:
        res = linregress(times_sec_ant2, antenna2_handle.phase_clock_samples[ant2_idxs])
        clock_bf_ant2 = res.slope*times_sec_ant2 + res.intercept
    else:
        phase_clock_state_ant2 = phase_clock_states[antenna2_handle.phase_clock_idxs]
        m2, b2 = clock_best_fit_line(phase_clock_state_ant2, clock_poly_len)
        clock_bf_ant2 = b2 + m2*times_sec_ant2 
    clock_vals_ant2 = antenna2_handle.phase_clock_samples[ant2_idxs]
    
    diff_clock_variation = clock_vals_ant2-clock_bf_ant2 - (clock_vals_ant1-clock_bf_ant1)
    return diff_clock_variation

def get_clock_derivs_baseline(ref_antenna, clock_poly_length, antenna1_handle, antenna2_handle, baseline_handle, phase_clock_states):
    """ Get the clock state partial derivatives for the clock function minus best fit global linear clock """
    _, ant1_idxs, _ = np.intersect1d(antenna1_handle.times_gps, \
            baseline_handle.datetime_array[baseline_handle.phase_data_idxs], return_indices=True)
    _, ant2_idxs, _ = np.intersect1d(antenna2_handle.times_gps, \
            baseline_handle.datetime_array[baseline_handle.phase_data_idxs], return_indices=True)

    dc_bar_dx = np.zeros((len(baseline_handle.datetime_array[baseline_handle.phase_data_idxs]),len(phase_clock_states)))
    times_sec_ant1 = (antenna1_handle.times_gps[ant1_idxs]-antenna1_handle.times_gps[0])/np.timedelta64(1,'s')
    if antenna1_handle.antenna_name != ref_antenna:
        phase_clock_state_ant1 = phase_clock_states[antenna1_handle.phase_clock_idxs]
        dm_dc, db_dc = clock_best_fit_derivs(phase_clock_state_ant1, clock_poly_length)
        for idx, time_gps in enumerate(baseline_handle.datetime_array[baseline_handle.phase_data_idxs]):
            clock_jac_ant1 = poly_jac_at_epoch(phase_clock_state_ant1, clock_poly_length, time_gps, antenna1_handle.phase_clock_start)
            dc_bar_dx[idx, antenna1_handle.phase_clock_idxs] = -(clock_jac_ant1-(times_sec_ant1[idx]*dm_dc + db_dc))

    times_sec_ant2 = (antenna2_handle.times_gps[ant2_idxs]-antenna2_handle.times_gps[0])/np.timedelta64(1,'s')
    if antenna2_handle.antenna_name != ref_antenna:
        phase_clock_state_ant2 = phase_clock_states[antenna2_handle.phase_clock_idxs]
        dm_dc, db_dc = clock_best_fit_derivs(phase_clock_state_ant2, clock_poly_length)
        for idx, time_gps in enumerate(baseline_handle.datetime_array[baseline_handle.phase_data_idxs]):
            clock_jac_ant2 = poly_jac_at_epoch(phase_clock_state_ant2, clock_poly_length, time_gps, antenna2_handle.phase_clock_start)
            dc_bar_dx[idx, antenna2_handle.phase_clock_idxs] = clock_jac_ant2-(times_sec_ant2[idx]*dm_dc + db_dc)

    return dc_bar_dx

def find_diff_meas_phase(antenna1_handle, antenna2_handle, baseline_handle, \
                         amb_state, store_handle, combination_type='WL'):
    """ Calculate the phase measurement residuals for a given baseline """
    data_ant1 = antenna1_handle.antenna_data
    data_ant2 = antenna2_handle.antenna_data
    
    _, ant1_idxs, _ = np.intersect1d(antenna1_handle.times_gps, baseline_handle.datetime_array, return_indices=True)
    _, ant2_idxs, _ = np.intersect1d(antenna2_handle.times_gps, baseline_handle.datetime_array, return_indices=True)
    
    diff_cp_model = data_ant2.cp_model.values[ant2_idxs] - data_ant1.cp_model.values[ant1_idxs]
    diff_cp_data = baseline_handle.cp_diff
   
    # get data and model carrier phase observations
    if store_handle.iono_free is True:
        diff_cp_data = deepcopy(baseline_handle.cp_combination) # deepcopy to avoid changing elements in baseline array
        diff_cp_dual_model = data_ant2.cp_dual_model.values[ant2_idxs] - data_ant1.cp_dual_model.values[ant1_idxs]  
        diff_cp_model = baseline_handle.combination_model(diff_cp_model, diff_cp_dual_model, combination_type)
        wavelength = baseline_handle.comb_wavelength
    else:
        diff_cp_data = deepcopy(baseline_handle.cp_diff)
        wavelength = baseline_handle.wavelength

    # subtract float or integer ambiguities from data
    slip_slices_arr = baseline_handle.slip_slices_arr
    for idx, slip_slice in enumerate(slip_slices_arr):
        diff_cp_data[slip_slice] = diff_cp_data[slip_slice] + wavelength*amb_state[idx]
    
    residuals_phase = diff_cp_data - diff_cp_model
    residuals_phase = residuals_phase[baseline_handle.phase_data_idxs]

    if store_handle.iono_comp_l4r and not store_handle.iono_free:
        source_array = [store_handle.source_time_dict[time] for time in baseline_handle.datetime_array[baseline_handle.phase_data_idxs]]
        stec_vals = store_handle.interp_l4r(baseline_handle.datetime_array[baseline_handle.phase_data_idxs], source_array, antenna1_handle.l4r_name, antenna2_handle.l4r_name)
        residuals_phase -= ALPHA_IONO/baseline_handle.f1**2*const.c*stec_vals

    return residuals_phase

def find_diff_meas_phase_vlbi(baseline_handle, amb_state, combination_type='WL', iono_free=False):
    """ Calculate the phase measurement residuals for a given baseline """
    phase_delay_model = baseline_handle.phase_delay_model
   
    # get data and model carrier phase observations
    if iono_free is True:
        phase_delays = baseline_handle.combination_model(baseline_handle.phase_delays, \
                baseline_hande.phase_delays_dual, combination_type) # deepcopy to avoid changing elements in baseline array
        phase_delay_dual_model = baseline_handle.phase_delay_dual_model  
        phase_delay_model = baseline_handle.combination_model(phase_delay_model, phase_delay_dual_model, combination_type)
        wavelength = baseline_handle.comb_wavelength
    else:
        phase_delays = deepcopy(baseline_handle.phase_delays) # deepcopy to avoid changing elements in baseline array
        wavelength = baseline_handle.wavelength

    # subtract float or integer ambiguities from data
    slip_slices_arr = baseline_handle.slip_slices_arr
    for idx, slip_slice in enumerate(slip_slices_arr):
        phase_delays[slip_slice] = phase_delays[slip_slice] + wavelength*amb_state[idx]

    residuals_phase = phase_delays - phase_delay_model
    residuals_phase = residuals_phase[baseline_handle.phase_data_idxs]

    return residuals_phase

def find_diff_model_phase(antenna1_handle, antenna2_handle, baseline_handle, combination_type='WL', iono_free=False):
    """ Calculate the phase model difference for a given baseline """
    data_ant1 = antenna1_handle.antenna_data
    data_ant2 = antenna2_handle.antenna_data
    
    _, ant1_idxs, _ = np.intersect1d(antenna1_handle.times_gps, baseline_handle.datetime_array, return_indices=True)
    _, ant2_idxs, _ = np.intersect1d(antenna2_handle.times_gps, baseline_handle.datetime_array, return_indices=True)
    
    diff_cp_model = data_ant2.cp_model.values[ant2_idxs] - data_ant1.cp_model.values[ant1_idxs]
   
    # get data and model carrier phase observations
    if iono_free is True:
        diff_cp_data = deepcopy(baseline_handle.cp_combination) # deepcopy to avoid changing elements in baseline array
        diff_cp_dual_model = data_ant2.cp_dual_model.values[ant2_idxs] - data_ant1.cp_dual_model.values[ant1_idxs]  
        diff_cp_model = baseline_handle.combination_model(diff_cp_model, diff_cp_dual_model, combination_type)
        wavelength = baseline_handle.comb_wavelength
    else:
        diff_cp_data = deepcopy(baseline_handle.cp_diff)
        wavelength = baseline_handle.wavelength

    return diff_cp_model


def find_cont_penalty(cont_penalty, ref_antenna, clock_poly_length, phase_clock_states, store_handle, antenna_handles, \
        baselines, baseline_handles, full_amb, use_phase_weights, combination_type):
    """ Find the continuity penalty residuals on all baselines for given phase and phase model"""
    n_amb = 0 
    residuals_diff_full = np.array([])
    for jdx, baseline in enumerate(baselines): # generate differential measurements on the baselines
        antenna1_handle = antenna_handles[baseline[0]]
        antenna2_handle = antenna_handles[baseline[1]]
        baseline_handle = baseline_handles[jdx]
        full_amb_baseline = full_amb[n_amb:n_amb+baseline_handle.n_amb_state]
        n_amb += baseline_handle.n_amb_state 
        if store_handle.sol_type == 'VLBI':
            residuals_baseline_phase = find_diff_meas_phase_vlbi(baseline_handle,\
                full_amb_baseline, store_handle, combination_type)
        elif store_handle.sol_type == 'GNSS':
            residuals_baseline_phase = find_diff_meas_phase(antenna1_handle, antenna2_handle, baseline_handle,\
                full_amb_baseline, store_handle, combination_type)
        phase_weight_mat = get_obs_weights(store_handle, antenna1_handle, antenna2_handle, 'phase', \
                baseline_handle.phase_data_idxs, baseline_handle, use_phase_weights)
        avg_weight = np.mean(np.diag(phase_weight_mat))
        phase_clock_variation = get_clock_variation_baseline(ref_antenna, clock_poly_length, \
                antenna1_handle, antenna2_handle, baseline_handle, phase_clock_states)
        residuals_diff = np.sqrt(cont_penalty) * avg_weight * np.diff(residuals_baseline_phase+phase_clock_variation)
        residuals_diff_full = np.concatenate((residuals_diff_full, residuals_diff))
     
    return residuals_diff_full

#def find_closure_model(baseline1_handle, baseline2_handle, baseline3_handle, combination_type='WL', iono_free=False):
#    """ Enforce phase closure explicitly for a 3-antenna combination """
#    phase_delay_model1 = baseline1_handle.phase_delay_model
#    phase_delay_model2 = baseline2_handle.phase_delay_model
#    phase_delay_model3 = baseline3_handle.phase_delay_model
#   
#    # get data and model carrier phase observations
#    if iono_free is True:
#        phase_delay_dual_model1 = baseline1_handle.phase_delay_dual_model  
#        phase_delay_dual_model2 = baseline2_handle.phase_delay_dual_model  
#        phase_delay_dual_model3 = baseline3_handle.phase_delay_dual_model  
#        phase_delay_model1 = baseline_handle.combination_model(phase_delay_model1, phase_delay_dual_model1, combination_type)
#        phase_delay_model2 = baseline_handle.combination_model(phase_delay_model2, phase_delay_dual_model2, combination_type)
#        phase_delay_model3 = baseline_handle.combination_model(phase_delay_model3, phase_delay_dual_model3, combination_type)
#
#    # find common epochs
#    dt_array = np.intersect1d(baseline1_handle.datetime_array[baseline1_handle.phase_data_idxs],\
#            baseline2_handle.datetime_array[baseline2_handle.phase_data_idxs])
#    dt_array = np.intersect1d(dt_array, baseline3_handle.datetime_array[baseline3_handle.phase_data_idxs])
#    _, idxs_bl1, _ = np.intersect1d(baseline1_handle.datetime_array[baseline1_handle.phase_data_idxs],\
#            dt_array, return_indices=True)
#    _, idxs_bl2, _ = np.intersect1d(baseline2_handle.datetime_array[baseline2_handle.phase_data_idxs],\
#            dt_array, return_indices=True)
#    _, idxs_bl3, _ = np.intersect1d(baseline3_handle.datetime_array[baseline3_handle.phase_data_idxs],\
#            dt_array, return_indices=True)
#    closure_residuals = phase_delay_model1[idxs_bl1] - phase_delay_model2[idxs_bl2] + phase_delay_model3[idxs_bl3]
#    
#    return closure_residuals

def find_closure_model(sol_type, baseline1_handle, baseline2_handle, baseline3_handle, antenna_handles, baseline_idxs, baselines, combination_type='WL', iono_free=False):
    """ Find phase closure of model for a 3-antenna combination """
    if sol_type == 'VLBI':
        phase_delay1 = baseline1_handle.phase_delay_model
        phase_delay2 = baseline2_handle.phase_delay_model
        phase_delay3 = baseline3_handle.phase_delay_model
   
        # get data and model carrier phase observations
        if iono_free is True:
            phase_delay_dual1 = baseline1_handle.phase_delay_dual_model
            phase_delay_dual2 = baseline2_handle.phase_delay_dual_model 
            phase_delay_dual3 = baseline3_handle.phase_delay_dual_model
            phase_delay1 = baseline1_handle.combination_model(baseline1_handle.phase_delay_model, \
                    baseline1_handle.phase_delay_dual_model, combination_type)
            phase_delay2 = baseline2_handle.combination_model(baseline2_handle.phase_delay_model, \
                    baseline2_handle.phase_delay_dual_model, combination_type)
            phase_delay3 = baseline3_handle.combination_model(baseline3_handle.phase_delay_model, \
                    baseline3_handle.phase_delay_dual_model, combination_type)
    elif sol_type == 'GNSS':
        antenna1_handle = antenna_handles[baselines[baseline_idxs[0]][0]]
        antenna2_handle = antenna_handles[baselines[baseline_idxs[0]][1]]
        phase_delay1 = find_diff_model_phase(antenna1_handle, antenna2_handle, baseline1_handle, combination_type, iono_free)
        antenna3_handle = antenna_handles[baselines[baseline_idxs[1]][0]]
        antenna4_handle = antenna_handles[baselines[baseline_idxs[1]][1]]
        phase_delay2 = find_diff_model_phase(antenna3_handle, antenna4_handle, baseline2_handle, combination_type, iono_free)
        antenna5_handle = antenna_handles[baselines[baseline_idxs[2]][0]]
        antenna6_handle = antenna_handles[baselines[baseline_idxs[2]][1]]
        phase_delay3 = find_diff_model_phase(antenna5_handle, antenna6_handle, baseline3_handle, combination_type, iono_free)

    # find common epochs
    dt_array = np.intersect1d(baseline1_handle.datetime_array[baseline1_handle.phase_data_idxs],\
            baseline2_handle.datetime_array[baseline2_handle.phase_data_idxs])
    dt_array = np.intersect1d(dt_array, baseline3_handle.datetime_array[baseline3_handle.phase_data_idxs])
    _, idxs_bl1, _ = np.intersect1d(baseline1_handle.datetime_array,\
            dt_array, return_indices=True)
    _, idxs_bl2, _ = np.intersect1d(baseline2_handle.datetime_array,\
            dt_array, return_indices=True)
    _, idxs_bl3, _ = np.intersect1d(baseline3_handle.datetime_array,\
            dt_array, return_indices=True)

    closure_sum = np.zeros(4)
    closure_sum[0] = np.sum(phase_delay1[idxs_bl1] + phase_delay2[idxs_bl2] + phase_delay3[idxs_bl3])
    closure_sum[1] = np.sum(-phase_delay1[idxs_bl1] + phase_delay2[idxs_bl2] + phase_delay3[idxs_bl3])
    closure_sum[2] = np.sum(phase_delay1[idxs_bl1] - phase_delay2[idxs_bl2] + phase_delay3[idxs_bl3])
    closure_sum[3] = np.sum(phase_delay1[idxs_bl1] + phase_delay2[idxs_bl2] - phase_delay3[idxs_bl3])
    closure_idx = np.argmin(np.abs(closure_sum))
    if closure_idx == 0:
        closure_residuals = phase_delay1[idxs_bl1] + phase_delay2[idxs_bl2] + phase_delay3[idxs_bl3]
    elif closure_idx == 1:
        closure_residuals = -phase_delay1[idxs_bl1] + phase_delay2[idxs_bl2] + phase_delay3[idxs_bl3]
    elif closure_idx == 2:
        closure_residuals = phase_delay1[idxs_bl1] - phase_delay2[idxs_bl2] + phase_delay3[idxs_bl3]
    else:
        closure_residuals = phase_delay1[idxs_bl1] + phase_delay2[idxs_bl2] - phase_delay3[idxs_bl3]
    
    return dt_array, closure_residuals


def find_closure_meas(sol_type, baseline1_handle, baseline2_handle, baseline3_handle, integer_amb1, integer_amb2, integer_amb3, combination_type='WL', iono_free=False):
    """ Find phase closure measurements for a 3-antenna combination """
    if sol_type == 'VLBI':
        phase_delay1 = baseline1_handle.phase_delays.copy()
        phase_delay2 = baseline2_handle.phase_delays.copy()
        phase_delay3 = baseline3_handle.phase_delays.copy()
   
        # get data and model carrier phase observations
        if iono_free is True:
            phase_delay_dual1 = baseline1_handle.phase_delay_dual 
            phase_delay_dual2 = baseline2_handle.phase_delay_dual 
            phase_delay_dual3 = baseline3_handle.phase_delay_dual  
            phase_delay1 = baseline1_handle.combination_model(baseline1_handle.phase_delays, \
                    baseline1_handle.phase_delays_dual, combination_type)
            phase_delay2 = baseline2_handle.combination_model(baseline2_handle.phase_delays, \
                    baseline2_handle.phase_delays_dual, combination_type)
            phase_delay3 = baseline3_handle.combination_model(baseline3_handle.phase_delays, \
                    baseline3_handle.phase_delays_dual, combination_type)
    elif sol_type == 'GNSS':
        phase_delay1 = baseline1_handle.cp_diff.copy()
        phase_delay2 = baseline2_handle.cp_diff.copy()
        phase_delay3 = baseline3_handle.cp_diff.copy()
        if iono_free is True:
            phase_delay1 = baseline1_handle.cp_combination
            phase_delay2 = baseline2_handle.cp_combination
            phase_delay3 = baseline3_handle.cp_combination

    # adjust for integer fix
    wavelength = baseline1_handle.wavelength
    for idx, slip_slice in enumerate(baseline1_handle.slip_slices_arr):
        phase_delay1[slip_slice] += wavelength*integer_amb1[idx]
    for idx, slip_slice in enumerate(baseline2_handle.slip_slices_arr):
        phase_delay2[slip_slice] += wavelength*integer_amb2[idx]
    for idx, slip_slice in enumerate(baseline3_handle.slip_slices_arr):
        phase_delay3[slip_slice] += wavelength*integer_amb3[idx]

    # find common epochs
    dt_array = np.intersect1d(baseline1_handle.datetime_array[baseline1_handle.phase_data_idxs],\
            baseline2_handle.datetime_array[baseline2_handle.phase_data_idxs])
    dt_array = np.intersect1d(dt_array, baseline3_handle.datetime_array[baseline3_handle.phase_data_idxs])
    _, idxs_bl1, _ = np.intersect1d(baseline1_handle.datetime_array,\
            dt_array, return_indices=True)
    _, idxs_bl2, _ = np.intersect1d(baseline2_handle.datetime_array,\
            dt_array, return_indices=True)
    _, idxs_bl3, _ = np.intersect1d(baseline3_handle.datetime_array,\
            dt_array, return_indices=True)

    closure_sum = np.zeros(4)
    closure_sum[0] = np.sum(phase_delay1[idxs_bl1] + phase_delay2[idxs_bl2] + phase_delay3[idxs_bl3])
    closure_sum[1] = np.sum(-phase_delay1[idxs_bl1] + phase_delay2[idxs_bl2] + phase_delay3[idxs_bl3])
    closure_sum[2] = np.sum(phase_delay1[idxs_bl1] - phase_delay2[idxs_bl2] + phase_delay3[idxs_bl3])
    closure_sum[3] = np.sum(phase_delay1[idxs_bl1] + phase_delay2[idxs_bl2] - phase_delay3[idxs_bl3])
    closure_idx = np.argmin(np.abs(closure_sum))
    if closure_idx == 0:
        closure_residuals = phase_delay1[idxs_bl1] + phase_delay2[idxs_bl2] + phase_delay3[idxs_bl3]
    elif closure_idx == 1:
        closure_residuals = -phase_delay1[idxs_bl1] + phase_delay2[idxs_bl2] + phase_delay3[idxs_bl3]
    elif closure_idx == 2:
        closure_residuals = phase_delay1[idxs_bl1] - phase_delay2[idxs_bl2] + phase_delay3[idxs_bl3]
    else:
        closure_residuals = phase_delay1[idxs_bl1] + phase_delay2[idxs_bl2] - phase_delay3[idxs_bl3]
    
    return dt_array, closure_residuals

def find_diff_meas(antenna1_handle, antenna2_handle, baseline_handle=None, store_handle=None):
    """ Calculate the pseudorange measurement residuals for a given baseline """
    data_ant1 = antenna1_handle.antenna_data
    data_ant2 = antenna2_handle.antenna_data

    if baseline_handle is not None:
        _, ant1_idxs, _ = np.intersect1d(antenna1_handle.times_gps, baseline_handle.datetime_array[baseline_handle.range_data_idxs], return_indices=True)
        _, ant2_idxs, _ = np.intersect1d(antenna2_handle.times_gps, baseline_handle.datetime_array[baseline_handle.range_data_idxs], return_indices=True)
        diff_pr_data = baseline_handle.pr_diff[baseline_handle.range_data_idxs]
    else:
        times, ant1_idxs, ant2_idxs = np.intersect1d(antenna1_handle.times_gps, antenna2_handle.times_gps, return_indices=True)
        diff_pr_data = data_ant2.pr_data.values[ant2_idxs] - data_ant1.pr_data.values[ant1_idxs]

    diff_pr_model = data_ant2.pr_model.values[ant2_idxs] - data_ant1.pr_model.values[ant1_idxs] 
    residuals = diff_pr_data - diff_pr_model

    if store_handle.iono_comp_l4r and not store_handle.iono_free:
        if baseline_handle is not None:
            source_array = [store_handle.source_time_dict[time] for time in baseline_handle.datetime_array[baseline_handle.range_data_idxs]]
            stec_vals = store_handle.interp_l4r(baseline_handle.datetime_array[baseline_handle.range_data_idxs], source_array, antenna1_handle.l4r_name, antenna2_handle.l4r_name)
            residuals += ALPHA_IONO/baseline_handle.f1**2*const.c*stec_vals
        else:
            source_array = np.array([store_handle.source_time_dict[time] for time in times])
            idxs_bl = store_handle.find_l4r_interpable(times, source_array, antenna1_handle.l4r_name, antenna2_handle.l4r_name) 
            stec_vals = store_handle.interp_l4r(times[idxs_bl], source_array[idxs_bl], antenna1_handle.l4r_name, antenna2_handle.l4r_name)
            #mask = np.ones(times.shape[0], dtype=bool)
            #mask[idxs_bl] = False
            #residuals[mask] = 0 # turn off measurements with no TEC
            #if antenna2_handle.antenna_name == 'HN-VLBA' and np.any(residuals>200): breakpoint()
            residuals[idxs_bl] += ALPHA_IONO/(1.57542e9)**2*const.c*stec_vals

    #if antenna1_handle.antenna_name == 'PIE1' or antenna1_handle.antenna_name == 'PTVB': #True:# antenna_handle.antenna_name == 'FD_VLBA':
    #    corr_el_fig = plt.figure()
    #    corr_el_ax = corr_el_fig.add_subplot(111)
    #    corr_az_fig = plt.figure()
    #    corr_az_ax = corr_az_fig.add_subplot(111)
    #    azim_arr = antenna1_handle.azim_arr[ant1_idxs]
    #    elev_arr = antenna1_handle.elev_arr[ant1_idxs]
    #    for sat in store_handle.source_array:
    #        # get epochs of satellite
    #        if sat[0] == 'G':
    #            marker='x'
    #        elif sat[0] == 'E':
    #            marker='+'
    #        else:
    #            marker='1'
    #        idxs_sat = []
    #        for idx, time in enumerate(antenna1_handle.times_gps[ant1_idxs]):
    #            if sat == store_handle.source_time_dict[time]:
    #                idxs_sat.append(idx)
    #        idx_sat = np.array(idxs_sat)
    #        corr_el_ax.plot(elev_arr[idxs_sat], residuals[idxs_sat], marker=marker, linestyle='None', label=sat)
    #        corr_az_ax.plot(azim_arr[idxs_sat], residuals[idxs_sat], marker=marker, linestyle='None', label=sat)
    #    corr_el_ax.set_ylabel('PR error (m, data-model)')
    #    corr_el_ax.set_xlabel('elevation (deg)')
    #    corr_az_ax.set_ylabel('PR error (m, data-model)')
    #    corr_az_ax.set_xlabel('azimuth (deg)')
    #    corr_el_fig.savefig(antenna1_handle.antenna_name + antenna2_handle.antenna_name + 'diff_corr_fig_el.png')
    #    corr_az_fig.savefig(antenna1_handle.antenna_name + antenna2_handle.antenna_name + 'diff_corr_fig_az.png')
    #    plt.close(corr_el_fig)
    #    plt.close(corr_az_fig)
    #if antenna1_handle.antenna_name == 'PIE1' or antenna1_handle.antenna_name == 'PTVB': #True:# antenna_handle.antenna_name == 'FD_VLBA':
    #    obs_fig = plt.figure()
    #    obs_ax = obs_fig.add_subplot(111)
    #    for sat in store_handle.source_array:
    #        # get epochs of satellite
    #        if sat[0] == 'G':
    #            marker='x'
    #        elif sat[0] == 'E':
    #            marker='+'
    #        else:
    #            marker='1'
    #        idxs_sat = []
    #        for idx, time in enumerate(antenna1_handle.times_gps[ant1_idxs]):
    #            if sat == store_handle.source_time_dict[time]:
    #                idxs_sat.append(idx)
    #        idx_sat = np.array(idxs_sat)
    #        obs_ax.plot(antenna1_handle.times_gps[ant1_idxs][idxs_sat], residuals[idxs_sat], marker=marker, linestyle='None', label=sat)
    #        #  plot elevation and azimuth 
    #    time_deltas_full = (antenna1_handle.times_gps - antenna1_handle.times_gps[0])/np.timedelta64(1, 's')
    #    full_time_hr = np.round(time_deltas_full[-1]/3600) 
    #    interval_hr = int(np.ceil(full_time_hr/8))       
    #    obs_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    #    obs_ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hr))  # Adjust interval as needed
    #    obs_fig.autofmt_xdate()  # Auto-rotate date labels
    #    obs_ax.set_ylabel('PR diff error (m)')
    #    obs_fig.savefig(antenna1_handle.antenna_name +antenna2_handle.antenna_name+ 'diff_error.png')
    #    plt.close(obs_fig)
    #    breakpoint()

    return residuals

def get_obs_weights(store_handle, antenna1_handle, antenna2_handle, obs_type='range', use_idxs=[], baseline_handle=None, use_phase_weights=False):
    """ Calculate weights for observations """
    if baseline_handle is not None:
        times_gps, ant1_idxs, _ = np.intersect1d(antenna1_handle.times_gps, baseline_handle.datetime_array[use_idxs], return_indices=True)
        _, ant2_idxs, _ = np.intersect1d(antenna2_handle.times_gps, baseline_handle.datetime_array[use_idxs], return_indices=True)
    else:
        times_gps, ant1_idxs, ant2_idxs = np.intersect1d(antenna1_handle.times_gps, antenna2_handle.times_gps, return_indices=True)
           
    use_cov_kernel = False
    if obs_type == 'range':
        if baseline_handle is not None:
            if baseline_handle.use_cov_kernel_range is True:
                use_cov_kernel = True
    elif obs_type == 'phase':
        if baseline_handle.use_cov_kernel_phase is True:
            use_cov_kernel = True

    if baseline_handle is not None and use_cov_kernel is True:
        variables = ['azimuth', 'elevation', 'time', 'noise']
        azimuth = np.radians(antenna1_handle.azim_arr[ant1_idxs])
        elevation = np.radians(antenna1_handle.elev_arr[ant1_idxs])
        times = (times_gps - times_gps[0])/np.timedelta64(1, 's')
        X = np.vstack((azimuth, elevation, times))
        cov = baseline_handle.build_covariance_matrix(obs_type, X, variables)
    else:
        if store_handle.sol_type == 'VLBI':
            if obs_type == 'range':
                var_diag = baseline_handle.grdel_err[use_idxs]**2
            elif obs_type == 'phase':
                if use_phase_weights is False:
                    #weights = np.ones(len(baseline_handle.phdel_err))
                    var_diag = 100*np.mean(baseline_handle.grdel_err[use_idxs]**2)*np.ones(len(baseline_handle.phdel_err[use_idxs]))
                    #weights = np.sqrt(np.mean(1/baseline_handle.grdel_err**2))/100*np.ones(len(baseline_handle.phdel_err))
                else:
                    var_diag = baseline_handle.phdel_err[use_idxs]**2
                    #weights = 1/baseline_handle.phdel_err

            if baseline_handle is not None and use_cov_kernel is False:
                # additive baseline uncertainty
                if obs_type == 'range':
                    q_add = baseline_handle.q_range
                elif obs_type == 'phase':
                    q_add = baseline_handle.q_phase
                var_diag += q_add**2
                cov = np.diag(var_diag)

        elif store_handle.sol_type == 'GNSS':
            if baseline_handle is None:
                if obs_type == 'range':
                    #weights = 1/np.sqrt(antenna1_handle.pr_errors[ant1_idxs]**2+antenna2_handle.pr_errors[ant2_idxs]**2)
                    var_diag = antenna1_handle.pr_errors[ant1_idxs]**2+antenna2_handle.pr_errors[ant2_idxs]**2
                elif obs_type == 'phase':
                    if use_phase_weights is False:
                        var_diag = 100*np.mean(antenna1_handle.pr_errors[ant1_idxs]**2+antenna2_handle.pr_errors[ant2_idxs]**2)*np.ones(len(ant1_idxs))
                    else:
                        var_diag = antenna1_handle.cp_errors[ant1_idxs]**2+antenna2_handle.cp_errors[ant2_idxs]**2
                cov = np.diag(var_diag)
            elif obs_type == 'range' and baseline_handle.q_range != 0 and baseline_handle.q_range_satellite != 0:
                # Construct cofactor matrices Q1 and Q2
                Q_eta = construct_Q_eta(store_handle, baseline_handle, use_idxs)
                Q_epsilon = np.eye(Q_eta.shape[0])  # Identity matrix
                cov = baseline_handle.q_range_satellite**2*Q_eta + baseline_handle.q_range**2*Q_epsilon
            elif obs_type == 'phase' and baseline_handle.q_phase != 0 and baseline_handle.q_phase_satellite != 0:
                Q_eta = construct_Q_eta(store_handle, baseline_handle, use_idxs)
                Q_epsilon = np.eye(Q_eta.shape[0])  # Identity matrix
                cov = baseline_handle.q_phase_satellite**2*Q_eta + baseline_handle.q_phase**2*Q_epsilon
            elif obs_type == 'range':
                cov = np.diag(antenna1_handle.pr_errors[ant1_idxs]**2 + antenna2_handle.pr_errors[ant2_idxs]**2 + baseline_handle.q_range**2)
            elif obs_type == 'phase':
                cov = np.diag(antenna1_handle.cp_errors[ant1_idxs]**2 + antenna2_handle.cp_errors[ant2_idxs]**2 + baseline_handle.q_phase**2)

    weight_mat = np.linalg.inv(cov)
    weight_chol = np.linalg.cholesky(weight_mat) # cholesky decomposition
    return weight_chol

def construct_float_amb(diff_phase_data, diff_phase_model, wavelength):
    """ Construct the initial, coarse float ambiguity estimate """
    N_arr = 1/wavelength*(diff_phase_model-diff_phase_data)
    return N_arr

def find_sigmas(data_series, nsig=3.5):
    """ Find the standard deviation of a data series by iteratively removing outliers
        provide mask for data elements within nsig of 0
    """
    noi = np.sum(data_series**2) # total noise variance
    nel = len(data_series) # number of elements
    n_iter = int(np.floor(nel/2)) # max number of iterations

    flagged = np.isnan(data_series)
    #np.zeros_like(data_series, dtype=bool)
    #flagged = np.bitwise_or(flagged, np.isnan(data_series))
    for _ in range(n_iter):
        blanked = 0 # have we blanked a value
        
        for idx, data_el in enumerate(data_series):
            if flagged[idx] == 0 and data_el**2 > nsig**2*(noi/nel):
                noi = noi - data_el**2
                nel = nel - 1
                flagged[idx] = 1
                blanked = 1

        if blanked == 0: break # break if no values were removed this iteration
    if nel == 0:
        idxs_include = ~np.isnan(data_series)
        sigma = np.std(data_series)
        #print('find_sigmas -- all data flagged')
    else:
        sigma = np.sqrt(noi/nel)
        idxs_include = ~flagged
    return sigma, idxs_include

def remove_outliers(residuals, baseline_handles, phase_only=False, phase=True):
    """Remove outliers from phase residuals before re-computing LS solution"""
    num_samples = 0
    for jdx, baseline_handle in enumerate(baseline_handles): # generate differential measurements on the baselines
        if phase_only is False:
            residuals_range= residuals[num_samples:\
                                    num_samples + len(baseline_handle.range_data_idxs)]
            num_samples = num_samples + len(baseline_handle.range_data_idxs) 
            sigma_range, use_idxs_range = find_sigmas(residuals_range)
        
        if phase is True:
            residuals_phase = residuals[num_samples:\
                                     num_samples + len(baseline_handle.phase_data_idxs)]
            num_samples = num_samples + len(baseline_handle.phase_data_idxs) 
            sigma_phase, use_idxs_phase = find_sigmas(residuals_phase)
        else:
            use_idxs_phase = use_idxs_range
        
        if phase_only is False:
            # restrict phase points to good range points
            common_idxs, placement_range, placement_phase = np.intersect1d(baseline_handle.range_data_idxs, baseline_handle.phase_data_idxs, return_indices=True)
            restrict_idxs_phase = np.bitwise_and(use_idxs_range[placement_range], use_idxs_phase[placement_phase])
            # turn to false idxs with use_idxs_range=False and use_idxs_phase=True
            use_idxs_phase[placement_phase] = restrict_idxs_phase 

            baseline_handle.range_data_idxs = baseline_handle.range_data_idxs[use_idxs_range]

        baseline_handle.phase_data_idxs = baseline_handle.phase_data_idxs[use_idxs_phase]

def iterative_remove_outliers(store_handle, ls_sol, bounds, ls_args, res_fcn, jac, sol_type, n_ao_state, n_grav_state, phase_only=False, phase=True):
    """ Iteratively remove outliers, adjusting the least squares solution each time """
    delta_samples = np.inf
    len_last = np.inf
    baselines = ls_args[1]
    state_expanded = ls_sol.x
    antenna_handles = ls_args[3]
    baseline_handles = ls_args[9]
    while np.abs(delta_samples)>0:
        residuals = ls_sol.fun
        len_pts = 0
        num_samples = 0
        delta_range = 0
        delta_phase = 0
        for jdx, baseline_handle in enumerate(baseline_handles): # generate differential measurements on the baselines
            T_medfilt = 1800 # s
            t_delta = np.mean(np.diff((baseline_handle.datetime_array-baseline_handle.datetime_array[0])/np.timedelta64(1, 's')))
            N_window = int(np.round(T_medfilt/t_delta))
            if N_window % 2 == 0:
                N_window += 1 # must be odd
            if phase_only is False:
                residuals_range = residuals[num_samples:\
                                        num_samples + len(baseline_handle.range_data_idxs)]
                sigma_range, use_idxs_range = find_sigmas(residuals_range-medfilt(residuals_range, kernel_size=N_window))
                num_samples = num_samples + len(baseline_handle.range_data_idxs) 

            if phase is True:
                residuals_phase = residuals[num_samples:\
                                         num_samples + len(baseline_handle.phase_data_idxs)]
                sigma_phase, use_idxs_phase = find_sigmas(residuals_phase-medfilt(residuals_phase, kernel_size=N_window))
                num_samples = num_samples + len(baseline_handle.phase_data_idxs) 
            else:
                use_idxs_phase = use_idxs_range
            
            if phase_only is False:
                # restrict phase points to good range points
                common_idxs, placement_range, placement_phase = np.intersect1d(baseline_handle.range_data_idxs, baseline_handle.phase_data_idxs, return_indices=True)
                restrict_idxs_phase = np.bitwise_and(use_idxs_range[placement_range], use_idxs_phase[placement_phase])
                # turn to false idxs with use_idxs_range=False and use_idxs_phase=True
                use_idxs_phase[placement_phase] = restrict_idxs_phase 
                if len(use_idxs_phase)> len(baseline_handle.phase_data_idxs):
                    use_idxs_phase = use_idxs_phase[:len(baseline_handle.phase_data_idxs)]

                delta_range += len(baseline_handle.range_data_idxs) - np.sum(use_idxs_range)
                baseline_handle.range_data_idxs = baseline_handle.range_data_idxs[use_idxs_range]
                len_pts += len(use_idxs_range)

            delta_phase += len(baseline_handle.phase_data_idxs) - np.sum(use_idxs_phase)
            baseline_handle.phase_data_idxs = baseline_handle.phase_data_idxs[use_idxs_phase] 
            len_pts += len(use_idxs_phase)

        if store_handle.stochastic_clock or store_handle.stochastic_trop:
            ref_antenna = ls_args[0]
            baselines = ls_args[1]
            clock_idxs = ls_args[4]
            trop_idxs = ls_args[6]
            disb_idxs = ls_args[8]
            if phase is True:
                amb_state_idxs = ls_args[13]
                phase_clock_idxs = ls_args[17]
                phase_disb_idxs = ls_args[18]
                state_expanded, _, clock_idxs, trop_idxs, disb_idxs, phase_clock_idxs, phase_disb_idxs, amb_state_idxs = \
                        adjust_stoch_params(store_handle, antenna_handles, baseline_handles,\
                        baselines, ref_antenna, state_expanded, clock_idxs, trop_idxs, disb_idxs, n_ao_state, \
                        n_grav_state, phase, phase_clock_idxs, phase_disb_idxs, amb_state_idxs, phase_only)
            else:
                state_expanded, _, clock_idxs, trop_idxs, disb_idxs, _, _, _ = adjust_stoch_params(store_handle, antenna_handles, baseline_handles,\
                        baselines, ref_antenna, state_expanded, clock_idxs, trop_idxs, disb_idxs, n_ao_state, n_grav_state)
            ls_args_list = list(ls_args)
            ls_args_list[4] = clock_idxs
            ls_args_list[6] = trop_idxs
            ls_args_list[8] = disb_idxs
            if phase is True: 
                ls_args_list[13] = amb_state_idxs
                ls_args_list[17] = phase_clock_idxs
                ls_args_list[18] = phase_disb_idxs
            ls_args = tuple(ls_args_list)
 
        delta_samples = len_last-len_pts
        len_last = len_pts
        bound_low = bounds[0][:len(state_expanded)]
        bound_high = bounds[1][:len(state_expanded)]
        bounds = (bound_low, bound_high)
        ls_sol = least_squares(res_fcn, state_expanded, jac=jac, method='trf',\
            max_nfev=100, bounds=bounds, verbose=0, x_scale = 'jac', xtol=1e-15,\
            args=ls_args)
        if delta_samples < np.inf: print('removed ' + str(delta_samples) +' outliers')

    return ls_sol, ls_args, bounds

def trim_amb(baseline_handles, start):
    """ find slip slices with insufficient data length and remove them """
    n_amb = 0
    del_inds = []
    for jdx, baseline_handle in enumerate(baseline_handles):
        good_slip_slices = []
        len_arr = []
        n_amb_orig = baseline_handle.n_amb_state # integers are immutable
        for slip_idx, slip_slice in enumerate(baseline_handle.slip_slices_arr):
            intersect_idxs = np.intersect1d(slip_slice, baseline_handle.phase_data_idxs)
            if len(intersect_idxs) < MIN_SLICE:
                del_inds.append(start + n_amb + slip_idx) 
                baseline_handle.n_amb_state -= 1
            len_arr.append(len(intersect_idxs))
        n_amb += n_amb_orig
        baseline_handle.trim_phase_idxs(len_arr)

    return del_inds

def trim_amb_state(baseline_handles, state_expanded, amb_state_idxs):
    """trim the ambiguity state to drop estimated parameters with too little data"""
    del_inds = trim_amb(baseline_handles, amb_state_idxs.start)
    state_test = state_expanded + 0.0

    if len(del_inds) > 0:
        state_expanded = np.delete(state_expanded, del_inds)
    amb_state_idxs = slice(amb_state_idxs.start, len(state_expanded))
    
    return state_expanded, amb_state_idxs

def trim_amb_Zdom(baseline_handles, iZt):
    """trim the ambiguity state in the Z domain"""
    # iZt is the mapping from Z back to A domain
    del_inds = trim_amb(baseline_handles, 0)
    
    if len(del_inds) > 0:
        iZt = np.delete(iZt, del_inds, axis=0)

    return iZt

def gen_phase_clock_state(store_handle, antenna_handles, baseline_handles, baselines, ref_antenna, state_expanded, end_state, clock_idxs, \
        clock_poly_length, trop_idxs, trop_poly_length, disb_idxs, phase_clock_idxs=None, phase_disb_idxs=None, amb_state_idxs=None):
    """ Generate the phase clock state covering only datetimes currently selected in the dataset """
    n_clock_phase = 0
    grdel_state = state_expanded[:end_state]
    if amb_state_idxs is not None:
        amb_state = state_expanded[amb_state_idxs]
    if phase_clock_idxs is not None:
        phase_clock_states = state_expanded[phase_clock_idxs]
    clock_states = state_expanded[clock_idxs]

    if store_handle.estimate_disb is True:
        disb_states = state_expanded[disb_idxs]
    if store_handle.estimate_phase_disb is True:
        if phase_disb_idxs is not None:
            phase_disb_states = state_expanded[phase_disb_idxs]
        else:
            phase_disb_states = disb_states*0.0 # initialize as 0

    if store_handle.global_linear_clock is True:
        idx_start = 1
    elif store_handle.global_quadratic_clock is True:
        idx_start = 2
    else:
        idx_start = 0
    
    phase_clock_state_new = np.array([])
    for idx, antenna_handle in enumerate(antenna_handles):
        begin_date = np.datetime64('2200-01-01T00:00:00.0')
        end_date = np.datetime64('1900-01-01T00:00:00.0')
        for jdx, baseline in enumerate(baselines):
            # get location of earliest phase datapoint
            if idx in baseline:
                baseline_handle = baseline_handles[jdx]
                begin_date = min(begin_date, baseline_handle.datetime_array[baseline_handle.phase_data_idxs[0]])
                end_date = max(end_date, baseline_handle.datetime_array[baseline_handle.phase_data_idxs[-1]])
            else:
                continue

        if ref_antenna == antenna_handle.antenna_name: 
            continue
        else:
            if begin_date == antenna_handle.phase_clock_start and end_date == antenna_handle.phase_clock_end \
                    and phase_clock_idxs is not None:
                # no change -- use old phase clock as initial values
                phase_clock_state_ant = phase_clock_states[antenna_handle.phase_clock_idxs]
                nclock_phase_antenna = len(phase_clock_state_ant)-1
            elif begin_date == antenna_handle.times_gps[0] and end_date == antenna_handle.times_gps[-1]:
                # phase clock initialized as range clock state
                clock_state_ant = clock_states[antenna_handle.range_clock_idxs] # range clock state
                phase_clock_state_ant = clock_state_ant
                nclock_phase_antenna = len(clock_state_ant)-1
            else:
                # resample clock state to new epochs
                if clock_poly_length > 0:
                    phase_exp_length = (end_date-begin_date)/np.timedelta64(1, 's')
                    nclock_phase_antenna = int(np.ceil(phase_exp_length/clock_poly_length)) # number of piecewise-linear intervals
                    new_epochs = [begin_date]
                    for interval in range(nclock_phase_antenna):
                        new_epochs.append(begin_date + clock_poly_length*(interval+1)*np.timedelta64(1, 's'))
                    phase_clock_state_ant  = np.zeros(len(new_epochs)+idx_start)
                    nclock_phase_antenna = len(phase_clock_state_ant)-1
                else:
                    phase_clock_state_ant  = np.zeros(idx_start+1)
                    nclock_phase_antenna = len(phase_clock_state_ant)-1

                if phase_clock_idxs is not None and clock_poly_length > 0:
                    # fit clock function to previous phase clock function
                    begin_past = antenna_handle.phase_clock_start
                    phase_clock_past_ant = phase_clock_states[antenna_handle.phase_clock_idxs]
                    phase_clock_past_samples  = sample_poly_at_interval(phase_clock_past_ant[idx_start:], clock_poly_length, new_epochs, begin_past)
                    diffs_clock = np.diff(np.array(phase_clock_past_samples))
                    phase_clock_state_ant[:idx_start] = phase_clock_past_ant[:idx_start]
                    phase_clock_state_ant[idx_start] = phase_clock_past_samples[0]
                    phase_clock_state_ant[idx_start+1:] = diffs_clock
                elif clock_poly_length > 0:
                    # fit clock function to range clock function
                    begin_past = antenna_handle.times_gps[0]
                    clock_past_ant = clock_states[antenna_handle.range_clock_idxs] # range clock state
                    clock_past_samples  = sample_poly_at_interval(clock_past_ant[idx_start:], clock_poly_length, new_epochs, begin_past)
                    diffs_clock = np.diff(np.array(clock_past_samples).flatten())
                    phase_clock_state_ant[:idx_start] = clock_past_ant[:idx_start]
                    phase_clock_state_ant[idx_start] = clock_past_samples[0]
                    phase_clock_state_ant[idx_start+1:] = diffs_clock

            phase_clock_samples = np.zeros(len(antenna_handle.times_gps))
            if store_handle.global_linear_clock is True or store_handle.global_quadratic_clock is True:
                phase_clock_state_global = phase_clock_state_ant[:idx_start]
                phase_clock_samples = sample_global_poly_at_interval(phase_clock_state_global, antenna_handle.times_gps)
            else:
                phase_clock_samples = np.zeros(len(antenna_handle.times_gps))
            
            if clock_poly_length > 0: 
                phase_clock_samples += sample_poly_at_interval(phase_clock_state_ant[idx_start:], clock_poly_length, \
                        antenna_handle.times_gps, begin_date)
            else:
                # only global clock model
                phase_clock_samples += phase_clock_state_ant[idx_start]

            antenna_handle.hold_clock(phase_clock_samples, phase_delay=True)

            antenna_phase_clock_idxs = slice(n_clock_phase, n_clock_phase+nclock_phase_antenna+1)
            antenna_handle.hold_phase_clock_params(begin_date, end_date, antenna_phase_clock_idxs)
            phase_clock_state_new = np.append(phase_clock_state_new, phase_clock_state_ant)
            n_clock_phase = n_clock_phase + nclock_phase_antenna+1
    
    phase_clock_idxs = slice(len(grdel_state), len(grdel_state)+len(phase_clock_state_new))
    state_expanded = np.append(grdel_state, phase_clock_state_new)

    if store_handle.estimate_phase_disb is True:
        phase_disb_idxs = slice(len(state_expanded), len(state_expanded)+len(phase_disb_states))
        state_expanded = np.append(state_expanded, phase_disb_states)
    else:
        phase_disb_idxs = slice(len(state_expanded), len(state_expanded))

    first_amb = len(state_expanded)
    if amb_state_idxs is not None:
        state_expanded = np.append(state_expanded, amb_state)
    amb_state_idxs = slice(first_amb,len(state_expanded))

    return state_expanded, phase_clock_idxs, phase_disb_idxs, amb_state_idxs

def adjust_stoch_params(store_handle, antenna_handles, baseline_handles, baselines, ref_antenna, state_expanded, clock_idxs, \
        trop_idxs, disb_idxs, n_ao_state, n_grav_state, phase=False, phase_clock_idxs=None, phase_disb_idxs=None, amb_state_idxs=None, phase_only=False):
    """ Adjust the state elements in stochastic models to deal with deactivated indices (epochs not estimated need to be removed from the state) """
    n_clock = 0
    n_clock_phase = 0
    n_trop = 0
    pos_state = state_expanded[:clock_idxs.start]
    if phase_only is False:
        clock_states = state_expanded[clock_idxs]
        clock_state_new = np.array([])
    if amb_state_idxs is not None:
        amb_state = state_expanded[amb_state_idxs]
    if phase is True:
        phase_clock_states = state_expanded[phase_clock_idxs]
        phase_clock_state_new = np.array([])
    if store_handle.estimate_trop is True:
        trop_states = state_expanded[trop_idxs]
        trop_state_new = np.array([])
    if store_handle.estimate_disb is True and phase_only is False:
        disb_states = state_expanded[disb_idxs]
    else:
        disb_states = []
    if phase is True and store_handle.estimate_phase_disb is True:
        if phase_disb_idxs is not None:
            phase_disb_states = state_expanded[phase_disb_idxs]
            if len(disb_states) == 0 and phase_only is False:
                disb_states = phase_disb_states.copy() # resurrect the disb states after phase only solution
        else:
            phase_disb_states = disb_states*0.0 # initialize as 0
            #phase_disb_states = np.zeroes(len(antenna_handles)-1)# initialize as 0

    ao_idxs = slice(clock_idxs.stop,clock_idxs.stop+n_ao_state)
    grav_idxs = slice(clock_idxs.stop+n_ao_state,clock_idxs.stop+n_ao_state+n_grav_state)
    ao_state = state_expanded[ao_idxs]
    grav_state = state_expanded[grav_idxs]

    if store_handle.global_linear_clock is True:
        idx_start = 1
    elif store_handle.global_quadratic_clock is True:
        idx_start = 2
    else:
        idx_start = 0

    for idx, antenna_handle in enumerate(antenna_handles):
        # initialize observation time arrays
        if store_handle.stochastic_clock is True:
            if phase_only is False:
                times_full_range = np.array([], dtype=np.datetime64)
            if phase is True:
                times_full_phase = np.array([], dtype=np.datetime64)
        if store_handle.stochastic_trop is True:
            times_full_trop = np.array([], dtype=np.datetime64)

        for jdx, baseline in enumerate(baselines):
            # find union of all times with active observations
            if idx in baseline:
                baseline_handle = baseline_handles[jdx]
                if store_handle.stochastic_clock is True:
                    if phase_only is False:
                        times_full_range = np.union1d(times_full_range, baseline_handle.datetime_array[baseline_handle.range_data_idxs])
                    if phase is True:
                        times_full_phase = np.union1d(times_full_phase, baseline_handle.datetime_array[baseline_handle.phase_data_idxs])
                if store_handle.stochastic_trop is True:
                    if phase_only is False:
                        times_full_trop = np.union1d(times_full_trop, baseline_handle.datetime_array[baseline_handle.range_data_idxs])
                    if phase is True:
                        times_full_trop = np.union1d(times_full_trop, baseline_handle.datetime_array[baseline_handle.phase_data_idxs])
            else:
                continue

        if ref_antenna == antenna_handle.antenna_name: 
            continue
        else:
            if phase_only is False:
                clock_state_ant = clock_states[antenna_handle.range_clock_idxs] # range clock state
                if store_handle.global_linear_clock is True or store_handle.global_quadratic_clock is True:
                    clock_state_global = clock_state_ant[:idx_start]
                    if store_handle.stochastic_clock is True:
                        clock_samples = sample_global_poly_at_interval(clock_state_global, times_full_range, \
                                antenna_handle.times_gps[0], antenna_handle.times_gps[-1])
                    else:
                        clock_samples = sample_global_poly_at_interval(clock_state_global, antenna_handle.times_gps)
                elif store_handle.stochastic_clock is True:
                    clock_samples = np.zeros(len(times_full_range))
                else:    
                    clock_samples = np.zeros(len(clock_state_ant))

            if phase is True and phase_clock_idxs is not None:
                phase_clock_state_ant = phase_clock_states[antenna_handle.phase_clock_idxs]
            else:
                phase_clock_state_ant = clock_states[antenna_handle.range_clock_idxs]

            if phase is True:
                if store_handle.global_linear_clock is True or store_handle.global_quadratic_clock is True:
                    phase_clock_state_global = phase_clock_state_ant[:idx_start]
                    if store_handle.stochastic_clock is True:
                        phase_clock_samples = sample_global_poly_at_interval(phase_clock_state_global, times_full_phase,\
                                antenna_handle.times_gps[0], antenna_handle.times_gps[-1])
                    else:
                        phase_clock_samples = sample_global_poly_at_interval(phase_clock_state_global, antenna_handle.times_gps)
                elif store_handle.stochastic_clock is True:
                    phase_clock_samples = np.zeros(len(times_full_phase))
                else:
                    phase_clock_samples = np.zeros(len(phase_clock_state_ant))

            if store_handle.stochastic_clock is True:
                if phase_only is False:
                    clock_state_adj = np.zeros(len(times_full_range)+idx_start)
                    clock_state_adj[idx_start:] = sample_stoch_params_at_times(clock_state_ant[idx_start:], antenna_handle.clock_times, times_full_range)
                    if idx_start>0:
                        clock_state_adj[:idx_start] = clock_state_ant[:idx_start]
                    clock_state_ant = clock_state_adj
                    clock_samples += clock_state_adj[idx_start:] 

                    antenna_clock_idxs = slice(len(clock_state_new), len(clock_state_new) + len(clock_state_adj))
                    antenna_handle.hold_clock(clock_samples)
                    antenna_handle.hold_range_clock_params(antenna_clock_idxs, times_full_range)

                if phase is True:
                    # do the same for phase clock state
                    if phase_clock_idxs is not None:
                        # interpolate w/ interp1d to get new phase clock state
                        phase_clock_state_adj = np.zeros(len(times_full_phase)+idx_start)
                        phase_clock_state_adj[idx_start:] = sample_stoch_params_at_times(phase_clock_state_ant[idx_start:], antenna_handle.phase_clock_times, times_full_phase)
                        if idx_start>0:
                            phase_clock_state_adj[:idx_start] = phase_clock_state_ant[:idx_start]
                        phase_clock_state_ant = phase_clock_state_adj
                        phase_clock_samples += phase_clock_state_ant[idx_start:]
                    else:
                        # phase clock initialized as range clock state
                        phase_clock_samples += clock_state_ant[idx_start:]
                        phase_clock_state_ant = clock_state_ant.copy()

                    antenna_handle.hold_clock(phase_clock_samples, phase_delay=True)
                    antenna_phase_clock_idxs = slice(len(phase_clock_state_new), len(phase_clock_state_new) + len(phase_clock_state_ant))
                    antenna_handle.hold_phase_clock_params(times_full_phase[0], times_full_phase[-1], antenna_phase_clock_idxs, times_full_phase)

            if store_handle.estimate_trop is True and antenna_handle.estimate_trop is True:
                trop_state_ant = trop_states[antenna_handle.trop_idxs]
                #if phase is True:
                #    breakpoint()

                if store_handle.stochastic_trop is True:
                    # interpolate w/ interp1d to get new range clock state
                    trop_state_ant = sample_stoch_params_at_times(trop_state_ant, antenna_handle.trop_times, times_full_trop)
                    antenna_trop_idxs = slice(len(trop_state_new), len(trop_state_new) + len(trop_state_ant))
                    antenna_handle.hold_trop(trop_state_ant)
                    antenna_handle.hold_trop_params(antenna_trop_idxs, times_full_trop)
            
            if store_handle.stochastic_clock is True:
                if phase_only is False:
                    clock_state_new = np.append(clock_state_new, clock_state_ant)
                if phase is True:
                    phase_clock_state_new = np.append(phase_clock_state_new, phase_clock_state_ant)
            if store_handle.estimate_trop is True and antenna_handle.estimate_trop is True:
                trop_state_new = np.append(trop_state_new, trop_state_ant)
     
    if store_handle.stochastic_clock is True and phase_only is False:
        state_expanded = np.append(pos_state, clock_state_new)
        clock_idxs = slice(len(pos_state), len(state_expanded))
    else:
        state_expanded = pos_state

    state_expanded = np.append(state_expanded, ao_state)
    state_expanded = np.append(state_expanded, grav_state)

    if store_handle.estimate_trop is True:
        trop_idxs = slice(len(state_expanded),len(state_expanded)+len(trop_state_new))
        state_expanded = np.append(state_expanded, trop_state_new)
    else:
        trop_idxs = slice(len(state_expanded),len(state_expanded))

    if store_handle.estimate_disb is True and phase_only is False:
        disb_idxs = slice(len(state_expanded), len(state_expanded)+len(disb_states))
        state_expanded = np.append(state_expanded, disb_states)
    else:
        disb_idxs = slice(len(state_expanded), len(state_expanded))
    end_range_state = len(state_expanded)

    if phase is True:
        phase_clock_idxs = slice(end_range_state, end_range_state+len(phase_clock_state_new)) # phase clock polynomials are stored after positions/range clock
        state_expanded = np.append(state_expanded, phase_clock_state_new)

    if store_handle.estimate_phase_disb is True and phase is True:
        phase_disb_idxs = slice(len(state_expanded), len(state_expanded)+len(phase_disb_states))
        state_expanded = np.append(state_expanded, phase_disb_states)
    else:
        phase_disb_idxs = slice(len(state_expanded), len(state_expanded))

    first_amb = len(state_expanded)
    if amb_state_idxs is not None:
        state_expanded = np.append(state_expanded, amb_state)

    amb_state_idxs = slice(first_amb, len(state_expanded))

    return state_expanded, end_range_state, clock_idxs, trop_idxs, disb_idxs, phase_clock_idxs, phase_disb_idxs, amb_state_idxs


def get_process_variance_times(store_handle, antenna_handle, model_type, phase=False, eval_times=None, only_mat=False, clock_mat_type='rw'):
    """ Generate process variance values for a stochastic clock or troposphere model
        See Herring (1990) -- Geodesy by Radio Interferometry (or Joe Skeens' dissertation) for details
    """
    if model_type == 'clock':
        if eval_times is not None:
            duration_arr = np.array([store_handle.duration_dict[time] for time in eval_times[1:]])
            times_sec = (eval_times-eval_times[0])/np.timedelta64(1, 's')
        elif phase is False:
            duration_arr = np.array([store_handle.duration_dict[time] for time in antenna_handle.clock_times[1:]])
            times_sec = (antenna_handle.clock_times-antenna_handle.clock_times[0])/np.timedelta64(1, 's')
        else:
            duration_arr = np.array([store_handle.duration_dict[time] for time in antenna_handle.phase_clock_times[1:]])
            times_sec = (antenna_handle.phase_clock_times-antenna_handle.phase_clock_times[0])/np.timedelta64(1, 's')

        times_diff_sec = np.diff(times_sec)
        duration_arr[times_diff_sec-duration_arr<0] = 0 # sometimes in VLBI we have measurements with a shifted reference time
        if only_mat is True:
            # we need the variance component matrix, not the full process variance
            if clock_mat_type == 'rw':
                process_variance = times_diff_sec - duration_arr/6
                if np.any(process_variance<0):
                    process_variance[process_variance<0] = times_diff_sec
            elif clock_mat_type == 'irw':
                process_variance = times_diff_sec**2*times_sec[:-1] + times_diff_sec**3/3 + duration_arr**3/120 
            else:
                raise ValueError('Unknown clock model type ' + str(clock_mat_type) +' (should be either rw or irw)')
        else:
            process_variance = antenna_handle.clock_psd_irw*(times_diff_sec**2*times_sec[:-1] + times_diff_sec**3/3 + duration_arr**3/120)/FACTOR_IRW \
                              + antenna_handle.clock_psd_rw*(times_diff_sec - duration_arr/6)/FACTOR_RW

    elif model_type == 'trop':
        duration_arr = np.array([store_handle.duration_dict[time] for time in antenna_handle.trop_times[1:]])
        times_sec = (antenna_handle.trop_times-antenna_handle.trop_times[0])/np.timedelta64(1, 's')
        times_diff_sec = np.diff(times_sec)
        duration_arr[times_diff_sec-duration_arr<0] = 0 # sometimes in VLBI we have measurements with a shifted reference time
        if only_mat is True:
            process_variance = times_diff_sec - duration_arr/6
        else:
            process_variance = antenna_handle.trop_psd_rw*(times_diff_sec - duration_arr/6)/FACTOR_RW
    else:
        raise ValueError('Unknown model type: ' + model_type)

    if np.any(process_variance<0): breakpoint()
    
    return process_variance

def union_of_slices(array, *slices):
    """ Get an array with indices from union of multiple slice objects """
    indices = set()
    for s in slices:
        indices.update(range(*s.indices(len(array))))
    return array[sorted(indices)]

def set_bounds_phase_clock(bound_low, bound_high, clock_bound, store_handle, antenna_handles, baseline_handles, baselines, ref_antenna, state_expanded, end_state, clock_idxs, \
        clock_poly_length, trop_idxs, disb_idxs, phase_clock_idxs, phase_disb_idxs, n_ao_state, n_grav_state, amb_state_idxs=None):
    """ Set the phase clock bounds to +/- clock_bound*100% from the range clock function, fix bounds length """
    n_clock_phase = 0
    grdel_state = state_expanded[:end_state]
    if amb_state_idxs is not None:
        amb_state = state_expanded[amb_state_idxs]
    if phase_clock_idxs is not None:
        phase_clock_states = state_expanded[phase_clock_idxs]
    clock_states = state_expanded[clock_idxs]
    
    bound_low_rxpos = bound_low[slice(0,(len(antenna_handles)-1)*3)]
    bound_high_rxpos = bound_high[slice(0,(len(antenna_handles)-1)*3)]
    bound_low_clock = bound_low[clock_idxs]
    bound_high_clock = bound_high[clock_idxs]
    bound_low_ao = bound_low[slice(clock_idxs.stop,clock_idxs.stop+n_ao_state)]
    bound_high_ao = bound_high[slice(clock_idxs.stop,clock_idxs.stop+n_ao_state)]
    bound_low_grav = bound_low[slice(clock_idxs.stop+n_ao_state,clock_idxs.stop+n_ao_state+n_grav_state)]
    bound_high_grav = bound_high[slice(clock_idxs.stop+n_ao_state,clock_idxs.stop+n_ao_state+n_grav_state)]
    bound_low_trop = bound_low[trop_idxs]
    bound_high_trop = bound_high[trop_idxs]
    bound_low_disb = bound_low[disb_idxs]
    bound_high_disb = bound_high[disb_idxs]
    bound_low_phase_disb = bound_low[phase_disb_idxs]
    bound_high_phase_disb = bound_high[phase_disb_idxs]
    if amb_state_idxs is not None:
        bound_low_amb = bound_low[amb_state_idxs]
        bound_high_amb = bound_high[amb_state_idxs]
    
    bound_low_pc = np.array([])
    bound_high_pc = np.array([]) 
    for idx, antenna_handle in enumerate(antenna_handles):
        phase_times = np.array([], dtype=np.datetime64) 
        begin_date = np.datetime64('2200-01-01T00:00:00.0')
        end_date = np.datetime64('1900-01-01T00:00:00.0')
        for jdx, baseline in enumerate(baselines):
            # get location of earliest phase datapoint
            if idx in baseline:
                baseline_handle = baseline_handles[jdx]
                begin_date = min(begin_date, baseline_handle.datetime_array[baseline_handle.phase_data_idxs[0]])
                end_date = max(end_date, baseline_handle.datetime_array[baseline_handle.phase_data_idxs[-1]])
                phase_times = np.union1d(phase_times, baseline_handle.datetime_array[baseline_handle.phase_data_idxs])
            else:
                continue

        if ref_antenna == antenna_handle.antenna_name: 
            continue
        else:
            #phase_exp_length = (end_date-begin_date)/np.timedelta64(1, 's')
            #nclock_phase_antenna = int(np.ceil(phase_exp_length/clock_poly_length)) # number of piecewise-linear intervals
            if store_handle.global_linear_clock is True:
                idx_start = 1
            elif store_handle.global_quadratic_clock is True:
                idx_start = 2
            else:
                idx_start = 0
            phase_clock_state_ant = phase_clock_states[antenna_handle.phase_clock_idxs][idx_start:]
            nclock_phase_antenna = len(phase_clock_state_ant)-1

            # set bounds for the phase clock function to agree with the range clock function
            clock_state_ant = clock_states[antenna_handle.range_clock_idxs][idx_start:] # range clock state

            # initialize bounds
            bound_low_pc_ant = np.zeros(len(phase_clock_states[antenna_handle.phase_clock_idxs]))
            bound_high_pc_ant = np.zeros(len(phase_clock_states[antenna_handle.phase_clock_idxs]))

            bound_low_pc_ant[0] = -np.inf
            bound_high_pc_ant[0] = np.inf
            bound_low_pc_ant[:idx_start] = -np.inf*np.ones(idx_start)
            bound_high_pc_ant[:idx_start] = np.inf*np.ones(idx_start)
            if store_handle.stochastic_clock is False:
                # only global clock model
                bound_low_pc_ant[idx_start] = -np.inf
                bound_high_pc_ant[idx_start] = np.inf

            if store_handle.stochastic_clock is True:
                range_clock_samples = sample_stoch_params_at_times(clock_state_ant, antenna_handle.clock_times, phase_times)
                range_clock_state_ant = range_clock_samples
                bound_low_pc_ant[idx_start:] = range_clock_samples - clock_bound
                bound_high_pc_ant[idx_start:] = range_clock_samples + clock_bound
            else:
                # we cant use the range clock state directly to set bounds because the phase clock may be sampled at 
                # different epochs, sample the range clock at these epochs explicitly
                if clock_poly_length>0:
                    epochs = [begin_date]
                    for interval in range(nclock_phase_antenna):
                        epochs.append(begin_date + clock_poly_length*(interval+1)*np.timedelta64(1, 's'))

                    range_clock_samples  = sample_poly_at_interval(clock_state_ant, clock_poly_length, epochs, begin_date)
                    diffs_clock = np.diff(np.array(range_clock_samples).flatten())
                    range_clock_state_ant = np.zeros_like(phase_clock_state_ant)
                    range_clock_state_ant[0] = range_clock_samples[0]
                    range_clock_state_ant[1:] = diffs_clock
                    for epoch_idx, epoch in enumerate(epochs[1:]):
                        if epoch_idx == 0:
                            N_pt = np.sum(phase_times < epoch)
                        else:
                            N_pt = np.sum(np.bitwise_and(phase_times < epoch, phase_times > epochs[epoch_idx-1]))
                        if N_pt == 0: N_pt = 1 
                        dt = clock_poly_length/N_pt # average cadence of measurements
                        delta_T = baseline_handle.wavelength/dt*clock_poly_length # shift for phase-aliased slope
                        bound_low_pc_ant[idx_start+epoch_idx+1] = range_clock_state_ant[epoch_idx+1] - delta_T*clock_bound
                        bound_high_pc_ant[idx_start+epoch_idx+1] = range_clock_state_ant[epoch_idx+1] + delta_T*clock_bound

            if (np.any(bound_low_pc_ant[idx_start:]>phase_clock_state_ant) or np.any(bound_high_pc_ant[idx_start:] < phase_clock_state_ant))\
                    and clock_poly_length>0:
                state_expanded[phase_clock_idxs][antenna_handle.phase_clock_idxs][idx_start:] = range_clock_state_ant

            bound_low_pc = np.append(bound_low_pc, bound_low_pc_ant)
            bound_high_pc = np.append(bound_high_pc, bound_high_pc_ant)

            antenna_phase_clock_idxs = slice(n_clock_phase, n_clock_phase+nclock_phase_antenna+1)
            n_clock_phase = n_clock_phase + nclock_phase_antenna+1
    bound_low = np.concatenate((bound_low_rxpos, bound_low_clock, bound_low_ao, bound_low_grav, bound_low_trop, bound_low_disb, bound_low_pc, bound_low_phase_disb))
    bound_high = np.concatenate((bound_high_rxpos, bound_high_clock, bound_high_ao, bound_high_grav, bound_high_trop, bound_high_disb, bound_high_pc, bound_high_phase_disb))
    if amb_state_idxs is not None:
        bound_low = np.append(bound_low, bound_low_amb)
        bound_high = np.append(bound_high, bound_high_amb)

    return bound_low, bound_high, state_expanded

def analyze_ls_solution(sol_type, plot_results, ref_antenna, clock_idxs, trop_idxs, disb_idxs, ls_sol, store_handle, antenna_handles, sol_name, baselines,\
                        n_ao_state, baseline_handles=[], phase_delay=False, phase_only=False, phase_clock_idxs=[], phase_disb_idxs=[], use_phase_weights=False, integer_amb=[]):
    """Analyze the least-squares solution, print formal sigmas, plot relevant residuals"""
    if sol_type == 'GNSS':
        label1='pseudorange'
        label2='carrier phase'
    elif sol_type == 'VLBI':
        label1='group delay'
        label2='phase delay'
    else:
        raise ValueError('Unknown solution type ' + sol_type)
    iono_free = store_handle.iono_free
    final_state = ls_sol.x
    clock_states = final_state[clock_idxs]
    trop_states = final_state[trop_idxs]

    residuals = ls_sol.fun  # Residuals of the solution
    if phase_delay is True:
        if phase_only is False:
            residuals_range, residuals_phase = get_residuals(residuals, baseline_handles, phase_delay, phase_only) 
            residuals_obs = np.concatenate((residuals_range,residuals_phase))
        else:
            residuals_obs = get_residuals(residuals, baseline_handles, phase_delay, phase_only) 
    else:
        residuals_obs = residuals 

    #unit_var = get_unit_var_full(residuals_obs, final_state)
    unit_var = get_unit_var_full(ls_sol.fun, final_state)
    #weights = get_weights(store_handle, antenna_handles, baselines, baseline_handles, phase_delay, phase_only, use_phase_weights)
    
    J = ls_sol.jac  # Jacobian of the solution
    #J = J[:len(residuals_obs),:]
    if np.linalg.cond(J.T@ J) > 1e9:
        cov_matrix_full = pinv(J.T @ J)
    else:
        cov_matrix_full = np.linalg.inv(J.T @ J)

    cov_matrix_full *= unit_var

    if phase_only is False: cov_matrix_full_clock = cov_matrix_full[clock_idxs,clock_idxs]
    if len(trop_states)>0:
        cov_matrix_trop = cov_matrix_full[trop_idxs,trop_idxs]
    if phase_delay is True: 
        phase_clock_states = final_state[phase_clock_idxs]
        cov_matrix_full_phase_clock = cov_matrix_full[phase_clock_idxs,phase_clock_idxs]

    if store_handle.estimate_disb:
        if phase_only is False:
            disb_states = final_state[disb_idxs]
            cov_disb = cov_matrix_full[disb_idxs,disb_idxs]
    if store_handle.estimate_phase_disb:
        if phase_delay is True:
            phase_disb_states = final_state[phase_disb_idxs]
            cov_phase_disb = cov_matrix_full[phase_disb_idxs,phase_disb_idxs]

    count=0
    count_ao = 0
    count_grav = 0
    print('\n')
    print('For solution ' + sol_name +':')
    print("Condition number: {:.2e}".format(np.linalg.cond(ls_sol.jac.T@ls_sol.jac)))
    for idx, antenna_handle in enumerate(antenna_handles):
        times_gps = antenna_handle.times_gps
        time_deltas_full = (times_gps - times_gps[0])/np.timedelta64(1, 's')
        full_time_hr = np.round(time_deltas_full[-1]/3600) 
        interval_hr = int(np.ceil(full_time_hr/8))       
        #  plot elevation and azimuth 
        plt.figure(figsize=(10, 6))
        plt.plot(times_gps, antenna_handle.elev_arr, marker='x', linestyle='None', color='b', label='elevation') 
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=interval_hr))  # Adjust interval as needed
        plt.gcf().autofmt_xdate()  # Auto-rotate date labels
        plt.title('source geometry (' + antenna_handle.antenna_name + ')')
        plt.ylabel('elevation (deg)')
        plt.grid(True)
        if iono_free is True:
            plt.savefig(sol_type+'_'+sol_name+'_elev_' + antenna_handle.antenna_name + 'ionofree.png')
        else:
            plt.savefig(sol_type+'_'+sol_name+'_elev_' + antenna_handle.antenna_name + '.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(times_gps, antenna_handle.azim_arr, marker='+', linestyle='None', color='r', label='azimuth') 
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=interval_hr))  # Adjust interval as needed
        plt.gcf().autofmt_xdate()  # Auto-rotate date labels
        plt.title('source geometry (' + antenna_handle.antenna_name + ')')
        plt.ylabel('azimuth (deg)')
        plt.grid(True)
        if iono_free is True:
            plt.savefig(sol_type+'_'+sol_name+'_azim_'+ antenna_handle.antenna_name + 'ionofree.png')
        else:
            plt.savefig(sol_type+'_'+sol_name+'_azim_'+ antenna_handle.antenna_name + '.png')
        plt.close()

        if antenna_handle.estimate_ao is True:
            ao_ant = final_state[clock_idxs.stop + count_ao]
            print(antenna_handle.antenna_name + ' (reference antenna):') 
            print('axis offset: ' + str(np.round(ao_ant,6)) + ' m')
            cov_ao = cov_matrix_full[clock_idxs.stop+count_ao,clock_idxs.stop+count_ao]  # un-scaled covariance matrix
            print('sigma_ao: ' + str(np.round(np.sqrt(cov_ao)),6))
            count_ao += 1
        if antenna_handle.estimate_grav_def is True:
            #grav_ant_sin = final_state[clock_idxs.stop + n_ao_state + count_grav]
            #grav_ant_cos = final_state[clock_idxs.stop + n_ao_state + count_grav+1]
            #print(antenna_handle.antenna_name + ' (reference antenna):') 
            #print('grav. def. sin: ' + str(np.round(grav_ant_sin,6)) + ' m')
            #print('grav. def. cos: ' + str(np.round(grav_ant_cos,6)) + ' m')
            #cov_grav_sin = cov_matrix_full[clock_idxs.stop+n_ao_state+count_grav,clock_idxs.stop+n_ao_state+count_grav] 
            #cov_grav_cos = cov_matrix_full[clock_idxs.stop+n_ao_state+count_grav+1,clock_idxs.stop+n_ao_state+count_grav+1] 
            #print('sigma_grav_sin: ' + str(np.round(np.sqrt(cov_grav_sin),6)))
            #print('sigma_grav_cos: ' + str(np.round(np.sqrt(cov_grav_cos),6)))
            grav_ant_e = final_state[clock_idxs.stop + n_ao_state + count_grav]
            grav_ant_e2 = final_state[clock_idxs.stop + n_ao_state + count_grav+1]
            print(antenna_handle.antenna_name + ' (reference antenna):') 
            print('grav. def. e: ' + str(np.round(grav_ant_e,6)) + ' m')
            print('grav. def. e^2: ' + str(np.round(grav_ant_e2,6)) + ' m')
            cov_grav_e = cov_matrix_full[clock_idxs.stop+n_ao_state+count_grav,clock_idxs.stop+n_ao_state+count_grav] 
            cov_grav_e2 = cov_matrix_full[clock_idxs.stop+n_ao_state+count_grav+1,clock_idxs.stop+n_ao_state+count_grav+1] 
            print('sigma_grav_e: ' + str(np.round(np.sqrt(cov_grav_e),6)))
            print('sigma_grav_e^2: ' + str(np.round(np.sqrt(cov_grav_e2),6)))
            count_grav += 2

            # plot gravitational deformation
            #grav_measured = grav_ant_sin*np.sin(np.deg2rad(antenna_handle.elev_arr))\
            #        + grav_ant_cos*np.cos(np.deg2rad(antenna_handle.elev_arr))
            grav_measured = grav_ant_e*np.deg2rad(antenna_handle.elev_arr)\
                    + grav_ant_e2*np.deg2rad(antenna_handle.elev_arr)**2
            elev_model = np.linspace(0,90,100)
            #grav_model = grav_ant_sin*np.sin(np.deg2rad(elev_model))\
            #        + grav_ant_cos*np.cos(np.deg2rad(elev_model))
            grav_model = grav_ant_e*np.deg2rad(elev_model)\
                    + grav_ant_e2*np.deg2rad(elev_model)**2

            plt.figure(figsize=(10, 6))
            plt.plot(elev_model, grav_model*1e3, zorder=1, color='b') 
            plt.scatter(antenna_handle.elev_arr, grav_measured*1e3, marker='x', linestyle='None', zorder=2, color='r') 
            plt.xlabel('elevation (deg)')
            plt.ylabel('delay due to grav. def. (mm)')
            plt.grid(True)
            plt.savefig(sol_type+'_'+sol_name+'_grav_def_'+antenna_handle.antenna_name+'.png')
            plt.close()


        if ref_antenna == antenna_handle.antenna_name: # compute corrected ranges once
            ref_idx = idx
            continue
        else:
            # get elements of state vector for this antenna
            rxpos_state = final_state[count*3:count*3+3]
            clock_state = clock_states[antenna_handle.range_clock_idxs]
            if phase_delay is True:
                phase_clock_state = phase_clock_states[antenna_handle.phase_clock_idxs]
            print(antenna_handle.antenna_name + ':') 
            print('initial pos: x: ' + str(antenna_handle.ref_pos[0]) + \
                          str(' y: ') + str(antenna_handle.ref_pos[1]) + \
                          str(' z: ') + str(antenna_handle.ref_pos[2]))
            print('final pos: x: ' + str(np.round(rxpos_state[0],6)) + \
                    str(' y: ') + str(np.round(rxpos_state[1],6)) + str(' z: ') + str(np.round(rxpos_state[2],6)))

            pos_diff = rxpos_state - antenna_handle.ref_pos
            pos_diff_NEU = antenna_handle.R_mat.T@pos_diff
            print('position difference (XYZ): x: ' + str(np.round(pos_diff[0],6)) + str(' y: ') + \
                    str(np.round(pos_diff[1],6)) + str(' z: ') + str(np.round(pos_diff[2],6)))
            print('position difference (NEU): N: ' + str(np.round(pos_diff_NEU[0],6)) + str(' E: ') + \
                    str(np.round(pos_diff_NEU[1],6)) + str(' U: ') + str(np.round(pos_diff_NEU[2],6)))
            
            cov_matrix_rxpos = cov_matrix_full[count*3:count*3+3,count*3:count*3+3]
            if phase_only is False: 
                cov_matrix_clock = cov_matrix_full_clock[antenna_handle.range_clock_idxs,\
                                                     antenna_handle.range_clock_idxs]  
                sigmas_clock = np.sqrt(np.diag(cov_matrix_clock))

            if phase_delay is True:
                cov_matrix_phase_clock = cov_matrix_full_phase_clock[antenna_handle.phase_clock_idxs,\
                                                      antenna_handle.phase_clock_idxs]  
                sigmas_phase_clock = np.sqrt(np.diag(cov_matrix_phase_clock))

            sigmas_rxpos = np.sqrt(np.diag(cov_matrix_rxpos))
            cov_NEU = antenna_handle.R_mat.T@cov_matrix_rxpos@antenna_handle.R_mat
            sigmas_NEU = np.sqrt(np.diag(cov_NEU))
                 
            print('formal pos. errors: sigma_xx: ' + str(np.round(sigmas_rxpos[0],6)) +  \
                    ' sigma_yy: ' + str(np.round(sigmas_rxpos[1],6)) +  ' sigma_zz: ' + str(np.round(sigmas_rxpos[2],6)))
            print('formal pos. errors (NEU): sigma_NN: ' + str(np.round(sigmas_NEU[0],6)) +  \
                    ' sigma_EE: ' + str(np.round(sigmas_NEU[1],6)) +  ' sigma_UU: ' + str(np.round(sigmas_NEU[2],6)))

            if phase_only is False: 
                print('bulk clock offset (range): ' + str(np.round(antenna_handle.clock_samples[0]*1e9/const.c,6)) + ' ns')
                print('sigma_clock (range): ' + str(np.round(sigmas_clock[0],6)))
            if phase_delay is True: 
                print('bulk clock offset (phase): ' + str(np.round(antenna_handle.phase_clock_samples[0]*1e9/const.c,6)) + ' ns')
                print('sigma_clock (phase): ' + str(np.round(sigmas_phase_clock[0],6)))
            if len(trop_states)>0 and antenna_handle.estimate_trop is True:
                trop_state = trop_states[antenna_handle.trop_idxs]
                cov_trop = cov_matrix_trop[antenna_handle.trop_idxs,antenna_handle.trop_idxs]
                sigmas_trop = np.sqrt(np.diag(cov_trop))
                print('bulk dZWD (m): ' + str(np.round(trop_state[0],6)))
                print('sigma trop (m): ' + str(np.round(sigmas_trop[0],6)))

            if store_handle.estimate_disb is True:
                print('ref system is ' + store_handle.ref_system)
                if phase_only is False:
                    disb_ant = disb_states[antenna_handle.range_disb_idxs]
                    sigmas_disb_ant = np.sqrt(np.diag(cov_disb))[antenna_handle.range_disb_idxs]
                    for idx, system in enumerate(store_handle.disb_systems):
                        if system == 'G':
                            print('GPS code DISB: ' + str(np.round(disb_ant[idx],4)) + ' m')
                            print('GPS code DISB uncertainty: ' + str(np.round(sigmas_disb_ant[idx],4)) + ' m')
                        elif system == 'E':
                            print('Galileo code DISB: ' + str(np.round(disb_ant[idx],4)) + ' m')
                            print('Galileo code DISB uncertainty: ' + str(np.round(sigmas_disb_ant[idx],4)) + ' m')
                        if system == 'C':
                            print('BeiDou code DISB: ' + str(np.round(disb_ant[idx],4)) + ' m')
                            print('BeiDou code DISB uncertainty: ' + str(np.round(sigmas_disb_ant[idx],4)) + ' m')
            if store_handle.estimate_phase_disb is True:
                if phase_delay is True:
                    phase_disb_ant = phase_disb_states[antenna_handle.phase_disb_idxs]
                    sigmas_phase_disb_ant = np.sqrt(np.diag(cov_phase_disb))[antenna_handle.phase_disb_idxs]
                    for idx, system in enumerate(store_handle.disb_systems):
                        if system == 'G':
                            print('GPS phase DISB: ' + str(np.round(phase_disb_ant[idx],4)) + ' m')
                            print('GPS phase DISB uncertainty: ' + str(np.round(sigmas_phase_disb_ant[idx],4)) + ' m')
                        elif system == 'E':
                            print('Galileo phase DISB: ' + str(np.round(phase_disb_ant[idx],4)) + ' m')
                            print('Galileo phase DISB uncertainty: ' + str(np.round(sigmas_phase_disb_ant[idx],4)) + ' m')
                        if system == 'C':
                            print('BeiDou phase DISB: ' + str(np.round(phase_disb_ant[idx],4)) + ' m')
                            print('BeiDou phase DISB uncertainty: ' + str(np.round(sigmas_phase_disb_ant[idx],4)) + ' m')

            count+=1

    if len(baseline_handles)>0:
        if phase_delay is True and phase_only is False:
            residuals_range, residuals_phase = get_residuals(residuals, baseline_handles,phase_delay, phase_only) 
            residuals_cs=np.concatenate((residuals_range,residuals_phase))
        else:
            residuals_cs = get_residuals(residuals, baseline_handles, phase_delay, phase_only) 
        #chi_sq = np.sum(residuals_cs**2)/(len(residuals_cs)-len(ls_sol.x))
        chi_sq = np.sum(ls_sol.fun**2)/(len(ls_sol.fun)-len(ls_sol.x))
        print(f'chi-squared: {chi_sq:.3f}')

    if plot_results is False:
        return
     
    num_samples = 0
    for jdx, baseline in enumerate(baselines): # generate differential measurements on the baselines
        antenna1_handle = antenna_handles[baseline[0]]
        antenna2_handle = antenna_handles[baseline[1]]      
        print('For baseline ' + antenna2_handle.antenna_name + '--' + antenna1_handle.antenna_name)

        if len(baseline_handles) == 0:
            _, ant1_idxs, ant2_idxs = np.intersect1d(antenna1_handle.times_gps, \
                    antenna2_handle.times_gps, return_indices=True)
        else:
            baseline_handle = baseline_handles[jdx]
            _, ant1_idxs, _ = np.intersect1d(antenna1_handle.times_gps, \
                    baseline_handle.datetime_array, return_indices=True)
            _, ant2_idxs, _ = np.intersect1d(antenna2_handle.times_gps, \
                    baseline_handle.datetime_array, return_indices=True)

        if phase_only is False:
            if len(baseline_handles) == 0 or store_handle.stochastic_clock is False:
                diff_clock = antenna2_handle.clock_samples[ant2_idxs ] - \
                              antenna1_handle.clock_samples[ant1_idxs]
            else:
                _, ant1_idxs_clock, ant1_dt = np.intersect1d(antenna1_handle.clock_times, \
                        baseline_handle.datetime_array, return_indices=True)
                _, ant2_idxs_clock, ant2_dt = np.intersect1d(antenna2_handle.clock_times, \
                        baseline_handle.datetime_array, return_indices=True)
                diff_clock = np.zeros(len(baseline_handle.datetime_array))
                diff_clock[ant2_dt] += antenna2_handle.clock_samples[ant2_idxs_clock]
                diff_clock[ant1_dt] -= antenna1_handle.clock_samples[ant1_idxs_clock]

        if phase_delay is True:
            if store_handle.stochastic_clock is False:
                diff_clock_phase = antenna2_handle.phase_clock_samples[ant2_idxs] - \
                              antenna1_handle.phase_clock_samples[ant1_idxs]
            else:
                _, ant1_idxs_phase, ant1_dt = np.intersect1d(antenna1_handle.phase_clock_times, \
                        baseline_handle.datetime_array, return_indices=True)
                _, ant2_idxs_phase, ant2_dt = np.intersect1d(antenna2_handle.phase_clock_times, \
                        baseline_handle.datetime_array, return_indices=True)
                diff_clock_phase = np.zeros(len(baseline_handle.datetime_array))
                diff_clock_phase[ant2_dt] += antenna2_handle.phase_clock_samples[ant2_idxs_phase]
                diff_clock_phase[ant1_dt] -= antenna1_handle.phase_clock_samples[ant1_idxs_phase]

        if len(trop_states)>0:
            if len(baseline_handles) == 0 or store_handle.stochastic_trop is False:
                diff_trop = antenna2_handle.trop_samples[ant2_idxs ] - \
                              antenna1_handle.trop_samples[ant1_idxs]
            else:
                _, ant1_idxs_trop, ant1_dt = np.intersect1d(antenna1_handle.trop_times, \
                        baseline_handle.datetime_array, return_indices=True)
                _, ant2_idxs_trop, ant2_dt = np.intersect1d(antenna2_handle.trop_times, \
                        baseline_handle.datetime_array, return_indices=True)

                diff_trop = np.zeros(len(baseline_handle.datetime_array))
                diff_trop[ant2_dt] += antenna2_handle.trop_samples[ant2_idxs_trop]
                diff_trop[ant1_dt] -= antenna1_handle.trop_samples[ant1_idxs_trop]

        if phase_delay is False: # part of function residuals belong to phase residuals
            if len(baseline_handles) == 0:
                times_gps = np.intersect1d(antenna1_handle.times_gps, antenna2_handle.times_gps)
                residuals_range = residuals[num_samples:num_samples+len(times_gps)]
                num_samples = num_samples + len(times_gps) 
                range_weight_mat = get_obs_weights(store_handle, antenna1_handle, antenna2_handle, 'range')
                residuals_range = np.linalg.inv(range_weight_mat)@residuals_range
            else:
                times_gps = baseline_handle.datetime_array[baseline_handle.range_data_idxs]
                residuals_range = residuals[num_samples:num_samples+len(baseline_handle.range_data_idxs)]
                diff_clock = diff_clock[baseline_handle.range_data_idxs]
                num_samples = num_samples + len(baseline_handle.range_data_idxs) 
                range_weight_mat = get_obs_weights(store_handle, antenna1_handle, antenna2_handle, 'range',\
                            baseline_handle.range_data_idxs, baseline_handle)
                residuals_range = np.linalg.inv(range_weight_mat)@residuals_range
                if baseline_handle.use_cov_kernel_range is False:
                    print(f'q_range (m): {baseline_handle.q_range:.5f}')
                    print(f'q_range (ps): {baseline_handle.q_range/const.c*1e12:.5f}')
            dt_range = to_datetime(times_gps)
            dc_range = diff_clock
        elif phase_only is False:
            baseline_handle = baseline_handles[jdx]
            times_gps = baseline_handle.datetime_array
            residuals_range = residuals[num_samples:\
                                     num_samples + len(baseline_handle.range_data_idxs)]
            num_samples = num_samples + len(baseline_handle.range_data_idxs) 
            residuals_phase = residuals[num_samples:\
                                     num_samples + len(baseline_handle.phase_data_idxs)]
            num_samples = num_samples + len(baseline_handle.phase_data_idxs)
            
            range_weight_mat = get_obs_weights(store_handle, antenna1_handle, antenna2_handle, 'range', \
                    baseline_handle.range_data_idxs, baseline_handle)
            phase_weight_mat = get_obs_weights(store_handle, antenna1_handle, antenna2_handle, 'phase', \
                    baseline_handle.phase_data_idxs, baseline_handle, use_phase_weights)
            residuals_range = np.linalg.inv(range_weight_mat)@residuals_range
            residuals_phase = np.linalg.inv(phase_weight_mat)@residuals_phase
            
            dt_range = to_datetime(times_gps[baseline_handle.range_data_idxs])
            dc_range = diff_clock[baseline_handle.range_data_idxs]
            if baseline_handle.use_cov_kernel_range is False:
                print(f'q_range (m): {baseline_handle.q_range:.5f}')
                print(f'q_range (ps): {baseline_handle.q_range/const.c*1e12:.5f}')

            if baseline_handle.use_cov_kernel_phase is False:
                print(f'q_phase (m): {baseline_handle.q_phase:.5f}')
                print(f'q_phase (ps): {baseline_handle.q_phase/const.c*1e12:.5f}')
        else:
            baseline_handle = baseline_handles[jdx]
            times_gps = baseline_handle.datetime_array
            residuals_phase = residuals[num_samples:\
                                     num_samples + len(baseline_handle.phase_data_idxs)]
            num_samples = num_samples + len(baseline_handle.phase_data_idxs)

            phase_weight_mat = get_obs_weights(store_handle, antenna1_handle, antenna2_handle, 'phase', baseline_handle.phase_data_idxs, baseline_handle, use_phase_weights)
            residuals_phase = np.linalg.inv(phase_weight_mat)@residuals_phase
            if baseline_handle.use_cov_kernel_phase is False:
                print(f'q_phase (m): {baseline_handle.q_phase:.5f}')
                print(f'q_phase (ps): {baseline_handle.q_phase/const.c*1e12:.5f}')
        
        time_deltas_full = (times_gps - times_gps[0])/np.timedelta64(1, 's')
        full_time_hr = np.round(time_deltas_full[-1]/3600) 
        interval_hr = int(np.ceil(full_time_hr/8))       
        
        # print baseline-specific quantities
        #if baseline[0] < ref_idx:
        #    count_1 = baseline[0]
        #else:
        #    count_1 = baseline[0]-1
        #if baseline[1] < ref_idx:
        #    count_2 = baseline[1]
        #else:
        #    count_2 = baseline[1]-1

        #if antenna2_handle.antenna_name == ref_antenna:
        #    R_mat = antenna2_handle.R_mat.T
        #    var_r2 = np.zeros((3,3))
        #    cov_r1_r2 = np.zeros((3,3))
        #    r2 = antenna2_handle.ref_pos
        #    r1 = final_state[count_1*3:count_1*3+3]
        #    var_r1 = cov_matrix_full[count_1*3:count_1*3+3,count_1*3:count_1*3+3]
        #else:
        #    r2 = final_state[count_2*3:count_2*3+3]
        #    R_mat = antenna1_handle.R_mat.T
        #    var_r2 = cov_matrix_full[count_2*3:count_2*3+3,count_2*3:count_2*3+3]
        #    if antenna1_handle.antenna_name == ref_antenna:
        #        r1 = antenna1_handle.ref_pos
        #        var_r1 = np.zeros((3,3))
        #        cov_r1_r2 = np.zeros((3,3))
        #    else:
        #        r1 = final_state[count_1*3:count_1*3+3]
        #        var_r1 = cov_matrix_full[count_1*3:count_1*3+3,count_1*3:count_1*3+3]
        #        cov_r1_r2 = cov_matrix_full[count_1*3:count_1*3+3,count_2*3:count_2*3+3]
        #b_12 = r2-r1
        #b_12_NEU = R_mat@b_12
        #var_b = var_r1 + var_r2 - 2*cov_r1_r2
        #var_b_NEU = R_mat@var_b@R_mat.T
        #sigmas_b = np.sqrt(np.diag(var_b))
        #sigmas_b_NEU = np.sqrt(np.diag(var_b_NEU))
        #L_mag = np.linalg.norm(b_12)
        #sig_L = 1/L_mag*np.sqrt(b_12[0]**2*var_b[0,0] + b_12[1]**2*var_b[1,1] + b_12[2]**2*var_b[2,2]\
        #        + 2*b_12[0]*b_12[1]*var_b[0,1] + 2*b_12[0]*b_12[2]*var_b[0,2] + 2*b_12[1]*b_12[2]*var_b[1,2])
        #print(f"baseline length (m): {L_mag:.4f}")
        #print(f"sigma_LL: {sig_L:.5f}")
        #print(f"baseline vector (XYZ, m): {b_12[0]:.4f}, {b_12[1]:.4f}, {b_12[2]:.4f}")
        #print('formal baseline errors (XYZ, m): sigma_xx: ' + str(np.round(sigmas_b[0],6)) +  \
        #            ' sigma_yy: ' + str(np.round(sigmas_b[1],6)) +  ' sigma_zz: ' + str(np.round(sigmas_b[2],6)))
        #print(f"baseline vector (NEU, m): {b_12_NEU[0]:.4f}, {b_12_NEU[1]:.4f}, {b_12_NEU[2]:.4f}")
        #print('formal baseline errors (NEU, m): sigma_NN: ' + str(np.round(sigmas_NEU[0],6)) +  \
        #            ' sigma_EE: ' + str(np.round(sigmas_b_NEU[1],6)) +  ' sigma_UU: ' + str(np.round(sigmas_b_NEU[2],6)))
        
        source_array = np.array([store_handle.source_time_dict[time] for time in times_gps])
        if phase_only is False:
            # Create a DataFrame
            data = DataFrame({
                'Datetime': dt_range,
                'Residuals': residuals_range,
                'Clock': dc_range})
            
            if len(baseline_handles)>0:
                elev_arr = antenna2_handle.elev_arr[ant2_idxs][baseline_handle.range_data_idxs]
                azim_arr = antenna2_handle.azim_arr[ant2_idxs][baseline_handle.range_data_idxs]
            else:
                elev_arr = antenna2_handle.elev_arr[ant2_idxs]
                azim_arr = antenna2_handle.azim_arr[ant2_idxs]
            # plot measurement residuals
            plt.figure(figsize=(10, 6))
            plt.plot(elev_arr, data['Residuals'].to_numpy(), marker='x', linestyle='None', color='b') 
            
            plt.title('elevation ' + label1+ ' residuals (' + antenna2_handle.antenna_name + '—' + antenna1_handle.antenna_name + ')')
            plt.xlabel('elevation (deg)')
            plt.ylabel('meas. residuals (m)')
            plt.grid(True)
            if iono_free is True:
                plt.savefig(sol_type+'_'+sol_name+'_elev_residuals_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + 'ionofree.png')
            else:
                plt.savefig(sol_type+'_'+sol_name+'_elev_residuals_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '.png')
            plt.close()

            # Set the datetime as the index
            data.set_index('Datetime', inplace=True)
            index_array = data.index.to_numpy() 

            # plot measurement residuals
            plt.figure(figsize=(10, 6))
            plt.plot(index_array, data['Residuals'].to_numpy(), marker='x', linestyle='None', color='b')
            
            # Formatting the date on the x-axis
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=interval_hr))  # Adjust interval as needed
            plt.gcf().autofmt_xdate()  # Auto-rotate date labels
            
            plt.title('final '+label1+' residuals (' + antenna2_handle.antenna_name + '—' + antenna1_handle.antenna_name + ')')
            plt.ylabel('meas. residuals (m)')
            plt.grid(True)
            if iono_free is True:
                plt.savefig(sol_type+'_'+sol_name+'_meas_residuals_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + 'ionofree.png')
            else:
                plt.savefig(sol_type+'_'+sol_name+'_meas_residuals_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '.png')
            plt.close()

            # plot clock residuals
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.scatter(index_array, (data['Clock'].to_numpy() + data['Residuals'].to_numpy())/const.c*1e12, marker='.', \
                    color='g', label='res. + clock', zorder=1)
            ax1.plot(index_array, data['Clock'].to_numpy()/const.c*1e12, linestyle='--', color='slategray', label='clock function', zorder=0)
            ax1.legend()

            # Formatting the date on the x-axis
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            ax1.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hr))  # Adjust interval as needed
            fig.autofmt_xdate()  # Auto-rotate date labels

            WRMS = np.sqrt(np.sum((range_weight_mat@data['Residuals'].to_numpy())**2)/np.trace(range_weight_mat@range_weight_mat.T))
            WRMS = WRMS/const.c*1e12
            #data_ps = data['Residuals'].to_numpy()/const.c*1e12
            #WRMS = np.sqrt(np.mean(data_ps**2))

            ax1.set_title('postfit residuals + clock function (' + antenna2_handle.antenna_name + '—' +\
                antenna1_handle.antenna_name + ')'+ ' WRMS: ' + str(np.round(WRMS,decimals=1))+ ' ps')
            #ax1.set_xlabel('date + hr')
            ax1.set_ylabel('range + clock function (ps)')
            ax1.grid(True)

            if iono_free is True:
                fig.savefig(sol_type+'_'+sol_name+'_clock_residuals_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '_ionofree.png')
            else:
                fig.savefig(sol_type+'_'+sol_name+'_clock_residuals_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '.png')
            plt.close(fig)

            ## plot clock residuals
            #fig, ax1 = plt.subplots(figsize=(10, 6))
            #data_full = (data['Clock'].to_numpy() + data['Residuals'].to_numpy())/const.c*1e12
            #data_clock = data['Clock'].to_numpy()/const.c*1e12
            #time_array = (index_array-index_array[0])/np.timedelta64(1, 's')
            #res = linregress(time_array,data_clock)
            #best_fit_line = time_array*res.slope + res.intercept
            #data_full_var = data_full - best_fit_line
            #data_clock_var = data_clock - best_fit_line
            #ax1.scatter(index_array, data_full_var, marker='o', \
            #        color='g', label='res. + clock', zorder=1)
            #ax1.plot(index_array, data_clock_var, linestyle='--', color='slategray', label='clock variation', zorder=0)
            #ax1.legend()

            ## Formatting the date on the x-axis
            #ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            #ax1.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hr))  # Adjust interval as needed
            #fig.autofmt_xdate()  # Auto-rotate date labels
            #
            #ax1.set_title('postfit residuals + clock variation (' + antenna2_handle.antenna_name + '—' + antenna1_handle.antenna_name + ')'\
            #        + ' WRMS: ' + str(np.round(WRMS,decimals=1))+ ' ps')
            ##ax1.set_xlabel('date + hr')
            #ax1.set_ylabel('range + clock function (ps)')
            #ax1.grid(True)

            #if iono_free is True:
            #    fig.savefig(sol_type+'_'+sol_name+'_clock_variation_residuals_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '_ionofree.png')
            #else:
            #    fig.savefig(sol_type+'_'+sol_name+'_clock_variation_residuals_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '.png')
            #plt.close(fig)

            # plot gr_del residuals by source
            #fig, ax = plt.subplots(figsize=(10, 6))
            #for src in np.unique(source_array):
            #    # get epochs of source
            #    idxs_src = []
            #    for idx, time in enumerate(times_gps):
            #        if src == store_handle.source_time_dict[time]:
            #            idxs_src.append(idx)
            #    idx_src = np.array(idxs_src)
            #    if phase_delay is True:
            #        idxs_src, _, range_idxs = np.intersect1d(idxs_src, baseline_handle.range_data_idxs, return_indices=True)
            #        ax.plot(time_deltas_full[idxs_src], residuals_range[range_idxs], marker='x', linestyle='None', label=src)
            #    else:
            #        ax.plot(time_deltas_full[idxs_src], residuals_range[idxs_src], marker='x', linestyle='None', label=src)
            #ax.set_title('by-source ' +label1+' residuals (' + antenna2_handle.antenna_name + '—' + antenna1_handle.antenna_name + ')')
            #ax.set_xlabel('time (sec)')
            #ax.set_ylabel('meas. residuals (m)')
            ##ax.legend()
            ## Formatting the date on the x-axis
            #if iono_free is True:
            #    fig.savefig('./slip_figs/'+sol_type+'_'+sol_name+'_full_meas_residuals_'+\
            #            antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + 'ionofree.png')
            #else:
            #    fig.savefig('./slip_figs/'+sol_type+'_'+sol_name+'_full_meas_residuals_'+\
            #            antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '.png')       
            #plt.close(fig)

            # plot clock function
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax2 = ax1.twinx()
            ax1.plot(index_array, data['Clock'].to_numpy(), linestyle='-', color='b')

            # Formatting the date on the x-axis
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            ax1.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hr))  # Adjust interval as needed
            fig.autofmt_xdate()  # Auto-rotate date labels
            
            ax1.set_title('diff. clock function (' + antenna2_handle.antenna_name + '—' + antenna1_handle.antenna_name + ')')
            #ax1.set_xlabel('date + hr')
            ax1.set_ylabel('diff. clock (m)')
            ax1.grid(True)

            ax2.plot(index_array, data['Clock'].to_numpy()*1e6/const.c, linestyle='-', color='b')
            ax2.set_ylabel('(microsec)')
            if iono_free is True:
                fig.savefig(sol_type+'_'+sol_name+'_clock_fcn_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '_ionofree.png')
            else:
                fig.savefig(sol_type+'_'+sol_name+'_clock_fcn_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '.png')
            plt.close(fig)

            if len(trop_states)>0 and len(baseline_handles)>0 and phase_only is False:
                # plot dZWD function
                fig, ax1 = plt.subplots(figsize=(10, 6))
                ax1.plot(index_array, diff_trop[baseline_handle.range_data_idxs], linestyle='-', color='b')
                # Formatting the date on the x-axis
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                ax1.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hr))  # Adjust interval as needed
                fig.autofmt_xdate()  # Auto-rotate date labels
                ax1.set_title('differential zenith wet delay (' + antenna2_handle.antenna_name + '—' + antenna1_handle.antenna_name + ')')
                ax1.set_ylabel('dZWD (m)')
                ax1.grid(True)

                if iono_free is True:
                    fig.savefig(sol_type+'_'+sol_name+'_dZWD_fcn_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '_ionofree.png')
                else:
                    fig.savefig(sol_type+'_'+sol_name+'_dZWD_fcn_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '.png')
                plt.close(fig)

            # plot histogram of range postfit residuals
            residuals_range_sorted = np.sort(residuals_range)
            normal_dist = norm.pdf(residuals_range_sorted, np.mean(residuals_range), np.std(residuals_range))
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(residuals_range, bins='fd', density=True)
            ax.plot(residuals_range_sorted, normal_dist, label='normal dist.')
            ax.set_title(label1+' residuals distribution (' + antenna2_handle.antenna_name + '—' + antenna1_handle.antenna_name + ')')
            ax.set_xlabel('meas. residuals (m)')
            ax.set_ylabel('probability density')
            ax.legend()
            # Formatting the date on the x-axis
            if iono_free is True:
                fig.savefig(sol_type+'_'+sol_name+'_hist_meas_residuals_'+\
                        antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + 'ionofree.png')
            else:
                fig.savefig(sol_type+'_'+sol_name+'_hist_meas_residuals_'+\
                        antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '.png')       
            plt.close(fig)

            #for src in np.unique(source_array):
            #    residuals_src = []
            #    # get epochs of source
            #    idxs_src = []
            #    for idx, time in enumerate(times_gps):
            #        if src == store_handle.source_time_dict[time]:
            #            idxs_src.append(idx)
            #    idx_src = np.array(idxs_src)
            #    if len(idxs_src) > 10:
            #        fig, ax = plt.subplots(figsize=(10, 6))
            #        if phase_delay is True:
            #            idxs_src, _, range_idxs = np.intersect1d(idxs_src, baseline_handle.range_data_idxs, return_indices=True)
            #            runs_range_idxs = consecutive_idxs(range_idxs)
            #        else:
            #            runs_range_idxs = consecutive_idxs(idxs_src)

            #        for run_range in runs_range_idxs:
            #            residuals_src.extend(residuals_range[run_range]-np.mean(residuals_range[run_range]))
            #        residuals_src = np.array(residuals_src)
            #        residuals_src_sorted = np.sort(residuals_src)
            #        normal_dist = norm.pdf(residuals_src_sorted, np.mean(residuals_src), np.std(residuals_src))
            #        ax.hist(residuals_src, bins='fd', density=True)
            #        ax.plot(residuals_src_sorted, normal_dist, label='normal dist.')
            #        ax.set_title('source ' + src + ' ' + label1+ ' distribution (' + \
            #                antenna2_handle.antenna_name + '—' + antenna1_handle.antenna_name + ')')
            #        ax.set_ylabel('probability density')
            #        ax.set_xlabel('meas. residuals (m)')
            #        ax.legend()
            #        # Formatting the date on the x-axis
            #        if iono_free is True:
            #            fig.savefig(sol_type+'_'+sol_name+'_hist_meas_residuals_'+src+'_'+\
            #                    antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + 'ionofree.png')
            #        else:
            #            fig.savefig(sol_type+'_'+sol_name+'_hist_meas_residuals_'+src+'_'+\
            #                    antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '.png')       
            #        plt.close(fig)

        if phase_delay is True: # plot phase residuals
            dt_phase = baseline_handle.datetime_array[baseline_handle.phase_data_idxs]
            diff_clock_phase = diff_clock_phase[baseline_handle.phase_data_idxs]

            data_phase = DataFrame({
                'Datetime': dt_phase,
                'Residuals': residuals_phase,
                'Clock': diff_clock_phase})
            data_phase.set_index('Datetime', inplace=True)
             
            phase_index_array = data_phase.index.to_numpy()
            # plot measurement residuals
            plt.figure(figsize=(10, 6))
            plt.plot(phase_index_array, data_phase['Residuals'].to_numpy(), marker='x', linestyle='None', color='b')         

            # Formatting the date on the x-axis
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=interval_hr))  # Adjust interval as needed
            plt.gcf().autofmt_xdate()  # Auto-rotate date labels
            
            plt.title('final '+ label2+' residuals (' + antenna2_handle.antenna_name + '—' + antenna1_handle.antenna_name + ')')
            #plt.xlabel('date + hr')
            plt.ylabel('meas. residuals (m)')
            plt.grid(True)
            if iono_free is True:
                plt.savefig(sol_type+'_'+sol_name+'_phase_meas_residuals_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + 'ionofree.png')
            else:
                plt.savefig(sol_type+'_'+sol_name+'_phase_meas_residuals_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '.png')
            plt.close()

            # plot elevation residuals
            plt.figure(figsize=(10, 6))
            elev_arr = antenna1_handle.elev_arr[ant1_idxs][baseline_handle.phase_data_idxs]
            plt.plot(elev_arr, data_phase['Residuals'].to_numpy(), marker='x', linestyle='None', color='b') 
            plt.title('elevation ' + label2+ ' residuals (' + antenna2_handle.antenna_name + '—' + antenna1_handle.antenna_name + ')')
            plt.xlabel('elevation (deg)')
            plt.ylabel('meas. residuals (m)')
            plt.grid(True)
            if iono_free is True:
                plt.savefig(sol_type+'_'+sol_name+'_phase_elev_residuals_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + 'ionofree.png')
            else:
                plt.savefig(sol_type+'_'+sol_name+'_phase_elev_residuals_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '.png')
            plt.close()

            # plot clock function
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax2 = ax1.twinx()
            ax1.plot(phase_index_array, data_phase['Clock'].to_numpy(), linestyle='-', color='b')

            # Formatting the date on the x-axis
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            ax1.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hr))  # Adjust interval as needed
            fig.autofmt_xdate()  # Auto-rotate date labels
            
            ax1.set_title('diff. clock function (' + antenna2_handle.antenna_name + '—' + antenna1_handle.antenna_name + ')')
            #ax1.set_xlabel('date + hr')
            ax1.set_ylabel('diff. clock (m)')
            ax1.grid(True)

            ax2.plot(phase_index_array, data_phase['Clock'].to_numpy()*1e6/const.c, linestyle='-', color='b')
            ax2.set_ylabel('(microsec)')
            if iono_free is True:
                fig.savefig(sol_type+'_'+sol_name+'_phase_clock_fcn_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '_ionofree.png')
            else:
                fig.savefig(sol_type+'_'+sol_name+'_phase_clock_fcn_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '.png')
            plt.close(fig)

            if len(trop_states)>0 and phase_only is True:
                # plot dZWD function
                fig, ax1 = plt.subplots(figsize=(10, 6))
                ax1.plot(phase_index_array, diff_trop[baseline_handle.phase_data_idxs], linestyle='-', color='b')
                # Formatting the date on the x-axis
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                ax1.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hr))  # Adjust interval as needed
                fig.autofmt_xdate()  # Auto-rotate date labels
                ax1.set_title('differential zenith wet delay (' + antenna2_handle.antenna_name + '—' + antenna1_handle.antenna_name + ')')
                ax1.set_ylabel('dZWD (m)')
                ax1.grid(True)

                if iono_free is True:
                    fig.savefig(sol_type+'_'+sol_name+'_dZWD_fcn_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '_ionofree.png')
                else:
                    fig.savefig(sol_type+'_'+sol_name+'_dZWD_fcn_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '.png')
                plt.close(fig)

            # plot clock residuals
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.scatter(phase_index_array, (data_phase['Clock'].to_numpy() + data_phase['Residuals'].to_numpy())/const.c*1e12, marker='o', \
                    color='g', label='res. + clock', zorder=1)
            ax1.plot(phase_index_array, data_phase['Clock'].to_numpy()/const.c*1e12, linestyle='--', color='slategray', label='clock function', zorder=0)
            ax1.legend()
            WRMS_phase = np.sqrt(np.sum((phase_weight_mat@data_phase['Residuals'].to_numpy())**2)/np.trace(phase_weight_mat@phase_weight_mat.T))
            WRMS_phase = WRMS_phase/const.c*1e12
            #WRMS = np.sqrt(np.mean((data_phase['Residuals']/const.c*1e12)**2))

            # Formatting the date on the x-axis
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            ax1.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hr))  # Adjust interval as needed
            fig.autofmt_xdate()  # Auto-rotate date labels
            
            ax1.set_title('postfit residuals + clock function (' + antenna2_handle.antenna_name + '—' + antenna1_handle.antenna_name + ')'\
                    + ' WRMS: ' + str(np.round(WRMS_phase,decimals=1))+ ' ps')
            #ax1.set_xlabel('date + hr')
            ax1.set_ylabel('phase + clock function (ps)')
            ax1.grid(True)

            # residuals txt file
            time_str = phase_index_array.astype('datetime64[s]').astype(str)
            value_str  = np.char.mod('%.6f', data_phase['Residuals'].to_numpy()/const.c*1e12)
            out        = np.column_stack([time_str, value_str])
            np.savetxt("phase_res.txt", out, fmt="%s %s")

            if iono_free is True:
                fig.savefig(sol_type+'_'+sol_name+'_phase_clock_residuals_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '_ionofree.png')
            else:
                fig.savefig(sol_type+'_'+sol_name+'_phase_clock_residuals_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '.png')
            plt.close(fig)

            if phase_only is False:
                # plot clock residuals
                fig, ax1 = plt.subplots(figsize=(10, 6))
                ax1.scatter(phase_index_array, (data_phase['Clock'].to_numpy() + data_phase['Residuals'].to_numpy())/const.c*1e12, marker='o', \
                        label='phase res. + clock', zorder=10)
                ax1.plot(phase_index_array, data_phase['Clock'].to_numpy()/const.c*1e12, linestyle='--', label='phase clock function', zorder=0)

                ax1.scatter(index_array, (data['Clock'].to_numpy() + data['Residuals'].to_numpy())/const.c*1e12, marker='o', label='range res. + clock', zorder=5)
                ax1.plot(index_array, data['Clock'].to_numpy()/const.c*1e12, linestyle='--', label='range clock function', zorder=1)
                ax1.legend()
                WRMS_range = np.sqrt(np.sum((range_weight_mat@data['Residuals'].to_numpy())**2)/np.trace(range_weight_mat@range_weight_mat.T))
                WRMS_range = WRMS_range/const.c*1e12
                #WRMS_range = np.sqrt(np.mean((data['Residuals']/const.c*1e12)**2))

                # Formatting the date on the x-axis
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                ax1.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hr))  # Adjust interval as needed
                fig.autofmt_xdate()  # Auto-rotate date labels
                
                ax1.set_title('postfit residuals + clock function (' + antenna2_handle.antenna_name + '—' + antenna1_handle.antenna_name + ')'\
                        + ' WRMS (range) : ' + str(np.round(WRMS_range,decimals=1))+ ' ps,' + ' WRMS (phase) : ' + str(np.round(WRMS_phase,decimals=1))+ ' ps'   )
                #ax1.set_xlabel('date + hr')
                ax1.set_ylabel('residuals + clock function (ps)')
                ax1.grid(True)

                if iono_free is True:
                    fig.savefig(sol_type+'_'+sol_name+'_full_clock_residuals_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '_ionofree.png')
                else:
                    fig.savefig(sol_type+'_'+sol_name+'_full_clock_residuals_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '.png')
                plt.close(fig)


            # plot clock residuals
            fig, ax1 = plt.subplots(figsize=(10, 6))
            data_phase_full = (data_phase['Clock'].to_numpy() + data_phase['Residuals'].to_numpy())/const.c*1e12
            data_phase_clock = data_phase['Clock'].to_numpy()/const.c*1e12
            time_array = (phase_index_array-phase_index_array[0])/np.timedelta64(1, 's')
            res = linregress(time_array,data_phase_clock)
            best_fit_line = time_array*res.slope + res.intercept
            data_phase_full_var = data_phase_full - best_fit_line
            data_phase_clock_var = data_phase_clock - best_fit_line
            ax1.scatter(phase_index_array, data_phase_full_var, marker='o', \
                    color='g', label='res. + clock', zorder=1)
            ax1.plot(phase_index_array, data_phase_clock_var, linestyle='--', color='slategray', label='clock variation', zorder=0)
            ax1.legend()
            # Formatting the date on the x-axis
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            ax1.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hr))  # Adjust interval as needed
            fig.autofmt_xdate()  # Auto-rotate date labels
            
            ax1.set_title('postfit residuals + clock variation (' + antenna2_handle.antenna_name + '—' + antenna1_handle.antenna_name + ')'\
                    + ' WRMS: ' + str(np.round(WRMS_phase,decimals=1))+ ' ps')
            #ax1.set_xlabel('date + hr')
            ax1.set_ylabel('phase + clock function (ps)')
            ax1.grid(True)

            if iono_free is True:
                fig.savefig(sol_type+'_'+sol_name+'_phase_clock_variation_residuals_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '_ionofree.png')
            else:
                fig.savefig(sol_type+'_'+sol_name+'_phase_clock_variation_residuals_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '.png')
            plt.close(fig)

            #if phase_only is False:
            #    # plot phase delay - group delay
            #    fig, ax1 = plt.subplots(figsize=(10, 6))
            #    _, _, idxs_group = np.intersect1d(baseline_handle.phase_data_idxs, baseline_handle.range_data_idxs, return_indices = True)
            #    data_full_phase = (data_phase['Clock'].to_numpy() + data_phase['Residuals'].to_numpy())/const.c*1e12
            #    data_full_range = (data['Clock'].to_numpy() + data['Residuals'].to_numpy())/const.c*1e12
            #    ax1.scatter(phase_index_array, data_full_range[idxs_group]-data_full_phase, marker='o', \
            #            color='g', label='res. + clock', zorder=1)
            #    # Formatting the date on the x-axis
            #    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            #    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hr))  # Adjust interval as needed
            #    fig.autofmt_xdate()  # Auto-rotate date labels
            #    
            #    ax1.set_title('observables comparison (' + antenna2_handle.antenna_name + '—' + antenna1_handle.antenna_name + ')')
            #    #ax1.set_xlabel('date + hr')
            #    ax1.set_ylabel('range - phase (ps)')
            #    ax1.grid(True)

            #    if iono_free is True:
            #        fig.savefig(sol_type+'_'+sol_name+'_range_phase_comparison_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '_ionofree.png')
            #    else:
            #        fig.savefig(sol_type+'_'+sol_name+'_range_phase_comparison_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '.png')
            #    plt.close(fig)

            #    # plot a posteriori phase delay - (gr del model + phase windup)
            #    if sol_type == 'VLBI':
            #        fig, ax1 = plt.subplots(figsize=(10, 6))
            #        phase_delays = baseline_handle.phase_delays[baseline_handle.phase_data_idxs]/const.c*1e12
            #        grdel_model = np.array(baseline_handle.group_delay_model)[baseline_handle.range_data_idxs][idxs_group]/const.c*1e12
            #        phase_windup_model = (baseline_handle.cpw_2-baseline_handle.cpw_1)[baseline_handle.phase_data_idxs]/const.c*1e12
            #        ax1.scatter(phase_index_array, phase_delays - grdel_model - phase_windup_model, marker='o', \
            #                color='g', zorder=1)
            #        # Formatting the date on the x-axis
            #        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            #        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hr))  # Adjust interval as needed
            #        fig.autofmt_xdate()  # Auto-rotate date labels
            #        
            #        RMS = np.sqrt(np.mean((phase_delays-grdel_model-phase_windup_model)**2))
            #        ax1.set_title('phase delay compatibility (' + antenna2_handle.antenna_name + '—' + antenna1_handle.antenna_name + ')' \
            #                + ' RMS: ' + str(np.round(RMS,decimals=1))+ ' ps')
            #        #ax1.set_xlabel('date + hr')
            #        ax1.set_ylabel('phDel - grDel post. - windup (ps)')
            #        ax1.grid(True)

            #        if iono_free is True:
            #            fig.savefig(sol_type+'_'+sol_name+'_range_posterior_phase_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '_ionofree.png')
            #        else:
            #            fig.savefig(sol_type+'_'+sol_name+'_range_posterior_phase_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '.png')
            #        plt.close(fig)

            #        if True:
            #            elev_arr = antenna2_handle.elev_arr[ant2_idxs][baseline_handle.phase_data_idxs]
            #            azim_arr = antenna2_handle.azim_arr[ant2_idxs][baseline_handle.phase_data_idxs]

            #            fig, ax1 = plt.subplots(figsize=(10, 6))
            #            ax1.scatter(elev_arr, phase_delays - grdel_model - phase_windup_model, marker='o', \
            #                    color='g', zorder=1)
            #            ax1.set_title('phase delay compatibility (' + antenna2_handle.antenna_name + '—' + antenna1_handle.antenna_name + ')' \
            #                    + ' RMS: ' + str(np.round(RMS,decimals=1))+ ' ps')
            #            ax1.set_xlabel('elevation (deg)')
            #            ax1.set_ylabel('phDel - grDel post. - windup (ps)')
            #            ax1.grid(True)

            #            if iono_free is True:
            #                fig.savefig(sol_type+'_'+sol_name+'_range_posterior_elev_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '_ionofree.png')
            #            else:
            #                fig.savefig(sol_type+'_'+sol_name+'_range_posterior_elev_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '.png')
            #            plt.close(fig)

            #            fig, ax1 = plt.subplots(figsize=(10, 6))
            #            ax1.scatter(azim_arr, phase_delays - grdel_model - phase_windup_model, marker='o', \
            #                    color='g', zorder=1)
            #            ax1.set_title('phase delay compatibility (' + antenna2_handle.antenna_name + '—' + antenna1_handle.antenna_name + ')' \
            #                    + ' RMS: ' + str(np.round(RMS,decimals=1))+ ' ps')
            #            ax1.set_xlabel('azimuth (deg)')
            #            ax1.set_ylabel('phDel - grDel post. - windup (ps)')
            #            ax1.grid(True)

            #            if iono_free is True:
            #                fig.savefig(sol_type+'_'+sol_name+'_range_posterior_azim_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '_ionofree.png')
            #            else:
            #                fig.savefig(sol_type+'_'+sol_name+'_range_posterior_azim_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '.png')
            #            plt.close(fig)


                ## plot clock variation residuals
                #fig, ax1 = plt.subplots(figsize=(10, 6))
                #ax1.scatter(phase_index_array, data_phase_full_var, marker='o', \
                #        label='phase res. + clock', zorder=10)
                #ax1.plot(phase_index_array, data_phase_clock_var, linestyle='--', label='phase clock function', zorder=0)

                #ax1.scatter(index_array, data_full_var, marker='o', label='range res. + clock', zorder=5)
                #ax1.plot(index_array, data_clock_var, linestyle='--', label='range clock function', zorder=1)
                #ax1.legend()

                ## Formatting the date on the x-axis
                #ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                #ax1.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hr))  # Adjust interval as needed
                #fig.autofmt_xdate()  # Auto-rotate date labels
                #
                #ax1.set_title('postfit residuals + clock variation (' + antenna2_handle.antenna_name + '—' + antenna1_handle.antenna_name + ')'\
                #        + ' WRMS (range) : ' + str(np.round(WRMS_range,decimals=1))+ ' ps,' + ' WRMS (phase) : ' + str(np.round(WRMS_phase,decimals=1))+ ' ps'   )
                ##ax1.set_xlabel('date + hr')
                #ax1.set_ylabel('residuals + clock variation (ps)')
                #ax1.grid(True)

                #if iono_free is True:
                #    fig.savefig(sol_type+'_'+sol_name+'_full_clock_variation_residuals_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '_ionofree.png')
                #else:
                #    fig.savefig(sol_type+'_'+sol_name+'_full_clock_variation_residuals_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '.png')
                #plt.close(fig)

            # plot carrier phase windup
            if sol_type == 'VLBI':
                fig, ax1 = plt.subplots(figsize=(10, 6))
                ax1.scatter(phase_index_array, baseline_handle.cpw_1[baseline_handle.phase_data_idxs]/const.c*1e12, color='r', label=antenna1_handle.antenna_name, zorder=0)
                ax1.scatter(phase_index_array, baseline_handle.cpw_2[baseline_handle.phase_data_idxs]/const.c*1e12, color='b', label=antenna2_handle.antenna_name, zorder=0)
                ax1.scatter(phase_index_array, (baseline_handle.cpw_2-baseline_handle.cpw_1)[baseline_handle.phase_data_idxs]/const.c*1e12, color='k', label='differential', zorder=1)
                ax1.legend()
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                ax1.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hr))  # Adjust interval as needed
                fig.autofmt_xdate()  # Auto-rotate date labels
                ax1.set_title('phase windup (' + antenna2_handle.antenna_name + '—' + antenna1_handle.antenna_name + ')')
                #ax1.set_xlabel('date + hr')
                ax1.set_ylabel('phase windup (ps)')
                ax1.grid(True)
                if iono_free is True:
                    fig.savefig(sol_type+'_'+sol_name+'_phase_windup_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '_ionofree.png')
                else:
                    fig.savefig(sol_type+'_'+sol_name+'_phase_windup_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '.png')
                plt.close(fig)

                # plot carrier phase windup by source
                fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 8))
                for src in np.unique(source_array):
                    # get epochs of source
                    idxs_src = []
                    for idx, time in enumerate(times_gps):
                        if src == store_handle.source_time_dict[time]:
                            idxs_src.append(idx)
                    idx_src = np.array(idxs_src)
                    _, _, phase_idxs = np.intersect1d(idxs_src, baseline_handle.phase_data_idxs, return_indices=True)
                    #ax1.plot(phase_index_array[phase_idxs], \
                    #        baseline_handle.cpw_1[baseline_handle.phase_data_idxs[phase_idxs]]/const.c*1e12, marker='o')
                    #ax2.plot(phase_index_array[phase_idxs], baseline_handle.cpw_2[baseline_handle.phase_data_idxs[phase_idxs]]/const.c*1e12, marker='o')
                    ax1.plot(phase_index_array[phase_idxs], \
                            baseline_handle.cpw_1[baseline_handle.phase_data_idxs[phase_idxs]]/baseline_handle.wavelength, marker='o')
                    ax2.plot(phase_index_array[phase_idxs], baseline_handle.cpw_2[baseline_handle.phase_data_idxs[phase_idxs]]/baseline_handle.wavelength, marker='o')
                    ax3.plot(phase_index_array[phase_idxs], (baseline_handle.cpw_2-baseline_handle.cpw_1)[baseline_handle.phase_data_idxs[phase_idxs]]/const.c*1e12, marker='o')
                ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                ax3.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hr))  # Adjust interval as needed
                fig.autofmt_xdate()  # Auto-rotate date labels
                ax1.set_title(antenna2_handle.antenna_name + '—' + antenna1_handle.antenna_name + ' differential feed rotation')
                #ax3.set_xlabel('date + hr')
                ax1.set_ylabel(antenna1_handle.antenna_name + ' (cycles)')
                ax2.set_ylabel(antenna2_handle.antenna_name + ' (cycles)')
                ax3.set_ylabel('rotation delay (ps)')
                ax1.grid(True)
                ax2.grid(True)
                ax3.grid(True)
                if iono_free is True:
                    fig.savefig(sol_type+'_'+sol_name+'_phase_windup_by_src_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '_ionofree.png')
                else:
                    fig.savefig(sol_type+'_'+sol_name+'_phase_windup_by_src_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '.png')
                plt.close(fig)

            # plot clock residuals by source
            fig, ax1 = plt.subplots(figsize=(10, 6))
            for src in np.unique(source_array):
                # get epochs of source
                idxs_src = []
                for idx, time in enumerate(times_gps):
                    if src == store_handle.source_time_dict[time]:
                        idxs_src.append(idx)
                idx_src = np.array(idxs_src)
                _, _, phase_idxs = np.intersect1d(idxs_src, baseline_handle.phase_data_idxs, return_indices=True)
                ax1.scatter(phase_index_array[phase_idxs], \
                        (data_phase['Clock'].to_numpy()[phase_idxs] + data_phase['Residuals'].to_numpy()[phase_idxs])/const.c*1e12
                        , marker='o', zorder=1)
            ax1.plot(phase_index_array, data_phase['Clock'].to_numpy()/const.c*1e12, linestyle='--', color='slategray', label='clock function', zorder=0)
            ax1.legend()
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            ax1.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hr))  # Adjust interval as needed
            fig.autofmt_xdate()  # Auto-rotate date labels
            ax1.set_title('postfit residuals + clock function (' + antenna2_handle.antenna_name + '—' + antenna1_handle.antenna_name + ')')
            #ax1.set_xlabel('date + hr')
            ax1.set_ylabel('phase + clock function (ps)')
            ax1.grid(True)
            if iono_free is True:
                fig.savefig(sol_type+'_'+sol_name+'_phase_clock_residuals_by_src_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '_ionofree.png')
            else:
                fig.savefig(sol_type+'_'+sol_name+'_phase_clock_residuals_by_src_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '.png')
            plt.close(fig)

            # plot zoomed data within 4 sigma of mean
            sigma, use_idxs = find_sigmas(residuals_phase)
            data_phase_zoom = DataFrame({
                'Datetime': dt_phase[use_idxs],
                'Residuals': residuals_phase[use_idxs]})
            data_phase_zoom.set_index('Datetime', inplace=True)
            phase_zoom_index = data_phase_zoom.index.to_numpy()
            # plot measurement residuals
            plt.figure(figsize=(10, 6))
            plt.plot(phase_zoom_index, data_phase_zoom['Residuals'].to_numpy(), marker='x', linestyle='None', color='b')
            
            # Formatting the date on the x-axis
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=interval_hr))  # Adjust interval as needed
            plt.gcf().autofmt_xdate()  # Auto-rotate date labels
            
            plt.title('final phase residuals (' + antenna2_handle.antenna_name + '—' + antenna1_handle.antenna_name + ')')
            #plt.xlabel('date + hr')
            plt.ylabel('meas. residuals (m)')
            plt.grid(True)
            if iono_free is True:
                plt.savefig(sol_type+'_'+sol_name+'_phase_meas_residuals_'+\
                        antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + 'ionofree_zoom.png')
            else:
                plt.savefig(sol_type+'_'+sol_name+'_phase_meas_residuals_'+\
                        antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '_zoom.png')
            plt.close()

            # ------------------Plot residuals by source-----------------#
            slips = []
            for slip_arr in baseline_handle.slip_slices_arr: 
                # first index of slip_arr is the index of a cycle slip
                if slip_arr[0] != 0: slips.append(slip_arr[0])
            
            #if len(source_array) > 0:
            #    fig2, ax2 = plt.subplots(figsize=(10, 6))
            #    for src in np.unique(source_array):
            #       # get epochs of source
            #       idxs_src = []
            #       for idx, time in enumerate(times_gps):
            #           if src == store_handle.source_time_dict[time]:
            #               idxs_src.append(idx)
            #       idx_src = np.array(idxs_src)
            #       idxs_pdi, idxs_phase, _ =  np.intersect1d(baseline_handle.phase_data_idxs, idxs_src, return_indices=True)
            #       # find which slips are in the data of this source, get their locations(indices) in idxs_src
            #       slips_src, slips_src_idxs, _ =  np.intersect1d(idxs_pdi, slips, return_indices=True)
            #       
            #       fig, ax = plt.subplots(figsize=(10, 6))
            #       for slip_slice in baseline_handle.slip_slices_arr:
            #           slip_slice_src = np.intersect1d(idxs_src, slip_slice)
            #           slip_idxs, _, idxs_phase_slice = np.intersect1d(slip_slice_src, baseline_handle.phase_data_idxs, return_indices=True)
            #           if len(slip_idxs)>0:
            #               ax.plot(time_deltas_full[slip_idxs], residuals_phase[idxs_phase_slice], marker='x', linestyle='None')
            #               ax2.plot(time_deltas_full[slip_idxs], residuals_phase[idxs_phase_slice], marker='x', linestyle='None')
            #                      
            #       if len(slips_src) >0:
            #           ax.vlines(x=time_deltas_full[slips_src], ymin=np.amin(residuals_phase[idxs_phase]),\
            #                 ymax=np.amax(residuals_phase[idxs_phase]), colors='k')
            #       ax.set_title('carrier phase residuals ' + str(src))
            #       ax.set_ylabel('phase (m)')
            #       ax.set_xlabel('time (sec)')

            #       if iono_free is True:
            #           fig.savefig('./slip_figs/'+sol_type+'_'+sol_name+'_'+antenna2_handle.antenna_name+'_' \
            #                   + antenna1_handle.antenna_name+'_residuals_'+str(src)+'_ionofree.png')
            #       else:
            #           fig.savefig('./slip_figs/'+sol_type+'_'+sol_name+'_'+antenna2_handle.antenna_name+'_' \
            #                   + antenna1_handle.antenna_name+'_residuals_'+str(src)+'.png')
            #       plt.close(fig)

            #    ax2.set_xlabel('time (sec)')
            #    ax2.set_ylabel('phase (m)')
            #    if iono_free is True:
            #        fig2.savefig('./slip_figs/'+sol_type+'_'+sol_name+'_'+antenna2_handle.antenna_name+'_' \
            #                + antenna1_handle.antenna_name+'_full_residuals_ionofree.png')
            #    else:
            #        fig2.savefig('./slip_figs/'+sol_type+'_'+sol_name+'_'+antenna2_handle.antenna_name+'_' \
            #                + antenna1_handle.antenna_name+'_full_residuals.png')
            #    plt.close(fig2)

            # plot allan variation of residuals
            tau_values_phase = generate_tau_values(baseline_handle.datetime_array[baseline_handle.phase_data_idxs], 200)
            tau_values_phase, allan_var_phase, allan_dev_phase = compute_allan_variance(residuals_phase/const.c*1e12, \
                                               baseline_handle.datetime_array[baseline_handle.phase_data_idxs], tau_values_phase)
            if len(tau_values_phase)>0:
                fig, ax1 = plt.subplots(figsize=(10, 6))
                ax1.plot(tau_values_phase, allan_dev_phase)
                ax1.set_xlabel('averaging time (s)')
                ax1.set_ylabel('allan deviation (ps/s)')
                ax1.set_yscale('log') 
                ax1.set_xscale('log') 
                ax1.grid(True)
                if iono_free is True:
                    fig.savefig(sol_type+'_'+sol_name+'_allan_dev_phase_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '_ionofree.png')
                else:
                    fig.savefig(sol_type+'_'+sol_name+'_allan_dev_phase_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '.png')
                plt.close(fig)

                if phase_only is False:
                    tau_values_range = generate_tau_values(baseline_handle.datetime_array[baseline_handle.range_data_idxs], 200)
                    tau_values_range, allan_var_range, allan_dev_range = compute_allan_variance(residuals_range/const.c*1e12, \
                                                       baseline_handle.datetime_array[baseline_handle.phase_data_idxs], tau_values_range)
                    fig, ax2 = plt.subplots(figsize=(10, 6))
                    ax2.plot(tau_values_range, allan_dev_range)
                    ax2.set_xlabel('averaging time (s)')
                    ax2.set_ylabel('allan deviation (ps/s)')
                    ax2.set_yscale('log') 
                    ax2.set_xscale('log') 
                    ax2.grid(True)
                    if iono_free is True:
                        fig.savefig(sol_type+'_'+sol_name+'_allan_dev_range_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '_ionofree.png')
                    else:
                        fig.savefig(sol_type+'_'+sol_name+'_allan_dev_range_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '.png')
                    plt.close(fig)

            # plot histogram of phase postfit residuals
            residuals_phase_sorted = np.sort(residuals_phase)
            normal_dist = norm.pdf(residuals_phase_sorted, np.mean(residuals_phase), np.std(residuals_phase))
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(residuals_phase, bins='fd', density=True)
            ax.plot(residuals_phase_sorted, normal_dist, label='normal dist.')
            ax.set_title('phase residuals distribution (' + antenna2_handle.antenna_name + '—' + antenna1_handle.antenna_name + ')')
            ax.set_xlabel('meas. residuals (m)')
            ax.set_ylabel('probability density')
            ax.legend()
            # Formatting the date on the x-axis
            if iono_free is True:
                fig.savefig(sol_type+'_'+sol_name+'_hist_phase_meas_residuals_'+\
                        antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + 'ionofree.png')
            else:
                fig.savefig(sol_type+'_'+sol_name+'_hist_phase_meas_residuals_'+\
                        antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '.png')       
            plt.close(fig)

            #if len(source_array) > 0:
            #    for src in np.unique(source_array):
            #        residuals_src = []
            #        # get epochs of source
            #        idxs_src = []
            #        for idx, time in enumerate(times_gps):
            #            if src == store_handle.source_time_dict[time]:
            #                idxs_src.append(idx)
            #        idx_src = np.array(idxs_src)
            #        idxs_src, _, phase_idxs = np.intersect1d(idxs_src, baseline_handle.phase_data_idxs, return_indices=True)
            #        runs_phase_idxs = consecutive_idxs(phase_idxs)
            #        if len(idxs_src) > 10:
            #            for run_phase in runs_phase_idxs:
            #                residuals_src.extend(residuals_phase[run_phase]-np.mean(residuals_phase[run_phase]))
            #            residuals_src = np.array(residuals_src)
            #            residuals_src_sorted = np.sort(residuals_src)
            #            normal_dist = norm.pdf(residuals_src_sorted, np.mean(residuals_src), np.std(residuals_src))
            #            fig, ax = plt.subplots(figsize=(10, 6))
            #            ax.hist(residuals_src, bins='fd', density=True)
            #            ax.plot(residuals_src_sorted, normal_dist, label='normal dist.')
            #            ax.set_title('source ' + src + ' ' + label2+' distribution (' \
            #                    + antenna2_handle.antenna_name + '—' + antenna1_handle.antenna_name + ')')
            #            ax.set_ylabel('probability density')
            #            ax.set_xlabel('meas. residuals (m)')
            #            ax.legend()
            #            if iono_free is True:
            #                fig.savefig(sol_type+'_'+sol_name+'_hist_phase_meas_residuals_'+src+'_'+\
            #                        antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + 'ionofree.png')
            #            else:
            #                fig.savefig(sol_type+'_'+sol_name+'_hist_phase_meas_residuals_'+src+'_'+\
            #                        antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '.png')       
            #            plt.close(fig)
    if phase_delay is True and len(antenna_handles)>2:
        # first use integer amb to get amb_state by baseline, then send this to find_closure_meas
        # generate phase closure plots
        n_amb = 0
        baseline_amb = []
        for baseline_handle in baseline_handles:
            full_amb_baseline = integer_amb[n_amb:n_amb+baseline_handle.n_amb_state]
            n_amb += baseline_handle.n_amb_state 
            baseline_amb.append([full_amb_baseline])

        baselines_closure = list(itertools.combinations(range(len(antenna_handles)), 3))
        if sol_type == 'VLBI':
            for baseline_closure in baselines_closure[:5]:
                # plot first 5 closures (n choose 3 expands very fast!)
                baseline_idxs = find_baselines_indices_for_triplet(baselines, baseline_closure)
                baseline1_handle = baseline_handles[baseline_idxs[0]]
                baseline2_handle = baseline_handles[baseline_idxs[1]]
                baseline3_handle = baseline_handles[baseline_idxs[2]]
                integer_amb1 = baseline_amb[baseline_idxs[0]][0]
                integer_amb2 = baseline_amb[baseline_idxs[1]][0]
                integer_amb3 = baseline_amb[baseline_idxs[2]][0]
                times_closure, closure_meas = find_closure_meas(sol_type, baseline1_handle, baseline2_handle, baseline3_handle, \
                        integer_amb1, integer_amb2, integer_amb3)
                antenna1_handle = antenna_handles[baseline_closure[0]]
                antenna2_handle = antenna_handles[baseline_closure[1]]
                antenna3_handle = antenna_handles[baseline_closure[2]]

                closure_meas *= 1e12/const.c
                RMS = np.sqrt(np.sum(closure_meas**2)/len(closure_meas))

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(times_closure, closure_meas, marker='.', color='g')
                # Formatting the date on the x-axis
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hr))  # Adjust interval as needed
                fig.autofmt_xdate()  # Auto-rotate date labels
                ax.set_title('phase closures (' + antenna1_handle.antenna_name + '—' + antenna2_handle.antenna_name + '—' + \
                        antenna3_handle.antenna_name + ')'+ ' RMS: ' + str(np.round(RMS,decimals=1))+ ' ps')
                ax.set_ylabel('phase closure (ps)')
                ax.grid(True)

                if iono_free is True:
                    fig.savefig(sol_type+'_'+sol_name+'_baseline_closures_'+antenna1_handle.antenna_name+'_' + antenna2_handle.antenna_name + '_' + antenna3_handle.antenna_name+'_ionofree.png')
                else:
                    fig.savefig(sol_type+'_'+sol_name+'_baseline_closures_'+antenna1_handle.antenna_name+'_' + antenna2_handle.antenna_name + '_' + antenna3_handle.antenna_name+'.png')
                plt.close(fig)

                times_closure, closure_model = find_closure_model(sol_type, baseline1_handle, baseline2_handle, baseline3_handle, antenna_handles, baseline_idxs, baselines)

                closure_model *= 1e12/const.c
                RMS = np.sqrt(np.sum(closure_model**2)/len(closure_model))

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(times_closure, closure_model, marker='.', color='g')
                # Formatting the date on the x-axis
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hr))  # Adjust interval as needed
                fig.autofmt_xdate()  # Auto-rotate date labels
                ax.set_title('model phase closures (' + antenna1_handle.antenna_name + '—' + antenna2_handle.antenna_name + '—' + \
                        antenna3_handle.antenna_name + ')'+ ' RMS: ' + str(np.round(RMS,decimals=1))+ ' ps')
                ax.set_ylabel('phase closure (ps)')
                ax.grid(True)

                if iono_free is True:
                    fig.savefig(sol_type+'_'+sol_name+'_baseline_model_closures_'+antenna1_handle.antenna_name+'_' + antenna2_handle.antenna_name + '_' + antenna3_handle.antenna_name+'_ionofree.png')
                else:
                    fig.savefig(sol_type+'_'+sol_name+'_baseline_model_closures_'+antenna1_handle.antenna_name+'_' + antenna2_handle.antenna_name + '_' + antenna3_handle.antenna_name+'.png')
                plt.close(fig)

    if store_handle.stochastic_clock is True:
        # plot Allan deviation curve for clock parameters
        for idx, antenna_handle in enumerate(antenna_handles):
            if antenna_handle.antenna_name == ref_antenna:
                ref_handle = antenna_handle
                ref_idx = idx
                break
        for idx, antenna_handle in enumerate(antenna_handles):
            if antenna_handle.antenna_name == ref_antenna:
                continue
            q_baseline = None
            # find the baseline handle with the reference antenna
            for jdx, baseline in enumerate(baselines):
                if idx in baseline and ref_idx in baseline:
                    if phase_delay is True:
                        q_baseline = baseline_handle.q_phase*1e12/const.c
                    else:
                        q_baseline = baseline_handle.q_range*1e12/const.c

            if q_baseline is None: continue
            phi_rw = antenna_handle.clock_psd_rw/3600   #(1e24/(const.c**2*FACTOR_RW)) # ps^2/s
            phi_irw = antenna_handle.clock_psd_irw/3600**3  #(1e24/(const.c**2*FACTOR_IRW)) # ps^2/s^3
            exp_len_sec = (antenna_handle.times_gps[-1] - antenna_handle.times_gps[0])/np.timedelta64(1, 's')
            #delta_t = np.mean(np.diff((antenna_handle.times_gps-antenna_handle.times_gps[0])/np.timedelta64(1,'s'))) # meas interval
            #times_sec = np.arange(exp_len_sec)+1
            times_sec = np.logspace(0,np.log10(exp_len_sec),300)
            AVAR_q = 3*q_baseline**2/times_sec**2
            AVAR_rw = phi_rw/times_sec
            AVAR_irw = phi_irw*times_sec/3
            ADEV = np.sqrt(AVAR_q + AVAR_rw + AVAR_irw)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(times_sec, ADEV, color='g')
            ax.plot(times_sec, np.sqrt(AVAR_q), linestyle='--', label='white noise')
            ax.plot(times_sec, np.sqrt(AVAR_rw), linestyle='--', label='random walk')
            ax.plot(times_sec, np.sqrt(AVAR_irw), linestyle='--', label='integrated random walk')
            ax.set_xlabel('averaging time (sec)')
            ax.set_ylabel('ADEV (ps/s)')
            ax.set_title('clock model ADEV (' + antenna_handle.antenna_name + ')')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True)

            if iono_free is True:
                fig.savefig(sol_type+'_'+sol_name+'_ADEV_'+antenna_handle.antenna_name+'_ionofree.png')
            else:
                fig.savefig(sol_type+'_'+sol_name+'_ADEV_'+antenna_handle.antenna_name+'.png')
            plt.close(fig)

    baselines_full = list(itertools.combinations(range(len(antenna_handles)), 2))
    for jdx, baseline in enumerate(baselines_full): # generate differential measurements on the baselines
        antenna1_handle = antenna_handles[baseline[0]]
        antenna2_handle = antenna_handles[baseline[1]]      

        # print baseline-specific quantities
        print('For baseline ' + antenna2_handle.antenna_name + '--' + antenna1_handle.antenna_name)
        if baseline[0] < ref_idx:
            count_1 = baseline[0]
        else:
            count_1 = baseline[0]-1
        if baseline[1] < ref_idx:
            count_2 = baseline[1]
        else:
            count_2 = baseline[1]-1

        if antenna2_handle.antenna_name == ref_antenna:
            R_mat = antenna2_handle.R_mat.T
            var_r2 = np.zeros((3,3))
            cov_r1_r2 = np.zeros((3,3))
            r2 = antenna2_handle.ref_pos
            r1 = final_state[count_1*3:count_1*3+3]
            var_r1 = cov_matrix_full[count_1*3:count_1*3+3,count_1*3:count_1*3+3]
        else:
            r2 = final_state[count_2*3:count_2*3+3]
            R_mat = antenna1_handle.R_mat.T
            var_r2 = cov_matrix_full[count_2*3:count_2*3+3,count_2*3:count_2*3+3]
            if antenna1_handle.antenna_name == ref_antenna:
                r1 = antenna1_handle.ref_pos
                var_r1 = np.zeros((3,3))
                cov_r1_r2 = np.zeros((3,3))
            else:
                r1 = final_state[count_1*3:count_1*3+3]
                var_r1 = cov_matrix_full[count_1*3:count_1*3+3,count_1*3:count_1*3+3]
                cov_r1_r2 = cov_matrix_full[count_1*3:count_1*3+3,count_2*3:count_2*3+3]
        b_12 = r2-r1
        b_12_NEU = R_mat@b_12
        var_b = var_r1 + var_r2 - 2*cov_r1_r2
        var_b_NEU = R_mat@var_b@R_mat.T
        sigmas_b = np.sqrt(np.diag(var_b))
        sigmas_b_NEU = np.sqrt(np.diag(var_b_NEU))
        L_mag = np.linalg.norm(b_12)
        sig_L = 1/L_mag*np.sqrt(b_12[0]**2*var_b[0,0] + b_12[1]**2*var_b[1,1] + b_12[2]**2*var_b[2,2]\
                + 2*b_12[0]*b_12[1]*var_b[0,1] + 2*b_12[0]*b_12[2]*var_b[0,2] + 2*b_12[1]*b_12[2]*var_b[1,2])
        print(f"baseline length (m): {L_mag:.4f}")
        print(f"sigma_LL: {sig_L:.5f}")
        print(f"baseline vector (XYZ, m): {b_12[0]:.4f}, {b_12[1]:.4f}, {b_12[2]:.4f}")
        print('formal baseline errors (XYZ, m): sigma_xx: ' + str(np.round(sigmas_b[0],6)) +  \
                    ' sigma_yy: ' + str(np.round(sigmas_b[1],6)) +  ' sigma_zz: ' + str(np.round(sigmas_b[2],6)))
        print(f"baseline vector (NEU, m): {b_12_NEU[0]:.4f}, {b_12_NEU[1]:.4f}, {b_12_NEU[2]:.4f}")
        print('formal baseline errors (NEU, m): sigma_NN: ' + str(np.round(sigmas_NEU[0],6)) +  \
                    ' sigma_EE: ' + str(np.round(sigmas_b_NEU[1],6)) +  ' sigma_UU: ' + str(np.round(sigmas_b_NEU[2],6)))


def cov_to_corr(cov):
    """ Convert covariance to correlation matrix """
    std_dev = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std_dev, std_dev)
    return corr

def fit_transform(X):
    """Fits the scaler on X and returns the scaled data along with the mean and std for future use."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X_scaled = (X - mean) / std
    return X_scaled, mean, std

def kernel_periodic(dx, ls):
    """ evaluate a periodic kernel function with length scale ls """ 
    return np.exp(-2 * np.sin(0.5 * dx)**2 / ls**2)

def kernel_sqexp(dx, ls):
    """ evaluate a squared-exponential kernel function with length scale ls """ 
    return np.exp(-0.5 * dx**2 / ls**2)

def load_kernel_from_csv(file_path):
    """
    Load the CSV file from az.summary(trace).to_csv() and return the means of all distributions in order.

    Parameters:
    - file_path: str, path to the CSV file.

    Returns:
    - tuple: means of (ls_azimuth, ls_elevation, ls_time, noise).
    """
    # Load the CSV file into a DataFrame
    df = read_csv(file_path, index_col=0)
    
    # Extract the 'mean' column as a list
    #means = df['mean'].tolist()

    # Ensure the correct order of hyperparameters
    ls_azimuth = df.loc['ls_azimuth', 'mean']
    ls_elevation = df.loc['ls_elevation', 'mean']
    ls_time = df.loc['ls_time', 'mean']
    noise = df.loc['noise', 'mean']
    
    return ls_azimuth, ls_elevation, ls_time, noise


def fit_transform(X):
    """Fits the scaler on X and returns the scaled data along with the mean and std for future use."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X_scaled = (X - mean) / std
    return X_scaled, mean, std

def save_kernel_parameters(filename, kernels):
    """
    Save the estimated kernel parameters, types, and scaling standard deviations to a file.

    Args:
        filename (str): The path to the file where parameters will be saved.
        kernels (dict): A dictionary where each key is a kernel name, and each value is a dictionary with keys:
            - 'type': The type of the kernel (e.g., 'sq-exp', 'periodic', 'white').
            - 'amplitude': The amplitude parameter.
            - 'length_scale': The length scale parameter (if applicable).
            - 'scaling_std': The standard deviation used for scaling the feature (if applicable).
    """
    # Ensure all parameter values are serializable
    for kernel_name, params in kernels.items():
        for key, value in params.items():
            if isinstance(value, np.generic):
                params[key] = value.item()

    # Save to JSON file
    with open(filename, 'w') as f:
        json.dump({'kernels': kernels}, f, indent=4)

def load_kernel_parameters(filename):
    """
    Load the estimated kernel parameters, types, and scaling standard deviations from a file.

    Args:
        filename (str): The path to the file from which parameters will be loaded.

    Returns:
        dict: A dictionary where each key is a kernel name, and each value is a dictionary with kernel parameters.
    """
    try:
        # Load from JSON file
        with open(filename, 'r') as f:
            data = json.load(f)

        kernels = data['kernels']
        return kernels
    except Exception as e:
        print(f"Error loading kernel parameters: {e}")
        return None

def mcmc_correlation(obs_type, datetime_array, sol_type, ls_sol, ls_args, bounds, res_fcn, jac):
    """Perform a Markov Chain Monte Carlo analysis to estimate the correlation between group and phase delay measurements"""
    import pymc as pm
    import arviz as az
    if obs_type == 'range':
        (ref_antenna, baselines, store_handle, antenna_handles, clock_idxs, clock_poly_length,
         trop_idxs, trop_poly_length, disb_idxs, baseline_handles) = ls_args
        phase_only = False
        phase_delay = False
    elif obs_type == 'phase':
        (ref_antenna, baselines, store_handle, antenna_handles, clock_idxs,
         clock_poly_length, trop_idxs, trop_poly_length, disb_idxs, baseline_handles,
         phase_delay, phase_only, use_amb_state, amb_state_idxs, use_phase_weights, tikhonov_lambda, cont_penalty, phase_clock_idxs,
         phase_disb_idxs, combination_type, integer_amb) = ls_args

    # Prepare data: time, azimuth, elevation
    time = np.array([])
    azimuth = np.array([])
    elevation = np.array([])
    for jdx, baseline in enumerate(baselines):
        baseline_handle = baseline_handles[jdx]
        antenna1_handle = antenna_handles[baseline[0]]
        antenna2_handle = antenna_handles[baseline[1]]
        azimuth_baseline = np.radians(antenna1_handle.azim_arr)
        elevation_baseline = np.radians(antenna1_handle.elev_arr)
        if phase_only is False:
            times_gps_range = baseline_handle.datetime_array[baseline_handle.phase_data_idxs]
            time_range = (times_gps_range - times_gps_range[0]) / np.timedelta64(1, 's')
            _, ant1_idxs, _ = np.intersect1d(antenna1_handle.times_gps,
                                             times_gps_range, return_indices=True)
            azimuth_range = azimuth_baseline[ant1_idxs]
            elevation_range = elevation_baseline[ant1_idxs]
            time = np.concatenate((time, time_range))
            azimuth = np.concatenate((azimuth, azimuth_range))
            elevation = np.concatenate((elevation, elevation_range))
        if phase_delay is True:
            times_gps_phase = baseline_handle.datetime_array[baseline_handle.phase_data_idxs]
            time_phase = (times_gps_phase - times_gps_phase[0]) / np.timedelta64(1, 's')
            _, ant1_idxs, _ = np.intersect1d(antenna1_handle.times_gps,
                                             times_gps_phase, return_indices=True)
            azimuth_phase = azimuth_baseline[ant1_idxs]
            elevation_phase = elevation_baseline[ant1_idxs]
            time = np.concatenate((time, time_phase))
            azimuth = np.concatenate((azimuth, azimuth_phase))
            elevation = np.concatenate((elevation, elevation_phase))

    # Scale elevation and time
    elevation_scaled, elev_mean, elev_std = fit_transform(elevation)
    time_scaled, time_mean, time_std = fit_transform(time)
    # Stack features into a (n_samples, n_features) array
    X_scaled = np.column_stack((azimuth, elevation_scaled, time_scaled))

    # Get residuals
    residuals = ls_sol.fun  # Residuals of the solution
    if phase_delay is True and phase_only is False:
        residuals_range, residuals_phase = get_residuals_unweighted(residuals, store_handle, baseline_handles, \
                baselines, antenna_handles, phase_delay, phase_only)
        residuals_full = np.concatenate((residuals_range, residuals_phase))
    else:
        residuals_full = get_residuals_unweighted(residuals, store_handle, baseline_handles, \
                    baselines, antenna_handles, phase_delay, phase_only)

    # Define the Bayesian model to estimate the covariance
    with pm.Model() as model:
        # Define priors for amplitude parameters
        wrms = np.sqrt(np.mean(residuals_full**2))
        prior_scale = wrms * 1.5  # Adjust based on your data

        amp_azimuth = pm.HalfNormal('amp_azimuth', sigma=prior_scale)
        amp_elevation = pm.HalfNormal('amp_elevation', sigma=prior_scale)
        amp_time = pm.HalfNormal('amp_time', sigma=prior_scale)
        noise = pm.HalfNormal('sigma', sigma=prior_scale / 2)  # Adjust as appropriate

        # Define priors for length scales
        ls_azimuth = pm.HalfNormal('ls_azimuth', sigma=1.0)
        ls_elevation = pm.HalfNormal('ls_elevation', sigma=1.0)
        ls_time = pm.HalfNormal('ls_time', sigma=1.0)

        # Define kernels with amplitude parameters
        input_dim = 3

        periodic_azimuth = amp_azimuth ** 2 * pm.gp.cov.Periodic(input_dim, period=2 * np.pi, ls=ls_azimuth, active_dims=[0])
        rbf_elevation = amp_elevation ** 2 * pm.gp.cov.ExpQuad(input_dim, ls=ls_elevation, active_dims=[1])
        rbf_time = amp_time ** 2 * pm.gp.cov.ExpQuad(input_dim, ls=ls_time, active_dims=[2])

        # Combine kernels additively
        kernel = periodic_azimuth + rbf_elevation + rbf_time

        # Instantiate the GP
        gp = pm.gp.Marginal(cov_func=kernel)

        # Likelihood
        y_obs = gp.marginal_likelihood("y_obs", X=X_scaled, y=residuals_full, sigma=noise)

        # Sample from the posterior
        trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.95, nuts={"max_treedepth": 15})

        # Extract mean hyperparameters
        mean_amp_azimuth = trace.posterior['amp_azimuth'].mean().item()
        mean_amp_elevation = trace.posterior['amp_elevation'].mean().item()
        mean_amp_time = trace.posterior['amp_time'].mean().item()
        mean_ls_azimuth = trace.posterior['ls_azimuth'].mean().item()
        mean_ls_elevation = trace.posterior['ls_elevation'].mean().item()
        mean_ls_time = trace.posterior['ls_time'].mean().item()
        mean_noise = trace.posterior['sigma'].mean().item()

        # Re-define kernels with mean hyperparameters
        periodic_azimuth = mean_amp_azimuth ** 2 * pm.gp.cov.Periodic(input_dim, period=2 * np.pi,
                                                                      ls=mean_ls_azimuth, active_dims=[0])
        rbf_elevation = mean_amp_elevation ** 2 * pm.gp.cov.ExpQuad(input_dim, ls=mean_ls_elevation, active_dims=[1])
        rbf_time = mean_amp_time ** 2 * pm.gp.cov.ExpQuad(input_dim, ls=mean_ls_time, active_dims=[2])
        kernel_mean = periodic_azimuth + rbf_elevation + rbf_time

        cov_noise = np.eye(len(X_scaled)) * mean_noise ** 2
        cov_mean = kernel_mean(X_scaled).eval()
        cov_total = cov_mean + cov_noise

    # get MAP estimates from Gaussian kernel density estimation
    from  scipy.stats import gaussian_kde
    samples_ls_elev = trace.posterior["ls_elevation"].values.flatten()
    kde_ls_elev = gaussian_kde(samples_ls_elev)
    xgrid_ls_elev = np.linspace(min(samples_ls_elev), max(samples_ls_elev), 1000)
    mode_ls_elevation = xgrid_ls_elev[np.argmax(kde_ls_elev(xgrid_ls_elev))]

    samples_amp_elev = trace.posterior["amp_elevation"].values.flatten()
    kde_amp_elev = gaussian_kde(samples_amp_elev)
    xgrid_amp_elev = np.linspace(min(samples_amp_elev), max(samples_amp_elev), 1000)
    mode_amp_elevation = xgrid_amp_elev[np.argmax(kde_amp_elev(xgrid_amp_elev))]

    samples_ls_azim = trace.posterior["ls_azimuth"].values.flatten()
    kde_ls_azim = gaussian_kde(samples_ls_azim)
    xgrid_ls_azim = np.linspace(min(samples_ls_azim), max(samples_ls_azim), 1000)
    mode_ls_azimuth = xgrid_ls_azim[np.argmax(kde_ls_azim(xgrid_ls_azim))]

    samples_amp_azim = trace.posterior["amp_azimuth"].values.flatten()
    kde_amp_azim = gaussian_kde(samples_amp_azim)
    xgrid_amp_azim = np.linspace(min(samples_amp_azim), max(samples_amp_azim), 1000)
    mode_amp_azimuth = xgrid_amp_azim[np.argmax(kde_amp_azim(xgrid_amp_azim))]

    samples_ls_time = trace.posterior["ls_time"].values.flatten()
    kde_ls_time = gaussian_kde(samples_ls_time)
    xgrid_ls_time = np.linspace(min(samples_ls_time), max(samples_ls_time), 1000)
    mode_ls_time = xgrid_ls_time[np.argmax(kde_ls_time(xgrid_ls_time))]

    samples_amp_time = trace.posterior["amp_time"].values.flatten()
    kde_amp_time = gaussian_kde(samples_amp_time)
    xgrid_amp_time = np.linspace(min(samples_amp_time), max(samples_amp_time), 1000)
    mode_amp_time = xgrid_amp_time[np.argmax(kde_amp_time(xgrid_amp_time))]

    samples_sigma = trace.posterior["sigma"].values.flatten()
    kde_sigma = gaussian_kde(samples_sigma)
    xgrid_sigma = np.linspace(min(samples_sigma), max(samples_sigma), 1000)
    mode_noise = xgrid_sigma[np.argmax(kde_sigma(xgrid_sigma))]

    # ANALYZE RESULTS
    print(f"MMSE Amplitude Azimuth: {mean_amp_azimuth}")
    print(f"MMSE Amplitude Elevation: {mean_amp_elevation}")
    print(f"MMSE Amplitude Time: {mean_amp_time}")
    print(f"MMSE Amplitude Noise: {mean_noise}")
    print(f"MMSE Length-Scale Azimuth: {mean_ls_azimuth}")
    print(f"MMSE Length-Scale Elevation: {mean_ls_elevation}")
    print(f"MMSE Length-Scale Time: {mean_ls_time}")
    print('\n')
    print(f"MAP Amplitude Azimuth: {mode_amp_azimuth}")
    print(f"MAP Amplitude Elevation: {mode_amp_elevation}")
    print(f"MAP Amplitude Time: {mode_amp_time}")
    print(f"MAP Amplitude Noise: {mode_noise}")
    print(f"MAP Length-Scale Azimuth: {mode_ls_azimuth}")
    print(f"MAP Length-Scale Elevation: {mode_ls_elevation}")
    print(f"MAP Length-Scale Time: {mode_ls_time}")
    
    save_var = 'MAP'
    if save_var == 'MAP':
        amp_azimuth = mode_amp_azimuth
        ls_azimuth = mode_ls_azimuth
        amp_elevation = mode_amp_elevation
        ls_elevation = mode_ls_elevation
        amp_time = mode_amp_time
        ls_time = mode_ls_time
        amp_noise = mode_noise
    elif save_var == 'MMSE':
        amp_azimuth = mean_amp_azimuth
        ls_azimuth = mean_ls_azimuth
        amp_elevation = mean_amp_elevation
        ls_elevation = mean_ls_elevation
        amp_time = mean_amp_time
        ls_time = mean_ls_time
        amp_noise = mean_noise

    kernels = {
        'azimuth': {
            'type': 'periodic',
            'amplitude': amp_azimuth,
            'length_scale': ls_azimuth,
            'scaling_std': 1
        },
        'elevation': {
            'type': 'sq-exp',
            'amplitude': amp_elevation,
            'length_scale': ls_elevation,
            'scaling_std': elev_std
        },
        'time': {
            'type': 'sq-exp',
            'amplitude': amp_time,
            'length_scale': ls_time,
            'scaling_std': time_std
        },
        'noise': {
            'type': 'white',
            'amplitude': amp_noise  # Noise doesn't have a length scale or std scale
        }
    }

    for baseline_handle in baseline_handles:
        baseline_handle.hold_covariance_kernels(kernels)

    kernel_filename = 'kernel_parameters_' + str(obs_type)+\
            '_'+antenna1_handle.antenna_name+'_'+antenna2_handle.antenna_name+'.json'

    # Save the parameters
    save_kernel_parameters(kernel_filename, kernels)
    print('kernel file output to ' + kernel_filename)

    kernel_name = "kernel_summary_" + str(obs_type)+\
            '_'+antenna1_handle.antenna_name+'_'+antenna2_handle.antenna_name
    az.summary(trace).to_csv(kernel_name+'.csv')
    print('kernel performance summary output to ' + kernel_name+'.csv')
    
    # ANALYZE RESULTS
    # Visualize the R-hat diagnostic
    axes = az.plot_trace(trace, var_names=['ls_azimuth', 'ls_elevation', 'ls_time'], figsize=(10, 13.5))
    axes[0,0].set_title('azimuth length-scale posterior dist.')
    axes[0,0].set_ylabel('probability density')
    axes[0,1].set_title('azimuth noise samples')
    axes[1,0].set_title('elevation length-scale posterior dist.')
    axes[1,0].set_ylabel('probability density')
    axes[1,1].set_title('elevation noise samples')
    axes[2,0].set_title('time length-scale posterior dist.')
    axes[2,0].set_xlabel('kernel length scale')
    axes[2,0].set_ylabel('probability density')
    axes[2,1].set_title('time noise samples')
    fig = axes[0, 0].figure
    Rhat_plot_name = "Rhat_trace_plot_lengthscale_"+obs_type+\
            '_'+antenna1_handle.antenna_name+'_'+antenna2_handle.antenna_name
    fig.savefig(Rhat_plot_name+'.png', dpi=300, bbox_inches="tight")

    axes = az.plot_trace(trace, var_names=['amp_azimuth', 'amp_elevation', 'amp_time', 'sigma'], figsize=(10, 13.5))
    axes[0,0].set_title('azimuth amplitude posterior dist.')
    axes[0,0].set_ylabel('probability density')
    axes[0,1].set_title('azimuth noise samples')
    axes[1,0].set_title('elevation amplitude posterior dist.')
    axes[1,0].set_ylabel('probability density')
    axes[1,1].set_title('elevation noise samples')
    axes[2,0].set_title('time amplitude posterior dist.')
    axes[2,0].set_ylabel('probability density')
    axes[2,1].set_title('time noise samples')
    axes[3,0].set_title('white noise amplitude posterior')
    axes[3,0].set_xlabel('kernel amplitude')
    axes[3,0].set_ylabel('probability density')
    axes[3,1].set_title('white noise samples')
    fig = axes[0, 0].figure
    Rhat_plot_name = "Rhat_trace_plot_amplitude_"+obs_type+\
            '_'+antenna1_handle.antenna_name+'_'+antenna2_handle.antenna_name
    fig.savefig(Rhat_plot_name+'.png', dpi=300, bbox_inches="tight")

    # Plot correlation matrix
    # Convert to correlation matrix
    std_dev = np.sqrt(np.diag(cov_total))
    corr_mean = cov_total / np.outer(std_dev, std_dev)
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(corr_mean, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax, label='Correlation Coefficient')
    ax.set_title('Posterior Mean Correlation Matrix')
    post_corr_name = 'mcmc_posterior_mean_correlation_matrix_'+obs_type+\
            '_'+antenna1_handle.antenna_name+'_'+antenna2_handle.antenna_name
    plt.savefig(post_corr_name+'.png')
    plt.close(fig)

    # plot the covariances
    # elevation
    N = 100
    elev = np.linspace(0,90,N)
    X_elev = np.radians(elev)/elev_std
    K_elev_mmse = kernel_sqexp(X_elev, mean_ls_elevation) 
    K_elev_map = kernel_sqexp(X_elev, mode_ls_elevation) 
    azim = np.linspace(0,360,N)
    K_azim_mmse = kernel_periodic(np.radians(azim), mean_ls_azimuth)
    K_azim_map = kernel_periodic(np.radians(azim), mode_ls_azimuth)
    time = np.linspace(0,2*3600,N)
    X_time = time/time_std
    K_time_mmse = kernel_sqexp(X_time, mean_ls_time)
    K_time_map = kernel_sqexp(X_time, mode_ls_time)

    # compare to least squares results
    lsq_sol_corr, ls_kernels = ls_correlation(obs_type, datetime_array, sol_type, ls_sol, ls_args, bounds, res_fcn, jac, False)
    K_elev_lsq = kernel_sqexp(X_elev, ls_kernels['elevation']['length_scale'])
    K_azim_lsq = kernel_periodic(np.radians(azim), ls_kernels['azimuth']['length_scale'])
    K_time_lsq = kernel_sqexp(X_time, ls_kernels['time']['length_scale'])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(elev, K_elev_mmse, label='MMSE')
    ax.plot(elev, K_elev_map, label='MAP')
    ax.plot(elev, K_elev_lsq, label='LSQ-VCE')
    ax.legend()
    ax.set_xlabel('elevation difference (deg)')
    ax.set_ylabel('correlation')
    kernel_elev_name = 'elevation_correlation_function_'+obs_type+\
            '_'+antenna1_handle.antenna_name+'_'+antenna2_handle.antenna_name
    plt.savefig(kernel_elev_name+'.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(azim, K_azim_mmse, label='MMSE')
    ax.plot(azim, K_azim_map, label='MAP')
    ax.plot(azim, K_azim_lsq, label='LSQ-VCE')
    ax.legend()
    ax.set_xlabel('azimuth difference (deg)')
    ax.set_ylabel('correlation')
    kernel_azim_name = 'azimuth_correlation_function_'+obs_type+\
            '_'+antenna1_handle.antenna_name+'_'+antenna2_handle.antenna_name
    plt.savefig(kernel_azim_name+'.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(time, K_time_mmse, label='MMSE')
    ax.plot(time, K_time_map, label='MAP')
    ax.plot(time, K_time_lsq, label='LSQ-VCE')
    ax.legend()
    ax.set_xlabel('time difference (sec)')
    ax.set_ylabel('correlation')
    kernel_time_name = 'time_correlation_function_'+obs_type+\
            '_'+antenna1_handle.antenna_name+'_'+antenna2_handle.antenna_name
    plt.savefig(kernel_time_name+'.png')
    plt.close(fig)

    ls_sol = least_squares(res_fcn, ls_sol.x, jac=jac, method='trf',\
        max_nfev=100, bounds=bounds, verbose=2, x_scale = 'jac', xtol=1e-15,\
        args=ls_args)

    return ls_sol

def ls_correlation(obs_type, datetime_array, sol_type, ls_sol, ls_args, bounds, res_fcn, jac, save_kernel=True):
    """Estimate the correlation between group and phase delay measurements using least squares."""
    if obs_type == 'range':
        (ref_antenna, baselines, store_handle, antenna_handles, clock_idxs, clock_poly_length,
         trop_idxs, trop_poly_length, disb_idxs, baseline_handles) = ls_args
        phase_only = False
        phase_delay = False
    elif obs_type == 'phase':
        (ref_antenna, baselines, store_handle, antenna_handles, clock_idxs,
         clock_poly_length, trop_idxs, trop_poly_length, disb_idxs, baseline_handles,
         phase_delay, phase_only, use_amb_state, amb_state_idxs, use_phase_weights, tikhonov_lambda, cont_penalty, phase_clock_idxs,
         phase_disb_idxs, combination_type, integer_amb) = ls_args

    # Prepare data: time, azimuth, elevation
    time = np.array([])
    azimuth = np.array([])
    elevation = np.array([])
    for jdx, baseline in enumerate(baselines):
        baseline_handle = baseline_handles[jdx]
        antenna1_handle = antenna_handles[baseline[0]]
        antenna2_handle = antenna_handles[baseline[1]]
        azimuth_baseline = np.radians(antenna1_handle.azim_arr)
        elevation_baseline = np.radians(antenna1_handle.elev_arr)
        if phase_only is False:
            times_gps_range = baseline_handle.datetime_array[baseline_handle.phase_data_idxs]
            time_range = (times_gps_range - times_gps_range[0]) / np.timedelta64(1, 's')
            _, ant1_idxs, _ = np.intersect1d(antenna1_handle.times_gps,
                                             times_gps_range, return_indices=True)
            azimuth_range = azimuth_baseline[ant1_idxs]
            elevation_range = elevation_baseline[ant1_idxs]
            time = np.concatenate((time, time_range))
            azimuth = np.concatenate((azimuth, azimuth_range))
            elevation = np.concatenate((elevation, elevation_range))
        if phase_delay is True:
            times_gps_phase = baseline_handle.datetime_array[baseline_handle.phase_data_idxs]
            time_phase = (times_gps_phase - times_gps_phase[0]) / np.timedelta64(1, 's')
            _, ant1_idxs, _ = np.intersect1d(antenna1_handle.times_gps,
                                             times_gps_phase, return_indices=True)
            azimuth_phase = azimuth_baseline[ant1_idxs]
            elevation_phase = elevation_baseline[ant1_idxs]
            time = np.concatenate((time, time_phase))
            azimuth = np.concatenate((azimuth, azimuth_phase))
            elevation = np.concatenate((elevation, elevation_phase))

    # Scale elevation and time
    elevation_scaled, elev_mean, elev_std = fit_transform(elevation)
    time_scaled, time_mean, time_std = fit_transform(time)
    # Stack features into a (n_samples, n_features) array
    X_scaled = np.column_stack((azimuth, elevation_scaled, time_scaled))

    # Get residuals
    residuals = ls_sol.fun  # Residuals of the solution
    if phase_delay is True and phase_only is False:
        residuals_range, residuals_phase = get_residuals_unweighted(residuals, store_handle, baseline_handles, \
                baselines, antenna_handles, phase_delay, phase_only)
        residuals_full = np.concatenate((residuals_range, residuals_phase))
    else:
        residuals_full = get_residuals_unweighted(residuals, store_handle, baseline_handles, \
                    baselines, antenna_handles, phase_delay, phase_only)

    # Compute the empirical covariance matrix
    emp_cov = np.outer(residuals_full, residuals_full)

    # THIS IS USELESS ^^^^^ DO NOT USE THIS FUNCTION IN ITS CURRENT STATE, NOT A GOOD REPRESENTATION OF COVARIANCE

    # Extract the upper triangle indices
    iu = np.triu_indices_from(emp_cov)

    # Flatten the upper triangle of the empirical covariance matrix
    emp_cov_vec = emp_cov[iu]

    # Define the residual function for least squares
    # Build the kernel matrices
    input_dim = 3

    # Compute pairwise differences
    azimuth_diff = np.subtract.outer(X_scaled[:, 0], X_scaled[:, 0])
    elevation_diff = np.subtract.outer(X_scaled[:, 1], X_scaled[:, 1])
    time_diff = np.subtract.outer(X_scaled[:, 2], X_scaled[:, 2])
    def residual_func(params):
        amp_azimuth, amp_elevation, amp_time, amp_noise, ls_azimuth, ls_elevation, ls_time = params

        # Periodic kernel for azimuth
        K_azimuth = amp_azimuth ** 2 * np.exp(-2 * np.sin(0.5 * azimuth_diff) ** 2 / ls_azimuth ** 2)

        # RBF kernels for elevation and time
        K_elevation = amp_elevation ** 2 * np.exp(-0.5 * (elevation_diff ** 2) / ls_elevation ** 2)
        K_time = amp_time ** 2 * np.exp(-0.5 * (time_diff ** 2) / ls_time ** 2)

        # Noise kernel
        K_noise = amp_noise ** 2 * np.eye(len(residuals_full))

        # Total covariance matrix
        cov_model = K_azimuth + K_elevation + K_time + K_noise

        # Flatten the upper triangle of the model covariance matrix
        cov_model_vec = cov_model[iu]

        # Compute residuals as difference between empirical and model covariance
        res = emp_cov_vec - cov_model_vec

        return res

    # Initial guesses for parameters
    wrms = np.sqrt(np.mean(residuals_full ** 2))
    initial_params = [wrms, wrms, wrms, wrms / 2, 1.0, 1.0, 1.0]

    # Bounds for parameters (ensure positive amplitudes and length scales)
    lower_bounds = [1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8]
    upper_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]

    # Perform least squares estimation
    result = least_squares(residual_func, initial_params, bounds=(lower_bounds, upper_bounds))

    # Extract estimated parameters
    amp_azimuth, amp_elevation, amp_time, amp_noise, ls_azimuth, ls_elevation, ls_time = result.x

    # Evaluate parameter uncertainties
    residuals_vce = result.fun
    m = len(residuals_vce)
    n = len(result.x)
    degrees_of_freedom = m - n
    s_sq = 2 * result.cost / degrees_of_freedom  # Since result.cost is 0.5 * sum(residuals**2)
    # Jacobian at the solution
    J = result.jac  # Shape (m, n)
    cov_params = s_sq * np.linalg.inv(J.T @ J)
    param_std = np.sqrt(np.diag(cov_params))

    if np.linalg.cond(J.T@J)>1e14:
        print('Least squares solution is unreliable (large condition number)')

    # Print parameter estimates and uncertainties
    param_names = ['amp_azimuth', 'amp_elevation', 'amp_time', 'amp_noise',
                   'ls_azimuth', 'ls_elevation', 'ls_time']
    print("\nEstimated Parameters and Uncertainties (LS-VCE):")
    for name, value, std in zip(param_names, result.x, param_std):
        print(f"{name}: {value:.6f} ± {std:.6f}")

    kernels = {
        'azimuth': {
            'type': 'periodic',
            'amplitude': amp_azimuth,
            'length_scale': ls_azimuth,
            'scaling_std': 1
        },
        'elevation': {
            'type': 'sq-exp',
            'amplitude': amp_elevation,
            'length_scale': ls_elevation,
            'scaling_std': elev_std
        },
        'time': {
            'type': 'sq-exp',
            'amplitude': amp_time,
            'length_scale': ls_time,
            'scaling_std': time_std
        },
        'noise': {
            'type': 'white',
            'amplitude': amp_noise  # Noise doesn't have a length scale or std scale
        }
    }

    # Reconstruct the covariance matrix using estimated parameters
    # Compute pairwise differences
    azimuth_diff = np.subtract.outer(X_scaled[:, 0], X_scaled[:, 0])
    elevation_diff = np.subtract.outer(X_scaled[:, 1], X_scaled[:, 1])
    time_diff = np.subtract.outer(X_scaled[:, 2], X_scaled[:, 2])

    # Periodic kernel for azimuth
    K_azimuth = amp_azimuth ** 2 * np.exp(-2 * np.sin(0.5 * azimuth_diff) ** 2 / ls_azimuth ** 2)

    # RBF kernels for elevation and time
    K_elevation = amp_elevation ** 2 * np.exp(-0.5 * (elevation_diff ** 2) / ls_elevation ** 2)
    K_time = amp_time ** 2 * np.exp(-0.5 * (time_diff ** 2) / ls_time ** 2)

    # Noise kernel
    K_noise = amp_noise ** 2 * np.eye(len(residuals_full))

    # Total covariance matrix
    cov_total = K_azimuth + K_elevation + K_time + K_noise
    # Convert to correlation matrix
    std_dev = np.sqrt(np.diag(cov_total))
    corr_mean = cov_total / np.outer(std_dev, std_dev)

    # Plot correlation matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(corr_mean, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax, label='Correlation Coefficient')
    ax.set_title('Posterior Mean Correlation Matrix')
    post_corr_name = 'ls_posterior_mean_correlation_matrix_'+obs_type+\
            '_'+antenna1_handle.antenna_name+'_'+antenna2_handle.antenna_name
    plt.savefig(post_corr_name+'.png')
    plt.close(fig)
    
    if save_kernel is True: 
        kernel_filename = 'kernel_parameters_' + str(obs_type)+\
                '_'+antenna1_handle.antenna_name+'_'+antenna2_handle.antenna_name+'.json'
        save_kernel_parameters(kernel_filename, kernels)

        for baseline_handle in baseline_handles:
            baseline_handle.hold_covariance_kernels(kernels)

        # plot the covariances
        # elevation
        N = 100
        elev = np.linspace(0,90,N)
        X_elev = np.radians(elev)/elev_std
        azim = np.linspace(0,180,N)
        time = np.linspace(0,time[-2],N)
        X_time = time/time_std

        # compare to least squares results
        K_elev_lsq = kernel_sqexp(X_elev, kernels['elevation']['length_scale'])
        K_azim_lsq = kernel_periodic(np.radians(azim), kernels['azimuth']['length_scale'])
        K_time_lsq = kernel_sqexp(X_time, kernels['time']['length_scale'])

        # compute 1-sigma bounds
        K_elev_lsq_p1s = kernel_sqexp(X_elev, kernels['elevation']['length_scale']+param_std[5])
        if kernels['elevation']['length_scale']-param_std[5] > 0:
            K_elev_lsq_m1s = kernel_sqexp(X_elev, kernels['elevation']['length_scale']-param_std[5])
        else:
            K_elev_lsq_m1s = kernel_sqexp(X_elev, 0)

        K_azim_lsq_p1s = kernel_periodic(np.radians(azim), kernels['azimuth']['length_scale']+param_std[4])
        if kernels['azimuth']['length_scale']-param_std[4] > 0:
            K_azim_lsq_m1s = kernel_periodic(np.radians(azim), kernels['azimuth']['length_scale']-param_std[4])
        else:
            K_azim_lsq_m1s = kernel_periodic(np.radians(azim), 0)

        K_time_lsq_p1s = kernel_sqexp(X_time, kernels['time']['length_scale']+param_std[6])
        if kernels['time']['length_scale']-param_std[6] > 0:
            K_time_lsq_m1s = kernel_sqexp(X_time, kernels['time']['length_scale']-param_std[6])
        else:
            K_time_lsq_m1s = kernel_sqexp(X_time, 0)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(elev, K_elev_lsq, label='LSQ-VCE')
        ax.plot(elev, K_elev_lsq_p1s, linestyle='--', color='r')
        ax.plot(elev, K_elev_lsq_m1s, linestyle='--', color='r')
        ax.set_xlabel('elevation difference (deg)')
        ax.set_ylabel('correlation')
        kernel_elev_name = 'elevation_correlation_function_'+obs_type+\
                '_'+antenna1_handle.antenna_name+'_'+antenna2_handle.antenna_name
        plt.savefig(kernel_elev_name+'.png')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(azim, K_azim_lsq, label='LSQ-VCE')
        ax.plot(azim, K_azim_lsq_p1s, linestyle='--', color='r')
        ax.plot(azim, K_azim_lsq_m1s, linestyle='--', color='r')
        ax.set_xlabel('azimuth difference (deg)')
        ax.set_ylabel('correlation')
        kernel_azim_name = 'azimuth_correlation_function_'+obs_type+\
                '_'+antenna1_handle.antenna_name+'_'+antenna2_handle.antenna_name
        plt.savefig(kernel_azim_name+'.png')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(time, K_time_lsq, label='LSQ-VCE')
        ax.plot(time, K_time_lsq_p1s, linestyle='--', color='r')
        ax.plot(time, K_time_lsq_m1s, linestyle='--', color='r')
        ax.set_xlabel('time difference (sec)')
        ax.set_ylabel('correlation')
        kernel_time_name = 'time_correlation_function_'+obs_type+\
                '_'+antenna1_handle.antenna_name+'_'+antenna2_handle.antenna_name
        plt.savefig(kernel_time_name+'.png')
        plt.close(fig)

        # Optionally, re-run the least squares solution with the updated weights
        ls_sol = least_squares(res_fcn, ls_sol.x, jac=jac, method='trf',
                               max_nfev=100, bounds=bounds, verbose=2, x_scale='jac', xtol=1e-15,
                               args=ls_args)

    return ls_sol, kernels


def compute_allan_variance(residuals, datetime_array, tau_values):
    """
    Computes the Allan variance for a given set of residuals and tau values.

    Parameters:
    - residuals: np.ndarray, residuals from the least squares solution.
    - datetime_array: np.ndarray, observation timestamps (numpy datetime64).
    - tau_values: np.ndarray, array of averaging times (tau) in the same units as the timestamps.

    Returns:
    - tau_values_used: np.ndarray, tau values for which Allan variance was computed.
    - allan_variance: np.ndarray, Allan variance for each tau value.
    - allan_deviation: np.ndarray, Allan deviation (square root of Allan variance) for each tau value.
    """
    # Convert timestamps to seconds if they are in datetime64
    timestamps = (datetime_array - datetime_array[0]) / np.timedelta64(1, 's')
    
    total_time = timestamps[-1] - timestamps[0]  # Total duration of the data
    allan_variance = []
    tau_values_used = []

    for tau in tau_values:
        # Calculate the number of non-overlapping segments
        num_segments = int(np.floor(total_time / tau))
        if num_segments < 2:
            continue  # Not enough segments to compute variance

        segment_averages = []

        for k in range(num_segments):
            # Define segment start and end times
            start_time = timestamps[0] + k * tau
            end_time = start_time + tau

            # Find indices within this segment
            indices_in_segment = np.where((timestamps >= start_time) & (timestamps < end_time))[0]
            if len(indices_in_segment) == 0:
                continue  # No data in this segment, skip it

            # Compute the average residual for the segment
            avg_residual = np.mean(residuals[indices_in_segment])
            segment_averages.append(avg_residual)

        segment_averages = np.array(segment_averages)
        if len(segment_averages) < 2:
            continue  # Not enough data to compute variance

        # Compute Allan variance
        variance = 0.5 * np.mean((segment_averages[1:] - segment_averages[:-1]) ** 2)
        allan_variance.append(variance)
        tau_values_used.append(tau)

    allan_variance = np.array(allan_variance)
    allan_deviation = np.sqrt(allan_variance)  # Allan deviation is the square root of variance

    return np.array(tau_values_used), allan_variance, allan_deviation


def generate_tau_values(datetime_array, num_points=50):
    """
    Generates a reasonable set of tau values for Allan variance computation from timestamps.

    Parameters:
    - timestamps: np.ndarray, numpy datetime64 array of timestamps.
    - num_points: int, number of tau values to generate (logarithmic spacing).

    Returns:
    - tau_values: np.ndarray, tau values in seconds.
    """
    times_in_seconds = (datetime_array - datetime_array[0]) / np.timedelta64(1, 's')
    
    # Compute the minimum and maximum time intervals
    min_interval = np.min(np.diff(times_in_seconds))  # Smallest time step
    max_interval = times_in_seconds[-1] - times_in_seconds[0]  # Total duration
    
    # Generate tau values logarithmically spaced between min_interval and max_interval
    tau_values = np.logspace(np.log10(min_interval), np.log10(max_interval / 2), num=num_points)
    
    return tau_values

def generate_correlated_noise(cov_matrix, rng=None):
    """
    Generate random noise consistent with the given covariance matrix.

    Parameters:
    - cov_matrix: np.ndarray, the covariance matrix.
    - rng: result of np.random.default_rng() -- random seed for reproducibility.

    Returns:
    - correlated_noise: np.ndarray, random noise consistent with the covariance matrix.
    """
    if rng is None:
        rng = np.random.default_rng()  # seed can be None or an int
    n = cov_matrix.shape[0]
    
    # Cholesky decomposition
    L = np.linalg.cholesky(cov_matrix)
    
    # Generate uncorrelated random noise
    z = rng.standard_normal(n)
    
    # Correlate the noise using the Cholesky factor
    correlated_noise = L @ z
    
    return correlated_noise

def consecutive_idxs(data, stepsize=1):
    """ return consecutive ranges of indices """
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def plot_time_units(ref_antenna, ls_sol_grdel, ls_sol_phdel, sol_type, store_handle, antenna_handles, baselines,\
                        iono_free, baseline_handles, clock_poly_length, clock_idxs, phase_clock_idxs):
    """Analyze the phase-only and range-only least-squares solutions, plot relevant residuals"""
    if sol_type == 'GNSS':
        label = 'pseudorange'
        label2 = 'carrier phase'
    elif sol_type == 'VLBI':
        label = 'group delay'
        label2 = 'phase delay'
    residuals_range = ls_sol_grdel.fun  # Residuals of the solution
    residuals_phase = ls_sol_phdel.fun  # Residuals of the solution
    num_samples_phase = 0
    num_samples_range = 0
    for idx, antenna_handle in enumerate(antenna_handles):
        if ref_antenna == antenna_handle.antenna_name: # compute corrected ranges once
            ref_idx = idx

    if store_handle.global_linear_clock is True:
        idx_start = 1
    elif store_handle.global_quadratic_clock is True:
        idx_start = 2
    else:
        idx_start = 0

    for jdx, baseline in enumerate(baselines): # generate differential measurements on the baselines
        baseline_handle = baseline_handles[jdx]
        #if baseline[1] == ref_idx: # ensure that reference antenna is subtracted (want estimated-reference)
        #    antenna1_handle = antenna_handles[baseline[1]]
        #    antenna2_handle = antenna_handles[baseline[0]]
        #else:
        #    antenna1_handle = antenna_handles[baseline[0]]
        #    antenna2_handle = antenna_handles[baseline[1]]

        antenna1_handle = antenna_handles[baseline[0]]
        antenna2_handle = antenna_handles[baseline[1]]      

        clock_state = ls_sol_grdel.x[clock_idxs] 
        phase_clock_state = ls_sol_phdel.x[phase_clock_idxs]

        if antenna1_handle.antenna_name != ref_antenna:
            clock_state_ant1 = clock_state[antenna1_handle.range_clock_idxs]
            phase_clock_state_ant1 = phase_clock_state[antenna1_handle.phase_clock_idxs]

            if store_handle.global_linear_clock is True or store_handle.global_quadratic_clock is True:
                clock_state_global_ant1 = clock_state_ant1[:idx_start]
                phase_clock_state_global_ant1 = phase_clock_state_ant1[:idx_start]
                if store_handle.stochastic_clock is True:
                    clock_samples_ant1 = sample_global_poly_at_interval(clock_state_global_ant1, antenna1_handle.clock_times,\
                            antenna1_handle.times_gps[0], antenna1_handle.times_gps[-1])
                    phase_clock_samples_ant1 = sample_global_poly_at_interval(phase_clock_state_global_ant1, antenna1_handle.phase_clock_times,\
                            antenna1_handle.times_gps[0], antenna1_handle.times_gps[-1])
                else:
                    clock_samples_ant1 = sample_global_poly_at_interval(clock_state_global_ant1, antenna1_handle.times_gps)
                    phase_clock_samples_ant1 = sample_global_poly_at_interval(phase_clock_state_global_ant1, antenna1_handle.times_gps)
                clock_state_ant1 = clock_state_ant1[idx_start:]
                phase_clock_state_ant1 = phase_clock_state_ant1[idx_start:]
            else:
                if store_handle.stochastic_clock is True:
                    clock_samples_ant1 = np.zeros(len(clock_state_ant1))
                    phase_clock_samples_ant1 = np.zeros(len(phase_clock_state_ant1))
                else:
                    clock_samples_ant1 = np.zeros(len(antenna1_handle.times_gps))
                    phase_clock_samples_ant1 = np.zeros(len(antenna1_handle.times_gps))

            if store_handle.stochastic_clock is False and clock_poly_length>0:
                clock_samples_ant1 += sample_poly_at_interval(clock_state_ant1, clock_poly_length, antenna1_handle.times_gps)
                phase_clock_samples_ant1 += sample_poly_at_interval(phase_clock_state_ant1,\
                                clock_poly_length, antenna1_handle.times_gps, antenna1_handle.phase_clock_start)
            elif store_handle.stochastic_clock is True:
                clock_samples_ant1 += clock_state_ant1
                phase_clock_samples_ant1 += phase_clock_state_ant1
            else:
                # only global clock model
                clock_samples_ant1 += clock_state_ant1[0] # bulk offset
                phase_clock_samples_ant1 += phase_clock_state_ant1[0] # bulk offset
        else:
            clock_samples_ant1 = antenna1_handle.clock_samples
            phase_clock_samples_ant1 = antenna1_handle.phase_clock_samples

        if antenna2_handle.antenna_name != ref_antenna:
            clock_state_ant2 = clock_state[antenna2_handle.range_clock_idxs]
            phase_clock_state_ant2 = phase_clock_state[antenna2_handle.phase_clock_idxs]

            if store_handle.global_linear_clock is True or store_handle.global_quadratic_clock is True:
                clock_state_global_ant2 = clock_state_ant2[:idx_start]
                phase_clock_state_global_ant2 = phase_clock_state_ant2[:idx_start]
                if store_handle.stochastic_clock is True:
                    clock_samples_ant2 = sample_global_poly_at_interval(clock_state_global_ant2, antenna2_handle.clock_times,\
                            antenna2_handle.times_gps[0], antenna2_handle.times_gps[-1])
                    phase_clock_samples_ant2 = sample_global_poly_at_interval(phase_clock_state_global_ant2, antenna2_handle.phase_clock_times,\
                            antenna2_handle.times_gps[0], antenna2_handle.times_gps[-1])
                else:
                    clock_samples_ant2 = sample_global_poly_at_interval(clock_state_global_ant2, antenna2_handle.times_gps)
                    phase_clock_samples_ant2 = sample_global_poly_at_interval(phase_clock_state_global_ant2, antenna2_handle.times_gps)
                clock_state_ant2 = clock_state_ant2[idx_start:]
                phase_clock_state_ant2 = phase_clock_state_ant2[idx_start:]
            else:
                if store_handle.stochastic_clock is True:
                    clock_samples_ant2 = np.zeros(len(clock_state_ant2))
                    phase_clock_samples_ant2 = np.zeros(len(phase_clock_state_ant2))
                else:
                    clock_samples_ant2 = np.zeros(len(antenna2_handle.times_gps))
                    phase_clock_samples_ant2 = np.zeros(len(antenna2_handle.times_gps))

            if store_handle.stochastic_clock is False and clock_poly_length>0:
                clock_samples_ant2 = sample_poly_at_interval(clock_state_ant2, clock_poly_length, antenna2_handle.times_gps)
                phase_clock_samples_ant2 = sample_poly_at_interval(phase_clock_state_ant2,\
                                clock_poly_length, antenna2_handle.times_gps, antenna2_handle.phase_clock_start)
            elif store_handle.stochastic_clock is True:
                clock_samples_ant2 = clock_state_ant2
                phase_clock_samples_ant2 = phase_clock_state_ant2
            else:
                clock_samples_ant2 += clock_state_ant2[0]
                phase_clock_samples_ant2 += phase_clock_state_ant2[0]
        else:
            clock_samples_ant2 = antenna2_handle.clock_samples
            phase_clock_samples_ant2 = antenna2_handle.phase_clock_samples

        if store_handle.stochastic_clock is False:
            _, ant1_idxs, ant2_idxs = np.intersect1d(antenna1_handle.times_gps, \
                     antenna2_handle.times_gps, return_indices=True)
            diff_clock = antenna2_handle.clock_samples[ant2_idxs ] - \
                        antenna1_handle.clock_samples[ant1_idxs]
            diff_clock_phase = antenna2_handle.phase_clock_samples[ant2_idxs ] - \
                        antenna1_handle.phase_clock_samples[ant1_idxs]
        else:
            _, ant1_idxs_clock, ant1_dt = np.intersect1d(antenna1_handle.clock_times, \
                    baseline_handle.datetime_array, return_indices=True)
            _, ant2_idxs_clock, ant2_dt = np.intersect1d(antenna2_handle.clock_times, \
                    baseline_handle.datetime_array, return_indices=True)
            diff_clock = np.zeros(len(baseline_handle.datetime_array))
            diff_clock[ant2_dt] += antenna2_handle.clock_samples[ant2_idxs_clock]
            diff_clock[ant1_dt] -= antenna1_handle.clock_samples[ant1_idxs_clock]
            _, ant1_idxs_phase, ant1_dt = np.intersect1d(antenna1_handle.phase_clock_times, \
                    baseline_handle.datetime_array, return_indices=True)
            _, ant2_idxs_phase, ant2_dt = np.intersect1d(antenna2_handle.phase_clock_times, \
                    baseline_handle.datetime_array, return_indices=True)
            diff_clock_phase = np.zeros(len(baseline_handle.datetime_array))

            diff_clock_phase[ant2_dt] += antenna2_handle.phase_clock_samples[ant2_idxs_phase]
            diff_clock_phase[ant1_dt] -= antenna1_handle.phase_clock_samples[ant1_idxs_phase]

        dc_range = diff_clock[baseline_handle.range_only_idxs]
        dc_phase = diff_clock_phase[baseline_handle.phase_data_idxs]       

        # adjust phase clock to range clock
        N_int = np.round(np.mean(diff_clock-diff_clock_phase)/baseline_handle.wavelength)
        dc_phase = dc_phase + N_int*baseline_handle.wavelength

        baseline_handle = baseline_handles[jdx]
        residuals_range_bl = residuals_range[num_samples_range:\
                                 num_samples_range + len(baseline_handle.range_only_idxs)]
        num_samples_range = num_samples_range + len(baseline_handle.range_only_idxs)
        residuals_phase_bl = residuals_phase[num_samples_phase:\
                                 num_samples_phase + len(baseline_handle.phase_data_idxs)]
        num_samples_phase = num_samples_phase + len(baseline_handle.phase_data_idxs)

        range_weight_mat = get_obs_weights(store_handle, antenna1_handle, antenna2_handle, 'range', \
                    baseline_handle.range_only_idxs, baseline_handle)
        phase_weight_mat = get_obs_weights(store_handle, antenna1_handle, antenna2_handle, 'phase', \
                    baseline_handle.phase_data_idxs, baseline_handle, True)

        residuals_range_bl = np.linalg.inv(range_weight_mat)@residuals_range_bl
        residuals_phase_bl = np.linalg.inv(phase_weight_mat)@residuals_phase_bl
         
        full_time_hr = (baseline_handle.datetime_array[-1]-baseline_handle.datetime_array[0])/(3600*np.timedelta64(1,'s'))
        interval_hr = int(np.ceil(full_time_hr/8))        

        datetimes = to_datetime(baseline_handle.datetime_array)
        dt_phase = datetimes[baseline_handle.phase_data_idxs]
        dt_range = datetimes[baseline_handle.range_only_idxs]

        phase_amb = baseline_handle.wavelength/const.c*1e12 # ps

        # plot the non-outliers to see trends better
        dt_range_f = DataFrame({'Datetime': dt_range})
        dt_phase_f = DataFrame({'Datetime': dt_phase})
        dt_range_f.set_index('Datetime', inplace=True)
        range_index_array = dt_range_f.index.to_numpy()
        dt_phase_f.set_index('Datetime', inplace=True)
        phase_index_array = dt_phase_f.index.to_numpy()
        WRMS_range = np.sqrt(np.sum((range_weight_mat@residuals_range_bl)**2)/np.trace(range_weight_mat@range_weight_mat.T))/const.c*1e12
        WRMS_phase = np.sqrt(np.sum((phase_weight_mat@residuals_phase_bl)**2)/np.trace(phase_weight_mat@phase_weight_mat.T))/const.c*1e12
        RMS_phase= np.sqrt(np.sum(residuals_phase_bl**2)/len(residuals_phase_bl))/const.c*1e12
        print('phase RMS: ' + str(RMS_phase))

        data_range_full = dc_range + residuals_range_bl
        time_array = (range_index_array-range_index_array[0])/np.timedelta64(1, 's')
        res_range = linregress(time_array, dc_range)
        best_fit_line_range = time_array*res_range.slope + res_range.intercept
        data_range_full_var = data_range_full - best_fit_line_range
        data_range_clock_var = dc_range - best_fit_line_range

        data_phase_full = dc_phase + residuals_phase_bl
        time_array = (phase_index_array-phase_index_array[0])/np.timedelta64(1, 's')
        res_phase = linregress(time_array, dc_phase)
        best_fit_line_phase = time_array*res_phase.slope + res_phase.intercept
        data_phase_full_var = data_phase_full - best_fit_line_phase
        data_phase_clock_var = dc_phase - best_fit_line_phase

        # plot residuals
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.scatter(phase_index_array, residuals_phase_bl/const.c*1e12, marker='o', \
                label=label2, zorder=10, color=custom_colors[0])
        ax1.scatter(range_index_array, residuals_range_bl/const.c*1e12, marker='s', \
                label=label, zorder=5, color=custom_colors[1])
        #ax1.errorbar(phase_index_array, residuals_phase_bl/const.c*1e12, yerr=1/phase_weights/const.c*1e12, marker='o', \
        #        linestyle='', label='phase res.', zorder=10, linestyle='', color=custom_colors[0], elinewidth=0.5, capsize=5)
        #ax1.errorbar(range_index_array, residuals_range_bl/const.c*1e12, yerr=1/range_weights/const.c*1e12, marker='o', \
        #        linestyle='', label=label+' res.', zorder=5, color=custom_colors[1], elinewidth=0.5, capsize=5)
        ax1.legend()
        half_interval = phase_amb / 2
        # Find the largest gap in datetime values by first converting them to numeric values
        x_numeric = mdates.date2num(phase_index_array)
        x_gaps = np.diff(x_numeric)  # Calculate differences between consecutive x values
        max_gap_idx = np.argmax(x_gaps)  # Find index of the largest gap
        x_clear_numeric = (x_numeric[max_gap_idx] + x_numeric[max_gap_idx + 1]) / 2  # Midpoint of the largest gap

        # Convert this numeric x back to a datetime for plotting
        x_clear_datetime = mdates.num2date(x_clear_numeric)
        # Calculate average distance between points and set x_span to 1/4 of that
        average_gap = np.mean(x_gaps)
        x_span_numeric = average_gap / 4  # Quarter of the average gap

        x_span_timedelta = to_timedelta(x_span_numeric, unit='D')  # Since x_numeric is in days
        
        # Plot the ambiguity interval centered at y=0 with a small width for the horizontal lines
        #ax1.hlines([-half_interval, half_interval], xmin=x_clear_datetime - x_span_timedelta, xmax=x_clear_datetime + x_span_timedelta, color='red')
        #ax1.vlines(x_clear_datetime, ymin=-half_interval, ymax=half_interval, color='red')
        
        # Optional: add label for the ambiguity interval
        #ax1.text(x_clear_datetime, 0, 'Ambiguity Interval', color='red', va='center')

        # Formatting the date on the x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hr))  # Adjust interval as needed
        fig.autofmt_xdate()  # Auto-rotate date labels
        
        ax1.set_title(antenna2_handle.antenna_name + '—' + antenna1_handle.antenna_name\
                +', phase ambiguity ' + str(np.round(phase_amb,decimals=1)) + ' ps, WRMS (' + label +  ') : ' + str(np.round(WRMS_range,decimals=1))+ ' ps,'\
                + ' WRMS (phase) : ' + str(np.round(WRMS_phase,decimals=1))+ ' ps'   )
        ax1.set_ylabel('residuals (ps)')
        ax1.grid(True)
        if iono_free is True:
            fig.savefig(sol_type+'_' + label.replace(' ','_') + '_only_phase_only_meas_residuals_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '_ionofree.png')
        else:
            fig.savefig(sol_type+'_' + label.replace(' ','_') + '_only_phase_only_meas_residuals_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '.png')
        plt.close(fig)

        # plots with windup effect included
        if store_handle.sol_type == 'VLBI':
            residuals_windup = (residuals_phase_bl+(baseline_handle.cpw_2-baseline_handle.cpw_1)[baseline_handle.phase_data_idxs])
            WRMS_windup = np.sqrt(np.sum((phase_weight_mat@residuals_windup)**2)/np.trace(phase_weight_mat@phase_weight_mat.T))/const.c*1e12
            #fig, ax1 = plt.subplots(figsize=(10, 6))
            #ax1.scatter(phase_index_array, residuals_windup/const.c*1e12, marker='o', \
            #        label=label2, zorder=10, color=custom_colors[0])
            #ax1.scatter(range_index_array, residuals_range_bl/const.c*1e12, marker='s', \
            #        label=label, zorder=5, color=custom_colors[1])
            #ax1.legend()

            ## Formatting the date on the x-axis
            #ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            #ax1.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hr))  # Adjust interval as needed
            #fig.autofmt_xdate()  # Auto-rotate date labels
            #ax1.set_title(antenna2_handle.antenna_name + '—' + antenna1_handle.antenna_name\
            #        +', WRMS (' + label +  ') : ' + str(np.round(WRMS_range,decimals=1))+ ' ps,' + ' WRMS (phase) : ' + str(np.round(WRMS_windup,decimals=1))+ ' ps'   )
            #ax1.set_ylabel('residuals (ps)')
            #ax1.grid(True)
            #if iono_free is True:
            #    fig.savefig(sol_type+'_' + label.replace(' ','_') + '_only_phase_only_meas_residuals_nowindup_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '_ionofree.png')
            #else:
            #    fig.savefig(sol_type+'_' + label.replace(' ','_') + '_only_phase_only_meas_residuals_nowindup'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '.png')
            #plt.close(fig)

            # plot of phase with/without windup effect included
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.scatter(phase_index_array, residuals_phase_bl/const.c*1e12, marker='o', \
                    label='differential feed rotation corrected', zorder=5, color=custom_colors[0])
            ax1.scatter(phase_index_array, residuals_windup/const.c*1e12, marker='d', \
                    label='without correction', zorder=10, color=custom_colors[2])
            ax1.legend()

            # Formatting the date on the x-axis
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            ax1.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hr))  # Adjust interval as needed
            fig.autofmt_xdate()  # Auto-rotate date labels
            ax1.set_title(antenna2_handle.antenna_name + '—' + antenna1_handle.antenna_name\
                    + ', WRMS (corrected) : ' + str(np.round(WRMS_phase,decimals=1))+ ' ps,' + ' WRMS (without correction) : ' + str(np.round(WRMS_windup,decimals=1))+ ' ps'   )
            ax1.set_ylabel('phase residuals (ps)')
            ax1.grid(True)
            if iono_free is True:
                fig.savefig('phase_only_meas_residuals_nowindup_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '_ionofree.png')
            else:
                fig.savefig('phase_only_meas_residuals_nowindup'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '.png')
            plt.close(fig)

        # plot clock residuals
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.scatter(phase_index_array, data_phase_full/const.c*1e12, marker='o', \
                label='phase res. + clock', zorder=10, color=custom_colors[0])
        #ax1.errorbar(phase_index_array, data_phase_full/const.c*1e12, yerr=1/phase_weights/const.c*1e12, marker='o', \
        #        linestyle='', label='phase res. + clock', zorder=10, color=custom_colors[0], elinewidth=0.5, capsize=5)
        ax1.plot(phase_index_array, dc_phase/const.c*1e12, linestyle='--', label='phase clock function', zorder=0, color=custom_colors[0])
        ax1.scatter(range_index_array, data_range_full/const.c*1e12, marker='s', \
                label=label+' res. + clock', zorder=5, color=custom_colors[1])
        #ax1.errorbar(range_index_array, data_range_full/const.c*1e12, yerr=1/range_weights/const.c*1e12, marker='o', \
        #        linestyle='', label=label+' res. + clock', zorder=5, color=custom_colors[1], elinewidth=0.5, capsize=5)
        ax1.plot(range_index_array, dc_range/const.c*1e12, linestyle='--', label=label+' clock function', zorder=1, color=custom_colors[1])
        ax1.legend()

        # Formatting the date on the x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hr))  # Adjust interval as needed
        fig.autofmt_xdate()  # Auto-rotate date labels
        ax1.set_title(antenna2_handle.antenna_name + '—' + antenna1_handle.antenna_name\
                + ', WRMS (' + label +  ') : ' + str(np.round(WRMS_range,decimals=1))+ ' ps,' + ' WRMS (phase) : ' + str(np.round(WRMS_phase,decimals=1))+ ' ps'   )
        ax1.set_ylabel('residuals + clock function (ps)')
        ax1.grid(True)
        if iono_free is True:
            fig.savefig(sol_type+'_'+label.replace(' ','_')+'_only_phase_only_clock_residuals_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '_ionofree.png')
        else:
            fig.savefig(sol_type+'_'+label.replace(' ','_')+'_only_phase_only_clock_residuals_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '.png')
        plt.close(fig)

        ## plot clock variation residuals
        #fig, ax1 = plt.subplots(figsize=(10, 6))
        #ax1.scatter(phase_index_array, data_phase_full_var/const.c*1e12, marker='o', \
        #        label='phase res. + clock', zorder=10, color=custom_colors[0])
        ##ax1.errorbar(phase_index_array, data_phase_full_var/const.c*1e12, yerr=1/phase_weights/const.c*1e12, marker='o', \
        ##        linestyle='', label='phase res. + clock', zorder=10, color=custom_colors[0], elinewidth=0.5, capsize=5)
        #ax1.plot(phase_index_array, data_phase_clock_var/const.c*1e12, linestyle='--', label='phase clock function', zorder=0, color=custom_colors[0])
        #ax1.scatter(range_index_array, data_range_full_var/const.c*1e12, marker='s', \
        #        label=label+' res. + clock', zorder=5, color=custom_colors[1])
        ##ax1.errorbar(range_index_array, data_range_full_var/const.c*1e12, yerr=1/range_weights/const.c*1e12, marker='o', \
        ##        linestyle='', label=label+' res. + clock', zorder=5, color=custom_colors[1], elinewidth=0.5, capsize=5)
        #ax1.plot(range_index_array, data_range_clock_var/const.c*1e12, linestyle='--', label=label+' clock function', zorder=1, color=custom_colors[1])
        #ax1.legend()

        ## Formatting the date on the x-axis
        #ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        #ax1.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hr))  # Adjust interval as needed
        #fig.autofmt_xdate()  # Auto-rotate date labels
        #
        #ax1.set_title(antenna2_handle.antenna_name + '—' + antenna1_handle.antenna_name\
        #        + ', WRMS (' + label +  ') : ' + str(np.round(WRMS_range,decimals=1))+ ' ps,' + ' WRMS (phase) : ' + str(np.round(WRMS_phase,decimals=1))+ ' ps'   )
        #ax1.set_ylabel('residuals + clock variation (ps)')
        #ax1.grid(True)
        #if iono_free is True:
        #    fig.savefig(sol_type+'_'+label.replace(' ','_')+'_only_phase_only_clock_variation_residuals_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '_ionofree.png')
        #else:
        #    fig.savefig(sol_type+'_'+label.replace(' ','_')+'_only_phase_only_clock_variation_residuals_'+antenna2_handle.antenna_name+'_' + antenna1_handle.antenna_name + '.png')
        #plt.close(fig)

def detect_unresolved_amb_vlbi(iono_free, amb_state, baseline_handles):
    """ find if any cycle slips remain in the data """
    slips_detected = False 
    slips_full = []
    slips_baseline = []
    for jdx, baseline_handle in enumerate(baseline_handles): # generate differential measurements on the baselines
        times_diff = (baseline_handle.datetime_array-baseline_handle.datetime_array[0])/np.timedelta64(1, 's')         
     
        phase_delay_model = baseline_handle.phase_delay_model
   
        # get data and model carrier phase observations
        if iono_free is True:
            phase_delays = baseline_handle.combination_model(baseline_handle.phase_delays, \
                    baseline_hande.phase_delays_dual, combination_type) # deepcopy to avoid changing elements in baseline array
            phase_delay_dual_model = baseline_handle.phase_delay_dual_model  
            phase_delay_model = baseline_handle.combination_model(phase_delay_model, phase_delay_dual_model, combination_type)
            wavelength = baseline_handle.comb_wavelength
        else:
            phase_delays = deepcopy(baseline_handle.phase_delays) # deepcopy to avoid changing elements in baseline array
            wavelength = baseline_handle.wavelength

        # subtract float or integer ambiguities from data
        slip_slices_arr = baseline_handle.slip_slices_arr
        for idx, slip_slice in enumerate(slip_slices_arr):
            phase_delays[slip_slice] = phase_delays[slip_slice] + wavelength*amb_state[idx]

        sigma_pd, use_idxs_pd = find_sigmas(phase_delays-phase_delay_model)
        slips = slip_detect_full(baseline_handle.f1, phase_delays, phase_delay_model, times_diff)
        if len(slips) > 0:
            slips = np.intersect1d(slips,np.argwhere(use_idxs_pd)) # take only non-outlier slips
            slips = np.intersect1d(slips, baseline_handle.phase_data_idxs)
            slips_full.extend(slips)
        slips_baseline.append(slips)
    if len(slips_full) > 0: slips_detected = True

    return slips_detected, slips_full, slips_baseline

def detect_unresolved_amb_gnss(store_handle, amb_state, antenna_handles, baselines, baseline_handles):
    """ find if any cycle slips remain in the data """
    slips_detected = False 
    slips_full = []
    slips_baseline = []
    for jdx, baseline in enumerate(baselines): # generate differential measurements on the baselines
        baseline_handle = baseline_handles[jdx]
        antenna1_handle = antenna_handles[baseline[0]]
        antenna2_handle = antenna_handles[baseline[1]]
        data_ant1 = antenna1_handle.antenna_data
        data_ant2 = antenna2_handle.antenna_data
        
        _, ant1_idxs, _ = np.intersect1d(antenna1_handle.times_gps, baseline_handle.datetime_array, return_indices=True)
        _, ant2_idxs, _ = np.intersect1d(antenna2_handle.times_gps, baseline_handle.datetime_array, return_indices=True)
        times_diff = (baseline_handle.datetime_array-baseline_handle.datetime_array[0])/np.timedelta64(1, 's')         
        
        diff_cp_model = data_ant2.cp_model.values[ant2_idxs] - data_ant1.cp_model.values[ant1_idxs]

        if store_handle.iono_comp_l4r and not store_handle.iono_free:
            source_array = [store_handle.source_time_dict[time] for time in baseline_handle.datetime_array]
            stec_vals = store_handle.interp_l4r(baseline_handle.datetime_array, source_array, antenna1_handle.l4r_name, antenna2_handle.l4r_name)
            diff_cp_model += ALPHA_IONO/baseline_handle.f1**2*const.c*stec_vals
   
        # get data and model carrier phase observations
        if store_handle.iono_free is True:
            diff_cp_data = deepcopy(baseline_handle.cp_combination) # deepcopy to avoid changing elements in baseline array
            diff_cp_dual_model = data_ant2.cp_dual_model.values[ant2_idxs] - data_ant1.cp_dual_model.values[ant1_idxs]  
            diff_cp_model = baseline_handle.combination_model(diff_cp_model, diff_cp_dual_model, combination_type)
            wavelength = baseline_handle.comb_wavelength
        else:
            diff_cp_data = deepcopy(baseline_handle.cp_diff)
            wavelength = baseline_handle.wavelength

        # subtract float or integer ambiguities from data
        slip_slices_arr = baseline_handle.slip_slices_arr
        for idx, slip_slice in enumerate(slip_slices_arr):
            diff_cp_data[slip_slice] = diff_cp_data[slip_slice] + wavelength*amb_state[idx]
        
        sigma_pd, use_idxs_pd = find_sigmas(diff_cp_data-diff_cp_model)
        slips = slip_detect_full(baseline_handle.f1, diff_cp_data, diff_cp_model, times_diff)
        if len(slips) > 0:
            slips = np.intersect1d(slips,np.argwhere(use_idxs_pd)) # take only non-outlier slips
            slips = np.intersect1d(slips, baseline_handle.phase_data_idxs)
            slips_full.extend(slips)
        slips_baseline.append(slips)
    if len(slips_full) > 0: slips_detected = True

    return slips_detected, slips_full, slips_baseline

def resolve_float_amb(store_handle, state_expanded, bounds, n_ao_state, n_grav_state, ls_args, res_fcn, jac, sol_type, recursive_amb, L_curve=False):
    """ Resolve the floating point carrier phase ambiguities on each baseline with LAMBDA """
    N_CANDS = 3
    N_MAX = 100
    PS = 0.98
    n_last = 0
    n_fixed = 0
    n_tot = np.inf
    n_iter = 0
    (ref_antenna, baselines, store_handle, antenna_handles, clock_idxs,  \
          clock_poly_length, trop_idxs, trop_poly_length, disb_idxs, baseline_handles, \
          phase_delay, phase_only, use_amb_state, amb_state_idxs, use_phase_weights, tikhonov_lambda, cont_penalty, phase_clock_idxs, \
          phase_disb_idxs, combination_type) = ls_args # retrieve arguments of least_squares function
    while n_tot > n_fixed:
        if n_iter == 0:
            if L_curve is True:
                NUM=150 
                tikhonov_array = np.logspace(-16,-2,num=NUM)
                #cont_array = np.logspace(-1,2,num=NUM*0)
                cont_array = [0]
                res_array = np.zeros((len(tikhonov_array),len(cont_array)))
                cond_array = np.zeros((len(tikhonov_array),len(cont_array)))
                wrms_array = np.zeros((len(tikhonov_array),len(cont_array)))
                wrms_full_array = np.zeros((len(tikhonov_array),len(cont_array)))
                state_array = np.zeros((len(state_expanded[amb_state_idxs]), len(tikhonov_array), len(cont_array)))
                for idx, tikhonov_test in enumerate(tikhonov_array):
                    for jdx, cont_penalty in enumerate(cont_array):
                        if not jdx%10:
                            iter_num = idx*len(cont_array)+jdx
                            print(f'L curve iteration {iter_num} of {len(tikhonov_array)*len(cont_array)}')
                        store_handle.hold_state(state_expanded*0)
                        ls_phfloat = least_squares(res_fcn, state_expanded, jac=jac, method='trf',\
                            max_nfev=100, bounds=bounds, verbose=0, x_scale = 'jac', xtol=1e-15,\
                            args=(ref_antenna, baselines, store_handle, antenna_handles, clock_idxs, \
                                  clock_poly_length, trop_idxs, trop_poly_length, disb_idxs, baseline_handles, \
                                  phase_delay, phase_only, use_amb_state, amb_state_idxs, use_phase_weights, tikhonov_test, cont_penalty, phase_clock_idxs, \
                                  phase_disb_idxs, combination_type))
                        residuals = ls_phfloat.fun
                        residuals_range, residuals_phase = get_residuals(residuals, baseline_handles) 
                        norm_res = np.sum(np.concatenate((residuals_range,residuals_phase))**2)
                        res_array[idx,jdx] = norm_res
                        state_array[:,idx,jdx] = ls_phfloat.x[amb_state_idxs]
                        phase_weight_mat = get_obs_weights(store_handle, antenna_handles[0], antenna_handles[1], 'phase', \
                                baseline_handles[0].phase_data_idxs, baseline_handles[0], use_phase_weights)
                        wrms_array[idx,jdx] = np.sqrt(np.sum(ls_phfloat.fun**2))
                        wrms_full_array[idx,jdx] = np.sqrt(np.sum(residuals_phase**2)/np.trace(phase_weight_mat@phase_weight_mat.T))/const.c*1e12
                        cond_array[idx,jdx] = np.linalg.cond(ls_phfloat.jac.T@ls_phfloat.jac)

                # First plot: x axis is tikhonov_array, y axis is cont_array, color axis is res_array
                if len(cont_array)>1 and len(tikhonov_array)>1:
                    plt.figure(figsize=(8, 6))
                    #plt.contourf(tikhonov_array, cont_array, res_array.T, norm=colors.LogNorm(), cmap='viridis')
                    plt.contourf(tikhonov_array, cont_array, res_array.T, cmap='viridis')
                    plt.xscale('log')
                    plt.colorbar(label='$||y-h(x)||^2$')
                    plt.ylabel('$\\alpha$')
                    plt.xlabel('$\\lambda$')
                    plt.savefig(store_handle.sol_type+'_'+antenna_handles[0].antenna_name+'_'+'_l_curve_res.png')
                    
                    # Find the minimum value in res_array and corresponding tikhonov_array, cont_array
                    min_idx_res = np.unravel_index(np.argmin(res_array), res_array.shape)
                    best_tikhonov_res = tikhonov_array[min_idx_res[0]]
                    best_cont_res = cont_array[min_idx_res[1]]
                    print(f'Best Tikhonov parameter for residuals: {best_tikhonov_res}')
                    print(f'Best Continuity penalty for residuals: {best_cont_res}')
                    
                    # Second plot: x axis is tikhonov_array, y axis is cont_array, color axis is normalized state_array
                    # Normalize state_array along the first axis (i-axis)
                    state_norm = np.abs(state_array)
                    for i in range(state_array.shape[0]):
                        state_norm[i] /= np.max(state_norm[i])
                    
                    # Compute the norm over the first axis (i-axis) of the normalized state array
                    state_norm_sum = np.linalg.norm(state_norm, axis=0)
                    
                    plt.figure(figsize=(8, 6))
                    #plt.contourf(tikhonov_array, cont_array, state_norm_sum.T, norm=colors.LogNorm(), cmap='plasma')
                    plt.contourf(tikhonov_array, cont_array, state_norm_sum.T, cmap='viridis')
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.colorbar(label='$||x^*||^2$')
                    plt.ylabel('$\\alpha$')
                    plt.xlabel('$\\lambda$')

                    plt.savefig(store_handle.sol_type+'_'+antenna_handles[0].antenna_name+'_'+'_l_curve_state.png')
                    
                    # Find the minimum value in state_norm_sum and corresponding tikhonov_array, cont_array
                    min_idx_state = np.unravel_index(np.argmin(state_norm_sum), state_norm_sum.shape)
                    best_tikhonov_state = tikhonov_array[min_idx_state[0]]
                    best_cont_state = cont_array[min_idx_state[1]]
                    print(f'Best Tikhonov parameter for state norm: {best_tikhonov_state}')
                    print(f'Best Continuity penalty for state norm: {best_cont_state}')

                    plt.figure(figsize=(8, 6))
                    plt.contourf(tikhonov_array, cont_array, wrms_array.T, cmap='viridis')
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.colorbar(label='phase WRMS (ps)')
                    plt.ylabel('$\\alpha$')
                    plt.xlabel('$\\lambda$')
                    plt.savefig(store_handle.sol_type+'_'+antenna_handles[0].antenna_name+'_'+'_l_curve_wrms.png')
                    
                    # Find the minimum value in state_norm_sum and corresponding tikhonov_array, cont_array
                    min_idx_wrms = np.unravel_index(np.argmin(wrms_array), wrms_array.shape)
                    best_tikhonov_wrms = tikhonov_array[min_idx_wrms[0]]
                    best_cont_wrms = cont_array[min_idx_wrms[1]]
                    print(f'Best Tikhonov parameter for WRMS: {best_tikhonov_wrms}')
                    print(f'Best Continuity penalty for WRMS: {best_cont_wrms}')
                elif len(tikhonov_array) > 1:
                    #state_norm = np.abs(state_array)
                    #for i in range(state_array.shape[0]):
                    #    state_norm[i] /= np.abs(state_norm[i][0])
                    plt.figure(figsize=(8, 6))
                    plt.plot(tikhonov_array, res_array)
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.ylabel('$||y-h(p)||^2$')
                    plt.xlabel('$\\lambda_t$')
                    plt.savefig(store_handle.sol_type+'_'+antenna_handles[0].antenna_name+'_'+'_l_curve_res.png')
                    
                    # Find the minimum value in res_array and corresponding tikhonov_array, cont_array
                    min_idx_res = np.argmin(res_array)
                    best_tikhonov_res = tikhonov_array[min_idx_res]
                    print(f'Best Tikhonov parameter for residuals: {best_tikhonov_res}')
                    
                    # Second plot: x axis is tikhonov_array, y axis is cont_array, color axis is normalized state_array
                    # Normalize state_array along the first axis (i-axis)
                    #state_norm = np.abs(state_array)
                    #for i in range(state_array.shape[0]):
                    #     state_norm[i] /= np.max(state_norm[i])
                    
                    # Compute the norm over the first axis (i-axis) of the normalized state array
                    #state_norm_sum = np.linalg.norm(state_array, axis=0)
                    #state_norm_sum = np.linalg.norm(state_norm, axis=0)
                    state_norm_sum = np.linalg.norm(state_array, axis=0)
                    
                    plt.figure(figsize=(8, 6))
                    plt.plot(tikhonov_array, state_norm_sum)
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.ylabel('$||\\Gamma p||$')
                    plt.xlabel('$\\lambda_t$')
                    plt.savefig(store_handle.sol_type+'_'+antenna_handles[0].antenna_name+'_'+'_l_curve_state.png')
                    
                    # Find the minimum value in state_norm_sum and corresponding tikhonov_array, cont_array
                    min_idx_state = np.argmin(state_norm_sum)
                    best_tikhonov_state = tikhonov_array[min_idx_state]
                    print(f'Best Tikhonov parameter for state norm: {best_tikhonov_state}')

                    plt.figure(figsize=(8, 6))
                    plt.plot(tikhonov_array, wrms_array)
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.ylabel('phase WRMS (ps)')
                    plt.xlabel('$\\lambda_t$')
                    plt.savefig(store_handle.sol_type+'_'+antenna_handles[0].antenna_name+'_'+'_l_curve_wrms.png')


                    # single plot
                    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

                     # Plot condition number
                    axs[0].plot(tikhonov_array, cond_array)
                    axs[0].set_yscale('log')
                    axs[0].set_ylabel('condition number')                  
                    
                    # Plot WRMS
                    axs[1].plot(tikhonov_array, res_array)
                    axs[1].set_yscale('log')
                    axs[1].set_ylabel('$||y-h(p)||^2$')

                    # Plot state norm
                    axs[2].plot(tikhonov_array, state_norm_sum)
                    axs[2].set_yscale('log')
                    axs[2].set_ylabel('$||\\Gamma p||$')
                    axs[2].set_xscale('log')
                    axs[2].set_xlabel('$\\lambda_t$')
                    fig.tight_layout()
                    plt.savefig(f"{store_handle.sol_type}_{antenna_handles[0].antenna_name}_l_curve_combined.png")
                    plt.close(fig)

                    plt.figure(figsize=(8, 6))
                    plt.plot(tikhonov_array, cond_array)
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.ylabel('condition number')
                    plt.xlabel('$\\lambda_t$')
                    plt.savefig(store_handle.sol_type+'_'+antenna_handles[0].antenna_name+'_'+'_l_curve_cond.png')

                    plt.figure(figsize=(8, 6))
                    plt.plot(wrms_full_array, state_norm_sum)
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.ylabel('$||\\Gamma p||$')
                    plt.xlabel('$\\lambda_t$')
                    plt.savefig(store_handle.sol_type+'_'+antenna_handles[0].antenna_name+'_'+'_l_curve_res_state.png')

                    plt.figure(figsize=(8, 6))
                    plt.plot(res_array, state_norm_sum, zorder=0)
                    sc = plt.scatter(res_array, state_norm_sum, c=tikhonov_array, cmap='viridis', s=100, norm=colors.LogNorm(),zorder=1)
                    cbar = plt.colorbar(sc)
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.xlabel('$||y-h(p)||$')
                    plt.ylabel('$||\\Gamma p||$')
                    cbar.set_label('$\\lambda_t$')
                    plt.savefig(store_handle.sol_type+'_'+antenna_handles[0].antenna_name+'_'+'_l_curve_res_state_tik.png')

                    
                    # Find the minimum value in state_norm_sum and corresponding tikhonov_array, cont_array
                    min_idx_wrms = np.argmin(wrms_array)
                    best_tikhonov_wrms = tikhonov_array[min_idx_wrms]
                    print(f'Best Tikhonov parameter for WRMS: {best_tikhonov_wrms}')
                raise ValueError("Exit the program")

  
            # initial attempt with fully floated ambiguities
            cov_cond = np.inf 
            iter_cond = 0
            while cov_cond > 1e13:
                # dynamically adjust tikhonov_lambda upwards
                if iter_cond > 0:
                    tikhonov_lambda *= 1e2
                ls_phfloat = least_squares(res_fcn, state_expanded, jac=jac, method='trf',\
                    max_nfev=100, bounds=bounds, verbose=2, x_scale = 'jac', xtol=1e-15,\
                    args=(ref_antenna, baselines, store_handle, antenna_handles, clock_idxs, \
                          clock_poly_length, trop_idxs, trop_poly_length, disb_idxs, baseline_handles, \
                          phase_delay, phase_only, use_amb_state, amb_state_idxs, use_phase_weights, \
                          tikhonov_lambda, cont_penalty, phase_clock_idxs, phase_disb_idxs, combination_type))
                J = ls_phfloat.jac  # Jacobian of the solution
                cov_cond = np.linalg.cond(J.T@J)
                iter_cond += 1
                print('iter_cond:' + str(iter_cond))
                print(iter_cond)
                if tikhonov_lambda > 1e-4: break

            state_full = ls_phfloat.x
            clock_state_full = state_full[clock_idxs]
            disb_state_full = state_full[disb_idxs]
            n_tot = len(state_full[amb_state_idxs])
        else: 
            # some ambiguities are fixed, now do phase only solution for remaining ambiguities
            rxpos_ao_grav = ls_phfloat.x[:3*(len(antenna_handles)-1)+n_ao_state+n_grav_state]
            trop_state = ls_phfloat.x[trop_idxs]
            disb_state = ls_phfloat.x[disb_idxs]
            phase_disb_state = ls_phfloat.x[phase_disb_idxs]
            phase_clock = ls_phfloat.x[phase_clock_idxs]
            state_phfloat = np.concatenate((rxpos_ao_grav,trop_state,phase_clock,phase_disb_state,amb_state))
            rxpos_idxs = slice(0,(len(antenna_handles)-1)*3)
            clock_idxs = slice((len(antenna_handles)-1)*3,(len(antenna_handles)-1)*3)
            ao_idxs = slice(clock_idxs.stop,clock_idxs.stop+n_ao_state)
            grav_idxs = slice(clock_idxs.stop+n_ao_state,clock_idxs.stop+n_ao_state+n_grav_state)
            trop_idxs = slice(grav_idxs.stop, grav_idxs.stop+len(trop_state))
            disb_idxs = slice(trop_idxs.stop, trop_idxs.stop)
            phase_clock_idxs = slice(trop_idxs.stop,trop_idxs.stop+len(phase_clock))
            phase_disb_idxs = slice(phase_clock_idxs.stop, phase_clock_idxs.stop+len(phase_disb_state))
            amb_state_idxs = slice(phase_disb_idxs.stop,len(state_phfloat))
            phase_only=True
            store_handle.hold_state(state_phfloat+1e-9)

            print('Attempt to fix remaining float ambiguities') 
            (bound_low, bound_high) = bounds
            bound_low_reduced = union_of_slices(bound_low, rxpos_idxs, ao_idxs, grav_idxs, trop_idxs, disb_idxs, phase_clock_idxs, phase_disb_idxs, amb_state_idxs)
            bound_high_reduced = union_of_slices(bound_high, rxpos_idxs, ao_idxs, grav_idxs, trop_idxs, disb_idxs, phase_clock_idxs, phase_disb_idxs, amb_state_idxs)
            ls_phfloat = least_squares(res_fcn, state_phfloat, jac=jac, method='trf',\
                max_nfev=100, bounds=(bound_low_reduced, bound_high_reduced), verbose=2, x_scale = 'jac', xtol=1e-15,\
                args=(ref_antenna, baselines, store_handle, antenna_handles, clock_idxs, \
                      clock_poly_length, trop_idxs, trop_poly_length, disb_idxs, baseline_handles, \
                      phase_delay, phase_only, use_amb_state, amb_state_idxs, use_phase_weights, tikhonov_lambda, cont_penalty, phase_clock_idxs, \
                      phase_disb_idxs, combination_type, z_par, iZt))

        # temp -- plot each stage
        #sol_name = 'rs_float_'+str(n_iter)
        #analyze_ls_solution(sol_type, True, ref_antenna, clock_idxs, trop_idxs, ls_phfloat, store_handle, antenna_handles, sol_name, baselines,\
        #            n_ao_state, baseline_handles, phase_delay, phase_only, phase_clock_idxs)
        
        float_amb = ls_phfloat.x[amb_state_idxs]
        residuals = ls_phfloat.fun

        unit_var = get_unit_var_full(ls_phfloat.fun, ls_phfloat.x)
        J = ls_phfloat.jac  # Jacobian of the solution
        cov_matrix = np.linalg.inv(J.T @ J)
        cov_matrix *= unit_var

        if n_last == 0: # first iteration
            var_cov_amb = cov_matrix[amb_state_idxs,amb_state_idxs]  # Covariance of ambiguities
            if ((var_cov_amb-var_cov_amb.T)<1e-10).all()==False: 
                var_cov_amb = (var_cov_amb.T + var_cov_amb)/2
                print('ambiguities not symmetric--symmetrized')
                #corr_test = np.corrcoef(J,rowvar=False) # correlation between parameters
                #cond_num = np.linalg.cond(J.T @ J)
                #for i in range(corr_test.shape[0]): corr_test[i,i] = 0 # dont care about diagonal, should be 1
                #bad_params = np.argwhere(corr_test>0.995) # these parameters are nearly collinear
                #bad_vals = corr_test[corr_test>0.995] # degree of collinearity

            # compute the Z-transformation
            print(f'shape var_cov_amb: ({var_cov_amb.shape[0]}, {var_cov_amb.shape[1]})', flush=True)
            Q_zhat,Z,L,D,z_hat,iZt = LAMBDA.decorrel(var_cov_amb, float_amb)
            print('done with decorrelation', flush=True)
            z_hat = Z.T @ float_amb
            Q_ba = cov_matrix[:amb_state_idxs.start,amb_state_idxs]
            Q_bz = Q_ba @ Z

            # do integer least-squares to find best candidates, accept probability >0.98 correctly fixed
            if len(z_hat) < N_MAX:
                z_par,sqnorm,Q_zz_par,Z_transform_par,Ps,n_fixed,z_fixed = LAMBDA.parsearch(z_hat, Q_zhat, Z, L, D, P0=PS, ncands=N_CANDS)
                #z_par,sqnorm,Q_zz_par,Z_transform_par,Ps,n_fixed,z_fixed = LAMBDA.parsearch_fast(z_hat, Q_zhat, Z, L, D, P0=PS, ncands=N_CANDS)
                # evaluate performance of MLAMBDA
                #import time
                #start_time = time.time()
                #for i in range(100): 
                #    z_par1,sqnorm,Q_zz_par,Z_transform_par,Ps,n_fixed,z_fixed = LAMBDA.parsearch_fast(z_hat, Q_zhat, Z, L, D, P0=PS, ncands=N_CANDS)
                #time_1 = time.time()
                #for i in range(100):
                #    z_par2,sqnorm,Q_zz_par,Z_transform_par,Ps,n_fixed,z_fixed = LAMBDA.parsearch(z_hat, Q_zhat, Z, L, D, P0=PS, ncands=N_CANDS)
                #time_2 = time.time()
                #parsearch_fast_len = time_1-start_time
                #parsearch_len = time_2-time_1

                slips_candidate = []
                cost_candidate = []
                # z_fixed -- all ambiguities (successfully fixed and unfixed) where unfixed are adjusted by fixed (length n)
                # z_par -- the subset of successfully fixed ambiguities (length n_fixed)
                # this is confusing nomenclature but is how it's written in the LAMBDA doc
                if N_CANDS > 1 and n_fixed > 0:
                    rxpos_ao_grav = ls_phfloat.x[:3*(len(antenna_handles)-1)+n_ao_state+n_grav_state]
                    trop_state = ls_phfloat.x[trop_idxs]
                    phase_clock = ls_phfloat.x[phase_clock_idxs]
                    phase_disb_state = ls_phfloat.x[phase_disb_idxs]
                    rxpos_idxs = slice(0,(len(antenna_handles)-1)*3)
                    clock_idxs_eval = slice((len(antenna_handles)-1)*3,(len(antenna_handles)-1)*3)
                    ao_idxs_eval = slice(clock_idxs_eval.stop,clock_idxs_eval.stop+n_ao_state)
                    grav_idxs_eval = slice(clock_idxs_eval.stop+n_ao_state,clock_idxs_eval.stop+n_ao_state+n_grav_state)
                    trop_idxs_eval = slice(grav_idxs_eval.stop,grav_idxs_eval.stop+len(ls_phfloat.x[trop_idxs]))
                    disb_idxs_eval = slice(trop_idxs_eval.stop,trop_idxs_eval.stop)
                    phase_clock_idxs_eval = slice(trop_idxs_eval.stop,trop_idxs_eval.stop+len(phase_clock))
                    phase_disb_idxs_eval = slice(phase_clock_idxs_eval.stop,phase_clock_idxs_eval.stop+len(phase_disb_state))
                    phase_only=True
                    num_remaining = n_tot-n_fixed
                    (bound_low, bound_high) = bounds
                    if n_iter == 0:
                        bound_low_reduced = union_of_slices(bound_low, rxpos_idxs, ao_idxs_eval, grav_idxs_eval, trop_idxs_eval, phase_clock_idxs_eval, phase_disb_idxs_eval, amb_state_idxs)
                        bound_high_reduced = union_of_slices(bound_high, rxpos_idxs, ao_idxs_eval, grav_idxs_eval, trop_idxs_eval, phase_clock_idxs_eval, phase_disb_idxs_eval, amb_state_idxs)

                    print('evaluating ' + str(N_CANDS) + ' ambiguity candidates') 
                    for jdx in range(N_CANDS):
                        print(jdx+1)
                        if num_remaining > 0 and n_fixed > 0:
                            z_par_eval = np.ndarray.flatten(np.array(z_par[:,jdx]))
                            amb_state = np.ndarray.flatten(np.round(z_fixed[:num_remaining,jdx]))
                            state_phfloat = np.concatenate((rxpos_ao_grav,trop_state,phase_clock,phase_disb_state,amb_state))
                            store_handle.hold_state(state_phfloat+1e-9)
                            amb_state_idxs_eval = slice(phase_disb_idxs_eval.stop,len(state_phfloat))
                            bound_low_reduced = bound_low_reduced[:len(state_phfloat)]
                            bound_high_reduced = bound_high_reduced[:len(state_phfloat)]
                            ls_phfloat_eval = least_squares(res_fcn, state_phfloat, jac=jac, method='trf',\
                                max_nfev=100, bounds=(bound_low_reduced, bound_high_reduced), verbose=2, x_scale = 'jac', xtol=1e-15,\
                                args=(ref_antenna, baselines, store_handle, antenna_handles, clock_idxs_eval, \
                                      clock_poly_length, trop_idxs_eval, trop_poly_length, disb_idxs_eval, baseline_handles, phase_delay, phase_only, \
                                      use_amb_state, amb_state_idxs_eval, use_phase_weights, tikhonov_lambda, cont_penalty, phase_clock_idxs_eval, \
                                      phase_disb_idxs_eval, combination_type, z_par_eval, iZt))
                            full_z_vec = np.concatenate((amb_state, z_par_eval))
                            a_fixed = np.round(iZt.dot(full_z_vec))
                        else:
                            a_fixed = np.round(iZt.dot(z_fixed[:,jdx]))
                            state_phfloat = np.concatenate((rxpos_ao_grav,trop_state,phase_clock))
                            store_handle.hold_state(state_phfloat+1e-9)
                            amb_state_idxs_eval = []
                            bound_low_reduced = bound_low_reduced[:len(state_phfloat)]
                            bound_high_reduced = bound_high_reduced[:len(state_phfloat)]
                            ls_phfloat_eval = least_squares(res_fcn, state_phfloat, jac=jac, method='trf',\
                                max_nfev=100, bounds=(bound_low_reduced, bound_high_reduced), verbose=2, x_scale = 'jac', xtol=1e-15,\
                                args=(ref_antenna, baselines, store_handle, antenna_handles, clock_idxs_eval, \
                                      clock_poly_length, trop_idxs_eval, trop_poly_length, disb_idxs_eval, baseline_handles, phase_delay, phase_only, \
                                      False, amb_state_idxs_eval, use_phase_weights, tikhonov_lambda, cont_penalty, phase_clock_idxs_eval, \
                                      phase_disb_idxs_eval, combination_type, a_fixed))
                        if sol_type == 'VLBI':
                            slip_detected, slips, _ = detect_unresolved_amb_vlbi(store_handle.iono_free, a_fixed, baseline_handles)
                        elif sol_type == 'GNSS': 
                            slip_detected, slips, _ = detect_unresolved_amb_gnss(store_handle, a_fixed, antenna_handles, baselines, baseline_handles)
                        slips_candidate.append(len(slips))
                        cost_candidate.append(ls_phfloat_eval.cost)
                    #min_slip_jdx = np.argmin(slips_candidate) # if identical, will return 0 
                    min_slip_jdx = np.argmin(cost_candidate) # if identical, will return 0 
                    print('selected candidate ' + str(min_slip_jdx+1) + ' of ' + str(N_CANDS))
                    z_par = np.ndarray.flatten(np.array(z_par[:,min_slip_jdx]))
                    a_fixed = np.round(iZt.dot(z_fixed[:,min_slip_jdx]))
                    z_fixed = z_fixed[:,min_slip_jdx]
                else:
                    z_par = np.ndarray.flatten(np.array(z_par))
                    a_fixed = np.round(iZt.dot(z_fixed))
            else:
                print('Too many ambiguities for LAMBDA. Using integer bootstrapping.')
                if recursive_amb is True:
                    z_fixed = LAMBDA.bootstrap(z_hat, L)
                    a_fixed = np.round(iZt.dot(z_fixed))
                    n_fixed = n_tot
                    num_remaining = 0
                else:
                    z_par,Q_zz_par,Z_transform_par,Ps,n_fixed,z_fixed = parbootstrap(z_hat, Q_zhat, Z, L, D, P0=PS)
                    a_fixed = np.round(iZt.dot(z_fixed))

            #a_fixed, cost = mlambda.mlambda(var_cov_amb, float_amb, p=1)

            # see lambda.pdf pp.9 distributed with LAMBDA.py for explanation of this math
            ls_phfloat.x = ls_phfloat.x[:amb_state_idxs.start]
            ls_phfloat.x = ls_phfloat.x - Q_bz @ np.linalg.inv(Q_zhat) @ (z_hat-np.ndarray.flatten(z_fixed))     

        else:
            # some fixed ambiguities, try to resolve the remaining ones
            # Update the Q_zz variance-covariance and float ambiguity estimate
            Q_zminor = cov_matrix[amb_state_idxs,amb_state_idxs]  # Covariance of ambiguities
 
            Q_bz_minor = cov_matrix[:amb_state_idxs.start,amb_state_idxs]
             
            # rerun parsearch with reduced number of ambiguities, re-calculated covariance
            Z_minor = Z[:len(float_amb),:len(float_amb)]
            L_minor = L[:len(float_amb),:len(float_amb)]
            D_minor = D[:len(float_amb),:len(float_amb)]
            z_par_minor, sqnorm, Qz_minor_par,Z_transform_par_minor,Ps_minor,n_fixed_minor,z_fixed_minor = \
                    LAMBDA.parsearch(float_amb, Q_zminor, Z_minor, L_minor, D_minor, P0=PS, ncands=N_CANDS)
            #z_par_minor, sqnorm, Qz_minor_par,Z_transform_par_minor,Ps_minor,n_fixed_minor,z_fixed_minor = \
            #        LAMBDA.parsearch_fast(float_amb, Q_zminor, Z_minor, L_minor, D_minor, P0=PS, ncands=N_CANDS)
            
            if recursive_amb is True or n_fixed_minor > 0:
                Q_bz = cov_matrix[:amb_state_idxs.start,amb_state_idxs]
                ls_phfloat.x = ls_phfloat.x[:amb_state_idxs.start]
                slips_candidate = []
                if N_CANDS > 1 and n_fixed_minor > 0:
                    for jdx in range(N_CANDS):
                        z_fixed = np.concatenate((np.ndarray.flatten(np.round(z_fixed_minor[:,jdx])),z_par))
                        a_fixed = np.round(iZt.dot(z_fixed))
                        if sol_type == 'VLBI':
                            slip_detected, slips, _ = detect_unresolved_amb_vlbi(store_handle.iono_free, a_fixed, baseline_handles)
                        elif sol_type == 'GNSS': 
                            slip_detected, slips, _ = detect_unresolved_amb_gnss(store_handle, a_fixed, antenna_handles, baselines, baseline_handles)
                        slips_candidate.append(len(slips))
                    min_slip_jdx = np.argmin(slips_candidate)
                    z_fixed = np.concatenate((np.ndarray.flatten(np.round(z_fixed_minor[:,min_slip_jdx])),z_par))
                    if len(z_par_minor) > 0:
                        z_fixed_minor = z_fixed_minor[:,min_slip_jdx]
                        z_par = np.concatenate((np.ndarray.flatten(z_par_minor[:,min_slip_jdx]), z_par))
                    a_fixed = np.round(iZt.dot(z_fixed))
                else:
                    z_fixed = np.concatenate((np.ndarray.flatten(z_fixed_minor), z_par))
                    if len(z_par_minor) > 0:
                        z_par = np.concatenate((np.ndarray.flatten(z_par_minor), z_par))
                    a_fixed = np.round(iZt.dot(z_fixed))
                # update state for the information we've gained from fixing an ambiguity
                ls_phfloat.x = ls_phfloat.x - Q_bz @ np.linalg.inv(Q_zminor) @ (float_amb-np.ndarray.flatten(z_fixed_minor))

                # place fixed ambiguities in right place, remove from float state
                n_fixed = n_fixed + n_fixed_minor 

        num_remaining = n_tot-n_fixed
        amb_state = np.ndarray.flatten(z_fixed[:num_remaining])
        if n_fixed <= n_last:
            print('Insufficient confidence in ' + str(len(a_fixed)-n_fixed) +' float ambiguities')
            if recursive_amb is True: 
                print('fixing ambiguities anyway (recursive solution)')
                if n_fixed > 0: 
                    z_fixed = np.concatenate((np.ndarray.flatten(np.round(z_fixed_minor)), z_par))
                a_fixed = np.round(iZt.dot(z_fixed))
                num_remaining = 0
                n_tot=n_fixed
            else:
                break
        elif n_last==0:
            print('Fixed ' + str(n_fixed) +' of ' + str(n_tot) + ' float ambiguities')
        else:
            print('Fixed an additional ' + str(n_fixed-n_last) +' float ambiguities')
        n_last = n_fixed
        n_iter += 1
    
    z_fixed = np.ndarray.flatten(z_fixed)
    if n_iter > 1 or num_remaining != 0:
        rxpos_state = ls_phfloat.x[:3*(len(antenna_handles)-1)]
        clock_state = clock_state_full
        phase_clock_state = ls_phfloat.x[phase_clock_idxs]
        ao_idxs = slice(clock_idxs.stop,clock_idxs.stop+n_ao_state)
        grav_idxs = slice(clock_idxs.stop+n_ao_state,clock_idxs.stop+n_ao_state+n_grav_state)
        ao_state = ls_phfloat.x[ao_idxs]
        grav_state = ls_phfloat.x[grav_idxs]
        trop_state = ls_phfloat.x[trop_idxs]
        disb_state = disb_state_full
        phase_disb_state = ls_phfloat.x[phase_disb_idxs]
        if num_remaining == 0:
            state_full = np.concatenate((rxpos_state,clock_state,ao_state,grav_state,trop_state,disb_state,phase_clock_state,phase_disb_state))
        else:
            state_full = np.concatenate((rxpos_state,clock_state,ao_state,grav_state,trop_state,disb_state,phase_clock_state,phase_disb_state,amb_state))
    else:
        state_full = ls_phfloat.x
    return a_fixed, ls_phfloat, state_full, num_remaining, z_fixed, iZt, phase_only


def iterative_weight_adjust_simple(store_handle, state_expanded, bounds, ls_args, res_fcn, jac, sol_type, observable='range'):
    """ Newton iteration to adjust measurements weights such that reduced chi-squared = 1"""
    tol = 1e-3
    n_iter = 0
    chi_sq = np.inf
    baselines = ls_args[1]
    antenna_handles = ls_args[3]
    baseline_handles = ls_args[9]
    if observable == 'range':
        phase_delay=False
        phase_only=False
    if observable == 'phase':
        use_phase_weights = ls_args[14]
        phase_delay=True
        phase_only=True

    while np.abs(chi_sq-1) > tol:
        ls_sol = least_squares(res_fcn, state_expanded, jac=jac, method='trf',\
            max_nfev=100, bounds=bounds, verbose=0, x_scale = 'jac', xtol=1e-15,\
            args=ls_args)
        residuals = get_residuals(ls_sol.fun, baseline_handles, phase_delay, phase_only)
        weights_full = np.zeros((len(residuals),len(residuals)))
        chi_sq = np.sum(residuals**2)/(len(residuals)-len(ls_sol.x))
        num_samples= 0 
        for jdx, baseline in enumerate(baselines):
            antenna1_handle = antenna_handles[baseline[0]]
            antenna2_handle = antenna_handles[baseline[1]]
            baseline_handle = baseline_handles[jdx]
            if observable == 'range':
                range_weight_mat = get_obs_weights(store_handle, antenna1_handle, antenna2_handle, 'range', \
                        baseline_handle.range_data_idxs, baseline_handle)
                weights_full[num_samples:num_samples+len(baseline_handle.range_data_idxs),\
                        num_samples:num_samples+len(baseline_handle.range_data_idxs)] = range_weight_mat
                num_samples = num_samples + len(baseline_handle.range_data_idxs) 
            elif observable == 'phase':
                phase_weight_mat = get_obs_weights(store_handle, antenna1_handle, antenna2_handle, 'phase',\
                        baseline_handle.phase_data_idxs, baseline_handle, use_phase_weights)
                weights_full[num_samples:num_samples+len(baseline_handle.phase_data_idxs), \
                        num_samples:num_samples+len(baseline_handle.phase_data_idxs)] = phase_weight_mat
                num_samples = num_samples + len(baseline_handle.phase_data_idxs)

        if observable == 'range':
            q_b = baseline_handle.q_range
        elif observable == 'phase':
            q_b = baseline_handle.q_phase
        if q_b == 0:
            if (observable == 'range' and baseline_handle.use_cov_kernel_range is True) \
                    or (observable == 'phase' and baseline_handle.use_cov_kernel_phase is True):
                    q_b = 1
            else:
                if (chi_sq-1) > 0:
                    q_b = 1/np.mean(np.diag(weights_full))
                else:
                    q_b = -1/np.mean(np.diag(weights_full))

        q_b += (chi_sq-1)/ \
                (1/(len(residuals)-len(ls_sol.x)) * np.sum((weights_full@residuals)**2 * 2*q_b))

        if q_b < 0:
            # bad iteration, just choose a smaller step
            if observable == 'range':
                q_b = baseline_handles[0].q_range/2
            elif observable == 'phase':
                q_b = baseline_handles[0].q_phase/2

        for baseline_handle in baseline_handles:
            if observable == 'range':
                baseline_handle.q_range = q_b
            elif observable == 'phase':
                baseline_handle.q_phase = q_b

        print(f'iteration {n_iter}')
        print(f'chi_sq: {chi_sq}')
        n_iter += 1
        if n_iter > 30:
            print(f'failed to adjust chi-squared final value: {chi_sq:.3f}')
            break

    return ls_sol 

def iterative_weight_adjust_baseline(store_handle, state_expanded, bounds, ls_args, res_fcn, jac, sol_type, observable='range'):
    """Implement LS-VCE to estimate variance components for GNSS observations.
       sigma^2_eta -- inter-satellite noise term (one draw from dist. for every satellite switch)
    """
    max_iter = 30
    tol_var = 1e-6
    n_iter = 0
    chi_sq = np.inf
    baselines = ls_args[1]
    antenna_handles = ls_args[3]
    baseline_handles = ls_args[9]
    if observable == 'range':
        phase_delay = False
        phase_only = False
    if observable == 'phase':
        use_phase_weights = ls_args[14]
        phase_delay = True
        phase_only = True

    # Initialization -- get initial guess for eta
    sigma_eta_squared = np.ones(len(baseline_handles)) 

    while n_iter < max_iter:
        sigma_eta_squared_new = np.ones(len(baseline_handles)) 

        # Perform least squares adjustment
        ls_sol = least_squares(res_fcn, state_expanded, jac=jac, method='trf',
                               max_nfev=100, bounds=bounds, verbose=0, x_scale='jac', xtol=1e-15,
                               args=ls_args)
        residuals = get_residuals(ls_sol.fun, baseline_handles, phase_delay, phase_only)

        num_samples = 0 
        for jdx, baseline in enumerate(baselines):
            antenna1_handle = antenna_handles[baseline[0]]
            antenna2_handle = antenna_handles[baseline[1]]
            baseline_handle = baseline_handles[jdx]
            if observable == 'range':
                weight_mat = get_obs_weights(store_handle, antenna1_handle, antenna2_handle, 'range', \
                        baseline_handle.range_data_idxs, baseline_handle)
                residuals_baseline = residuals[num_samples:num_samples+len(baseline_handle.range_data_idxs)]
                A_mat = ls_sol.jac[num_samples:num_samples+len(baseline_handle.range_data_idxs),:]
                use_idxs = baseline_handle.range_data_idxs
                num_samples = num_samples + len(baseline_handle.range_data_idxs) 
            elif observable == 'phase':
                weight_mat = get_obs_weights(store_handle, antenna1_handle, antenna2_handle, 'phase',\
                        baseline_handle.phase_data_idxs, baseline_handle, use_phase_weights)
                residuals_baseline = residuals[num_samples:num_samples+len(baseline_handle.phase_data_idxs)]
                A_mat = ls_sol.jac[num_samples:num_samples+len(baseline_handle.phase_data_idxs),:]
                use_idxs = baseline_handle.phase_data_idxs
                num_samples = num_samples + len(baseline_handle.phase_data_idxs)

            r = (np.linalg.inv(weight_mat) @ residuals_baseline).reshape(-1, 1)
            W_full = weight_mat @ weight_mat.T
            P_A = np.eye(W_full.shape[0]) - A_mat @ np.linalg.inv(A_mat.T @ W_full @ A_mat) @ A_mat.T @ W_full
            K_full = W_full @ P_A 

            # Construct cofactor matrix Q_eta
            Q_eta = construct_Q_eta(store_handle, baseline_handle, use_idxs)

            # Compute N_eta and s_eta
            N_eta = np.trace(K_full @ Q_eta @ K_full @ Q_eta)
            s_eta = r.T @ K_full @ Q_eta @ K_full @ r

            # Solve for variance component
            sigma_squared_baseline = s_eta / N_eta
            sigma_eta_squared_new[jdx] = max(sigma_squared_baseline[0, 0], 1e-10)

            if observable == 'range':
                baseline_handle.q_range = np.sqrt(sigma_squared_baseline[0, 0])
            elif observable == 'phase':
                baseline_handle.q_phase = np.sqrt(sigma_squared_baseline[0, 0])

        # Check convergence
        rel_change_eta = np.abs(sigma_eta_squared_new - sigma_eta_squared) / sigma_eta_squared
        var_change = max(rel_change_eta)
        if var_change < tol_var:
            break

        # Update variance components
        sigma_eta_squared = sigma_eta_squared_new
        print(f'iteration {n_iter}')
        print(f'max sigma^2_eta change: {var_change}')

        n_iter += 1

    # Final least squares adjustment with estimated variance components
    ls_sol = least_squares(res_fcn, state_expanded, jac=jac, method='trf',
                           max_nfev=100, bounds=bounds, verbose=0, x_scale='jac', xtol=1e-15,
                           args=ls_args)

    return ls_sol

def iterative_weight_adjust_gps_paper(store_handle, state_expanded, bounds, ls_args, res_fcn, jac, sol_type, observable='range'):
    """Implement LS-VCE to estimate variance components for GNSS observations.
       sigma^2_eta -- inter-satellite noise term (one draw from dist. for every satellite switch)
    """
    max_iter = 30
    tol_var = 1e-9
    n_iter = 0
    chi_sq = np.inf
    baselines = ls_args[1]
    antenna_handles = ls_args[3]
    baseline_handles = ls_args[9]
    if observable == 'range':
        phase_delay = False
        phase_only = False
    if observable == 'phase':
        use_phase_weights = ls_args[14]
        phase_delay = True
        phase_only = True

    # Initialization -- get initial guess for eta
    sigma_eta_squared = np.ones(len(baseline_handles)) 

    while n_iter < max_iter:
        sigma_eta_squared_new = np.ones(len(baseline_handles)) 

        # Perform least squares adjustment
        ls_sol = least_squares(res_fcn, state_expanded, jac=jac, method='trf',
                               max_nfev=100, bounds=bounds, verbose=0, x_scale='jac', xtol=1e-15,
                               args=ls_args)
        residuals = get_residuals(ls_sol.fun, baseline_handles, phase_delay, phase_only)
        A_mat = ls_sol.jac

        num_samples = 0 
        weights = np.zeros((len(residuals),len(residuals)))
        var_apr = np.zeros(len(residuals))
        for jdx, baseline in enumerate(baselines):
            antenna1_handle = antenna_handles[baseline[0]]
            antenna2_handle = antenna_handles[baseline[1]]
            baseline_handle = baseline_handles[jdx]
            if observable == 'range':
                range_weight_mat = get_obs_weights(store_handle, antenna_handles[baseline[0]], antenna_handles[baseline[1]], 'range',\
                        baseline_handle.range_data_idxs, baseline_handle)
                weights[num_samples:num_samples + len(baseline_handle.range_data_idxs),\
                        num_samples:num_samples + len(baseline_handle.range_data_idxs)] = range_weight_mat
                if sol_type == 'VLBI':
                    var_apr[num_samples:num_samples + len(baseline_handle.range_data_idxs)] = baseline_handle.grdel_err[baseline_handle.range_data_idxs]**2
                num_samples += len(baseline_handle.range_data_idxs)
            elif observable == 'phase':
                phase_weight_mat = get_obs_weights(store_handle, antenna1_handle, antenna2_handle, 'phase',\
                        baseline_handle.phase_data_idxs, baseline_handle, use_phase_weights)
                weights[num_samples:num_samples + len(baseline_handle.phase_data_idxs),\
                        num_samples:num_samples + len(baseline_handle.phase_data_idxs)] = phase_weight_mat
                if sol_type == 'VLBI':
                    var_apr[num_samples:num_samples + len(baseline_handle.phase_data_idxs)] = baseline_handle.phdel_err[baseline_handle.phase_data_idxs]**2
                num_samples += len(baseline_handle.phase_data_idxs)
        
        r = (np.linalg.inv(weights) @ residuals).reshape(-1, 1)
        W_full = weights @ weights.T
        P_A = np.eye(W_full.shape[0]) - A_mat @ np.linalg.inv(A_mat.T @ W_full @ A_mat) @ A_mat.T @ W_full
        r = P_A @ r
        K_full = W_full @ P_A 
        var_apr_mat = np.diag(var_apr)
 
        num_samples = 0 
        for jdx, baseline in enumerate(baselines):
            antenna1_handle = antenna_handles[baseline[0]]
            antenna2_handle = antenna_handles[baseline[1]]
            baseline_handle = baseline_handles[jdx]
            Q_eta_full = np.zeros_like(weights)

            # Construct cofactor matrix Q_eta_full
            if observable == 'range':
                Q_eta = construct_Q_eta(store_handle, baseline_handle, baseline_handle.range_data_idxs)
                Q_eta_full[num_samples:num_samples+len(baseline_handle.range_data_idxs),\
                        num_samples:num_samples+len(baseline_handle.range_data_idxs)] = Q_eta
                num_samples = num_samples + len(baseline_handle.range_data_idxs) 
            elif observable == 'phase':
                Q_eta = construct_Q_eta(store_handle, baseline_handle, baseline_handle.phase_data_idxs)
                Q_eta_full[num_samples:num_samples+len(baseline_handle.phase_data_idxs),\
                        num_samples:num_samples+len(baseline_handle.phase_data_idxs)] = Q_eta
                num_samples = num_samples + len(baseline_handle.phase_data_idxs)

            # Compute N_eta and s_eta
            N_eta = np.trace(Q_eta_full @ K_full @ Q_eta_full @ K_full)
            if sol_type == 'VLBI':
                # incorporate prior fringe fitting error info
                s_eta = r.T @ K_full @ Q_eta_full @ K_full @ r - np.trace(Q_eta_full @ K_full @ var_apr_mat @ K_full)
            else:
                s_eta = r.T @ K_full @ Q_eta_full @ K_full @ r 

            # Solve for variance component
            sigma_squared_baseline = s_eta / N_eta
            sigma_eta_squared_new[jdx] = max(sigma_squared_baseline[0, 0], 1e-10)

            if observable == 'range':
                baseline_handle.q_range = np.sqrt(sigma_squared_baseline[0, 0])
            elif observable == 'phase':
                baseline_handle.q_phase = np.sqrt(sigma_squared_baseline[0, 0])

        # Check convergence
        rel_change_eta = np.abs(sigma_eta_squared_new - sigma_eta_squared) / sigma_eta_squared
        var_change = max(rel_change_eta)
        if var_change < tol_var:
            break

        # Update variance components
        sigma_eta_squared = sigma_eta_squared_new
        print(f'iteration {n_iter}')
        print(f'max sigma^2_eta change: {var_change}')
        chi_sq = np.sum(residuals**2)/(len(residuals)-len(ls_sol.x))
        print(f'chi-squared: {chi_sq:.3f}')

        n_iter += 1

    # Final least squares adjustment with estimated variance components
    ls_sol = least_squares(res_fcn, state_expanded, jac=jac, method='trf',
                           max_nfev=100, bounds=bounds, verbose=0, x_scale='jac', xtol=1e-15,
                           args=ls_args)

    return ls_sol


def _row_scale(mat: np.ndarray, scale: np.ndarray):
    return scale[:, None] * mat  # diag(scale) @ mat


def _col_scale(mat: np.ndarray, scale: np.ndarray):
    return mat * scale           # mat @ diag(scale)

def iterative_weight_adjust_ls_vce(
    store_handle,
    state_expanded,
    bounds,
    ls_args,
    res_fcn,
    jac,
    sol_type,
    observable="range",
    no_PSD=False,
):
    max_iter, tol_var = 30, 1e-4
    FIT_IRW = True

    ref_antenna, baselines = ls_args[0], ls_args[1]
    antenna_handles, baseline_handles = ls_args[3], ls_args[9]
    if observable == "phase":
        phase_delay, use_phase_weights = True, ls_args[14]
        phase_only = True
    else:
        phase_delay, use_phase_weights = False, None
        phase_only = False

    ref_handle = next(a for a in antenna_handles if a.antenna_name == ref_antenna)
    sig_hat_prev, n_iter = None, 0

    while n_iter < max_iter:
        # ---------------- Non‑linear LS step -----------------------------
        ls_sol = least_squares(
            res_fcn, state_expanded, jac=jac, method="trf", max_nfev=100,
            bounds=bounds, verbose=0, x_scale="jac", xtol=1e-15, args=ls_args,
        )

        residuals = get_residuals(ls_sol.fun, baseline_handles, phase_delay, phase_only)
        m = len(ls_sol.fun)
        w_diag, var_apr = np.zeros(m), np.zeros(m)
        num_samples = 0 # start in lower right block diagonal 
        # ---------------- Observation weights ---------------------------
        for jdx, (ai_idx, aj_idx) in enumerate(baselines):
            bl = baseline_handles[jdx]
            ai, aj = antenna_handles[ai_idx], antenna_handles[aj_idx]
            if observable == "range":
                idxs = bl.range_data_idxs
                W = get_obs_weights(store_handle, ai, aj, "range", idxs, bl)
                w_vec = np.diag(W) if W.ndim == 2 else np.asarray(W)
                if sol_type == "VLBI":
                    var_apr[num_samples:num_samples+len(w_vec)] = bl.grdel_err[idxs] ** 2
            else:
                idxs = bl.phase_data_idxs
                W = get_obs_weights(store_handle, ai, aj, "phase", idxs, bl, use_phase_weights)
                w_vec = np.diag(W) if W.ndim == 2 else np.asarray(W)
                if sol_type == "VLBI":
                    var_apr[num_samples:num_samples+len(w_vec)] = bl.phdel_err[idxs] ** 2
            w_diag[num_samples:num_samples+len(w_vec)] = w_vec
            num_samples += len(w_vec)

        # ---------------- Stochastic clocks -----------------------------
        if store_handle.stochastic_clock:
            for ant in antenna_handles:
                if ant.antenna_name == ref_antenna:
                    continue
                if observable == "range":
                    pv = get_process_variance_times(store_handle, ant, "clock")
                    pv_ref = get_process_variance_times(store_handle, ref_handle, "clock", False, ant.clock_times)
                    sig = np.sqrt(pv + pv_ref)
                else:
                    pv = get_process_variance_times(store_handle, ant, "clock", phase_delay)
                    pv_ref = get_process_variance_times(store_handle, ref_handle, "clock", phase_delay, ant.phase_clock_times)
                    sig = np.sqrt(pv + pv_ref)
                n = len(sig)
                w_diag[num_samples:num_samples+n] = 1.0 / sig
                num_samples += n

        # ---------------- Stochastic troposphere ------------------------
        if store_handle.stochastic_trop:
            for ant in antenna_handles:
                if ant.antenna_name == ref_antenna or not getattr(ant, "estimate_trop", False):
                    continue
                pv = get_process_variance_times(store_handle, ant, "trop")
                sig = np.sqrt(pv)
                n = len(sig)
                w_diag[num_samples:num_samples+n] = 1.0 / sig
                num_samples += n

        # ---------------- Weighted LS algebra --------------------------- 
        inv_w, q_inv = 1.0 / w_diag, w_diag ** 2
        J = ls_sol.jac
        A = _row_scale(J, inv_w)
        cov = np.linalg.inv(J.T @ J)
        y_u = (inv_w * ls_sol.fun).reshape(-1, 1)
        y_u[len(residuals):] *= 0
        P_perp = np.eye(m) - _col_scale(A @ cov @ A.T, q_inv)
        v = P_perp @ y_u

        # ---------------- Build Q vectors -------------------------------
        num_samples = 0 
        Q_vecs = []
        for jdx, baseline in enumerate(baselines):
            antenna1_handle = antenna_handles[baseline[0]]
            antenna2_handle = antenna_handles[baseline[1]]
            baseline_handle = baseline_handles[jdx]

            Q_full = np.zeros(m)
            if observable == 'range':
                Q_eta = construct_Q_eta(store_handle, baseline_handle, baseline_handle.range_data_idxs)
                Q_full[num_samples:num_samples+len(baseline_handle.range_data_idxs)] = np.diag(Q_eta)
                num_samples += len(baseline_handle.range_data_idxs) 
            elif observable == 'phase':
                Q_eta = construct_Q_eta(store_handle, baseline_handle, baseline_handle.phase_data_idxs)
                Q_full[num_samples:num_samples+len(baseline_handle.phase_data_idxs)] = np.diag(Q_eta)
                num_samples += len(baseline_handle.phase_data_idxs)
            Q_vecs.append(Q_full)

        # stochastic PSD matrices (clock / trop)
        if not no_PSD:
            if store_handle.stochastic_clock is True:
                for idx, antenna_handle in enumerate(antenna_handles):
                    if ref_antenna != antenna_handle.antenna_name:
                        Q_full_rw = np.zeros(m)
                        Q_full_irw = np.zeros(m)
                        if observable == 'range':
                            Q_clock_rw = get_process_variance_times(store_handle, antenna_handle, 'clock', phase_delay, only_mat=True, clock_mat_type='rw')
                            Q_clock_irw = get_process_variance_times(store_handle, antenna_handle, 'clock', phase_delay, only_mat=True, clock_mat_type='irw')
                            Q_full_rw[num_samples:num_samples+len(antenna_handle.clock_times)-1] = Q_clock_rw
                            Q_full_irw[num_samples:num_samples+len(antenna_handle.clock_times)-1] = Q_clock_irw
                            num_samples += len(antenna_handle.clock_times)-1
                        elif observable == 'phase':
                            Q_clock_rw = get_process_variance_times(store_handle, antenna_handle, 'clock', phase_delay, only_mat=True, clock_mat_type='rw')
                            Q_clock_irw = get_process_variance_times(store_handle, antenna_handle, 'clock', phase_delay, only_mat=True, clock_mat_type='irw')
                            Q_full_rw[num_samples:num_samples+len(antenna_handle.phase_clock_times)-1] = Q_clock_rw
                            Q_full_irw[num_samples:num_samples+len(antenna_handle.phase_clock_times)-1] = Q_clock_irw
                            num_samples += len(antenna_handle.phase_clock_times)-1

                        Q_vecs.append(Q_full_rw/FACTOR_RW)
                        if FIT_IRW is True:
                            Q_vecs.append(Q_full_irw/FACTOR_IRW)


            if store_handle.stochastic_trop is True:
                for idx, antenna_handle in enumerate(antenna_handles):
                    if ref_antenna != antenna_handle.antenna_name and antenna_handle.estimate_trop is True:
                        Q_full_rw = np.zeros(m)
                        Q_trop = get_process_variance_times(store_handle, antenna_handle, 'trop', only_mat=True)
                        Q_full_rw[num_samples:num_samples+len(antenna_handle.trop_times)-1] = Q_trop
                        Q_vecs.append(Q_full_rw/FACTOR_RW)
                        num_samples += len(antenna_handle.trop_times)-1

        # ---------------- Assemble N and L --------------------------------
        Q_v = q_inv * v.ravel()
        Q_P = _row_scale(P_perp, q_inv)
        Q_P_sq = Q_P ** 2
        diag_QP_var = np.sum(Q_P_sq * var_apr[None, :], axis=1)

        n_q = len(Q_vecs)
        N = np.zeros((n_q, n_q))
        L = np.zeros(n_q)
        for k, qk in enumerate(Q_vecs):
            L[k] = 0.5 * (np.dot(qk, Q_v ** 2) - np.dot(qk, diag_QP_var))
            for l, ql in enumerate(Q_vecs):
                N[k, l] = 0.5 * np.dot(qk, Q_P_sq @ ql)


        N = 0.5 * (N + N.T)                     # make N exactly symmetric
        s_diag = np.sqrt(np.diag(N))            # σ_i  (may contain zeros)
        s_diag[s_diag == 0.0] = 1.0             # guard against div-by-zero
        S_inv = 1.0 / s_diag                    # diag(1/σ_i)
        
        N_s = _row_scale(_col_scale(N, S_inv), S_inv)
        L_s = S_inv * L                     # element-wise multiply
        
        # 3.  Condition number in the scaled basis
        cond_N_s = np.linalg.cond(N_s)
        C_target = 1e12 
        
        if cond_N_s <= C_target:
            # no ridge necessary
            sig_hat_s = np.linalg.solve(N_s, L_s)
            lam_s     = 0.0
        else:
            print('using ridge regression to stabilize LS-VCE')
            # 4.  Smallest λ that brings κ down to 1e12 *in the scaled basis*
            eigs   = np.linalg.eigvalsh(N_s)
            s_min, s_max = eigs[0], eigs[-1]
            s_min  = max(s_min, np.spacing(s_max))           # IEEE guard
            lam_s  = (s_max - C_target * s_min) / (C_target - 1.0)
        
            sig_hat_s = np.linalg.solve(N_s + lam_s*np.eye(len(N_s)), L_s)
        
        # 5.  Back-transform to physical units:  x = S · x_s
        sig_hat = S_inv * sig_hat_s        # element-wise
        sig_hat_save = sig_hat.copy()
        
        print(f"cond(N_s)        : {cond_N_s:.2e}")
        if lam_s:
            print(f"λ (scaled)    : {lam_s:.2e}")
            print(f"cond(N_s+λI)     : {np.linalg.cond(N_s + lam_s*np.eye(len(N_s))):.2e}")

        # set non-zero
        idx = 0
        for bl in baseline_handles:
            sig_hat[idx] = max(sig_hat[idx], 1e-10)
            idx += 1
        if not no_PSD and store_handle.stochastic_clock:
            for ant in antenna_handles:
                if ant.antenna_name == ref_antenna:
                    continue
                sig_hat[idx] = max(sig_hat[idx], 1e-2)
                idx += 1
                if FIT_IRW:
                    sig_hat[idx] = max(sig_hat[idx], 1e-3)
                    idx += 1
        if not no_PSD and store_handle.stochastic_trop:
            for ant in antenna_handles:
                if ant.antenna_name == ref_antenna or not getattr(ant, "estimate_trop", False):
                    continue
                sig_hat[idx] = max(sig_hat[idx], 1e-2)
                idx += 1


        if sig_hat_prev is not None:
            sig_hat[len(baseline_handles):] = np.exp(np.log(sig_hat_prev[len(baseline_handles):]) + \
                    0.5*(np.log(sig_hat[len(baseline_handles):]) - np.log(sig_hat_prev[len(baseline_handles):]))) # damp solution to prevent ping-ponging

        # ---------------- Update handles + print -------------------------
        idx = 0
        for bl in baseline_handles:
            s2 = max(sig_hat[idx], 1e-10)
            if observable == "range":
                bl.q_range = np.sqrt(s2)
            else:
                bl.q_phase = np.sqrt(s2)
            idx += 1

        if not no_PSD and store_handle.stochastic_clock:
            for ant in antenna_handles:
                if ant.antenna_name == ref_antenna:
                    continue
                print(f'for antenna {ant.antenna_name}:')
                if True: #observable == 'range':
                    if False: #ant.antenna_name == 'FDV2':
                        ant.clock_psd_rw = min(max(sig_hat[idx], 1e-2), 4e6)
                    else:
                        #ant.clock_psd_rw = min(max(sig_hat[idx], 1e-2), 345)
                        #ant.clock_psd_rw = min(max(sig_hat[idx], 1e-2), 50)
                        ant.clock_psd_rw = min(max(sig_hat[idx], 1e-2), 50)
                else:
                    ant.clock_psd_rw = max(sig_hat[idx], 1e-3)
                print(f"clock psd rw: {ant.clock_psd_rw} ps^2/hr")
                idx += 1
                if FIT_IRW:
                    if True: # observable == 'range':
                        if ant.ppp_clock_active:
                            #ant.clock_psd_irw = min(max(sig_hat[idx], 1e-3), 78)
                            ant.clock_psd_irw = min(max(sig_hat[idx], 1e-3), 0.0001)
                        else:
                            #ant.clock_psd_irw = min(max(sig_hat[idx], 1e-3), 780)
                            ant.clock_psd_irw = min(max(sig_hat[idx], 1e-3), 1e9)
                    else:
                        ant.clock_psd_irw = max(sig_hat[idx], 1e-4)
                    print(f"clock psd irw: {ant.clock_psd_irw} ps^2/hr^3")
                    idx += 1

        if not no_PSD and store_handle.stochastic_trop:
            for ant in antenna_handles:
                if ant.antenna_name == ref_antenna or not getattr(ant, "estimate_trop", False):
                    continue
                print(f'for antenna {ant.antenna_name}:')
                if True: #observable == 'range':
                    if ant.ppp_clock_active: 
                        ant.trop_psd_rw = min(max(sig_hat[idx], 1e-3), 10)
                    else:
                        ant.trop_psd_rw = min(max(sig_hat[idx], 1e-3), 30)
                print(f"trop psd rw: {ant.trop_psd_rw} ps^2/hr")
                idx += 1

        # ---------------- Diagnostics & convergence ----------------------
        chi_sq = np.sum(ls_sol.fun ** 2) / (m - len(ls_sol.x))
        print(f"iteration {n_iter}")
        print(f"chi-squared: {chi_sq:.3f}")

        if sig_hat_prev is not None:
            rel = np.linalg.norm(sig_hat - sig_hat_prev) / np.linalg.norm(sig_hat)
            #max_delta = np.max(rel)
            print(f"relative change: {rel}")
            if rel < tol_var:
                break
        sig_hat_prev = sig_hat.copy()
        n_iter += 1

    # ---------------- Final LS with tuned weights ------------------------
        # ---------------- Final LS with tuned weights ------------------------
    return least_squares(
        res_fcn, state_expanded, jac=jac, method="trf", max_nfev=100,
        bounds=bounds, verbose=0, x_scale="jac", xtol=1e-15, args=ls_args,
    )


def iterative_weight_adjust_ls_vce_alt(store_handle, state_expanded, bounds, ls_args, res_fcn, jac, sol_type, observable='range', no_PSD=False):
    """Implement LS-VCE to estimate variance components (https://link.springer.com/content/pdf/10.1007/s00190-007-0157-x.pdf) .
    """
    max_iter = 30
    tol_var = 1e-4
    n_iter = 0
    chi_sq = np.inf
    ref_antenna = ls_args[0]
    baselines = ls_args[1]
    antenna_handles = ls_args[3]
    clock_idxs = ls_args[4]
    trop_idxs = ls_args[6]
    baseline_handles = ls_args[9]
    if observable == 'range':
        phase_delay = False
        phase_only = False
    if observable == 'phase':
        use_phase_weights = ls_args[14]
        phase_clock_idxs = ls_args[17]
        phase_delay = True
        phase_only = True

    if store_handle.global_linear_clock is True:
        idx_start = 1
    elif store_handle.global_quadratic_clock is True:
        idx_start = 2
    else:
        idx_start = 0

    sig_hat_prev = np.ones(len(baseline_handles)) 
    while n_iter < max_iter:
        sigma_eta_squared_new = np.ones(len(baseline_handles)) 

        # Perform least squares adjustment
        ls_sol = least_squares(res_fcn, state_expanded, jac=jac, method='trf',
                               max_nfev=100, bounds=bounds, verbose=0, x_scale='jac', xtol=1e-15,
                               args=ls_args)
        residuals = get_residuals(ls_sol.fun, baseline_handles, phase_delay, phase_only)

        num_samples = 0 
        weights = np.zeros((len(ls_sol.fun),len(ls_sol.fun)))
        var_apr = np.zeros(len(ls_sol.fun))
        for jdx, baseline in enumerate(baselines):
            antenna1_handle = antenna_handles[baseline[0]]
            antenna2_handle = antenna_handles[baseline[1]]
            baseline_handle = baseline_handles[jdx]
            if observable == 'range':
                range_weight_mat = get_obs_weights(store_handle, antenna_handles[baseline[0]], antenna_handles[baseline[1]], 'range',\
                        baseline_handle.range_data_idxs, baseline_handle)
                weights[num_samples:num_samples + len(baseline_handle.range_data_idxs),\
                        num_samples:num_samples + len(baseline_handle.range_data_idxs)] = range_weight_mat
                if sol_type == 'VLBI':
                    var_apr[num_samples:num_samples + len(baseline_handle.range_data_idxs)] = baseline_handle.grdel_err[baseline_handle.range_data_idxs]**2
                num_samples += len(baseline_handle.range_data_idxs)
            elif observable == 'phase':
                phase_weight_mat = get_obs_weights(store_handle, antenna1_handle, antenna2_handle, 'phase',\
                        baseline_handle.phase_data_idxs, baseline_handle, use_phase_weights)
                weights[num_samples:num_samples + len(baseline_handle.phase_data_idxs),\
                        num_samples:num_samples + len(baseline_handle.phase_data_idxs)] = phase_weight_mat
                if sol_type == 'VLBI':
                    var_apr[num_samples:num_samples + len(baseline_handle.phase_data_idxs)] = baseline_handle.phdel_err[baseline_handle.phase_data_idxs]**2
                num_samples += len(baseline_handle.phase_data_idxs)

        if store_handle.stochastic_clock is True:
            for idx, antenna_handle in enumerate(antenna_handles):
                if ref_antenna == antenna_handle.antenna_name:
                    ref_handle = antenna_handle

            for idx, antenna_handle in enumerate(antenna_handles):
                if ref_antenna != antenna_handle.antenna_name:
                    if observable == 'range':
                        process_variance_clock = get_process_variance_times(store_handle, antenna_handle, 'clock')
                        process_variance_ref_clock = get_process_variance_times(store_handle, ref_handle, 'clock', False, antenna_handle.clock_times)
                        weights[num_samples:num_samples + len(antenna_handle.clock_times)-1,\
                                num_samples:num_samples + len(antenna_handle.clock_times)-1] =\
                                np.diag(1/np.sqrt(process_variance_clock + process_variance_ref_clock))
                        num_samples += len(antenna_handle.clock_times)-1

                    if observable == 'phase':
                        process_variance_phase_clock = get_process_variance_times(store_handle, antenna_handle, 'clock', phase_delay)
                        process_variance_ref_phase_clock = get_process_variance_times(store_handle, ref_handle, 'clock', phase_delay, antenna_handle.phase_clock_times)
                        weights[num_samples:num_samples + len(antenna_handle.phase_clock_times)-1,\
                                num_samples:num_samples + len(antenna_handle.phase_clock_times)-1] =\
                                np.diag(1/np.sqrt(process_variance_phase_clock + process_variance_ref_phase_clock))
                        num_samples += len(antenna_handle.phase_clock_times)-1

        if store_handle.stochastic_trop is True:
            for idx, antenna_handle in enumerate(antenna_handles):
                if ref_antenna != antenna_handle.antenna_name and antenna_handle.estimate_trop is True:
                    process_variance_trop = get_process_variance_times(store_handle, antenna_handle, 'trop')
                    weights[num_samples:num_samples + len(antenna_handle.trop_times)-1,\
                            num_samples:num_samples + len(antenna_handle.trop_times)-1] =\
                            np.diag(1/np.sqrt(process_variance_trop))
                    num_samples += len(antenna_handle.trop_times)-1

        var_apr_mat = np.diag(var_apr)
        A_mat = np.linalg.inv(weights)@ls_sol.jac
        n_param = ls_sol.jac.shape[1]
        cov = np.linalg.inv(ls_sol.jac.T @ ls_sol.jac)
        res_unweighted = (np.linalg.inv(weights) @ ls_sol.fun).reshape(-1, 1)
        y_com = res_unweighted.copy()
        #y_com[len(residuals):] = 0 # stochastic model "residuals" are zero for this least squares problem
        Q_com_inv = weights.T@weights
        P_A_perp = np.eye(len(y_com))-A_mat@cov@A_mat.T@Q_com_inv
        v_vec = P_A_perp@y_com
        
        # flags for controlling options in this code
        ESTIMATE_PSD = True 
        if no_PSD is True:
            # override to prevent over-permissive clock model in initial range solution
            ESTIMATE_PSD = False
        FIT_IRW = True

        num_samples = 0 # start in lower right block diagonal 
        Q_arr = []
        for jdx, baseline in enumerate(baselines):
            antenna1_handle = antenna_handles[baseline[0]]
            antenna2_handle = antenna_handles[baseline[1]]
            baseline_handle = baseline_handles[jdx]
            Q_eta_full = np.zeros_like(weights)

            # Construct cofactor matrix Q_eta_full
            if observable == 'range':
                Q_eta = construct_Q_eta(store_handle, baseline_handle, baseline_handle.range_data_idxs)
                Q_eta_full[num_samples:num_samples+len(baseline_handle.range_data_idxs),\
                        num_samples:num_samples+len(baseline_handle.range_data_idxs)] = Q_eta
                num_samples = num_samples + len(baseline_handle.range_data_idxs) 
            elif observable == 'phase':
                Q_eta = construct_Q_eta(store_handle, baseline_handle, baseline_handle.phase_data_idxs)
                Q_eta_full[num_samples:num_samples+len(baseline_handle.phase_data_idxs),\
                        num_samples:num_samples+len(baseline_handle.phase_data_idxs)] = Q_eta
                num_samples = num_samples + len(baseline_handle.phase_data_idxs)
            Q_arr.append(Q_eta_full)

        if ESTIMATE_PSD is True:
            if store_handle.stochastic_clock is True:
                for idx, antenna_handle in enumerate(antenna_handles):
                    if ref_antenna != antenna_handle.antenna_name:
                        Q_clock_full_rw = np.zeros_like(weights) 
                        Q_clock_full_irw = np.zeros_like(weights) 
                        if observable == 'range':
                            Q_clock_rw = get_process_variance_times(store_handle, antenna_handle, 'clock', phase_delay, only_mat=True, clock_mat_type='rw')
                            Q_clock_irw = get_process_variance_times(store_handle, antenna_handle, 'clock', phase_delay, only_mat=True, clock_mat_type='irw')
                            Q_clock_full_rw[num_samples:num_samples + len(antenna_handle.clock_times)-1,\
                                    num_samples:num_samples + len(antenna_handle.clock_times)-1] = np.diag(Q_clock_rw/FACTOR_RW)
                            Q_clock_full_irw[num_samples:num_samples + len(antenna_handle.clock_times)-1,\
                                    num_samples:num_samples + len(antenna_handle.clock_times)-1] = np.diag(Q_clock_irw/FACTOR_IRW)
                            num_samples += len(antenna_handle.clock_times)-1
                        elif observable == 'phase':
                            Q_clock_rw = get_process_variance_times(store_handle, antenna_handle, 'clock', phase_delay, only_mat=True, clock_mat_type='rw')
                            Q_clock_irw = get_process_variance_times(store_handle, antenna_handle, 'clock', phase_delay, only_mat=True, clock_mat_type='irw')
                            Q_clock_full_rw[num_samples:num_samples + len(antenna_handle.phase_clock_times)-1,\
                                    num_samples:num_samples + len(antenna_handle.phase_clock_times)-1] = np.diag(Q_clock_rw/FACTOR_RW)
                            Q_clock_full_irw[num_samples:num_samples + len(antenna_handle.phase_clock_times)-1,\
                                    num_samples:num_samples + len(antenna_handle.phase_clock_times)-1] = np.diag(Q_clock_irw/FACTOR_IRW)
                            num_samples += len(antenna_handle.phase_clock_times)-1

                        Q_arr.append(Q_clock_full_rw)
                        if FIT_IRW is True:
                            Q_arr.append(Q_clock_full_irw)


            if store_handle.stochastic_trop is True:
                for idx, antenna_handle in enumerate(antenna_handles):
                    if ref_antenna != antenna_handle.antenna_name and antenna_handle.estimate_trop is True:
                        Q_trop_full = np.zeros_like(weights)
                        Q_trop = get_process_variance_times(store_handle, antenna_handle, 'trop', only_mat=True)
                        Q_trop_full[num_samples:num_samples + len(antenna_handle.trop_times)-1,\
                                num_samples:num_samples + len(antenna_handle.trop_times)-1] = np.diag(Q_trop/FACTOR_RW)
                        Q_arr.append(Q_trop_full)
                        num_samples += len(antenna_handle.trop_times)-1
        
        N_arr = np.zeros((len(Q_arr),len(Q_arr)))
        L_vec = np.zeros(len(Q_arr))
        Q_v = Q_com_inv@v_vec
        Q_P = Q_com_inv@P_A_perp
        Q_P_var = Q_P@var_apr_mat@Q_P

        # currently using diagonal weights, simplify for << run time 
        diag_Q_P      = np.diag(Q_P)          # 1-D   (m,)
        diag_Q_P_var  = np.diag(Q_P_var)      # 1-D   (m,)
        Q_v_sq        = (Q_v.ravel())**2      # 1-D   (m,)
        diag_Q_P_sq   = diag_Q_P**2           # 1-D   (m,)
        for kdx, Q_k in enumerate(Q_arr):
            #L_vec[kdx] = 0.5*Q_v.T@Q_k@Q_v - 0.5*np.trace(Q_k@Q_P_var)
            qk = Q_k.diagonal() 
            L_vec[kdx] = 0.5 * (np.dot(qk, Q_v_sq) - np.dot(qk, diag_Q_P_var))
            for ldx, Q_l in enumerate(Q_arr):
                #N_arr[kdx,ldx] = 0.5*np.trace(Q_k@Q_P@Q_l@Q_P)
                ql = Q_l.diagonal()
                N_arr[kdx, ldx] = 0.5 * np.dot(qk, diag_Q_P_sq * ql)

        sig_hat = np.linalg.solve(N_arr, L_vec)

        sig_count = 0
        for jdx, baseline in enumerate(baselines):
            baseline_handle = baseline_handles[jdx]
            sigma_squared_baseline = max(sig_hat[jdx], 1e-10)

            if observable == 'range':
                baseline_handle.q_range = np.sqrt(sigma_squared_baseline)
            elif observable == 'phase':
                baseline_handle.q_phase = np.sqrt(sigma_squared_baseline)
            sig_count += 1
 
        if ESTIMATE_PSD is True:
            if store_handle.stochastic_clock is True:
                for idx, antenna_handle in enumerate(antenna_handles):
                    if ref_antenna != antenna_handle.antenna_name:
                        antenna_handle.clock_psd_rw = max(sig_hat[sig_count], 1e-3)
                        sig_count += 1
                        if FIT_IRW is True:
                            antenna_handle.clock_psd_irw = max(sig_hat[sig_count], 1e-1)
                            sig_count += 1

                        print('clock psd rw: ' + str(antenna_handle.clock_psd_rw) + ' cm^2/day')
                        if FIT_IRW is True:
                            print('clock psd irw: ' + str(antenna_handle.clock_psd_irw) + ' cm^2/day^3')

            if store_handle.stochastic_trop is True:
                for idx, antenna_handle in enumerate(antenna_handles):
                    if ref_antenna != antenna_handle.antenna_name and antenna_handle.estimate_trop is True:
                        antenna_handle.trop_psd_rw = max(sig_hat[sig_count], 1e-3)
                        sig_count += 1
                        print('trop psd rw: ' + str(antenna_handle.trop_psd_rw) + ' cm^2/day')

        # Check convergence
        rel_change_arr = sig_hat
        if n_iter > 0:
            rel_change_eta = np.abs(rel_change_arr - rel_change_prev) / np.abs(rel_change_arr)
            rel_change_prev = rel_change_arr
            var_change = max(rel_change_eta)
            if var_change < tol_var:
                break
            print(f'max sigma^2_eta change: {var_change}')
        else:
            rel_change_prev = rel_change_arr

        # Update variance components
        print(f'iteration {n_iter}')
        chi_sq = np.sum(ls_sol.fun**2)/(len(ls_sol.fun)-len(ls_sol.x))
        print(f'chi-squared: {chi_sq:.3f}')
        n_iter += 1

    # Final least squares adjustment with estimated variance components
    ls_sol = least_squares(res_fcn, state_expanded, jac=jac, method='trf',
                           max_nfev=100, bounds=bounds, verbose=0, x_scale='jac', xtol=1e-15,
                           args=ls_args)

    return ls_sol


def iterative_weight_adjust_ls_vce_old(store_handle, state_expanded, bounds, ls_args, res_fcn, jac, sol_type, observable='range', no_PSD=False):
    """Implement LS-VCE to estimate variance components (https://link.springer.com/content/pdf/10.1007/s00190-007-0157-x.pdf) .
    """
    max_iter = 30
    tol_var = 1e-4
    n_iter = 0
    chi_sq = np.inf
    ref_antenna = ls_args[0]
    baselines = ls_args[1]
    antenna_handles = ls_args[3]
    clock_idxs = ls_args[4]
    trop_idxs = ls_args[6]
    baseline_handles = ls_args[9]
    if observable == 'range':
        phase_delay = False
        phase_only = False
    if observable == 'phase':
        use_phase_weights = ls_args[14]
        phase_clock_idxs = ls_args[17]
        phase_delay = True
        phase_only = True

    if store_handle.global_linear_clock is True:
        idx_start = 1
    elif store_handle.global_quadratic_clock is True:
        idx_start = 2
    else:
        idx_start = 0

    sig_hat_prev = np.ones(len(baseline_handles)) 
    while n_iter < max_iter:
        sigma_eta_squared_new = np.ones(len(baseline_handles)) 

        # Perform least squares adjustment
        ls_sol = least_squares(res_fcn, state_expanded, jac=jac, method='trf',
                               max_nfev=100, bounds=bounds, verbose=0, x_scale='jac', xtol=1e-15,
                               args=ls_args)
        residuals = get_residuals(ls_sol.fun, baseline_handles, phase_delay, phase_only)

        num_samples = 0 
        weights = np.zeros((len(ls_sol.fun),len(ls_sol.fun)))
        var_apr = np.zeros(len(ls_sol.fun))
        for jdx, baseline in enumerate(baselines):
            antenna1_handle = antenna_handles[baseline[0]]
            antenna2_handle = antenna_handles[baseline[1]]
            baseline_handle = baseline_handles[jdx]
            if observable == 'range':
                range_weight_mat = get_obs_weights(store_handle, antenna_handles[baseline[0]], antenna_handles[baseline[1]], 'range',\
                        baseline_handle.range_data_idxs, baseline_handle)
                weights[num_samples:num_samples + len(baseline_handle.range_data_idxs),\
                        num_samples:num_samples + len(baseline_handle.range_data_idxs)] = range_weight_mat
                if sol_type == 'VLBI':
                    var_apr[num_samples:num_samples + len(baseline_handle.range_data_idxs)] = baseline_handle.grdel_err[baseline_handle.range_data_idxs]**2
                num_samples += len(baseline_handle.range_data_idxs)
            elif observable == 'phase':
                phase_weight_mat = get_obs_weights(store_handle, antenna1_handle, antenna2_handle, 'phase',\
                        baseline_handle.phase_data_idxs, baseline_handle, use_phase_weights)
                weights[num_samples:num_samples + len(baseline_handle.phase_data_idxs),\
                        num_samples:num_samples + len(baseline_handle.phase_data_idxs)] = phase_weight_mat
                if sol_type == 'VLBI':
                    var_apr[num_samples:num_samples + len(baseline_handle.phase_data_idxs)] = baseline_handle.phdel_err[baseline_handle.phase_data_idxs]**2
                num_samples += len(baseline_handle.phase_data_idxs)

        if store_handle.stochastic_clock is True:
            for idx, antenna_handle in enumerate(antenna_handles):
                if ref_antenna == antenna_handle.antenna_name:
                    ref_handle = antenna_handle

            for idx, antenna_handle in enumerate(antenna_handles):
                if ref_antenna != antenna_handle.antenna_name:
                    if observable == 'range':
                        process_variance_clock = get_process_variance_times(store_handle, antenna_handle, 'clock')
                        process_variance_ref_clock = get_process_variance_times(store_handle, ref_handle, 'clock', False, antenna_handle.clock_times)
                        weights[num_samples:num_samples + len(antenna_handle.clock_times)-1,\
                                num_samples:num_samples + len(antenna_handle.clock_times)-1] =\
                                np.diag(1/np.sqrt(process_variance_clock + process_variance_ref_clock))
                        num_samples += len(antenna_handle.clock_times)-1

                    if observable == 'phase':
                        process_variance_phase_clock = get_process_variance_times(store_handle, antenna_handle, 'clock', phase_delay)
                        process_variance_ref_phase_clock = get_process_variance_times(store_handle, ref_handle, 'clock', phase_delay, antenna_handle.phase_clock_times)
                        weights[num_samples:num_samples + len(antenna_handle.phase_clock_times)-1,\
                                num_samples:num_samples + len(antenna_handle.phase_clock_times)-1] =\
                                np.diag(1/np.sqrt(process_variance_phase_clock + process_variance_ref_phase_clock))
                        num_samples += len(antenna_handle.phase_clock_times)-1

        if store_handle.stochastic_trop is True:
            for idx, antenna_handle in enumerate(antenna_handles):
                if ref_antenna != antenna_handle.antenna_name and antenna_handle.estimate_trop is True:
                    process_variance_trop = get_process_variance_times(store_handle, antenna_handle, 'trop')
                    weights[num_samples:num_samples + len(antenna_handle.trop_times)-1,\
                            num_samples:num_samples + len(antenna_handle.trop_times)-1] =\
                            np.diag(1/np.sqrt(process_variance_trop))
                    num_samples += len(antenna_handle.trop_times)-1

        var_apr_mat = np.diag(var_apr)
        #A_mat = ls_sol.jac
        A_mat = np.linalg.inv(weights)@ls_sol.jac
        n_param = ls_sol.jac.shape[1]
        cov = np.linalg.inv(ls_sol.jac.T @ ls_sol.jac)
        res_unweighted = (np.linalg.inv(weights) @ ls_sol.fun).reshape(-1, 1)

        # get null space matrix B
        U, s, Vt = np.linalg.svd(A_mat, full_matrices=True)
        # Determine the effective rank by comparing singular values to a tolerance.
        rtol = 1e-8
        tol = rtol * s[0] if s.size > 0 else rtol
        rank = (s > tol).sum()
        # The columns of U from 'rank' to m-1 form an orthonormal basis for the left null space of H.
        B_mat = U[:, rank:]
        #t_vec = B_mat.T @ res_unweighted # ls_sol.fun
        t_vec = B_mat.T @ y_com # ls_sol.fun
        
        # flags for controlling options in this code
        ESTIMATE_PSD = True 
        if no_PSD is True:
            # override to prevent over-permissive clock model in initial range solution
            ESTIMATE_PSD = False
        ESTIMATE_CHI_SQ = True
        EM_UPDATE = False
        FIT_IRW = False
        initiated = False

        num_samples = 0 # start in lower right block diagonal 
        for jdx, baseline in enumerate(baselines):
            antenna1_handle = antenna_handles[baseline[0]]
            antenna2_handle = antenna_handles[baseline[1]]
            baseline_handle = baseline_handles[jdx]
            Q_eta_full = np.zeros_like(weights)

            # Construct cofactor matrix Q_eta_full
            if observable == 'range':
                Q_eta = construct_Q_eta(store_handle, baseline_handle, baseline_handle.range_data_idxs)
                Q_eta_full[num_samples:num_samples+len(baseline_handle.range_data_idxs),\
                        num_samples:num_samples+len(baseline_handle.range_data_idxs)] = Q_eta
                num_samples = num_samples + len(baseline_handle.range_data_idxs) 
            elif observable == 'phase':
                Q_eta = construct_Q_eta(store_handle, baseline_handle, baseline_handle.phase_data_idxs)
                Q_eta_full[num_samples:num_samples+len(baseline_handle.phase_data_idxs),\
                        num_samples:num_samples+len(baseline_handle.phase_data_idxs)] = Q_eta
                num_samples = num_samples + len(baseline_handle.phase_data_idxs)

            if ESTIMATE_CHI_SQ is True:
                if jdx == 0:
                    initiated=True
                    H_vh = vh_operator(B_mat.T @ Q_eta_full @ B_mat)[:,np.newaxis]
                else:
                    H_vh = np.hstack((H_vh, vh_operator(B_mat.T @ Q_eta_full @ B_mat)[:,np.newaxis]))

        if ESTIMATE_PSD is True:
            if EM_UPDATE is True:
                phi_arr = []
            if store_handle.stochastic_clock is True:
                for idx, antenna_handle in enumerate(antenna_handles):
                    if ref_antenna != antenna_handle.antenna_name:
                        Q_clock_full_rw = np.zeros_like(weights) 
                        Q_clock_full_irw = np.zeros_like(weights) 
                        if observable == 'range':
                            Q_clock_rw = get_process_variance_times(store_handle, antenna_handle, 'clock', phase_delay, only_mat=True, clock_mat_type='rw')
                            Q_clock_irw = get_process_variance_times(store_handle, antenna_handle, 'clock', phase_delay, only_mat=True, clock_mat_type='irw')
                            Q_clock_full_rw[num_samples:num_samples + len(antenna_handle.clock_times)-1,\
                                    num_samples:num_samples + len(antenna_handle.clock_times)-1] = np.diag(Q_clock_rw/FACTOR_RW)
                            Q_clock_full_irw[num_samples:num_samples + len(antenna_handle.clock_times)-1,\
                                    num_samples:num_samples + len(antenna_handle.clock_times)-1] = np.diag(Q_clock_irw/FACTOR_IRW)
                        elif observable == 'phase':
                            Q_clock_rw = get_process_variance_times(store_handle, antenna_handle, 'clock', phase_delay, only_mat=True, clock_mat_type='rw')
                            Q_clock_irw = get_process_variance_times(store_handle, antenna_handle, 'clock', phase_delay, only_mat=True, clock_mat_type='irw')
                            Q_clock_full_rw[num_samples:num_samples + len(antenna_handle.phase_clock_times)-1,\
                                    num_samples:num_samples + len(antenna_handle.phase_clock_times)-1] = np.diag(Q_clock_rw/FACTOR_RW)
                            Q_clock_full_irw[num_samples:num_samples + len(antenna_handle.phase_clock_times)-1,\
                                    num_samples:num_samples + len(antenna_handle.phase_clock_times)-1] = np.diag(Q_clock_irw/FACTOR_IRW)

                        if EM_UPDATE is True:
                            if observable == 'range':
                                n_epochs = len(antenna_handle.clock_times)-1
                                clock_idxs_range = np.arange(clock_idxs.start+idx_start, clock_idxs.stop)
                                idxs_ant = clock_idxs_range[antenna_handle.range_clock_idxs]
                            elif observable == 'phase':
                                n_epochs = len(antenna_handle.phase_clock_times)-1
                                clock_idxs_phase = np.arange(phase_clock_idxs.start+idx_start, phase_clock_idxs.stop)
                                idxs_ant = clock_idxs_phase[antenna_handle.phase_clock_idxs]

                            S_w_arr = []
                            G_sum_rw_arr = []
                            if FIT_IRW is True:
                                G_sum_irw_arr = []
                            for k in range(n_epochs):
                                diff_mu = res_unweighted[num_samples+k]
                                diff_var = cov[idxs_ant[k+1],idxs_ant[k+1]] + cov[idxs_ant[k],idxs_ant[k]] - 2*cov[idxs_ant[k],idxs_ant[k+1]]
                                S_w = diff_mu**2 + diff_var
                                S_w_arr.append(S_w)
                                G_sum_rw_arr.append(Q_clock_rw[k])
                                if FIT_IRW is True:
                                    G_sum_irw_arr.append(Q_clock_irw[k])

                            if FIT_IRW is True:
                                G = np.column_stack((np.array(G_sum_rw_arr)/FACTOR_RW, np.array(G_sum_irw_arr)))
                                S = np.array(S_w_arr)
                                phi = np.linalg.solve(G.T@G, G.T@S)
                                phi_arr.append(phi[0])
                                phi_arr.append(phi[1])
                            else:
                                phi_arr.append(np.sum(np.array(S_w_arr))*FACTOR_RW/np.sum(np.array(G_sum_rw_arr)))
                        else:
                            if initiated is False:
                                H_vh = vh_operator(B_mat.T @ Q_clock_full_rw @ B_mat)[:,np.newaxis]
                                initiated = True
                            else:
                                H_vh = np.hstack((H_vh, vh_operator(B_mat.T @ Q_clock_full_rw @ B_mat)[:,np.newaxis]))
                            if FIT_IRW is True:
                                H_vh = np.hstack((H_vh, vh_operator(B_mat.T @ Q_clock_full_irw @ B_mat)[:,np.newaxis]))

                        if observable == 'range':
                            num_samples += len(antenna_handle.clock_times)-1
                        elif observable == 'phase':
                            num_samples += len(antenna_handle.phase_clock_times)-1

            if store_handle.stochastic_trop is True:
                for idx, antenna_handle in enumerate(antenna_handles):
                    if ref_antenna != antenna_handle.antenna_name and antenna_handle.estimate_trop is True:
                        Q_trop_full = np.zeros_like(weights)
                        Q_trop = get_process_variance_times(store_handle, antenna_handle, 'trop', only_mat=True)
                        Q_trop_full[num_samples:num_samples + len(antenna_handle.trop_times)-1,\
                                num_samples:num_samples + len(antenna_handle.trop_times)-1] = np.diag(Q_trop/FACTOR_RW)

                        if EM_UPDATE is True:
                            n_epochs = len(antenna_handle.trop_times)-1
                            trop_idxs = np.arange(trop_idxs.start, trop_idxs.stop)
                            idxs_ant = trop_idxs[antenna_handle.trop_idxs]
                            S_w = 0
                            G_sum_rw = 0
                            for k in range(n_epochs):
                                diff_mu = res_unweighted[num_samples+k]
                                diff_var = cov[idxs_ant[k+1],idxs_ant[k+1]] + cov[idxs_ant[k],idxs_ant[k]] - 2*cov[idxs_ant[k],idxs_ant[k+1]]
                                S_w += diff_mu**2 + diff_var
                                G_sum_rw += Q_clock_rw[k]
                            phi_arr.append(S_w/G_sum_rw*FACTOR_RW)
                        else:
                            H_vh = np.hstack((H_vh, vh_operator(B_mat.T @ Q_trop_full @ B_mat)[:,np.newaxis]))
                        
                        num_samples += len(antenna_handle.trop_times)-1
        
        y_vh = vh_operator(t_vec@t_vec.T - B_mat.T@var_apr_mat@B_mat)
        sig_hat = np.linalg.lstsq(H_vh, y_vh, rcond=None)[0]
        if ESTIMATE_PSD is True:
            Q_com_inv = weights.T@weights
            y_com = np.concatenate((res_unweighted[:len(residuals)].flatten(), np.zeros(len(antenna_handles[1].clock_times)-1).flatten()))
            P_A_perp = np.eye(len(y_com))-A_mat@cov@A_mat.T@Q_com_inv
            test_perp = P_A_perp@A_mat # should be 0s, compare against B
            n_11 = 0.5*np.trace(Q_eta_full@Q_com_inv@P_A_perp@Q_eta_full@Q_com_inv@P_A_perp)
            n_12 = 0.5*np.trace(Q_eta_full@Q_com_inv@P_A_perp@Q_clock_full_rw@Q_com_inv@P_A_perp)
            n_21 = 0.5*np.trace(Q_clock_full_rw@Q_com_inv@P_A_perp@Q_eta_full@Q_com_inv@P_A_perp)
            n_22 = 0.5*np.trace(Q_clock_full_rw@Q_com_inv@P_A_perp@Q_clock_full_rw@Q_com_inv@P_A_perp)
            v_vec = P_A_perp@y_com
            l_1 = 0.5*v_vec.T@Q_com_inv@Q_eta_full@Q_com_inv@v_vec - 0.5*np.trace(Q_eta_full@Q_com_inv@P_A_perp@var_apr_mat@Q_com_inv@P_A_perp)
            l_2 = 0.5*v_vec.T@Q_com_inv@Q_clock_full_rw@Q_com_inv@v_vec - 0.5*np.trace(Q_clock_full_rw@Q_com_inv@P_A_perp@var_apr_mat@Q_com_inv@P_A_perp)
            N_arr = np.array([[n_11, n_12],[n_21,n_22]])
            L_vec = np.array([l_1, l_2])
            sig_hat_simple = np.linalg.solve(N_arr, L_vec)
        #sig_hat  = nnls(H_vh, y_vh)[0] 

        sig_count = 0
        if ESTIMATE_CHI_SQ is True:
            for jdx, baseline in enumerate(baselines):
                baseline_handle = baseline_handles[jdx]
                sigma_squared_baseline = max(sig_hat[jdx], 1e-10)

                if observable == 'range':
                    baseline_handle.q_range = np.sqrt(sigma_squared_baseline)
                elif observable == 'phase':
                    baseline_handle.q_phase = np.sqrt(sigma_squared_baseline)
                sig_count += 1
 
        if ESTIMATE_PSD is True:
            em_count = 0
            if store_handle.stochastic_clock is True:
                for idx, antenna_handle in enumerate(antenna_handles):
                    if ref_antenna != antenna_handle.antenna_name:
                        if EM_UPDATE is True:
                            antenna_handle.clock_psd_rw = phi_arr[em_count]
                            em_count += 1
                            if FIT_IRW is True:
                                antenna_handle.clock_psd_irw = phi_arr[em_count]
                                em_count += 1
                        else:
                            antenna_handle.clock_psd_rw = max(sig_hat[sig_count], 1e-3)
                            sig_count += 1
                            if FIT_IRW is True:
                                antenna_handle.clock_psd_irw = max(sig_hat[sig_count], 1e-1)
                                sig_count += 1

                        print('clock psd rw: ' + str(antenna_handle.clock_psd_rw) + ' cm^2/day')
                        if FIT_IRW is True:
                            print('clock psd irw: ' + str(antenna_handle.clock_psd_irw) + ' cm^2/day^3')

            if store_handle.stochastic_trop is True:
                for idx, antenna_handle in enumerate(antenna_handles):
                    if ref_antenna != antenna_handle.antenna_name and antenna_handle.estimate_trop is True:
                        if EM_UPDATE is True:
                            antenna_handle.trop_psd_rw = phi_arr[em_count]
                            em_count += 1
                        else:
                            antenna_handle.trop_psd_rw = max(sig_hat[sig_count], 1e-3)
                            sig_count += 1
                        print('trop psd rw: ' + str(antenna_handle.trop_psd_rw) + ' cm^2/day')

        # Check convergence
        if EM_UPDATE is True and ESTIMATE_PSD is True:
            rel_change_arr = np.concatenate((sig_hat.flatten(),np.array(phi_arr).flatten()))
        else:
            rel_change_arr = sig_hat
        if n_iter > 0:
            rel_change_eta = np.abs(rel_change_arr - rel_change_prev) / np.abs(rel_change_arr)
            rel_change_prev = rel_change_arr
            var_change = max(rel_change_eta)
            if var_change < tol_var:
                break
            print(f'max sigma^2_eta change: {var_change}')
        else:
            rel_change_prev = rel_change_arr

        # Update variance components
        print(f'iteration {n_iter}')
        chi_sq = np.sum(ls_sol.fun**2)/(len(ls_sol.fun)-len(ls_sol.x))
        print(f'chi-squared: {chi_sq:.3f}')

        # after you build weights, A_mat, B_mat
        #print("orthogonality ‖Bᵀ H‖:",
        #      np.linalg.norm(B_mat.T @ A_mat))
        #print("cond(H_vhᵀ H_vh):",
        #      np.linalg.cond(H_vh.T @ H_vh))
        #print("range vh PSD col magnitude:",
        #np.linalg.norm(H_vh[:, -1]))

        n_iter += 1

    # Final least squares adjustment with estimated variance components
    ls_sol = least_squares(res_fcn, state_expanded, jac=jac, method='trf',
                           max_nfev=100, bounds=bounds, verbose=0, x_scale='jac', xtol=1e-15,
                           args=ls_args)

    return ls_sol

def vh_operator(A):
    """
    Applies the vh (half-vectorization) operator to a symmetric matrix A.
    
    Parameters:
    A (array_like): A square (symmetric) matrix.
    
    Returns:
    np.ndarray: A vector containing the lower triangular part of A (including the diagonal),
                stacked column-wise.
    """
    A = np.asarray(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError("Input matrix must be square.")
    # Use numpy's tril_indices to extract indices for the lower triangular part
    lower_indices = np.tril_indices(A.shape[0])
    return A[lower_indices]

def iterative_weight_adjust(store_handle, state_expanded, bounds, ls_args, res_fcn, jac, sol_type, observable='range'):
    """Implement chi-squared fitting based on expectation value of residuals as shown at https://astrogeo.org/progs/psolve_20241125/doc/upwei_02.pdf.
    """
    max_iter = 5
    #tol_var = 1e-3
    tol_var = 1e-2
    n_iter = 0
    chi_sq = np.inf
    ref_antenna = ls_args[0]
    baselines = ls_args[1]
    antenna_handles = ls_args[3]
    clock_idxs = ls_args[4]
    trop_idxs = ls_args[6]
    baseline_handles = ls_args[9]
    if observable == 'range':
        phase_delay = False
        phase_only = False
    if observable == 'phase':
        use_phase_weights = ls_args[14]
        phase_clock_idxs = ls_args[17]
        phase_delay = True
        phase_only = True

    while n_iter < max_iter:
        # Perform least squares adjustment
        ls_sol = least_squares(res_fcn, state_expanded, jac=jac, method='trf',
                               max_nfev=100, bounds=bounds, verbose=2, x_scale='jac', xtol=1e-15,
                               args=ls_args)
        residuals = get_residuals(ls_sol.fun, baseline_handles, phase_delay, phase_only)
        weights = np.zeros((len(residuals),len(residuals)))
        var_apr = np.zeros(len(residuals))
        num_samples = 0
        for jdx, baseline in enumerate(baselines):
            antenna1_handle = antenna_handles[baseline[0]]
            antenna2_handle = antenna_handles[baseline[1]]
            baseline_handle = baseline_handles[jdx]
            if observable == 'range':
                range_weight_mat = get_obs_weights(store_handle, antenna_handles[baseline[0]], antenna_handles[baseline[1]], 'range',\
                        baseline_handle.range_data_idxs, baseline_handle)
                weights[num_samples:num_samples + len(baseline_handle.range_data_idxs),\
                        num_samples:num_samples + len(baseline_handle.range_data_idxs)] = range_weight_mat
                num_samples += len(baseline_handle.range_data_idxs)
            elif observable == 'phase':
                phase_weight_mat = get_obs_weights(store_handle, antenna1_handle, antenna2_handle, 'phase',\
                        baseline_handle.phase_data_idxs, baseline_handle, use_phase_weights)
                weights[num_samples:num_samples + len(baseline_handle.phase_data_idxs),\
                        num_samples:num_samples + len(baseline_handle.phase_data_idxs)] = phase_weight_mat
                num_samples += len(baseline_handle.phase_data_idxs)

        #if store_handle.stochastic_clock is True or store_handle.stochastic_clock is True:
        #    # need to stabilize the covariance with constraints from state transition equations
        #    if store_handle.stochastic_clock is True:
        #        idx_clock = 0
        #        if phase_delay is True:
        #            idx_phase_clock = 0
        #        for antenna_handle in antenna_handles:
        #            if ref_antenna != antenna_handle.antenna_name:
        #                process_variance_clock = get_process_variance_times(store_handle, antenna_handle, 'clock')
        #                weights[np.arange(num_samples,num_samples+len(antenna_handle.clock_times)-1),\
        #                        np.arange(num_samples,num_samples+len(antenna_handle.clock_times)-1)] = 1/np.sqrt(process_variance_clock)
        #                num_samples += len(antenna_handle.clock_times)-1

        #                if phase_delay is True:
        #                    process_variance_phase_clock = get_process_variance_times(store_handle, antenna_handle, 'clock', phase_delay)
        #                    weights[np.arange(num_samples,num_samples+len(antenna_handle.phase_clock_times)-1),\
        #                            np.arange(num_samples,num_samples+len(antenna_handle.phase_clock_times)-1)] = 1/np.sqrt(process_variance_phase_clock)
        #                    num_samples += len(antenna_handle.phase_clock_times)-1

        #    if store_handle.stochastic_trop is True:
        #        idx_trop = 0
        #        for antenna_handle in antenna_handles:
        #            if ref_antenna != antenna_handle.antenna_name:
        #                process_variance_trop = get_process_variance_times(store_handle, antenna_handle, 'trop')
        #                weights[np.arange(num_samples,num_samples+len(antenna_handle.trop_times)-1),\
        #                        np.arange(num_samples,num_samples+len(antenna_handle.trop_times)-1)] = 1/np.sqrt(process_variance_trop)
        #                num_samples += len(antenna_handle.trop_times)-1
        
        H_mat = np.linalg.inv(weights) @ ls_sol.jac[:len(weights),:]
        alpha_inv = weights @ weights.T
        V_mat = np.linalg.inv(ls_sol.jac.T @ ls_sol.jac)
        H_V_H = H_mat @ V_mat @ H_mat.T
        
        num_samples = 0 
        for jdx, baseline in enumerate(baselines):
            antenna1_handle = antenna_handles[baseline[0]]
            antenna2_handle = antenna_handles[baseline[1]]
            baseline_handle = baseline_handles[jdx]
            alpha_b_inv = np.zeros_like(weights)

            # Construct cofactor matrix Q_eta_full
            if observable == 'range':
                range_weight_mat = get_obs_weights(store_handle, antenna_handles[baseline[0]], antenna_handles[baseline[1]], 'range',\
                        baseline_handle.range_data_idxs, baseline_handle)
                R_b = np.sum(residuals[num_samples:num_samples+len(baseline_handle.range_data_idxs)]**2)
                N_b = len(baseline_handle.range_data_idxs)
                H_b = np.linalg.inv(range_weight_mat) @ ls_sol.jac[num_samples:num_samples+N_b,:]
                alpha_b_inv[num_samples:num_samples+len(baseline_handle.range_data_idxs), num_samples:num_samples+len(baseline_handle.range_data_idxs)] =\
                        range_weight_mat@range_weight_mat.T
                num_samples = num_samples + N_b
            elif observable == 'phase':
                phase_weight_mat = get_obs_weights(store_handle, antenna1_handle, antenna2_handle, 'phase',\
                        baseline_handle.phase_data_idxs, baseline_handle, use_phase_weights)
                R_b = np.sum(residuals[num_samples:num_samples+len(baseline_handle.phase_data_idxs)]**2)
                N_b = len(baseline_handle.phase_data_idxs)
                H_b = np.linalg.inv(phase_weight_mat) @ ls_sol.jac[num_samples:num_samples + N_b,:]
                alpha_b_inv[num_samples:num_samples+len(baseline_handle.phase_data_idxs), num_samples:num_samples+len(baseline_handle.phase_data_idxs)] =\
                        phase_weight_mat@phase_weight_mat.T
                num_samples = num_samples + N_b

            sigma_squared_baseline = (R_b - N_b + np.trace(H_V_H@alpha_b_inv))/(np.trace(alpha_b_inv)-np.trace(alpha_b_inv@H_V_H@alpha_b_inv))
            #if sigma_squared_baseline < 0: 
            #    sigma_squared_baseline=0
            #    break
            if observable == 'range':
                baseline_handle.q_range = np.sqrt(sigma_squared_baseline + baseline_handle.q_range**2)
            elif observable == 'phase':
                baseline_handle.q_phase = np.sqrt(sigma_squared_baseline + baseline_handle.q_phase**2)

        chi_sq = np.sum(ls_sol.fun**2)/(len(ls_sol.fun)-len(ls_sol.x))

        # Check convergence
        if np.abs(chi_sq-1)<tol_var:
            break
        
        # Update variance components
        print(f'iteration {n_iter}')
        print(f'chi-squared: {chi_sq:.3f}')

        n_iter += 1

    # Final least squares adjustment with estimated variance components
    ls_sol = least_squares(res_fcn, state_expanded, jac=jac, method='trf',
                           max_nfev=100, bounds=bounds, verbose=0, x_scale='jac', xtol=1e-15,
                           args=ls_args)

    return ls_sol

def iterative_weight_adjust_LS_VCE_full(store_handle, state_expanded, bounds, ls_args, res_fcn, jac, sol_type, observable='range'):
    """Implement LS-VCE to estimate variance components for GNSS observations.
       sigma^2_eta -- inter-satellite noise term (one draw from dist. for every satellite switch)
       sigma^2_epsilon -- intra-satellite noise term (random for every observation)
    """
    max_iter = 30
    tol_var = 1e-6
    n_iter = 0
    chi_sq = np.inf
    baselines = ls_args[1]
    antenna_handles = ls_args[3]
    baseline_handles = ls_args[9]
    if observable == 'range':
        phase_delay=False
        phase_only=False
    if observable == 'phase':
        use_phase_weights = ls_args[14]
        phase_delay=True
        phase_only=True

    # Initialization -- get initial guess for eta, epsilon
    sigma_eta_squared = np.ones(len(baseline_handles)) 
    sigma_epsilon_squared = np.ones(len(baseline_handles))

    while n_iter < max_iter:
        sigma_eta_squared_new = np.ones(len(baseline_handles)) 
        sigma_epsilon_squared_new = np.ones(len(baseline_handles))

        # Perform least squares adjustment
        ls_sol = least_squares(res_fcn, state_expanded, jac=jac, method='trf',
                               max_nfev=100, bounds=bounds, verbose=0, x_scale='jac', xtol=1e-15,
                               args=ls_args)
        residuals = get_residuals(ls_sol.fun, baseline_handles, phase_delay, phase_only)

        num_samples= 0 
        for jdx, baseline in enumerate(baselines):
            antenna1_handle = antenna_handles[baseline[0]]
            antenna2_handle = antenna_handles[baseline[1]]
            baseline_handle = baseline_handles[jdx]
            if observable == 'range':
                weight_mat = get_obs_weights(store_handle, antenna1_handle, antenna2_handle, 'range', \
                        baseline_handle.range_data_idxs, baseline_handle)
                residuals_baseline = residuals[num_samples:num_samples+len(baseline_handle.range_data_idxs)]
                A_mat = ls_sol.jac[num_samples:num_samples+len(baseline_handle.range_data_idxs),:]
                use_idxs = baseline_handle.range_data_idxs
                num_samples = num_samples + len(baseline_handle.range_data_idxs) 
            elif observable == 'phase':
                weight_mat = get_obs_weights(store_handle, antenna1_handle, antenna2_handle, 'phase',\
                        baseline_handle.phase_data_idxs, baseline_handle, use_phase_weights)
                residuals_baseline = residuals[num_samples:num_samples+len(baseline_handle.phase_data_idxs)]
                A_mat = ls_sol.jac[num_samples:num_samples+len(baseline_handle.phase_data_idxs),:]
                use_idxs = baseline_handle.phase_data_idxs
                num_samples = num_samples + len(baseline_handle.phase_data_idxs)

            r = (np.linalg.inv(weight_mat)@residuals_baseline).reshape(-1, 1)
            W_full = weight_mat@weight_mat.T
            P_A = np.eye(W_full.shape[0])-A_mat@np.linalg.inv(A_mat.T@W_full@A_mat)@A_mat.T@W_full
            K_full = W_full @ P_A 

            # Construct cofactor matrices Q1 and Q2
            Q_eta = construct_Q_eta(store_handle, baseline_handle, use_idxs)
            Q_epsilon = np.eye(Q_eta.shape[0])  # Identity matrix

            if np.all(Q_eta == Q_epsilon):
                # Compute N_beta_gamma and s_beta
                N = np.trace(K_full @ Q_epsilon @ K_full @ Q_epsilon)
                s = r.T @ K_full @ Q_epsilon @ K_full @ r

                # Solve for variance components
                sigma_squared_baseline = s/N
                sigma_epsilon_squared_new[jdx] = sigma_squared_baseline[0,0]

                # Ensure variance is positive
                sigma_epsilon_squared_new[jdx] = max(sigma_epsilon_squared_new[jdx], 1e-10)

                if observable == 'range':
                    baseline_handle.q_range = np.sqrt(sigma_squared_baseline[0,0])
                elif observable == 'phase':
                    baseline_handle.q_phase = np.sqrt(sigma_squared_baseline[0,0])

                print('sigma_epsilon')
                print(np.sqrt(sigma_epsilon_squared_new))
            else:
                # Compute N_beta_gamma and s_beta
                Qs = [Q_eta, Q_epsilon]
                N = np.zeros((2, 2))
                s = np.zeros((2, 1))
                for beta in range(2):
                    for gamma in range(2):
                        N[beta, gamma] = np.trace(K_full @ Qs[gamma] @ K_full @ Qs[beta])
                    s[beta] = r.T @ K_full @ Qs[beta] @ K_full @ r

                # Solve for variance components
                sigma_squared_baseline = np.linalg.solve(N, s)
                sigma_eta_squared_new[jdx] = sigma_squared_baseline[0, 0]
                sigma_epsilon_squared_new[jdx] = sigma_squared_baseline[1, 0]

                # Ensure variances are positive
                sigma_eta_squared_new[jdx] = max(sigma_eta_squared_new[jdx], 1e-10)
                sigma_epsilon_squared_new[jdx] = max(sigma_epsilon_squared_new[jdx], 1e-10)

                if observable == 'range':
                    baseline_handle.q_range_satellite = np.sqrt(sigma_squared_baseline[0, 0])
                    baseline_handle.q_range = np.sqrt(sigma_squared_baseline[1, 0])
                elif observable == 'phase':
                    baseline_handle.q_phase_satellite = np.sqrt(sigma_squared_baseline[0, 0])
                    baseline_handle.q_phase = np.sqrt(sigma_squared_baseline[1, 0])

                print('sigma_eta')
                print(np.sqrt(sigma_eta_squared_new))
                print('sigma_epsilon')
                print(np.sqrt(sigma_epsilon_squared_new))

        # Check convergence
        rel_change_eta = np.abs(sigma_eta_squared_new - sigma_eta_squared) / sigma_eta_squared
        rel_change_epsilon = np.abs(sigma_epsilon_squared_new - sigma_epsilon_squared) / sigma_epsilon_squared
        
        var_change = max(max(rel_change_eta), max(rel_change_epsilon))
        if var_change < tol_var:
            break

        # Update variance components
        sigma_eta_squared = sigma_eta_squared_new
        sigma_epsilon_squared = sigma_epsilon_squared_new
        print(f'iteration {n_iter}')
        print(f'max sigma^2 change: {var_change}')
        n_iter += 1

    # Final least squares adjustment with estimated variance components
    ls_sol = least_squares(res_fcn, state_expanded, jac=jac, method='trf',
                           max_nfev=100, bounds=bounds, verbose=0, x_scale='jac', xtol=1e-15,
                           args=ls_args)

    return ls_sol
    
def construct_Q_eta(store_handle, baseline_handle, use_idxs):
    """ Construct a correlation matrix for consecutive GNSS observations of the same satellite """
    times = baseline_handle.datetime_array[use_idxs]

    source_time = []
    for time in times:
        source_time.append(store_handle.source_time_dict[time])
    source_time = np.array(source_time)

    Q_eta = np.zeros((len(use_idxs),len(use_idxs)))
    for source in store_handle.source_array:
        source_idxs = np.argwhere(np.equal(source_time, source))
        for run_idxs in consecutive_idxs(source_idxs.flatten()):
            if len(run_idxs)>0:
                Q_eta[run_idxs[0]:run_idxs[-1]+1,run_idxs[0]:run_idxs[-1]+1] = np.ones((len(run_idxs),len(run_idxs)))

    return Q_eta

def parbootstrap(zhat,Qzhat,Z,L,D,P0=0.995):
    """ Use integer bootstrapping to resolve ambiguities up to a confidence level """
    # Get the number of ambiguities from the dimension of the vc-matrix
    n = len(Qzhat)

    # Compute the bootstrapped success rate if all ambiguities would be fixed
    Ps = LAMBDA.SR_IB_2(D)

    k=0
    while Ps<P0 and k<(n-1):
        k+=1
        
        # Compute the bootstrapped success rate if the last n-k+1 ambiguities 
        # would be fixed
        Ps = LAMBDA.SR_IB_2(D[0][k:])

    if Ps > P0:
    
        zpar = LAMBDA.bootstrap(zhat[k:], L[k:,k:])
    
        Qzpar = Qzhat[k:,k:]
        Zpar  = Z[:,k:]
    
        # First k-1 ambiguities are adjusted based on correlation with the 
        # fixed ambiguities
        QP = Qzhat[:k,k:].dot(np.linalg.inv(Qzhat[k:,k:])) 
    
        zfixed=np.zeros(k)
        zfixed = zhat[:k] - QP.dot(zhat[k:]-zpar)
        zfixed = np.concatenate((zfixed,zpar),axis=0)
        nfixed = n-k
    else:
    
        zpar   = []
        Qzpar  = []
        Zpar   = []
        Ps     = np.nan
        zfixed = zhat
        nfixed = 0
        
    return zpar,Qzpar,Zpar,Ps,nfixed,zfixed  

def find_baselines_indices_for_triplet(baselines, triplet):
    """ Find the indices of the baselines in this closure combination """
    baselines_indices_in_triplet = []
    triplet_combinations = list(itertools.combinations(triplet, 2))
    for jdx, baseline in enumerate(baselines):
        if baseline in triplet_combinations:
            baselines_indices_in_triplet.append(jdx)
    return baselines_indices_in_triplet

def calc_residuals(state, ref_antenna, baselines, store_handle, antenna_handles, \
        clock_idxs, clock_poly_length, trop_idxs, trop_poly_length, disb_idxs, \
        baseline_handles=[], phase_delay=False, phase_only=False, use_amb_state=False, amb_state_idxs=[], \
        use_phase_weights=False, tikhonov_lambda=0, cont_penalty=0, phase_clock_idxs=[], phase_disb_idxs=[], combination_type=None,\
        integer_amb=[], inverse_Z_transform = []):
    """ Calculate the residuals vector from the current state vector. This function will be called in the estimator during iteration."""
    # set up state elements into appropriate arrays, check what has changed from previous iteration
    if phase_only is False:
        clock_states = state[clock_idxs]
    if phase_delay is True: 
        phase_clock_states = state[phase_clock_idxs]
    if use_amb_state is True:
        amb_state = state[amb_state_idxs]
    if store_handle.estimate_trop is True:
        trop_states = state[trop_idxs]
    if store_handle.estimate_disb is True:
        if phase_only is False:
            disb_states = state[disb_idxs]
    if store_handle.estimate_phase_disb is True:
        if phase_delay is True:
            phase_disb_states = state[phase_disb_idxs]
  
    if store_handle.global_linear_clock is True:
        idx_start = 1
    elif store_handle.global_quadratic_clock is True:
        idx_start = 2
    else:
        idx_start = 0

    count = 0
    count_ao = 0
    count_grav = 0
    for idx, antenna_handle in enumerate(antenna_handles):
        if antenna_handle.estimate_ao is True:
            ao_ant = state[clock_idxs.stop + count_ao]
            antenna_handle.axis_offset = ao_ant 
            count_ao += 1

        if antenna_handle.estimate_grav_def is True:
            grav_def_ant = state[clock_idxs.stop + count_ao + count_grav:clock_idxs.stop + count_ao + count_grav+2]
            antenna_handle.grav_def_model = grav_def_ant
            count_grav += 2

        if ref_antenna == antenna_handle.antenna_name:
            if store_handle.sol_type == 'GNSS' and (antenna_handle.estimate_ao is True or antenna_handle.estimate_grav_def is True):
                data_corrected = store_handle.correct_PR_CP(antenna_handle, phase=True)
                antenna_handle.hold_data(data_corrected)
        else:            
            # get elements of state vector for this antenna
            rxpos_state = state[count*3:count*3+3]
            times_gps = antenna_handle.times_gps 

            if phase_only is False:
                clock_state = clock_states[antenna_handle.range_clock_idxs]

            if phase_delay is True:
                phase_clock_state = phase_clock_states[antenna_handle.phase_clock_idxs]
            
            # get clock samples from state
            if phase_only is False:
                if store_handle.global_linear_clock is True or store_handle.global_quadratic_clock is True:
                    clock_state_global = clock_state[:idx_start]
                    if store_handle.stochastic_clock is True:
                        clock_samples = sample_global_poly_at_interval(clock_state_global, antenna_handle.clock_times, times_gps[0], times_gps[-1])
                    else:
                        clock_samples = sample_global_poly_at_interval(clock_state_global, times_gps)
                    clock_state = clock_state[idx_start:]
                else:
                    if store_handle.stochastic_clock is True:
                        clock_samples = np.zeros(len(clock_state))
                    else:
                        clock_samples = np.zeros(len(times_gps))

                if store_handle.stochastic_clock is False and clock_poly_length>0:
                    clock_samples += sample_poly_at_interval(clock_state, clock_poly_length, times_gps)
                elif store_handle.stochastic_clock is True:
                    clock_samples += clock_state
                else:
                    # havent added in bulk clock offset yet
                    clock_samples += clock_state[0]

                if antenna_handle.ppp_clock_active is True:
                    if store_handle.stochastic_clock is True:
                        antenna_handle.interp_ppp_clock(antenna_handle.clock_times)
                    clock_samples += antenna_handle.ppp_clock_samples

            if store_handle.estimate_trop is True and antenna_handle.estimate_trop is True:
                trop_state = trop_states[antenna_handle.trop_idxs]

            if store_handle.stochastic_trop is False and store_handle.estimate_trop is True and antenna_handle.estimate_trop is True:
                trop_samples = sample_poly_at_interval(trop_state, trop_poly_length, times_gps)
                antenna_handle.hold_trop(trop_samples)
            elif store_handle.stochastic_trop is True and antenna_handle.estimate_trop is True:
                trop_samples = trop_state
                antenna_handle.hold_trop(trop_samples)

            if store_handle.estimate_disb is True and phase_only is False:
                disb_state = disb_states[antenna_handle.range_disb_idxs]
                antenna_handle.hold_disb(disb_state)

            if phase_delay is True:
                if store_handle.global_linear_clock is True or store_handle.global_quadratic_clock is True:
                    phase_clock_state_global = phase_clock_state[:idx_start]
                    if store_handle.stochastic_clock is True:
                        phase_clock_samples = sample_global_poly_at_interval(phase_clock_state_global, antenna_handle.phase_clock_times, times_gps[0], times_gps[-1])
                    else:
                        phase_clock_samples = sample_global_poly_at_interval(phase_clock_state_global, times_gps)
                    phase_clock_state = phase_clock_state[idx_start:]
                else:
                    if store_handle.stochastic_clock is True:
                        phase_clock_samples = np.zeros(len(phase_clock_state))
                    else:
                        phase_clock_samples = np.zeros(len(times_gps))

                if store_handle.stochastic_clock is False and clock_poly_length>0:
                    phase_clock_samples += sample_poly_at_interval(phase_clock_state,\
                            clock_poly_length, times_gps, antenna_handle.phase_clock_start)
                elif store_handle.stochastic_clock is True:
                    phase_clock_samples += phase_clock_state
                else:
                    # havent added in bulk clock offset yet
                    phase_clock_samples += phase_clock_state[0]

                if antenna_handle.ppp_clock_active is True:
                    if store_handle.stochastic_clock is True:
                        antenna_handle.interp_ppp_clock(antenna_handle.phase_clock_times)
                    phase_clock_samples += antenna_handle.ppp_clock_samples

                if store_handle.estimate_phase_disb is True:
                    phase_disb_state = phase_disb_states[antenna_handle.phase_disb_idxs]
                    antenna_handle.hold_disb(phase_disb_state, phase_delay)

            rxpos_series, R_obj = store_handle.compute_tides(times_gps, rxpos_state, antenna_handle.antenna_name) 
            antenna_handle.update_pos_series(rxpos_series, R_obj, rxpos_state)
            if phase_only is False:
                antenna_handle.hold_clock(clock_samples)
            if phase_delay is True:
                antenna_handle.hold_clock(phase_clock_samples, phase_delay) 

            if store_handle.sol_type == 'VLBI':
                store_handle.compute_azel(times_gps, antenna_handle)
            elif store_handle.sol_type == 'GNSS':
                # Correct pseudorange/carrier phase data for the state
                data_corrected = store_handle.correct_PR_CP(antenna_handle, phase_delay, phase_only)
                antenna_handle.hold_data(data_corrected)

            count+=1

    if phase_delay is True:
        if use_amb_state is True:
            if len(inverse_Z_transform)>0: # doing iterative LAMBDA
                full_z_vec = np.concatenate((amb_state, integer_amb))
                full_amb = inverse_Z_transform.dot(full_z_vec)
            else: # doing first float ambiguity adjustment
                full_amb = amb_state

    residuals = np.array([],dtype=float)
    n_amb = 0 
    for jdx, baseline in enumerate(baselines): # generate differential measurements on the baselines
        antenna1_handle = antenna_handles[baseline[0]]
        antenna2_handle = antenna_handles[baseline[1]]
        if len(baseline_handles)>0:
            baseline_handle = baseline_handles[jdx]

        if store_handle.sol_type == 'VLBI':
            _, ant1_idxs, _ = np.intersect1d(antenna1_handle.times_gps, \
                    baseline_handle.datetime_array, return_indices=True)
            _, ant2_idxs, _ = np.intersect1d(antenna2_handle.times_gps, \
                    baseline_handle.datetime_array, return_indices=True)

            # get clock samples
            if phase_only is False:
                if store_handle.stochastic_clock is False:
                    clock_samples = antenna2_handle.clock_samples[ant2_idxs] - \
                            antenna1_handle.clock_samples[ant1_idxs]
                else:
                    _, ant1_idxs_clock, ant1_dt = np.intersect1d(antenna1_handle.clock_times, \
                            baseline_handle.datetime_array, return_indices=True)
                    _, ant2_idxs_clock, ant2_dt = np.intersect1d(antenna2_handle.clock_times, \
                            baseline_handle.datetime_array, return_indices=True)
                    clock_samples = np.zeros(len(baseline_handle.datetime_array))
                    clock_samples[ant2_dt] += antenna2_handle.clock_samples[ant2_idxs_clock]
                    clock_samples[ant1_dt] -= antenna1_handle.clock_samples[ant1_idxs_clock]
            else:
                clock_samples = []

            if phase_delay is True:
                if store_handle.stochastic_clock is False:
                    phase_clock_samples = antenna2_handle.phase_clock_samples[ant2_idxs] - \
                                antenna1_handle.phase_clock_samples[ant1_idxs]
                else:
                    _, ant1_idxs_phase, ant1_dt = np.intersect1d(antenna1_handle.phase_clock_times, \
                            baseline_handle.datetime_array, return_indices=True)
                    _, ant2_idxs_phase, ant2_dt = np.intersect1d(antenna2_handle.phase_clock_times, \
                            baseline_handle.datetime_array, return_indices=True)
                    phase_clock_samples = np.zeros(len(baseline_handle.datetime_array))
                    phase_clock_samples[ant2_dt] += antenna2_handle.phase_clock_samples[ant2_idxs_phase]
                    phase_clock_samples[ant1_dt] -= antenna1_handle.phase_clock_samples[ant1_idxs_phase]
            
            if store_handle.estimate_trop is True:
                if len(baseline_handles) == 0 or store_handle.stochastic_trop is False:
                    trop_samples1 = antenna1_handle.trop_samples[ant1_idxs] 
                    trop_samples2 = antenna2_handle.trop_samples[ant2_idxs] 
                else:
                    trop_samples1 = np.zeros(len(baseline_handle.datetime_array))
                    trop_samples2 = np.zeros(len(baseline_handle.datetime_array))
                    if antenna1_handle.estimate_trop is True:
                        _, ant1_idxs_trop, ant1_dt = np.intersect1d(antenna1_handle.trop_times, \
                                baseline_handle.datetime_array, return_indices=True)
                        trop_samples1[ant1_dt] += antenna1_handle.trop_samples[ant1_idxs_trop]
                    if antenna2_handle.estimate_trop is True:
                        _, ant2_idxs_trop, ant2_dt = np.intersect1d(antenna2_handle.trop_times, \
                                baseline_handle.datetime_array, return_indices=True)
                        trop_samples2[ant2_dt] += antenna2_handle.trop_samples[ant2_idxs_trop]
            else:
                trop_samples1 = []
                trop_samples2 = []

            if store_handle.src_type == 'GNSS':
                if phase_delay is True:
                    store_handle.model_group_phase_vlbi(antenna1_handle, antenna2_handle, baseline_handle, \
                            baseline_handle.datetime_array, clock_samples, trop_samples1, trop_samples2,\
                            phase_delay, phase_clock_samples, phase_only)
                else:
                    store_handle.model_group_phase_vlbi(antenna1_handle, antenna2_handle, baseline_handle, \
                            baseline_handle.datetime_array, clock_samples, trop_samples1, trop_samples2)
            elif store_handle.src_type == 'VLBI':
                if phase_delay is True:
                    store_handle.model_group_phase_farfield(antenna1_handle, antenna2_handle, baseline_handle, \
                            baseline_handle.datetime_array, clock_samples, trop_samples1, trop_samples2,\
                            phase_delay, phase_clock_samples, phase_only)
                else:
                    store_handle.model_group_phase_farfield(antenna1_handle, antenna2_handle, baseline_handle, \
                            baseline_handle.datetime_array, clock_samples, trop_samples1, trop_samples2)

        if phase_delay is True:
            # residual includes pseudorange and carrier phase difference
            if phase_only is False:
                if store_handle.sol_type == 'VLBI':
                    residuals_baseline = baseline_handle.group_delays-baseline_handle.group_delay_model
                    residuals_baseline = residuals_baseline[baseline_handle.range_data_idxs]
                elif store_handle.sol_type == 'GNSS':
                    residuals_baseline = find_diff_meas(antenna1_handle, antenna2_handle, baseline_handle, store_handle)
                range_weight_mat = get_obs_weights(store_handle, antenna1_handle, antenna2_handle, 'range',\
                        baseline_handle.range_data_idxs, baseline_handle)
                residuals_baseline = range_weight_mat@residuals_baseline
            else:
                residuals_baseline = np.array([])

            if use_amb_state is True:
                full_amb_baseline = full_amb[n_amb:n_amb+baseline_handle.n_amb_state]
                n_amb += baseline_handle.n_amb_state 
                if store_handle.sol_type == 'VLBI':
                    residuals_baseline_phase = find_diff_meas_phase_vlbi(baseline_handle,\
                        full_amb_baseline, combination_type, store_handle.iono_free)
                elif store_handle.sol_type == 'GNSS':
                    residuals_baseline_phase = find_diff_meas_phase(antenna1_handle, antenna2_handle, baseline_handle,\
                        full_amb_baseline, store_handle, combination_type)
            else:
                integer_amb_baseline = integer_amb[n_amb:n_amb+baseline_handle.n_amb_state]
                n_amb += baseline_handle.n_amb_state 
                if store_handle.sol_type == 'VLBI':
                    residuals_baseline_phase = find_diff_meas_phase_vlbi(baseline_handle,\
                        integer_amb_baseline, combination_type, store_handle.iono_free)
                elif store_handle.sol_type == 'GNSS':
                    residuals_baseline_phase = find_diff_meas_phase(antenna1_handle, antenna2_handle, baseline_handle,\
                        integer_amb_baseline, store_handle, combination_type)

            phase_weight_mat = get_obs_weights(store_handle, antenna1_handle, antenna2_handle, 'phase', \
                    baseline_handle.phase_data_idxs, baseline_handle, use_phase_weights)
            residuals_baseline_phase = phase_weight_mat@residuals_baseline_phase
            residuals_baseline = np.concatenate((residuals_baseline, residuals_baseline_phase))
        else:
            if store_handle.sol_type == 'VLBI':
                residuals_baseline = (baseline_handle.group_delays-baseline_handle.group_delay_model)[baseline_handle.range_data_idxs]
            elif store_handle.sol_type == 'GNSS':
                if len(baseline_handles)>0:
                    residuals_baseline = find_diff_meas(antenna1_handle, antenna2_handle, baseline_handle, store_handle)
                else:
                    residuals_baseline = find_diff_meas(antenna1_handle, antenna2_handle, None, store_handle)
            if len(baseline_handles)>0:
                range_weight_mat = get_obs_weights(store_handle, antenna1_handle, antenna2_handle, 'range', \
                        baseline_handle.range_data_idxs, baseline_handle)
            else:
                range_weight_mat = get_obs_weights(store_handle, antenna1_handle, antenna2_handle, 'range')
            residuals_baseline = range_weight_mat@residuals_baseline
            #if np.any(np.abs(residuals_baseline)>1e6): breakpoint()

        if jdx == 0:
            residuals = residuals_baseline
        else:
            residuals = np.concatenate((residuals, residuals_baseline))

    if store_handle.stochastic_clock is True:
        for antenna_handle in antenna_handles:
            if ref_antenna == antenna_handle.antenna_name:
                ref_handle = antenna_handle

        for antenna_handle in antenna_handles:
            if ref_antenna != antenna_handle.antenna_name:
                if phase_only is False:
                    process_variance_clock = get_process_variance_times(store_handle, antenna_handle, 'clock')
                    process_variance_ref_clock = get_process_variance_times(store_handle, ref_handle, 'clock', False, antenna_handle.clock_times)
                    clock_samples = antenna_handle.clock_samples.copy()
                    if antenna_handle.ppp_clock_active:
                        antenna_handle.interp_ppp_clock(antenna_handle.clock_times)
                        clock_samples -= antenna_handle.ppp_clock_samples
                    if store_handle.global_linear_clock is True or store_handle.global_quadratic_clock is True:
                        clock_state_global = clock_states[antenna_handle.range_clock_idxs][:idx_start]
                        clock_global = sample_global_poly_at_interval(clock_state_global, antenna_handle.clock_times,\
                                antenna_handle.times_gps[0], antenna_handle.times_gps[-1])
                        diff_clock = np.diff(clock_samples-clock_global)
                    else:
                        diff_clock = np.diff(clock_samples)
                    residuals = np.concatenate((residuals, 1/np.sqrt(process_variance_clock + process_variance_ref_clock)*diff_clock))
                if phase_delay is True:
                    process_variance_phase_clock = get_process_variance_times(store_handle, antenna_handle, 'clock', phase_delay)
                    process_variance_ref_phase_clock = get_process_variance_times(store_handle, ref_handle, 'clock', phase_delay, antenna_handle.phase_clock_times)
                    phase_clock_samples = antenna_handle.phase_clock_samples.copy()

                    if antenna_handle.ppp_clock_active:
                        antenna_handle.interp_ppp_clock(antenna_handle.phase_clock_times)
                        phase_clock_samples -= antenna_handle.ppp_clock_samples

                    if store_handle.global_linear_clock is True or store_handle.global_quadratic_clock is True:
                        phase_clock_state_global = phase_clock_states[antenna_handle.phase_clock_idxs][:idx_start]
                        phase_clock_global = sample_global_poly_at_interval(phase_clock_state_global, antenna_handle.phase_clock_times,\
                                antenna_handle.times_gps[0], antenna_handle.times_gps[-1])
                        diff_phase_clock = np.diff(phase_clock_samples-phase_clock_global)
                    else:
                        diff_phase_clock = np.diff(phase_clock_samples)
                    residuals = np.concatenate((residuals, 1/np.sqrt(process_variance_phase_clock + process_variance_ref_phase_clock)*diff_phase_clock))
                #if np.any(np.abs(residuals)>1e6): breakpoint()

    if store_handle.stochastic_trop is True:
        for antenna_handle in antenna_handles:
            if ref_antenna != antenna_handle.antenna_name and antenna_handle.estimate_trop is True:
                process_variance_trop = get_process_variance_times(store_handle, antenna_handle, 'trop')
                residuals = np.concatenate((residuals, 1/np.sqrt(process_variance_trop)*np.diff(antenna_handle.trop_samples)))

    #if phase_delay is True and len(antenna_handles)>2:
    #    # explicitly enforce phase closure
    #    baselines_closure = list(itertools.combinations(range(len(antenna_handles)), 3))
    #    for baseline_closure in baselines_closure:
    #        baseline_idxs = find_baselines_indices_for_triplet(baselines, baseline_closure)
    #        baseline1_handle = baseline_handles[baseline_idxs[0]]
    #        baseline2_handle = baseline_handles[baseline_idxs[1]]
    #        baseline3_handle = baseline_handles[baseline_idxs[2]]
    #        residuals_baseline = find_closure_model(baseline1_handle, baseline2_handle, baseline3_handle, \
    #                combination_type, store_handle.iono_free)
    #        residuals = np.concatenate((residuals, residuals_baseline))

    if cont_penalty > 0 and phase_delay is True and clock_poly_length>0: #use_amb_state is True:
        n_amb = 0 
        for jdx, baseline in enumerate(baselines): # generate differential measurements on the baselines
            antenna1_handle = antenna_handles[baseline[0]]
            antenna2_handle = antenna_handles[baseline[1]]
            baseline_handle = baseline_handles[jdx]
            if use_amb_state is True:
                full_amb_baseline = full_amb[n_amb:n_amb+baseline_handle.n_amb_state]
                n_amb += baseline_handle.n_amb_state 
                if store_handle.sol_type == 'VLBI':
                    residuals_baseline_phase = find_diff_meas_phase_vlbi(baseline_handle,\
                        full_amb_baseline, combination_type, store_handle.iono_free)
                elif store_handle.sol_type == 'GNSS':
                    residuals_baseline_phase = find_diff_meas_phase(antenna1_handle, antenna2_handle, baseline_handle,\
                        full_amb_baseline, store_handle, combination_type)
            else:
                integer_amb_baseline = integer_amb[n_amb:n_amb+baseline_handle.n_amb_state]
                n_amb += baseline_handle.n_amb_state 
                if store_handle.sol_type == 'VLBI':
                    residuals_baseline_phase = find_diff_meas_phase_vlbi(baseline_handle,\
                        integer_amb_baseline, combination_type, store_handle.iono_free)
                elif store_handle.sol_type == 'GNSS':
                    residuals_baseline_phase = find_diff_meas_phase(antenna1_handle, antenna2_handle, baseline_handle,\
                        integer_amb_baseline, store_handle, combination_type)
            phase_weight_mat = get_obs_weights(store_handle, antenna1_handle, antenna2_handle, 'phase', \
                    baseline_handle.phase_data_idxs, baseline_handle, use_phase_weights)
            avg_weight = np.mean(np.diag(phase_weight_mat))
            phase_clock_variation = get_clock_variation_baseline(ref_antenna, clock_poly_length, \
                    antenna1_handle, antenna2_handle, baseline_handle, phase_clock_states)
            residuals_diff = np.sqrt(cont_penalty) * avg_weight * np.diff(residuals_baseline_phase+phase_clock_variation)
            residuals = np.concatenate((residuals, residuals_diff))

    if tikhonov_lambda > 0 and use_amb_state is True:
        # ||Ax-b||^2 + lambda * ||L x||^2  = ||[Ax - b; sqrt(lambda) L x]||^2
        n_amb = 0 
        for jdx, baseline in enumerate(baselines): # generate differential measurements on the baselines
            antenna1_handle = antenna_handles[baseline[0]]
            antenna2_handle = antenna_handles[baseline[1]]
            baseline_handle = baseline_handles[jdx]
            full_amb_baseline = full_amb[n_amb:n_amb+baseline_handle.n_amb_state]
            n_amb += baseline_handle.n_amb_state 
            phase_weight_mat = get_obs_weights(store_handle, antenna1_handle, antenna2_handle, 'phase', \
                    baseline_handle.phase_data_idxs, baseline_handle, use_phase_weights)
            avg_weight = np.mean(np.diag(phase_weight_mat))
            residuals = np.concatenate((residuals, np.sqrt(tikhonov_lambda)*avg_weight*full_amb_baseline))

    store_handle.hold_state(state)
    if False: #True:
        antenna1_handle = antenna_handles[0]
        antenna2_handle = antenna_handles[2]
        times, ant1_idxs, ant2_idxs = np.intersect1d(antenna1_handle.times_gps, antenna2_handle.times_gps, return_indices=True)
        source_array = np.array([store_handle.source_time_dict[time] for time in times])
        breakpoint()
        print(residuals[residuals>1e5])
        print(source_array[residuals>1e5])
        print(times[residuals>1e5])

    return residuals

def calc_jac(state, ref_antenna, baselines, store_handle, antenna_handles, \
        clock_idxs, clock_poly_length, trop_idxs, trop_poly_length, disb_idxs, \
        baseline_handles=[], phase_delay=False, phase_only=False, use_amb_state=False, amb_state_idxs=[], \
        use_phase_weights=False, tikhonov_lambda=0, cont_penalty=0, phase_clock_idxs=[], phase_disb_idxs = [], combination_type=None,\
        integer_amb=[], inverse_Z_transform = []):
    """ Calculate the analytical Jacobian from the state vector. This function will be called in the estimator during iteration."""
    # set up state elements in appropriate arrays
    if phase_only is False:
        clock_states = state[clock_idxs]
    if phase_delay is True: 
        phase_clock_states = state[phase_clock_idxs]
        if use_amb_state is True:
            amb_state = state[amb_state_idxs]
    if store_handle.estimate_trop is True:
        trop_states = state[trop_idxs]

    if store_handle.estimate_disb is True:
        if phase_only is False:
            disb_states = state[disb_idxs]
    if store_handle.estimate_phase_disb is True:
        if phase_delay is True:
            phase_disb_states = state[phase_disb_idxs]

    n_measurements = 0
    if phase_delay is True:
        for jdx, baseline in enumerate(baselines): 
            baseline_handle = baseline_handles[jdx]
            n_measurements += len(baseline_handle.phase_data_idxs)
            if phase_only is False:
                n_measurements += len(baseline_handle.range_data_idxs)
    else:
        if len(baseline_handles)>0:
            for jdx, baseline_handle in enumerate(baseline_handles): 
                n_measurements += len(baseline_handle.range_data_idxs)
        else:
            for jdx, baseline in enumerate(baselines): 
                datetime_array = np.intersect1d(antenna_handles[baseline[0]].times_gps, antenna_handles[baseline[1]].times_gps)
                n_measurements += len(datetime_array)

    analytical_jac = np.zeros((n_measurements, len(state))) # full Jacobian
    # set up idxs as list instead of slice for some later logic
    clock_idxs_list = np.arange(clock_idxs.start,clock_idxs.stop)
    if phase_delay is True:
        phase_clock_idxs_list = np.arange(phase_clock_idxs.start,phase_clock_idxs.stop)
    if store_handle.estimate_trop is True:
        trop_idxs_list = np.arange(trop_idxs.start,trop_idxs.stop)

    if store_handle.estimate_disb is True and phase_only is False:
        disb_idxs_list = np.arange(disb_idxs.start,disb_idxs.stop)
    if store_handle.estimate_phase_disb is True:
        if phase_delay is True:
            phase_disb_idxs_list = np.arange(phase_disb_idxs.start,phase_disb_idxs.stop)

    ao_state_ant = []
    n_ao_state = 0
    for idx, antenna_handle in enumerate(antenna_handles):
        if antenna_handle.estimate_ao is True:
            ao_ant = state[clock_idxs.stop + count_ao]
            ao_state_ant.append(ao_ant)
            n_ao_state += 1
        else:
            ao_ant = None
            ao_state_ant.append(None)

    count_rx = 0
    count_ao = 0
    count_grav = 0
    weights = np.zeros((n_measurements,n_measurements))
    for idx, antenna_handle in enumerate(antenna_handles):
        if ref_antenna == antenna_handle.antenna_name:
            clock_state = []
            phase_clock_state = []
            if antenna_handle.estimate_ao is True:
                ao_ant = ao_state_ant[idx]
                if store_handle.sol_type == 'GNSS':
                    antenna_jac = store_handle.compute_analytical_jac(antenna_handle, \
                            clock_state, clock_poly_length, [], trop_poly_length, [], ao_ant, phase=True)
                len_ao = 1
                count_ao += 1
        else:
            # get elements of state vector for this antenna
            if antenna_handle.estimate_ao is True:
                ao_ant = ao_state_ant[idx]
                len_ao = 1
                count_ao += 1
            else:
                len_ao = 0
                ao_ant = None
            if antenna_handle.estimate_grav_def is True:
                len_grav = 2
                count_grav += 2
            else:
                len_grav = 0

            if phase_only is False:
                clock_state = clock_states[antenna_handle.range_clock_idxs]
            else:
                clock_state = []

            if phase_delay is True:
                phase_clock_state = phase_clock_states[antenna_handle.phase_clock_idxs]
            else:
                phase_clock_state = []
            
            if store_handle.estimate_trop is True and antenna_handle.estimate_trop is True:
                trop_state = trop_states[antenna_handle.trop_idxs]
            else:
                trop_state = []

            if store_handle.estimate_disb is True and phase_only is False:
                disb_state = disb_states[antenna_handle.range_disb_idxs]
                antenna_disb_idxs = disb_idxs_list[antenna_handle.range_disb_idxs]
            else:
                disb_state = []
            if store_handle.estimate_phase_disb is True and phase_delay is True:
                phase_disb_state = phase_disb_states[antenna_handle.phase_disb_idxs]
                antenna_phase_disb_idxs = phase_disb_idxs_list[antenna_handle.phase_disb_idxs]
            else:
                phase_disb_state = []

            if store_handle.sol_type == 'GNSS':
                # Correct pseudorange/carrier phase data for the state
                antenna_jac = store_handle.compute_analytical_jac(antenna_handle, \
                        clock_state, clock_poly_length, trop_state, trop_poly_length, \
                        disb_state, ao_ant, phase_delay, phase_clock_state, phase_disb_state, phase_only)

        # place this antenna's Jacobian in the right place in the full Jacobian
        num_samples = 0
        for jdx, baseline in enumerate(baselines):
            if len(baseline_handles) > 0: 
                baseline_handle = baseline_handles[jdx]
                datetime_array = baseline_handle.datetime_array
                _, ant1_idxs, _ = np.intersect1d(antenna_handles[baseline[0]].times_gps, \
                        datetime_array, return_indices=True)
                _, ant2_idxs, _ = np.intersect1d(antenna_handles[baseline[1]].times_gps, \
                        datetime_array, return_indices=True)
            else:
                datetime_array, ant1_idxs, ant2_idxs = np.intersect1d(antenna_handles[baseline[0]].times_gps, \
                        antenna_handles[baseline[1]].times_gps, return_indices=True)

            if idx in baseline:
                if idx == baseline[0]:
                    # antenna is subtracted in these residuals
                    sign=1
                    ant_deriv=1
                    ant_idxs = ant1_idxs
                else: 
                    sign=-1
                    ant_deriv=2
                    ant_idxs = ant2_idxs

                if store_handle.sol_type == 'VLBI': 
                    ao_ant1 = ao_state_ant[baseline[0]]
                    ao_ant2 = ao_state_ant[baseline[1]]
                    if phase_delay is True and (antenna_handle.antenna_name != ref_antenna or \
                            (antenna_handle.estimate_ao is True or antenna_handle.estimate_grav_def is True)):
                        if store_handle.src_type == 'GNSS':
                            antenna_jac = store_handle.compute_analytical_jac_vlbi(antenna_handles[baseline[0]], antenna_handles[baseline[1]], baseline_handle,\
                                    ant_deriv, datetime_array, clock_state, clock_poly_length, trop_state, trop_poly_length, ao_ant1, ao_ant2, phase_delay, \
                                    phase_clock_state, phase_only)
                        elif store_handle.src_type == 'VLBI':
                            antenna_jac = store_handle.compute_analytical_jac_farfield(antenna_handles[baseline[0]], antenna_handles[baseline[1]], baseline_handle,\
                                    ant_deriv, datetime_array, clock_state, clock_poly_length, trop_state, trop_poly_length, ao_ant1, ao_ant2, phase_delay, \
                                    phase_clock_state, phase_only)
                    elif antenna_handle.antenna_name != ref_antenna or (antenna_handle.estimate_ao is True or antenna_handle.estimate_grav_def is True):
                        if store_handle.src_type == 'GNSS':
                            antenna_jac = store_handle.compute_analytical_jac_vlbi(antenna_handles[baseline[0]], antenna_handles[baseline[1]], baseline_handle,\
                                    ant_deriv, datetime_array, clock_state, clock_poly_length, trop_state, trop_poly_length, ao_ant1, ao_ant2)
                        elif store_handle.src_type == 'VLBI':
                            antenna_jac = store_handle.compute_analytical_jac_farfield(antenna_handles[baseline[0]], antenna_handles[baseline[1]], baseline_handle,\
                                    ant_deriv, datetime_array, clock_state, clock_poly_length, trop_state, trop_poly_length, ao_ant1, ao_ant2)

                if phase_delay is True:
                    if antenna_handle.estimate_ao is True:
                        # deposit partials for antenna axis offset length on this baseline
                        if store_handle.sol_type == 'VLBI':
                            if phase_only is False:
                                analytical_jac[num_samples:num_samples + len(baseline_handle.range_data_idxs), clock_idxs.stop-1+count_ao] = \
                                        sign*antenna_jac[baseline_handle.range_data_idxs,3+len(clock_state):3+len(clock_state)+len_ao].flatten()
                            analytical_jac[num_samples + len(baseline_handle.range_data_idxs):num_samples + \
                                    len(baseline_handle.range_data_idxs) + len(baseline_handle.phase_data_idxs), \
                                    clock_idxs.stop-1+count_ao] = sign*antenna_jac[\
                                    baseline_handle.phase_data_idxs,3+len(clock_state):3+len(clock_state)+len_ao].flatten()
                        elif store_handle.sol_type == 'GNSS':
                            if phase_only is False:
                                analytical_jac[num_samples:num_samples + len(baseline_handle.range_data_idxs), clock_idxs.stop-1+count_ao] = \
                                        sign*antenna_jac[ant_idxs[baseline_handle.range_data_idxs],3+len(clock_state):3+len(clock_state)+len_ao].flatten()
                            analytical_jac[num_samples + len(baseline_handle.range_data_idxs):num_samples + \
                                    len(baseline_handle.range_data_idxs) + len(baseline_handle.phase_data_idxs), \
                                    clock_idxs.stop-1+count_ao] = sign*antenna_jac[\
                                    ant_idxs[baseline_handle.phase_data_idxs],3+len(clock_state):3+len(clock_state)+len_ao].flatten()

                    if antenna_handle.estimate_grav_def is True:
                        # deposit partials for antenna gravitational deformation on this baseline
                        if store_handle.sol_type == 'VLBI':
                            if phase_only is False:
                                analytical_jac[num_samples:num_samples + len(baseline_handle.range_data_idxs), \
                                        clock_idxs.stop+n_ao_state+count_grav-len_grav:clock_idxs.stop+n_ao_state+count_grav] = \
                                        sign*antenna_jac[baseline_handle.range_data_idxs,3+len(clock_state)+len_ao:3+len(clock_state)+len_ao+len_grav]
                                analytical_jac[num_samples + len(baseline_handle.range_data_idxs):num_samples + \
                                        len(baseline_handle.range_data_idxs) + len(baseline_handle.phase_data_idxs), \
                                        clock_idxs.stop+n_ao_state+count_grav-len_grav:clock_idxs.stop+n_ao_state+count_grav] = sign*antenna_jac[\
                                        baseline_handle.phase_data_idxs,3+len(clock_state)+len_ao:3+len(clock_state)+len_ao+len_grav]
                            else:
                                analytical_jac[num_samples:num_samples + len(baseline_handle.phase_data_idxs), \
                                        clock_idxs.stop+n_ao_state+count_grav-len_grav:clock_idxs.stop+n_ao_state+count_grav] = sign*antenna_jac[\
                                        baseline_handle.phase_data_idxs,3+len(clock_state)+len_ao:3+len(clock_state)+len_ao+len_grav]
                        elif store_handle.sol_type == 'GNSS':
                            if phase_only is False:
                                analytical_jac[num_samples:num_samples + len(baseline_handle.range_data_idxs), \
                                        clock_idxs.stop+n_ao_state+count_grav-len_grav:clock_idxs.stop+n_ao_state+count_grav] = \
                                        sign*antenna_jac[ant_idxs[baseline_handle.range_data_idxs],3+len(clock_state)+len_ao:3+len(clock_state)+len_ao+len_grav]
                                analytical_jac[num_samples + len(baseline_handle.range_data_idxs):num_samples + \
                                        len(baseline_handle.range_data_idxs) + len(baseline_handle.phase_data_idxs), \
                                        clock_idxs.stop+n_ao_state+count_grav-2:clock_idxs.stop+n_ao_state+count_grav] = sign*antenna_jac[\
                                        ant_idxs[baseline_handle.phase_data_idxs],3+len(clock_state)+len_ao:3+len(clock_state)+len_ao+len_grav]
                            else:
                                analytical_jac[num_samples:num_samples + len(baseline_handle.phase_data_idxs), \
                                        clock_idxs.stop+n_ao_state+count_grav-2:clock_idxs.stop+n_ao_state+count_grav] = sign*antenna_jac[\
                                        ant_idxs[baseline_handle.phase_data_idxs],3+len(clock_state)+len_ao:3+len(clock_state)+len_ao+len_grav]

                    if antenna_handle.estimate_ao is True or antenna_handle.estimate_grav_def is True:
                        if phase_only is False:
                            range_weight_mat = get_obs_weights(store_handle, antenna_handles[baseline[0]], antenna_handles[baseline[1]], 'range',\
                                    baseline_handle.range_data_idxs, baseline_handle)
                            weights[num_samples:num_samples + len(baseline_handle.range_data_idxs), \
                                    num_samples:num_samples + len(baseline_handle.range_data_idxs)] = range_weight_mat
                        phase_weight_mat = get_obs_weights(store_handle, antenna_handles[baseline[0]], antenna_handles[baseline[1]], 'phase',\
                                baseline_handle.phase_data_idxs, baseline_handle, use_phase_weights)
                        weights[num_samples:num_samples + len(baseline_handle.phase_data_idxs),\
                                num_samples:num_samples + len(baseline_handle.phase_data_idxs)] = phase_weight_mat

                    if ref_antenna == antenna_handle.antenna_name:
                        # only axis offset can be estimated for reference antenna
                        if phase_only is False:
                            num_samples += len(baseline_handle.range_data_idxs)
                        num_samples += len(baseline_handle.phase_data_idxs)
                        continue

                    if store_handle.estimate_trop is True and antenna_handle.estimate_trop is True:
                        # deposit troposphere partials for range and phase measurements on this baseline
                        antenna_trop_idxs = trop_idxs_list[antenna_handle.trop_idxs]

                    if phase_only is False:
                        antenna_clock_idxs = clock_idxs_list[antenna_handle.range_clock_idxs]
                        # deposit rxpos/clock/trop/disb partials for range measurements on this baseline
                        if store_handle.sol_type == 'VLBI':
                            analytical_jac[num_samples:num_samples + len(baseline_handle.range_data_idxs), count_rx*3:count_rx*3+3] = \
                                    -antenna_jac[baseline_handle.range_data_idxs,:3]                       
                            analytical_jac[num_samples:num_samples + len(baseline_handle.range_data_idxs), antenna_clock_idxs] = \
                                    sign*antenna_jac[baseline_handle.range_data_idxs,3:3+len(clock_state)]
                            if store_handle.estimate_trop is True and antenna_handle.estimate_trop is True:
                                analytical_jac[num_samples:num_samples + len(baseline_handle.range_data_idxs), antenna_trop_idxs] = \
                                        sign*antenna_jac[baseline_handle.range_data_idxs,3+len(clock_state)+len_ao+len_grav:3+len(clock_state)+len_ao+len_grav+len(trop_state)]  
                        elif store_handle.sol_type == 'GNSS':
                            analytical_jac[num_samples:num_samples + len(baseline_handle.range_data_idxs), count_rx*3:count_rx*3+3] = \
                                    sign*antenna_jac[ant_idxs[baseline_handle.range_data_idxs],:3]                       
                            analytical_jac[num_samples:num_samples + len(baseline_handle.range_data_idxs), antenna_clock_idxs] = \
                                    sign*antenna_jac[ant_idxs[baseline_handle.range_data_idxs],3:3+len(clock_state)]
                            if store_handle.estimate_trop is True and antenna_handle.estimate_trop is True:
                                analytical_jac[num_samples:num_samples + len(baseline_handle.range_data_idxs), antenna_trop_idxs] = \
                                        sign*antenna_jac[ant_idxs[baseline_handle.range_data_idxs],3+len(clock_state)+len_ao+len_grav:3+len(clock_state)+len_ao+len_grav+len(trop_state)]
                            if store_handle.estimate_disb is True:
                                analytical_jac[num_samples:num_samples + len(baseline_handle.range_data_idxs), antenna_disb_idxs] = \
                                        sign*antenna_jac[ant_idxs[baseline_handle.range_data_idxs],\
                                        3+len(clock_state)+len_ao+len_grav+len(trop_state):3+len(clock_state)+len_ao+len_grav+len(trop_state)+len(disb_state)]

                        range_weight_mat = get_obs_weights(store_handle, antenna_handles[baseline[0]], antenna_handles[baseline[1]], 'range',\
                                baseline_handle.range_data_idxs, baseline_handle)
                        weights[num_samples:num_samples + len(baseline_handle.range_data_idxs),\
                                num_samples:num_samples + len(baseline_handle.range_data_idxs)] = range_weight_mat
                        num_samples += len(baseline_handle.range_data_idxs)
                       
                    antenna_phase_clock_idxs = phase_clock_idxs_list[antenna_handle.phase_clock_idxs]
                    # deposit rxpos/clock partials for phase measurements on this baseline
                    if store_handle.sol_type == 'VLBI':
                        analytical_jac[num_samples:num_samples + len(baseline_handle.phase_data_idxs), count_rx*3:count_rx*3+3] = \
                                -antenna_jac[baseline_handle.phase_data_idxs,:3]                        
                        analytical_jac[num_samples:num_samples + len(baseline_handle.phase_data_idxs), antenna_phase_clock_idxs] = \
                                sign*antenna_jac[baseline_handle.phase_data_idxs,\
                                3+len(clock_state)+len(trop_state)+len_ao+len_grav:3+len(clock_state)+len(trop_state)+len_ao+len_grav+len(phase_clock_state)]
                        if store_handle.estimate_trop is True and antenna_handle.estimate_trop is True:
                            analytical_jac[num_samples:num_samples + len(baseline_handle.phase_data_idxs), antenna_trop_idxs] = \
                                    sign*antenna_jac[baseline_handle.phase_data_idxs,3+len(clock_state)+len_ao+len_grav:3+len(clock_state)+len_ao+len_grav+len(trop_state)]  
                    elif store_handle.sol_type == 'GNSS':
                        analytical_jac[num_samples:num_samples + len(baseline_handle.phase_data_idxs), count_rx*3:count_rx*3+3] = \
                                sign*antenna_jac[ant_idxs[baseline_handle.phase_data_idxs],:3]                        
                        analytical_jac[num_samples:num_samples + len(baseline_handle.phase_data_idxs), antenna_phase_clock_idxs] = \
                                sign*antenna_jac[ant_idxs[baseline_handle.phase_data_idxs],\
                                3+len(clock_state)+len(trop_state)+len_ao+len_grav+len(disb_state):3+len(clock_state)+len(trop_state)+len_ao+len_grav+len(disb_state)+len(phase_clock_state)]
                        if store_handle.estimate_trop is True and antenna_handle.estimate_trop is True:
                            analytical_jac[num_samples:num_samples + len(baseline_handle.phase_data_idxs), antenna_trop_idxs] = \
                                    sign*antenna_jac[ant_idxs[baseline_handle.phase_data_idxs],3+len(clock_state)+len_ao+len_grav:3+len(clock_state)+len_ao+len_grav+len(trop_state)]
                        if store_handle.estimate_phase_disb is True:
                            analytical_jac[num_samples:num_samples + len(baseline_handle.phase_data_idxs), antenna_phase_disb_idxs] = \
                                    sign*antenna_jac[ant_idxs[baseline_handle.phase_data_idxs],\
                                    3+len(clock_state)+len(trop_state)+len_ao+len_grav+len(disb_state)+len(phase_clock_state):\
                                    3+len(clock_state)+len(trop_state)+len_ao+len_grav+len(disb_state)+len(phase_clock_state)+len(phase_disb_state)]

                    phase_weight_mat = get_obs_weights(store_handle, antenna_handles[baseline[0]], antenna_handles[baseline[1]], 'phase',\
                            baseline_handle.phase_data_idxs, baseline_handle, use_phase_weights)
                    weights[num_samples:num_samples + len(baseline_handle.phase_data_idxs),\
                            num_samples:num_samples + len(baseline_handle.phase_data_idxs)] = phase_weight_mat
                    num_samples += len(baseline_handle.phase_data_idxs)
                else:
                    # range-only solution
                    if antenna_handle.estimate_ao is True:
                        # deposit partials for antenna axis offset length on this baseline
                        if store_handle.sol_type == 'VLBI':
                            analytical_jac[num_samples:num_samples + len(baseline_handle.range_data_idxs), clock_idxs.stop-1+count_ao] = \
                                    sign*antenna_jac[baseline_handle.range_data_idxs,3+len(clock_state):3+len(clock_state)+len_ao].flatten()
                        elif store_handle.sol_type == 'GNSS':
                            if len(baseline_handles)>0:
                                analytical_jac[num_samples:num_samples + len(baseline_handle.range_data_idxs), clock_idxs.stop-1+count_ao] = \
                                        sign*antenna_jac[ant_idxs[baseline_handle.range_data_idxs],3+len(clock_state):3+len(clock_state)+len_ao].flatten()
                            else:
                                analytical_jac[num_samples:num_samples + len(datetime_array), clock_idxs.stop-1+count_ao] = \
                                        sign*antenna_jac[ant_idxs,3+len(clock_state):3+len(clock_state)+len_ao].flatten()


                    if antenna_handle.estimate_grav_def is True:
                        # deposit partials for antenna gravitational deformation on this baseline
                        if store_handle.sol_type == 'VLBI':
                            analytical_jac[num_samples:num_samples + len(baseline_handle.range_data_idxs), \
                                    clock_idxs.stop+n_ao_state+count_grav-len_grav:clock_idxs.stop+n_ao_state+count_grav] = \
                                    sign*antenna_jac[baseline_handle.range_data_idxs,3+len(clock_state)+len_ao:3+len(clock_state)+len_ao+len_grav]
                        elif store_handle.sol_type == 'GNSS':
                            if len(baseline_handles)>0:
                                analytical_jac[num_samples:num_samples + len(baseline_handle.range_data_idxs), \
                                        clock_idxs.stop+n_ao_state+count_grav-len_grav:clock_idxs.stop+n_ao_state+count_grav] = \
                                        sign*antenna_jac[ant_idxs[baseline_handle.range_data_idxs],3+len(clock_state)+len_ao:3+len(clock_state)+len_ao+len_grav]
                            else:
                                analytical_jac[num_samples:num_samples + len(datetime_array), \
                                        clock_idxs.stop+n_ao_state+count_grav-len_grav:clock_idxs.stop+n_ao_state+count_grav] = \
                                        sign*antenna_jac[ant_idxs,3+len(clock_state)+len_ao:3+len(clock_state)+len_ao+len_grav].flatten()

                    if antenna_handle.estimate_ao is True or antenna_handle.estimate_grav_def is True:
                        range_weight_mat = get_obs_weights(store_handle, antenna_handles[baseline[0]], antenna_handles[baseline[1]], 'range',\
                                baseline_handle.range_data_idxs, baseline_handle)
                        weights[num_samples:num_samples + len(baseline_handle.range_data_idxs), \
                                num_samples:num_samples + len(baseline_handle.range_data_idxs)] = range_weight_mat

                    if ref_antenna == antenna_handle.antenna_name:
                        # only axis offset can be estimated for reference antenna
                        if len(baseline_handles)>0:
                            num_samples += len(baseline_handle.range_data_idxs)
                        else:
                            num_samples += len(datetime_array)
                        continue

                    if store_handle.estimate_trop is True and antenna_handle.estimate_trop is True:
                        # deposit troposphere partials for range and phase measurements on this baseline
                        antenna_trop_idxs = trop_idxs_list[antenna_handle.trop_idxs]

                    # deposit rxpos/clock/trop/disb partials for range measurements on this baseline
                    if len(baseline_handles)>0:
                        antenna_clock_idxs = clock_idxs_list[antenna_handle.range_clock_idxs]
                        if store_handle.sol_type == 'VLBI':
                            analytical_jac[num_samples:num_samples + len(baseline_handle.range_data_idxs), count_rx*3:count_rx*3+3] = \
                                    -antenna_jac[baseline_handle.range_data_idxs,:3]  
                            analytical_jac[num_samples:num_samples + len(baseline_handle.range_data_idxs), antenna_clock_idxs] = \
                                    sign*antenna_jac[baseline_handle.range_data_idxs,3:3+len(clock_state)] 
                            if store_handle.estimate_trop is True and antenna_handle.estimate_trop is True:
                                analytical_jac[num_samples:num_samples + len(baseline_handle.range_data_idxs), antenna_trop_idxs] = \
                                        sign*antenna_jac[baseline_handle.range_data_idxs,3+len(clock_state)+len_ao+len_grav:3+len(clock_state)+len_ao+len_grav+len(trop_state)]  
                        elif store_handle.sol_type == 'GNSS':
                            analytical_jac[num_samples:num_samples + len(baseline_handle.range_data_idxs), count_rx*3:count_rx*3+3] = \
                                    sign*antenna_jac[ant_idxs[baseline_handle.range_data_idxs],:3]  
                            analytical_jac[num_samples:num_samples + len(baseline_handle.range_data_idxs), antenna_clock_idxs] = \
                                    sign*antenna_jac[ant_idxs[baseline_handle.range_data_idxs],3:3+len(clock_state)]
                            if store_handle.estimate_trop is True and antenna_handle.estimate_trop is True:
                                analytical_jac[num_samples:num_samples + len(baseline_handle.range_data_idxs), antenna_trop_idxs] = \
                                        sign*antenna_jac[ant_idxs[baseline_handle.range_data_idxs],3+len(clock_state)+len_ao+len_grav:3+len(clock_state)+len_ao+len_grav+len(trop_state)]
                            if store_handle.estimate_disb is True:
                                analytical_jac[num_samples:num_samples + len(baseline_handle.range_data_idxs), antenna_disb_idxs] = \
                                        sign*antenna_jac[ant_idxs[baseline_handle.range_data_idxs],\
                                        3+len(clock_state)+len_ao+len_grav+len(trop_state):3+len(clock_state)+len_ao+len_grav+len(trop_state)+len(disb_state)]


                        range_weight_mat = get_obs_weights(store_handle, antenna_handles[baseline[0]], antenna_handles[baseline[1]], 'range',\
                                baseline_handle.range_data_idxs, baseline_handle)
                        weights[num_samples:num_samples + len(baseline_handle.range_data_idxs),\
                                num_samples:num_samples + len(baseline_handle.range_data_idxs)] = range_weight_mat
                        num_samples += len(baseline_handle.range_data_idxs)
                    else:
                        antenna_clock_idxs = clock_idxs_list[antenna_handle.range_clock_idxs]
                        analytical_jac[num_samples:num_samples + len(datetime_array), count_rx*3:count_rx*3+3] = \
                                sign*antenna_jac[ant_idxs,:3]  
                        analytical_jac[num_samples:num_samples + len(datetime_array), antenna_clock_idxs] = \
                                sign*antenna_jac[ant_idxs,3:3+len(clock_state)]
                        if store_handle.estimate_trop is True and antenna_handle.estimate_trop is True:
                            analytical_jac[num_samples:num_samples + len(datetime_array), antenna_trop_idxs] = \
                                    sign*antenna_jac[ant_idxs,3+len(clock_state)+len_ao+len_grav:3+len(clock_state)+len_ao+len_grav+len(trop_state)]

                        if store_handle.estimate_disb is True:
                            analytical_jac[num_samples:num_samples + len(datetime_array), antenna_disb_idxs] = \
                                    sign*antenna_jac[ant_idxs,\
                                    3+len(clock_state)+len_ao+len_grav+len(trop_state):3+len(clock_state)+len_ao+len_grav+len(trop_state)+len(disb_state)]

                        range_weight_mat = get_obs_weights(store_handle, antenna_handles[baseline[0]], antenna_handles[baseline[1]], 'range')
                        weights[num_samples:num_samples + len(datetime_array),\
                                num_samples:num_samples + len(datetime_array)] = range_weight_mat
                        num_samples += len(datetime_array)
            else:
                # track number of samples passed
                if phase_delay is True:
                    baseline_handle = baseline_handles[jdx]
                    if phase_only is False:
                        num_samples += len(baseline_handle.range_data_idxs)
                    num_samples += len(baseline_handle.phase_data_idxs)
                else:
                    if len(baseline_handles)>0:
                        baseline_handle = baseline_handles[jdx]
                        num_samples += len(baseline_handle.range_data_idxs)
                    else:
                        num_samples += len(datetime_array)

        # increment receiver state position 
        if ref_antenna != antenna_handle.antenna_name:
            count_rx += 1

    if use_amb_state is True:
        # deposit float ambiguity partials for phase measurements on this baseline
        # Note -- no sign b/c estimated on a per-baseline, not per-antenna basis
        n_amb = 0 
        num_samples = 0
        for jdx, baseline in enumerate(baselines):
            baseline_handle = baseline_handles[jdx]
            if store_handle.iono_free:
                wavelength = baseline_handle.comb_wavelength
            else:
                wavelength = baseline_handle.wavelength

            if phase_only is False: 
                num_samples += len(baseline_handle.range_data_idxs)

            for slip_idx, slip_slice in enumerate(baseline_handle.slip_slices_arr):
                idxs_slice, _, idxs_residuals = np.intersect1d(slip_slice, baseline_handle.phase_data_idxs, return_indices=True)
                idxs_full = idxs_residuals + num_samples 
                if len(inverse_Z_transform)>0:
                    # each Z-domain ambiguity contributes to each of the A-domain ambiguities
                    analytical_jac[idxs_full, amb_state_idxs.start:]\
                            = wavelength*inverse_Z_transform[slip_idx+n_amb, :len(amb_state)]
                else:
                    analytical_jac[idxs_full, amb_state_idxs.start+n_amb+slip_idx] = wavelength
                
            num_samples += len(baseline_handle.phase_data_idxs)
            n_amb += baseline_handle.n_amb_state

    if store_handle.stochastic_clock is True:
        if store_handle.global_linear_clock is True:
            idx_start = 1
        elif store_handle.global_quadratic_clock is True:
            idx_start = 2
        else:
            idx_start = 0


        for antenna_handle in antenna_handles:
            if ref_antenna == antenna_handle.antenna_name:
                ref_handle = antenna_handle

        idx_clock = 0
        idx_phase_clock = 0
        for idx, antenna_handle in enumerate(antenna_handles):
            if ref_antenna != antenna_handle.antenna_name:
                if idx_clock != 0 and phase_only is False:
                    # need to skip an index when we move to a new antenna handle
                    idx_clock += 1 + idx_start 
                if idx_phase_clock != 0 and phase_delay is True:
                    idx_phase_clock += 1 + idx_start
                if phase_only is False:
                    jac_ant_clock = np.zeros((len(antenna_handle.clock_times)-1, analytical_jac.shape[1]))
                    process_variance_clock = get_process_variance_times(store_handle, antenna_handle, 'clock')
                    process_variance_ref_clock = get_process_variance_times(store_handle, ref_handle, 'clock', False, antenna_handle.clock_times)
                    for jdx in range(len(antenna_handle.clock_times)-1):
                        jac_ant_clock[jdx, clock_idxs.start+idx_clock+idx_start] = -1/np.sqrt(process_variance_clock[jdx] + process_variance_ref_clock[jdx])
                        jac_ant_clock[jdx, clock_idxs.start+idx_clock+1+idx_start] = 1/np.sqrt(process_variance_clock[jdx] + process_variance_ref_clock[jdx])
                        idx_clock += 1
                    analytical_jac = np.vstack([analytical_jac, jac_ant_clock])

                if phase_delay is True:
                    jac_ant_phase_clock = np.zeros((len(antenna_handle.phase_clock_times)-1, analytical_jac.shape[1]))
                    process_variance_phase_clock = get_process_variance_times(store_handle, antenna_handle, 'clock', phase_delay)
                    process_variance_ref_phase_clock = get_process_variance_times(store_handle, ref_handle, 'clock', phase_delay, antenna_handle.phase_clock_times)
                    for jdx in range(len(antenna_handle.phase_clock_times)-1):
                        jac_ant_phase_clock[jdx, phase_clock_idxs.start+idx_phase_clock+idx_start] = -1/np.sqrt(process_variance_phase_clock[jdx] + process_variance_ref_phase_clock[jdx])
                        jac_ant_phase_clock[jdx, phase_clock_idxs.start+idx_phase_clock+1+idx_start] = 1/np.sqrt(process_variance_phase_clock[jdx] + process_variance_ref_phase_clock[jdx])
                        idx_phase_clock += 1
                    analytical_jac = np.vstack([analytical_jac, jac_ant_phase_clock])

    if store_handle.stochastic_trop is True:
        idx_trop = 0
        for idx, antenna_handle in enumerate(antenna_handles):
            if ref_antenna != antenna_handle.antenna_name and antenna_handle.estimate_trop is True:
                if idx_trop != 0: 
                    # need to skip an index when we move to a new antenna handle
                    idx_trop += 1 
                process_variance_trop = get_process_variance_times(store_handle, antenna_handle, 'trop')
                jac_ant_trop = np.zeros((len(antenna_handle.trop_times)-1, analytical_jac.shape[1]))
                for jdx in range(len(antenna_handle.trop_times)-1):
                    jac_ant_trop[jdx, trop_idxs.start+idx_trop] = -1/np.sqrt(process_variance_trop[jdx])
                    jac_ant_trop[jdx, trop_idxs.start+idx_trop+1] = 1/np.sqrt(process_variance_trop[jdx])
                    idx_trop += 1
                analytical_jac = np.vstack([analytical_jac, jac_ant_trop])

    if cont_penalty > 0 and phase_delay is True and clock_poly_length>0: #use_amb_state is True:
        # initialize differenced Jacobian matrix
        cont_jac = np.array([], dtype=float)
        if use_amb_state is True:
            if len(inverse_Z_transform)>0: # doing iterative LAMBDA
                full_z_vec = np.concatenate((amb_state, integer_amb))
                full_amb = inverse_Z_transform.dot(full_z_vec)
            else: # doing first float ambiguity adjustment
                full_amb = amb_state
        else:
            full_amb = integer_amb

        num_samples = 0
        for jdx, baseline in enumerate(baselines):
            baseline_handle = baseline_handles[jdx]
            antenna1_handle = antenna_handles[baseline[0]]
            antenna2_handle = antenna_handles[baseline[1]]
            phase_weight_mat = get_obs_weights(store_handle, antenna1_handle, antenna2_handle, 'phase', \
                    baseline_handle.phase_data_idxs, baseline_handle, use_phase_weights)
            avg_weight = np.mean(np.diag(phase_weight_mat))
            if phase_only is False: 
                num_samples += len(baseline_handle.range_data_idxs)
            # get the difference of relevant rows
            clock_derivs = get_clock_derivs_baseline(ref_antenna, clock_poly_length, \
                    antenna1_handle, antenna2_handle, baseline_handle, phase_clock_states)
            dh_dx = np.diff(analytical_jac[num_samples:num_samples + len(baseline_handle.phase_data_idxs)], axis=0)
            dc_dx = np.diff(clock_derivs, axis=0)
            cont_jac_baseline = np.zeros_like(dh_dx)
            cont_jac_baseline += dh_dx
            cont_jac_baseline[:,phase_clock_idxs] = cont_jac_baseline[:,phase_clock_idxs] + dc_dx
            cont_jac_baseline = np.sqrt(cont_penalty) * avg_weight * cont_jac_baseline
            if len(cont_jac) == 0:
                cont_jac = cont_jac_baseline
            else:
                cont_jac = np.vstack([cont_jac, cont_jac_baseline])
            num_samples += len(baseline_handle.phase_data_idxs)

        analytical_jac = np.vstack([analytical_jac, cont_jac])

    if tikhonov_lambda>0 and use_amb_state is True:
        # modify the analytical jacobian for the regularization
        if len(inverse_Z_transform)>0: # doing iterative LAMBDA
            full_z_vec = np.concatenate((amb_state, integer_amb))
            full_amb = inverse_Z_transform.dot(full_z_vec)
        else: # doing first float ambiguity adjustment
            full_amb = amb_state
        amb_idxs = list(range(len(full_amb)))
        tikhonov_L = np.zeros((len(amb_idxs), analytical_jac.shape[1]))
        n_amb = 0 
        for jdx, baseline in enumerate(baselines): # generate differential measurements on the baselines
            antenna1_handle = antenna_handles[baseline[0]]
            antenna2_handle = antenna_handles[baseline[1]]
            baseline_handle = baseline_handles[jdx]
            amb_idxs_baseline = amb_idxs[n_amb:n_amb+baseline_handle.n_amb_state]
            n_amb += baseline_handle.n_amb_state 
            phase_weight_mat = get_obs_weights(store_handle, antenna1_handle, antenna2_handle, 'phase', \
                    baseline_handle.phase_data_idxs, baseline_handle, use_phase_weights)
            avg_weight = np.mean(np.diag(phase_weight_mat))
            if len(inverse_Z_transform)>0:
                tikhonov_L_baseline = avg_weight*inverse_Z_transform[amb_idxs_baseline,:len(amb_state)]
                tikhonov_L[amb_idxs_baseline,amb_state_idxs.start:] += tikhonov_L_baseline
            else:
                for index in amb_idxs_baseline:
                    tikhonov_L[index, index+amb_state_idxs.start] = avg_weight
        analytical_jac = np.vstack([analytical_jac, np.sqrt(tikhonov_lambda) * tikhonov_L])

    analytical_jac[:weights.shape[0],:] = weights@analytical_jac[:weights.shape[0],:]

    return analytical_jac
