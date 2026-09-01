#!/usr/bin/python
"""
Generate local tie vectors using either GNSS (pseudorange, carrier phase) 
or VLBI (group delay, phase delay) data with observations of either 
GNSS satellites or natural radio sources.

==============================================================================
  This file is part of the PyLocalTie software package.  It has been prepared 
  under the NASA Open-Source Science initiative.

  This is free software; you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License as published
  by the Free Software Foundation; either version 3.0 of the License, or
  any later version.

  We are distributing this in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License; if not, write to the Free Software Foundation,
  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110, USA

  This software was developed by Applied Research Laboratories at the
  University of Texas at Austin, under NASA Grants 80NSSC24K0828 and 
  80NSSC20K1732.

  Copyright 2025, The Board of Regents of The University of Texas System
==============================================================================
"""

import LAMBDA
import os
use_custom_version = os.getenv('USE_CUSTOM_GEORINEX', 'false').lower() == 'true'
if use_custom_version:
    import sys
    sys.path.insert(0, '/home/jskeens/oscar_dir/scratch/jskeens')
    sys.path.insert(0, '/trashcan/scratch/jskeens')
    from georinex_custom import load
    #sys.path.insert(0, '/sgl/ceph/work/jskeens')
    #try: 
    #    from georinex_custom import load
    #except: 
    #    sys.path.insert(0, '/home/jskeens/oscar_dir/scratch/jskeens')
    #    sys.path.insert(0, '/trashcan/scratch/jskeens')
    #    from georinex_custom import load
else:
    from georinex import load

import xarray as xr
import datetime
import itertools
import random
import re
from pandas import Timestamp
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import least_squares
from scipy.interpolate import make_interp_spline
from scipy.stats import linregress, mode
import scipy.constants as const
from copy import deepcopy
import argparse
import matplotlib.dates as mdates

from gnsstk import std_vector_string, Position, AntennaStore, AntexData, OceanLoadTides, PoleTides, \
                  AtmLoadTides, SolarSystem, GlobalTropModel, SaasTropModel, NeillTropModel

from single_diff_tools import import_key_gnss, import_data_vlbi, import_data_vlbi_farfield, import_data_vlbi_ngs, import_data_vlbi_vgosdb, \
                  import_data_vlbi_vda, write_SINEX, datetime64_to_mjd, map_datasets, import_data_nc_sim,\
                  find_common_epochs, BaselineInfo, AntennaInfo, GNSSTKStores, ECEF2ECI, slip_detect_MW, slip_detect_single_freq,\
                  slip_detect_phase_delay, sample_poly_at_interval, trim_amb_Zdom, trim_amb_state, gen_phase_clock_state, adjust_stoch_params, thin_data,\
                  analyze_ls_solution, resolve_float_amb, construct_float_amb, remove_outliers, iterative_remove_outliers, calc_residuals, \
                  calc_jac, plot_time_units, read_src, date_to_common, date_to_mjd, detect_unresolved_amb_vlbi, \
                  detect_unresolved_amb_gnss, set_bounds_phase_clock, union_of_slices, get_residuals, iterative_weight_adjust, iterative_weight_adjust_ls_vce, \
                  iterative_weight_adjust_LS_VCE_full, sample_global_poly_at_interval, gen_key, VMF3Model, get_spanning_tree, \
                  read_thermal_deformation_coeffs, get_obs_weights, find_cont_penalty, mcmc_correlation, ls_correlation, load_kernel_parameters, vlbi_transform_data,\
                  write_vda_phase, NavStore

MIN_SLICE = 0
MIN_TROP_DIST = 0.5e3 # 1 km
MU_EARTH = 3.9860044188e14
TK_LAMBDA = 1e-9
#TK_LAMBDA = 1e-9
CT_PENALTY = 1000
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

def np_datetime64(arg, unit='ns'):
    """
    Convert an ISO-8601 string to numpy.datetime64.

    A single line `np.datetime64(arg)` also works, but wrapping it lets
    you control the unit and emit a clean argparse error when parsing
    fails.
    """
    try:
        return np.datetime64(arg, unit)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid datetime '{arg}': {exc}")

def add_args_to_parser(parser_in):
    """Add arguments to parser."""
    parser.add_argument("-r", dest="rinex_files", action="append", type=str, metavar="FILE", nargs="+")
    parser.add_argument("-e", dest="eph_files", action="append", type=str, nargs="+", help = 'Ephemeris file. Add files for day before and after experiment too.')
    parser.add_argument("--src_type", default=None, required=True,
                         help="Type of sources observed in the experiment (GNSS or VLBI)"
                         )
    parser.add_argument("--key_file", default=None,
                         help="Name of key file determining schedule. Source prefixes must match RINEX satellite names"
                         )
    parser.add_argument("--src_file", default=None,
                         help="Name of source file (i.e. glo.src) giving far-field source positions in RA/Dec if using fringe_file"
                         )
    parser.add_argument("--fringe_file", default=None,
                         help="PIMA fringe file to read. Antenna names supplied with --antenna_name must match those in fri file"
                         )
    parser.add_argument("--nc_sim_file", default=None,
                         help="Simulated data file from sim_local_tie.py to read. Works with either GNSS or VLBI data. Antenna names supplied with --antenna_name must match those in file."
                         )
    parser.add_argument("--ngs_file", default=None,
                         help="NGS data file to read. Antenna names supplied with --antenna_name must match those in file"
                         )
    parser.add_argument("--vda_file", default=None,
                         help="vgosda data file to read. antenna names supplied with --antenna_name must match those in file"
                         )
    parser.add_argument("--output_vda_file", default=None,
                         help="If specified (and --vda_file specified), will write phase delay solution to vgosda file for transfer to other software"
                         )
    parser.add_argument("--L4R_file", default=None,
                         help="L4R ionosphere correction file to read for long baseline observations. Antenna names supplied with --antenna_name must match those in file"
                         )
    parser.add_argument("--L4R_names",
                         help="Usage: --L4R_names {ANTENNA_NAME}. Set the GNSS station name to use for L4R. Order of --antenna_names.",
                         default=[], 
                         action='append')
    parser.add_argument("--ZWD_files", default=[], action='append',
                         help="ZWD correction file (proc log from PrecisePos) to read for long baseline observations. USE THE SAME TROP MODEL USED BY PRECISEPOS WITH THIS OPTION"
                         )
    parser.add_argument("--ZWD_antennas",
                         help="Usage: --ZWD_antennas {ANTENNA_NAME}. Set the GNSS station name to use for L4R. Order of --antenna_names.",
                         default=[], 
                         action='append')
    parser.add_argument("--vgosDB", default=None,
                         help="untarred vgosDB directory location to read. Antenna names supplied with --antenna_name must match subdirectory locations"
                         )
    parser.add_argument("--band", default=None,
                         help="If vgosDB, specify band to read."
                         )
    parser.add_argument("--utc2gps", default=None, type=float,
                         help="Number of seconds GPS is ahead of UTC (reqd for ngs_file)"
                         )
    parser.add_argument("--ant_offset", action="append", type=str, nargs="+", help="Receiver offset  as 'N E U' (m), for receivers with a monument different from phase center")
    parser.add_argument("--ant_offset_names",
                         help="Usage: --ant_offset_names {ANTENNA_NAME}. Receivers with an NEU offset, order of ant_offset",
                         default=[], 
                         action='append')
    parser.add_argument("--rxpos", action="append", type=str, nargs="+", help="Receiver position  as 'X Y Z' (m)")
    parser.add_argument("--OBX_file", default=None, action="extend", nargs="+",
                         help="Name of high-rate IGS OBX file describing GNSS satellite orientations."
                         )
    parser.add_argument("--clock_file", default=None, nargs="?",
                         help="Text file giving clock bias in meters for reference clock. Column 1 second of day, column 2 clock value at epoch. See example."
                         )
    parser.add_argument("--bulk_clock", dest="bulk_clocks", action="append", type=float, nargs="+", 
                         help = 'Bulk clock offsets for the antennas (m), order of antenna input. Convention: positive late (same as SNAPPER). (required)')
    parser.add_argument("--antex_file",
                         help="Name of antex file that holds phase center data for included antennas."
                         )
    parser.add_argument("--atmfile",
                         help="Atmospheric loading file."
                         )
    parser.add_argument("--oceanfile",
                         help="Ocean loading file.",
                         )
    parser.add_argument("--SSEfile",
                         help="Solar System ephemeris binary file.",
                         )
    parser.add_argument("--ant_info_file",
                         help="File containing VLBI telescope thermal deformation coefficients.",
                         )
    parser.add_argument("--earthfile",
                         help="EOP file (finals.all).",
                         )
    parser.add_argument("--ref_antenna",
                         help="Name of the reference antenna (required, this position is not estimated).",
                         )
    parser.add_argument(
        "--begin_experiment",
        metavar="YYYY-MM-DDThh:mm[:ss]",
        type=np_datetime64,               
        help="ISO-8601 start time (e.g. 2025-07-31T08:00). Used for gen_key if key file not supplied (GNSS data)"
    )
    parser.add_argument(
        "--end_experiment",
        metavar="YYYY-MM-DDThh:mm[:ss]",
        type=np_datetime64,
        help="ISO-8601 end time (e.g. 2025-07-31T18:00). Used for gen_key if key file not supplied (GNSS data)"
    )
    parser.add_argument("--vlbi_antennas",
                         help="Usage: --vlbi_antenna {ANTENNA_NAME}. Designate an antenna as VLBI. Will have to supply axis offset",
                         default=[], 
                         action='append')
    parser.add_argument("--cable_cal_antennas",
                         help="Usage: --cable_cal_antennas {ANTENNA_NAME}. Designate that an antenna has a cable calibration file that will be supplied with --cable_cal_files",
                         default=[], 
                         action='append')
    parser.add_argument("--cable_cal_files",
                         help="Usage: --cable_cal_files {FILE_NAME}. Supply a DiFX cable calibration file associated with an antenna given in --cable_cal_antennas (same order of arguments)",
                         default=[], 
                         action='append')
    parser.add_argument("--ppp_clock_antennas",
                         help="Usage: --ppp_clock_antennas {ANTENNA_NAME}. Designate that an antenna has a Precise Point Positioning clock file that will be supplied with --ppp_clock_files",
                         default=[], 
                         action='append')
    parser.add_argument("--ppp_clock_files",
                         help="Usage: --ppp_clock_files {FILE_NAME}. Supply a Precise Point Positioning clock file for an antenna given in --ppp_clock_antennas (same order of arguments). Clock file is column 1 second of day, column 2 clock value (m).",
                         default=[], 
                         action='append')
    parser.add_argument("--v3gr_files",
                         help="Usage: --v3gr_files {FILE_NAME}. Give a VMF3 GRID file for the V3GR trop function",
                         default=[], 
                         action='append')
    parser.add_argument("--vmf3_files",
                         help="Usage: --vmf3_files {FILE_NAME}. Give a VMF3 station-based zwd/zhd + mapping function file for the VMF3 trop function.",
                         default=[], 
                         action='append')
    parser.add_argument("--vmf3_stations",
                         help="Usage: --vmf3_stations {STATION_NAME}. Specify which VMF3 station will be used to find VMF3 trop parameters, should be same length as --antenna_names.",
                         default=[], 
                         action='append')
    parser.add_argument("--vmf3_grad_files",
                         help="Usage: --vmf3_files {FILE_NAME}. Give a VMF3 station-based east/north gradient file for the VMF3 trop function",
                         default=[], 
                         action='append')
    parser.add_argument("--orog_file",
                         help="Usage: --orog_file {FILE_NAME}. Give an orography file for the V3GR trop function",
                         type=str)
    parser.add_argument("--ell_file",
                         help="Usage: --ell_file {FILE_NAME}. Give a VMF3 station position file for the VMF3 trop function",
                         type=str)
    parser.add_argument("--weather_cal_antennas",
                         help="Usage: --weather_cal_antennas {ANTENNA_NAME}. Designate that an antenna has a weather calibration file that will be supplied with --cable_cal_files",
                         default=[], 
                         action='append')
    parser.add_argument("--weather_cal_files",
                         help="Usage: --weather_cal_files {FILE_NAME}. Supply a DiFX weather calibration file associated with an antenna given in --weather_cal_antennas (same order of arguments)",
                         default=[], 
                         action='append')
    parser.add_argument("--axis_offsets",
                         help="VLBI antenna axis offset in meters. Repeat argument in same order for more than 1 VLBI antenna.",
                         default=[], 
                         action='append')
    parser.add_argument("--clock_poly_length",
                         type=float,
                         default=15,
                         help="Length of interval for piecewise-linear clock model (min).",
                         )
    parser.add_argument("--trop_poly_length",
                         type=float,
                         default=0,
                         help="Length of interval for piecewise-linear troposepher wet zenith delay model (min), 0=disabled.",
                         )
    parser.add_argument("--stochastic_clock",
                         action="store_true",
                         default=False,
                         help="Use a stochastic (Kalman filter-like) model for clock bias. Replaces clock_poly_length",
                         )
    parser.add_argument("--baseline_strategy",
                         type=str,
                         default='Obs-Max',
                         help="If using GNSS observables, choose a strategy for forming the spanning tree of baselines. Options: Obs-Max, Shortest",
                         )
    parser.add_argument("--stochastic_trop",
                         action="store_true",
                         default=False,
                         help="Use a stochastic (Kalman filter-like) model for clock bias. Replaces trop_poly_length",
                         )
    parser.add_argument("--estimate_disb",
                         action="store_true",
                         default=False,
                         help="Estimate differential inter-GNSS system (GPS, Galileo, BeiDou) code biases caused by receiver hardware (necessary for heterogeneous receivers)." 
                               "NB: this assume that all included antennas have observations from the same set of systems.",
                         )
    parser.add_argument("--estimate_phase_disb",
                         action="store_true",
                         default=False,
                         help="Estimate differential inter-GNSS system (GPS, Galileo, BeiDou) phase biases caused by receiver hardware (usually not necessary)." 
                               "NB: this assume that all included antennas have observations from the same set of systems.",
                         )
    parser.add_argument("--plot_intermediate_results",
                         action="store_true",
                         default=False,
                         help="Plot intermediate analysis results. WARNING: LOTS of plots.",
                         )
    parser.add_argument("--recursive_amb",
                         action="store_true",
                         default=False,
                         help="Do recursive ambiguity fixing (fix and redefine floating point ambiguities until no cycle slip detections found).",
                         )
    parser.add_argument("--trop_global",
                         action="store_true",
                         default=False,
                         help="Do tropospheric delay modeling with Global model.",
                         )
    parser.add_argument("--trop_saas",
                         action="store_true",
                         default=False,
                         help="Do tropospheric delay modeling with Saastamoinen model.",
                         )
    parser.add_argument("--trop_neill",
                         action="store_true",
                         default=False,
                         help="Do tropospheric delay modeling with Neill model.",
                         )
    parser.add_argument("--trop_vmf3",
                         action="store_true",
                         default=False,
                         help="Do tropospheric delay modeling with station-based VMF3 model (highest accuracy). Need --vmf3_files and --ell_file for this model.",
                         )
    parser.add_argument("--trop_v3gr",
                         action="store_true",
                         default=False,
                         help="Do tropospheric delay modeling with V3GR (VMF3 GRID) model. Need --v3gr_files and --orog_file for this model.",
                         )
    parser.add_argument("--igs_data",
                         action="store_true",
                         default=False,
                         help="Trigger if inputting 30-second cadence IGS data. Needed for proper time referencing logic",
                         )
    parser.add_argument("--estimate_ao",
                         help="Usage: --estimate_ao {ANTENNA_NAME}. Estimate the axis offset of a VLBI telescope.",
                         default=[], 
                         action='append')
    parser.add_argument("--estimate_grav_def",
                         help="Usage: --estimate_grav_def {ANTENNA_NAME}. Estimate the gravitational deformation of a VLBI telescope.",
                         default=[], 
                         action='append')
    parser.add_argument("--global_linear_clock",
                         action="store_true",
                         default=False,
                         help="Estimate a full-experiment linear clock (along with either stochastic or polynomial clock model).",
                         )
    parser.add_argument("--global_quadratic_clock",
                         action="store_true",
                         default=False,
                         help="Estimate a full-experiment quadratic clock (along with either stochastic or polynomial clock model).",
                         )
    parser.add_argument("--tikhonov_reg",
                         action="store_true",
                         default=False,
                         help="Regularize the float ambiguity solution via Tikhonov (L2) regularization.",
                         )
    parser.add_argument("--continuity_penalty",
                         action="store_true",
                         default=False,
                         help="Enforce a continuity penalty to aid in phase delay ambiguity resolution.",
                         )
    parser.add_argument("--do_mcmc_correlation",
                         action="store_true",
                         default=False,
                         help="Estimate the correlation between measurements using Markov Chain Monte Carlo (MCMC) sampling.",
                         )
    parser.add_argument("--do_ls_vce_correlation",
                         action="store_true",
                         default=False,
                         help="Estimate the correlation between measurements using Least Squares Variance Component Estimation  (LS-VCE).",
                         )
    parser.add_argument("--load_covariance_kernel_range",
                         default=None,
                         help="Covariance kernel file to model a full rank covariance matrix for range (group delay or pseudorange) measurements.",
                         )
    parser.add_argument("--load_covariance_kernel_phase",
                         default=None,
                         help="Covaraince kernel file to model a full rank covariance matrix for phase measurements.",
                         )
    parser.add_argument("--analytical_delay",
                         action="store_true",
                         default=False,
                         help="Use the adapted Jaron, Nothnagel (2019) analytical delay model (recommended) instead of iterative light time model",
                         )
    parser.add_argument("--analytical_Jac",
                         action="store_true",
                         default=False,
                         help="Use an analytical Jacobian from the adapted Jaron, Nothnagel (2019) delay model instead of a numerical approximation. "\
                                 "NB: this can be used independent of --analytical_delay. You should really use this unless you have reason not to.",
                         )
    parser.add_argument("--dither_phase",
                         action="store_true",
                         default=False,
                         help="Add randomized integer wavelength offsets to toy phase data to replicate VLBI data",
                         )
    parser.add_argument("--trop_T",
                         type=float,
                         default=None,
                         help="Temperature at the sites, degrees C.",
                         )
    parser.add_argument("--trop_P",
                         type=float,
                         default=None,
                         help="Pressure at the sites, millibar.",
                         )
    parser.add_argument("--trop_H",
                         type=float,
                         default=None,
                         help="Relative humidity at the sites.",
                         )
    parser.add_argument("--antenna_name", 
                        dest="antenna_names",
                        help="Name of antennas in RINEX files. Order of file input. Should match loading rinex files.", 
                        default=[], 
                        action='append')
    parser.add_argument("--linked_clock", 
                        dest="linked_clocks",
                        help="Specify pairs of antennas that have feeds from the same clock as --linked_clocks \"ANT1--ANT2\" . Important for stochastic clock modeling.", 
                        default=[], 
                        action='append')
    parser.add_argument("--sta_code", 
                        dest="sta_codes",
                        help="Length 4 alphanumeric station code for SINEX. Order of file input. Should match antenna_name order. (Optional)", 
                        default=[], 
                        action='append')
    parser.add_argument("--domes_name", 
                        dest="domes_names",
                        help="Length 9 alphanumeric DOMES ID for SINEX. Order of file input. Should match antenna_name order. (Optional)", 
                        default=[], 
                        action='append')
    parser.add_argument("--antenna_type", 
                        dest="antenna_types",
                        help="Antenna types of antennas in data files (e.g. GNSS, XY-N, XY-E, Az-El, Equa, BWG, Nasmyth). Order of file input."+\
                              " If GNSS, specify ANTEX type in place of GNSS.", 
                        default=[], 
                        action='append')
    parser.add_argument("--iono_free", action="store_true", default=False, help = 'Use ionosphere free combination in '\
            +'GNSS antenna to compensate in VLBI. CURRENTLY NOT FULLY IMPLEMENTED.')
    parser.add_argument("--ionex_files", dest="ionex_files", action="append", type=str, nargs="+", help = 'Compensate for ionosphere with an IONEX model. Repeat for multiple days.')
    parser.add_argument("--iono_freq", type=str, default='L2',
                       help = 'Carrier frequency to use in iono-free combination (L2 or L5). NB: will only use GPS satellites with L2.')
   
def define_amb_state_vlbi(store_handle, baseline_handles):
    """ Detect cycle slips in carrier phase measurements. This will determine the number of float ambiguity states
        Get rough float ambiguity estimate from PR solution rxpos/clock
    """
    amb_state = np.array([]) 
    amb_int_total = np.array([])
    for jdx, baseline_handle in enumerate(baseline_handles): # generate differential measurements on the baselines
        source_array = [store_handle.source_time_dict[time] for time in baseline_handle.datetime_array]
        times_diff = (baseline_handle.datetime_array-baseline_handle.datetime_array[0])/np.timedelta64(1, 's')         
        if store_handle.iono_free: # get float ambiguity for combination
            phase_delays = baseline_handle.combination_model(baseline_handle.phase_delays, \
                               baseline_hande.phase_delays_dual, combination_type) 
            phase_delay_model = baseline_handle.phase_delay_model
            phase_delay_dual_model = baseline_handle.phase_delay_dual_model  
            phase_delay_model = baseline_handle.combination_model(phase_delay_model, phase_delay_dual_model, combination_type)
            slips, m_Bw_arr, threshod_arr = slip_detect_MW(baseline_handle.f1, baseline_handle.f2, baseline_handle.phase_delays, \
                    baseline_handle.phase_delays_dual, baseline_handle.group_delays, baseline_handle.group_delays_dual, times_diff, source_array, plot=True)
            N_float = construct_float_amb(phase_delays, phase_delay_model, baseline_handle.comb_wavelength)
        else:
            amb_int = np.rint((baseline_handle.phase_delay_model-baseline_handle.phase_delays)/baseline_handle.wavelength)
            amb_int_total = np.concatenate((amb_int_total, amb_int))
            baseline_handle.phase_delays = baseline_handle.phase_delays + amb_int * baseline_handle.wavelength
            #slips = slip_detect_phase_delay(baseline_handle.f1, baseline_handle.phase_delays, baseline_handle.phase_delay_model, \
            #         times_diff, source_array)
            slips = slip_detect_single_freq(baseline_handle.f1, baseline_handle.phase_delays, baseline_handle.phase_delay_model, \
                     times_diff, source_array)
            N_float = construct_float_amb(baseline_handle.phase_delays, \
                        baseline_handle.phase_delay_model, baseline_handle.wavelength)
        baseline_handle.save_phase_slips(slips)

        # initialize ambiguity state
        n_amb_state = 0
        len_N_good_idx = []
        slices_arr = []
        for src in np.unique(source_array): # get slips for each source data series
            idxs_src = np.ndarray.flatten(np.argwhere(np.array(source_array) == src))
            # find which slips are in the data of this source, get their locations(indices) in idxs_src
            slips_src, _, slips_src_idxs = np.intersect1d(np.array(baseline_handle.slips), idxs_src, return_indices=True)
            if len(slips_src)>0:
                for jdx in range(len(slips_src)+1):
                    if jdx == 0:
                        slice_idxs = idxs_src[:slips_src_idxs[0]]
                    elif jdx == len(slips_src):
                        slice_idxs = idxs_src[slips_src_idxs[jdx-1]:]
                    else:
                        slice_idxs = idxs_src[slips_src_idxs[jdx-1]:slips_src_idxs[jdx]]
                    N_idxs = N_float[slice_idxs]
                    N_good = N_idxs[~np.isnan(N_idxs)]
                    if len(N_good) >= MIN_SLICE and len(N_good) > 0:
                        N_interval = np.mean(N_good)
                        amb_state = np.append(amb_state, N_interval)
                        n_amb_state += 1
                    slices_arr.append(slice_idxs)
                    len_N_good_idx.append(len(N_good))
            else:
                N_idxs = N_float[idxs_src]
                N_good = N_idxs[~np.isnan(N_idxs)]
                if len(N_good) >= MIN_SLICE and len(N_good) > 0:
                    N_interval = np.mean(N_good)
                    amb_state = np.append(amb_state, N_interval)
                    n_amb_state += 1
                slices_arr.append(idxs_src)
                len_N_good_idx.append(len(N_good))
        baseline_handle.hold_slip_slices(slices_arr, n_amb_state)
        baseline_handle.trim_phase_idxs(len_N_good_idx)
        baseline_handle.phase_data_idxs = baseline_handle.range_data_idxs

    return amb_state, amb_int_total

def define_amb_state_gnss(store_handle, baselines, antenna_handles, baseline_handles):
    """ Define the ambiguity state for GNSS solution type """
    # Get rough float ambiguity estimate from PR solution rxpos/clock
    amb_state = np.array([])
    amb_int_total = np.array([])
    for jdx, baseline in enumerate(baselines):
        baseline_handle = baseline_handles[jdx]
        antenna1_handle = antenna_handles[baseline[0]]
        antenna2_handle = antenna_handles[baseline[1]]
        wavelength = baseline_handle.wavelength
        slices_arr = []
        baseline_handle.get_phase_idxs(store_handle.iono_free)
        source_array = [store_handle.source_time_dict[time] for time in baseline_handle.datetime_array]
        _, ant1_idxs, _ = np.intersect1d(antenna1_handle.times_gps, baseline_handle.datetime_array, return_indices=True)
        _, ant2_idxs, _ = np.intersect1d(antenna2_handle.times_gps, baseline_handle.datetime_array, return_indices=True)
        times_diff = (baseline_handle.datetime_array-baseline_handle.datetime_array[0])/np.timedelta64(1, 's')

        diff_cp_model = antenna2_handle.antenna_data.cp_model.values[ant2_idxs] - antenna1_handle.antenna_data.cp_model.values[ant1_idxs]
        if store_handle.iono_comp_l4r and not store_handle.iono_free:
            ALPHA_IONO=1.345e9
            # compensate for ionosphere when getting first guess for ambiguities
            source_array = [store_handle.source_time_dict[time] for time in baseline_handle.datetime_array]
            stec_vals = store_handle.interp_l4r(baseline_handle.datetime_array, source_array, antenna1_handle.l4r_name, antenna2_handle.l4r_name)
            diff_cp_model += ALPHA_IONO/baseline_handle.f1**2*const.c*stec_vals 
        
        N_int = np.rint((diff_cp_model-baseline_handle.cp_diff)/wavelength)
        baseline_handle.cp_diff += N_int*wavelength

        if store_handle.iono_free: # get float ambiguity for combination
            diff_cp_dual_model = data_ant2.cp_dual_model.values[ant2_idxs] - data_ant1.cp_dual_model.values
            wavelength2 = const.c/baseline_handle.f2
            N_int2 = np.rint((diff_cp_dual_model-baseline_handle.cp_dual)/wavelength2)
            baseline_handle.cp_dual += N_int2*wavelength2
            N_float = construct_float_amb(baseline_handle.cp_combination, \
                        baseline_handle.cp_model_combination, baseline_handle.comb_wavelength)
        else:
            N_float = construct_float_amb(baseline_handle.cp_diff, diff_cp_model, wavelength)

        #amb_int = N_int[baseline_handle.range_data_idxs]
        amb_int = N_int
        amb_int_total = np.concatenate((amb_int_total, amb_int))
    
        if store_handle.iono_free:
            # detect slips
            slips, m_Bw_arr, threshod_arr = slip_detect_MW(baseline_handle.f1, baseline_handle.f2, baseline_handle.cp_diff, baseline_handle.cp_dual,\
                           baseline_handle.pr_diff, baseline_handle.pr_dual, times_diff, source_array, plot=True)
        else: # need to detect slips with single-frequency
            slips = slip_detect_single_freq(baseline_handle.f1, baseline_handle.cp_diff, diff_cp_model, \
                     times_diff, source_array)

        baseline_handle.save_phase_slips(slips)
        if store_handle.iono_free: 
            baseline_handle.combination_measurement('WL')
            baseline_handle.combination_model(diff_cp_model, diff_cp_dual_model, 'WL')

        n_amb_state = 0
        len_N_good_idx = []
        for sat in np.unique(source_array): # get slips for each satellite data series
            idxs_sat = np.ndarray.flatten(np.argwhere(np.array(source_array) == sat))
            # find which slips are in the data of this satellite, get their locations(indices) in idxs_sat
            slips_sat, _, slips_sat_idxs = np.intersect1d(np.array(baseline_handle.slips), idxs_sat, return_indices=True)
            if len(slips_sat)>0:
                for jdx in range(len(slips_sat)+1):
                    if jdx == 0:
                        slice_idxs = idxs_sat[:slips_sat_idxs[0]]
                    elif jdx == len(slips_sat):
                        slice_idxs = idxs_sat[slips_sat_idxs[jdx-1]:]
                    else:
                        slice_idxs = idxs_sat[slips_sat_idxs[jdx-1]:slips_sat_idxs[jdx]]
                    N_idxs = N_float[slice_idxs]
                    N_good = N_idxs[~np.isnan(N_idxs)]
                    if len(N_good) >= MIN_SLICE:
                        N_interval = np.mean(N_good)
                        amb_state = np.append(amb_state, N_interval)
                        n_amb_state += 1
                    slices_arr.append(slice_idxs)
                    len_N_good_idx.append(len(N_good))
            else:
                N_idxs = N_float[idxs_sat]
                N_good = N_idxs[~np.isnan(N_idxs)]
                if len(N_good) >= MIN_SLICE:
                    N_interval = np.mean(N_good)
                    amb_state = np.append(amb_state, N_interval)
                    n_amb_state += 1
                slices_arr.append(idxs_sat)
                len_N_good_idx.append(len(N_good))

        baseline_handle.hold_slip_slices(slices_arr, n_amb_state)
        baseline_handle.trim_phase_idxs(len_N_good_idx)
        baseline_handle.phase_data_idxs = baseline_handle.range_data_idxs

    return amb_state, amb_int_total

def redefine_amb_state(sol_type, store_handle, baseline_handles, baselines, antenna_handles):
    """ Detect cycle slips in carrier phase measurements without differentiating by source.
        Redefine the ambiguity states
    """
    amb_state = np.array([]) 
    for jdx, baseline_handle in enumerate(baseline_handles): # generate differential measurements on the baselines
        if sol_type == 'GNSS':
            if store_handle.iono_free: # get float ambiguity for combination
                N_float = construct_float_amb(baseline_handle.cp_combination, \
                            baseline_handle.cp_model_combination, baseline_handle.comb_wavelength)
            else:
                baseline = baselines[jdx]
                antenna1_handle = antenna_handles[baseline[0]]
                antenna2_handle = antenna_handles[baseline[1]]
                _, ant1_idxs, _ = np.intersect1d(antenna1_handle.times_gps, baseline_handle.datetime_array, return_indices=True)
                _, ant2_idxs, _ = np.intersect1d(antenna2_handle.times_gps, baseline_handle.datetime_array, return_indices=True)
                diff_cp_model = antenna2_handle.antenna_data.cp_model.values[ant2_idxs] - antenna1_handle.antenna_data.cp_model.values[ant1_idxs]
                N_float = construct_float_amb(baseline_handle.cp_diff, \
                            diff_cp_model, const.c/baseline_handle.f1)
        elif sol_type == 'VLBI':
            if store_handle.iono_free: # get float ambiguity for combination
                phase_delays = baseline_handle.combination_model(baseline_handle.phase_delays, \
                                   baseline_hande.phase_delays_dual, combination_type) 
                phase_delay_model = baseline_handle.phase_delay_model
                phase_delay_dual_model = baseline_handle.phase_delay_dual_model  
                phase_delay_model = baseline_handle.combination_model(phase_delay_model, phase_delay_dual_model, combination_type)
                N_float = construct_float_amb(phase_delays, phase_delay_model, baseline_handle.comb_wavelength)
            else:
                N_float = construct_float_amb(baseline_handle.phase_delays, \
                            baseline_handle.phase_delay_model, baseline_handle.wavelength)

        # initialize ambiguity state
        n_amb_state = 0
        slices_arr = []
        idxs_full = np.ndarray.flatten(np.argwhere(np.ones(len(baseline_handle.datetime_array), dtype=bool)))
        for jdx in range(len(baseline_handle.slips)+1):
            if len(baseline_handle.slips)>0:
                if jdx == 0:
                    slice_idxs = idxs_full[:baseline_handle.slips[0]]
                elif jdx == len(baseline_handle.slips):
                    slice_idxs = idxs_full[baseline_handle.slips[jdx-1]:]
                else:
                    slice_idxs = idxs_full[baseline_handle.slips[jdx-1]:baseline_handle.slips[jdx]]
            else:
                slice_idxs = idxs_full
            N_idxs = N_float[slice_idxs]
            N_good = N_idxs[~np.isnan(N_idxs)]
            if len(N_good) >= MIN_SLICE and len(N_good) > 0:
                N_interval = np.mean(N_good)
                amb_state = np.append(amb_state, N_interval)
                n_amb_state += 1
            slices_arr.append(slice_idxs)
        baseline_handle.hold_slip_slices(slices_arr, n_amb_state)
        #baseline_handle.trim_phase_idxs(len_N_good_idx)

    return amb_state

def define_baseline_handles_gnss(store_handle, clock_state, clock_poly_length, antenna_handles, ref_antenna, baselines):
    """ Define the baseline handle constructs for GNSS solution type """
    baseline_handles = []
    f1 = 1575.42*1e6
    wavelength1 = const.c/f1
    for jdx, baseline in enumerate(baselines): # generate differential measurements on the baselines
       antenna1_handle = antenna_handles[baseline[0]]
       antenna2_handle = antenna_handles[baseline[1]]

       data_ant1 = antenna1_handle.antenna_data
       data_ant2 = antenna2_handle.antenna_data
       if store_handle.iono_comp_l4r:
           ds_baseline = store_handle.sel_l4r_baseline(antenna1_handle.l4r_name, antenna2_handle.l4r_name)
           svs = data_ant1.sv.values
           idxs_interpable = store_handle.find_l4r_interpable(antenna1_handle.times_gps, svs, antenna1_handle.l4r_name, antenna2_handle.l4r_name)
           times_good = antenna1_handle.times_gps[idxs_interpable]
           times_gps, ant2_idxs, _ = np.intersect1d(antenna2_handle.times_gps, times_good, return_indices=True)
           _, ant1_idxs, _ = np.intersect1d(antenna1_handle.times_gps, times_gps, return_indices=True)
       else:
           times_gps, ant1_idxs, ant2_idxs = np.intersect1d(antenna1_handle.times_gps, antenna2_handle.times_gps, return_indices=True)

       # apparently, there are some situations with pseudorange but no carrier phase data, e.g. NLIB 10/27/25 E12 @ 15:29:00
       # get rid of these data
       idxs_nonnan = np.bitwise_and(~np.isnan(data_ant1.cp_data[ant1_idxs]),~np.isnan(data_ant2.cp_data[ant2_idxs]))
       #ant1_idxs = ant1_idxs[idxs_nonnan]
       #ant2_idxs = ant2_idxs[idxs_nonnan]
       #times_gps = times_gps[idxs_nonnan]
       
       source_array = [store_handle.source_time_dict[time] for time in times_gps]
       diff_cp_data = data_ant2.cp_data.values[ant2_idxs] - data_ant1.cp_data.values[ant1_idxs]
       diff_pr_data = data_ant2.pr_data.values[ant2_idxs] - data_ant1.pr_data.values[ant1_idxs]
 
       if store_handle.iono_free:
           if store_handle.iono_freq == 'L2':
               f2 = 1227.60*1e6
               diff_pr_dual = data_ant2.P2.values[ant2_idxs] - data_ant1.P2.values[ant1_idxs]
    
           elif store_handle.iono_freq == 'L5':
               f2 = 1176.45*1e6
               diff_pr_dual = data_ant2.C5.values[ant2_idxs] - data_ant1.C5.values[ant1_idxs]
           diff_cp_dual_model = data_ant2.cp_dual_model.values[ant2_idxs] - data_ant1.cp_dual_model.values
           diff_cp_dual = data_ant2.cp_dual.values[ant2_idxs] - data_ant1.cp_dual.values[ant1_idxs]
           wavelength2 = const.c/f2
           N_int = np.rint((diff_cp_dual_model-diff_cp_dual)/wavelength2)
           diff_cp_dual = diff_cp_dual + N_int*wavelength2

       # initialize a baseline object to hold differential phase info
       baseline_handle = BaselineInfo(times_gps, f1)
       baseline_handle.prepare_l1_frequency(diff_pr_data, diff_cp_data)
       baseline_handle.hold_range_idxs(np.arange(len(diff_pr_data))[idxs_nonnan])
       baseline_handle.get_phase_idxs(store_handle.iono_free)
       if store_handle.iono_free: 
           diff_cp_model = data_ant2.cp_model.values[ant2_idxs] - data_ant1.cp_model.values[ant1_idxs]
           baseline_handle.prepare_dual_frequency(diff_cp_dual, f2)
           baseline_handle.combination_measurement('WL')
           baseline_handle.combination_model(diff_cp_model, diff_cp_dual_model, 'WL')
       baseline_handles.append(baseline_handle)

    return baseline_handles

def update_baseline_handles_vlbi(store_handle, antenna_handles, baseline_handles, baselines):
    """ Use only dSTEC-defined data for L4R long baseline solution """
    for jdx, baseline in enumerate(baselines): # generate differential measurements on the baselines
       antenna1_handle = antenna_handles[baseline[0]]
       antenna2_handle = antenna_handles[baseline[1]]
       baseline_handle = baseline_handles[jdx]

       # select only data points covered by l4r
       source_array = [store_handle.source_time_dict[time] for time in baseline_handle.datetime_array]
       bl_idxs = store_handle.find_l4r_interpable(baseline_handle.datetime_array, source_array, antenna1_handle.l4r_name, antenna2_handle.l4r_name)
 
       baseline_handle.group_delays = baseline_handle.group_delays[bl_idxs]
       baseline_handle.phase_delays = baseline_handle.phase_delays[bl_idxs]
       baseline_handle.datetime_array = baseline_handle.datetime_array[bl_idxs]
       _, _, baseline_handle.range_data_idxs = np.intersect1d(baseline_handle.range_data_idxs, bl_idxs, return_indices=True)
       _, _, baseline_handle.phase_data_idxs = np.intersect1d(baseline_handle.phase_data_idxs, bl_idxs, return_indices=True)
       if store_handle.iono_free:
            baseline_handle.group_delays_dual = baseline_handle.group_delays_dual[bl_idxs]
            baseline_handle.phase_delays_dual = baseline_handle.phase_delays_dual[bl_idxs]

    return baseline_handles

def find_clock_trop_params(sol_type, antenna_handles, store_handle, trop_poly_length, clock_poly_length, ref_antenna):
    """ Find the number of clock and troposphere state elements based on the input arguments """
    if (store_handle.stochastic_clock or store_handle.stochastic_trop) and sol_type=='GNSS':
        # figure out which clock epochs will actually be state elements -- there must be intersecting data at two antennas
        for idx, antenna_handle in enumerate(antenna_handles):
            if ref_antenna == antenna_handle.antenna_name:
                ref_handle = antenna_handle
            times_gps_antenna = antenna_handle.times_gps
            for jdx, antenna_handle_comp in enumerate(antenna_handles):
                if idx == jdx: continue
                if jdx == 0 or (idx == 0 and jdx == 1):
                    intersect_times = np.intersect1d(antenna_handle.times_gps, antenna_handle_comp.times_gps)
                else:
                    intersect_times = np.union1d(intersect_times, np.intersect1d(antenna_handle.times_gps, antenna_handle_comp.times_gps))

            if store_handle.stochastic_clock:
                antenna_handle.clock_times = intersect_times
                antenna_handle.phase_clock_times = intersect_times
            if store_handle.stochastic_trop:
                antenna_handle.trop_times = intersect_times

    elif (store_handle.stochastic_clock or store_handle.stochastic_trop) and sol_type=='VLBI':
        # for a VLBI-style solution, we know that we will have data on a baseline for each time in times_gps
        # just define the clock_times and trop_times arrays
        for idx, antenna_handle in enumerate(antenna_handles):
            if ref_antenna == antenna_handle.antenna_name:
                ref_handle = antenna_handle
            if store_handle.stochastic_clock:
                antenna_handle.clock_times = antenna_handle.times_gps
            if store_handle.stochastic_trop: 
                antenna_handle.trop_times = antenna_handle.times_gps
    else:
        for idx, antenna_handle in enumerate(antenna_handles):
            if ref_antenna == antenna_handle.antenna_name:
                ref_handle = antenna_handle

    n_clock_tot = 0
    n_trop_tot = 0
    for antenna_handle in antenna_handles:
        if ref_antenna != antenna_handle.antenna_name:
            # number of clock polynomials is len(antenna_handles)-1, n_clock + 1 clock parameters per polynomial
            exp_length = (antenna_handle.times_gps[-1]-antenna_handle.times_gps[0])/np.timedelta64(1, 's')
            if (store_handle.global_linear_clock or store_handle.global_quadratic_clock) and\
                    exp_length < clock_poly_length:
                raise ValueError('Cannot have two global clock models (clock_poly_length > experiment length and global clock model activated)')
            elif store_handle.global_linear_clock is False and store_handle.global_quadratic_clock is False and\
                    clock_poly_length == 0 and store_handle.stochastic_clock is False:
                raise ValueError('Must have a clock model')
            elif store_handle.global_linear_clock and store_handle.global_quadratic_clock:
                raise ValueError('Cannot have two global clock models (both global linear and global quadratic clock model activated)')
            if store_handle.stochastic_clock is False and clock_poly_length >0:
                nclock = 1+int(np.ceil(exp_length/clock_poly_length)) # number of piecewise-linear intervals
            elif store_handle.stochastic_clock:
                nclock = len(antenna_handle.clock_times)
            else:
                # only global clock model
                nclock = 1
            if store_handle.global_linear_clock:
                nclock += 1
            elif store_handle.global_quadratic_clock:
                nclock += 2
            n_clock_tot += nclock

            if store_handle.stochastic_trop is False and trop_poly_length > 0:
                ntrop = 1+int(np.ceil(exp_length/trop_poly_length)) # number of piecewise-linear intervals
            elif store_handle.stochastic_trop:
                ntrop = len(antenna_handle.trop_times)
            else:
                ntrop = 0
            if np.linalg.norm(np.array(ref_handle.ref_pos)-np.array(antenna_handle.ref_pos))< MIN_TROP_DIST:
                # override -- distance to reference too small
                ntrop = 0
            n_trop_tot += ntrop

    return n_clock_tot, n_trop_tot

def assimilate_amb_state(sol_type, a_fixed, baseline_handles, iono_free):
    """ Add the previous ambiguity fix to the phase measurements before defining a new one """
    a_fixed = np.rint(a_fixed) # ensure all integers
    n_amb = 0
    for jdx, baseline_handle in enumerate(baseline_handles): # generate differential measurements on the baselines
        amb_baseline = a_fixed[n_amb:n_amb+baseline_handle.n_amb_state]
        n_amb += baseline_handle.n_amb_state
        # get data and model carrier phase observations
        if iono_free:
            wavelength = baseline_handle.comb_wavelength
        else:
            wavelength = baseline_handle.wavelength

        # subtract float or integer ambiguities from data
        slip_slices_arr = baseline_handle.slip_slices_arr
        for idx, slip_slice in enumerate(slip_slices_arr):
            if sol_type == 'VLBI':
                baseline_handle.phase_delays[slip_slice] = baseline_handle.phase_delays[slip_slice] + wavelength*amb_baseline[idx]
            elif sol_type == 'GNSS':
                if iono_free:
                    baseline_handle.cp_combination[slip_slice] = baseline_handle.cp_combination[slip_slice] + wavelength*amb_baseline[idx]
                else:
                    baseline_handle.cp_diff[slip_slice] = baseline_handle.cp_diff[slip_slice] + wavelength*amb_baseline[idx]

def update_measurements_vlbi(store_handle, src_type, baselines, baseline_handles, antenna_handles):
    """ Update the VLBI measurement model for a new state definition """
    for jdx, baseline in enumerate(baselines):
        baseline_handle = baseline_handles[jdx]
        antenna1_handle = antenna_handles[baseline[0]]
        antenna2_handle = antenna_handles[baseline[1]]
        _, ant1_idxs, _ = np.intersect1d(antenna1_handle.times_gps, \
                baseline_handle.datetime_array, return_indices=True)
        _, ant2_idxs, _ = np.intersect1d(antenna2_handle.times_gps, \
                baseline_handle.datetime_array, return_indices=True)

        # get clock samples
        clock_samples = antenna2_handle.clock_samples[ant2_idxs] - \
                    antenna1_handle.clock_samples[ant1_idxs]
        phase_clock_samples = antenna2_handle.clock_samples[ant2_idxs] - \
                antenna1_handle.clock_samples[ant1_idxs]

        if store_handle.estimate_trop:
            trop_samples1 = antenna1_handle.trop_samples[ant1_idxs]
            trop_samples2 = antenna2_handle.trop_samples[ant2_idxs]
        else:
            trop_samples1 = []
            trop_samples2 = []

        if store_handle.src_type == 'GNSS':
            store_handle.model_group_phase_vlbi(antenna1_handle, antenna2_handle, baseline_handle, \
                                     baseline_handle.datetime_array, clock_samples, trop_samples1, trop_samples2,\
                                     True, phase_clock_samples)
        elif store_handle.src_type == 'VLBI':
            store_handle.model_group_phase_farfield(antenna1_handle, antenna2_handle, baseline_handle, \
                                     baseline_handle.datetime_array, clock_samples, trop_samples1, trop_samples2,\
                                     True, phase_clock_samples)

def lstsq_estimation(sol_type, plot_intermediate_results, ref_antenna, store_handle, antenna_handles, baselines, baseline_handles, \
                     clock_poly_length, trop_poly_length, clock_file=None, estimate_AO=False, \
                     analytical_Jac=False, tikhonov_reg=False, cont_reg=False, recursive_amb=False, do_mcmc_correlation=False,
                     do_ls_vce_correlation=False, covariance_kernel_range=None, covariance_kernel_phase=None, igs_data=False, baseline_strategy='Obs-Max', band=None):
    """
    Take the single-source data and produce a differential position estimate via least-squares adjustment
    """
    #VLBI_like=False
    VLBI_like=True

    n_clock_tot, n_trop_tot = find_clock_trop_params(sol_type, antenna_handles, store_handle, trop_poly_length, clock_poly_length, ref_antenna)
    if VLBI_like and sol_type == 'GNSS':
        # we want all of the times included in correct_PR_CP so that they can be included in the 
        # data reduction model in vlbi_transform_data
        n_clock_tot = 0
        for antenna_handle in antenna_handles:
            antenna_handle.clock_times = antenna_handle.times_gps
            antenna_handle.phase_clock_times = antenna_handle.times_gps
            n_clock_tot += len(antenna_handle.clock_times)

    # set up first rxpos series
    for antenna_handle in antenna_handles:
        rxpos_series, R_obj  = store_handle.compute_tides(antenna_handle.times_gps, antenna_handle.ref_pos, antenna_handle.antenna_name) 
        antenna_handle.update_pos_series(rxpos_series, R_obj, antenna_handle.ref_pos)
        store_handle.compute_azel(antenna_handle.times_gps, antenna_handle)

    # (len(antenna_handles)-1)*3 position parameters (XYZ for all but reference station)
    n_ao_state = 0
    n_grav_state = 0
    n_grav_state_hold = 0
    HOLD_GRAV = True # only turn on gravity deformation estimation after 
    grav_antennas = []
    antenna_names = []
    for idx, antenna_handle in enumerate(antenna_handles):
        antenna_names.append(antenna_handle.antenna_name)
        if ref_antenna == antenna_handle.antenna_name:
            ref_handle = antenna_handle
        if antenna_handle.estimate_ao:
            n_ao_state += 1
        if antenna_handle.estimate_grav_def:
            if HOLD_GRAV:
                n_grav_state = 0
                n_grav_state_hold += 2
                grav_antennas.append(antenna_handle.antenna_name)
                antenna_handle.estimate_grav_def = False
            else:
                n_grav_state += 2

    n_rxpos = (len(antenna_handles)-1)*3 
    if store_handle.estimate_disb:
        n_disb_tot = (len(store_handle.systems)-1)*(len(antenna_handles)-1) # range DISB states
    else:
        n_disb_tot = 0

    if sol_type == 'VLBI' and store_handle.iono_comp_l4r:
        # use only measurements in l4r dataset
        baseline_handles = update_baseline_handles_vlbi(store_handle, antenna_handles, baseline_handles, baselines)

    nparams = n_rxpos + n_clock_tot + n_trop_tot + n_ao_state + n_grav_state + n_disb_tot
    state = np.zeros(int(nparams)) # positions + clocks    
 
    clock_idxs = slice(n_rxpos,n_rxpos+n_clock_tot) # clock polynomials are stored after positions
    trop_idxs = slice(n_rxpos+n_clock_tot+n_ao_state+n_grav_state,n_rxpos+n_clock_tot+n_ao_state+n_grav_state+n_trop_tot)
    disb_idxs = slice(n_rxpos+n_clock_tot+n_ao_state+n_grav_state+n_trop_tot,n_rxpos+n_clock_tot+n_ao_state+n_grav_state+n_trop_tot+n_disb_tot)
    count = 0
    n_clock_loop = 0
    n_trop_loop = 0
    n_disb_loop = 0
    n_disb_loop_phase = 0
    for idx, antenna_handle in enumerate(antenna_handles):
        if ref_antenna != antenna_handle.antenna_name: # compute corrected ranges once
            exp_length = (antenna_handle.times_gps[-1]-antenna_handle.times_gps[0])/np.timedelta64(1, 's')
            if store_handle.stochastic_clock is False and clock_poly_length > 0:
                nclock = 1+int(np.ceil(exp_length/clock_poly_length)) # number of piecewise-linear intervals
            elif store_handle.stochastic_clock:
                nclock = len(antenna_handle.clock_times)
            else:
                nclock = 1
            if store_handle.global_linear_clock:
                nclock += 1
                idx_start = 1 
            elif store_handle.global_quadratic_clock:
                nclock += 2
                idx_start = 2
            else:
                idx_start = 0
            rxpos = antenna_handle.ref_pos
            state[count*3:count*3+3] = rxpos
            try: state[n_rxpos+n_clock_loop+idx_start] = antenna_handle.bulk_clock
            except: breakpoint()

            if store_handle.stochastic_clock is False:
                antenna_handle.hold_range_clock_params(slice(n_clock_loop,n_clock_loop + nclock), antenna_handle.times_gps)
            else:
                antenna_handle.hold_range_clock_params(slice(n_clock_loop,n_clock_loop + nclock), antenna_handle.clock_times)
            # only add troposphere states if they can be estimated (>MIN_TROP_DIST from reference antenna)
            if store_handle.estimate_trop:
                antenna_handle.estimate_trop = True
                if np.linalg.norm(np.array(antenna_handle.ref_pos) - np.array(ref_handle.ref_pos)) < MIN_TROP_DIST:
                    print('Troposphere parameters for antenna ' + antenna_handle.antenna_name +' disabled '\
                            +'(too close to reference antenna ' + ref_handle.antenna_name + ')' )
                    antenna_handle.estimate_trop = False
                    ntrop = 0
                    antenna_handle.hold_trop_params(slice(n_trop_loop, n_trop_loop), antenna_handle.times_gps)
                elif store_handle.stochastic_trop is False:
                    ntrop = 1+int(np.ceil(exp_length/trop_poly_length)) # number of piecewise-linear intervals
                    antenna_handle.hold_trop_params(slice(n_trop_loop, n_trop_loop + ntrop), antenna_handle.times_gps)
                else:
                    ntrop = len(antenna_handle.trop_times)
                    antenna_handle.hold_trop_params(slice(n_trop_loop, n_trop_loop + ntrop), antenna_handle.trop_times)
                #state[n_rxpos+n_clock_tot+n_ao_state+n_grav_state+n_trop_loop] = 0
            else:
                ntrop = 0

            if store_handle.estimate_disb:
                # no need to set a value in state, these are initialized as 0
                ndisb = len(store_handle.disb_systems)
                antenna_handle.hold_range_disb_params(slice(n_disb_loop, n_disb_loop + ndisb))
                n_disb_loop += ndisb
            else:
                ndisb = 0
            if store_handle.estimate_phase_disb:
                antenna_handle.hold_phase_disb_params(slice(n_disb_loop_phase, n_disb_loop_phase + ndisb))
                n_disb_loop_phase += ndisb

            n_trop_loop += ntrop
            n_clock_loop += nclock
            count += 1 
        else: 
            ref_idx = idx
            antenna_handle.clock_times = antenna_handle.times_gps
        
        # compute corrected ranges
        if ref_antenna == antenna_handle.antenna_name and clock_file is not None:
            # load clock and build interpolator function
            ppp_data = np.loadtxt(clock_file)
            time_sec_of_day = ppp_data[:,0]
            cb_data_m = ppp_data[:,2]
            interp_fcn = make_interp_spline(time_sec_of_day, cb_data_m)
 
            # correct reference data for PPP clock
            sec_of_day_data = (antenna_handle.times_gps - antenna_handle.times_gps.astype('datetime64[D]')).astype('timedelta64[s]').astype(float)
            clock_samples = interp_fcn(sec_of_day_data)
        else: 
            if store_handle.stochastic_clock is False:
                clock_samples = np.zeros(len(antenna_handle.times_gps)) + antenna_handle.bulk_clock
            else:
                clock_samples = np.zeros(len(antenna_handle.clock_times)) + antenna_handle.bulk_clock

        if antenna_handle.ppp_clock_active:
            if store_handle.stochastic_clock:
                antenna_handle.interp_ppp_clock(antenna_handle.clock_times)
            clock_samples += antenna_handle.ppp_clock_samples
            #if store_handle.stochastic_clock is False:
            #    antenna_handle.clock_times = antenna_handle.times_gps
            #    antenna_handle.phase_clock_times = antenna_handle.times_gps
            #if store_handle.stochastic_trop is False:
            #    antenna_handle.trop_times = antenna_handle.times_gps
        
        antenna_handle.hold_clock(clock_samples)
        antenna_handle.hold_clock(clock_samples.copy(), phase_delay=True) # ref clock is also ref phase clock
        if store_handle.stochastic_clock and ref_antenna == antenna_handle.antenna_name:
            antenna_handle.phase_clock_times = antenna_handle.clock_times

        if store_handle.estimate_trop:
            if store_handle.stochastic_trop is False:
                trop_samples = np.zeros(len(antenna_handle.times_gps))
            else:
                trop_samples = np.zeros(len(antenna_handle.trop_times))
            antenna_handle.hold_trop(trop_samples)
        
        if sol_type == 'GNSS':
            antenna_handle.get_pr_data(store_handle.iono_free, store_handle.iono_freq)
            antenna_handle.get_cp_data(store_handle.iono_free, store_handle.iono_freq)
            data_corrected = store_handle.correct_PR_CP(antenna_handle, phase=True)
            antenna_handle.hold_data(data_corrected)



    if sol_type == 'GNSS' and VLBI_like:
        # convert data to 1 meas per scan by antenna
        vlbi_transform_data(store_handle, antenna_handles, igs_data)
        n_clock_tot, n_trop_tot = find_clock_trop_params(sol_type, antenna_handles, store_handle, trop_poly_length, clock_poly_length, ref_antenna)
        nparams = n_rxpos + n_clock_tot + n_trop_tot + n_ao_state + n_grav_state + n_disb_tot
        state = np.zeros(int(nparams)) # positions + clocks    
        clock_idxs = slice(n_rxpos,n_rxpos+n_clock_tot) # clock polynomials are stored after positions
        trop_idxs = slice(n_rxpos+n_clock_tot+n_ao_state+n_grav_state,n_rxpos+n_clock_tot+n_ao_state+n_grav_state+n_trop_tot)
        disb_idxs = slice(n_rxpos+n_clock_tot+n_ao_state+n_grav_state+n_trop_tot,n_rxpos+n_clock_tot+n_ao_state+n_grav_state+n_trop_tot+n_disb_tot)
        count = 0
        n_clock_loop = 0
        n_trop_loop = 0
        for idx, antenna_handle in enumerate(antenna_handles):
            if ref_antenna != antenna_handle.antenna_name: # compute corrected ranges once
                exp_length = (antenna_handle.times_gps[-1]-antenna_handle.times_gps[0])/np.timedelta64(1, 's')
                if store_handle.stochastic_clock is False and clock_poly_length > 0:
                    nclock = 1+int(np.ceil(exp_length/clock_poly_length)) # number of piecewise-linear intervals
                elif store_handle.stochastic_clock:
                    nclock = len(antenna_handle.clock_times)
                else:
                    nclock = 1
                if store_handle.global_linear_clock:
                    nclock += 1
                    idx_start = 1 
                elif store_handle.global_quadratic_clock:
                    nclock += 2
                    idx_start = 2
                else:
                    idx_start = 0
                rxpos = antenna_handle.ref_pos
                state[count*3:count*3+3] = rxpos
                #state[n_rxpos+n_clock_loop+idx_start] = antenna_handle.bulk_clock
                if store_handle.stochastic_clock:
                    state[n_rxpos+n_clock_loop+idx_start:n_rxpos+n_clock_loop+nclock] += antenna_handle.bulk_clock
                else:
                    state[n_rxpos+n_clock_loop+idx_start] = antenna_handle.bulk_clock
                antenna_handle.hold_range_clock_params(slice(n_clock_loop,n_clock_loop + nclock), antenna_handle.clock_times)
                if store_handle.estimate_trop and antenna_handle.estimate_trop:
                    if store_handle.stochastic_trop is False:
                        ntrop = 1+int(np.ceil(exp_length/trop_poly_length)) # number of piecewise-linear intervals
                        antenna_handle.hold_trop_params(slice(n_trop_loop, n_trop_loop + ntrop), antenna_handle.times_gps)
                    else:
                        ntrop = len(antenna_handle.trop_times)
                        antenna_handle.hold_trop_params(slice(n_trop_loop, n_trop_loop + ntrop), antenna_handle.trop_times)
                    state[n_rxpos+n_clock_tot+n_ao_state+n_grav_state+n_trop_loop] = 0
                else:
                    ntrop = 0
                n_trop_loop += ntrop
                n_clock_loop += nclock
                count += 1 
            else: 
                ref_idx = idx

    if sol_type == 'GNSS':
        # take only linearly independent baselines in GNSS data
        if baseline_strategy == 'Obs-Max':
            maximize=True
        else:
            maximize=False
        baselines_old = baselines
        baselines = get_spanning_tree(antenna_handles, maximize)

    count_ao = 0
    if estimate_AO:
        for idx, antenna_handle in enumerate(antenna_handles):
            if antenna_handle.is_VLBI:
                # add axis offset to state
                state[nparams-n_ao_state+count_ao] = antenna_handle.axis_offset
                count_ao += 1

    # gravitational deformation is initialized as zeros, so no need to set initial state

    if sol_type == 'VLBI':
        # initialize prefit data
        for jdx, baseline in enumerate(baselines):
            baseline_handle = baseline_handles[jdx]
            antenna1_handle = antenna_handles[baseline[0]]
            antenna2_handle = antenna_handles[baseline[1]]
            _, ant1_idxs, _ = np.intersect1d(antenna1_handle.times_gps, \
                    baseline_handle.datetime_array, return_indices=True)
            _, ant2_idxs, _ = np.intersect1d(antenna2_handle.times_gps, \
                    baseline_handle.datetime_array, return_indices=True)

            # load kernels if they are supplied
            if covariance_kernel_range is not None:
                kernels_range = load_kernel_parameters(covariance_kernel_range)
                for baseline_handle in baseline_handles:
                    baseline_handle.hold_covariance_kernels(kernels_range)

            if covariance_kernel_phase is not None:
                kernels_phase = load_kernel_parameters(covariance_kernel_phase)
                for baseline_handle in baseline_handles:
                    baseline_handle.hold_covariance_kernels(kernels_phase, True) 

            # get clock samples
            clock_samples = antenna2_handle.clock_samples[ant2_idxs] - \
                        antenna1_handle.clock_samples[ant1_idxs]
            phase_clock_samples = antenna2_handle.clock_samples[ant2_idxs] - \
                    antenna1_handle.clock_samples[ant1_idxs]
            if store_handle.estimate_trop:
                trop_samples1 = antenna1_handle.trop_samples[ant1_idxs]
                trop_samples2 = antenna2_handle.trop_samples[ant2_idxs]
            else:
                trop_samples1 = []
                trop_samples2 = []

            if store_handle.src_type == 'GNSS':
                store_handle.model_group_phase_vlbi(antenna1_handle, antenna2_handle, baseline_handle, \
                                         baseline_handle.datetime_array, clock_samples, trop_samples1, trop_samples2, \
                                         True, phase_clock_samples)
            elif store_handle.src_type == 'VLBI':
                store_handle.model_group_phase_farfield(antenna1_handle, antenna2_handle, baseline_handle, \
                                         baseline_handle.datetime_array, clock_samples, trop_samples1, trop_samples2, \
                                         True, phase_clock_samples)
    
    # bias the initial state a bit for testing
    #rx_bias = 20 # 10 cm
    #state[0:3] = state[0:3] + rx_bias 

    # bound the optimization
    pos_bound = np.inf
    rxpos_state_length = 3*(len(antenna_handles)-1) # number of state elements that are rx pos
    bound_low = np.zeros_like(state)
    bound_low[:rxpos_state_length] = state[:rxpos_state_length]-pos_bound
    bound_low[rxpos_state_length:] = bound_low[rxpos_state_length:] - np.inf
    
    bound_high = np.zeros_like(state)
    bound_high[rxpos_state_length:] = bound_high[rxpos_state_length:] + np.inf
    bound_high[:rxpos_state_length] = state[:rxpos_state_length]+pos_bound

    if analytical_Jac:
        jac = calc_jac
    else:
        jac = '2-point'
    
    # Do the pseudorange (group delay) solution 
    store_handle.hold_state(state+1e-9)
    print('Executing initial range-based solution:')
    ls_args = (ref_antenna, baselines, store_handle, antenna_handles, clock_idxs, clock_poly_length, \
                   trop_idxs, trop_poly_length, disb_idxs, baseline_handles)
    bounds = (bound_low, bound_high)
    ls_grdel = least_squares(calc_residuals, state, jac=jac, method='trf', x_scale='jac',\
            max_nfev=100, bounds=bounds, verbose=2, xtol=1e-15, args=ls_args)
    #ls_grdel_test = least_squares(calc_residuals, state, jac='2-point', method='trf', x_scale='jac',\
    #        max_nfev=100, bounds=bounds, verbose=2, xtol=1e-15, args=ls_args)

    # define new, expanded state -- add phase clock and carrier phase ambiguities
    grdel_state = ls_grdel.x
    end_range_state = len(grdel_state) # number of RXPOS + PR clock + trop parameters
    state_expanded = np.zeros(end_range_state)
    state_expanded[:end_range_state] = grdel_state
    clock_state = state_expanded[clock_idxs]
    use_phase_weights = False # dont use phase weights until we've fixed ambiguities -- they are horribly optimistic

    # initialize the phase model and define the ambiguity state
    if sol_type == 'GNSS':
        baseline_handles = define_baseline_handles_gnss(store_handle, clock_state, clock_poly_length, antenna_handles, ref_antenna, baselines)
        # load kernels if they are supplied
        if covariance_kernel_range is not None:
            kernels_range = load_kernel_parameters(covariance_kernel_range)
            for baseline_handle in baseline_handles:
                baseline_handle.hold_covariance_kernels(kernels_range)

        if covariance_kernel_phase is not None:
            kernels_phase = load_kernel_parameters(covariance_kernel_phase)
            for baseline_handle in baseline_handles:
                baseline_handle.hold_covariance_kernels(kernels_phase, True)

    if store_handle.stochastic_clock or store_handle.stochastic_trop:
        state_expanded, end_range_state, clock_idxs, trop_idxs, disb_idxs, _, _, _ = adjust_stoch_params(store_handle, antenna_handles, baseline_handles,\
                baselines, ref_antenna, state_expanded, clock_idxs, trop_idxs, disb_idxs, n_ao_state, n_grav_state)
    else:
        state_expanded = ls_grdel.x

    bound_low = bounds[0][:len(state_expanded)]
    bound_high = bounds[1][:len(state_expanded)]
    bounds = (bound_low, bound_high)

    # update baseline_handles in the lists_args tuple, need to convert to list because tuple is immutable
    ls_args_list = list(ls_args)
    ls_args_list[4] = clock_idxs
    ls_args_list[6] = trop_idxs
    ls_args_list[8] = disb_idxs
    ls_args_list[9] = baseline_handles
    ls_args = tuple(ls_args_list)
    # adjust range weights so chi_squared = 1
    if covariance_kernel_range is None:
        print('running iterative weight adjustment', flush=True)
        print('\n')
        if sol_type == 'GNSS' and store_handle.vlbi_like is False:
            ls_grdel = iterative_weight_adjust_LS_VCE_full(store_handle, state_expanded, bounds, ls_args, calc_residuals, jac, sol_type, 'range')
        elif (store_handle.stochastic_clock or store_handle.stochastic_trop):
            # we cant use the iterative weight adjust function for baseline-specific weighting with a stochastic paramteric model and >1 baseline
            ls_grdel = iterative_weight_adjust_ls_vce(store_handle, state_expanded, bounds, ls_args, calc_residuals, jac, sol_type, 'range', no_PSD=True)
        else:
            ls_grdel = iterative_weight_adjust(store_handle, state_expanded, bounds, ls_args, calc_residuals, jac, sol_type, 'range')
    print('iteratively removing outliers', flush=True)
    ls_grdel, ls_args, bounds = iterative_remove_outliers(store_handle, ls_grdel, bounds, ls_args, calc_residuals, \
            jac, sol_type, n_ao_state, n_grav_state, phase_only=False, phase=False)
    clock_idxs = ls_args[4]
    trop_idxs = ls_args[6]
    disb_idxs = ls_args[8]
    state_expanded = ls_grdel.x
    end_range_state = len(state_expanded) # number of RXPOS + PR clock + trop parameters
    if covariance_kernel_range is None:
        print('re-running iterative weight adjustment', flush=True)
        print('\n')
        if sol_type == 'GNSS' and store_handle.vlbi_like is False:
            ls_grdel = iterative_weight_adjust_LS_VCE_full(store_handle, ls_grdel.x, bounds, ls_args, calc_residuals, jac, sol_type, 'range')
        elif (store_handle.stochastic_clock or store_handle.stochastic_trop):
            if len(baseline_handles)==1:
                ls_grdel = iterative_weight_adjust_ls_vce(store_handle, ls_grdel.x, bounds, ls_args, calc_residuals, jac, sol_type, 'range', no_PSD=True)
            else:
                ls_grdel = iterative_weight_adjust_ls_vce(store_handle, ls_grdel.x, bounds, ls_args, calc_residuals, jac, sol_type, 'range', no_PSD=True)
        else:
            ls_grdel = iterative_weight_adjust(store_handle, ls_grdel.x, bounds, ls_args, calc_residuals, jac, sol_type, 'range')
    grdel_clock_idxs = clock_idxs
    for baseline_handle in baseline_handles:
        baseline_handle.save_range_idxs()

    # analyze group delay LS solution
    sol_name='grdel'
    analyze_ls_solution(sol_type, plot_intermediate_results, ref_antenna, clock_idxs, trop_idxs, disb_idxs, ls_grdel, store_handle, antenna_handles, \
            sol_name, baselines, n_ao_state, baseline_handles)
    write_SINEX(sol_type, sol_name, ls_grdel, store_handle, antenna_handles, ref_antenna)

    if sol_type == 'VLBI':
        # update phase data before slip detection
        for jdx, baseline in enumerate(baselines):
            baseline_handle = baseline_handles[jdx]
            antenna1_handle = antenna_handles[baseline[0]]
            antenna2_handle = antenna_handles[baseline[1]]
            _, ant1_idxs, _ = np.intersect1d(antenna1_handle.times_gps, \
                    baseline_handle.datetime_array, return_indices=True)
            _, ant2_idxs, _ = np.intersect1d(antenna2_handle.times_gps, \
                    baseline_handle.datetime_array, return_indices=True)

            # get clock samples
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
            phase_clock_samples = clock_samples.copy()

            if store_handle.estimate_trop:
                if len(baseline_handles) == 0 or store_handle.stochastic_trop is False:
                    trop_samples1 = antenna1_handle.trop_samples[ant1_idxs]
                    trop_samples2 = antenna2_handle.trop_samples[ant2_idxs]
                else:
                    trop_samples1 = np.zeros(len(baseline_handle.datetime_array))
                    trop_samples2 = np.zeros(len(baseline_handle.datetime_array))
                    if antenna1_handle.estimate_trop:
                        _, ant1_idxs_trop, ant1_dt = np.intersect1d(antenna1_handle.trop_times, \
                                baseline_handle.datetime_array, return_indices=True)
                        trop_samples1[ant1_dt] += antenna1_handle.trop_samples[ant1_idxs_trop]
                    if antenna2_handle.estimate_trop:
                        _, ant2_idxs_trop, ant2_dt = np.intersect1d(antenna2_handle.trop_times, \
                                baseline_handle.datetime_array, return_indices=True)
                        trop_samples2[ant2_dt] += antenna2_handle.trop_samples[ant2_idxs_trop]
            else:
                trop_samples1 = []
                trop_samples2 = []

            if store_handle.src_type == 'GNSS':
                store_handle.model_group_phase_vlbi(antenna1_handle, antenna2_handle, baseline_handle, \
                                         baseline_handle.datetime_array, clock_samples, trop_samples1, trop_samples2,\
                                         True, phase_clock_samples)
            elif store_handle.src_type == 'VLBI':
                store_handle.model_group_phase_farfield(antenna1_handle, antenna2_handle, baseline_handle, \
                                         baseline_handle.datetime_array, clock_samples, trop_samples1, trop_samples2, \
                                         True, phase_clock_samples)
        amb_state, amb_int = define_amb_state_vlbi(store_handle, baseline_handles)
    elif sol_type == 'GNSS':
        clock_state = ls_grdel.x[clock_idxs]
        if store_handle.global_linear_clock:
            idx_start = 1
        elif store_handle.global_quadratic_clock:
            idx_start = 2
        else:
            idx_start = 0
        # update for estimated PR state, calculate phase model for the first time
        for antenna_handle in antenna_handles:
            times_gps = antenna_handle.times_gps
            if ref_antenna != antenna_handle.antenna_name:
                clock_state_ant = clock_state[antenna_handle.range_clock_idxs]
                if store_handle.global_linear_clock or store_handle.global_quadratic_clock:
                    clock_state_global = clock_state_ant[:idx_start]
                    if store_handle.stochastic_clock is False:
                        clock_samples = sample_global_poly_at_interval(clock_state_global, times_gps)
                    else:
                        clock_samples = sample_global_poly_at_interval(clock_state_global, antenna_handle.clock_times)
                    clock_state_ant = clock_state_ant[idx_start:]
                else:
                    clock_samples = np.zeros(len(times_gps))

                if store_handle.stochastic_clock is False and clock_poly_length > 0:
                    clock_samples += sample_poly_at_interval(clock_state_ant, clock_poly_length, times_gps)
                elif store_handle.stochastic_clock:
                    clock_samples += clock_state_ant
                antenna_handle.hold_clock(clock_samples)
                antenna_handle.phase_clock_times = antenna_handle.clock_times
                antenna_handle.hold_clock(clock_samples, phase_delay=True)
                data_corrected = store_handle.correct_PR_CP(antenna_handle, phase=True)
                antenna_handle.hold_data(data_corrected)
        amb_state, amb_int = define_amb_state_gnss(store_handle, baselines, antenna_handles, baseline_handles)

    if do_mcmc_correlation:
        # use Markov Chain Monte Carlo to estimate covariance of measurements
        ls_grdel = mcmc_correlation('range', datetime_array, sol_type, ls_grdel, ls_args, bounds, calc_residuals, jac)
        ## fit a new iterative q value so chi-2=1
        #ls_grdel = iterative_weight_adjust(store_handle, ls_grdel.x, bounds, ls_args, calc_residuals, jac, sol_type, 'range')
        sol_name='grdel_mcmc'
        analyze_ls_solution(sol_type, plot_intermediate_results, ref_antenna, clock_idxs, trop_idxs, disb_idxs, ls_grdel, store_handle, antenna_handles, \
            sol_name, baselines, n_ao_state, baseline_handles)
    elif do_ls_vce_correlation:
        # use Markov Chain Monte Carlo to estimate covariance of measurements
        ls_grdel, _ = ls_correlation('range', datetime_array, sol_type, ls_grdel, ls_args, bounds, calc_residuals, jac, True)
        sol_name='grdel_ls_vce'
        analyze_ls_solution(sol_type, plot_intermediate_results, ref_antenna, clock_idxs, trop_idxs, disb_idxs, ls_grdel, store_handle, antenna_handles, \
            sol_name, baselines, n_ao_state, baseline_handles)

    # create phase clock and add to state
    # NB: phase clock will not necessarily cover the same epochs as range clock
    if store_handle.stochastic_clock is False:
        state_expanded, phase_clock_idxs, phase_disb_idxs, _ = gen_phase_clock_state(store_handle, antenna_handles, baseline_handles,\
                baselines, ref_antenna, state_expanded, end_range_state, clock_idxs, clock_poly_length, trop_idxs, trop_poly_length, disb_idxs)

    if store_handle.stochastic_clock or store_handle.stochastic_trop:
        state_expanded, end_range_state, clock_idxs, trop_idxs, disb_idxs, phase_clock_idxs, phase_disb_idxs, _ = adjust_stoch_params(store_handle, antenna_handles, baseline_handles,\
                baselines, ref_antenna, state_expanded, clock_idxs, trop_idxs, disb_idxs, n_ao_state, n_grav_state, phase=True)
        
    # add ambiguities to state
    state_expanded = np.append(state_expanded, amb_state)
    amb_state_idxs = slice(phase_disb_idxs.stop,len(state_expanded))

    # same bounds, more state elements
    bound_low_expanded = np.zeros_like(state_expanded)
    bound_low_expanded[:rxpos_state_length] = state_expanded[:rxpos_state_length]-pos_bound
    bound_low_expanded[rxpos_state_length:] = bound_low_expanded[rxpos_state_length:] - np.inf

    bound_high_expanded = np.zeros_like(state_expanded)
    bound_high_expanded[rxpos_state_length:] = bound_high_expanded[rxpos_state_length:] + np.inf
    bound_high_expanded[:rxpos_state_length] = state_expanded[:rxpos_state_length]+pos_bound

    #CLOCK_BOUND=baseline_handle.wavelength # 1 ambiguity interval
    #CLOCK_BOUND=0.7
    CLOCK_BOUND=np.inf # no constraint
    bound_low_expanded, bound_high_expanded, state_expanded = set_bounds_phase_clock(bound_low_expanded, bound_high_expanded,\
            CLOCK_BOUND, store_handle, antenna_handles, baseline_handles, baselines, ref_antenna, state_expanded, end_range_state, \
            clock_idxs, clock_poly_length, trop_idxs, disb_idxs, phase_clock_idxs, phase_disb_idxs, n_ao_state, n_grav_state, amb_state_idxs)
    L_curve = False
    # compute float ambiguity carrier phase solution
    phase_delay = True
    phase_only = False
    use_amb_state = True

    if store_handle.iono_free: 
        combination_type = 'WL'
        print('Executing wide-lane float ambiguity solution:', flush=True)
    else: 
        combination_type = None
        print('Executing float ambiguity solution:', flush=True)

    if tikhonov_reg:
        tikhonov_lambda = TK_LAMBDA
    else:
        tikhonov_lambda = 0

    if cont_reg:
        cont_penalty = CT_PENALTY
    else:
        cont_penalty = 0

    store_handle.hold_state(state_expanded-1e-9)
    ls_phfloat = least_squares(calc_residuals, state_expanded, jac=jac, method='trf',\
            max_nfev=100, bounds=(bound_low_expanded, bound_high_expanded), verbose=2, x_scale = 'jac', xtol=1e-15,\
            args=(ref_antenna, baselines, store_handle, antenna_handles, clock_idxs, \
                   clock_poly_length, trop_idxs, trop_poly_length, disb_idxs, baseline_handles, phase_delay, phase_only, \
                   use_amb_state, amb_state_idxs, use_phase_weights, tikhonov_lambda, cont_penalty, phase_clock_idxs, \
                   phase_disb_idxs, combination_type))

    #sol_name = 'amb_temp'
    #analyze_ls_solution(sol_type, plot_intermediate_results, ref_antenna, clock_idxs, trop_idxs, disb_idxs, ls_phfloat, store_handle, antenna_handles, sol_name, baselines,\
    #                    n_ao_state, baseline_handles, phase_delay, phase_only, phase_clock_idxs, phase_disb_idxs)
   
    # remove the outlier phase data after the first float solution
    residuals = ls_phfloat.fun
    remove_outliers(residuals, baseline_handles)
    
    state_expanded = ls_phfloat.x
    # trim the phase state to remove unused state elements
    state_expanded, amb_state_idxs = trim_amb_state(baseline_handles, state_expanded, amb_state_idxs)
    if store_handle.stochastic_clock is False:
        state_expanded, phase_clock_idxs, phase_disb_idxs, amb_state_idxs = gen_phase_clock_state(store_handle, antenna_handles, baseline_handles, \
                baselines, ref_antenna, state_expanded, end_range_state, clock_idxs, clock_poly_length, trop_idxs, trop_poly_length, \
                disb_idxs, phase_clock_idxs, phase_disb_idxs, amb_state_idxs)
    if store_handle.stochastic_clock or store_handle.stochastic_trop:
        state_expanded, end_range_state, clock_idxs, trop_idxs, disb_idxs, phase_clock_idxs, phase_disb_idxs, amb_state_idxs = adjust_stoch_params(store_handle, antenna_handles, baseline_handles,\
                            baselines, ref_antenna, state_expanded, clock_idxs, trop_idxs, disb_idxs, n_ao_state, n_grav_state, phase_delay, phase_clock_idxs, phase_disb_idxs, amb_state_idxs)
    bound_low_expanded, bound_high_expanded, state_expanded = set_bounds_phase_clock(bound_low_expanded, bound_high_expanded,\
            CLOCK_BOUND, store_handle, antenna_handles, baseline_handles, baselines, ref_antenna, state_expanded, end_range_state, \
            clock_idxs, clock_poly_length, trop_idxs, disb_idxs, phase_clock_idxs, phase_disb_idxs, n_ao_state, n_grav_state, amb_state_idxs)
    store_handle.hold_state(state_expanded+1e-9)
    ls_args=(ref_antenna, baselines, store_handle, antenna_handles, clock_idxs, \
                   clock_poly_length, trop_idxs, trop_poly_length, disb_idxs, baseline_handles, phase_delay, phase_only, use_amb_state, \
                  amb_state_idxs, use_phase_weights, tikhonov_lambda, cont_penalty, phase_clock_idxs, phase_disb_idxs, combination_type)
    # recompute the solution without outliers starting from previous solution -- should improve phase data, converge quickly
    print('Executing outlier-free float ambiguity solution:', flush=True)
    # resolve the integer ambiguity from the floating point estimation with LAMBDA
    bounds=(bound_low_expanded, bound_high_expanded)
    afixed, ls_phfloat, phfloat_state, n_unresolved, z_fixed, iZt, phase_only_res = resolve_float_amb(store_handle, state_expanded, bounds, \
            n_ao_state, n_grav_state, ls_args, calc_residuals, jac, sol_type, recursive_amb, L_curve=L_curve)

    if recursive_amb:
        if sol_type == 'GNSS':
            slip_detected, slips_full_last, slips_baseline = detect_unresolved_amb_gnss(store_handle, afixed, antenna_handles, baselines, baseline_handles)
        elif sol_type == 'VLBI':
            slip_detected, slips_full_last, slips_baseline = detect_unresolved_amb_vlbi(store_handle.iono_free, afixed, baseline_handles)
        slips_full_last2 = slips_full_last
        if slip_detected:
            max_iter = 30
            #max_iter = 4
            iterations = 0 
            while slip_detected:
                print(str(len(slips_full_last)) + ' cycle slips detected, re-running ambiguity fix')
                assimilate_amb_state(sol_type, afixed, baseline_handles, store_handle.iono_free)
                for jdx, baseline_handle in enumerate(baseline_handles):
                   if len(slips_baseline)>0:
                       baseline_handle.save_phase_slips(slips_baseline[jdx])
                   else:
                       baseline_handle.save_phase_slips([])
                amb_state = redefine_amb_state(sol_type, store_handle, baseline_handles, baselines, antenna_handles)
                state_expanded = np.append(state_expanded[:phase_disb_idxs.stop], amb_state)
                store_handle.hold_state(state_expanded+1e-9)
                amb_state_idxs = slice(phase_disb_idxs.stop,len(state_expanded))
                bound_low_expanded = np.zeros_like(state_expanded)
                bound_low_expanded[:rxpos_state_length] = state_expanded[:rxpos_state_length]-pos_bound
                bound_low_expanded[rxpos_state_length:] = bound_low_expanded[rxpos_state_length:] - np.inf
                bound_high_expanded = np.zeros_like(state_expanded)
                bound_high_expanded[rxpos_state_length:] = bound_high_expanded[rxpos_state_length:] + np.inf
                bound_high_expanded[:rxpos_state_length] = state_expanded[:rxpos_state_length]+pos_bound
                bound_low_expanded, bound_high_expanded, state_expanded = set_bounds_phase_clock(bound_low_expanded, bound_high_expanded,\
                        CLOCK_BOUND, store_handle, antenna_handles, baseline_handles, baselines, ref_antenna, state_expanded, end_range_state, \
                        clock_idxs, clock_poly_length, trop_idxs, disb_idxs, phase_clock_idxs, phase_disb_idxs, n_ao_state, n_grav_state, amb_state_idxs)
                bounds=(bound_low_expanded, bound_high_expanded)
                ls_args=(ref_antenna, baselines, store_handle, antenna_handles, clock_idxs, \
                               clock_poly_length, trop_idxs, trop_poly_length, disb_idxs, baseline_handles, phase_delay, phase_only, use_amb_state, \
                              amb_state_idxs, use_phase_weights, tikhonov_lambda, cont_penalty, phase_clock_idxs, phase_disb_idxs, combination_type)

                # update measurement model after redefining state -- calling residuals fcn will do this
                res = calc_residuals(state_expanded+1e-9, ref_antenna, baselines, store_handle, antenna_handles, clock_idxs, \
                               clock_poly_length, trop_idxs, trop_poly_length, disb_idxs, baseline_handles, phase_delay, phase_only, use_amb_state, \
                              amb_state_idxs, use_phase_weights, tikhonov_lambda, cont_penalty, phase_clock_idxs, phase_disb_idxs, combination_type)
                afixed, ls_phfloat, phfloat_state, n_unresolved, z_fixed, iZt, phase_only_res = resolve_float_amb(store_handle, state_expanded, bounds, \
                        n_ao_state, n_grav_state, ls_args, calc_residuals, jac, sol_type, recursive_amb)
                iterations += 1 
                if sol_type == 'GNSS':
                    slip_detected, slips_full, slips_baseline = detect_unresolved_amb_gnss(store_handle, afixed, antenna_handles, baselines, baseline_handles)
                elif sol_type == 'VLBI':
                    slip_detected, slips_full, slips_baseline = detect_unresolved_amb_vlbi(store_handle.iono_free, afixed, baseline_handles)
                if iterations > max_iter or slips_full == slips_full_last: 
                    break
                if slips_full==slips_full_last2:
                    if len(slips_full) < len(slips_full_last):
                        break
                slips_full_last2 = slips_full_last
                slips_full_last = slips_full
        if slip_detected is False:
            print('All ambiguities successfully resolved!')
        else:
            print('Problem with ambiguity resolution. ' + str(len(slips_full_last)) + ' cycle slips or outliers remain. Review phase residuals')

    # Show float ambiguity solution
    if store_handle.iono_free: 
        sol_name = 'amb_float_WL'
    else:
        sol_name = 'amb_float'

    if phase_only_res is False: 
        # already analyzed solution in function if iterative LAMBDA invoked
        analyze_ls_solution(sol_type, plot_intermediate_results, ref_antenna, clock_idxs, trop_idxs, disb_idxs, ls_phfloat, store_handle, antenna_handles, sol_name, baselines,\
                            n_ao_state, baseline_handles, phase_delay, phase_only_res, phase_clock_idxs, phase_disb_idxs, False, afixed)
    if n_unresolved == len(afixed):
        print('No ambiguites were resolved -- floating point phase solution is the best we can do. Exiting...')
        exit(0)
            
    # do the ionosphere-free combination and float ambiguity resolution
    if store_handle.iono_free:
        # generate fixed widelane solution (for debugging purposes)
        print('Executing widelane fixed ambiguity solution:', flush=True)
        state_WL = phfloat_state[:amb_state_idxs.start]
        ls_phfixed_WL = least_squares(calc_residuals, state_WL, jac=jac, method='trf',\
            max_nfev=100, bounds=(bound_low_expanded[:amb_state_idxs.start], bound_high_expanded[:amb_state_idxs.start]), verbose=2, x_scale = 'jac', xtol=1e-15,\
            args=(ref_antenna, baselines, store_handle, antenna_handles, clock_idxs, \
                   clock_poly_length, trop_idxs, trop_poly_length, disb_idxs, baseline_handles, phase_delay, phase_only, \
                   use_amb_state, amb_state_idxs, use_phase_weights, tikhonov_lambda, cont_penalty, phase_clock_idxs, phase_disb_idxs,  \
                   combination_type))
        sol_name = 'amb_fixed_WL'
        analyze_ls_solution(sol_type, plot_intermediate_results, ref_antenna, clock_idxs, trop_idxs, disb_idxs, ls_phfixed_WL, store_handle, antenna_handles, sol_name, baselines,\
                        n_ao_state, baseline_handles, phase_delay, phase_only, phase_clock_idxs, phase_disb_idxs)

        bound_low_expanded, bound_high_expanded, state_WL = set_bounds_phase_clock(bound_low_expanded, bound_high_expanded,\
                CLOCK_BOUND, store_handle, antenna_handles, baseline_handles, baselines, ref_antenna, state_WL, end_range_state, \
                clock_idxs, clock_poly_length, trop_idxs, disb_idxs, phase_clock_idxs, phase_disb_idxs, n_ao_state, n_grav_state, amb_state_idxs)


        state_NL = phfloat_state[:amb_state_idxs.start]
        n_amb = 0 
        for idx, baseline in enumerate(baselines):
            baseline_handle =  baseline_handles[idx]
            afixed_baseline = afixed[n_amb:n_amb+baseline_handle.n_amb_state]
            baseline_handle.resolve_widelane_amb(afixed_baseline) 

            # now do IF combination
            antenna1_handle = antenna_handles[baseline[0]]
            antenna2_handle = antenna_handles[baseline[1]]
            data_ant1 = antenna1_handle.antenna_data
            data_ant2 = antenna2_handle.antenna_data

            diff_cp_model = data_ant2.cp_model.values - data_ant1.cp_model.values  
            diff_cp_dual_model = data_ant2.cp_dual_model.values - data_ant1.cp_dual_model.values  
            baseline_handle.combination_measurement('IF')
            baseline_handle.combination_model(diff_cp_model, diff_cp_dual_model, 'IF')

            # get new float ambiguity estimate for IF combination
            N_float = construct_float_amb(baseline_handle.cp_combination, \
                        baseline_handle.cp_model_combination, baseline_handle.comb_wavelength)
            n_amb = n_amb + baseline_handle.n_amb_state
            for idx, slip_slice in enumerate(baseline_handle.slip_slices_arr):
                N_idxs = N_float[slip_slice]
                N_good = N_idxs[~np.isnan(N_idxs)]
                N_interval = np.mean(N_good)
                state_NL = np.append(state_NL, N_interval)

        combination_type = 'IF'
        store_handle.hold_state(state_NL+1e-9)
        print('Executing ionosphere-free float ambiguity solution:', flush=True)
        ls_phfloat_NL = least_squares(calc_residuals, state_NL, jac=jac, method='trf',\
                max_nfev=100, bounds=(bound_low_expanded, bound_high_expanded), verbose=2, x_scale = 'jac', xtol=1e-15,\
                args=(ref_antenna, baselines, store_handle, antenna_handles, clock_idxs, \
                       clock_poly_length, trop_idxs, trop_poly_length, disb_idxs, baseline_handles, phase_delay, phase_only,\
                       use_amb_state, amb_state_idxs, use_phase_weights, tikhonov_lambda, cont_penalty, phase_clock_idxs, \
                       phase_disb_idxs, combination_type))
        
        # remove the outlier phase data after the IF float solution
        remove_outliers(ls_phfloat_NL.fun, baseline_handles)
        
        state_NL = ls_phfloat_NL.x

        # adjust the ambiguity state to remove any ambiguities with no data remaining
        state_NL, amb_state_idxs = trim_amb_state(baseline_handles, state_NL, amb_state_idxs)
        # regenerate the phase clock state 
        if store_handle.stochastic_clock is False:
            state_NL, phase_clock_idxs, phase_disb_idxs, amb_state_idxs = gen_phase_clock_state(store_handle, antenna_handles, baseline_handles,\
                baselines, ref_antenna, state_NL, end_range_state, clock_idxs,  clock_poly_length, trop_idxs, trop_poly_length, \
                disb_idxs, phase_clock_idxs, phase_disb_idxs, amb_state_idxs)

        if store_handle.stochastic_clock or store_handle.stochastic_trop:
            state_NL, end_range_state, clock_idxs, trop_idxs, disb_idxs, phase_clock_idxs, phase_disb_idxs, amb_state_idxs = adjust_stoch_params(store_handle, antenna_handles, baseline_handles,\
                                baselines, ref_antenna, state_NL, clock_idxs, trop_idxs, disb_idxs, n_ao_state, n_grav_state, phase_delay, phase_clock_idxs, phase_disb_idxs, amb_state_idxs)

        bound_low_expanded, bound_high_expanded, state_NL = set_bounds_phase_clock(bound_low_expanded, bound_high_expanded,\
                CLOCK_BOUND, store_handle, antenna_handles, baseline_handles, baselines, ref_antenna, state_NL, end_range_state, \
                clock_idxs, clock_poly_length, trop_idxs, disb_idxs, phase_clock_idxs, phase_disb_idxs, n_ao_state, n_grav_state, amb_state_idxs)
        store_handle.hold_state(state_NL+1e-9)
        print('Executing outlier-free ionosphere-free float ambiguity solution:', flush=True)
        bounds=(bound_low_expanded, bound_high_expanded)
        ls_args=(ref_antenna, baselines, store_handle, antenna_handles, clock_idxs, \
                       clock_poly_length, trop_idxs, trop_poly_length, disb_idxs, baseline_handles, phase_delay, phase_only, \
                       use_amb_state, amb_state_idxs, use_phase_weights, tikhonov_lambda, cont_penalty, phase_clock_idxs, \
                       phase_disb_idxs, combination_type)
        afixed, ls_phfloat_NL, phfloat_NL_state, n_unresolved, z_fixed, iZt, phase_only_res = resolve_float_amb(store_handle, state_NL, bounds, n_ao_state, n_grav_state, ls_args, \
                calc_residuals, jac, sol_type, recursive_amb)
        sol_name = 'amb_float_NL'
        analyze_ls_solution(sol_type, plot_intermediate_results, ref_antenna, clock_idxs, trop_idxs, disb_idxs, ls_phfloat_NL, store_handle, antenna_handles, sol_name, baselines,\
                        n_ao_state, baseline_handles, phase_delay, phase_only_res, phase_clock_idxs, phase_disb_idxs)
    
    # start preparing fixed ambiguity solution
    end_range_state = len(grdel_state)
    if store_handle.iono_free:
        state_apriori_phfix = phfloat_NL_state
    else:
        state_apriori_phfix = phfloat_state
    store_handle.hold_state(state_apriori_phfix+1e-9)
    use_phase_weights = True
    
    # Remove outlier data pre-LS adjustment after resolving ambiguity
    if n_unresolved == 0:
        # if n_unresolved is 0, we have successfully fixed all ambiguities, otherwise we are still estimating a subset 
        # of ambiguities in Z-domain
        use_amb_state = False
        amb_state_idxs = slice(amb_state_idxs.start,amb_state_idxs.start)
        integer_amb = afixed
        residuals = calc_residuals(state_apriori_phfix, ref_antenna, baselines, store_handle, antenna_handles, clock_idxs, \
                       clock_poly_length, trop_idxs, trop_poly_length, disb_idxs, baseline_handles, phase_delay, phase_only, \
                       use_amb_state, amb_state_idxs, use_phase_weights, tikhonov_lambda, cont_penalty, phase_clock_idxs, \
                       phase_disb_idxs, combination_type, integer_amb)
    else:
        use_amb_state = True
        amb_state_idxs = slice(amb_state_idxs.start, len(state_apriori_phfix))
        z_par = z_fixed[n_unresolved:]
        residuals = calc_residuals(state_apriori_phfix, ref_antenna, baselines, store_handle, antenna_handles, clock_idxs, \
                       clock_poly_length, trop_idxs, trop_poly_length, disb_idxs, baseline_handles, phase_delay, phase_only,\
                       use_amb_state, amb_state_idxs, use_phase_weights, tikhonov_lambda, cont_penalty, phase_clock_idxs, \
                       phase_disb_idxs, combination_type, z_par, iZt)

    # set regularization parameters to 0 -- dont bias final positioning solution
    tikhononv_lambda = 0
    cont_penalty = 0
    #CLOCK_BOUND=np.inf # dont need constraint once ambiguities are fixed

    remove_outliers(residuals, baseline_handles)
    if store_handle.stochastic_clock or store_handle.stochastic_trop:
        state_apriori_phfix, end_range_state, clock_idxs, trop_idxs, disb_idxs, phase_clock_idxs, phase_disb_idxs, amb_state_idxs =\
                adjust_stoch_params(store_handle, antenna_handles, baseline_handles,\
                            baselines, ref_antenna, state_apriori_phfix, clock_idxs, trop_idxs, disb_idxs, n_ao_state, n_grav_state, \
                            phase_delay, phase_clock_idxs, phase_disb_idxs, amb_state_idxs)

    # Do the first fixed carrier phase solution
    bound_low_phfixed, bound_high_phfixed, state_apriori_phfix = set_bounds_phase_clock(bound_low_expanded, bound_high_expanded,\
            CLOCK_BOUND, store_handle, antenna_handles, baseline_handles, baselines, ref_antenna, state_apriori_phfix, end_range_state, \
            clock_idxs, clock_poly_length, trop_idxs, disb_idxs, phase_clock_idxs, phase_disb_idxs, n_ao_state, n_grav_state, amb_state_idxs)
    
    # adjust the state to remove any ambiguities with no data remaining
    if n_unresolved != 0:
        # the number of ambiguities can't change in Z-domain, simply remove rows in the mapping back to a-domain
        iZt = trim_amb_Zdom(baseline_handles, iZt)

    bound_low_phfix, bound_high_phfix, _ = set_bounds_phase_clock(bound_low_phfixed, bound_high_phfixed,\
            CLOCK_BOUND, store_handle, antenna_handles, baseline_handles, baselines, ref_antenna, state_apriori_phfix, end_range_state, \
            clock_idxs, clock_poly_length, trop_idxs, disb_idxs, phase_clock_idxs, phase_disb_idxs, n_ao_state, n_grav_state, amb_state_idxs)

    clock_state_pre_fix = state_apriori_phfix[clock_idxs]
    trop_state_pre_fix = state_apriori_phfix[trop_idxs]
    disb_state_pre_fix = state_apriori_phfix[disb_idxs]
    disb_idxs_prefix = disb_idxs
    if store_handle.stochastic_clock or store_handle.stochastic_trop:
        state_apriori_phfix, end_range_state, clock_idxs, trop_idxs, disb_idxs, phase_clock_idxs, phase_disb_idxs, amb_state_idxs =\
                adjust_stoch_params(store_handle, antenna_handles, baseline_handles,\
                    baselines, ref_antenna, state_apriori_phfix, clock_idxs, trop_idxs, disb_idxs, n_ao_state, n_grav_state, phase_delay,\
                    phase_clock_idxs, phase_disb_idxs, amb_state_idxs, phase_only=True)
   
    if HOLD_GRAV:
        n_grav_state = n_grav_state_hold
    rxpos = state_apriori_phfix[:3*(len(antenna_handles)-1)]
    rxpos_idxs = slice(0,len(rxpos))
    ao_idxs = slice(clock_idxs.stop, clock_idxs.stop+n_ao_state)
    grav_idxs = slice(clock_idxs.stop+n_ao_state, clock_idxs.stop+n_ao_state+n_grav_state)
    clock_idxs_phfix = slice((len(antenna_handles)-1)*3,(len(antenna_handles)-1)*3)
    ao_idxs_phfix = slice(clock_idxs_phfix.stop,clock_idxs_phfix.stop+n_ao_state)
    grav_idxs_phfix = slice(clock_idxs_phfix.stop+n_ao_state,clock_idxs_phfix.stop+n_ao_state+n_grav_state)
    if store_handle.stochastic_clock is False and store_handle.stochastic_trop is False:
        # remove clock states
        ao_state = state_apriori_phfix[ao_idxs]
        grav_state = state_apriori_phfix[grav_idxs]
        trop_state = state_apriori_phfix[trop_idxs]
        phase_clock = state_apriori_phfix[phase_clock_idxs]
        phase_disb_state = state_apriori_phfix[phase_disb_idxs]
        if store_handle.stochastic_trop:
            trop_state = trop_state[:len(phase_clock)]
        state_phfinal = np.concatenate((rxpos,ao_state,grav_state,trop_state,phase_clock,phase_disb_state))
        trop_idxs_phfix = slice(grav_idxs_phfix.stop,grav_idxs_phfix.stop+n_trop_tot)
        disb_idxs_phfix = slice(trop_idxs_phfix.stop,trop_idxs_phfix.stop)
    else:
        # we already took these steps in adjust_stoch_params
        state_phfinal = state_apriori_phfix
        phase_clock = state_phfinal[phase_clock_idxs]
        phase_disb_state = state_apriori_phfix[phase_disb_idxs]
        trop_idxs_phfix = trop_idxs
        disb_idxs_phfix = disb_idxs

    if HOLD_GRAV and n_grav_state>0:
        grav_state = np.zeros(n_grav_state)
        ao_state = state_apriori_phfix[ao_idxs]
        trop_state = state_apriori_phfix[trop_idxs]
        phase_clock = state_apriori_phfix[phase_clock_idxs]
        phase_disb_state = state_apriori_phfix[phase_disb_idxs]
        if store_handle.stochastic_trop :
            trop_state = trop_state[:len(phase_clock)-idx_start]
        state_phfinal = np.concatenate((rxpos,ao_state,grav_state,trop_state,phase_clock,phase_disb_state))
        if stochastic_trop is False:
            trop_idxs_phfix = slice(grav_idxs_phfix.stop,grav_idxs_phfix.stop+n_trop_tot)
        else:
            trop_idxs_phfix = slice(grav_idxs_phfix.stop,grav_idxs_phfix.stop+len(phase_clock))
        disb_idxs_phfix = slice(trop_idxs_phfix.stop,trop_idxs_phfix.stop)

        bound_low_phfix=np.concatenate((bound_low_phfix,-np.ones(2)*np.inf))
        bound_high_phfix=np.concatenate((bound_high_phfix,np.ones(2)*np.inf))
        for antenna_handle in antenna_handles:
            if antenna_handle.antenna_name in grav_antennas:
                antenna_handle.estimate_grav_def = True

    phase_clock_idxs_phfix = slice(disb_idxs_phfix.stop,disb_idxs_phfix.stop+len(phase_clock))
    phase_disb_idxs_phfix = slice(phase_clock_idxs_phfix.stop, phase_clock_idxs_phfix.stop+len(phase_disb_state))
    if n_unresolved > 0: 
        state_phfinal = np.concatenate((state_phfinal,state_apriori_phfix[amb_state_idxs]))
        amb_state_idxs_phfix = slice(phase_disb_idxs_phfix.stop,len(state_phfinal))
    else:
        amb_state_idxs_phfix = slice(phase_disb_idxs_phfix.stop,phase_disb_idxs_phfix.stop)
    bound_low_phfinal = union_of_slices(bound_low_phfix, rxpos_idxs, ao_idxs_phfix, grav_idxs_phfix, trop_idxs_phfix, disb_idxs_phfix, phase_clock_idxs_phfix, phase_disb_idxs_phfix, amb_state_idxs_phfix)
    bound_high_phfinal = union_of_slices(bound_high_phfix, rxpos_idxs, ao_idxs_phfix, grav_idxs_phfix, trop_idxs_phfix, disb_idxs_phfix, phase_clock_idxs_phfix, phase_disb_idxs_phfix, amb_state_idxs_phfix)

    print('Executing initial phase-only fixed ambiguity solution:', flush=True)
    # NB: clock_idxs, though passed, are not used in final least_squares or subsequent analysis due to phase_only param
    phase_only=True
    bounds = (bound_low_phfinal, bound_high_phfinal)
    if n_unresolved == 0:
        ls_args = (ref_antenna, baselines, store_handle, antenna_handles, clock_idxs_phfix, \
                       clock_poly_length, trop_idxs_phfix, trop_poly_length, disb_idxs_phfix, baseline_handles, \
                       phase_delay, phase_only, use_amb_state, amb_state_idxs_phfix, use_phase_weights, tikhonov_lambda, cont_penalty, phase_clock_idxs_phfix, \
                       phase_disb_idxs_phfix, combination_type, integer_amb)
    else:
        ls_args = (ref_antenna, baselines, store_handle, antenna_handles, clock_idxs_phfix, \
                       clock_poly_length, trop_idxs_phfix, trop_poly_length, disb_idxs_phfix, baseline_handles, phase_delay, phase_only, \
                       use_amb_state, amb_state_idxs_phfix, use_phase_weights, tikhonov_lambda, cont_penalty, phase_clock_idxs_phfix, \
                       phase_disb_idxs_phfix, combination_type, z_par, iZt)
    ls_phfixed_phaseonly_first = least_squares(calc_residuals, state_phfinal, jac=jac, method='trf',\
            max_nfev=100, bounds=bounds, verbose=2, x_scale = 'jac', xtol=1e-15,\
            args=ls_args)

    if covariance_kernel_phase is None:
        print('running iterative weight adjustment')
        print('\n') 
        if sol_type == 'GNSS' and store_handle.vlbi_like is False:
            ls_phfixed_phaseonly_first = iterative_weight_adjust_LS_VCE_full(store_handle, ls_phfixed_phaseonly_first.x, bounds,\
                    ls_args, calc_residuals, jac, sol_type, 'phase')
        elif (store_handle.stochastic_clock or store_handle.stochastic_trop):
            ls_phfixed_phaseonly_first = iterative_weight_adjust_ls_vce(store_handle, ls_phfixed_phaseonly_first.x, bounds,\
                    ls_args, calc_residuals, jac, sol_type, 'phase', no_PSD=True)
        else:
            ls_phfixed_phaseonly_first = iterative_weight_adjust(store_handle, ls_phfixed_phaseonly_first.x, bounds,\
                    ls_args, calc_residuals, jac, sol_type, 'phase')


    residuals = ls_phfixed_phaseonly_first.fun
    state_phfinal = ls_phfixed_phaseonly_first.x
    print('iteratively removing outliers')
    print('\n')
    ls_phfixed_phaseonly_first, ls_args, bounds = iterative_remove_outliers(store_handle, ls_phfixed_phaseonly_first, bounds, \
            ls_args, calc_residuals, jac, sol_type, n_ao_state, n_grav_state, phase_only, phase_delay)
    if n_unresolved == 0:
        (ref_antenna, baselines, store_handle, antenna_handles, clock_idxs_phfix, \
                       clock_poly_length, trop_idxs_phfix, trop_poly_length, disb_idxs_phfix, baseline_handles, \
                       phase_delay, phase_only, use_amb_state, amb_state_idxs_phfix, use_phase_weights, tikhonov_lambda, cont_penalty, phase_clock_idxs_phfix, \
                       phase_disb_idxs_phfix, combination_type, integer_amb) = ls_args

    else:
        (ref_antenna, baselines, store_handle, antenna_handles, clock_idxs_phfix, \
                       clock_poly_length, trop_idxs_phfix, trop_poly_length, disb_idxs_phfix, baseline_handles, phase_delay, phase_only, \
                       use_amb_state, amb_state_idxs_phfix, use_phase_weights, tikhonov_lambda, cont_penalty, phase_clock_idxs_phfix, \
                       phase_disb_idxs_phfix, combination_type, z_par, iZt) = ls_args

    #remove_outliers(residuals, baseline_handles, phase_only)
    # second weight adjustment after residuals removed
    if covariance_kernel_phase is None:
        print('re-running iterative weight adjustment')
        print('\n')
        if sol_type == 'GNSS' and store_handle.vlbi_like is False:
            ls_phfixed_phaseonly_first = iterative_weight_adjust_LS_VCE_full(store_handle, ls_phfixed_phaseonly_first.x, bounds,\
                    ls_args, calc_residuals, jac, sol_type, 'phase')
        elif (store_handle.stochastic_clock or store_handle.stochastic_trop):
            ls_phfixed_phaseonly_first = iterative_weight_adjust_ls_vce(store_handle, ls_phfixed_phaseonly_first.x, bounds,\
                    ls_args, calc_residuals, jac, sol_type, 'phase')
        else:
            ls_phfixed_phaseonly_first = iterative_weight_adjust(store_handle, ls_phfixed_phaseonly_first.x, bounds,\
                    ls_args, calc_residuals, jac, sol_type, 'phase')
    end_range_state = len(rxpos)+n_ao_state+n_grav_state+n_trop_tot
    state_hold = state_phfinal
    if store_handle.stochastic_clock is False:
        state_phfinal, phase_clock_idxs_phfix, phase_disb_idxs, _ = gen_phase_clock_state(store_handle, antenna_handles, baseline_handles,\
                baselines, ref_antenna, state_phfinal, end_range_state, clock_idxs_phfix, clock_poly_length, trop_idxs_phfix, trop_poly_length, \
                disb_idxs_phfix, phase_clock_idxs_phfix, phase_disb_idxs_phfix, amb_state_idxs_phfix)

    if store_handle.stochastic_clock or store_handle.stochastic_trop:
        state_phfinal, end_range_state, clock_idxs_phfix, trop_idxs_phfix, disb_idxs_phfix, phase_clock_idxs_phfix, phase_disb_idxs_phfix, amb_state_idxs_phfix =\
                adjust_stoch_params(store_handle, antenna_handles, baseline_handles,\
                            baselines, ref_antenna, state_phfinal, clock_idxs_phfix, trop_idxs_phfix, disb_idxs_phfix, n_grav_state, n_ao_state, phase_delay, \
                            phase_clock_idxs_phfix, phase_disb_idxs_phfix, amb_state_idxs_phfix, phase_only=True)

    bound_low_phfinal = bound_low_phfinal[:len(state_phfinal)]
    bound_high_phfinal = bound_high_phfinal[:len(state_phfinal)]
    bounds=(bound_low_phfinal, bound_high_phfinal)

    print('Executing final phase-only fixed ambiguity solution:')
    if n_unresolved == 0:
        ls_args = (ref_antenna, baselines, store_handle, antenna_handles, clock_idxs_phfix, \
                       clock_poly_length, trop_idxs_phfix, trop_poly_length, disb_idxs_phfix, baseline_handles, \
                       phase_delay, phase_only, use_amb_state, amb_state_idxs_phfix, use_phase_weights, tikhonov_lambda, cont_penalty, phase_clock_idxs_phfix, \
                       phase_disb_idxs_phfix, combination_type, integer_amb)

        ls_phfixed_phaseonly = least_squares(calc_residuals, state_phfinal, jac=jac, method='trf',\
                max_nfev=100, bounds=(bound_low_phfinal, bound_high_phfinal), verbose=2, x_scale = 'jac', xtol=1e-15,\
                args=ls_args)
    else:
        ls_args = (ref_antenna, baselines, store_handle, antenna_handles, clock_idxs_phfix, \
                       clock_poly_length, trop_idxs_phfix, trop_poly_length, disb_idxs_phfix, baseline_handles, phase_delay, \
                       phase_only, use_amb_state, amb_state_idxs_phfix, use_phase_weights, tikhonov_lambda, cont_penalty, phase_clock_idxs_phfix, \
                       phase_disb_idxs_phfix, combination_type, z_par, iZt)
    sol_name='final_phase_only'
    analyze_ls_solution(sol_type, True, ref_antenna, clock_idxs_phfix, trop_idxs_phfix, disb_idxs_phfix, ls_phfixed_phaseonly, store_handle,\
                        antenna_handles, sol_name, baselines, n_ao_state, baseline_handles, phase_delay, phase_only, \
                        phase_clock_idxs_phfix, phase_disb_idxs_phfix, use_phase_weights, integer_amb)
    write_SINEX(sol_type, sol_name, ls_phfixed_phaseonly, store_handle, antenna_handles, ref_antenna)

    if do_mcmc_correlation:
        # use Markov Chain Monte Carlo to estimate covariance of measurements
        ls_phfixed_phaseonly = mcmc_correlation('phase', datetime_array, sol_type, ls_phfixed_phaseonly, ls_args, bounds, calc_residuals, jac)
        #ls_phfixed_phaseonly = iterative_weight_adjust(store_handle, ls_phfixed_phaseonly.x, bounds,\
        #         ls_args, calc_residuals, jac, sol_type, 'phase')
        sol_name='phase_only_mcmc'
        analyze_ls_solution(sol_type, True, ref_antenna, clock_idxs_phfix, trop_idxs_phfix, disb_idxs_phfix, ls_phfixed_phaseonly, store_handle,\
                        antenna_handles, sol_name, baselines, n_ao_state, baseline_handles, phase_delay, phase_only, \
                        phase_clock_idxs_phfix, phase_disb_idxs_phfix, use_phase_weights, integer_amb)
    elif do_ls_vce_correlation :
        ls_phfixed_phaseonly,_ = ls_correlation('phase', datetime_array, sol_type, ls_phfixed_phaseonly, ls_args, bounds, calc_residuals, jac, True)
        sol_name='phase_only_ls_vce'
        analyze_ls_solution(sol_type, True, ref_antenna, clock_idxs_phfix, trop_idxs_phfix, disb_idxs_phfix, ls_phfixed_phaseonly, store_handle,\
                        antenna_handles, sol_name, baselines, n_ao_state, baseline_handles, phase_delay, phase_only, \
                        phase_clock_idxs_phfix, phase_disb_idxs_phfix, use_phase_weights, integer_amb)

    plot_time_units(ref_antenna, ls_grdel, ls_phfixed_phaseonly, sol_type, store_handle,\
                        antenna_handles, baselines, store_handle.iono_free, baseline_handles, \
                        clock_poly_length, grdel_clock_idxs, phase_clock_idxs_phfix)

    # generate a new phase clock state accounting for removed epochs
    #end_range_state = len(grdel_state)
    end_range_state = len(rxpos)+len(clock_state_pre_fix)+n_ao_state+n_grav_state+n_trop_tot+n_disb_tot
    # put the range clock state back in 
    ao_state_phfix = ls_phfixed_phaseonly.x[ao_idxs_phfix] 
    grav_state_phfix =  ls_phfixed_phaseonly.x[grav_idxs_phfix] 
    state_apriori_phfix = np.concatenate((state_apriori_phfix[:clock_idxs.start], clock_state_pre_fix, ao_state_phfix, grav_state_phfix,\
            trop_state_pre_fix, disb_state_pre_fix, ls_phfixed_phaseonly.x[phase_clock_idxs_phfix.start:]))
    if store_handle.stochastic_clock is False:
        state_apriori_phfix, phase_clock_idxs, phase_disb_idxs, _ = gen_phase_clock_state(store_handle, antenna_handles, baseline_handles,\
                baselines, ref_antenna, state_apriori_phfix, end_range_state, clock_idxs, clock_poly_length, trop_idxs, trop_poly_length, disb_idxs_prefix, \
                phase_clock_idxs, phase_disb_idxs, amb_state_idxs)

    if store_handle.stochastic_clock or store_handle.stochastic_trop:
        state_apriori_phfix, end_range_state, clock_idxs, trop_idxs, disb_idxs, phase_clock_idxs, phase_disb_idxs, amb_state_idxs = \
                adjust_stoch_params(store_handle, antenna_handles, baseline_handles,\
                            baselines, ref_antenna, state_apriori_phfix, clock_idxs, trop_idxs, disb_idxs_prefix, \
                            n_ao_state, n_grav_state, phase_delay, phase_clock_idxs, phase_disb_idxs, amb_state_idxs, phase_only=False)

    phase_only=False
    store_handle.hold_state(state_apriori_phfix+1e-9)
    print('Executing initial PR+CP fixed ambiguity solution:')
    if n_unresolved == 0:
        ls_args = (ref_antenna, baselines, store_handle, antenna_handles, clock_idxs, \
                       clock_poly_length, trop_idxs, trop_poly_length, disb_idxs, baseline_handles, phase_delay, phase_only,\
                       use_amb_state, amb_state_idxs, use_phase_weights, tikhonov_lambda, cont_penalty, phase_clock_idxs, \
                       phase_disb_idxs, combination_type, integer_amb)
    else:
        ls_args = (ref_antenna, baselines, store_handle, antenna_handles, clock_idxs, \
                       clock_poly_length, trop_idxs, trop_poly_length, disb_idxs, baseline_handles, phase_delay, phase_only, \
                       use_amb_state, amb_state_idxs, use_phase_weights, tikhonov_lambda, cont_penalty, phase_clock_idxs, \
                       phase_disb_idxs, combination_type, z_par, iZt)
    bound_low_phfixed = union_of_slices(bound_low_phfix, rxpos_idxs, clock_idxs, ao_idxs, grav_idxs, trop_idxs, disb_idxs, phase_clock_idxs, phase_disb_idxs, amb_state_idxs)
    bound_high_phfixed = union_of_slices(bound_high_phfix, rxpos_idxs, clock_idxs,  ao_idxs, grav_idxs, trop_idxs, disb_idxs, phase_clock_idxs, phase_disb_idxs, amb_state_idxs)
    bounds = (bound_low_phfixed, bound_high_phfixed)
    ls_phfixed_initial = least_squares(calc_residuals, state_apriori_phfix, jac=jac, method='trf',\
            max_nfev=100, bounds=bounds, verbose=2, x_scale = 'jac', xtol=1e-15,\
            args = ls_args)

    # do one last outlier removal before final solutions
    residuals = ls_phfixed_initial.fun
    #remove_outliers(residuals, baseline_handles)
    if n_unresolved != 0:
        iZt = trim_amb_Zdom(baseline_handles, iZt)
    if store_handle.stochastic_clock is False:
        state_phfix, phase_clock_idxs, phase_disb_idxs, _ = gen_phase_clock_state(store_handle, antenna_handles, baseline_handles,\
                baselines, ref_antenna, ls_phfixed_initial.x, end_range_state, clock_idxs, clock_poly_length, trop_idxs, trop_poly_length, \
                disb_idxs, phase_clock_idxs, phase_disb_idxs, amb_state_idxs)
    else:
        state_phfix = ls_phfixed_initial.x

    bound_low_phfix, bound_high_phfix, state_phfix = set_bounds_phase_clock(bound_low_phfixed, bound_high_phfixed,\
            CLOCK_BOUND, store_handle, antenna_handles, baseline_handles, baselines, ref_antenna, state_phfix, end_range_state, \
            clock_idxs, clock_poly_length, trop_idxs, disb_idxs, phase_clock_idxs, phase_disb_idxs, n_ao_state, n_grav_state, amb_state_idxs)

    print('Executing final PR+CP fixed ambiguity solution:')
    if n_unresolved == 0:
        ls_phfixed_final = least_squares(calc_residuals, state_phfix, jac=jac, method='trf',\
                max_nfev=100, bounds=(bound_low_phfix, bound_high_phfix), verbose=2, x_scale = 'jac', xtol=1e-15,\
                args=(ref_antenna, baselines, store_handle, antenna_handles, clock_idxs, \
                       clock_poly_length, trop_idxs, trop_poly_length, disb_idxs, baseline_handles, phase_delay, phase_only,\
                       use_amb_state, amb_state_idxs, use_phase_weights, tikhonov_lambda, cont_penalty, phase_clock_idxs, \
                       phase_disb_idxs, combination_type, integer_amb))
    else:
        ls_phfixed_final = least_squares(calc_residuals, state_phfix, jac=jac, method='trf',\
                max_nfev=100, bounds=(bound_low_phfix, bound_high_phfix), verbose=2, x_scale = 'jac', xtol=1e-15,\
                args=(ref_antenna, baselines, store_handle, antenna_handles, clock_idxs, \
                       clock_poly_length, trop_idxs, trop_poly_length, disb_idxs, baseline_handles, phase_delay, phase_only, \
                       use_amb_state, amb_state_idxs, use_phase_weights, tikhonov_lambda, cont_penalty, phase_clock_idxs, \
                       phase_disb_idxs, combination_type, z_par, iZt))

    sol_name='final_combined'
    analyze_ls_solution(sol_type, True, ref_antenna, clock_idxs, trop_idxs, disb_idxs, ls_phfixed_final, store_handle, antenna_handles, sol_name, baselines,\
                        n_ao_state, baseline_handles, phase_delay, phase_only, phase_clock_idxs, phase_disb_idxs, use_phase_weights, integer_amb)
    write_SINEX(sol_type, sol_name, ls_phfixed_final, store_handle, antenna_handles, ref_antenna)

    # extract ambiguities for comparison to simulated data
    int_amb_compare = -amb_int + 0.0
    n_amb = 0 
    n_samples = 0
    for baseline_handle in baseline_handles:
        slip_slices_arr = baseline_handle.slip_slices_arr
        for idx, slip_slice in enumerate(slip_slices_arr): 
            slip_slice = slip_slice + n_samples
            int_amb_compare[slip_slice] -= integer_amb[idx+n_amb]
        if sol_type == 'GNSS':
            n_samples += len(baseline_handle.cp_diff)
        elif sol_type == 'VLBI':
            n_samples += len(baseline_handle.phase_delays)
        n_amb += baseline_handle.n_amb_state

    return int_amb_compare, baseline_handles


if __name__ == '__main__':    
    ### Parse command-line options
    parser = argparse.ArgumentParser()
    add_args_to_parser(parser)
    args = parser.parse_args()
        
    if args.ref_antenna is None: raise ValueError('Must choose a reference antenna with --ref_antenna')

    if args.L4R_file is not None:
        l4r_names = [l4r_name for l4r_name in args.L4R_names]
        l4r_data = xr.open_dataset(args.L4R_file, engine='netcdf4')

    # are we processing GNSS data (pseudoranges, carrier phases) or VLBI data (group delay, phase delay)?
    if args.rinex_files is not None:
        sol_type = 'GNSS' # we will be dealing with pseudoranges and carrier phases
        rinex_files = [rinex_file[0] for rinex_file in args.rinex_files]
    else:
        sol_type = 'VLBI'
    if args.src_type == 'GNSS':
        # Note the need to specify args.eph_files[0], as ArgParse wraps the list in a list.
        if args.eph_files is not None:
            #eph_files = [item for sublist in args.eph_files for item in sublist]
            eph_files = [eph_file for eph_file in args.eph_files]
        else:
            raise InsufficientDataError("Invalid run configuration: No ephemeris files specified")
        # initialize Nav Store
        nav_store = NavStore(eph_files)
        
        ## temp
        #clk_file = eph_files[1]
        #if nav_store.ndf.addDataSource(clk_file[0]) is False:
        #    raise Exception('Failed to load RINEX clock file')

        eph_sats = nav_store.get_sat_ids(sat_system=1) # GPS
        eph_sats.extend(nav_store.get_sat_ids(sat_system=2)) # Galileo 
        #eph_sats.extend(nav_store.get_sat_ids(sat_system=3)) # GLONASS 
        eph_sats.extend(nav_store.get_sat_ids(sat_system=7)) # Beidou 
   
        rinex_sats = []
        for sat in eph_sats:
           system = sat.system.name
           if system == 'GPS':
               system_code = 'G'
               #continue
           elif system == 'Galileo':
               system_code = 'E'
               #continue
           elif system == 'BeiDou':
               system_code = 'C'
               #continue
           

           sv_id = str(sat.id)
           if len(sv_id) == 1: sv_id = '0' + sv_id
           rinex_id = system_code + sv_id

           if args.L4R_file is not None:
               # restrict to only satellites with data in the dataset
               if rinex_id not in l4r_data.sv.values:
                   continue
           
           rinex_sats.append(rinex_id)

    elif args.src_type == 'VLBI':
        if args.fringe_file is not None:
            src_dict = read_src(args.src_file)
        nav_store = None
    else:
        raise ValueError('Unknown solution type (--src_type): '+args.src_type)

    antenna_names = [antenna_name for antenna_name in args.antenna_names]
    antenna_types = [antenna_type for antenna_type in args.antenna_types]
    if args.trop_vmf3 == True:
        vmf3_stations = [vmf3_station for vmf3_station in args.vmf3_stations]
    if sol_type == 'VLBI' or args.rxpos is not None:
        if args.rxpos is None and args.vgosDB is None and args.vda_file is None:
            raise ValueError('Must supply receiver positions for VLBI solution (--rxpos)')
        if args.rxpos is not None:
            rxpos_all = []
            for rxpos_arg in args.rxpos:
                rxpos = [float(pos_comp) for pos_comp in rxpos_arg[0].split()]
                rxpos_all.append(rxpos)

    if args.ant_offset is not None:
        offset_all = []
        for offset_ant in args.ant_offset:
            offset = [float(pos_comp) for pos_comp in offset_ant[0].split()]
            offset_all.append(offset)

    vlbi_antennas = [vlbi_antenna for vlbi_antenna in args.vlbi_antennas]

    if len(args.sta_codes)==len(args.antenna_names) and len(args.domes_names)==len(args.antenna_names):
        sta_codes = [sta_code for sta_code in args.sta_codes]
        domes_names = [domes_name for domes_name in args.domes_names]

    ao_antennas = [ao_antenna for ao_antenna in args.estimate_ao]
    grav_def_antennas = [grav_def_antenna for grav_def_antenna in args.estimate_grav_def]
    cable_cal_antennas = [cable_cal_antenna for cable_cal_antenna in args.cable_cal_antennas]
    cable_cal_files = [cable_cal_file for cable_cal_file in args.cable_cal_files]
    weather_cal_antennas = [weather_cal_antenna for weather_cal_antenna in args.weather_cal_antennas]
    weather_cal_files = [weather_cal_file for weather_cal_file in args.weather_cal_files]
    zwd_antennas = [zwd_antenna for zwd_antenna in args.ZWD_antennas]
    zwd_files = [zwd_file for zwd_file in args.ZWD_files]
    ppp_clock_antennas = [ppp_clock_antenna for ppp_clock_antenna in args.ppp_clock_antennas]
    ppp_clock_files = [ppp_clock_file for ppp_clock_file in args.ppp_clock_files]
    ant_offset_names = [ant_offset_name for ant_offset_name in args.ant_offset_names]
    weather_cal_files = [weather_cal_file for weather_cal_file in args.weather_cal_files]

    if len(ao_antennas) > 0: 
        estimate_ao = True
    else:
        estimate_ao = False
    axis_offsets = [axis_offset for axis_offset in args.axis_offsets]
    if args.ref_antenna not in antenna_names:
        raise ValueError('reference antenna ' + args.ref_antenna +' not in antenna names')
   
    if args.iono_free:
        if args.iono_freq == 'L2':
            print('Using iono-free combination with L1/L2')
        elif args.iono_freq == 'L5':
            print('Using iono-free combination with L1/L5')
        else:
            raise ValueError('Unknown carrier frequency ' + args.iono_freq)

    if args.do_mcmc_correlation or args.do_ls_vce_correlation and (args.load_covariance_kernel_range is not None or \
            args.load_covariance_kernel_phase is not None):
        raise ValueError('Cannot estimate covariance when covariance kernel is supplied')

    baselines = list(itertools.combinations(range(len(antenna_names)), 2))

    duration_arr = []
    if sol_type == 'GNSS':
        # get data from rinex
        full_data = {}
        for rinex_file in rinex_files:
            rinex_name = rinex_file.split('.')
            if rinex_name[1][-1] == 'o' or rinex_name[1] =='rnx': 
                rinex_data_full = load(rinex_file)
                #rinex_data_full.to_netcdf(rinex_name[0]+'.nc')
            elif rinex_name[1] == 'nc':
                rinex_data_full = xr.open_dataset(rinex_file)
            else:
                raise ValueError('Unknown file extension ' + rinex_name[1])
            
            # use only satellites in ephemeris -- otherwise will cause GNSSTK errors
            started=0
            for sat in rinex_sats[1:]:
                try:
                    rinex_data_sat = rinex_data_full.sel(sv=sat)
                    if started == 0:
                        rinex_data = rinex_data_sat
                        started=1
                    else:
                        rinex_data = xr.concat((rinex_data, rinex_data_sat), dim='sv')
                except: 
                    continue # SV is in ephemeris but not data
            # if needed, map RINEX3 to RINEX2
            if rinex_data.version >= 3:
                rinex_data = map_datasets(rinex_data) 
            
            full_data[rinex_file] = rinex_data
            
        # parse or generate key file
        if args.key_file is not None:
            datetime_array, source_array, point_ra_dec_array, dt_key, duration_key, source_key, point_key  \
                    = import_key_gnss(rinex_files, full_data, args.key_file)
        else:
            for idx, rinex_file in enumerate(rinex_files):
                rinex_data = full_data[rinex_file]
                time_arr = rinex_data.time.values
                if idx == 0:
                    common_times = time_arr
                else:
                    #common_times = np.union1d(common_times,time_arr)
                    common_times = np.intersect1d(common_times,time_arr)

            if args.begin_experiment is not None:
                common_times = common_times[common_times>=args.begin_experiment]
            if args.end_experiment is not None:
                common_times = common_times[common_times<=args.end_experiment]
            start_date = common_times[0]
            #start_date = (start_date + np.timedelta64(15, 's')).astype('datetime64[30s]').astype('datetime64[ns]')
            end_date = common_times[-1]
            point_ra_dec_array = []
            datetime_array, source_array = gen_key(rinex_files, full_data, start_date, end_date, \
                                                           eph_sats, args.iono_free, args.iono_freq)

        # GNSS data has uniform sampling -- the mode of the time differences will be this interval
        times_sec = (datetime_array-datetime_array[0])/np.timedelta64(1,'s')
        avg_diff = mode(np.diff(times_sec), keepdims=True)[0][0]
        duration_dict = {}
        for time in datetime_array:
            duration_dict[time] = avg_diff
        baseline_handles = [] # we will fill this later for GNSS data
        # thin the RINEX data 
        thinned_data = thin_data(antenna_names, rinex_files, full_data, datetime_array, source_array)

    elif sol_type == 'VLBI':
        # parse key file
        if args.src_type == 'GNSS':
            if args.fringe_file is not None:
                baseline_handles, datetime_array, source_array, point_ra_dec_array, dt_ant, duration_dict = import_data_vlbi(args.fringe_file, \
                        antenna_names, baselines, rinex_sats, args.key_file)
            elif args.nc_sim_file is not None:
                baseline_handles, datetime_array, source_array, point_ra_dec_array, dt_ant = import_data_nc_sim(args.nc_sim_file, \
                         antenna_names, baselines, args.key_file)
        elif args.src_type == 'VLBI':
            if args.fringe_file is not None:
                baseline_handles, datetime_array, source_array, point_ra_dec_array, dt_ant, duration_dict = import_data_vlbi_farfield(args.fringe_file, antenna_names, baselines, src_dict)
            elif args.ngs_file is not None:
                if args.utc2gps is None:
                    raise ValueError('UTC2GPS cannot be none when reading NGS file')
                # DEPRECATED
                baseline_handles, datetime_array, source_array, point_ra_dec_array, dt_ant, duration_dict = import_data_vlbi_ngs(args.ngs_file, antenna_names, baselines, args.utc2gps)
            elif args.vda_file is not None:
                if args.band is None:
                    raise ValueError('Must select a band when reading vgosDA')
                baseline_handles, datetime_array, source_array, point_ra_dec_array, dt_ant, rxpos_all_db, duration_dict, baseline_observations = \
                        import_data_vlbi_vda(args.vda_file, antenna_names, baselines, args.band)
                if args.rxpos is None:
                    rxpos_all = rxpos_all_db
            elif args.vgosDB is not None:
                if args.band is None:
                    raise ValueError('Must select a band when reading vgosDB')
                if args.utc2gps is not None:
                    print('argument utc2gps has no effect with vgosDB data input (offset is in DB)')
                baseline_handles, datetime_array, source_array, point_ra_dec_array, dt_ant, rxpos_all_db, duration_dict = import_data_vlbi_vgosdb(args.vgosDB, antenna_names, baselines, args.band)
                if args.rxpos is None:
                    rxpos_all = rxpos_all_db
            elif args.nc_sim_file is not None:
                baseline_handles, datetime_array, source_array, point_ra_dec_array, dt_ant = import_data_nc_sim(args.nc_sim_file, \
                        antenna_names, baselines, args.key_file)

    # get antenna data from antex file
    antenna_store = AntennaStore()
    time_beg = date_to_common(datetime_array[0], 'GPS')
    if args.src_type == 'GNSS':
         antenna_store.includeAllSatellites()
         antenna_store.addANTEXfile(args.antex_file, time_beg) # define time b/c sat antennas defined by epoch
   
    # initialize solar system model
    sol_sys = SolarSystem()
    sol_sys.initializeWithBinaryFile(args.SSEfile)
    sol_sys.addFile(args.earthfile)

    # trim EOP store for input date/time 
    mjd_beg = datetime64_to_mjd(datetime_array[0]) 
    mjd_end = datetime64_to_mjd(datetime_array[-1]) 
    sol_sys.edit(int(mjd_beg-5), int(mjd_end+5))

    # have to use std_vector_string b/c typemap for list of strings to std_vec<std_string> isnt working
    ant_names_cpp = std_vector_string(antenna_names)

    # initialize Ocean Loading and Atmospheric Loading
    if args.oceanfile is not None:
        ocean_store = OceanLoadTides()
        ocean_store.initializeSites(ant_names_cpp, args.oceanfile)
    else:
        ocean_store = None
    if args.atmfile is not None:
        atm_store = AtmLoadTides()
        atm_store.initializeSites(ant_names_cpp, args.atmfile)
    else:
        atm_store = None

    if args.global_linear_clock and args.global_quadratic_clock:
        raise ValueError('Must choose --global_linear_clock or --global_quadratic clock (not both)')
     
    # initialize the stores handle
    store_handle = GNSSTKStores(sol_type, args.src_type, sol_sys, antenna_store, ocean_store, atm_store, nav_store, \
            args.iono_free, args.iono_freq, args.analytical_delay, args.stochastic_clock, args.stochastic_trop, args.global_linear_clock,\
            args.global_quadratic_clock, args.estimate_disb, args.estimate_phase_disb)
    if args.trop_poly_length > 0:
        store_handle.estimate_trop = True

    if args.src_type == 'GNSS':
        store_handle.build_antenna_map(source_array, datetime_array)

    store_handle.hold_source_array(source_array, datetime_array, duration_dict)
    store_handle.save_exp_weather(args.trop_T, args.trop_P, args.trop_H)

    if args.L4R_file is not None:
        store_handle.build_l4r_model(l4r_data)

    # initialize the antenna handles
    antenna_handles = []
    for antenna_idx, antenna_name in enumerate(antenna_names):
        antenna_type = antenna_types[antenna_idx]
        if sol_type == 'GNSS':
            antenna_data = thinned_data[antenna_name]
        if sol_type == 'GNSS' and args.rxpos is None:
            if args.rxpos is None:
                antenna_position = antenna_data.position 
        else:
            antenna_position = rxpos_all[antenna_idx]
        bulk_clock = args.bulk_clocks[antenna_idx][0]

        if args.trop_vmf3 or args.trop_global or args.trop_saas or args.trop_neill:
            tk_pos = Position(antenna_position[0], antenna_position[1], antenna_position[2])
            if args.trop_vmf3:
                vmf3_station = vmf3_stations[antenna_idx]
                tropModel = VMF3Model('station', vmf3_station)
                tropModel.load_station_vmf3_files(args.vmf3_files, args.ell_file)
                if len(args.vmf3_grad_files)>0:
                    tropModel.load_station_grad_files(args.vmf3_grad_files)
                #from gnsstk import WGS84Ellipsoid
                #ell_model = WGS84Ellipsoid()
                #pos_geodetic = tk_pos.asGeodetic(ell_model)
                #lat = pos_geodetic[0]
                #lon = pos_geodetic[1]
                #ht_el = pos_geodetic[2]
                #MJD = 60786.5 
                #elev = 5 
                #Gn_h, Ge_h, Gn_w, Ge_w = tropModel.interpolate_station_grad(MJD)
                #mfh, mfw, zhd, zwd = tropModel.interpolate_station_vmf3(lat, lon, ht_el, MJD, elev)
                #tropModel.load_v3gr_files(args.v3gr_files, args.orog_file)
                #Gn_h2, Ge_h2, Gn_w2, Ge_w2 = tropModel.interpolate_grad(lat, lon, MJD)
                #mfh2, mfw2, zhd2, zwd2 = tropModel.interpolate_vmf3(lat, lon, ht_el, MJD, elev)
            elif args.trop_v3gr:
                tropModel = VMF3Model('V3GR')
                tropModel.load_v3gr_files(args.v3gr_files, args.orog_file)
                # test against Matlab
                #from gnsstk import WGS84Ellipsoid
                #ell_model = WGS84Ellipsoid()
                #pos_geodetic = tk_pos.asGeodetic(ell_model)
                #lat = pos_geodetic[0]
                #lon = pos_geodetic[1]
                #ht_el = pos_geodetic[2]
                #MJD = 60686.7083 
                #elev = 60 
                #Gn_h, Ge_h, Gn_w, Ge_w = tropModel.interpolate_grad(lat, lon, MJD)
                #mfh, mfw, zhd, zwd = tropModel.interpolate_vmf3(lat, lon, ht_el, MJD, elev)

            elif args.trop_global:
                tropModel = GlobalTropModel(tk_pos, time_beg)
                if args.trop_H is not None:
                    tropModel.setHumidity(args.trop_H)
                else:
                    tropModel.setHumidity(50) # 50 percent is standard assumed humidity
            elif args.trop_saas:
                lat = tk_pos.geodeticLatitude()
                dt_datetime = datetime_array[0].astype('M8[D]').astype('O')
                day_of_year = dt_datetime.timetuple().tm_yday
                tropModel = SaasTropModel(lat, day_of_year)
                if args.trop_T is not None and args.trop_P is not None and args.trop_H is not None:
                    tropModel.setWeather(args.trop_T, args.trop_P, args.trop_H)
            elif args.trop_neill:
                lat = tk_pos.geodeticLatitude()
                ht = tk_pos.height()
                dt_datetime = datetime_array[0].astype('M8[D]').astype('O')
                day_of_year = dt_datetime.timetuple().tm_yday
                #tropModel = NeillTropModel(ht, lat, day_of_year)
                tropModel = NeillTropModel()
                tropModel.setReceiverLatitude(lat)
                tropModel.setReceiverHeight(ht)
                tropModel.setDayOfYear(time_beg)

            antenna_handle = AntennaInfo(antenna_name, antenna_position, antenna_type, bulk_clock, 0, args.dither_phase, tropModel) 
        else:
            antenna_handle = AntennaInfo(antenna_name, antenna_position, antenna_type, bulk_clock, 0, args.dither_phase) 

        if antenna_name == args.ref_antenna:
            # remove stochastic model PSD scalings for reference antenna
            antenna_handle.clock_psd_rw = 0
            antenna_handle.clock_psd_irw = 0
            antenna_handle.trop_psd_rw = 0

        if len(args.sta_codes)==len(args.antenna_names) and len(args.domes_names)==len(args.antenna_names):
            sta_code = sta_codes[antenna_idx]
            domes_name = domes_names[antenna_idx]
            antenna_handle.set_sinex_names(sta_code, domes_name)

        if antenna_name in weather_cal_antennas:
            idx_weather = [i for i, val in enumerate(weather_cal_antennas) if val == antenna_name][0]
            antenna_handle.load_weather_file(weather_cal_files[idx_weather])

        if antenna_name in cable_cal_antennas:
            idx_cable = [i for i, val in enumerate(cable_cal_antennas) if val == antenna_name][0]
            antenna_handle.load_cable_cal_file(cable_cal_files[idx_cable])


        if antenna_name in zwd_antennas:
            idx_zwd = [i for i, val in enumerate(zwd_antennas) if val == antenna_name][0]
            antenna_handle.load_zwd_file(zwd_files[idx_zwd])
        
        if antenna_name in ant_offset_names:
            idx_offset = [i for i, val in enumerate(ant_offset_names) if val == antenna_name][0]
            antenna_offset = offset_all[idx_offset]
            antenna_handle.save_offset(antenna_offset)

        if len(args.linked_clocks)>0:
            # currently a stub
            for clock_link in args.linked_clocks:
                ant_args = clock_link.split('--')
                if antenna_name in ant_args:
                    if antenna_name == ant_args[0]:
                        antenna_handle.linked_clocks.append(ant_args[1])
                    else:
                        antenna_handle.linked_clocks.append(ant_args[0])

        if antenna_name in vlbi_antennas:
            # phase center is handled by geometric calculation, no PCO
            idx_vlbi = [i for i, val in enumerate(vlbi_antennas) if val == antenna_name]
            axis_offset = float(np.array(axis_offsets)[idx_vlbi][0])
            if antenna_name in ao_antennas:
                estimate_ao_ant = True
            else:
                estimate_ao_ant = False
            if antenna_name in grav_def_antennas:
                estimate_grav_def = True
            else:
                estimate_grav_def = False


            if estimate_ao_ant and estimate_grav_def:
                raise ValueError('Cannot estimate both axis offset and gravitational deformation (cos(el) term in both)')
            antenna_handle.set_VLBI(axis_offset, point_ra_dec_array, datetime_array, estimate_ao_ant, estimate_grav_def)
        else:
            # antenna phase center variation
            antennaPCOData = AntexData()
            store_handle.antenna_store.getAntenna(antenna_handle.antenna_type, antennaPCOData) # memory leak, needs investigation
            antenna_handle.hold_PCO(antennaPCOData)      
            if antennaPCOData.nFreq <= 4 and args.iono_freq == 'L5':
                print('No L5 data in ANTEX file -- using L2 mapping (danger)')
        if sol_type == 'GNSS':
            antenna_handle.hold_data(antenna_data)
            if store_handle.estimate_disb:
                antenna_handle.hold_disb(np.zeros(len(store_handle.systems)))
            if store_handle.estimate_phase_disb:
                antenna_handle.hold_disb(np.zeros(len(store_handle.systems)), phase=True)
        elif sol_type == 'VLBI':
            times_antenna = dt_ant[antenna_name]
            antenna_handle.hold_times(times_antenna)
            antenna_handle.set_phase_clock()

        if store_handle.iono_comp_l4r:
            if antenna_name in vlbi_antennas:
                antenna_handle.set_l4r_name(np.array(l4r_names)[idx_vlbi][0])
            else:
                antenna_handle.set_l4r_name(antenna_handle.antenna_name)

        if antenna_name in ppp_clock_antennas:
            idx_clock = [i for i, val in enumerate(ppp_clock_antennas) if val == antenna_name][0]
            antenna_handle.load_ppp_clock(ppp_clock_files[idx_clock])
        if antenna_handle.ppp_clock_active: antenna_handle.interp_ppp_clock()
        antenna_handles.append(antenna_handle)

    # build the IONEX ionosphere store
    # build the OBX satellite attitude store
    if args.ionex_files is not None:
        if args.iono_free:
            raise Exception('IONEX file should not be used with ionosphere free combination')
        ionex_files = [item for sublist in args.ionex_files for item in sublist]
        store_handle.build_ionex_store(ionex_files)

    # build the OBX satellite attitude store
    if args.OBX_file is not None:
        if args.src_type=='VLBI': 
            print('OBX file ' + args.OBX_file + ' will not be used (VLBI solution)')
        else:
            store_handle.build_obx_store(args.OBX_file)

    if args.ant_info_file is not None:
        read_thermal_deformation_coeffs(args.ant_info_file, antenna_handles)
        print('Read antenna thermal deformation model. Need either vgosDB or temperature with --trop_T to use')

    clock_poly_length = args.clock_poly_length * 60 # convert to seconds
    trop_poly_length = args.trop_poly_length * 60 # convert to seconds
    phase_ambiguities, baseline_handles = lstsq_estimation(sol_type, args.plot_intermediate_results, args.ref_antenna, \
                     store_handle, antenna_handles, baselines, baseline_handles,\
                     clock_poly_length, trop_poly_length, args.clock_file, estimate_ao,\
                     args.analytical_Jac, args.tikhonov_reg, args.continuity_penalty, args.recursive_amb, args.do_mcmc_correlation,\
                     args.do_ls_vce_correlation, args.load_covariance_kernel_range, args.load_covariance_kernel_phase, args.igs_data, args.baseline_strategy)

    if args.vda_file is not None and args.output_vda_file is not None:
        write_vda_phase(baseline_observations, args.vda_file, args.output_vda_file, phase_ambiguities, baseline_handles, baselines, antenna_names)
