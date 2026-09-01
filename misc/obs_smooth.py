#!/usr/bin/python
"""
Smooth MDH obs and reduce cadence to 30 seconds

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

import os
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
import mdhpy as mdh

from gnsstk import std_vector_string, Position, AntennaStore, AntexData, OceanLoadTides, PoleTides, \
                  AtmLoadTides, SolarSystem, GlobalTropModel, SaasTropModel, NeillTropModel

from single_diff_tools import import_key_gnss, import_data_vlbi, import_data_vlbi_farfield, import_data_vlbi_ngs, import_data_vlbi_vgosdb, \
                  import_data_vlbi_vda, write_SINEX, datetime64_to_mjd, map_datasets, import_data_nc_sim,\
                  find_common_epochs, BaselineInfo, AntennaInfo, GNSSTKStores, ECEF2ECI, slip_detect_MW, slip_detect_single_freq,\
                  slip_detect_phase_delay, sample_poly_at_interval, trim_amb_Zdom, trim_amb_state, gen_phase_clock_state, adjust_stoch_params, thin_data,\
                  analyze_ls_solution, resolve_float_amb, construct_float_amb, remove_outliers, iterative_remove_outliers, calc_residuals, \
                  calc_jac, plot_time_units, read_src, date_to_common, date_to_mjd, detect_unresolved_amb_vlbi, \
                  detect_unresolved_amb_gnss, set_bounds_phase_clock, union_of_slices, get_residuals, iterative_weight_adjust, iterative_weight_adjust_ls_vce, \
                  iterative_weight_adjust_LS_VCE_full, sample_global_poly_at_interval, gen_key, \
                  read_thermal_deformation_coeffs, get_obs_weights, find_cont_penalty, mcmc_correlation, ls_correlation, load_kernel_parameters, vlbi_transform_obs, NavStore

def add_args_to_parser(parser_in):
    """Add arguments to parser."""
    parser.add_argument("--obs_file", help="MDH OBS data")
    parser.add_argument("--obs_file_out", help="MDH OBS output data file")
    parser.add_argument("-e", dest="eph_files", action="append", type=str, nargs="+", help = 'Ephemeris file. Add files for day before and after experiment too.')
    parser.add_argument("--rxpos", action="append", type=str, nargs="+", help="Receiver position  as 'X Y Z' (m)")
    parser.add_argument("--OBX_file", default=None, action="extend", nargs="+",
                         help="Name of high-rate IGS OBX file describing GNSS satellite orientations."
                         )
    parser.add_argument("--bulk_clock", type=float, 
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
    parser.add_argument("--vlbi_antenna", action="store_true",
                         help="Usage: --vlbi_antenna  Designate the antenna as VLBI. Will have to supply axis offset")
    parser.add_argument("--cable_cal_file",
                         help="Usage: --cable_cal_file {FILE_NAME}. Supply a DiFX cable calibration file associated with an antenna")
    parser.add_argument("--ppp_clock_file",
                         help="Usage: --ppp_clock_files {FILE_NAME}. Supply a Precise Point Positioning clock file for the antenna. Clock file is column 1 second of day, column 2 clock value (m).")
    parser.add_argument("--weather_cal_file",
                         help="Usage: --weather_cal_file {FILE_NAME}. Supply a DiFX weather calibration file")
    parser.add_argument("--axis_offset",
                         help="VLBI antenna axis offset in meters.",
                         type=float)
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
    parser.add_argument("--analytical_delay",
                         action="store_true",
                         default=False,
                         help="Use the adapted Jaron, Nothnagel (2019) analytical delay model (recommended) instead of iterative light time model",
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
                        type=str,
                        help="Name of antenna.")
    parser.add_argument("--nsec_out", 
                        type=float,
                        required=True,
                        help="Number of seconds to sample new obs data at (>=1)")
    parser.add_argument("--antenna_type", 
                        type=str,
                        help="Antenna type (e.g. GNSS, XY-N, XY-E, Az-El, Equa, BWG, Nasmyth)."+\
                              " If GNSS, specify ANTEX type in place of GNSS.")
    parser.add_argument("--iono_freq", type=str, default='L2',
                       help = 'Carrier frequency to use in iono-free combination (L2 or L5). NB: will only use GPS satellites with L2.')
    parser.add_argument("--system", type=str,
                       help = 'GNSS system to smooth if desired (ignore others), options: Galileo, GPS, BeiDou')
    parser.add_argument("--prn", type=str,
                       help = 'PRN to smooth (ignore others)')
    parser.add_argument(
        "--begin_obs",
        metavar="YYYY-MM-DDThh:mm[:ss]",
        type=np_datetime64,
        required=True,             
        help="ISO-8601 start time (e.g. 2025-07-31T08:00). Used for beginning of EOP range"
    )
    parser.add_argument(
        "--end_obs",
        metavar="YYYY-MM-DDThh:mm[:ss]",
        type=np_datetime64,
        required=True,
        help="ISO-8601 end time (e.g. 2025-07-31T18:00). Used for end of EOP range"
    )

freq_dict={
    1: 1575.42e6,
    2: 1227.60e6,
    5: 1176.45e6,
    8: 1561.098e6,
    10: 1176.45e6,
    11: 1207.14e6,
    29: 1176.45e6,
    30: 1207.14e6
}
system_dict={
    7: 'G',
    9: 'C',
    10: 'E'
}

def get_antenna_data_slow(dataset, times_gps):
    """ Convert an MDH datset to an xarray usable in single_diff_tools routines  """
    freq = freq_dict[dataset.attrs['carrierCode']]
    wavelength = const.c/freq
    idx_DMS = np.argwhere(np.array(dataset.dtype.names)=='demodulatorStatus')[0][0]
    idx_PR = np.argwhere(np.array(dataset.dtype.names)=='pseudorange')[0][0]
    idx_ADR = np.argwhere(np.array(dataset.dtype.names)=='accumulatedDeltaRange')[0][0]
    times_gps_corr = []
    if dataset.attrs['prnCode'] < 10:
        prn = '0' + str(dataset.attrs['prnCode'])
    else:
        prn = str(dataset.attrs['prnCode'])
    src = system_dict[dataset.attrs['systemCode']] + prn # RINEX3 ID
    obs_full = []
    src_time_dict = {}
    for idx, time in enumerate(times_gps):
        dataset_idx = list(dataset[idx])
        DMS = dataset_idx[idx_DMS]
        if DMS != 2:
            # only want CCL data
            continue
        pr_meas = dataset_idx[idx_PR]
        ADR_meas = dataset_idx[idx_ADR]
        obs_dict = {}
        obs_dict['pr_data'] = pr_meas
        obs_dict['cp_data'] = ADR_meas*wavelength

        # Create the new dataset 
        data_dict = {var_name: (["time"], value) for var_name, value in obs_dict.items()}
        obs_xarray = xr.Dataset(
                    {k: (("time",), np.atleast_1d(v)) for k, v in obs_dict.items()},
                        coords={"time": np.atleast_1d(time)},)
        obs_xarray = obs_xarray.assign_coords(sv=src)
        #obs_xarray.attrs = obs_time.attrs
 

        if len(obs_full)==0:
            obs_full = obs_xarray
        else:
            obs_full = xr.concat((obs_full, obs_xarray), dim='time')
        times_gps_corr.append(times_gps[idx])
        src_time_dict[times_gps[idx]] = src

    times_gps_corr = np.array(times_gps_corr)

    return obs_full, times_gps_corr, freq, src_time_dict

def get_antenna_data(dataset, times_gps, nsec):
    """ Convert an MDH datset to an xarray usable in single_diff_tools routines  """
    # --- constants / ids ---
    freq = freq_dict[dataset.attrs['carrierCode']]
    wavelength = const.c / freq
    prn = f"{int(dataset.attrs['prnCode']):02d}"
    src = system_dict[dataset.attrs['systemCode']] + prn  # RINEX 3 ID

    # 1) Filter rows: only DMS == 2 (CCL)
    dms = dataset['demodulatorStatus'][...]          # 1D array
    sel_idx = np.nonzero(dms == 2)[0]                # integer indices

    # 2) Use the selected indices to fetch the needed fields
    pr = dataset['pseudorange'][sel_idx]
    adr = dataset['accumulatedDeltaRange'][sel_idx]

    # 3) Times
    times_gps_corr = times_gps[sel_idx]

    # 4) Compute carrier-phase (meters)
    cp = adr * wavelength
    if dataset.attrs['cadence']/dataset.attrs['timeDenominator'] < 0.1 and nsec>1:
        # 50 Hz data -- decimate to 1 Hz
        pr = pr[::50]
        cp = cp[::50]
        times_gps_corr = times_gps_corr[::50]

    # --- build xarray in one go ---
    obs_full = xr.Dataset(
        data_vars={
            "pr_data": ("time", pr),
            "cp_data": ("time", cp),
        },
        coords={"time": times_gps_corr},
    ).assign_coords(sv=src)

    src_time_dict = {t: src for t in times_gps_corr}

    return obs_full, times_gps_corr, freq, src_time_dict

def extract_antenna_data(antenna_handle, mdh_out_file, times_gps_dset, dataset, freq, nsec_out):
    """ Convert an xarray back to an MDH dataset """
    wavelength = const.c/freq
    dset_out = mdh_out_file.create_dataset(dataset.name, shape=(len(antenna_handle.antenna_data.pr_data.values),), dtype=dataset.id.get_type(), compression='lzf')
    idx_PR = np.argwhere(np.array(dataset.dtype.names)=='pseudorange')[0][0]
    idx_ADR = np.argwhere(np.array(dataset.dtype.names)=='accumulatedDeltaRange')[0][0]
    for attr in dataset.attrs.keys():
        h5t = dataset.attrs.get_id(attr).get_type()
        if attr != "timeDenominator" and attr != "cadence" and attr != "startTime":
            #dset_out.attrs.create(attr, dataset.attrs[attr])              
            val = dataset.attrs[attr]           
            dset_out.attrs.create(attr, val, dtype=h5t)
        elif attr == "timeDenominator":
            dset_out.attrs.create("timeDenominator", 50, dtype=h5t)
        elif attr == "cadence":
            dset_out.attrs.create("cadence", int(nsec_out)*50, dtype=h5t)
        else:
            gps_epoch = np.datetime64("1980-01-06T00:00:00", "ns")
            sec_since_gps = (antenna_handle.times_gps[0] - gps_epoch) / np.timedelta64(1, "s")
            dset_out.attrs.create("startTime", int(sec_since_gps*50), dtype=h5t)
    _, _, idxs_dset = np.intersect1d(antenna_handle.times_gps, times_gps_dset, return_indices=True)

    for name in dataset.dtype.names:
        if name != "pseudorange" and name != "accumulatedDeltaRange":
            dset_out[name] = dataset[name][idxs_dset]

    buf = np.empty(dset_out.shape, dtype=dset_out.dtype)
    buf[...] = dset_out[...]
    buf['pseudorange'] = antenna_handle.antenna_data.pr_data.values
    buf['accumulatedDeltaRange'] =  antenna_handle.antenna_data.cp_data.values/wavelength
    buf['demodulatorStatus'] = 2 * np.bitwise_and(np.abs(antenna_handle.antenna_data.cp_data.values)>0, buf['demodulatorStatus']>0)
    dset_out[...] = buf

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

if __name__ == '__main__':    
    ### Parse command-line options
    parser = argparse.ArgumentParser()
    add_args_to_parser(parser)
    args = parser.parse_args()

    # are we processing GNSS data (pseudoranges, carrier phases) or VLBI data (group delay, phase delay)?
    sol_type = 'GNSS' # we will be dealing with pseudoranges and carrier phases

    # Note the need to specify args.eph_files[0], as ArgParse wraps the list in a list.
    if args.eph_files is not None:
        #eph_files = [item for sublist in args.eph_files for item in sublist]
        eph_files = [eph_file for eph_file in args.eph_files]
    else:
        raise InsufficientDataError("Invalid run configuration: No ephemeris files specified")
    # initialize Nav Store
    nav_store = NavStore(eph_files)
    
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
       rinex_sats.append(rinex_id)

    # get antenna data from antex file
    antenna_store = AntennaStore()
    antenna_store.includeAllSatellites()
    time_beg = date_to_common(args.begin_obs, 'GPS')
    antenna_store.addANTEXfile(args.antex_file, time_beg) # define time b/c sat antennas defined by epoch
   
    # initialize solar system model
    sol_sys = SolarSystem()
    sol_sys.initializeWithBinaryFile(args.SSEfile)
    sol_sys.addFile(args.earthfile)

    # trim EOP store for input date/time 
    mjd_beg = datetime64_to_mjd(args.begin_obs) 
    mjd_end = datetime64_to_mjd(args.end_obs) 
    sol_sys.edit(int(mjd_beg-5), int(mjd_end+5))

    # have to use std_vector_string b/c typemap for list of strings to std_vec<std_string> isnt working
    ant_names_cpp = std_vector_string([args.antenna_name])

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

    # initialize the stores handle
    store_handle = GNSSTKStores(sol_type, 'GNSS', sol_sys, antenna_store, ocean_store, atm_store, nav_store, False)
    store_handle.build_antenna_map(rinex_sats, [])
    store_handle.source_array = rinex_sats

    #store_handle.hold_source_array(source_array, datetime_array, duration_dict)
    store_handle.save_exp_weather(args.trop_T, args.trop_P, args.trop_H)

    # initialize the antenna handles
    antenna_position = [float(pos_comp) for pos_comp in args.rxpos[0][0].split()]

    if args.trop_global is True or args.trop_saas is True or args.trop_neill is True:
        tk_pos = Position(antenna_position[0], antenna_position[1], antenna_position[2])
        if args.trop_global is True:
            tropModel = GlobalTropModel(tk_pos, time_beg)
            if args.trop_H is not None:
                tropModel.setHumidity(args.trop_H)
            else:
                tropModel.setHumidity(50) # 50 percent is standard assumed humidity
        elif args.trop_saas is True:
            lat = tk_pos.geodeticLatitude()
            dt_datetime = datetime_array[0].astype('M8[D]').astype('O')
            day_of_year = dt_datetime.timetuple().tm_yday
            tropModel = SaasTropModel(lat, day_of_year)
            if args.trop_T is not None and args.trop_P is not None and args.trop_H is not None:
                tropModel.setWeather(args.trop_T, args.trop_P, args.trop_H)
        elif args.trop_neill is True:
            lat = tk_pos.geodeticLatitude()
            ht = tk_pos.height()
            dt_datetime = datetime_array[0].astype('M8[D]').astype('O')
            day_of_year = dt_datetime.timetuple().tm_yday
            tropModel = NeillTropModel()
            tropModel.setReceiverLatitude(lat)
            tropModel.setReceiverHeight(ht)
            tropModel.setDayOfYear(time_beg)

        antenna_handle = AntennaInfo(args.antenna_name, antenna_position, args.antenna_type, args.bulk_clock, False, tropModel) 
    else:
        antenna_handle = AntennaInfo(args.antenna_name, antenna_position, args.antenna_type, args.bulk_clock, False) 

    if args.weather_cal_file is not None:
        antenna_handle.load_weather_file(args.weather_cal_file)

    if args.cable_cal_file is not None:
        antenna_handle.load_cable_cal_file(args.cable_cal_file)
    
    if args.vlbi_antenna is True:
        axis_offset = args.axis_offset
        antenna_handle.set_VLBI(axis_offset, [], [], False, False)
    else:
        # antenna phase center variation
        antennaPCOData = AntexData()
        store_handle.antenna_store.getAntenna(antenna_handle.antenna_type, antennaPCOData) # memory leak, needs investigation
        antenna_handle.hold_PCO(antennaPCOData)      

    if args.ppp_clock_file is not None:
        antenna_handle.load_ppp_clock(args.ppp_clock_file)

    # build the OBX satellite attitude store
    if args.OBX_file is not None:
        store_handle.build_obx_store(args.OBX_file)

    if args.ant_info_file is not None:
        read_thermal_deformation_coeffs(args.ant_info_file, [antenna_handle])
        print('Read antenna thermal deformation model. Need either vgosDB or temperature with --trop_T to use')

    
    with mdh.MDHFile(args.obs_file, 'r') as mdh_file:
        with mdh.MDHFile(args.obs_file_out, 'w') as mdh_out_file:
            for idx, dataset in enumerate(mdh_file.matchingDatasets(SUBCLASS="OBSERVATIONS")):
                #if "Track_OBSERVATIONS_0000_PRN11_L1_E1C_2363_153429020" not in dataset.name: continue
                # temp -- neglect unwanted signals
                #if "B5b" in dataset.name or "B1I" in dataset.name or "B2I" in dataset.name or "E5b" in dataset.name \
                #        or "L5" in dataset.name: continue
                if args.system is not None:
                    if args.system =='GPS' and dataset.attrs['systemCode'] != 7:
                        continue
                    elif args.system =='Galileo' and dataset.attrs['systemCode'] != 10:
                        continue
                    elif args.system =='BeiDou' and dataset.attrs['systemCode'] != 9:
                        continue
                if args.prn is not None:
                    if int(args.prn)!=dataset.attrs['prnCode']:
                        continue
                print('processing dataset ' + dataset.name)
                times_mdh, time_denominator = mdh.findTimeArrayForDatasets([dataset])
                times_gps_dset = np.array(mdh.convert_time.mdh_to_date(times_mdh, dataset.attrs["timeDenominator"]),dtype='datetime64[ns]')
                antenna_handle.antenna_data, times_gps, freq, src_time_dict = get_antenna_data(dataset, times_gps_dset, args.nsec_out)
                if len(times_gps) < args.nsec_out: 
                    print('too few pts, skipping dataset')
                    continue
                if antenna_handle.antenna_data.sv.values not in rinex_sats:
                    print('satellite not in ephemerides -- skipping')
                    continue
                store_handle.source_time_dict = src_time_dict 
                antenna_handle.times_gps = times_gps
                antenna_handle.clock_times = antenna_handle.times_gps
                antenna_handle.phase_clock_times = antenna_handle.clock_times
                trop_samples = np.zeros(len(antenna_handle.times_gps))
                antenna_handle.hold_trop(trop_samples)

                clock_samples = np.zeros(len(antenna_handle.times_gps)) + antenna_handle.bulk_clock
                if antenna_handle.ppp_clock_active is True:
                    antenna_handle.interp_ppp_clock(times_gps)
                    clock_samples += antenna_handle.ppp_clock_samples
                
                antenna_handle.hold_clock(clock_samples)
                antenna_handle.hold_clock(clock_samples.copy(), phase_delay=True) # ref clock is also ref phase clock

                rxpos_series, R_obj  = store_handle.compute_tides(antenna_handle.times_gps, antenna_handle.ref_pos, antenna_handle.antenna_name) 
                antenna_handle.update_pos_series(rxpos_series, R_obj, antenna_handle.ref_pos)
                #store_handle.compute_azel(antenna_handle.times_gps, antenna_handle)
                
                data_corrected = store_handle.correct_PR_CP(antenna_handle, phase=True, phase_only=False, sim=False, freq=freq)
                antenna_handle.hold_data(data_corrected)

                # convert data to 1 meas per scan by antenna
                vlbi_transform_obs(store_handle, antenna_handle, freq, args.nsec_out)
                if len(antenna_handle.times_gps)>0:
                    extract_antenna_data(antenna_handle, mdh_out_file, times_gps_dset, dataset, freq, args.nsec_out)
