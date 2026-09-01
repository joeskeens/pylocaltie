#!/usr/bin/python
"""
Generate simulated data, either GNSS (pseudorange, carrier phase) or VLBI (group delay, phase delay),
with a key file, or observations from an existing VLBI data format (NGS, VGOSDB, VGOSDA).
Observations can be of either GNSS satellites or natural radio sources.

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
use_custom_version = os.getenv('USE_CUSTOM_GEORINEX', 'false').lower() == 'true'
if use_custom_version:
    import sys
    sys.path.insert(0, '/sgl/ceph/work/jskeens')
    from georinex_custom import load
else:
    from georinex import load

import xarray as xr
import datetime
import itertools
import random
import re
import numpy as np
from scipy.stats import mode
import scipy.constants as const
import argparse
import matplotlib.dates as mdates

from gnsstk import std_vector_string, Position, AntennaStore, AntexData, OceanLoadTides, PoleTides, \
                  AtmLoadTides, SolarSystem, GlobalTropModel, SaasTropModel, NeillTropModel

from single_diff_tools import import_key_gnss, import_data_vlbi, import_data_vlbi_farfield, import_data_vlbi_ngs, import_data_vlbi_vgosdb, \
                  import_data_vlbi_vda, datetime64_to_mjd, map_datasets, BaselineInfo, AntennaInfo, GNSSTKStores, read_src, date_to_common, \
                  generate_correlated_noise, read_thermal_deformation_coeffs, load_kernel_parameters, thin_data, consecutive_idxs, NavStore

def add_args_to_parser(parser_in):
    """Add arguments to parser."""
    #parser.add_argument("--auto_start_date", type=parse_date, default=None, help="For auto measurement generation, start date in the format 'yyyy-mm-dd HH:MM:SS'")
    #parser.add_argument("--auto_end_date", type=parse_date, default=None, help="For auto measurement generation, end date in the format 'yyyy-mm-dd HH:MM:SS'")
    #parser.add_argument("--auto_meas_cadence", type=float, default=None, help="For auto measurement generation, cadence of the output measurements (sec). Default: 30 sec")
    parser.add_argument("-r", dest="rinex_files", action="append", type=str, metavar="FILE", nargs="+")
    parser.add_argument("--mdh_out", default=None, type=str, help = 'name of output MDH obs file')
    parser.add_argument("--nc_out", default=[], action="append", type=str, help = 'Name of output xarray obs file. Give multiple names for multiple antennas with GNSS obs type. ')
    parser.add_argument("-e", dest="eph_files", action="append", type=str, nargs="+", help = 'Ephemeris file. Add files for day before and after experiment too.')
    parser.add_argument("--src_type", default=None, required=True,
                         help="Type of sources observed in the experiment (GNSS or VLBI)"
                         )
    parser.add_argument("--obs_type", default=None, required=True,
                         help="Type of observations--GNSS (pseudorange, carrier phase) or VLBI (group delay, phase delay)"
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
    parser.add_argument("--ngs_file", default=None,
                         help="NGS data file to read. Antenna names supplied with --antenna_name must match those in file"
                         )
    parser.add_argument("--vda_file", default=None,
                         help="vgosDA data file to read. Antenna names supplied with --antenna_name must match those in file"
                         )
    parser.add_argument("--vgosDB", default=None,
                         help="untarred vgosDB directory location to read. Antenna names supplied with --antenna_name must match subdirectory locations"
                         )
    parser.add_argument("--band", default=None,
                         help="If vgosDB, specify band to read."
                         )
    parser.add_argument("--utc2gps", default=None, type=float,
                         help="Number of seconds GPS is ahead of UTC (reqd for ngs_file)"
                         )
    parser.add_argument("--rxpos", action="append", type=str, nargs="+", help="Receiver position  as 'X Y Z' (m)")
    parser.add_argument("--OBX_file", default=None,
                         help="Name of high-rate IGS OBX file describing GNSS satellite orientations."
                         )
    parser.add_argument("--bulk_clock", dest="bulk_clocks", action="append", type=float, nargs="+", 
                         help = 'Bulk clock offsets for the antennas (microseconds), order of antenna input. Convention: positive late (same as diffproc). (required)')
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
    parser.add_argument("--vlbi_antennas",
                         help="Usage: --vlbi_antenna {ANTENNA_NAME}. Designate an antenna as VLBI. Will have to supply axis offset",
                         default=[], 
                         action='append')
    parser.add_argument("--axis_offsets",
                         help="VLBI antenna axis offset in meters. Repeat argument in same order for more than 1 VLBI antenna.",
                         default=[], 
                         action='append')
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
    parser.add_argument("--load_covariance_kernel_range",
                         default=None,
                         help="Covariance kernel file to model a full rank covariance matrix for range (group delay or pseudorange) measurements.",
                         )
    parser.add_argument("--load_covariance_kernel_phase",
                         default=None,
                         help="Covaraince kernel file to model a full rank covariance matrix for phase measurements.",
                         )
    parser.add_argument("--q_range",
                         type=float,
                         help="Covariance multiplier (m) to give proper noise variance for group delay meaurements (VLBI).",
                         )
    parser.add_argument("--q_phase",
                         type=float,
                         help="Covariance multiplier (m) to give proper noise variance for phase delay meaurements (VLBI).",
                         )
    parser.add_argument("--analytical_delay",
                         action="store_true",
                         default=False,
                         help="Use the adapted Jaron, Nothnagel (2019) analytical delay model (recommended) instead of iterative light time model",
                         )
    parser.add_argument("--dither_phase",
                         action="store_true",
                         default=False,
                         help="Add randomized integer wavelength offsets to phase data",
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
    parser.add_argument("--antenna_type", 
                        dest="antenna_types",
                        help="Antenna types of antennas to simulate (e.g. GNSS, XY-N, XY-E, Az-El, Equa, BWG, Nasmyth)."+\
                              " If GNSS, supply ANTEX type rather than GNSS.", 
                        default=[], 
                        action='append')
    parser.add_argument("--ionex_file", default=None,
                         help="Add ionosphere delay with an IONEX model.")
    parser.add_argument("--iono_free", action="store_true", default=False, help = 'Generate a second frequency to be used in ionosphere free combination. CURRENTLY NOT FULLY IMPLEMENTED.')
    parser.add_argument("--iono_freq", type=str, default='L2',
                                       help = 'Carrier frequency to use in iono-free combination (L2 or L5). NB: will only use GPS satellites with L2.')


def simulate_data(nc_out, mdh_out, obs_type, store_handle, antenna_handles, baselines, baseline_handles, \
                     covariance_kernel_range=None, covariance_kernel_phase=None, q_range=0, q_phase=0, dither_phase=False, seed=None):
    """
    Take the single-source data and produce a differential position estimate via least-squares adjustment
    """
    # set up first rxpos series
    n_clock_tot = 0
    n_trop_tot = 0
    for antenna_handle in antenna_handles:
        rxpos_series, R_obj  = store_handle.compute_tides(antenna_handle.times_gps, antenna_handle.ref_pos, antenna_handle.antenna_name) 
        antenna_handle.update_pos_series(rxpos_series, R_obj, antenna_handle.ref_pos)
        store_handle.compute_azel(antenna_handle.times_gps, antenna_handle)

    for idx, antenna_handle in enumerate(antenna_handles):
        clock_samples = np.zeros(len(antenna_handle.times_gps)) + antenna_handle.bulk_clock*const.c/1e6
        antenna_handle.hold_clock(clock_samples, phase_delay=True) # ref clock is also ref phase clock
        antenna_handle.hold_clock(clock_samples) 

    if dither_phase is True:
        #SEED= int(sha256(antenna_handle.antenna_name.encode('utf-8')).hexdigest(),16) % (2**32)
        rng = np.random.default_rng(seed=seed)

    if obs_type == 'VLBI':
        # generate VLBI measurements
        datetime_total = []
        for jdx, baseline in enumerate(baselines):
            baseline_handle = baseline_handles[jdx]
            datetime_total.extend(baseline_handle.datetime_array)
        datetime_total = np.unique(datetime_total)
        source_array = store_handle.source_array.tolist()
        #obs_data = xr.Dataset({}, coords={"time": datetime_total, "source": source_array})
        obs_data = xr.Dataset({}, coords={"time": datetime_total})

        rand_int_baseline = []
        group_delay_dict = {}
        phase_delay_dict = {}
        antenna_ids = set()
        baseline_ids = []
        antenna_positions = {}

        for jdx, baseline in enumerate(baselines):
            baseline_handle = baseline_handles[jdx]
            antenna1_handle = antenna_handles[baseline[0]]
            antenna2_handle = antenna_handles[baseline[1]]
            _, ant1_idxs, ant2_idxs = np.intersect1d(antenna1_handle.times_gps, \
                    antenna2_handle.times_gps, return_indices=True)

            antenna1_name = antenna1_handle.antenna_name
            antenna2_name = antenna2_handle.antenna_name
            antenna_ids.update([antenna1_name, antenna2_name])

            antenna_positions[antenna1_name] = antenna1_handle.ref_pos
            antenna_positions[antenna2_name] = antenna2_handle.ref_pos

            # Handle axis offsets
            ao_ant1 = antenna1_handle.axis_offset if antenna1_handle.is_VLBI else None
            ao_ant2 = antenna2_handle.axis_offset if antenna2_handle.is_VLBI else None

            # get clock samples
            clock_samples = antenna2_handle.clock_samples[ant2_idxs] - \
                        antenna1_handle.clock_samples[ant1_idxs]
            phase_clock_samples = antenna2_handle.clock_samples[ant2_idxs] - \
                    antenna1_handle.clock_samples[ant1_idxs]

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

            baseline_id = f"{antenna1_name}-{antenna2_name}"
            baseline_ids.append(baseline_id)

            group_delays_baseline = baseline_handle.group_delay_model
            phase_delays_baseline = baseline_handle.phase_delay_model
            sources_baseline = [store_handle.source_time_dict[time] for time in baseline_handle.datetime_array]

            # load kernels if they are supplied
            if covariance_kernel_range is not None:
                kernel_range = load_kernel_parameters(covariance_kernel_range)
                for baseline_handle in baseline_handles:
                    baseline_handle.hold_covariance_kernels(kernel_range)
                    variables = ['azimuth', 'elevation', 'time', 'noise']
                    azimuth = np.radians(antenna1_handle.azim_arr[ant1_idxs])
                    elevation = np.radians(antenna1_handle.elev_arr[ant1_idxs])
                    times = (baseline_handle.datetime_array - baseline_handle.datetime_array[0])/np.timedelta64(1, 's')
                    X = np.vstack((azimuth, elevation, times))
                    cov_range = baseline_handle.build_covariance_matrix('range', X, variables)

                    range_noise = generate_correlated_noise(cov_range, rng)
                    group_delays_baseline = group_delays_baseline + range_noise
            elif q_range is not None:
                range_noise = generate_correlated_noise(np.eye(len(group_delays_baseline))*q_range**2, rng)
                group_delays_baseline = group_delays_baseline + range_noise

            if covariance_kernel_phase is not None:
                if covariance_kernel_range is None:
                    variables = ['azimuth', 'elevation', 'time', 'noise']
                    azimuth = np.radians(antenna1_handle.azim_arr[ant1_idxs])
                    elevation = np.radians(antenna1_handle.elev_arr[ant1_idxs])
                    times = (baseline_handle.datetime_array - baseline_handle.datetime_array[0])/np.timedelta64(1, 's')
                    X = np.vstack((azimuth, elevation, times))

                kernel_phase = load_kernel_parameters(covariance_kernel_phase)
                for baseline_handle in baseline_handles:
                    baseline_handle.hold_covariance_kernels(kernel_phase, True) 
                    cov_phase = baseline_handle.build_covariance_matrix('phase', X, variables)
                    phase_noise = generate_correlated_noise(cov_phase, rng)
                    phase_delays_baseline = phase_delays_baseline + phase_noise

            elif q_phase is not None:
                phase_noise = generate_correlated_noise(np.eye(len(phase_delays_baseline))*q_phase**2, rng)
                phase_delays_baseline = phase_delays_baseline + phase_noise

            if dither_phase is True:
                MAX_N=3 # number of wavelengths that can be added or subtracted
                rand_int = rng.integers(-MAX_N, MAX_N, size=len(baseline_handle.datetime_array))
                phase_delays_baseline = phase_delays_baseline + baseline_handle.wavelength*rand_int
            else:
                rand_int = np.zeros(len(baseline_handle.datetime_array))

            rand_int_baseline.extend(rand_int)
            
            if store_handle.src_type=='VLBI':
                source_coord = [antenna1_handle.point_ra_dec_dict[time] for time in baseline_handle.datetime_array]
                ra_list, dec_list = zip(*source_coord)
                ra_list = np.array(ra_list)
                dec_list = np.array(dec_list)
            else:
                ra_list = np.nan*np.ones(len(baseline_handle.datetime_array))
                dec_list = np.nan*np.ones(len(baseline_handle.datetime_array))
             
            group_delay_da = xr.DataArray(
                group_delays_baseline,
                coords={
                    'time': baseline_handle.datetime_array,
                    'source': ('time', sources_baseline),
                    'source_ra': ('time', ra_list),
                    'source_dec': ('time', dec_list)
                },
                dims=['time'],
                name='group_delay'
            )
            phase_delay_da = xr.DataArray(
                phase_delays_baseline,
                coords={
                    'time': baseline_handle.datetime_array,
                    'source': ('time', sources_baseline),
                    'source_ra': ('time', ra_list),
                    'source_dec': ('time', dec_list)
                },
                dims=['time'],
                name='phase_delay'
            )

            # Store DataArrays in dictionaries using baseline_id as the key
            group_delay_dict[baseline_id] = group_delay_da
            phase_delay_dict[baseline_id] = phase_delay_da

        # Now, collect all antenna names and positions
        antenna_ids = list(antenna_ids)
        antenna_positions_array = np.array([antenna_positions[ant] for ant in antenna_ids])

        # Create a DataArray for antenna positions
        antenna_position_da = xr.DataArray(
            antenna_positions_array,
            coords={'antenna': antenna_ids, 'xyz': ['x', 'y', 'z']},
            dims=['antenna', 'xyz'],
            name='antenna_position'
        )

        # Assign the antenna positions to the dataset
        obs_data['antenna_position'] = antenna_position_da

        # Combine the DataArrays into the Dataset
        # First, create a new coordinate for baselines
        obs_data = obs_data.assign_coords(baseline=baseline_ids)

        # Align times across all baselines
        all_times = np.unique(np.concatenate([da.time.values for da in group_delay_dict.values()]))

        # Reindex DataArrays to the union of all times, filling missing data with NaNs
        for baseline_id in baseline_ids:
            group_delay_da = group_delay_dict[baseline_id].reindex(time=all_times)
            phase_delay_da = phase_delay_dict[baseline_id].reindex(time=all_times)

            # Expand DataArrays to include the 'baseline' dimension
            group_delay_da = group_delay_da.expand_dims('baseline')
            phase_delay_da = phase_delay_da.expand_dims('baseline')

            # Assign the baseline coordinate
            group_delay_da = group_delay_da.assign_coords(baseline=[baseline_id])
            phase_delay_da = phase_delay_da.assign_coords(baseline=[baseline_id])

            # Add DataArrays to the Dataset
            if 'group_delay' in obs_data:
                obs_data['group_delay'].loc[{'baseline': baseline_id}] = group_delay_da.sel(baseline=baseline_id)
            else:
                obs_data['group_delay'] = group_delay_da

            if 'phase_delay' in obs_data:
                obs_data['phase_delay'].loc[{'baseline': baseline_id}] = phase_delay_da.sel(baseline=baseline_id)
            else:
                obs_data['phase_delay'] = phase_delay_da

        # Update the coordinates
        obs_data = obs_data.assign_coords(time=all_times)

        # Set global attributes
        obs_data.attrs['version'] = 2.0
        obs_data.attrs['interval'] = 1.0
        obs_data.attrs['time_system'] = 'GPS'
        obs_data.attrs['frequency'] = baseline_handles[0].f1

        # Save to NetCDF if requested
        if nc_out is not None:
            obs_data.attrs['filename'] = nc_out[0]
            obs_data.to_netcdf(nc_out[0])

    elif obs_type == 'GNSS':
        VLBI_like = False
        if len(nc_out) < len(antenna_handles) and nc_out is not None:
            raise ValueError('Fewer file names than supplied antennas. These must be equal for GNSS data.')

        datetime_total = []
        if VLBI_like is False:
            for antenna_handle in antenna_handles:
                datetime_total.extend(antenna_handle.times_gps)
            datetime_total = np.unique(datetime_total)
        else:
            datetime_full = antenna_handles[0].times_gps
            for antenna_handle in antenna_handles:
                datetime_full = np.intersect1d(antenna_handle.times_gps, datetime_full)

            # take only first point -- we are creating a single point per observation
            datetime_total = []
            source_last = 'none'
            for time in datetime_full:
                source_time = store_handle.source_time_dict[time]
                if source_time != source_last:
                    datetime_total.append(time)
                source_last = source_time

        rand_int_antenna = []
        source_array = store_handle.source_array.tolist()
        for idx, antenna_handle in enumerate(antenna_handles):
            antenna_position = antenna_handle.ref_pos
            if store_handle.iono_free is False:
                pseudorange, carrier_phase = store_handle.correct_PR_CP(antenna_handle, phase=True, phase_only=False, sim=True)
            else:
                pseudorange, pseudorange_dual, carrier_phase, carrier_phase_dual = store_handle.correct_PR_CP(antenna_handle, phase=True, phase_only=False, sim=True)

            if VLBI_like is True:
                # take only 1 measurement per source observation
                _, ant_idxs, _ = np.intersect1d(antenna_handle.times_gps, datetime_total, return_indices=True)
                pseudorange = np.array(pseudorange)[ant_idxs]
                carrier_phase = np.array(carrier_phase)[ant_idxs]
                if store_handle.iono_free is True:
                    pseudorange_dual = np.array(pseudorange_dual)[ant_idxs]
                    carrier_phase_dual = np.array(carrier_phase_dual)[ant_idxs]

            f1 = 1575.42e6
            if store_handle.iono_free is True:
                if self.iono_freq == 'L2':
                    f2 = 1227.60*1e6
                elif self.iono_freq == 'L5':
                    f2 = 1176.45*1e6

            # generate noise -- inner noise by source and outer noise between sources
            if q_range is not None or q_phase is not None:
                source_array_full = []
                if VLBI_like is False:
                    for time in antenna_handle.times_gps:
                        source_array_full.append(store_handle.source_time_dict[time])
                else:
                    for time in datetime_total:
                        source_array_full.append(store_handle.source_time_dict[time])
                source_array_full = np.array(source_array_full)
                N_red = 2 # factor of within-satellite noise to inter-satellite noise

                if q_range is not None:
                    Q_eta = np.zeros((len(source_array_full),len(source_array_full)))
                    for source in store_handle.source_array:
                        source_idxs = np.argwhere(np.equal(source_array_full, source))
                        for run_idxs in consecutive_idxs(source_idxs.flatten()):
                            if len(run_idxs)>0:
                                Q_eta[run_idxs[0]:run_idxs[-1]+1,run_idxs[0]:run_idxs[-1]+1] = np.ones((len(run_idxs),len(run_idxs)))
                    q_range_inner = q_range / N_red
                    cov_range = Q_eta*q_range**2/2 + np.eye(len(pseudorange))*q_range_inner**2/2
                    range_noise = generate_correlated_noise(cov_range, rng)
                    pseudorange = pseudorange + range_noise

                    #for kdx, source in enumerate(source_array):
                    #    source_idxs = np.argwhere(source_array_full==source)
                    #    pseudorange[source_idxs] = pseudorange[source_idxs] + range_noise[kdx]

                    if store_handle.iono_free is True:
                        dual_range_noise = generate_correlated_noise(cov_range, rng)
                        pseudorange_dual = pseudorange_dual + dual_range_noise

                if q_phase is not None:
                    Q_eta = np.zeros((len(source_array_full),len(source_array_full)))
                    for source in store_handle.source_array:
                        source_idxs = np.argwhere(np.equal(source_array_full, source))
                        for run_idxs in consecutive_idxs(source_idxs.flatten()):
                            if len(run_idxs)>0:
                                Q_eta[run_idxs[0]:run_idxs[-1]+1,run_idxs[0]:run_idxs[-1]+1] = np.ones((len(run_idxs),len(run_idxs)))
                    q_phase_inner = q_phase / N_red
                    cov_phase = Q_eta*q_phase**2/2 + np.eye(len(carrier_phase))*q_phase_inner**2/2
                    phase_noise = generate_correlated_noise(cov_phase, rng)
                    carrier_phase = carrier_phase + phase_noise

                    if store_handle.iono_free is True:
                        dual_phase_noise = generate_correlated_noise(cov_phase, rng)
                        carrier_phase_dual = carrier_phase_dual + dual_phase_noise

            # save a data file for each antenna
            obs_data = xr.Dataset({}, coords={"time": datetime_total, "source": source_array})
            obs_data.attrs['version'] = 2.0
            obs_data.attrs['interval'] = 1.0
            obs_data.attrs['rinextype'] = 'obs'
            obs_data.attrs['time_system'] = 'GPS'
            obs_data.attrs['filename'] = nc_out[idx]
            obs_data.attrs['frequency'] = 1575.42e6
            obs_data.attrs['position'] = [antenna_position[0], antenna_position[1], antenna_position[2]]

            satellites = store_handle.source_array.tolist()
            if VLBI_like is False:
                times_gps = antenna_handle.times_gps
            else:
                times_gps = datetime_total


            if dither_phase is True and antenna_handle.antenna_type != 'GNSS':
                MAX_N=3 # number of wavelengths that can be added or subtracted
                rand_int = rng.integers(-MAX_N, MAX_N, size=len(times_gps))
                carrier_phase = carrier_phase + const.c/f1*rand_int
                if store_handle.iono_free is True:
                    carrier_phase_dual = carrier_phase_dual + const.c/f2*rand_int
            else:
                rand_int = np.zeros(len(times_gps))
            rand_int_antenna.append(rand_int)

            # Create 2D arrays for pr and cp with nan values
            pr_2d = np.full((len(times_gps), len(satellites)), np.nan)
            cp_2d = np.full((len(times_gps), len(satellites)), np.nan)
            SNR_2d = np.full((len(times_gps), len(satellites)), np.nan)
            if store_handle.iono_free is True:
                pr_dual_2d = np.full((len(times_gps), len(satellites)), np.nan)
                cp_dual_2d = np.full((len(times_gps), len(satellites)), np.nan)
                SNR_dual_2d = np.full((len(times_gps), len(satellites)), np.nan)

            SNR = np.zeros(len(pseudorange)) + 50 # 50 dB-Hz constant

            # Assign values to the source satellite at each epoch
            for jdx in range(len(times_gps)):
                sv_index = satellites.index(store_handle.source_time_dict[times_gps[jdx]])
                pr_2d[jdx, sv_index] = pseudorange[jdx]
                cp_2d[jdx, sv_index] = carrier_phase[jdx]*f1/const.c
                SNR_2d[jdx, sv_index] = SNR[jdx]
                if store_handle.iono_free is True:
                    pr_dual_2d[jdx, sv_index] = pseudorange_dual[jdx]
                    cp_dual_2d[jdx, sv_index] = carrier_phase_dual[jdx]*f2/const.c # need to 
                    SNR_dual_2d[jdx, sv_index] = SNR[jdx]

            pr_xarray = xr.DataArray(pr_2d, coords={'time': times_gps, 'sv': satellites})
            obs_data = obs_data.assign({'C1': pr_xarray})

            cp_xarray = xr.DataArray(cp_2d, coords={'time': times_gps, 'sv': satellites})
            obs_data = obs_data.assign({'L1': cp_xarray})

            SNR_xarray = xr.DataArray(SNR_2d, coords={'time': times_gps, 'sv': satellites})
            obs_data = obs_data.assign({'S1': SNR_xarray})

            if store_handle.iono_free is True:
                pr_dual_xarray = xr.DataArray(pr_dual_2d, coords={'time': times_gps, 'sv': satellites})
                cp_dual_xarray = xr.DataArray(cp_dual_2d, coords={'time': times_gps, 'sv': satellites})
                SNR_dual_xarray = xr.DataArray(SNR_dual_2d, coords={'time': times_gps, 'sv': satellites})
                if store_handle.iono_freq == 'L2':
                    obs_data = obs_data.assign({'C2': pr_dual_xarray})
                    obs_data = obs_data.assign({'L2': cp_dual_xarray})
                    obs_data = obs_data.assign({'S2': SNR_dual_xarray})
                elif store_handle.iono_freq == 'L5':
                    obs_data = obs_data.assign({'C5': pr_dual_xarray})
                    obs_data = obs_data.assign({'L5': cp_dual_xarray})
                    obs_data = obs_data.assign({'S5': SNR_dual_xarray})

            if mdh_out is not None:
                save_mdh_obs(obs_data)
            if nc_out is not None:
                obs_data.to_netcdf(nc_out[idx])
        
    if obs_type == 'VLBI':
        return np.array(rand_int_baseline)
    else:
        return datetime_total, np.array(rand_int_antenna)

        #baseline_handles = []
        #for jdx, baseline in enumerate(baselines): # generate differential measurements on the baselines
        #   antenna1_handle = antenna_handles[baseline[0]]
        #   antenna2_handle = antenna_handles[baseline[1]]
        #   times_gps, ant1_idxs, ant2_idxs = np.intersect1d(antenna1_handle.times_gps, antenna2_handle.times_gps, return_indices=True)
        #   source_array = [store_handle.source_time_dict[time] for time in times_gps]
        #   if store_handle.iono_free:
        #       if store_handle.iono_freq == 'L2':
        #           f2 = 1227.60*1e6
        #       elif store_handle.iono_freq == 'L5':
        #           f2 = 1176.45*1e6
        #   # initialize a baseline object to hold differential phase info
        #   baseline_handle = BaselineInfo(times_gps, f1)
        #   baseline_handles.append(baseline_handle)

        ## load kernels if they are supplied -- CURRENTLY UNUSED
        #if covariance_kernel_range is not None:
        #    ls_azimuth, ls_elevation, ls_time, noise = load_kernel_from_csv(covariance_kernel_range)
        #    for baseline_handle in baseline_handles:
        #        baseline_handle.hold_covariance_kernels(ls_azimuth, ls_elevation, ls_time, noise)

        #if covariance_kernel_phase is not None:
        #    ls_azimuth, ls_elevation, ls_time, noise = load_kernel_from_csv(covariance_kernel_phase)
        #    for baseline_handle in baseline_handles:
        #        baseline_handle.hold_covariance_kernels(ls_azimuth, ls_elevation, ls_time, noise, True) 

if __name__ == '__main__':    
    ### Parse command-line options
    parser = argparse.ArgumentParser()
    add_args_to_parser(parser)
    args = parser.parse_args()
        
    obs_type = args.obs_type
    if args.src_type == 'GNSS':
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
           elif system == 'Galileo':
               system_code = 'E'
           elif system == 'BeiDou':
               system_code = 'C'

           sv_id = str(sat.id)
           if len(sv_id) == 1: sv_id = '0' + sv_id
           
           rinex_id = system_code + sv_id
           rinex_sats.append(rinex_id)
    elif args.src_type == 'VLBI':
        if args.fringe_file is not None:
            src_dict = read_src(args.src_file)
        nav_store = []
    else:
        raise ValueError('Unknown solution type (--src_type): '+args.src_type)

    antenna_names = [antenna_name for antenna_name in args.antenna_names]
    antenna_types = [antenna_type for antenna_type in args.antenna_types]
    if obs_type == 'VLBI' or args.rxpos is not None:
        if args.rxpos is None and args.vgosDB is None and args.vda_file is None:
            raise ValueError('Must supply receiver positions for VLBI solution (--rxpos)')
        if args.rxpos is not None:
            rxpos_all = []
            for rxpos_arg in args.rxpos:
                rxpos = [float(pos_comp) for pos_comp in rxpos_arg[0].split()]
                rxpos_all.append(rxpos)

    vlbi_antennas = [vlbi_antenna for vlbi_antenna in args.vlbi_antennas]
    axis_offsets = [axis_offset for axis_offset in args.axis_offsets]
   
    if args.iono_free is True:
        if args.iono_freq == 'L2':
            print('Using iono-free combination with L1/L2')
        elif args.iono_freq == 'L5':
            print('Using iono-free combination with L1/L5')
        else:
            raise ValueError('Unknown carrier frequency ' + args.iono_freq)

    baselines = list(itertools.combinations(range(len(antenna_names)), 2))
    duration_arr = []

    if obs_type == 'GNSS' and args.rinex_files is not None:
        # get data from rinex
        rinex_files = [rinex_file[0] for rinex_file in args.rinex_files]
        full_data = {}
        for rinex_file in rinex_files:
            rinex_name = rinex_file.split('.')
            if rinex_name[1][-1] == 'o' or rinex_name[1] =='rnx': 
                rinex_data_full = load(rinex_file)
                rinex_data_full.to_netcdf(rinex_name[0]+'.nc')
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

    if obs_type == 'GNSS':
        # parse or generate key file
        if args.key_file is not None:
            if args.rinex_files is not None:
                datetime_array, source_array, point_ra_dec_array, dt_key, duration_key, source_key, point_key  \
                        = import_key_gnss(rinex_files, full_data, args.key_file)
            else: 
                sim_data_rate = 1 # time in seconds between measurements
                datetime_array, source_array, point_ra_dec_array, dt_key, duration_key, source_key, point_key  \
                        = import_key_gnss([], [], args.key_file, sim_data_rate)
        else:
            for idx, rinex_file in enumerate(rinex_files):
                rinex_data = full_data[rinex_file]
                time_arr = rinex_data.time.values
                if idx == 0:
                    common_times = time_arr
                else:
                    common_times = np.union1d(common_times,time_arr)

            start_date = common_times[0]
            end_date = common_times[-1]
            point_ra_dec_array = []
            datetime_array, source_array = gen_key(rinex_files, full_data, start_date, end_date, \
                                                           eph_sats, args.iono_free, args.iono_freq)

        # GNSS data has uniform sampling -- the mode of the time differences will be this interval
        times_sec = (datetime_array-datetime_array[0])/np.timedelta64(1,'s')
        avg_diff = mode(np.diff(times_sec), keepdims=True)[0][0]
        duration_arr = np.ones(len(datetime_array))*avg_diff

        if args.rinex_files is not None:
            thinned_data = thin_data(antenna_names, rinex_files, full_data, datetime_array, source_array)

        baseline_handles = [] # we will fill this later for GNSS data

    elif obs_type == 'VLBI':
        # parse key file
        if args.src_type == 'GNSS':
            baseline_handles, datetime_array, source_array, point_ra_dec_array, dt_ant, duration_arr = import_data_vlbi(args.fringe_file, \
                    antenna_names, baselines, rinex_sats, args.key_file)
        elif args.src_type == 'VLBI':
            if args.fringe_file is not None:
                baseline_handles, datetime_array, source_array, point_ra_dec_array, dt_ant, duration_arr = import_data_vlbi_farfield(args.fringe_file, antenna_names, baselines, src_dict)
            if args.ngs_file is not None:
                if args.utc2gps is None:
                    raise ValueError('UTC2GPS cannot be none when reading NGS file')
                baseline_handles, datetime_array, source_array, point_ra_dec_array, dt_ant = import_data_vlbi_ngs(args.ngs_file, antenna_names, baselines, args.utc2gps)
            if args.vda_file is not None:
                if args.band is None:
                    raise ValueError('Must select a band when reading vgosDA')
                baseline_handles, datetime_array, source_array, point_ra_dec_array, dt_ant, rxpos_all_db, duration_arr = import_data_vlbi_vda(args.vda_file, antenna_names, baselines, args.band)
                if args.rxpos is None:
                    rxpos_all = rxpos_all_db
            if args.vgosDB is not None:
                if args.band is None:
                    raise ValueError('Must select a band when reading vgosDB')
                if args.utc2gps is not None:
                    print('argument utc2gps has no effect with vgosDB data input (offset is in DB)')
                baseline_handles, datetime_array, source_array, point_ra_dec_array, dt_ant, rxpos_all_db, duration_arr = import_data_vlbi_vgosdb(args.vgosDB, antenna_names, baselines, args.band)
                if args.rxpos is None:
                    rxpos_all = rxpos_all_db

    # get antenna data from antex file
    antenna_store = AntennaStore()
    if args.src_type == 'GNSS':
         antenna_store.includeAllSatellites()
    time_beg = date_to_common(datetime_array[0], 'GPS')
    if len(antenna_names) > len(vlbi_antennas):
        antenna_store.addANTEXfile(args.antex_file, time_beg) # define time b/c sat antennas defined by epoch
   
    names = std_vector_string()

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
     
    # initialize the stores handle
    store_handle = GNSSTKStores(obs_type, args.src_type, sol_sys, antenna_store, ocean_store, atm_store, nav_store, \
            args.iono_free, args.iono_freq, args.analytical_delay)

    if args.src_type == 'GNSS':
        store_handle.build_antenna_map(source_array, datetime_array)
    elif args.src_type == 'VLBI':
        store_handle.hold_source_array(source_array, datetime_array, duration_arr)
    
    store_handle.save_exp_weather(args.trop_T, args.trop_P, args.trop_H)

    # initialize the antenna handles
    antenna_handles = []
    for antenna_idx, antenna_name in enumerate(antenna_names):
        antenna_type = antenna_types[antenna_idx]
        if obs_type == 'GNSS':
            if args.rinex_files is None:
                antenna_data = []
            else:
                antenna_data = thinned_data[antenna_name]

            if args.rxpos is None:
                if args.rinex_files is None:
                   raise ValueError('With no rinex files, need to supply rxpos')
                antenna_position = antenna_data.position 
            else:
                antenna_position = rxpos_all[antenna_idx]
        else:
            antenna_position = rxpos_all[antenna_idx]
        bulk_clock = args.bulk_clocks[antenna_idx][0]

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

            antenna_handle = AntennaInfo(antenna_name, antenna_position, antenna_type, bulk_clock, 0, False, tropModel) 
        else:
            antenna_handle = AntennaInfo(antenna_name, antenna_position, antenna_type, bulk_clock, 0, False) 

        if antenna_name in vlbi_antennas:
            # phase center is handled by geometric calculation, no PCO
            idx_vlbi = [i for i, val in enumerate(vlbi_antennas) if val == antenna_name]
            axis_offset = float(np.array(axis_offsets)[idx_vlbi][0])
            antenna_handle.set_VLBI(axis_offset, point_ra_dec_array, datetime_array, False)
        else:
            # antenna phase center variation
            antennaPCOData = AntexData()
            store_handle.antenna_store.getAntenna(antenna_handle.antenna_type, antennaPCOData) # memory leak, needs investigation
            antenna_handle.hold_PCO(antennaPCOData)      
            if antennaPCOData.nFreq <= 4 and args.iono_freq == 'L5':
                print('No L5 data in ANTEX file -- using L2 mapping (danger)')
        if obs_type == 'GNSS':
            if args.rinex_files is None:
                antenna_handle.hold_times(np.array(datetime_array))
            else:
                antenna_handle.hold_data(antenna_data)
        if obs_type == 'VLBI':
            times_antenna = dt_ant[antenna_name]
            antenna_handle.hold_times(times_antenna)
            antenna_handle.set_phase_clock()
        antenna_handles.append(antenna_handle)

    # build the IONEX ionosphere store
    if args.ionex_file is not None:
        if args.iono_free is True:
            raise Exception('IONEX file should not be used with ionosphere free combination')
        store_handle.build_ionex_store(args.ionex_file)

    # build the OBX satellite attitude store
    if args.OBX_file is not None:
        if args.src_type=='VLBI': 
            print('OBX file ' + args.OBX_file + ' will not be used (VLBI solution)')
        else:
            store_handle.build_obx_store(args.OBX_file)

    if args.ant_info_file is not None:
        read_thermal_deformation_coeffs(args.ant_info_file, antenna_handles)
        print('Read antenna thermal deformation model. Need either vgosDB or temperature with --trop_T to use')

    _ = simulate_data(args.nc_out, args.mdh_out, obs_type, store_handle, antenna_handles, baselines, baseline_handles,\
                      args.load_covariance_kernel_range, args.load_covariance_kernel_phase, args.q_range, args.q_phase, args.dither_phase)
