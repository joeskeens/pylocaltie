#!/usr/bin/python
"""
Tests for single_diff_tools.py and local_tie_estimator.py
This script runs a series of simulated collections with realistic noise and ensures that 
integer phase ambiguity resolution works in each.

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
import numpy as np
import pytest
import itertools
import os

use_custom_version = os.getenv('USE_CUSTOM_GEORINEX', 'false').lower() == 'true'
if use_custom_version:
    import sys
    sys.path.insert(0, '/sgl/ceph/work/jskeens')
    from georinex_custom import load
else:
    from georinex import load
import xarray as xr

from local_tie_estimator import lstsq_estimation
from sim_local_tie import simulate_data

from gnsstk import std_vector_string, Position, AntennaStore, AntexData, OceanLoadTides, PoleTides, AtmLoadTides, \
                  CommonTime, SolarSystem, GlobalTropModel, SaasTropModel, NeillTropModel

from single_diff_tools import import_key_gnss, import_data_vlbi, import_data_vlbi_farfield, import_data_vlbi_ngs, import_data_vlbi_vgosdb, \
                  import_data_vlbi_vda, write_SINEX, datetime64_to_mjd, map_datasets, BaselineInfo, AntennaInfo, GNSSTKStores, \
                  date_to_common, import_data_nc_sim, read_thermal_deformation_coeffs, thin_data, NavStore

np.set_printoptions(precision=5, linewidth=110)

def test_VLBI_VLBI():
    """
    Test that for VLBI observations of VLBI sources, we can simulate
    noisy data with randomized cycle slips and correctly estimate the 
    cycle slips we produced.
    """
    # setup experiment
    antenna_names = ["HART15M", "HARTRAO"]
    antenna_types = ["Az-El", "Equa"]
    rxpos_all = [[5085490.77200,  2668161.46300, -2768692.60400], [5085442.78000, 2668263.49000, -2768697.01400]]
    vlbi_antennas = ['HART15M', 'HARTRAO']
    axis_offsets = ['1.46400', '6.69140']
    ref_antenna = 'HARTRAO' 
    bulk_clocks = [[0.0], [0.0]]
    vgosDB = './test_data/15NOV09XA'
    band = 'X'
    SSEfile = './test_data/SolarSystem1960to2040.405.ssbin'
    atmfile = './test_data/atm_load.atl'
    oceanfile = './test_data/ocean_load.blq'
    earthfile = './test_data/finals.all'
    ant_info_file = './test_data/antenna-info.txt'
    analytical_delay = False
    nc_out = ['hartrao_15nov09.nc']
    obs_type = 'VLBI'
    src_type = obs_type
    q_range =  0.00412
    q_phase = 0.00345
    trop_poly_length = 60*60 
    clock_poly_length = 60*60

    baselines = list(itertools.combinations(range(len(antenna_names)), 2))
    baseline_handles, datetime_array, source_array, point_ra_dec_array, dt_ant, rxpos_all_db, duration_arr = import_data_vlbi_vgosdb(vgosDB, antenna_names, baselines, band)
    trop_H = 50
     
    antenna_store = AntennaStore()
    time_beg = date_to_common(datetime_array[0], 'GPS')

    if len(antenna_names) > len(vlbi_antennas):
        antenna_store.addANTEXfile(args.antex_file, time_beg) # define time b/c sat antennas defined by epoch
    
    names = std_vector_string()

    # initialize solar system model
    sol_sys = SolarSystem()
    sol_sys.initializeWithBinaryFile(SSEfile)
    sol_sys.addFile(earthfile)

    # trim EOP store for input date/time
    mjd_beg = datetime64_to_mjd(datetime_array[0])
    mjd_end = datetime64_to_mjd(datetime_array[-1])
    sol_sys.edit(int(mjd_beg-5), int(mjd_end+5))

    # have to use std_vector_string b/c typemap for list of strings to std_vec<std_string> isnt working
    ant_names_cpp = std_vector_string(antenna_names)

    # initialize Ocean Loading and Atmospheric Loading
    ocean_store = OceanLoadTides()
    ocean_store.initializeSites(ant_names_cpp, oceanfile)
    atm_store = AtmLoadTides()
    atm_store.initializeSites(ant_names_cpp, atmfile)

    # initialize the stores handle
    store_handle = GNSSTKStores(obs_type, src_type, sol_sys, antenna_store, ocean_store, atm_store, [], \
            False, None, analytical_delay)

    store_handle.hold_source_array(source_array, datetime_array, [])
    store_handle.estimate_trop = True

    antenna_handles = []
    for antenna_idx, antenna_name in enumerate(antenna_names):
        antenna_type = antenna_types[antenna_idx]
        bulk_clock = bulk_clocks[antenna_idx][0]
        antenna_position = rxpos_all[antenna_idx]
        tk_pos = Position(antenna_position[0], antenna_position[1], antenna_position[2])
        tropModel = GlobalTropModel(tk_pos, time_beg)
        tropModel.setHumidity(trop_H)
        antenna_handle = AntennaInfo(antenna_name, antenna_position, antenna_type, bulk_clock, False, tropModel)

        if antenna_name in vlbi_antennas:
            # phase center is handled by geometric calculation, no PCO
            idx_vlbi = [i for i, val in enumerate(vlbi_antennas) if val == antenna_name]
            axis_offset = float(np.array(axis_offsets)[idx_vlbi][0])
            antenna_handle.set_VLBI(axis_offset, point_ra_dec_array, datetime_array, False, False)
        else:
            # antenna phase center variation
            antennaPCOData = AntexData()
            store_handle.antenna_store.getAntenna(antenna_handle.antenna_type, antennaPCOData) # memory leak, needs investigation
            antenna_handle.hold_PCO(antennaPCOData)

        times_antenna = dt_ant[antenna_name]
        antenna_handle.hold_times(times_antenna)
        antenna_handle.set_phase_clock()
        antenna_handles.append(antenna_handle)

    read_thermal_deformation_coeffs(ant_info_file, antenna_handles)
    seed=42 # the answer to the ultimate question of life, the universe, and everything
    print('Simulating VLBI data...')
    simulated_integer_amb = simulate_data(nc_out, None, obs_type, store_handle, antenna_handles, baselines, baseline_handles,\
                      None, None, q_range, q_phase, True, seed)

    T1 = baseline_handles[0].T1
    T2 = baseline_handles[0].T2
    H1 = baseline_handles[0].H1
    H2 = baseline_handles[0].H2
    P1 = baseline_handles[0].P1
    P2 = baseline_handles[0].P2

    # import simulated data
    print('Loading simulated VLBI data...')
    baseline_handles, datetime_array, source_array, point_ra_dec_array, dt_ant = import_data_nc_sim(nc_out[0], \
                                    antenna_names, baselines)
    baseline_handles[0].save_weather(P1, H1, T1, P2, H2, T2)

    print('Estimating positions from simulated VLBI data (this will take a little while)...')
    estimated_integer_amb, baseline_handles = lstsq_estimation(obs_type, False, ref_antenna, store_handle, antenna_handles, baselines, baseline_handles,\
                     clock_poly_length, trop_poly_length, None, False, True, True, True, True, False, False, None, None)

    # shift estimated integer amb to simulated integer amb by 1 integer (this is still a success)
    estimated_integer_amb = estimated_integer_amb + int(np.round(simulated_integer_amb - estimated_integer_amb)[0])
    np.testing.assert_array_equal(estimated_integer_amb, simulated_integer_amb)
    print('VLBI-VLBI Test successful!!! \n')

def test_GNSS_VLBI():
    """
    Test that for VLBI observations of GNSS sources, we can simulate
    noisy data with randomized cycle slips and correctly estimate the 
    cycle slips we produced.
    """
    # setup experiment
    antenna_names = ["DBR205", "FD-VLBA"]
    antenna_types = ["TPSCR.G5        TPSH", "Az-El"]
    rxpos_all = [[-1324070.4781, -5332176.0011, 3231921.7985], [-1324009.4540, -5332181.9550, 3231962.3690]]
    vlbi_antennas = ['FD-VLBA']
    axis_offsets = ['2.1329']
    ref_antenna = 'DBR205' 
    bulk_clocks = [[0.1867], [0.088035]]
    fringe_file = './test_data/observables_final_pared.txt'
    antex_file = './test_data/igs20.atx'
    OBX_file = './test_data/COD0OPSFIN_20230250000_01D_30S_ATT.OBX'
    SSEfile = './test_data/SolarSystem1960to2040.405.ssbin'
    atmfile = './test_data/atm_load.atl'
    oceanfile = './test_data/ocean_load.blq'
    earthfile = './test_data/finals.all'
    ant_info_file = './test_data/antenna-info.txt'
    key_file='./test_data/uy001d_prn_vlbi.key'
    analytical_delay = True
    nc_out = ['uy001d_vlbi.nc']
    obs_type = 'VLBI'
    src_type = 'GNSS'
    q_range =  0.194
    q_phase = 0.0044
    trop_P = 840
    trop_T = 5
    trop_H = 80
    trop_poly_length = 0 
    clock_poly_length = 86400

    baselines = list(itertools.combinations(range(len(antenna_names)), 2))

    eph_files = [['./test_data/COD0MGXFIN_20230250000_01D_05M_ORB.SP3'], ['./test_data/COD0MGXFIN_20230250000_01D_30S_CLK.CLK']]
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

    baseline_handles, datetime_array, source_array, point_ra_dec_array, dt_ant, duration_dict = import_data_vlbi(fringe_file, \
        antenna_names, baselines, rinex_sats, key_file)
    antenna_store = AntennaStore()
    time_beg = date_to_common(datetime_array[0], 'GPS')
    antenna_store.includeAllSatellites()
    antenna_store.addANTEXfile(antex_file, time_beg) # define time b/c sat antennas defined by epoch
    
    names = std_vector_string()

    # initialize solar system model
    sol_sys = SolarSystem()
    sol_sys.initializeWithBinaryFile(SSEfile)
    sol_sys.addFile(earthfile)

    # trim EOP store for input date/time
    mjd_beg = datetime64_to_mjd(datetime_array[0])
    mjd_end = datetime64_to_mjd(datetime_array[-1])
    sol_sys.edit(int(mjd_beg-5), int(mjd_end+5))

    # have to use std_vector_string b/c typemap for list of strings to std_vec<std_string> isnt working
    ant_names_cpp = std_vector_string(antenna_names)

    # initialize Ocean Loading and Atmospheric Loading
    ocean_store = OceanLoadTides()
    ocean_store.initializeSites(ant_names_cpp, oceanfile)
    atm_store = AtmLoadTides()
    atm_store.initializeSites(ant_names_cpp, atmfile)

    # initialize the stores handle
    store_handle = GNSSTKStores(obs_type, src_type, sol_sys, antenna_store, ocean_store, atm_store, nav_store, \
            False, None, analytical_delay)
    store_handle.save_exp_weather(trop_T, trop_P, trop_H)
    store_handle.hold_source_array(source_array, datetime_array, duration_dict)
    store_handle.build_antenna_map(source_array, datetime_array)

    antenna_handles = []
    for antenna_idx, antenna_name in enumerate(antenna_names):
        antenna_type = antenna_types[antenna_idx]
        bulk_clock = bulk_clocks[antenna_idx][0]
        antenna_position = rxpos_all[antenna_idx]
        tk_pos = Position(antenna_position[0], antenna_position[1], antenna_position[2])
        tropModel = GlobalTropModel(tk_pos, time_beg)
        tropModel.setHumidity(trop_H)
        antenna_handle = AntennaInfo(antenna_name, antenna_position, antenna_type, bulk_clock, False, tropModel)

        if antenna_name in vlbi_antennas:
            # phase center is handled by geometric calculation, no PCO
            idx_vlbi = [i for i, val in enumerate(vlbi_antennas) if val == antenna_name]
            axis_offset = float(np.array(axis_offsets)[idx_vlbi][0])
            antenna_handle.set_VLBI(axis_offset, point_ra_dec_array, datetime_array, False, False)
        else:
            # antenna phase center variation
            antennaPCOData = AntexData()
            store_handle.antenna_store.getAntenna(antenna_handle.antenna_type, antennaPCOData) # memory leak, needs investigation
            antenna_handle.hold_PCO(antennaPCOData)

        times_antenna = dt_ant[antenna_name]    
        antenna_handle.hold_times(times_antenna)
        antenna_handle.set_phase_clock()        
        antenna_handles.append(antenna_handle)
                                                
    read_thermal_deformation_coeffs(ant_info_file, antenna_handles)
    store_handle.build_obx_store([OBX_file])

    seed=42 # the answer to the ultimate question of life, the universe, and everything
    print('Simulating VLBI observations of GNSS satellites...')
    simulated_integer_amb = simulate_data(nc_out, None, obs_type, store_handle, antenna_handles, baselines, baseline_handles,\
                      None, None, q_range, q_phase, True, seed)

    # import simulated data
    print('Loading simulated data...')
    baseline_handles, datetime_array, source_array, point_ra_dec_array, dt_ant = import_data_nc_sim(nc_out[0], \
                                    antenna_names, baselines)

    print('Estimating positions from simulated data (this will take a little while)...')
    estimated_integer_amb, baseline_handles = lstsq_estimation(obs_type, False, ref_antenna, store_handle, antenna_handles, baselines, baseline_handles,\
                     clock_poly_length, trop_poly_length, None, False, True, True, True, True, False, False, None, None)

    # shift estimated integer amb to simulated integer amb by 1 integer (this is still a success)
    estimated_integer_amb = estimated_integer_amb + int(np.round(simulated_integer_amb - estimated_integer_amb)[0])
    np.testing.assert_array_equal(estimated_integer_amb, simulated_integer_amb)
    print('GNSS-VLBI Test successful!!! \n')


def test_MULTI_GNSS_VLBI():
    """
    Test that for VLBI observations of GNSS sources, we can simulate
    noisy data with randomized cycle slips and correctly estimate the 
    cycle slips we produced.
    """
    # setup experiment
    antenna_names = ["DBR205", "DBR231", "FD-VLBA"]
    antenna_types = ["TPSCR.G5        TPSH", "TPSCR.G5        TPSH", "Az-El"]
    rxpos_all = [[-1324070.4781, -5332176.0011, 3231921.7985], [-1330748.6520, -5328115.2850, 3236419.9321], [-1324009.4540, -5332181.9550, 3231962.3690]]
    vlbi_antennas = ['FD-VLBA']
    axis_offsets = ['2.1329']
    ref_antenna = 'DBR205' 
    bulk_clocks = [[0.1867], [-0.059], [0.088035]]
    fringe_file = './test_data/observables_final_pared.txt'
    antex_file = './test_data/igs20.atx'
    OBX_file = './test_data/COD0OPSFIN_20230250000_01D_30S_ATT.OBX'
    SSEfile = './test_data/SolarSystem1960to2040.405.ssbin'
    atmfile = './test_data/atm_load.atl'
    oceanfile = './test_data/ocean_load.blq'
    earthfile = './test_data/finals.all'
    ant_info_file = './test_data/antenna-info.txt'
    key_file='./test_data/uy001d_prn_vlbi.key'
    analytical_delay = True
    nc_out = ['uy001d_vlbi.nc']
    obs_type = 'VLBI'
    src_type = 'GNSS'
    #q_range =  0.194 -- real value
    #q_phase = 0.0044 -- real value
    q_range =  0.08
    q_phase = 0.004
    trop_P = 840
    trop_T = 5
    trop_H = 80
    trop_poly_length = 0 
    clock_poly_length = 86400

    baselines = list(itertools.combinations(range(len(antenna_names)), 2))

    eph_files = [['./test_data/COD0MGXFIN_20230250000_01D_05M_ORB.SP3'], ['./test_data/COD0MGXFIN_20230250000_01D_30S_CLK.CLK']]
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

    baseline_handles, datetime_array, source_array, point_ra_dec_array, dt_ant, duration_dict = import_data_vlbi(fringe_file, \
        antenna_names, baselines, rinex_sats, key_file)
    antenna_store = AntennaStore()
    time_beg = date_to_common(datetime_array[0], 'GPS')
    antenna_store.includeAllSatellites()
    antenna_store.addANTEXfile(antex_file, time_beg) # define time b/c sat antennas defined by epoch
    
    names = std_vector_string()

    # initialize solar system model
    sol_sys = SolarSystem()
    sol_sys.initializeWithBinaryFile(SSEfile)
    sol_sys.addFile(earthfile)

    # trim EOP store for input date/time
    mjd_beg = datetime64_to_mjd(datetime_array[0])
    mjd_end = datetime64_to_mjd(datetime_array[-1])
    sol_sys.edit(int(mjd_beg-5), int(mjd_end+5))

    # have to use std_vector_string b/c typemap for list of strings to std_vec<std_string> isnt working
    ant_names_cpp = std_vector_string(antenna_names)

    # initialize Ocean Loading and Atmospheric Loading
    ocean_store = OceanLoadTides()
    ocean_store.initializeSites(ant_names_cpp, oceanfile)
    atm_store = AtmLoadTides()
    atm_store.initializeSites(ant_names_cpp, atmfile)

    # initialize the stores handle
    store_handle = GNSSTKStores(obs_type, src_type, sol_sys, antenna_store, ocean_store, atm_store, nav_store, \
            False, None, analytical_delay)
    store_handle.save_exp_weather(trop_T, trop_P, trop_H)
    store_handle.hold_source_array(source_array, datetime_array, duration_dict)
    store_handle.build_antenna_map(source_array, datetime_array)

    antenna_handles = []
    for antenna_idx, antenna_name in enumerate(antenna_names):
        antenna_type = antenna_types[antenna_idx]
        bulk_clock = bulk_clocks[antenna_idx][0]
        antenna_position = rxpos_all[antenna_idx]
        tk_pos = Position(antenna_position[0], antenna_position[1], antenna_position[2])
        tropModel = GlobalTropModel(tk_pos, time_beg)
        tropModel.setHumidity(trop_H)
        antenna_handle = AntennaInfo(antenna_name, antenna_position, antenna_type, bulk_clock, False, tropModel)

        if antenna_name in vlbi_antennas:
            # phase center is handled by geometric calculation, no PCO
            idx_vlbi = [i for i, val in enumerate(vlbi_antennas) if val == antenna_name]
            axis_offset = float(np.array(axis_offsets)[idx_vlbi][0])
            antenna_handle.set_VLBI(axis_offset, point_ra_dec_array, datetime_array, False, False)
        else:
            # antenna phase center variation
            antennaPCOData = AntexData()
            store_handle.antenna_store.getAntenna(antenna_handle.antenna_type, antennaPCOData) # memory leak, needs investigation
            antenna_handle.hold_PCO(antennaPCOData)

        times_antenna = dt_ant[antenna_name]    
        antenna_handle.hold_times(times_antenna)
        antenna_handle.set_phase_clock()        
        antenna_handles.append(antenna_handle)
                                                
    read_thermal_deformation_coeffs(ant_info_file, antenna_handles)
    store_handle.build_obx_store([OBX_file])

    seed=42 # the answer to the ultimate question of life, the universe, and everything
    print('Simulating VLBI observations of GNSS satellites...')
    simulated_integer_amb = simulate_data(nc_out, None, obs_type, store_handle, antenna_handles, baselines, baseline_handles,\
                      None, None, q_range, q_phase, True, seed)

    # import simulated data
    print('Loading simulated data...')
    baseline_handles, datetime_array, source_array, point_ra_dec_array, dt_ant = import_data_nc_sim(nc_out[0], \
                                    antenna_names, baselines)

    print('Estimating positions from simulated data (this will take a little while)...')
    estimated_integer_amb, baseline_handles = lstsq_estimation(obs_type, False, ref_antenna, store_handle, antenna_handles, baselines, baseline_handles,\
                     clock_poly_length, trop_poly_length, None, False, True, True, True, True, False, False, None, None)

    # shift estimated integer amb to simulated integer amb by 1 integer (this is still a success)
    idx_start = 0
    for baseline_handle in baseline_handles:
        idxs_handle = np.arange(idx_start,idx_start+len(baseline_handle.datetime_array))
        estimated_integer_amb[idxs_handle] = estimated_integer_amb[idxs_handle] + int(np.round(simulated_integer_amb[idxs_handle] - estimated_integer_amb[idxs_handle])[0])
        idx_start = idx_start + len(baseline_handle.datetime_array)
    try: np.testing.assert_array_equal(estimated_integer_amb, simulated_integer_amb)
    except: breakpoint()
    print('VLBI-Style Multi-GNSS Test successful!!! \n')


def test_GNSS_GNSS():
    """
    Test that for VLBI observations of GNSS sources, we can simulate
    noisy data with randomized cycle slips and correctly estimate the 
    cycle slips we produced.
    """
    # setup experiment
    antenna_names = ["DBR205", "FD-VLBA"]
    antenna_types = ["TPSCR.G5        TPSH", "Az-El"]
    rxpos_all = [[-1324070.4781, -5332176.0011, 3231921.7985], [-1324009.4540, -5332181.9550, 3231962.3690]]
    vlbi_antennas = ['FD-VLBA']
    axis_offsets = ['2.1329']
    ref_antenna = 'DBR205' 
    bulk_clocks = [[0.1867], [0.088035]]
    fringe_file = './test_data/observables_final_pared.txt'
    antex_file = './test_data/igs20.atx'
    OBX_file = './test_data/COD0OPSFIN_20230250000_01D_30S_ATT.OBX'
    SSEfile = './test_data/SolarSystem1960to2040.405.ssbin'
    atmfile = './test_data/atm_load.atl'
    oceanfile = './test_data/ocean_load.blq'
    earthfile = './test_data/finals.all'
    ant_info_file = './test_data/antenna-info.txt'
    key_file='./test_data/uy001d_prn_vlbi.key'
    analytical_delay = True
    nc_out = ['SIM_DBR205_GNSS.nc', 'SIM_FD_VLBA_GNSS.nc']
    obs_type = 'GNSS'
    src_type = 'GNSS'
    q_range =  0.02
    q_phase = 0.002
    trop_P = 840
    trop_T = 5
    trop_H = 80
    trop_poly_length = 0 
    clock_poly_length = 86400

    baselines = list(itertools.combinations(range(len(antenna_names)), 2))

    eph_files = [['./test_data/COD0MGXFIN_20230250000_01D_05M_ORB.SP3'], ['./test_data/COD0MGXFIN_20230250000_01D_30S_CLK.CLK']]

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

    sim_data_rate = 1 # time in seconds between measurements
    datetime_array, source_array, point_ra_dec_array, dt_key, duration_key, source_key, point_key  \
            = import_key_gnss([], [], key_file, sim_data_rate)

    baseline_handles = [] # we will fill this later for GNSS data

    antenna_store = AntennaStore()
    time_beg = date_to_common(datetime_array[0], 'GPS')
    antenna_store.includeAllSatellites()
    antenna_store.addANTEXfile(antex_file, time_beg) # define time b/c sat antennas defined by epoch
    
    names = std_vector_string()

    # initialize solar system model
    sol_sys = SolarSystem()
    sol_sys.initializeWithBinaryFile(SSEfile)
    sol_sys.addFile(earthfile)

    # trim EOP store for input date/time
    mjd_beg = datetime64_to_mjd(datetime_array[0])
    mjd_end = datetime64_to_mjd(datetime_array[-1])
    sol_sys.edit(int(mjd_beg-5), int(mjd_end+5))

    # have to use std_vector_string b/c typemap for list of strings to std_vec<std_string> isnt working
    ant_names_cpp = std_vector_string(antenna_names)

    # initialize Ocean Loading and Atmospheric Loading
    ocean_store = OceanLoadTides()
    ocean_store.initializeSites(ant_names_cpp, oceanfile)
    atm_store = AtmLoadTides()
    atm_store.initializeSites(ant_names_cpp, atmfile)

    # initialize the stores handle
    store_handle = GNSSTKStores(obs_type, src_type, sol_sys, antenna_store, ocean_store, atm_store, nav_store, \
            False, None, analytical_delay)
    store_handle.save_exp_weather(trop_T, trop_P, trop_H)
    store_handle.hold_source_array(source_array, datetime_array, [])
    store_handle.build_antenna_map(source_array, datetime_array)

    antenna_handles = []
    for antenna_idx, antenna_name in enumerate(antenna_names):
        antenna_type = antenna_types[antenna_idx]
        bulk_clock = bulk_clocks[antenna_idx][0]
        antenna_position = rxpos_all[antenna_idx]
        tk_pos = Position(antenna_position[0], antenna_position[1], antenna_position[2])
        tropModel = GlobalTropModel(tk_pos, time_beg)
        tropModel.setHumidity(trop_H)
        antenna_handle = AntennaInfo(antenna_name, antenna_position, antenna_type, bulk_clock, False, tropModel)

        if antenna_name in vlbi_antennas:
            # phase center is handled by geometric calculation, no PCO
            idx_vlbi = [i for i, val in enumerate(vlbi_antennas) if val == antenna_name]
            axis_offset = float(np.array(axis_offsets)[idx_vlbi][0])
            antenna_handle.set_VLBI(axis_offset, point_ra_dec_array, datetime_array, False, False)
        else:
            # antenna phase center variation
            antennaPCOData = AntexData()
            store_handle.antenna_store.getAntenna(antenna_handle.antenna_type, antennaPCOData) # memory leak, needs investigation
            antenna_handle.hold_PCO(antennaPCOData)

        antenna_handle.hold_times(np.array(datetime_array))
        antenna_handle.hold_trop(np.zeros(len(antenna_handle.times_gps)))
        antenna_handles.append(antenna_handle)
                                                
    read_thermal_deformation_coeffs(ant_info_file, antenna_handles)
    store_handle.build_obx_store([OBX_file])

    seed=42 # the answer to the ultimate question of life, the universe, and everything
    print('Simulating GNSS observations of GNSS satellites...')
    datetime_array_simulated, simulated_integer_amb = simulate_data(nc_out, None, obs_type, store_handle, antenna_handles, baselines, baseline_handles,\
                      None, None, q_range, q_phase, True, seed)

    # import simulated data
    print('Loading simulated data...')
    rinex_files = nc_out
    # get data from rinex
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
        
    datetime_array, source_array, point_ra_dec_array, dt_key, duration_key, source_key, point_key  \
            = import_key_gnss(rinex_files, full_data, key_file)

    baseline_handles = [] # we will fill this later for GNSS data
    thinned_data = thin_data(antenna_names, rinex_files, full_data, datetime_array, source_array)
    for antenna_handle in antenna_handles:
        antenna_data = thinned_data[antenna_handle.antenna_name]
        antenna_handle.hold_data(antenna_data)

    print('Estimating positions from simulated data (this will take a little while)...')
    estimated_integer_amb, baseline_handles = lstsq_estimation(obs_type, False, ref_antenna, store_handle, antenna_handles, baselines, baseline_handles,\
                     clock_poly_length, trop_poly_length, None, False, True, True, True, True, False, False, None, None)

    # shift estimated integer amb to simulated integer amb by 1 integer (this is still a success)
    common_epochs, idxs_estimated, idxs_simulated = np.intersect1d(baseline_handles[0].datetime_array, datetime_array_simulated, return_indices=True)
    simulated_integer_amb_baseline = simulated_integer_amb[1,:] - simulated_integer_amb[0,:]
    simulated_integer_amb_baseline = simulated_integer_amb_baseline[idxs_simulated]
    estimated_integer_amb = estimated_integer_amb + int(np.round(simulated_integer_amb_baseline - estimated_integer_amb)[0])
    np.testing.assert_array_equal(estimated_integer_amb, simulated_integer_amb_baseline)
    print('GNSS-GNSS Test successful!!! \n')

if __name__ == '__main__':
    test_VLBI_VLBI()
    test_GNSS_VLBI()
    test_GNSS_GNSS()
    test_MULTI_GNSS_VLBI()


