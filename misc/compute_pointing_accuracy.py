#!/usr/bin/python
"""
Use precise ephemerides to find high acccuracy right ascension and declination.
Compare these to the pointing polynomials used by the VLBA.
"""
import datetime
import numpy as np
import scipy.constants as const
import argparse
from gnsstk import std_vector_double, SatID, std_vector_GNSS, std_vector_SatID, std_vector_string,\
                  IonosphereFreeRange, SatelliteSystem, NavSearchOrder, NavSatelliteID, RinexSatID, Triple,\
                  Position, AntennaStore, AntexData, OceanLoadTides, PoleTides, \
                  AtmLoadTides, computeSolidEarthTides, \
                  AntexStream, EphTime, CommonTime, TimeSystem, SolarSystem, EarthOrientation, northEastUpGeodetic
from single_diff_tools import import_key_gnss, datetime64_to_mjd, AntennaInfo, GNSSTKStores, NavStore, date_to_common, read_thermal_deformation_coeffs, find_sigmas
import ast
import re
from collections import defaultdict
from scipy.stats import mode
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
available_fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')
for font in available_fonts: fm.fontManager.addfont(font)

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

def add_args_to_parser(parser_in):
    """Add arguments to parser."""
    parser.add_argument("--file_out", default='pointing_error.txt',
                         help="Name of file to write by-satellite pseudorange and doppler measurements to for use in software correlator"
                         )
    parser.add_argument("--pointing_file", default='FD.py',
                         help="Name of Python file containing pointing polynomials calculated by NRAO"
                         )
    parser.add_argument("--tle_file", default=None,
                         help="Name of two-line element file to compare to precise ephemerides orbit (optional, will plot differences at observation times)"
                         )
    parser.add_argument("--tle_name_rinex", default=None,
                         help="RINEX ID of satellite in the supplied two-line element file"
                         )
    parser.add_argument("-e", dest="eph_files", action="append", type=str, nargs="+", help = 'Ephemeris file. Add files for day before and after experiment too.')
    parser.add_argument("--key_file", default=None,
                         help="Name of key file determining schedule. Source prefixes must match RINEX satellite names"
                         )
    parser.add_argument("--rxpos", type=str, nargs="+", help="Receiver position  as 'X Y Z' (m)")
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
    parser.add_argument("--earthfile",
                         help="EOP file (finals.all).",
                         )
    parser.add_argument("--vlbi_antenna",
                         help="Usage: --vlbi_antenna. Designate the antenna as VLBI. Will have to supply axis offset",
                         default=False,
                         action="store_true")
    parser.add_argument("--axis_offset",
                         help="VLBI antenna axis offset in meters",
                         default=0)
    parser.add_argument("--analytical_delay",
                         action="store_true",
                         default=False,
                         help="Use the adapted Jaron, Nothnagel (2019) analytical delay model (recommended) instead of iterative light time model",
                         )
    parser.add_argument("--antenna_name",
                        type=str,
                        help="Name of antenna.")
    parser.add_argument("--antenna_type", 
                        type=str,
                        help="Antenna type (e.g. GNSS, XY-N, XY-E, Az-El, Equa, BWG, Nasmyth)."+\
                              " If GNSS, supply ANTEX type rather than GNSS.")

def write_file(file_out, max_error_scan, min_error_scan, avg_error_scan, max_error_source, min_error_source, avg_error_source, datetime_array_scan, source_array_scan): 
    """
    Writes pointing error data to a file line by line.

    The function writes the data in a formatted manner to `file_out`, 
    ensuring each line contains the corresponding label, pseudorange, and doppler.
    """
    with open(file_out, 'w') as f_handle:
        for idx, source in enumerate(np.unique(source_array_scan)):
            line1 = f"for all points of source {source}: \n"
            line2 = f"max pointing error: {max_error_source[idx]} arcmin \n"
            line3 = f"min pointing error: {min_error_source[idx]} arcmin \n"
            line4 = f"avg pointing error: {avg_error_source[idx]} arcmin \n"
            f_handle.write(line1)
            f_handle.write(line2)
            f_handle.write(line3)
            f_handle.write(line4)
            f_handle.write('\n')

        for idx, source in enumerate(source_array_scan):
            dt = datetime_array_scan[idx].astype('datetime64[ms]').astype(datetime.datetime)
            formatted_dt = f"{dt:%Y %m %d %H:%M:%S}"
            line1 = f"for source {source} at time {dt}: \n"
            line2 = f"max pointing error: {max_error_scan[idx]} arcmin \n"
            line3 = f"min pointing error: {min_error_scan[idx]} arcmin \n"
            line4 = f"avg pointing error: {avg_error_scan[idx]} arcmin \n"
            f_handle.write(line1)
            f_handle.write(line2)
            f_handle.write(line3)
            f_handle.write(line4)
            f_handle.write('\n')


def write_file_az_el(file_out, mjd_scan, az_scan, el_scan, source_arr): 
    """
    Writes pointing error data to a file line by line.

    The function writes the data in a formatted manner to `file_out`, 
    ensuring each line contains the corresponding label, pseudorange, and doppler.
    """
    file_out_az_el = file_out.split('.')[0] + '_az_el' + '.' + file_out.split('.')[1]
    with open(file_out_az_el, 'w') as f_handle:
        for idx, source in enumerate(source_arr):
            line1 = f"for source {source} at MJD {mjd_scan[idx]}: \n"
            line2 = f"az (deg): {az_scan[idx]}, el (deg) {el_scan[idx]} \n"
            f_handle.write(line1)
            f_handle.write(line2)
            #f_handle.write('\n')

def parse_orbit_file(filepath):
    data = defaultdict(list)
    current_source = None
    buffer = ''
    inside_append = False

    with open(filepath, 'r') as f:
        for line in f:
            stripped = line.strip()

            # Start of new append
            if stripped.startswith('source_') and '.append' in stripped:
                match = re.match(r"source_(\w+)_orbit\.append\(\s*(.*)", stripped)
                if match:
                    current_source = match.group(1)
                    content = match.group(2).rstrip('\\').strip()
                    buffer = content
                    inside_append = True

                    # One-liner
                    if content.endswith('})') or content.endswith('} )'):
                        try:
                            entry = ast.literal_eval(buffer.rstrip(')'))
                            data[current_source].append(entry)
                        except Exception as e:
                            print(f"Failed to parse entry:\n{buffer}\nError: {e}")
                        buffer = ''
                        inside_append = False
            elif inside_append:
                # Strip backslashes and whitespace, then add
                line_no_backslash = stripped.rstrip('\\').strip()
                buffer += ' ' + line_no_backslash
                if line_no_backslash.endswith('})') or line_no_backslash.endswith('} )'):
                    try:
                        entry = ast.literal_eval(buffer.rstrip(')'))
                        data[current_source].append(entry)
                    except Exception as e:
                        print(f"Failed to parse entry:\n{buffer}\nError: {e}")
                    buffer = ''
                    inside_append = False

    return dict(data)


def datetime64_to_mjd(dt):
    # Ensure datetime64 is in days to compute days since 1970-01-01
    epoch = np.datetime64('1970-01-01T00:00:00', 'ns')
    delta = dt - epoch
    return delta.astype('timedelta64[ns]').astype(float) / 86400e9 + 40587

def angular_difference_rad(a, b):
    """Returns minimal angular difference a - b in radians."""
    delta = a - b
    return (delta + np.pi) % (2 * np.pi) - np.pi

def find_scan_errors(src_dict, ra_arr, dec_arr, antenna_handle, datetime_array_scan, source_array_scan, duration_key):
    """ Take right ascension and declination from analytical model, compare to tracking polynomials, compute errors, plot
    """
    UTC2GPS_MJD = 18/86400
    max_error_scan = []
    min_error_scan = []
    avg_error_scan = []
    mjd_scan_arr = []
    source_arr_full = []
    day = int(datetime64_to_mjd(antenna_handle.times_gps[0]))
    times_scan_full = []
    for idx, source in enumerate(source_array_scan):
        tracking_poly = src_dict[source]
        scan_beg = datetime_array_scan[idx]
        scan_end = scan_beg + duration_key[idx]*np.timedelta64(1,'s')
        idxs_source = np.bitwise_and(antenna_handle.times_gps > scan_beg, antenna_handle.times_gps < scan_end)
        times_scan = antenna_handle.times_gps[idxs_source]
        times_scan_full.append(times_scan)
        mjd_scan = np.array([datetime64_to_mjd(time) for time in times_scan])
        mjd_scan_arr.append(mjd_scan)
        source_scan = np.array([source for time in times_scan])
        source_arr_full.append(source_scan)
        ra_scan = ra_arr[idxs_source]
        dec_scan = dec_arr[idxs_source]
        start_mjd = datetime64_to_mjd(scan_beg)
        end_mjd = datetime64_to_mjd(scan_end)
        found = False
        for poly in tracking_poly:
            # find the correct polynomial
            interval_beg = poly['interval'][0] + UTC2GPS_MJD
            interval_end = poly['interval'][1] + UTC2GPS_MJD
            interval_middle = (interval_beg + interval_end)/2
            if interval_middle > start_mjd and interval_middle < end_mjd:
                # right interval
                found = True
                T_mjd = mjd_scan - interval_beg
                ra_poly = poly['ra']
                dec_poly = poly['dec']
                ra_track = ra_poly[0] + ra_poly[1]*T_mjd + ra_poly[2]*T_mjd**2 + ra_poly[3]*T_mjd**3 + ra_poly[4]*T_mjd**4
                dec_track = dec_poly[0] + dec_poly[1]*T_mjd + dec_poly[2]*T_mjd**2 + dec_poly[3]*T_mjd**3 + dec_poly[4]*T_mjd**4
                error_rad = np.sqrt(angular_difference_rad(ra_track,ra_scan)**2 + angular_difference_rad(dec_track,dec_scan)**2)
                error_arcmin = np.rad2deg(error_rad)*60
                max_error_scan.append(np.amax(error_arcmin))
                min_error_scan.append(np.amin(error_arcmin))
                avg_error_scan.append(np.mean(error_arcmin))

                if idx == 0:
                    # plot a track
                    fig, ax = plt.subplots()
                    ax.scatter(np.rad2deg(ra_track[0]), np.rad2deg(dec_track[0]), marker='*')
                    ax.plot(np.rad2deg(ra_track), np.rad2deg(dec_track), label='tracking poly')
                    ax.scatter(np.rad2deg(ra_scan[0]), np.rad2deg(dec_scan[0]), marker='*')
                    ax.plot(np.rad2deg(ra_scan), np.rad2deg(dec_scan), label='ephemeris')
                    ax.set_xlabel('right ascension (deg)')
                    ax.set_ylabel('declination (deg)')
                    ax.set_title('single scan tracking for ' + source)
                    ax.set_aspect('equal', adjustable='datalim')
                    ax.legend()
                    fig.savefig('tracking_poly_scan_'+source+'_'+antenna_handle.antenna_name+'_MJD_'+str(day)+'.png')
                    plt.close(fig)
            else:
                continue

    max_error_scan = np.array(max_error_scan)
    min_error_scan = np.array(min_error_scan)
    avg_error_scan = np.array(avg_error_scan)

    fig, ax = plt.subplots()
    #ax.scatter(datetime_array_scan, max_error_scan, label='max', marker='+')
    #ax.scatter(datetime_array_scan, min_error_scan, label='min', marker='1')
    ax.scatter(datetime_array_scan, avg_error_scan, label='avg', marker='x')
    ax.set_ylabel('pointing error (arcmin)')
    ax.set_title('tracking error for ' + antenna_handle.antenna_name)
    interval_hr = 2
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hr))  # Adjust interval as needed
    fig.autofmt_xdate()  # Auto-rotate date labels
    #ax.legend()
    fig.savefig('tracking_error_'+antenna_handle.antenna_name+'_MJD_'+str(day)+'.png')
    plt.close(fig)

    # find bad sources
    sources_unique = np.unique(source_array_scan)
    avg_error_source = []
    min_error_source = []
    max_error_source = []
    for source in sources_unique:
        idxs_source = np.argwhere(np.array(source_array_scan)==source)
        avg_error_source.append(np.mean(avg_error_scan[idxs_source]))
        min_error_source.append(np.min(avg_error_scan[idxs_source]))
        max_error_source.append(np.max(avg_error_scan[idxs_source]))

    avg_error_source = np.array(avg_error_source)
    min_error_source = np.array(min_error_source)
    max_error_source = np.array(max_error_source)
    std_error, good_source_idx = find_sigmas(avg_error_source, nsig=4.0)
    good_sources = sources_unique[good_source_idx]

    # plot only good sources 
    fig, ax = plt.subplots()
    for source in good_sources:
        source_idxs = np.argwhere(np.array(source_array_scan)==source)
        times_source = np.array(datetime_array_scan)[source_idxs]
        errors_source = avg_error_scan[source_idxs]
        ax.plot(times_source, errors_source, marker='x')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hr))  # Adjust interval as needed
    ax.set_ylabel('pointing error (arcmin)')
    ax.set_title('tracking error for ' + antenna_handle.antenna_name)
    fig.autofmt_xdate()  # Auto-rotate date labels
    fig.savefig('tracking_error_'+antenna_handle.antenna_name+'_MJD_'+str(day)+'_good_sources.png')
    plt.close(fig)

    return max_error_scan, min_error_scan, avg_error_scan, avg_error_source, min_error_source, max_error_source, np.concatenate(mjd_scan_arr), np.concatenate(source_arr_full), np.concatenate(times_scan_full)

def sample_tle_at_epoch_array(satrec, epochs, frame='ITRS'):
    """ Retrieve TLE ephemeris data at requested epochs"""
    jd1 = np.array([epoch.jd1 for epoch in epochs])
    jd2 = np.array([epoch.jd2 for epoch in epochs])
    err, r, v = satrec.sgp4_array(jd1, jd2)
    if any(err):
        raise ValueError(f"SGP4 propagation failed!")
    else:
        teme_p = CartesianRepresentation(r.T*u.km)
        teme_v = CartesianDifferential(v.T*u.km/u.s)
        states = TEME(teme_p.with_differentials(teme_v), obstime=epochs)
        if frame == "ITRS":
            states = states.transform_to(ITRS(obstime=epochs))
        return states.cartesian.get_xyz().value.T, \
            states.cartesian.differentials["s"].get_d_xyz().value.T


def compare_sat_states(tle_rinexid, datetime_array_scan, source_array_scan, satrec, store_handle, antenna_handle):
    """ Compare TLE coordinates to precise ephemerides """
    idxs_source = np.argwhere(np.array(source_array_scan)==tle_rinexid)
    times_source = np.array(datetime_array_scan)[idxs_source].flatten()
    RSID = RinexSatID(str(tle_rinexid))
    tle_states = sample_tle_at_epoch_array(satrec, Time(times_source))
    tle_pos = tle_states[0]
    times_common = date_to_common(times_source, 'GPS')
    day = int(datetime64_to_mjd(datetime_array_scan[0]))
    dist_arr = []
    for idx, common_time in enumerate(times_common):
        sat_xvt = store_handle.nav_store.get_xvt(RSID, common_time)
        sat_coord = np.array([sat_xvt.x[0], sat_xvt.x[1], sat_xvt.x[2]])
        tle_pos_epoch = tle_pos[idx,:]*1e3
        dist = np.linalg.norm(tle_pos_epoch-sat_coord)
        dist_arr.append(dist)

    fig, ax = plt.subplots()
    dist_arr = np.array(dist_arr)/1e3 # convert to km
    ax.plot(times_source, dist_arr)
    interval_hr = 2
    ax.set_xlim(datetime_array_scan[0], datetime_array_scan[-1])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hr))  # Adjust interval as needed
    ax.set_ylabel('position error (km)')
    ax.set_title('TLE-ephemeris error for ' + tle_rinexid)
    fig.autofmt_xdate()  # Auto-rotate date labels
    fig.savefig('TLE_error_'+antenna_handle.antenna_name+'_MJD_'+str(day)+'_'+tle_rinexid+'.png')
    plt.close(fig)


if __name__ == '__main__':    
    ### Parse command-line options
    parser = argparse.ArgumentParser()
    add_args_to_parser(parser)
    args = parser.parse_args()
        
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


    antenna_name = args.antenna_name
    antenna_type = args.antenna_type

    for rxpos_arg in args.rxpos:
        rxpos = [float(pos_comp) for pos_comp in rxpos_arg.split()]

    vlbi_antenna = args.vlbi_antenna
    axis_offset = args.axis_offset
   
    datetime_array, source_array, point_ra_dec_array, dt_key, duration_key, source_key, point_key  \
            = import_key_gnss([], [], args.key_file, 0)

    times_sec = (datetime_array-datetime_array[0])/np.timedelta64(1,'s')
    # resample at 1 sec for each observation
    datetime_array_full = []
    source_array_full = []
    datetime_array_scan = datetime_array
    source_array_scan = source_array
    duration_dict = {}
    for idx, time in enumerate(datetime_array):
        duration_dict[time] = duration_key[idx]
        for jdx in range(int(duration_key[idx])):
            datetime_array_full.append(time+jdx*np.timedelta64(1, 's'))
            source_array_full.append(source_array[idx])
    datetime_array = datetime_array_full
    source_array = source_array_full

    # get antenna data from antex file
    antenna_store = AntennaStore()
    antenna_store.includeAllSatellites()
    time_beg = date_to_common(datetime_array[0], 'GPS')
    if args.antex_file is not None:
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
    ant_names_cpp = std_vector_string([antenna_name])
    
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
    store_handle = GNSSTKStores('GNSS', 'GNSS', sol_sys, antenna_store, ocean_store, atm_store, nav_store, \
            False, '', True)

    store_handle.build_antenna_map(source_array, datetime_array)
    store_handle.hold_source_array(source_array, datetime_array, duration_dict)

    # initialize the antenna handle
    antenna_position = rxpos
    antenna_handle = AntennaInfo(antenna_name, antenna_position, antenna_type, 0, False) 

    if vlbi_antenna is True:
        # phase center is handled by geometric calculation, no PCO
        antenna_handle.set_VLBI(float(axis_offset), point_ra_dec_array, datetime_array, False, False)
    else:
        # antenna phase center variation
        antennaPCOData = AntexData()
        store_handle.antenna_store.getAntenna(antenna_handle.antenna_type, antennaPCOData) # memory leak, needs investigation
        antenna_handle.hold_PCO(antennaPCOData)      

    antenna_handle.hold_times(np.array(datetime_array))

    good_idxs = []
    for idx, source in enumerate(source_array):
        if source in rinex_sats:
            good_idxs.append(idx)
    antenna_handle.times_gps = antenna_handle.times_gps[good_idxs]

    rxpos_series, R_obj  = store_handle.compute_tides(antenna_handle.times_gps, antenna_handle.ref_pos, antenna_handle.antenna_name) 
    antenna_handle.update_pos_series(rxpos_series, R_obj, antenna_handle.ref_pos)

    src_dict = parse_orbit_file(args.pointing_file) 
    ra_arr, dec_arr = store_handle.compute_ra_dec(antenna_handle, antenna_handle.times_gps)
    max_error_scan, min_error_scan, avg_error_scan, avg_error_source, min_error_source, max_error_source, mjd_scan, source_arr_full, times_scan_full = \
            find_scan_errors(src_dict, ra_arr, dec_arr, antenna_handle, datetime_array_scan, source_array_scan, duration_key)
    store_handle.compute_azel(times_scan_full, antenna_handle)
    az_scan = antenna_handle.azim_arr
    el_scan = antenna_handle.elev_arr

    if args.tle_file is not None: 
        # this requires sgp4 and astropy (heavy dependency!)
        from sgp4.api import Satrec
        from astropy.coordinates import (
            CartesianDifferential, CartesianRepresentation, TEME, ITRS
        )
        from astropy.time import Time
        from astropy import units as u
        from astropy.utils import iers
        with open(args.tle_file) as tle_file:
            lines = tle_file.readlines()
            s = lines[0]
            t = lines[1]
        iers_a = iers.IERS_A.open(args.earthfile)
        iers.earth_orientation_table.set(iers_a)
        satrec = Satrec.twoline2rv(s,t)
        compare_sat_states(args.tle_name_rinex, datetime_array_scan, source_array_scan, satrec, store_handle, antenna_handle)

    write_file(args.file_out, max_error_scan, min_error_scan, avg_error_scan, max_error_source, min_error_source, avg_error_source, datetime_array_scan, source_array_scan)
    write_file_az_el(args.file_out, mjd_scan, az_scan, el_scan, source_arr_full) 
    print("File written to " + args.file_out)
