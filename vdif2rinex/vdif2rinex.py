#!/usr/bin/python
"""
==============================================================================
  This file is part of the VDIF2RINEX software package.

  This is free software; you can redistribute it and/or modify
  it under the terms of the BSD 3-Clause License. See the LICENSE file
  distributed with this software package for the full license text.

  We are distributing this in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  BSD 3-Clause License for more details.

  This software was developed by Applied Research Laboratories at the
  University of Texas at Austin.

  Copyright 2026, The Board of Regents of The University of Texas System
==============================================================================
"""
import numpy as np
from scipy.stats import mode, linregress
import scipy.constants as const
import argparse
import os
import pickle
from datetime import datetime, timedelta, timezone
import gnsstk
import sys
sys.path.insert(0, '/home/jskeens/gnssvlbi/software/analysis')
from single_diff_tools import import_key_gnss, read_key, AntennaInfo, GNSSTKStores, date_to_common, find_sigmas, NavStore
from acq_routines import acq_prn_fft, analytic_signal_fft_1d, oversample_prn_code, shift_code_phase, compute_Gd_numeric_noncoherent,\
        compute_Gd_numeric_coherent, l1cp_overlay_s1, weil_code_from_wp, cs_hex_to_pm1, acq_aided, decode_nav_list, decode_e1b_with_time
import matplotlib.pyplot as plt
import xarray as xr
from numba import njit
from vdif_tools import VDIFHeader, VDIFStats, get_vdif_stats, unpack_vdif_chunk

RANGING_CODES={
  'G': {
  1575.42e6: ['CA', 'CP'],
  1227.6e6: ['CM'], # we will hand-off to CL after acquisition
  1176.45e6: ['Q5'],
  },
  'E': {
  1575.42e6: ['E1C'],
  1278.75e6: ['E6C'],
  1207.14e6: ['E5bQ'],
  1176.45e6: ['E5aQ'],
  },
  'C': {
  1575.42e6: ['B1CP'],
  1561.098e6: ['B1I'],
  1268.152e6: ['B3I'],
  1207.14e6: ['B2I'],
  1176.45e6: ['B2aQ'],
  },
}

GPS_BIII=[4,18,23,14,11,28,1,21,20]
CODE_PERIODS={
  'CA': 1e-3,
  'CP': 1e-2,
  'CM': 2e-2,
  'CL': 1.5,
  'Q5': 1e-3,
  'E1C': 4e-3,
  'E1B': 4e-3,
  'E5aQ': 1e-3,
  'E5bQ': 1e-3,
  'E6C': 1e-3,
  'B1CP': 1e-2,
  'B2aQ': 1e-3,
  'B1I': 1e-3,
  'B2I': 1e-3,
  'B3I': 1e-3,
}
CHIP_RATES={
  'CA': 1.023e6,
  'CD': 2.046e6,
  'CP': 2.046e6,
  'CM': 1.023e6,
  'CL': 1.023e6,
  'I5': 10.23e6,
  'Q5': 10.23e6,
  'E1B': 2.046e6,
  'E1C': 2.046e6,
  'E5aI': 10.23e6,
  'E5aQ': 10.23e6,
  'E5bI': 10.23e6,
  'E5bQ': 10.23e6,
  'E6C': 5.115e6,
  'B1CD': 2.046e6,
  'B1CP': 2.046e6,
  'B2aI': 10.23e6,
  'B2aQ': 10.23e6,
  'B2bI': 10.23e6,
  'B2bQ': 10.23e6,
  'B1I': 2.046e6,
  'B2I': 2.046e6,
  'B3I': 10.23e6,
}

BOC_SCP={ # BOC subcarrier period non-BOC --> 1
  'CA': 1,
  'CD': 1,
  'CP': 1,
  'CM': 1,
  'CL': 1,
  'I5': 1,
  'Q5': 1,
  'E1B': 2,
  'E1C': 2,
  'E5aQ': 1,
  'E5bQ': 1,
  'E6C': 1, 
  'B1CP': 2,
  'B2aQ': 1,
  'B1I': 1,
  'B2I': 1,
  'B3I': 1,
}

CODE_EML={
  'CA': 13.0/256,
  'CD': 26.0/256,
  'CP': 26.0/256,
  'CM': 13.0/256,
  'CL': 13.0/256,
  'I5': 26.0/256,
  'Q5': 128.0/256,
  'E1B': 26.0/256,
  'E1C': 26.0/256,
  'E5aI': 26.0/256,
  'E5aQ': 128.0/256,
  'E5bI': 128.0/256,
  'E5bQ': 128.0/256,
  'E6C': 64.0/256,
  'B1CD': 26.0/256,
  'B1CP': 26.0/256,
  'B2aI': 128.0/256,
  'B2aQ': 128.0/256,
  'B2bI': 128.0/256,
  'B2bQ': 128.0/256,
  'B1I': 26.0/256,
  'B2I': 26.0/256,
  'B3I': 128.0/256,
}

NAV_BIT_MS={
  'CA': 20,
  'CD': 10,
  'CP': 0,
  'CM': 20,
  'CL': 0,
  'I5': 10,
  'Q5': 0,
  'E1B': 4,
  'E1C': 0,
  'E6C': 0,
  'E5aI': 20,
  'E5aQ': 0,
  'E5bI': 20,
  'E5bQ': 0,
  'B1CD': 10,
  'B1CP': 0,
  'B2aI': 5,
  'B2aQ': 0,
  'B2bI': 1,
  'B2bQ': 0,
  'B1I': 20,
  'B2I': 20,
  'B3I': 20,
}
SECONDARY_BIT_MS={
  'CA': 0,
  'CD': 0,
  'CP': 10,
  'CM': 0,
  'CL': 0,
  'I5': 1,
  'Q5': 1,
  'E1B': 0,
  'E1C': 4,
  'E5aI': 0,
  'E5aQ': 1,
  'E5bI': 0,
  'E5bQ': 1,
  'E6C': 1,
  'B1CD': 0,
  'B1CP': 10,
  'B2aI': 0,
  'B2aQ': 1,
  'B2bI': 0,
  'B2bQ': 1,
  'B1I': 0,
  'B2I': 1,
  'B3I': 1,
}

PHASE_SHIFT_RC={
  'CA': 0,
  'CD': 0.25,
  'CP': 0.25,
  'CM': -0.25,
  'CL': -0.25,
  'I5': 0,
  'Q5': -0.25,
  'E1B': 0,
  'E1C': 0.5,
  'E5aI': 0,
  'E5aQ': 0.25,
  'E5bI': 0,
  'E5bQ': 0.25,
  'E6C': -0.5,
  'B1CD': 0,
  'B1CP': 0.25,
  'B2aI': 0,
  'B2aQ': 0.25,
  'B2bI': 0,
  'B2bQ': 0.25,
  'B1I': 0,
  'B2I': 0,
  'B3I': 0,
}

NH10_L5I=[1,1,1,1,-1,-1,1,-1,1,-1]
NH20_L5Q=[1,1,1,1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,-1,1]
NH20_B2I_B3I=[1,1,1,1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,-1,1]
B1I_PREAMBLE=[-1,-1,-1,1,1,1,-1,1,1,-1,1]
GPS_PREAMBLE=[-1,1,1,1,-1,1,-1,-1]
E1B_E5bI_PREAMBLE=[1,-1,1,-1,-1,1,1,1,1,1]
E5aI_PREAMBLE=[-1,1,-1,-1,1,-1,-1,-1,1,1,1,1]
E1C_SECONDARY=[1,1,-1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,-1,-1,1,1,-1,1]

# weil_code_from_wp(w, p, 3607, 1800)
B1CP_WP={
  1: (269, 1889),
  2: (1448, 1268),
  3: (1028, 1593),
  4: (1324, 1186),
  5: (822, 1239),
  6: (5, 1930),
  7: (155, 176),
  8: (458, 1696),
  9: (310, 26),
  10: (959, 1344),
  11: (1238, 1271),
  12: (1180, 1182),
  13: (1288, 1381),
  14: (334, 1604),
  15: (885, 1333),
  16: (1362, 1185),
  17: (181, 31),
  18: (1648, 704),
  19: (838, 1190),
  20: (313, 1646),
  21: (750, 1385),
  22: (225, 113),
  23: (1477, 860),
  24: (309, 1656),
  25: (108, 1921),
  26: (1457, 1173),
  27: (149, 1928),
  28: (322, 57),
  29: (271, 150),
  30: (576, 1214),
  31: (1103, 1148),
  32: (450, 1458),
  33: (399, 1519),
  34: (241, 1635),
  35: (1045, 1257),
  36: (164, 1687),
  37: (513, 1382),
  38: (687, 1514),
  39: (422, 1),
  40: (303, 1583),
  41: (324, 1806),
  42: (495, 1664),
  43: (725, 1338),
  44: (780, 1111),
  45: (367, 1706),
  46: (882, 1543),
  47: (631, 1813),
  48: (37, 228),
  49: (647, 2871),
  50: (1043, 2884),
  51: (24, 1823),
  52: (120, 75),
  53: (134, 11),
  54: (136, 63),
  55: (158, 1937),
  56: (214, 22),
  57: (335, 1768),
  58: (340, 1526),
  59: (661, 1402),
  60: (889, 1445),
  61: (929, 1680),
  62: (1002, 1290),
  63: (1149, 1245),
}

# weil_code_from_wp(w, p, 1021, 100)
B2aQ_WP={
  1: (123, 138),
  2: (55, 570),
  3: (40, 351),
  4: (139, 77),
  5: (31, 885),
  6: (175, 247),
  7: (350, 413),
  8: (450, 180),
  9: (478, 3),
  10: (8, 26),
  11: (73, 17),
  12: (97, 172),
  13: (213, 30),
  14: (407, 1008),
  15: (476, 646),
  16: (4, 158),
  17: (15, 170),
  18: (47, 99),
  19: (163, 53),
  20: (280, 179),
  21: (322, 925),
  22: (353, 114),
  23: (375, 10),
  24: (510, 584),
  25: (332, 60),
  26: (7, 3),
  27: (13, 684),
  28: (16, 263),
  29: (18, 545),
  30: (25, 22),
  31: (50, 546),
  32: (81, 190),
  33: (118, 303),
  34: (127, 234),
  35: (132, 38),
  36: (134, 822),
  37: (164, 57),
  38: (177, 669),
  39: (208, 697),
  40: (249, 93),
  41: (276, 18),
  42: (349, 66),
  43: (439, 318),
  44: (477, 133),
  45: (498, 98),
  46: (88, 70),
  47: (155, 132),
  48: (330, 26),
  49: (3, 354),
  50: (21, 58),
  51: (84, 41),
  52: (111, 182),
  53: (128, 944),
  54: (153, 205),
  55: (197, 23),
  56: (199, 1),
  57: (214, 792),
  58: (256, 641),
  59: (265, 83),
  60: (291, 7),
  61: (324, 111),
  62: (326, 96),
  63: (340, 92)
}

# cs_hex_to_pm1(HEX, 100)
E5aQ_CS_HEX={
  1: '83F6F69D8F6E15411FB8C9B1C',
  2: '66558BD3CE0C7792E83350525',
  3: '59A025A9C1AF0651B779A8381',
  4: 'D3A32640782F7B18E4DF754B7',
  5: 'B91FCAD7760C218FA59348A93',
  6: 'BAC77E933A779140F094FBF98',
  7: '537785DE280927C6B58BA6776',
  8: 'EFCAB4B65F38531ECA22257E2',
  9: '79F8CAE838475EA5584BEFC9B',
  10: 'CA5170FEA3A810EC606B66494',
  11: '1FC32410652A2C49BD845E567',
  12: 'FE0A9A7AFDAC44E42CB95D261',
  13: 'B03062DC2B71995D5AD8B7DBE',
  14: 'F6C398993F598E2DF4235D3D5',
  15: '1BB2FB8B5BF24395C2EF3C5A1',
  16: '2F920687D238CC7046EF6AFC9',
  17: '34163886FC4ED7F2A92EFDBB8',
  18: '66A872CE47833FB2DFD5625AD',
  19: '99D5A70162C920A4BB9DE1CA8',
  20: '81D71BD6E069A7ACCBEDC66CA',
  21: 'A654524074A9E6780DB9D3EC6',
  22: 'C3396A101BEDAF623CFC5BB37',
  23: 'C3D4AB211DF36F2111F2141CD',
  24: '3DFF25EAE761739265AF145C1',
  25: '994909E0757D70CDE389102B5',
  26: 'B938535522D119F40C25FDAEC',
  27: 'C71AB549C0491537026B390B7',
  28: '0CDB8C9E7B53F55F5B0A0597B',
  29: '61C5FA252F1AF81144766494F',
  30: '626027778FD3C6BB4BAA7A59D',
  31: 'E745412FF53DEBD03F1C9A633',
  32: '3592AC083F3175FA724639098',
  33: '52284D941C3DCAF2721DDB1FD',
  34: '73B3D8F0AD55DF4FE814ED890',
  35: '94BF16C83BD7462F6498E0282',
  36: 'A8C3DE1AC668089B0B45B3579',
  37: 'E23FFC2DD2C14388AD8D6BEC8',
  38: 'F2AC871CDF89DDC06B5960D2B',
  39: '06191EC1F622A77A526868BA1',
  40: '22D6E2A768E5F35FFC8E01796',
  41: '25310A06675EB271F2A09EA1D',
  42: '9F7993C621D4BEC81A0535703',
  43: 'D62999EACF1C99083C0B4A417',
  44: 'F665A7EA441BAA4EA0D01078C',
  45: '46F3D3043F24CDEABD6F79543',
  46: 'E2E3E8254616BD96CEFCA651A',
  47: 'E548231A82F9A01A19DB5E1B2',
  48: '265C7F90A16F49EDE2AA706C8',
  49: '364A3A9EB0F0481DA0199D7EA',
  50: '9810A7A898961263A0F749F56'
}

E5bQ_CS_HEX={
  1: 'CFF914EE3C6126A49FD5E5C94',
  2: 'FC317C9A9BF8C6038B5CADAB3',
  3: 'A2EAD74B6F9866E414393F239',
  4: '72F2B1180FA6B802CB84DF997',
  5: '13E3AE93BC52391D09E84A982',
  6: '77C04202B91B22C6D3469768E',
  7: 'FEBC592DD7C69AB103D0BB29C',
  8: '0B494077E7C66FB6C51942A77',
  9: 'DD0E321837A3D52169B7B577C',
  10: '43DEA90EA6C483E7990C3223F',
  11: '0366AB33F0167B6FA979DAE18',
  12: '99CCBBFAB1242CBE31E1BD52D',
  13: 'A3466923CEFDF451EC0FCED22',
  14: '1A5271F22A6F9A8D76E79B7F0',
  15: '3204A6BB91B49D1A2D3857960',
  16: '32F83ADD43B599CBFB8628E5B',
  17: '3871FB0D89DB77553EB613CC1',
  18: '6A3CBDFF2D64D17E02773C645',
  19: '2BCD09889A1D7FC219F2EDE3B',
  20: '3E49467F4D4280B9942CD6F8C',
  21: '658E336DCFD9809F86D54A501',
  22: 'ED4284F345170CF77268C8584',
  23: '29ECCE910D832CAF15E3DF5D1',
  24: '456CCF7FE9353D50E87A708FA',
  25: 'FB757CC9E18CBC02BF1B84B9A',
  26: '5686229A8D98224BC426BC7FC',
  27: '700A2D325EA14C4B7B7AA8338',
  28: '1210A330B4D3B507D854CBA3F',
  29: '438EE410BD2F7DBCDD85565BA',
  30: '4B9764CC455AE1F61F7DA432B',
  31: 'BF1F45FDDA3594ACF3C4CC806',
  32: 'DA425440FE8F6E2C11B8EC1A4',
  33: 'EE2C8057A7C16999AFA33FED1',
  34: '2C8BD7D8395C61DFA96243491',
  35: '391E4BB6BC43E98150CDDCADA',
  36: '399F72A9EADB42C90C3ECF7F0',
  37: '93031FDEA588F88E83951270C',
  38: 'BA8061462D873705E95D5CB37',
  39: 'D24188F88544EB121E963FD34',
  40: 'D5F6A8BB081D8F383825A4DCA',
  41: '0FA4A205F0D76088D08EAF267',
  42: '272E909FAEBC65215E263E258',
  43: '3370F35A674922828465FC816',
  44: '54EF96116D4A0C8DB0E07101F',
  45: 'DE347C7B27FADC48EF1826A2B', 
  46: '01B16ECA6FC343AE08C5B8944',
  47: '1854DB743500EE94D8FC768ED',
  48: '28E40C684C87370CD0597FAB4',
  49: '5E42C19717093353BCAAF4033',
  50: '64310BAD8EB5B36E38646AF01'
}

def add_args_to_parser(parser):
    """ Add arguments to parser """
    parser.add_argument("-i", dest="input_files", action="append", type=str, nargs="+", help = 'Input VDIF file(s) to track.')
    parser.add_argument("-i_x", dest="input_files_x", action="append", type=str, nargs="+", help = 'Input VDIF X polarization file(s) to track.')
    parser.add_argument("-i_y", dest="input_files_y", action="append", type=str, nargs="+", help = 'Input VDIF Y polarization file(s) to track.')
    parser.add_argument("-o", dest="output_files", action="append", type=str, nargs="+", help = 'Output RINEX file(s) (same length as VDIF).')
    parser.add_argument("-e", dest="eph_files", action="append", type=str, nargs="+", help = 'Ephemeris file. Add files for day before and after experiment too.')
    parser.add_argument("--rc_directory",
                        type=str,
                        required=True,
                        help="Filepath of directory with pickle ranging code files.")
    parser.add_argument("--key_file", default=None,
                         help="Name of key file determining schedule. Source prefixes must match RINEX satellite names"
                         )
    parser.add_argument("--satellite", action="append", type=str, nargs="+", help = 'Alternative to key_file. Supply a satellite to track per VDIF file in RINEX format. Will override key_file if supplied')    
    parser.add_argument("--num_channels",
                         type=int,
                         default=None,
                         help="Number of channels in the VDIF. Default: read from file.",
                         )
    parser.add_argument("--channel",
                         type=int,
                         default=0,
                         help="Channel to read.",
                         )
    parser.add_argument("--max_time",
                         type=int,
                         default=None,
                         help="Max number of seconds to correlate.",
                         )
    parser.add_argument("--blind_search",
                         action="store_true",
                         default=False,
                         help="Do a blind Doppler search instead of using ephemeris for aiding.",
                         )
    parser.add_argument("--short_circuit",
                         action="store_true",
                         default=False,
                         help="If .pkl files for models already exist, go straight to uprighting/RINEX writing.",
                         )
    parser.add_argument("--acq_cadence", default=1.0, type=float,
                         help = 'Cadence of output pseudorange and carrier phase measurements in RINEX file in seconds.')
    parser.add_argument("--center_freq", default=30e6, type=float,
                         help = 'Center frequency of GNSS carrier in the IF data. (usually around bandwidth/2)') 
    parser.add_argument("--sky_freq", default=1575.42e6, type=float,
                         help = 'Sky frequency of GNSS carrier in the IF data') 
    parser.add_argument("--rxpos", type=str, nargs="+", help="Receiver position  as 'X Y Z' (m)")
    parser.add_argument("--clock_offset", default=0, type=float,
                         help = 'clock offset for the antenna (microseconds). Convention: positive late.') 
    parser.add_argument("--clock_rate", default=0, type=float,
                         help = 'clock rate for the antenna (microseconds/sec). Convention: positive late.') 
    parser.add_argument("--vlbi_antenna",
                         help="Usage: --vlbi_antenna. Designate the antenna as VLBI for precise ephemeris aiding. Will have to supply axis offset",
                         default=False,
                         action="store_true")
    parser.add_argument("--ca_only",
                         help="Only correlate CA obs instead of CA + CP for GPS",
                         default=False,
                         action="store_true")
    parser.add_argument("--antenna_type", 
                        type=str,
                        help="Antenna type (e.g. GNSS, XY-N, XY-E, Az-El, Equa, BWG, Nasmyth)."+\
                              " If GNSS, supply ANTEX type rather than GNSS.")
    parser.add_argument("--weather_cal_file",
                         help="VLBA weather file for DiFX (used for troposphere model)",
                         )
    parser.add_argument("--earthfile",
                         help="Bulletin A EOP file for aiding (finals.all).",
                         )
    parser.add_argument("--axis_offset",
                         help="VLBI antenna axis offset in meters",
                         type=float,
                         default=0)
    parser.add_argument("--trop_global",
                         action="store_true",
                         default=False,
                         help="Do tropospheric delay modeling with Global model.",
                         )
    parser.add_argument("--trop_H",
                         type=float,
                         default=None,
                         help="Relative humidity at the site.",
                         )
    parser.add_argument("--antenna_name",
                        type=str,
                        help="Name of antenna.")
    parser.add_argument(
        "--cadence",
        type=int,
        default=1,
        help="cadence of output pseudorange/carrier phase measurements")
 
    parser.add_argument(
        "-t",
        "--thread",
        type=int,
        default=0,
        help="VDIF thread ID to process. (default: %(default)s)")

class NavUprighter():
    """ Holds information and functions for uprighting soft bits to fix half-cycle ambiguities """
    def __init__(self, system, ranging_code, prn):
        """
        Initializes object with system, ranging code, and PRN 

        Args:
            system (str): GNSS system character (G, E , C)
            ranging code (str): ranging code name (CA, CP, E1C, E5aQ, B1CP, B2aQ, etc.)
            prn (int): PRN/SVID for observed satellite
        """
        code_period = CODE_PERIODS[ranging_code]*1e3
        self.ranging_code = ranging_code
        nav_bit_length = NAV_BIT_MS[ranging_code]
        secondary_bit_length = SECONDARY_BIT_MS[ranging_code]

        if ranging_code == 'CA':
            self.carrier_code = 'L1'
            self.LNAV_H = _make_lnav_h()
            self.corr_code = np.array(GPS_PREAMBLE)
            self.mode = 'preamble'
            self.num_bits_secondary = int(nav_bit_length/code_period)
        elif ranging_code == 'CP':
            self.carrier_code = 'L1'
            self.mode = 'sec_code'
            self.corr_code = l1cp_overlay_s1(prn) 
            self.num_bits_secondary = int(secondary_bit_length/code_period)
        elif ranging_code == 'CL':
            self.carrier_code = 'L2'
        elif ranging_code == 'Q5':
            self.carrier_code = 'L5'
            self.mode = 'sec_code'
            self.corr_code = np.array(NH20_L5Q)
            self.num_bits_secondary = int(secondary_bit_length/code_period)
        elif ranging_code == 'E1C':
            self.carrier_code = 'L1'
            self.mode = 'sec_code'
            self.corr_code = np.array(E1C_SECONDARY)
            self.num_bits_secondary = int(secondary_bit_length/code_period)
        elif ranging_code == 'E1B':
            self.carrier_code = 'L1'
            self.mode = 'navigation'
        elif ranging_code == 'E5aQ':
            self.carrier_code = 'E5a'
            self.mode = 'sec_code'
            self.corr_code = cs_hex_to_pm1(E5aQ_CS_HEX[prn], 100)
            self.num_bits_secondary = int(secondary_bit_length/code_period)
        elif ranging_code == 'E5bQ':
            self.carrier_code = 'E5b'
            self.mode = 'sec_code'
            self.corr_code = cs_hex_to_pm1(E5bQ_CS_HEX[prn], 100)
            self.num_bits_secondary = int(secondary_bit_length/code_period)
        elif ranging_code == 'E6C':
            self.carrier_code = 'E6'
            self.mode = 'sec_code'
            self.corr_code = cs_hex_to_pm1(E5aQ_CS_HEX[prn], 100)
            self.num_bits_secondary = int(secondary_bit_length/code_period)
        elif ranging_code == 'B1CP':
            self.carrier_code = 'L1'
            self.mode = 'sec_code'
            w, p = B1CP_WP[prn]
            self.corr_code = weil_code_from_wp(w, p, 3607, 1800)
            self.num_bits_secondary = int(secondary_bit_length/code_period)
        elif ranging_code == 'B2aQ':
            self.carrier_code = 'B5a'
            self.mode = 'sec_code'
            w, p = B2aQ_WP[prn]
            self.corr_code = weil_code_from_wp(w, p, 1021, 100)
            self.num_bits_secondary = int(secondary_bit_length/code_period)
        elif ranging_code == 'B1I':
            self.carrier_code = 'B1'
            self.mode = 'preamble'
            self.corr_code = np.array(B1I_PREAMBLE)
            self.num_bits_secondary = int(nav_bit_length/code_period)
        elif ranging_code == 'B2I':
            self.carrier_code = 'B2'
            self.mode = 'sec_code'
            self.corr_code = np.array(NH20_B2I_B3I)
            self.num_bits_secondary = int(secondary_bit_length/code_period)
        elif ranging_code == 'B3I':
            self.carrier_code = 'B3'
            self.mode = 'sec_code'
            self.corr_code = np.array(NH20_B2I_B3I)
            self.num_bits_secondary = int(secondary_bit_length/code_period)

    def secondary_polarity_from_soft(self, soft_bits):
        """
        Estimate polarity (±1) using secondary-code correlation.

        Args:
          soft_bits (numpy ndarray): soft bits, 1 per ranging code period 
    
        Returns:
          pol: +1 (as-is) or -1 (invert)
          phi: best phase offset in soft-sample units (0 .. min(period_ss, len(soft_bits))-1)
          score: signed correlation at best offset
    
        Works identically for K=1 (secondary chip rate == soft sample rate).
        """
        soft_bits = np.asarray(soft_bits, dtype=float).ravel()
        len_corr_code = self.corr_code.size
        period_ss = self.num_bits_secondary * len_corr_code  # period in soft-sample units
    
        # Build one full period of the expanded code at soft rate:
        corr_code_repeat = np.repeat(self.corr_code, self.num_bits_secondary)  
    
        # Search phases within one secondary period
        n_phases = min(period_ss, len(soft_bits))
    
        best_phi = 0
        best_score = 0.0
    
        # Correlate for each phase hypothesis
        # Use as many samples as we have; wrap s cyclically.
        soft_sample_indices = np.arange(soft_bits.size)
        scores = []
        for phi in range(n_phases):
            # align secondary with soft bits: corr_code_repeat[(phi + n) % period_ss]
            idx = (phi + soft_sample_indices) % period_ss
            score = float(np.dot(soft_bits, corr_code_repeat[idx]))
            #print(score)
            scores.append(score)
            if abs(score) > abs(best_score):
                best_score = score
                best_phi = phi
        print(f'best_score: {best_score}')
        print(f'ratio: {best_score/np.median(np.abs(scores))}')
        pol = +1 if best_score >= 0 else -1
        
        if False: #self.ranging_code == 'B3I':
            hard_bits = np.sign(soft_bits)
            for phi in range(n_phases):
                idx = (phi + soft_sample_indices) % period_ss
                corr_idx = corr_code_repeat[idx]
                s1 = hard_bits[:100]-corr_idx[:100]
                s2 = hard_bits[:100]+corr_idx[:100]
                if np.all(s1 ==0) or np.all(s2==0):
                    for idx in range(30): print(np.dot(hard_bits[1000*idx:1000*(idx+1)],corr_idx[1000*idx:1000*(idx+1)]))
                    breakpoint()
            breakpoint()

        return pol, best_phi, best_score

    def preamble_polarity_from_soft(self, soft_bits):
        """
        Determine polarity using preamble comparison to hard_bits.
        Args:
            soft_bits : (N,) array_like real
    
        Returns:
            pol        : +1 or -1
        """
        soft_bits = soft_bits[::self.num_bits_secondary]
        _hard_bits = lambda s, inv=False: np.sign(-s) if inv else np.sign(s)
        scores={}
        for inv in (False,True):
            bits=_hard_bits(soft_bits,inv)
            best=0
            for off in range(30):
                n=(len(bits)-off)//30
                if n:
                    w=bits[off:off+30*n].reshape(n,30)
                    best=max(best,np.count_nonzero(np.all(w[:,:len(self.corr_code)]==self.corr_code,1)))
            scores[inv]=best

        if scores[True] > scores[False]:
            return -1
        else:
            return +1

    def preamble_polarity_ca(self, soft_bits):
        """
        GPS L1 C/A polarity via LNAV parity (word grid) + TLM preamble (sign).

        Step 1: parity equations are polarity-invariant — each row of LNAV_H
                sums 16 transmitted bits, so a wrong-polarity hypothesis still
                satisfies them. They lock the 30-bit word offset (~100% pass at
                the right offset, ~1.6% per word at any other).
        Step 2: at the locked grid, soft-correlate the 8-bit TLM preamble against
                every 10th word (10 word-slots per subframe). Only the TLM slot
                shows a consistent sign across subframes; that sign is polarity.

        Args:
            soft_bits (np.ndarray): soft bits, 1 per ranging code period.

        Returns:
            pol: +1 (as-is) or -1 (invert)
        """
        soft_bits = np.asarray(soft_bits, float).ravel()[::self.num_bits_secondary]
        bits = (soft_bits < 0).astype(np.uint8)
        N = len(bits)
    
        # 1) Lock the 30-bit word boundary via parity (polarity-invariant).
        best_off, best_pass = 0, -1
        for off in range(30):
            n = (N - off) // 30
            if n < 2:
                continue
            w = bits[off:off + 30*n].reshape(n, 30)
            v = np.empty((n - 1, 32), dtype=np.uint8)
            v[:, :2] = w[:-1, 28:30]   # D29*_prev, D30*_prev
            v[:, 2:] = w[1:]           # D1..D30 of current word
            passes = int(np.all(((v @ self.LNAV_H.T) & 1) == 0, axis=1).sum())
            if passes > best_pass:
                best_pass, best_off = passes, off
    
        # 2) Resolve polarity via TLM preamble at the right subframe position.
        #    Soft-correlate corr_code with words[sp::10] for each sp; the TLM
        #    sp gives the largest |score|; its sign is the polarity.
        n = (N - best_off) // 30
        sw = soft_bits[best_off:best_off + 30*n].reshape(n, 30)
        L = len(self.corr_code)
        scores = np.array([(sw[sp::10, :L] @ self.corr_code).sum()
                           for sp in range(10)])
        sp = int(np.argmax(np.abs(scores)))
        return +1 if scores[sp] >= 0 else -1

def _make_lnav_h():
    """ Create polarity check matrix for L1:CA """
    H = np.zeros((6, 32), dtype=np.uint8)
    rows = [
        [0, 2, 3, 4, 6, 7, 11, 12, 13, 14, 15, 18, 19, 21, 24, 26],   # D25
        [1, 3, 4, 5, 7, 8, 12, 13, 14, 15, 16, 19, 20, 22, 25, 27],   # D26
        [0, 2, 4, 5, 6, 8, 9, 13, 14, 15, 16, 17, 20, 21, 23, 28],    # D27
        [1, 3, 5, 6, 7, 9, 10, 14, 15, 16, 17, 18, 21, 22, 24, 29],   # D28
        [2, 4, 6, 7, 8, 10, 11, 15, 16, 17, 18, 19, 22, 23, 25, 30],  # D29
        [0, 1, 4, 6, 7, 9, 10, 11, 12, 14, 16, 20, 23, 24, 25, 31],   # D30
    ]
    for i, idx in enumerate(rows):
        H[i, idx] = 1
    return H

class RinexFile():
    """Reads in obs data and writes RINEX3"""
    rinex_version = 3.04
    data_types = ('pseudorange', 'accumulatedDeltaRange', 'dopplerFrequencyShift', 'snr')
    data_type_map = {'pseudorange':"C", 'accumulatedDeltaRange': "L", 'dopplerFrequencyShift': "D", 'snr': "S"}
    gps_cc_map = {"L1": "1", "L2": "2", "L5": "5"}
    glo_cc_map = {"G1": "1", "G2": "2", "G3": "3"}
    gal_cc_map = {"L1": "1", "E5a": "5", "E6": "6", "E5b": "7", "E5": "8"}
    bds_cc_map = {"L1": "1", "B1": "2", "B3": "6", "B5b": "7", "B5a": "5"}
    carrier_code_map = {"G": gps_cc_map, "R": glo_cc_map, "E": gal_cc_map, "C": bds_cc_map}

    gps_rc_map = {'CodelessY': "W", 'M': "M", 'Mprime': "M", 'MP': "M", 'P':"P", 'Y': "Y", 'CA':"C", 'CL': "L", 'CM':"S", 'CMCL':"X", 'I5':"I", 'Q5': "Q", 'CD':"S", 'CP':"L", 'CDCP':"X"}
    glo_rc_map = {v:k for k,v in {"C": "GFC", "P": "GFP"}.items()}
    gal_rc_map = {'E1A': 'A', 'E1B': 'B', 'E1C': 'C', 'E5aI': 'I', 'E5aQ': 'Q', 'E5bI': 'I', 'E5bQ': 'Q', 'E6C' : 'C'}
    bds_rc_map = {'B1CD': 'D','B1CP': 'P', 'B1I': 'I', 'B1Q': 'Q', 'B2I': 'I', 'B2Q': 'Q', 'B3I': 'I', 'B3Q': 'Q', "B2aQ": "P", "B2bQ": "P", "B2bI": "D"}
    ranging_code_map = {"G": gps_rc_map, "R": glo_rc_map, "E": gal_rc_map, "C": bds_rc_map}

    def __init__(self, rxpos, filename, obs_times):
        """
        Initializes hdf5 obs data for RINEX conversion 

        Args:
            rxpos (list): Receiver position [x, y, z]
        """
        self.rxpos = rxpos
        self.filename = filename
        self.header = gnsstk.Rinex3ObsHeader()
        self.header.fileType = "Observations"
        self.header.antennaPosition = gnsstk.Triple(*self.rxpos)
        self.start_common = date_to_common(obs_times[0], 'GPS')
        self.end_common = date_to_common(obs_times[-1], 'GPS')
        self.num_epochs = len(obs_times)
        self.obs_times = obs_times
        self.used_obs = np.array([], dtype=self.obs_times[0].dtype)
        self.typevec = []

    def gen_header(self, carrier_code, ranging_codes, system_code):
        """Generates RINEX header for carrier/ranging code combination. Call after hold_data()"""
        self.carrier_code = carrier_code
        self.system_code = system_code
        datatypes = ['pseudorange', 'accumulatedDeltaRange', 'dopplerFrequencyShift', 'snr']
        for ranging_code in ranging_codes:
            for dtype in datatypes:
                self.typevec.append(gnsstk.RinexObsID(RinexFile.hdf5_to_rinex_code(carrier_code, ranging_code, dtype, system_code), RinexFile.rinex_version))
        self.header.mapObsTypes[self.system_code] = self.typevec

        # set required fields' flags manually or gnsstk will barf
        self.header.version = RinexFile.rinex_version
        self.header.firstObs = gnsstk.CivilTime(self.start_common)
        self.header.lastObs = gnsstk.CivilTime(self.end_common)
        self.header.date = str(gnsstk.CivilTime(self.header.firstObs))
        self.header.fileSysSat = gnsstk.RinexSatID(-1, gnsstk.SatelliteSystem.Mixed)
        self.header.interval = 1

        self.dataset = gnsstk.std_vector_Rinex3ObsData(self.num_epochs)
        times_common = date_to_common(self.obs_times)
        for i, time in enumerate(times_common):
            self.dataset[i].time = time
            self.dataset[i].epochFlag = 0
            self.dataset[i].clockOffset = 0.
            self.dataset[i].numSVs = 0

        self.header.valid = self.header.allValid302 | self.header.allValid2
        self.header.validEoH = True
        self.header.valid |= gnsstk.Rinex3ObsHeader.validInterval
        self.header.valid |= gnsstk.Rinex3ObsHeader.validFirstTime
        self.header.valid |= gnsstk.Rinex3ObsHeader.validLastTime
        #self.header.fileAgency = "SGL"
        #self.header.fileProgram = "vdif2rinex.py"

    def hdf5_to_rinex_code(carrier_code, ranging_code, data_type, system_code):
        """Converts a set of HDF5 attributes into a 4-character RINEX obs code

        Accepts HDF5 attributes carrierCode, rangingCode, obs column name, and
        systemCode. Uses this set of attributes to find a best mapping into
        RINEX-style character codes.

        Args:
            carrier_code (str): HDF5 carrier code
            ranging_code (str): HDF5 ranging code
            data_type (str): HDF5 obs column name
            system_code (str): RINEX system code

        Returns:
            str: 4-character RINEX obs code

        Raises:
            RinexConversionError: Invalid HDF5 obs code
        """
        try:
            rinex_data_type = RinexFile.data_type_map[data_type]
        except KeyError:
            raise RinexConversionError("Invalid data type '" + data_type + "'")
        try:
            rinex_carrier_code = RinexFile.carrier_code_map[system_code][carrier_code]
        except KeyError:
            raise RinexConversionError("Invalid carrier code '" + carrier_code + "'")
        try:
            rinex_ranging_code = RinexFile.ranging_code_map[system_code][ranging_code]
        except KeyError:
            raise RinexConversionError("Invalid ranging code '"  + ranging_code + "'")
        return system_code + rinex_data_type + rinex_carrier_code + rinex_ranging_code

    def gen_dataset(self, ranging_code, times_gps, pseudorange, ADR, doppler, C_N0, prn):
        """Converts obs to Rinex dataset"""
        times_common = date_to_common(times_gps, 'GPS')
        obs_data = {}
        obs_data['pseudorange'] = pseudorange
        obs_data['accumulatedDeltaRange'] = ADR
        obs_data['dopplerFrequencyShift'] = doppler
        obs_data['snr'] = C_N0
        obs_dtype = lambda dtype: obs_data[dtype]

        rsid = gnsstk.RinexSatID('{}{}'.format(self.system_code, prn))

        #where = np.where(np.ones(self.num_epochs))[0] 
        _, _, where  = np.intersect1d(times_gps, self.obs_times, return_indices=True) 
        self.used_obs = np.union1d(times_gps, self.used_obs)
        valid_idx = gnsstk.std_vector_int(where.tolist())
        for dtype in RinexFile.data_types:
            data = obs_dtype(dtype).astype(float) 
            data = gnsstk.std_vector_double(data.tolist())
            rot_idx = self.header.getObsIndex(RinexFile.hdf5_to_rinex_code(self.carrier_code, ranging_code, dtype, self.system_code))
            gnsstk.writeEpochs(self.dataset, self.header, rsid, data, valid_idx, rot_idx)

    def write(self):
        """Writes data to RINEX file
        """
        self.header.firstObs = gnsstk.CivilTime(date_to_common(self.used_obs[0]))
        self.header.lastObs = gnsstk.CivilTime(date_to_common(self.used_obs[-1]))
        gnsstk.writeRinex3Obs(os.path.abspath(self.filename), self.header, self.dataset)

def process_vdif(vdif_files, vdif_files_dual, output_files, satellites, rc_dir, thread, num_channels, channel,\
        acq_cadence, f_IF, f_sky, store_handle, antenna_handle, aided=False, short_circuit=False, max_time=None):
    """ Read VDIF files, track GNSS signals """
    #if f_sky == 1176.45e6: short_circuit=True
    for idx, vdif_file_handle in enumerate(vdif_files):
        output_file = output_files[idx]

        # we assume one source per VDIF file 
        vdif_file = np.memmap(vdif_file_handle, dtype=np.uint8, mode='r')
        vdif_file_end = os.path.basename(vdif_file_handle)
        print(f'processing file {vdif_file_end}')

        # get VDIF file setup
        vdif_stats, header = get_vdif_stats(vdif_file, thread, num_channels)
        sample_rate = (vdif_stats.frames_per_sec * vdif_stats.data_bytes_per_frame * 8) \
            // (vdif_stats.bits_per_sample * vdif_stats.num_channels)
        consumed = 0
        total_bytes = vdif_file.shape[0]

        IS_COMPLEX = vdif_stats.is_complex
        BITS_PER_SAMPLE = vdif_stats.bits_per_sample
        BIT_MASK = (1 << vdif_stats.bits_per_sample_component) - 1
        frame_len = header.data_frame_len
        header_size = VDIFHeader.size()
        word_size = VDIFHeader.WORD_SIZE

        if vdif_files_dual is not None:
            vdif_file_handle_dual = vdif_files_dual[idx]
            vdif_file_end_dual = os.path.basename(vdif_file_handle_dual)
            vdif_file_dual = np.memmap(vdif_file_handle_dual, dtype=np.uint8, mode='r')
            print(f'processing file {vdif_file_end_dual}')

            # get VDIF file setup
            thread_dual = thread+1
            vdif_stats_dual, header_dual = get_vdif_stats(vdif_file_dual, thread_dual, args.num_channels)
            consumed_dual = 0
            #total_bytes_dual = os.fstat(vdif_file_dual.fileno()).st_size
            total_bytes_dual = vdif_file_dual.shape[0]
            frame_len_dual = header.data_frame_len

            start_sec = header.seconds_from_ref_epoch + header.frame_no/vdif_stats.frames_per_sec
            start_sec_dual = header_dual.seconds_from_ref_epoch + header_dual.frame_no/vdif_stats.frames_per_sec
            frame_diff = int(np.round((start_sec - start_sec_dual)*vdif_stats.frames_per_sec))
            if frame_diff > 0:
                # mismatch of start time -- need to align
                consumed_dual += frame_diff*frame_len_dual
            elif frame_diff < 0:
                consumed += abs(frame_diff)*frame_len


        # remove tz info from file_start_utc to avoid warning 
        time_gps = np.datetime64(vdif_stats.file_start_utc.astimezone(timezone.utc).replace(tzinfo=None)) + np.timedelta64(UTC2GPS, 's')
        if satellites is not None:
            source = satellites[0]
        else:
            try: 
                source = store_handle.source_time_dict[time_gps]
            except:
                # the baseband file was likely delayed due to slewing, etc. but we can
                # find the right time tag by just searching for the closest key
                keys = np.array(list(store_handle.source_time_dict.keys()))  # shape (N,), dtype datetime64
                deltas = np.abs(keys - np.datetime64(time_gps))
                closest_key = keys[np.argmin(deltas)]
                source = store_handle.source_time_dict[closest_key]

        time_shift = header.frame_no/vdif_stats.frames_per_sec # sec
        time_start = time_gps + np.timedelta64(int(time_shift*1e9), 'ns') # correct for start of VDIF

        if not short_circuit:
            samples_per_chunk = int(sample_rate * acq_cadence)
            samples_per_frame = (vdif_stats.data_bytes_per_frame * 8) // (BITS_PER_SAMPLE * vdif_stats.num_channels)
            assert samples_per_chunk % samples_per_frame == 0, \
                        f"samples_per_chunk ({samples_per_chunk}) must be a multiple of samples_per_frame ({samples_per_frame})"

            # process VDIF file
            soft_bits_full = []
            hard_bits_full = []
            model_full = []
            chunk_num = 0
            model = None
            if IS_COMPLEX or vdif_files_dual is not None:
                combine_q = True
            else:
                combine_q = False
           
            chan_mask = np.zeros(vdif_stats.num_channels, dtype=np.bool_)
            chan_mask[channel] = True

            while consumed < total_bytes:
                count_samples = 0
                i_out = []
                q_out = []
                i_out, q_out, bytes_read = unpack_vdif_chunk(
                        vdif_file, consumed, total_bytes, thread,
                        BITS_PER_SAMPLE, vdif_stats.bits_per_sample_component, BIT_MASK,
                        IS_COMPLEX, samples_per_chunk, header_size, word_size, vdif_stats.num_channels, chan_mask)
                consumed += bytes_read

                if consumed >= total_bytes: break
                if vdif_files_dual is not None:
                    q_out, _, bytes_read = unpack_vdif_chunk(
                            vdif_file_dual, consumed_dual, total_bytes_dual, thread_dual,
                            BITS_PER_SAMPLE, vdif_stats_dual.bits_per_sample_component, BIT_MASK,
                            IS_COMPLEX, samples_per_chunk, header_size, word_size, vdif_stats.num_channels, chan_mask)
                    consumed_dual += bytes_read
                    if consumed_dual >= total_bytes_dual: break
                print(vdif_stats)
                soft_bits_rc, hard_bits_rc, model = \
                        track(combine_q, f_sky, np.array(i_out,dtype=int), np.array(q_out,dtype=int), source, rc_dir, vdif_stats, \
                        sample_rate, f_IF, acq_cadence, store_handle, antenna_handle, aided, model, time_start+np.timedelta64(chunk_num, 's'), \
                        vdif_file_handle)
                chunk_num += 1
                print(f'processed second {chunk_num} for source {source}')
                soft_bits_full.append(soft_bits_rc)
                hard_bits_full.append(hard_bits_rc)
                model_full.append(model)
                if max_time is not None:
                    if chunk_num > max_time: break

            # save for easy re-start
            import pickle
            rc_filename_model = source + str(f_sky) + '_' +antenna_handle.antenna_name+ '_' + vdif_file_end+ '_model.pkl'
            rc_filename_hardbits = source + str(f_sky) +'_' +antenna_handle.antenna_name+ '_' + vdif_file_end +'_hb.pkl'
            rc_filename_softbits = source + str(f_sky) +'_' +antenna_handle.antenna_name+ '_' + vdif_file_end +'_sb.pkl'
            with open(rc_filename_model, 'wb') as f:
                pickle.dump(model_full, f)
            with open(rc_filename_hardbits, 'wb') as f:
                pickle.dump(hard_bits_full, f)
            with open(rc_filename_softbits, 'wb') as f:
                pickle.dump(soft_bits_full, f)
        else:
            # short circuit restart
            import pickle
            try:
                rc_filename_model = source + str(f_sky) + '_' +antenna_handle.antenna_name+ '_' + vdif_file_end + '_model.pkl'
                rc_filename_hardbits = source + str(f_sky) +'_' +antenna_handle.antenna_name+ '_' + vdif_file_end +'_hb.pkl'
                rc_filename_softbits = source + str(f_sky) +'_' +antenna_handle.antenna_name+ '_' + vdif_file_end +'_sb.pkl'
                with open(rc_filename_model, mode='rb') as f:
                    model_full = pickle.load(f)
                with open(rc_filename_hardbits, mode='rb') as f:
                    hard_bits_full = pickle.load(f)
                with open(rc_filename_softbits, mode='rb') as f:
                    soft_bits_full = pickle.load(f)
            except:
                rc_filename_model = source + str(f_sky) + '_' +antenna_handle.antenna_name+ '_' + vdif_file_handle.split("_")[-1]+ '_model.pkl'
                rc_filename_hardbits = source + str(f_sky) +'_' +antenna_handle.antenna_name+ '_' + vdif_file_handle.split("_")[-1]+'_hb.pkl'
                rc_filename_softbits = source + str(f_sky) +'_' +antenna_handle.antenna_name+ '_' + vdif_file_handle.split("_")[-1]+'_sb.pkl'
                with open(rc_filename_model, mode='rb') as f:
                    model_full = pickle.load(f)
                with open(rc_filename_hardbits, mode='rb') as f:
                    hard_bits_full = pickle.load(f)
                with open(rc_filename_softbits, mode='rb') as f:
                    soft_bits_full = pickle.load(f)

        print(f'uprighting bits and creating RINEX observables for source {source}')
        resolve_rinex_obs(output_file, f_sky, source, soft_bits_full, hard_bits_full, model_full, store_handle, antenna_handle, time_gps, time_shift, aided, short_circuit)

        print(f"Total bytes processed={consumed}")

def pair_sum(n):
    """
    Sum a numpy array n = [n_1, n_2, n_3, ...] to a new array,
                  n_mod = [n_1, n_2+n_3, n_4+n_5, ...]
    """
    first = n[:1]
    rest = n[1:]
    
    pairs = rest[: (rest.size // 2) * 2].reshape(-1, 2).sum(axis=1)
    
    n_mod = np.concatenate([first, pairs])
    return n_mod

@njit(cache=True)
def pair_sum_numba(n):
    """
    n = [n1, n2, n3, n4, ...]  ->  [n1, n2+n3, n4+n5, ...]
    Drops the last element if the tail after n1 has odd length.
    """
    m = n.shape[0]
    # after the first element, we form floor((m-1)/2) pair sums
    out_len = 1 + (m - 1) // 2
    out = np.empty(out_len, dtype=n.dtype)

    out[0] = n[0]

    j = 1
    i = 1
    # sum (n[1]+n[2]), (n[3]+n[4]), ...
    while i + 1 < m:
        out[j] = n[i] + n[i + 1]
        j += 1
        i += 2

    return out

def pair_mean(t):
    """
    t = [a0, a1, a2, a3, ...] where each ai is a numpy array

    Returns:
      [mean(a0), mean(concat(a1,a2)), mean(concat(a3,a4)), ...]
    Drops the final unpaired array if the count after the first is odd.
    """
    if len(t) == 0:
        return np.array([], dtype=float)

    means = [np.mean(t[0])]

    rest = t[1:]
    rest_even = rest[: (len(rest) // 2) * 2]   # trim last if odd
    for i in range(0, len(rest_even), 2):
        means.append(np.mean(np.concatenate((rest_even[i], rest_even[i+1]))))

    return np.asarray(means, dtype=float)

@njit(cache=True)
def pair_mean_numba(t):
    """
    t = [a0, a1, a2, a3, ...] where each ai is a numpy array

    Returns:
      [mean(a0), mean(concat(a1,a2)), mean(concat(a3,a4)), ...]
    Drops the final unpaired array if the count after the first is odd.
    """
    lengths = np.array([len(a) for a in t], dtype=np.int64)
    Lmax = 0
    for i in range(lengths.shape[0]):
        if lengths[i] > Lmax:
            Lmax = lengths[i]
    data2d = np.zeros((len(t), Lmax), dtype=np.float64)
    for i, a in enumerate(t):
        data2d[i, :len(a)] = np.asarray(a, dtype=np.float64)

    return pair_mean_padded(data2d, lengths)

@njit(cache=True)
def pair_mean_padded(data2d, lengths):
    """
    data2d: (K, Lmax) padded with zeros
    lengths: (K,) actual lengths for each row
    """
    K = lengths.shape[0]
    if K == 0:
        return np.empty(0, dtype=np.float64)

    out_len = 1 + (K - 1) // 2
    out = np.empty(out_len, dtype=np.float64)

    # mean of a0
    s0 = 0.0
    n0 = lengths[0]
    for j in range(n0):
        s0 += data2d[0, j]
    out[0] = s0 / n0 if n0 > 0 else np.nan

    out_i = 1
    i = 1
    while i + 1 < K:
        s = 0.0
        n = 0
        n1 = lengths[i]
        n2 = lengths[i + 1]
        for j in range(n1):
            s += data2d[i, j]
        for j in range(n2):
            s += data2d[i + 1, j]
        n = n1 + n2
        out[out_i] = s / n if n > 0 else np.nan
        out_i += 1
        i += 2

    return out


def wrap_to_pi(angle_rad):
    """Wrap to (-pi, pi]."""
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi

def wrap_to_pi_two(angle_rad):
    """Wrap to (-pi/2, pi/2]."""
    half_pi = 0.5 * np.pi
    return (angle_rad + half_pi) % np.pi - half_pi


def init_fD_fDdot_from_doubled_phasor(z_theta, weight, time_s):
    """
    Coarse init for (fD, fD_dot) using doubled-angle phasor z_theta=exp(2j*phase).
    For each fD_dot candidate, dechirp and FFT to pick fD.
    Returns: phi0, fD, fD_dot  (phi0 is for *single* phase, modulo pi)
    """
    fDdot_grid_hz_s = np.linspace(-10.0, 10.0, 41)  # Hz/s
    wsum = np.sum(weight)
    f_bins = np.fft.fftfreq(len(z_theta), d=time_s[1]-time_s[0])

    best_score = -np.inf
    best_fD = 0.0
    best_fDdot = 0.0
    best_phi0_double = 0.0

    for fDdot in fDdot_grid_hz_s:
        # dechirp for doubled-angle model: exp(-j*2*pi*fDdot*t^2)
        z_dechirped = z_theta * np.exp(-1j * 2.0*np.pi * fDdot * (time_s*time_s))

        # weighted FFT (simple amplitude weighting)
        spectrum = np.fft.fft(weight * z_dechirped)
        mag = np.abs(spectrum)
        #mag[0] = 0.0 # ignore DC
        k = int(np.argmax(mag))
        f2 = f_bins[k]                # this is the frequency of z(t), i.e. 2*fD
        fD = 0.5 * f2

        # score via coherent sum at that (fD, fDdot)
        ph = 4.0*np.pi * fD * time_s + 2.0*np.pi * fDdot * (time_s*time_s)
        coherent = np.sum(weight * z_theta * np.exp(-1j * ph))
        score = np.abs(coherent)
        if score > best_score:
            best_score = score
            best_fD = fD
            best_fDdot = fDdot
            best_phi0_double = np.angle(coherent)  # estimates 2*phi0 (mod 2pi)

    # convert doubled-angle phase offset to single-angle phi0 (mod pi)
    phi0 = wrap_to_pi_two(0.5 * best_phi0_double)

    return phi0, best_fD, best_fDdot, best_score

def fit_wrapped_phase_wls(z_theta, weight, jacobian, phi0, fD, fDdot, tol=1e-10, max_iter=50):
    """
    Unit-circle LS fit using doubled angle:
        z_k = exp(2j * phase_rad[k])
        zhat_k(p) = exp(2j * (jacobian[k,:] @ p))
        minimize sum_k w_k * |z_k - zhat_k(p)|^2

    jacobian is the design matrix for the *phase model* (not for zhat).
    [1, 2*pi*t, pi*t^2].
    """
    num_samples = z_theta.size

    params_init = np.array([phi0, fD, fDdot], float)
    params = params_init.copy()

    sqrt_weight = np.sqrt(weight)

    def cost_and_complex_resid(current_params):
        model_phase = jacobian @ current_params
        zhat = np.exp(2j * model_phase)
        resid = z_theta - zhat
        cost = float(np.sum(weight * (resid.real**2 + resid.imag**2)))
        return cost, resid, zhat

    cost, resid, zhat = cost_and_complex_resid(params)

    for iteration in range(1, int(max_iter) + 1):
        # Linearize: zhat(p+dp) ≈ zhat + (dzhat/dp) dp
        # dzhat/dp = 2j * zhat * J   (elementwise zhat times each row of J)
        dzhat_dp = (2j * zhat)[:, None] * jacobian  # (len(phase),3) complex

        # Build real LS system for dp:
        # minimize || sqrt(w) * (resid - dzhat_dp @ dp) ||_2 over complex residual
        A_real = np.vstack([
            (dzhat_dp.real * sqrt_weight[:, None]),
            (dzhat_dp.imag * sqrt_weight[:, None]),
        ])
        b_real = np.hstack([
            (resid.real * sqrt_weight),
            (resid.imag * sqrt_weight),
        ])

        step, *_ = np.linalg.lstsq(A_real, b_real, rcond=None)
        new_params = params + step

        new_cost, resid, zhat = cost_and_complex_resid(new_params)
        rel_improve = (cost - new_cost) / max(cost, 1e-300)

        if np.linalg.norm(step) < tol or (0.0 <= rel_improve < tol):
            return new_params, iteration, new_cost

        params, cost = new_params, new_cost

    return params, int(max_iter), cost

@njit(cache=True)
def wls_step_3x3_from_complex(D, r, w):
    """
    Solve for dx in (Re(D^H W D)) dx = Re(D^H W r)

    D: (N,3) complex128/complex64   (each row is complex Jacobian for sample k)
    r: (N,)  complex128/complex64   (complex residual)
    w: (N,)  float64/float32        (diagonal weights)

    Returns:
      dx: (3,) float64  (real parameter update)
    """
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)

    N = r.shape[0]
    for k in range(N):
        wk = w[k]
        rk = r[k]
        d0 = D[k, 0]
        d1 = D[k, 1]
        d2 = D[k, 2]

        cd0 = np.conjugate(d0)
        cd1 = np.conjugate(d1)
        cd2 = np.conjugate(d2)

        # b += w * Re(conj(d_i) * r)
        b[0] += wk * (cd0 * rk).real
        b[1] += wk * (cd1 * rk).real
        b[2] += wk * (cd2 * rk).real

        # A += w * Re(conj(d_i) * d_j)
        A[0, 0] += wk * (cd0 * d0).real
        A[0, 1] += wk * (cd0 * d1).real
        A[0, 2] += wk * (cd0 * d2).real
        A[1, 1] += wk * (cd1 * d1).real
        A[1, 2] += wk * (cd1 * d2).real
        A[2, 2] += wk * (cd2 * d2).real

    # Symmetrize
    A[1, 0] = A[0, 1]
    A[2, 0] = A[0, 2]
    A[2, 1] = A[1, 2]

    return np.linalg.solve(A, b)

@njit(cache=True)
def fit_wrapped_phase_wls_numba(z_theta, weight, jacobian, phi0, fD, fDdot,
                               tol=1e-10, max_iter=50):
    """
    Unit-circle LS fit using doubled angle:
        z_k = exp(2j * phase_rad[k])
        zhat_k(p) = exp(2j * (jacobian[k,:] @ p))
        minimize sum_k w_k * |z_k - zhat_k(p)|^2

    jacobian: (N,3) real design matrix for phase model [1, 2*pi*t, pi*t^2]
    z_theta: (N,) complex
    weight:  (N,) real weights
    """
    N = z_theta.size
    params = np.array([phi0, fD, fDdot], dtype=np.float64)

    # Work buffers (avoid realloc each iter)
    model_phase = np.empty(N, dtype=np.float64)
    zhat = np.empty(N, dtype=np.complex128)
    resid = np.empty(N, dtype=np.complex128)
    D = np.empty((N, 3), dtype=np.complex128)

    # initial cost
    for k in range(N):
        model_phase[k] = (jacobian[k, 0] * params[0] +
                          jacobian[k, 1] * params[1] +
                          jacobian[k, 2] * params[2])
        zhat[k] = np.exp(2j * model_phase[k])
        resid[k] = z_theta[k] - zhat[k]

    cost = 0.0
    for k in range(N):
        r = resid[k]
        cost += weight[k] * (r.real * r.real + r.imag * r.imag)

    for it in range(1, max_iter + 1):
        # Build D (complex Jacobian for zhat) and residual for current params
        for k in range(N):
            # model + zhat
            model_phase[k] = (jacobian[k, 0] * params[0] +
                              jacobian[k, 1] * params[1] +
                              jacobian[k, 2] * params[2])
            zh = np.exp(2j * model_phase[k])
            zhat[k] = zh

            # residual
            rk = z_theta[k] - zh
            resid[k] = rk

            # D_k = dzhat/dp = 2j * zhat * J_k
            fac = 2j * zh
            D[k, 0] = fac * jacobian[k, 0]
            D[k, 1] = fac * jacobian[k, 1]
            D[k, 2] = fac * jacobian[k, 2]

        step = wls_step_3x3_from_complex(D, resid, weight)
        new_params = params + step

        # compute new cost
        new_cost = 0.0
        for k in range(N):
            mp = (jacobian[k, 0] * new_params[0] +
                  jacobian[k, 1] * new_params[1] +
                  jacobian[k, 2] * new_params[2])
            zh = np.exp(2j * mp)
            r = z_theta[k] - zh
            new_cost += weight[k] * (r.real * r.real + r.imag * r.imag)

        denom = cost if cost > 1e-300 else 1e-300
        rel_improve = (cost - new_cost) / denom

        step_norm = np.sqrt(step[0]*step[0] + step[1]*step[1] + step[2]*step[2])
        if step_norm < tol or (0.0 <= rel_improve < tol):
            return new_params, it, new_cost

        params = new_params
        cost = new_cost

    return params, max_iter, cost

def wls_step(J, r, w):
    """
    Solve (J^T W J) dx = J^T W r
    with diagonal weights w (length N).
    """
    WJ = J * w[:, None]               # each row scaled by w_k
    A = J.T @ WJ                      # J^T W J
    b = J.T @ (w * r)               # -J^T W r
    return np.linalg.solve(A, b)

@njit(cache=True)
def wls_step_numba(J, r, w):
    """
    Solve (J^T W J) dx = (J^T W r)
    with diagonal weights w (length N).
    J: (N,p) float64
    r: (N,) float64
    w: (N,) float64  (use sqrt-weights? no: these are actual weights)
    """
    N, p = J.shape
    A = np.zeros((p, p), dtype=np.float64)
    b = np.zeros(p, dtype=np.float64)

    for k in range(N):
        wk = w[k]
        rk = r[k]
        for i in range(p):
            Ji = J[k, i]
            b[i] += wk * Ji * rk
            for j in range(p):
                A[i, j] += wk * Ji * J[k, j]

    return np.linalg.solve(A, b)


def _acq_coarse_select_slice(prn_code, sample_rate, chip_rate, C_slices, X_blk, T_a, f_IF, N_init,
                            f_center, samples_per_chip, decimation, f_max, num_accum):
    """
    Run coarse acquisition for each 1ms code slice; return best slice index and its results.
    """
    best = dict(CN0=-np.inf, m=0, f=None, t=None)
    for m in range(C_slices.shape[0]):
        f_c, t_c, CN0 = acq_prn_fft(
                C_slices, X_blk,
            T_a, f_IF, N_init, num_accum,
            f_center, samples_per_chip, decimation, f_max, m0=m
        )
        if CN0 > best["CN0"]:
            best.update(CN0=CN0, m=m, f=f_c, t=t_c)

    # second loop -- circshift and test the hypotheses to decide m/m+1 ambiguity
    dt = np.arange(N_init*num_accum) * T_a / N_init
    phase = 2*np.pi*(f_IF + best["f"]) * dt
    mixer = np.exp(-1j * phase)
    X_block = X_blk.ravel() * mixer
    best_score = 0
    code_length_samples = T_a*num_accum*sample_rate
    for m in range(C_slices.shape[0]):
        t_acq_test = best["t"] - (num_accum-1 - m) * T_a
        code_phase_prompt = t_acq_test * chip_rate
        #C_k_prompt = shift_code_phase(C_k, sample_rate, chip_rate, code_phase_prompt)
        C_k_prompt = oversample_prn_code(prn_code, sample_rate, chip_rate, code_length_samples, code_phase_prompt)
        score = np.sum(np.abs(np.vdot(C_k_prompt, X_block)))
        if score > best_score:
            best["m"] = m
            best_score = score

    return best["m"], best["f"], best["t"], best["CN0"]

def _acq_aided_select_slice(prn_code, sample_rate, chip_rate, C_slices, X_blk, T_a, f_IF, N_init,
                            f_center, samples_per_chip, num_accum):
    """
    Run coarse acquisition for each 1ms code slice; return best slice index and its results.
    """
    best = dict(CN0=-np.inf, m=0, t=None)
    for m in range(C_slices.shape[0]):
        t_c, CN0 = acq_aided(C_slices, X_blk, T_a, f_IF, N_init, num_accum, f_center, samples_per_chip, m0=m)
        if CN0 > best["CN0"]:
            best.update(CN0=CN0, m=m, t=t_c)

    # second loop -- circshift and test the hypotheses to decide m/m+1 ambiguity
    dt = np.arange(N_init*num_accum) * T_a / N_init
    phase = 2*np.pi*(f_IF + f_center) * dt
    mixer = np.exp(-1j * phase)
    X_block = X_blk.ravel() * mixer
    best_score = 0
    code_length_samples = T_a*num_accum*sample_rate
    for m in range(C_slices.shape[0]):
        t_acq_test = best["t"] - (num_accum-1 - m) * T_a
        code_phase_prompt = t_acq_test * chip_rate
        #C_k_prompt = shift_code_phase(C_k, sample_rate, chip_rate, code_phase_prompt)
        C_k_prompt = oversample_prn_code(prn_code, sample_rate, chip_rate, code_length_samples, code_phase_prompt)
        score = np.sum(np.abs(np.vdot(C_k_prompt, X_block)))
        if score > best_score:
            best["m"] = m
            best_score = score

    return best["m"], best["t"], best["CN0"]

@njit(cache=True)
def wrap_index_in_block(start_chip_phase, Nchips, sample_rate, chip_rate, N_k):
    # chip phase (chips) at each sample
    chip_time = (np.arange(N_k) * chip_rate / sample_rate) + (start_chip_phase % Nchips)
    chip_idx = np.floor(chip_time).astype(np.int64) % Nchips
    # wrap occurs where chip_idx decreases
    d = np.diff(chip_idx)
    wrap_candidates = np.where(d < 0)[0]
    if wrap_candidates.size == 0:
        return N_k  # no wrap inside this block
    return int(wrap_candidates[0] + 1)  # boundary sample where new code period begins

@njit(cache=True)
def vdot_numba(a, b):
    acc = 0.0 + 0.0j
    n = min(a.shape[0], b.shape[0])
    for i in range(n):
        acc += a[i].conjugate() * b[i]
    return acc

@njit(cache=True)
def get_tap_meas(
    X_blocks: np.ndarray,              # (num_blocks, N_k) complex
    dt_full: np.ndarray,               # (num_blocks*N_k,) float (seconds)
    num_blocks: int,
    N_k: int,
    ranging_code: str,
    prn_code: np.ndarray,              # (Nchips,) +/-1
    sample_rate: float,
    chip_rate: float,
    code_length_samples: int | float,
    eml: float,                        # chips (E-L spacing)
    t_0: float,                        # chips
    t_dot: float,                      # chips/s
    phi_0: float,                      # rad
    f_IF: float,                       # Hz
    f_D: float,                        # Hz
    f_D_dot: float,                    # Hz/s
    do_coherent: bool
):
    """
    Compute correlator-based measurements for one tracking iteration.

    Returns
    -------
    cost : float
    phase_arr : (num_blocks,) float
    d_arr : (num_blocks,) float
    weight_arr : (num_blocks,) float
    dt_true : (num_blocks,) float  # effective time tags per code period
    """

    S_early_arr = []
    S_late_arr = []
    S_prompt_arr = []
    dt_true = []

    for block in range(num_blocks):
        dt = dt_full[block * N_k : (block + 1) * N_k]

        code_phase_prompt = t_0 + t_dot * dt[0]
        if ranging_code == "CL":
            # CL doesn't advance by code period; track chips into the long code sequence
            code_phase_prompt += dt[0] * chip_rate

        code_phase_early = code_phase_prompt - eml / 2.0
        code_phase_late  = code_phase_prompt + eml / 2.0

        C_k_prompt = oversample_prn_code(prn_code, sample_rate, chip_rate, code_length_samples,
                                         code_phase_prompt, t_dot)
        C_k_early  = oversample_prn_code(prn_code, sample_rate, chip_rate, code_length_samples,
                                         code_phase_early,  t_dot)
        C_k_late   = oversample_prn_code(prn_code, sample_rate, chip_rate, code_length_samples,
                                         code_phase_late,   t_dot)

        sample_boundary = wrap_index_in_block(code_phase_prompt, len(prn_code), sample_rate, chip_rate, N_k)

        phase = phi_0 + 2.0 * np.pi * (f_IF + f_D) * dt + np.pi * f_D_dot * dt**2
        X_block = X_blocks[block, :] * np.exp(-1j * phase)

        if ranging_code != "CL":
            dt_true.append(dt[:sample_boundary])
            dt_true.append(dt[sample_boundary:])

            #S_prompt_1 = np.vdot(C_k_prompt[:sample_boundary], X_block[:sample_boundary])
            #S_prompt_2 = np.vdot(C_k_prompt[sample_boundary:],  X_block[sample_boundary:])
            #S_early_1  = np.vdot(C_k_early[:sample_boundary],  X_block[:sample_boundary])
            #S_early_2  = np.vdot(C_k_early[sample_boundary:],  X_block[sample_boundary:])
            #S_late_1   = np.vdot(C_k_late[:sample_boundary],   X_block[:sample_boundary])
            #S_late_2   = np.vdot(C_k_late[sample_boundary:],   X_block[sample_boundary:])
            S_prompt_1 = vdot_numba(C_k_prompt[:sample_boundary], X_block[:sample_boundary])
            S_prompt_2 = vdot_numba(C_k_prompt[sample_boundary:],  X_block[sample_boundary:])
            S_early_1  = vdot_numba(C_k_early[:sample_boundary],  X_block[:sample_boundary])
            S_early_2  = vdot_numba(C_k_early[sample_boundary:],  X_block[sample_boundary:])
            S_late_1   = vdot_numba(C_k_late[:sample_boundary],   X_block[:sample_boundary])
            S_late_2   = vdot_numba(C_k_late[sample_boundary:],   X_block[sample_boundary:])

            S_prompt_arr.extend([S_prompt_1, S_prompt_2])
            S_early_arr.extend([S_early_1,  S_early_2])
            S_late_arr.extend([S_late_1,   S_late_2])
        else:
            dt_true.append(dt)

            #S_prompt = np.vdot(C_k_prompt, X_block)
            #S_early  = np.vdot(C_k_early,  X_block)
            #S_late   = np.vdot(C_k_late,   X_block)
            S_prompt = vdot_numba(C_k_prompt, X_block)
            S_early  = vdot_numba(C_k_early,  X_block)
            S_late   = vdot_numba(C_k_late,   X_block)

            S_prompt_arr.append(S_prompt)
            S_early_arr.append(S_early)
            S_late_arr.append(S_late)

    # Collapse partial intervals back to one value per code period
    if ranging_code != "CL":
        S_prompt_arr = pair_sum_numba(np.asarray(S_prompt_arr))
        S_early_arr  = pair_sum_numba(np.asarray(S_early_arr))
        S_late_arr   = pair_sum_numba(np.asarray(S_late_arr))
        dt_true      = pair_mean_numba(dt_true)
    else:
        S_prompt_arr = np.asarray(S_prompt_arr)
        S_early_arr  = np.asarray(S_early_arr)
        S_late_arr   = np.asarray(S_late_arr)
        dt_true      = np.asarray([np.mean(x) for x in dt_true], dtype=np.float64)

    I_arr = np.real(S_prompt_arr)
    Q_arr = np.imag(S_prompt_arr)

    # squared phase discriminator
    if not do_coherent:
        E_arr = np.abs(S_early_arr)
        L_arr = np.abs(S_late_arr)
        # on first iter, use safe noise-resistant discriminators
        d_arr = 0.5 * (E_arr - L_arr) / (E_arr + L_arr)
        phase_arr = 0.5 * np.arctan2(2.0 * I_arr * Q_arr, I_arr**2 - Q_arr**2)
    else:
        # on subsequent iterations, hone in with decision-directed phase and dot product discriminator
        d_arr = np.real((S_early_arr-S_late_arr)*np.conjugate(S_prompt_arr))/(2*np.abs(S_prompt_arr)**2)
        hard_bits = np.sign(np.real(S_prompt_arr))
        phase_arr = np.arctan2(Q_arr*hard_bits, I_arr*hard_bits)

    weight_arr = (I_arr**2 + Q_arr**2) / np.median(np.abs(S_prompt_arr) ** 2)

    cost_tau = np.sum(weight_arr * d_arr**2 * 1e4)
    cost_phi = np.sum(weight_arr * (phase_arr / (2.0 * np.pi)) ** 2)

    return cost_tau, cost_phi, phase_arr, d_arr, weight_arr, dt_true, S_prompt_arr


UTC2GPS = 18
def track(is_complex, f_sky, i_out, q_out, source, rc_dir, vdif_stats, sample_rate, f_IF, \
        acq_cadence, store_handle, antenna_handle, aided, model, time_gps, vdif_file_handle):
    """
    Track GNSS signals in a chunk of baseband VDIF
    """
    sys = source[0]; prn=source[1:]
    if model is None:
        freq_codes = RANGING_CODES[sys]
        ranging_codes = freq_codes[f_sky]
    else:
        ranging_codes = model.keys()
    if is_complex:
        analytic_signal = np.array(i_out) + 1j*np.array(q_out)
    else:
        analytic_signal = analytic_signal_fft_1d(i_out)
    analytic_signal -= np.mean(analytic_signal)

    model_new = {}
    soft_bits_rc = {}
    hard_bits_rc = {}
    for ranging_code in ranging_codes:
        print(f'tracking ranging code {ranging_code}:')
        code_period = CODE_PERIODS[ranging_code]
        chip_rate = CHIP_RATES[ranging_code]
        eml = CODE_EML[ranging_code] 
        scp = BOC_SCP[ranging_code]

        if ranging_code == 'CL':
            code_period = CODE_PERIODS['CM']
        code_length_samples = code_period * sample_rate
        N_k = int(sample_rate * code_period) # block size 1 code period (coherent integration time)
        samples_per_chip = int(round(sample_rate/chip_rate))
        num_blocks = int(np.floor(len(i_out) / N_k))

        if len(analytic_signal) < sample_rate:
            # last, incomplete block--> drop last ms
            analytic_signal = analytic_signal[:num_blocks*N_k]
 
        X_blocks = analytic_signal.reshape(-1, N_k)

        prn_code = get_replica(rc_dir, ranging_code, str(int(prn)))
        C_k = oversample_prn_code(prn_code, sample_rate, chip_rate, code_length_samples)
        data_len_s = len(i_out) / sample_rate

        if model is None:
            if aided:
                # prepare for aided acquisition
                times_gps = np.array([time_gps, time_gps + np.timedelta64(int(data_len_s), 's')])
                antenna_handle.times_gps = times_gps
                source_array = np.repeat(source, len(times_gps))
                _, doppler_model = store_handle.sim_pr_simple(antenna_handle, times_gps, antenna_handle.bulk_clock*np.ones(len(times_gps)), f_sky, source_array)
                f_acq_init = doppler_model[0]
                f_acq_end = doppler_model[-1]

            low_decimation = 20
            high_decimation = 2000
            sbm = SECONDARY_BIT_MS[ranging_code]
            sbm = 0
            # sbm = 0 -- sometimes we need to switch this off--can get unlucky in either direction
            if sbm == 0: sbm = np.inf

            if sbm > 1: # > 1 ms
                T_a = 1e-3 # coherent integration time per block
                factor=1
            else:
                # using a 1 ms coherent integration will result in destructive interference across a bit flip
                # we thus need to sub-divide further to avoid this problem
                # find the best factor < 10 that evenly divides the number of samples
                for factor in range(2,10):
                    if len(analytic_signal)%(factor*1e3) == 0:
                        # get an integer division of 1 ms
                        T_a = 1e-3/factor
                        break
            N_init = int(sample_rate * T_a)
            n_code = int(code_period / T_a)
            X_acq = analytic_signal.reshape(-1, N_init)
            
            # ----- INIT (start) -----
            if n_code == 1:
                num_accum = 5
                X_init = X_acq[:num_accum,:]
                X_end = X_acq[-num_accum:,:] # NB: this indexing only works if data_len_s evenly divides by code_period 
                if aided:
                    t_acq_init, C_N0_init = acq_aided(C_k, X_init, T_a, f_IF, N_init, num_accum,\
                            f_acq_init, samples_per_chip) # fine acquisition
                    t_acq_end, C_N0_end = acq_aided(C_k, X_end, T_a, f_IF, N_init, num_accum,\
                            f_acq_end, samples_per_chip) # fine acquisition
                else:
                    f_acq_coarse, t_acq_coarse, C_N0_coarse = acq_prn_fft(C_k, X_init, T_a, f_IF, N_init, num_accum,\
                            0.0, samples_per_chip, low_decimation, 5e3) # coarse acquisition
                    f_acq_init, t_acq_init, C_N0_init = acq_prn_fft(C_k, X_init, T_a, f_IF, N_init, num_accum,\
                            f_acq_coarse, samples_per_chip, high_decimation, 2e2) # fine acquisition

                    f_acq_coarse, t_acq_coarse, C_N0_coarse = acq_prn_fft(C_k, X_end, T_a, f_IF, N_init, num_accum,\
                            0.0, samples_per_chip, low_decimation, 5e3) # coarse acquisition
                    f_acq_end, t_acq_end, C_N0_end = acq_prn_fft(C_k, X_end, T_a, f_IF, N_init, num_accum,\
                            f_acq_coarse, samples_per_chip, high_decimation, 2e2) # fine acquisition
            else:
                num_accum = n_code
                C_slices = C_k.reshape(n_code, N_init)
                X_init = np.asarray(X_acq[:num_accum,:], dtype=np.complex64)
                X_end = np.asarray(X_acq[-num_accum:,:], dtype=np.complex64)

                if aided:
                    m_init, t_acq_init, C_N0_init = _acq_aided_select_slice(
                        prn_code, sample_rate, chip_rate, C_slices, X_init, T_a, f_IF, N_init,
                        f_center=f_acq_init, samples_per_chip=samples_per_chip, num_accum=num_accum)
                    m_end, t_acq_end, C_N0_end = _acq_aided_select_slice(
                        prn_code, sample_rate, chip_rate, C_slices, X_end, T_a, f_IF, N_init,
                        f_center=f_acq_end, samples_per_chip=samples_per_chip, num_accum=num_accum)
                else:
                    # Coarse: pick best 1ms slice
                    m_init, f_acq_coarse, t_coarse, C_N0_coarse = _acq_coarse_select_slice(
                        prn_code, sample_rate, chip_rate, C_slices, X_init, T_a, f_IF, N_init,
                        f_center=0.0, samples_per_chip=samples_per_chip,
                        decimation=low_decimation, f_max=5e3, num_accum=num_accum)
                    # Fine: only on winning slice, centered at its coarse Doppler
                    f_acq_init, t_acq_init, C_N0_init = acq_prn_fft(C_slices, X_init, T_a, f_IF, N_init, \
                            num_accum, f_acq_coarse, samples_per_chip, high_decimation, 2e2, m0=m_init) # fine acquisition

                    m_end, f_acq_coarse, t_coarse, C_N0_coarse = _acq_coarse_select_slice(
                        prn_code, sample_rate, chip_rate, C_slices, X_end, T_a, f_IF, N_init, 
                        f_center=0.0, samples_per_chip=samples_per_chip,
                        decimation=low_decimation, f_max=5e3, num_accum=num_accum)
                    f_acq_end, t_acq_end, C_N0_end = acq_prn_fft(C_slices, X_end, T_a, f_IF, N_init, \
                            num_accum, f_acq_coarse, samples_per_chip, high_decimation, 2e2, m0=m_end) # fine acquisition

                t_acq_init -= (num_accum-1 - m_init) * T_a
                t_acq_end -= (num_accum-1 - m_end) * T_a

                doppler_avg = (f_acq_init + f_acq_end)/2
                range_rate_doppler = doppler_avg / f_sky
                range_rate_code = t_acq_end - t_acq_init 
                #if np.abs(range_rate_code)>10*np.abs(range_rate_doppler) and aided:
                if aided:
                    #print('adjusting code phase due to inconsistent range rate')
                    # we had an unlucky alignment that resulted in destructive interference.
                    # use Doppler to find the code phase change instead
                    if C_N0_init > C_N0_end:
                        # implicit multiplication by 1 sec
                        t_acq_end = t_acq_init + range_rate_doppler
                    else:
                        t_acq_init = t_acq_end - range_rate_doppler

                if ranging_code == 'CM':
                    # hand off from CM to CL
                    # we will keep coherent intervals at 20 ms (CM code length)
                    print('handing off to CL...')
                    ranging_code = 'CL'
                    code_period_cl = CODE_PERIODS[ranging_code]
                    eml = CODE_EML[ranging_code] 
                    prn_code = get_replica(rc_dir, ranging_code, str(int(prn)))
                    #code_length_samples = code_period_cl * sample_rate
                    code_frac_prompt_cm = t_acq_init * chip_rate

                    #C_k = oversample_prn_code(prn_code, sample_rate, chip_rate, code_length_samples, code_frac_prompt_cm)
                    num_offsets = int(code_period_cl/code_period) # 75

                    best_offset = np.nan
                    best_score = 0
                    scores = []
                    dt_full = np.arange(N_k*num_blocks) * code_period / N_k 
                    f_D = f_acq_init
                    f_D_dot = (f_acq_end - f_acq_init) / data_len_s  # Hz/s
                    phase = 2*np.pi*(f_IF + f_D) * dt_full + np.pi * f_D_dot * dt_full**2
                    mixer = np.exp(-1j * phase)
                    mixed_signal = analytic_signal*mixer

                    samples_per_cm = int(code_period * sample_rate)
                    # Score each possible CM-aligned offset as coherent sum across blocks with circular shift
                    scores = []
                    best_offset = np.nan
                    best_score = 0.0
                    for offset in range(num_offsets):
                        #code_indices = (int(code_frac_prompt_cm) + samples_per_cm*offset \
                        #        + np.arange(code_length_samples, dtype=int)) % int(code_length_samples)
                        #C_k_prompt = C_k[code_indices]
                        #S_prompt = np.vdot(C_k_prompt[:len(mixed_signal)], mixed_signal)
                        code_frac_prompt = code_frac_prompt_cm + offset*code_period*chip_rate
                        C_k_prompt = oversample_prn_code(prn_code, sample_rate, chip_rate, len(mixed_signal), code_frac_prompt)
                        S_prompt = np.vdot(C_k_prompt, mixed_signal)
                        score = np.abs(S_prompt)
                        scores.append(score)
                        if score > best_score:
                            best_score = score
                            best_offset = offset

                    t_acq_init += best_offset*code_period
                    t_acq_end += best_offset*code_period 

            if C_N0_init < 40 and C_N0_end < 40:
                continue # no detection of the ranging code -- go to next one

            t_0 = t_acq_init * chip_rate
            t_dot = (t_acq_end - t_acq_init) * chip_rate / data_len_s  # chip/s
            f_D = f_acq_init
            f_D_dot = (f_acq_end - f_acq_init) / data_len_s  # Hz/s

            code_frac_prompt = t_0
            #C_k_prompt = shift_code_phase(C_k, sample_rate, chip_rate, code_frac_prompt)
            C_k_prompt = oversample_prn_code(prn_code, sample_rate, chip_rate, code_length_samples, code_frac_prompt)
            dt = np.arange(N_k) * code_period / N_k
            mixer = np.exp(-1j * 2*np.pi*((f_IF+f_acq_init)*dt)) 
            S_prompt_init = np.sum(X_blocks[0,:] * mixer * C_k_prompt)
            phi_0 = np.angle(S_prompt_init)
            phi_0_hold = phi_0
            C_N0 = C_N0_init
            print(f'starting state: t_0 = {t_0}, t_dot = {t_dot}, phi_0 = {phi_0}, f_D = {f_D}, f_D_dot = {f_D_dot}, C_N0 = {C_N0}')
        else:
            # use model from previous iteration to initiate tracking in this iteration
            t_0_model, t_dot_model, phi_0_model, f_D_model, f_D_dot_model, C_N0_model = model[ranging_code]
            t_0 = t_0_model + t_dot_model*data_len_s
            if ranging_code == 'CL':
                t_0 += data_len_s*chip_rate
            t_dot = t_dot_model
            phi_0 = phi_0_model + 2*np.pi*f_D_model*data_len_s + np.pi*f_D_dot_model*data_len_s**2
            phi_0_hold = phi_0
            f_D = f_D_model + f_D_dot_model*data_len_s
            f_D_dot = f_D_dot_model
            code_frac_prompt = t_0
            C_N0 = C_N0_model
            print(f'starting state: t_0 = {t_0}, t_dot = {t_dot}, phi_0 = {phi_0}, f_D = {f_D}, f_D_dot = {f_D_dot}')

        cost = np.inf
        tolerance = 1e1
        max_iter = 15

        # batch adjustment to get code phase and ADR
        prev_cost = np.inf
        rel_tol = 1e-4
        dt_full = np.arange(N_k*num_blocks) * code_period / N_k 
        t_blk = (np.arange(num_blocks) + 0.5) * code_period # + 0.5 -- sample at middle of block interval

        # temp -- test for Doppler, code phase at each epoch for comparison
        #C_N0_arr = []
        #f_acq_arr = []
        #t_acq_arr = []
        #for block in range(num_blocks):
        #    X_block = X_blocks[block,:]
        #    f_acq_coarse, t_acq_coarse, C_N0_coarse = acq_prn_fft(C_k, X_block, code_period, f_IF, N_k, num_accum, 0.0, code_period, 1, 5e3) # coarse acquisition
        #    f_acq, t_acq, C_N0 = acq_prn_fft(C_k, X_block, code_period, f_IF, N_k, num_accum, f_acq_coarse, code_period, 10, 1.5e2) # fine acquisition
        #    f_acq_arr.append(f_acq)
        #    t_acq_arr.append(t_acq*chip_rate)
        #    C_N0_arr.append(C_N0)
        #fig_code, ax_code = plt.subplots(dpi=300)
        ##ax_code.plot(t_blk, t_acq_arr, label='acquisition')
        #ax_code.set_xlabel('t (sec)')
        #ax_code.set_ylabel('code offset (chips)')
        #fig_doppler, ax_doppler = plt.subplots(dpi=300)
        ##ax_doppler.plot(t_blk, f_acq_arr, label='acquisition')
        #ax_doppler.set_xlabel('t (sec)')
        #ax_doppler.set_ylabel('Doppler freq. (Hz)')
        #fig_IQ, ax_IQ = plt.subplots(dpi=300)
        #ax_IQ.set_xlabel('t (sec)')
        #ax_IQ.set_ylabel('Correlation mag.')
        #fig_CNO, ax_CNO = plt.subplots(dpi=300)
        #ax_CNO.plot(t_blk, C_N0_arr, label='acquisition')
        #ax_CNO.set_xlabel('t (sec)')
        #ax_CNO.set_ylabel('C/N0 (dB-Hz)')
        #fig_CNO.savefig('CNO_tracking.png')
        #plt.close(fig_CNO)

        #fig_psd, (ax_psd_bb, ax_psd_real) = plt.subplots(1, 2, dpi=300, figsize=(10, 4))
        #
        ## Analytic signal PSD, zoomed to signal band
        ##bw_plot = 128e6
        #ax_psd_bb.psd(analytic_signal, NFFT=min(4096, len(analytic_signal)),
        #              Fs=sample_rate, scale_by_freq=True)
        #ax_psd_bb.set_xlim(0, sample_rate/2)
        ##ax_psd_bb.set_xlim(15.02e6, 17.02e6)
        #ax_psd_bb.set_ylim(-80, -60)
        #ax_psd_bb.set_title(f'{source} {ranging_code} analytic')
        #ax_psd_bb.set_xlabel('Frequency (Hz)')
        #ax_psd_bb.set_ylabel('PSD (dB-Hz)')
        #
        ## Original real signal PSD
        #ax_psd_real.psd(np.array(i_out, dtype=np.float64), NFFT=min(4096, len(i_out)),
        #                Fs=sample_rate, scale_by_freq=True)
        #ax_psd_real.set_xlim(0, sample_rate/2)
        ##ax_psd_real.set_xlim(15.02e6, 17.02e6)
        #ax_psd_real.set_ylim(-80, -60)
        #ax_psd_real.set_title(f'{source} {ranging_code} real (I)')
        #ax_psd_real.set_xlabel('Frequency (Hz)')
        #ax_psd_real.set_ylabel('PSD (dB-Hz)')
        #
        #fig_psd.tight_layout()
        #fig_psd.savefig(f'psd_diag_{source}_{ranging_code}.png')
        #plt.close(fig_psd)

        #fig_psd, ax_psd = plt.subplots(1, 1, dpi=300, figsize=(5, 4))
        #
        ## Original real signal PSD
        #ax_psd.psd(np.array(i_out, dtype=np.float64), NFFT=min(4096, len(i_out)),
        #                Fs=sample_rate, scale_by_freq=True)
        ##ax_psd.set_xlim(0, sample_rate/2)
        ##ax_psd_real.set_xlim(15.02e6, 17.02e6)
        #ax_psd.set_ylim(-80, -60)
        #ax_psd.set_title(f'{source} {ranging_code} real (I)')
        #ax_psd.set_xlabel('Frequency (Hz)')
        #ax_psd.set_ylabel('PSD (dB-Hz)')
        #
        #fig_psd.tight_layout()
        #fig_psd.savefig(f'psd_diag_{source}_{ranging_code}.png')
        #plt.close(fig_psd)
        #if model is None:
        #    fig_psd, ax_psd = plt.subplots(1, 1, dpi=300, figsize=(5, 4))
        #    # Analytic signal PSD, zoomed to signal band
        #    ax_psd.psd(analytic_signal, NFFT=min(4096, len(analytic_signal)),
        #                    Fs=sample_rate, scale_by_freq=True)
        #    ax_psd.set_ylim(-80, -60)
        #    ax_psd.set_xlim(0, sample_rate/2)
        #    ax_psd.set_title(f'{source} {ranging_code} complex (I/Q)')
        #    ax_psd.set_xlabel('Frequency (Hz)')
        #    ax_psd.set_ylabel('PSD (dB-Hz)')
        #    
        #    fig_psd.tight_layout()
        #    fig_psd.savefig(f'psd_diag_{source}_{ranging_code}_{antenna_handle.antenna_name}_{str(f_sky)}_{vdif_file_handle.split("_")[-1]}.png')
        #    plt.close(fig_psd)
       
        if  False: #ranging_code == 'E1C':
            offset_test=0.5
            #offset_test=100
            test_chip = np.linspace(-offset_test, offset_test, 1000)
            style='beg'
            #style='end'
            if style == 'beg':
                code_phase_prompt = t_0 #+ t_dot * t_blk[0]
                dt = dt_full[0:N_k]
            elif style == 'end':
                code_phase_prompt = t_0 + t_dot * t_blk[-1]
                dt = dt_full[(num_blocks-1)*N_k:num_blocks*N_k]
            if ranging_code == 'CL' and style == 'end':
                # we arent repeating the code w/ CL, just advancing it b/c
                # N_k is not a full code period
                code_phase_prompt += dt[0]*chip_rate
            code_phase_samples = code_phase_prompt / chip_rate * sample_rate
            #sample_boundary = int(np.floor(code_phase_samples))
            sample_boundary =  wrap_index_in_block(code_phase_prompt, len(prn_code), sample_rate, chip_rate, N_k)
            phase = phi_0 + 2*np.pi*(f_IF + f_D) * dt + np.pi * f_D_dot * dt**2
            mixer = np.exp(-1j * phase)
            if style == 'beg':
                X_block = X_blocks[0,:] * mixer
            elif style == 'end':
                X_block = X_blocks[-1,:] * mixer
            def prompt_corr(code_phase_prompt_offset):
                code_frac_prompt = code_phase_prompt_offset
                #C_p = shift_code_phase(C_k, sample_rate, chip_rate, code_frac_prompt)
                C_p = oversample_prn_code(prn_code, sample_rate, chip_rate, code_length_samples, code_frac_prompt, t_dot)
                C_e = oversample_prn_code(prn_code, sample_rate, chip_rate, code_length_samples, code_frac_prompt+eml/2, t_dot)
                C_l = oversample_prn_code(prn_code, sample_rate, chip_rate, code_length_samples, code_frac_prompt-eml/2, t_dot)
                if ranging_code != 'CL':
                    P_1 = np.vdot(C_p[:sample_boundary], X_block[:sample_boundary])
                    P_2 = np.vdot(C_p[sample_boundary:], X_block[sample_boundary:])
                else:
                    P = np.vdot(C_p, X_block)
                    return P, P 
                return P_1, P_2

            def eml_corr(code_phase_prompt_offset):
                code_frac_prompt = code_phase_prompt_offset
                C_e = oversample_prn_code(prn_code, sample_rate, chip_rate, code_length_samples, code_frac_prompt+eml/2, t_dot)
                C_l = oversample_prn_code(prn_code, sample_rate, chip_rate, code_length_samples, code_frac_prompt-eml/2, t_dot)
                if ranging_code != 'CL':
                    E_1 = np.abs(np.vdot(C_e[:sample_boundary], X_block[:sample_boundary]))
                    E_2 = np.abs(np.vdot(C_e[sample_boundary:], X_block[sample_boundary:]))
                    L_1 = np.abs(np.vdot(C_l[:sample_boundary], X_block[:sample_boundary]))
                    L_2 = np.abs(np.vdot(C_l[sample_boundary:], X_block[sample_boundary:]))
                    E = E_1+E_2
                    L = L_1+L_2
                else:
                    E = np.vdot(C_e, X_block)
                    L = np.vdot(C_l, X_block)
                d = (E - L) / (E + L)
                return d

            power_arr = []
            d_arr = []
            beta = 1.0/scp # BOC subcarrier period non-BOC --> 1
            Gd = (2-(2-beta)*eml)/(2*(2-beta))
            for chip in test_chip:
                code_offset = code_phase_prompt + chip
                if style =='beg':
                    _, P = prompt_corr(code_offset)
                    d = eml_corr(code_offset)
                elif style == 'end':
                    P, _ = prompt_corr(code_offset)
                    d = eml_corr(code_offset)
                power = np.abs(P)**2
                power_arr.append(power)
                d_arr.append(d)
            power_arr = np.array(power_arr)
            power_arr /= np.amax(power_arr)
            d_arr = np.array(d_arr)
            d_arr_corr = -Gd*d_arr
            good_idxs = np.abs(test_chip)<0.05
            fig_chip, ax_chip = plt.subplots(dpi=300)
            ax_chip.plot(test_chip, power_arr, label='tau')
            ax_chip.plot(d_arr_corr[good_idxs], power_arr[good_idxs], label='k E-L/E+L')
            ax_chip.set_xlabel('Offset (chips)')
            ax_chip.set_ylabel('Correlation function')
            ax_chip.legend()
            fig_chip.savefig('corr_func_' + ranging_code + '.png')

            good_idxs_calc = np.abs(test_chip)<10
            beta = 1.0/scp # BOC subcarrier period non-BOC --> 1
            Gd_est = -d_arr[good_idxs_calc]/test_chip[good_idxs_calc]
            Gd_est_scalar = np.median(Gd_est)
            fig_gd, ax_gd = plt.subplots(dpi=300)
            ax_gd.plot(test_chip[good_idxs_calc], -d_arr[good_idxs_calc], label='data')
            ax_gd.plot(test_chip[good_idxs_calc], 1/Gd*test_chip[good_idxs_calc], label='(2-(2-beta)*eml)/(2*(2-beta))')
            ax_gd.set_xlabel('Offset (chips)')
            ax_gd.set_ylabel('(E-L)/(E+L)')
            ax_gd.legend()
            fig_gd.savefig('gd_func_' + ranging_code + '.png')
            breakpoint()
            #Gd_data = compute_Gd_numeric_noncoherent(X_blocks, t_blk, prn_code, code_length_samples, sample_rate, \
            #        chip_rate, code_period, f_IF, t_0, t_dot, phi_0, f_D, f_D_dot, eml, N_k, num_blocks)


        #Gd = compute_Gd_numeric_noncoherent(X_blocks, t_blk, C_k, sample_rate, chip_rate, code_period, f_IF, t_0, t_dot, phi_0, f_D, f_D_dot, eml, N_k, num_blocks)
        #Gd = compute_Gd_numeric_coherent(X_blocks, t_blk, C_k, sample_rate, chip_rate, code_period, f_IF, t_0, t_dot, phi_0, f_D, f_D_dot, eml, N_k, num_blocks)
        #print(f'Gd: {Gd}')

        beta = 1.0/BOC_SCP[ranging_code] # BOC subcarrier period non-BOC --> 1
        Gd = (2-(2-beta)*eml)/(2*(2-beta)) # analytic slope of true offset to code phase discriminator (E-L)/(E+L)
        do_coherent = False
        for it in range(max_iter):
            if it >0 and C_N0 > 45: 
                Gd=1 # dot product discriminator has unit gain
                do_coherent = True
            cost_tau, cost_phi, phase_arr, d_arr, weight_arr, dt_true, S_prompt_arr = get_tap_meas(X_blocks, dt_full, num_blocks, N_k, ranging_code,\
                prn_code, sample_rate, chip_rate, code_length_samples, eml, t_0, t_dot, phi_0, f_IF, f_D, f_D_dot, do_coherent)
            cost = cost_tau + cost_phi

            if it == 0:
                rel_impr = np.inf
            else:
                rel_impr = np.abs(prev_cost - cost) / prev_cost
           
            J_tau = np.column_stack([-np.ones(num_blocks)*Gd, -dt_true*Gd])
            J_phi = np.column_stack([np.ones(num_blocks), 2*np.pi * dt_true, np.pi * dt_true**2])
            dx_t   = wls_step_numba(J_tau, d_arr, weight_arr)

            z_theta = np.exp(2j * phase_arr)
            phi0, fD, fDdot, score = init_fD_fDdot_from_doubled_phasor(z_theta, weight_arr, dt_true)
            dx_phi, max_iter, phase_cost = fit_wrapped_phase_wls_numba(z_theta, weight_arr, J_phi, phi0, fD, fDdot, tol=1e-6, max_iter=50)

            #code_offset_time = d_arr + t_0 + t_dot*dt_true
            #res = linregress(t_blk, code_offset_time)
            #t_0_comp = res.intercept
            #t_dot_comp = res.slope
            #line_sp = res.intercept - t_0 + (res.slope-t_dot)*t_blk 
            #line_lsq = -dx_t[0] -dx_t[1]*dt_true
            #fig_test, ax_test = plt.subplots(dpi=300)
            #ax_test.plot(t_blk, d_arr, label='meas')
            #ax_test.plot(t_blk, line_sp, label='sp plot')
            #ax_test.plot(t_blk, Gd*line_lsq, label='lsq')
            #ax_test.set_xlabel('t (sec)')
            #ax_test.set_ylabel('code offset (chips)')
            #ax_test.legend()
            #fig_test.savefig('code_offset_' + antenna_handle.antenna_name + '_'+ ranging_code + '_' + str(f_sky) + '.png')
            #plt.close(fig_test)

            #if it == 10:
            #    model_check = dx_phi[0] + 2*np.pi*dx_phi[1]*t_blk + np.pi*dx_phi[2]*t_blk**2
            #    fig_test, ax_test = plt.subplots(dpi=300)
            #    ax_test.plot(t_blk, phase_arr, label='meas')
            #    ax_test.plot(t_blk, wrap_to_pi_two(model_check), label='model')
            #    ax_test.plot(t_blk, phase_arr-wrap_to_pi_two(model_check), label='diff')
            #    ax_test.set_xlabel('t (sec)')
            #    ax_test.set_ylabel('carrier offset (radians)')
            #    ax_test.legend()
            #    fig_test.savefig('carrier_offset_' + antenna_handle.antenna_name + '_' + ranging_code + '_' + str(f_sky) + vdif_file_handle.split("_")[-1]+'.png')
            #    plt.close(fig_test)
            
            phi_0 += dx_phi[0]
            f_D += dx_phi[1]
            f_D_dot += dx_phi[2]
            t_0 += dx_t[0]
            t_dot += dx_t[1]
            prev_cost = cost

            print(f'iter {it}: cost_tau = {cost_tau}, cost_phi = {cost_phi}')
            print(f'state: t_0 = {t_0}, t_dot = {t_dot}, phi_0 = {phi_0}, f_D = {f_D}, f_D_dot = {f_D_dot}')
            if rel_impr < rel_tol:
                print('finished tracking! breaking')
                break
        #ax_code.legend()
        #fig_code.savefig('code_tracking.png')
        #plt.close(fig_code)
        #ax_doppler.legend()
        #fig_doppler.savefig('doppler_tracking.png')
        #plt.close(fig_doppler)
     
        # extract soft bits
        soft_bits = np.real(S_prompt_arr)
        hard_bits = np.sign(soft_bits)

        # get C/N0 after tracking -- use narrowband wideband power ratio method
        z_wiped = S_prompt_arr * hard_bits  # only if you expect prompt to be mostly real under lock
        z_wiped = z_wiped[1:] # get rid of first partial interval
        
        # simple K = 1 version
        #NP = np.sum(z_wiped.real)**2 + np.sum(z_wiped.imag)**2
        #WP = np.sum(np.abs(z_wiped)**2)
        #mu = NP/WP
        #C_N0 = 10*np.log10(1/code_period*(mu-1)/(len(z_wiped)-mu))

        # --- block-averaged NWPR ---
        target_block_time = 40e-3  # 40 ms
        N_block = max(2, int(round(target_block_time / code_period)))
        K = len(z_wiped) // N_block
        if K < 1:
            raise ValueError(f"Need at least {N_block} prompts; have {len(z_wiped)}")
        
        z_blocks = z_wiped[:K * N_block].reshape(K, N_block)
        NP_k = np.abs(z_blocks.sum(axis=1))**2          # |Σ r|² per block (narrowband)
        WP_k = np.sum(np.abs(z_blocks)**2, axis=1)      # Σ |r|²  per block (wideband)
        mu_k = NP_k / WP_k
        mu_bar = mu_k.mean()
        
        C_N0 = 10 * np.log10((1.0 / code_period) * (mu_bar - 1.0) / (N_block - mu_bar))
        mu_std = mu_k.std(ddof=1) if K > 1 else np.nan  # runtime quality metric
        print(f'final C_N0: {C_N0}')

        if C_N0 < 35 and len(ranging_codes) == 1 and model is None:
            # nothing to do, we lost the signal. Maybe it isn't there at all
            exit()
        #HRT_CN0 = 10*np.log10(1/code_period*(np.abs(np.sum(z_wiped*z_wiped))/(2*len(z_wiped)*N_k)-1))

        n_slip = int(np.rint((phi_0 - phi_0_hold) / np.pi))
        if n_slip != 0:
            phi_0 -= n_slip * np.pi

        soft_bits_rc[ranging_code] = soft_bits
        hard_bits_rc[ranging_code] = hard_bits
        model_new[ranging_code] = (t_0, t_dot, phi_0, f_D, f_D_dot, C_N0)

        # temp -- test full model
        #dt_full = np.arange(N_k*num_blocks) * code_period / N_k 
        #phase = phi_0 + 2*np.pi*(f_D + f_IF) * dt_full + np.pi * f_D_dot * dt_full**2
        #mixer = np.exp(-1j * phase)
        #X_full = analytic_signal * mixer
        #code_phase_prompt = t_0
        #best_score = 0
        #code_length_samples = len(analytic_signal)
        #C_k_prompt = oversample_prn_code(prn_code, sample_rate, chip_rate, code_length_samples, code_phase_prompt, t_dot)
        #DC_signal = np.conj(X_full) * C_k_prompt
        #DC_signal *= DC_signal # remove bit flips

        #def block_average(x, N):
        #    return x[:len(x) - len(x) % N].reshape(-1, N).mean(axis=1)

        #def running_mean(x, N):
        #    return np.convolve(x, np.ones(N)/N, mode='valid')

        #def time_for_window(dt_full, N_win, N_avg):
        #    """Trim time array to match running_mean output, then block average."""
        #    n_rm = len(dt_full) - N_win + 1
        #    return block_average(dt_full[N_win//2 : N_win//2 + n_rm], N_avg)
        #    # centering the window offset so timestamps sit at the middle of each window
        #
        #N_avg = 1000
        #N_rc = int(code_period*sample_rate)
        #time_arr_10000  = time_for_window(dt_full, 10000,  N_avg)
        #time_arr_100000 = time_for_window(dt_full, 100000, N_avg)
        #time_arr_rc    = time_for_window(dt_full, N_rc,   N_avg)
        #
        #print('1')
        #mean_signal_10000 = running_mean_block_average(DC_signal, 10000, N_avg)
        #phase_arr_10000 = 0.5 * np.arctan2(2.0 * mean_signal_10000.real * mean_signal_10000.imag,\
        #        mean_signal_10000.real**2 - mean_signal_10000.imag**2)
        #print('2')
        #mean_signal_100000 = running_mean_block_average(DC_signal, 100000, N_avg)
        #phase_arr_100000 = 0.5 * np.arctan2(2.0 * mean_signal_100000.real * mean_signal_100000.imag,\
        #        mean_signal_100000.real**2 - mean_signal_100000.imag**2)
        #print('3')
        #mean_signal_rc = running_mean_block_average(DC_signal, N_rc, N_avg)
        #phase_arr_rc = 0.5 * np.arctan2(2.0 * mean_signal_rc.real * mean_signal_rc.imag,\
        #        mean_signal_rc.real**2 - mean_signal_rc.imag**2)
        #print('calc done')

        #fig_1, ax_1 = plt.subplots(dpi=300)
        #ax_1.scatter(time_arr_10000,  phase_arr_10000,  label='10000')
        #ax_1.scatter(time_arr_100000, phase_arr_100000, label='100000')
        #ax_1.scatter(time_arr_rc,    phase_arr_rc,    label='rc')
        #ax_1.set_xlabel('t (sec)')
        #ax_1.set_ylabel('phase (rad)')
        #ax_1.set_title(f'source {source}')
        #ax_1.legend()
        #fig_1.savefig('model_test_' + antenna_handle.antenna_name + '_'+source+ '_' + str(f_sky) + vdif_file_handle.split("_")[-1]+'.png')
        #plt.close(fig_1)
        #breakpoint()

        #ax_1.scatter(time_100, mean_signal_100.imag, label='100')
        #ax_1.scatter(time_1000, mean_signal_1000.imag, label='1000')
        ##ax_1.scatter(time_5000, mean_signal_5000.imag, label='5000')
        #ax_1.scatter(time_rc, mean_signal_rc.imag, label='rc')
        


    return soft_bits_rc, hard_bits_rc, model_new

@njit(cache=True)
def running_mean_block_average(x, N_win, N_avg):
    """
    Equivalent to block_average(running_mean(x, N_win), N_avg)
    but fused into one pass with a sliding window sum.
    O(len(x)) regardless of N_win.
    """
    n_rm  = len(x) - N_win + 1          # running-mean output length
    n_out = n_rm // N_avg                # block-average output length
    out   = np.empty(n_out, dtype=x.dtype)

    # --- initial window sum ---
    zero = x[0] - x[0]                  # dtype-safe zero (works for complex)
    win_sum = zero
    for j in range(N_win):
        win_sum += x[j]

    # --- fused sweep ---
    block_sum   = zero
    block_count = 0
    out_idx     = 0

    for i in range(n_rm):
        if i > 0:
            win_sum += x[i + N_win - 1] - x[i - 1]

        block_sum   += win_sum           # accumulate raw window sums
        block_count += 1

        if block_count == N_avg:
            out[out_idx] = block_sum / (N_avg * N_win)  # single division
            out_idx     += 1
            block_sum    = zero
            block_count  = 0
            if out_idx == n_out:
                break

    return out

def resolve_rinex_obs(output_file, f_sky, source, soft_bits_full, hard_bits_full, model_full, store_handle, antenna_handle, time_gps, time_shift, aided, short_circuit):
    """ Upright data bits, assimilate models, and produce RINEX observables """
    ranging_codes =  [key for key in model_full[0].keys()]
    nu_arr = []
    system_code = source[0]; prn=int(source[1:])
    for rc in ranging_codes:
        nav_uprighter = NavUprighter(system_code, rc, prn)
        nu_arr.append(nav_uprighter)

    shift_int = True
    # check to make sure each track is real, if not -- dont write to RINEX
    ranging_codes_use = []
    C_N0_RANGE = 10
    if antenna_handle.is_VLBI:
        C_N0_MIN = 52 # dB-Hz
    else:
        C_N0_MIN = 40
    use_idxs = np.zeros(len(model_full), dtype=bool)
    for idx, rc in enumerate(ranging_codes):
        C_N0_arr = []
        for jdx in range(len(model_full)):
            (t_0, t_dot, phi_0, f_D, f_D_dot, C_N0) = model_full[jdx][rc] 
            C_N0_arr.append(C_N0)
        C_N0_arr = np.array(C_N0_arr)
        if not np.all(C_N0_arr<C_N0_MIN):
            ranging_codes_use.append(rc)
            C_N0_max = np.max(C_N0_arr)
            use_idxs_rc = C_N0_arr > C_N0_max-C_N0_RANGE 
            use_idxs = np.bitwise_or(use_idxs, use_idxs_rc)

    num_epochs = int(np.sum(use_idxs))
    if time_shift > 0:
        if shift_int:
            # we shift up to next integer if there is a time shift
            time_start = time_gps + np.timedelta64(1, 's') + np.argwhere(use_idxs)[0][0]
        else:
            time_start = time_gps + np.timedelta64(int(time_shift*1e9), 'ns') + np.argwhere(use_idxs)[0][0]
    else:
        time_start = time_gps + np.argwhere(use_idxs)[0][0]

    # get observation times
    times_gps = []
    for jdx in range(len(model_full)):
        times_gps.append(time_gps + np.timedelta64(jdx, 's'))
    obs_times = np.array(times_gps)[use_idxs]

    rinex_file = RinexFile(antenna_handle.ref_pos, output_file, obs_times)
    rinex_file.gen_header(nav_uprighter.carrier_code, ranging_codes_use, system_code) # cc should be the same
    for idx, rc in enumerate(ranging_codes_use):
        soft_bits_rc = []
        hard_bits_rc = []
        phi_0_arr = []
        t_0_arr = []
        t_dot_arr = []
        f_D_arr = []
        f_D_dot_arr = []
        C_N0_arr = []
        models_rc = []
        times_gps = []
        times_rc_arr = []
        chip_rate = CHIP_RATES[rc]
        code_period = CODE_PERIODS[rc]
        for jdx in range(len(model_full)):
            models_rc.append(model_full[jdx][rc])
            #(t_0, t_dot, phi_0, f_D, f_D_dot) = model_full[jdx][rc] 
            (t_0, t_dot, phi_0, f_D, f_D_dot, C_N0) = model_full[jdx][rc] 
            t_0_arr.append(t_0)
            t_dot_arr.append(t_dot)
            phi_0_arr.append(phi_0)
            f_D_arr.append(f_D)
            f_D_dot_arr.append(f_D_dot)
            C_N0_arr.append(C_N0)
            hard_bits_rc.append(hard_bits_full[jdx][rc])
            soft_bits_rc.append(soft_bits_full[jdx][rc])
            times_gps.append(time_gps + np.timedelta64(jdx, 's'))
            times_rc = time_gps + np.timedelta64(int(code_period*1000), 'ms') * (np.arange(len(hard_bits_full[jdx][rc])-1)+1) \
                    + np.timedelta64(jdx, 's') + np.timedelta64(int(time_shift*1e9), 'ns')
            times_rc_arr.append(times_rc)

        t_0_arr = np.array(t_0_arr)
        t_dot_arr = np.array(t_dot_arr)
        phi_0_arr = np.array(phi_0_arr)
        f_D_arr = np.array(f_D_arr)
        f_D_dot_arr = np.array(f_D_dot_arr)
        C_N0_arr = np.array(C_N0_arr)
        hard_bits_arr = np.concatenate(hard_bits_rc)
        times_rc_arr = np.concatenate(times_rc_arr)
        #hard_bits_arr[hard_bits_arr==1] = 0
        #hard_bits_arr[hard_bits_arr==-1] = 1
        #soft_bits_arr = np.concatenate(soft_bits_rc)[1:] # drop first partial code
        soft_bits_arr = np.concatenate(soft_bits_rc) # drop first partial code
        times_gps = np.array(times_gps)

        if np.all(C_N0_arr < 40): 
            continue

        if time_shift > 0: 
            # shift to next integer second to account for frame start
            # (we are interpolating at ceil integer second and extrapolating at floored integer second)
            time_shift_start = time_shift*1e3 % int(code_period*1e3) # account for shifted start of file
            if shift_int is True:
                code_phase_m = (t_0_arr+t_dot_arr*(1-time_shift))/chip_rate * const.c
                ADR = -(phi_0_arr + 2*np.pi*f_D_arr*(1-time_shift) + np.pi*f_D_dot_arr*(1-time_shift)**2)/(2*np.pi)
                f_D_arr += f_D_dot_arr*(1-time_shift)
                times_gps += np.timedelta64(1, 's')
            else:
                code_phase_m = t_0_arr/chip_rate * const.c # ambiguous pseudorange
                ADR = -phi_0_arr/(2*np.pi)
                times_gps += np.timedelta64(int(time_shift*1e9), 'ns')
            code_phase_m -= time_shift_start/1e3 * const.c
        else:
            code_phase_m = t_0_arr/chip_rate * const.c # ambiguous pseudorange
            ADR = -phi_0_arr/(2*np.pi)

        antenna_handle.times_gps = times_gps
        source_array = np.repeat(source, len(times_gps))
        #rxpos_series, R_obj  = store_handle.compute_tides(antenna_handle.times_gps, antenna_handle.ref_pos, antenna_handle.antenna_name) 
        #antenna_handle.update_pos_series(rxpos_series, R_obj, antenna_handle.ref_pos)
        pr_model, _ = store_handle.sim_pr_simple(antenna_handle, times_gps, antenna_handle.bulk_clock*np.ones(len(times_gps)), f_sky, source_array)

        # now do uprighting of bits 
        nav_uprighter = nu_arr[idx]
        if rc != 'CL':
            if nav_uprighter.mode == 'sec_code':
                polarity, best_phi, best_score  = nav_uprighter.secondary_polarity_from_soft(soft_bits_arr)
            elif nav_uprighter.mode == 'preamble':
                if rc == 'CA':
                    polarity = nav_uprighter.preamble_polarity_ca(soft_bits_arr)
                else:
                    polarity = nav_uprighter.preamble_polarity_from_soft(soft_bits_arr)
                best_phi = None
            pr_amb = const.c*code_period # m 
        else: 
            polarity = np.sign(np.sum(soft_bits_arr))
            pr_amb = 0.5*const.c # ambiguity comes from misalignment of frame length (1 sec) with code period (1.5 sec)

        # now upright the ADR, resolve the pseudorange ambiguity and save off the obs
        wavelength = const.c/f_sky # m 

        pseudorange_resolved = -code_phase_m + pr_amb*np.rint((pr_model+code_phase_m)/pr_amb)
        pr_const = np.rint((pseudorange_resolved[0]/wavelength-ADR[0]))
        ADR += pr_const # adjust ADR to positive by shifting by rounded PR
        # temp 
        if rc == 'E1B':
            polarity = 1
            nav_arr, times = decode_e1b_with_time(hard_bits_arr, times_rc_arr)
            nav_results = decode_nav_list(nav_arr)
            for hnv_result in nav_results:
                if hnv_result.get('utc') is not None:
                    time_hnv_utc = hnv_result['utc']
                    print(time_hnv_utc)

        # resolve pseudorange ambiguity
        if polarity == -1:
            # resolve to full cycle ambiguity
            ADR += 0.5

        # apply appropriate aligning phase shift (RINEX table A23)
        phase_shift = PHASE_SHIFT_RC[rc]
        ADR += phase_shift

        # test misses in adjacent intervals
        if time_shift > 0: 
            # shift to next integer second to account for frame start
            # (we are interpolating at ceil integer second and extrapolating at floored integer second)
            code_miss_m = (t_0_arr+t_dot_arr*(2-time_shift))/chip_rate * const.c
            code_miss_m -= time_shift_start/1e3 * const.c
            ADR_miss = -(phi_0_arr + 2*np.pi*f_D_arr*(2-time_shift) + np.pi*f_D_dot_arr*(2-time_shift)**2)/(2*np.pi)
        else:
            code_miss_m = (t_0_arr+t_dot_arr)/chip_rate * const.c
            ADR_miss = -(phi_0_arr + 2*np.pi*f_D_arr + np.pi*f_D_dot_arr)/(2*np.pi)
        pr_miss = -code_miss_m[:-1] + pr_amb*np.rint((pr_model[1:]+code_miss_m[:-1])/pr_amb)
        ADR_miss += phase_shift
        ADR_diff = ADR[1:]-ADR_miss[:-1]
        pr_const_miss = 0.5*np.rint(2*np.median(ADR_diff))
        ADR_miss += pr_const_miss

        output_file_end = os.path.basename(output_file)
        fig_1, ax_1 = plt.subplots(dpi=300)
        ax_1.scatter(np.arange(len(pr_miss)), pseudorange_resolved[1:]-pr_miss,label='pseudorange', linestyle='-')
        ax_1.scatter(np.arange(len(pr_miss)), ADR[1:]-ADR_miss[:-1], label='ADR', linestyle='--')
        ax_1.set_xlabel('t (sec)')
        ax_1.set_ylabel('range miss m (PR), cycles (ADR)')
        ax_1.set_title(f'source {source}')
        ax_1.legend()
        if len(ranging_codes) > 1:
            fig_1.savefig('range_miss_' + antenna_handle.antenna_name + '_'+source+ '_' + str(f_sky) + output_file_end + rc +'.png')
        else:
            fig_1.savefig('range_miss_' + antenna_handle.antenna_name + '_'+source+ '_' + str(f_sky) + output_file_end+'.png')
        plt.close(fig_1)

        # cycle slip fixer (deal with earlier bug)
        ADR_diff = ADR[1:]-ADR_miss[:-1]
        iter_num = 0
        rc_multiple = len(soft_bits_full[0][rc])
        while np.any(np.abs(ADR_diff)>0.4):
            print(f'fixing cycle slip for {output_file_end} sat {source} ranging code {rc}')
            #if antenna_handle.antenna_name == 'dish-ftdavis': breakpoint()
            idx_slip = np.argwhere(np.abs(ADR_diff)>0.4)[0][0]
            sign_slip = np.sign(ADR_diff[idx_slip])
            ADR[idx_slip+1:] += -0.5*sign_slip
            ADR_miss[idx_slip+1:] += -0.5*sign_slip
            soft_bits_arr[(idx_slip+1)*rc_multiple:] = -soft_bits_arr[(idx_slip+1)*rc_multiple:]
            if nav_uprighter.mode == 'sec_code':
                polarity_check, best_phi, best_score  = nav_uprighter.secondary_polarity_from_soft(soft_bits_arr)
            elif nav_uprighter.mode == 'preamble':
                if rc == 'CA':
                    polarity_check = nav_uprighter.preamble_polarity_ca(soft_bits_arr)
                else:
                    polarity_check = nav_uprighter.preamble_polarity_from_soft(soft_bits_arr)
                best_phi = None
            if polarity != polarity_check:
                ADR += 0.5
                ADR_miss += 0.5
            fig_1, ax_1 = plt.subplots(dpi=300)
            ax_1.scatter(np.arange(len(pr_miss)), pseudorange_resolved[1:]-pr_miss,label='pseudorange', linestyle='-')
            ax_1.scatter(np.arange(len(pr_miss)), ADR[1:]-ADR_miss[:-1], label='ADR', linestyle='--')
            ax_1.set_xlabel('t (sec)')
            ax_1.set_ylabel('range miss m (PR), cycles (ADR)')
            ax_1.set_title(f'source {source}')
            ax_1.legend()
            if len(ranging_codes) > 1:
                fig_1.savefig('range_miss_' + antenna_handle.antenna_name + '_'+source+ '_' + str(f_sky) + output_file_end + rc +'.png')
            else:
                fig_1.savefig('range_miss_' + antenna_handle.antenna_name + '_'+source+ '_' + str(f_sky) + output_file_end+'.png')
            plt.close(fig_1)
            ADR_diff = ADR[1:]-ADR_miss[:-1]
            iter_num += 1 
            polarity = polarity_check
            if iter_num > 5:
                print(f'Unable to resolve cycle slips for {output_file_end} sat {source} ranging code {rc} ') 
                break
               
        # limit obs to within 10 dB-Hz of maximum, remove outliers
        C_N0_max = np.max(C_N0_arr)
        pr_sig, idxs_pr = find_sigmas(pseudorange_resolved[1:]-pr_miss, 3) # 3 sigma
        adr_sig, idxs_adr = find_sigmas(ADR[1:]-ADR_miss[:-1], 3)
        idxs_pr = np.concatenate(([idxs_pr[0]],idxs_pr))
        idxs_adr = np.concatenate(([idxs_adr[0]],idxs_adr))
        idxs_CN0 = C_N0_arr > C_N0_max-C_N0_RANGE
        use_idxs = idxs_pr & idxs_adr & idxs_CN0

        times_gps = times_gps[use_idxs]
        pseudorange_resolved = pseudorange_resolved[use_idxs]
        pr_model = np.array(pr_model)[use_idxs]
        ADR = ADR[use_idxs]
        f_D_arr = f_D_arr[use_idxs]
        C_N0_arr = C_N0_arr[use_idxs]
        soft_bits_arr = soft_bits_arr.reshape(-1, rc_multiple)[use_idxs].ravel()

        while np.any(np.diff(times_gps)/np.timedelta64(1,'s')>3):
            # large time discontinuity that likely destabilizes solution
            # take one side -- the larger side
            print('fixing time discontinuity')
            idx_diff = np.argwhere(np.diff(times_gps)/np.timedelta64(1,'s')>3)[0][0]
            if len(times_gps[idx_diff+1:]) > len(times_gps[:idx_diff+1]):
                times_gps = times_gps[idx_diff+1:]
                pseudorange_resolved = pseudorange_resolved[idx_diff+1:]
                pr_model = pr_model[idx_diff+1:]
                ADR = ADR[idx_diff+1:]
                f_D_arr = f_D_arr[idx_diff+1:]
                C_N0_arr = C_N0_arr[idx_diff+1:]
                soft_bits_arr = soft_bits_arr[(idx_diff+1)*rc_multiple:]
            else:
                times_gps = times_gps[:idx_diff+1]
                pseudorange_resolved = pseudorange_resolved[:idx_diff+1]
                pr_model = pr_model[:idx_diff+1]
                ADR = ADR[:idx_diff+1]
                f_D_arr = f_D_arr[:idx_diff+1]
                C_N0_arr = C_N0_arr[:idx_diff+1]
                soft_bits_arr = soft_bits_arr[:(idx_diff+1)*rc_multiple]

            if nav_uprighter.mode == 'sec_code':
                polarity_check, best_phi, best_score = nav_uprighter.secondary_polarity_from_soft(soft_bits_arr)
            elif nav_uprighter.mode == 'preamble':
                if rc == 'CA':
                    polarity_check = nav_uprighter.preamble_polarity_ca(soft_bits_arr)
                else:
                    polarity_check = nav_uprighter.preamble_polarity_from_soft(soft_bits_arr)

            if polarity != polarity_check:
                ADR += 0.5
            polarity = polarity_check

        
        fig_1, ax_1 = plt.subplots(dpi=300)
        ax_1.scatter(np.arange(len(ADR)),pseudorange_resolved-pr_model,label='pseudorange', linestyle='-')
        ax_1.scatter(np.arange(len(ADR)),ADR*wavelength-pr_model,label='carrier phase', linestyle='-')
        ax_1.set_xlabel('t (sec)')
        ax_1.set_ylabel('range (m)')
        ax_1.set_title(f'source {source}')
        ax_1.legend()
        if len(ranging_codes) > 1:
            fig_1.savefig('ORD_' + antenna_handle.antenna_name + '_'+source+ '_' + str(f_sky) + output_file_end + rc +'.png')
        else:
            fig_1.savefig('ORD_' + antenna_handle.antenna_name + '_'+source+ '_' + str(f_sky) + output_file_end+'.png')
        plt.close(fig_1)

        #fig_1, ax_1 = plt.subplots(dpi=300)
        #ax_1.scatter(np.arange(30),pseudorange_resolved,label='pseudorange', linestyle='-')
        #ax_1.scatter(np.arange(30), ADR, label='ADR', linestyle='--')
        #ax_1.set_xlabel('t (sec)')
        #ax_1.set_ylabel('range (m)')
        #ax_1.set_title(f'source {source}')
        #ax_1.legend()
        #fig_1.savefig('meas_'+source+'.png')
        #plt.close(fig_1)

        #fig_2, ax_2 = plt.subplots(dpi=300)
        #ax_2.scatter(np.arange(30),ADR-pseudorange_resolved, linestyle='-')
        #ax_2.set_xlabel('t (sec)')
        #ax_2.set_ylabel('ADR-pseudorange (m)')
        #fig_2.savefig('diff_'+source+'.png')
        #plt.close(fig_2)
        rinex_file.gen_dataset(rc, times_gps, pseudorange_resolved, ADR, f_D_arr, C_N0_arr, prn)

    rinex_file.write()


def get_replica(rc_dir, ranging_code, prn):
    """ Return ranging code for observed satellite """
    rc_file = rc_dir + '/' + ranging_code + '_' + prn + '.pkl'
    with open(rc_file, 'rb') as f:
        prn_code = pickle.load(f)
    return prn_code

def main():
    """
    """
    parser = argparse.ArgumentParser()
    add_args_to_parser(parser)
    args = parser.parse_args()

    for rxpos_arg in args.rxpos:
        rxpos = [float(pos_comp) for pos_comp in rxpos_arg.split()]

    if args.key_file is not None:
        datetime_array, source_array, point_ra_dec_array, dt_key, duration_key, source_key, point_key  \
                = import_key_gnss([], [], args.key_file, 0)
        _, _, _, source_array_key, _, _, datetime_array_key, scan_nums_key = read_key(args.key_file)
        times_sec = (datetime_array-datetime_array[0])/np.timedelta64(1,'s')
        avg_diff = mode(np.diff(times_sec), keepdims=True)[0][0]

        # cover the rest of the 1-second intervals up to the next observation
        # (sometimes the VDIF data start late)
        source_array_full = []
        datetime_array_full = []
        for idx, duration in enumerate(duration_key):
            for tim in range(int(duration)):
                datetime_array_full.append(datetime_array[idx]+np.timedelta64(tim,'s'))
                source_array_full.append(source_array[idx])
        datetime_array = np.array(datetime_array_full)
        source_array = np.array(source_array_full)
        duration_dict = {}
        for time in datetime_array:
            duration_dict[time] = avg_diff

    # initialize the antenna handle
    antenna_position = rxpos
    if args.trop_global is True:
        tk_pos = gnsstk.Position(antenna_position[0], antenna_position[1], antenna_position[2])
        if args.trop_global is True:
            tropModel = gnsstk.GlobalTropModel(tk_pos, time_beg)
            if args.trop_H is not None:
                tropModel.setHumidity(args.trop_H)
            else:
                tropModel.setHumidity(50) # 50 percent is standard assumed humidity
        antenna_handle = AntennaInfo(args.antenna_name, antenna_position, args.antenna_type, args.clock_offset, 0, False, tropModel)
    else:
        antenna_handle = AntennaInfo(args.antenna_name, antenna_position, args.antenna_type, args.clock_offset, 0, False)

    if args.vlbi_antenna:
        antenna_handle.set_VLBI(args.axis_offset, None, None, False, False)

    if args.eph_files is not None:
        # use aiding from precise ephemerides
        eph_files = [eph_file for eph_file in args.eph_files]
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

        if args.weather_cal_file is not None:
            antenna_handle.load_weather_file(args.weather_cal_file)
    else:
        nav_store = None

    aided = not args.blind_search

    store_handle = GNSSTKStores('GNSS', 'GNSS', None, None, None, None, nav_store, False, '', True)
    if args.key_file is not None:
        store_handle.hold_source_array(source_array, datetime_array, duration_dict)

    if args.satellite is not None:
        satellite = args.satellite[0]
    else:
        satellite = None

    if args.ca_only:
        RANGING_CODES['G'] = {1575.42e6: ['CA'], 1227.6e6: ['CM'], 1176.45e6: ['Q5'],}

    if args.input_files is not None:
        process_vdif(args.input_files[0], None, args.output_files[0], satellite, args.rc_directory, args.thread, \
                args.num_channels, args.channel, args.acq_cadence, args.center_freq, args.sky_freq, store_handle, antenna_handle, aided, args.short_circuit, args.max_time)
    elif args.input_files_x is not None and args.input_files_y is not None:
        process_vdif(args.input_files_x[0], args.input_files_y[0], args.output_files[0], satellite, args.rc_directory, args.thread, \
                args.num_channels, args.channel, args.acq_cadence, args.center_freq, args.sky_freq, store_handle, antenna_handle, aided, args.short_circuit, args.max_time)
    else:
        raise ValueError('Need to supply either R polarization or X + Y polarization input files')


if __name__ == "__main__":
    main()

