#!/usr/bin/env python3
"""
dd_to_sd.py -- undifference Bernese L4 double-difference residuals into
per-satellite single-difference ionosphere estimates (the L4R level-1 step,
Maennel & Rothacher 2016, Eq. 4).

Per baseline, per epoch:
  * Each .RES record is one DD: rows of the operator D are (+1 at sat1, -1 at sat2)
    over the unknown per-satellite single differences I_hat[sat].
  * Append the zero-mean constraint row (all ones, RHS 0): Sum_sat I_hat = 0.
  * Solve the (ndd+1 x nsat) system for I_hat (least squares / min-norm).
  * Convert L4 metres -> slant TEC with PER-SYSTEM frequencies (GLONASS is FDMA,
    per-satellite). Absolute level (GIM) is NOT added here -- that's the next step.

Output per (baseline, epoch): {sat: (sd_L4_m, sd_stec_TECU)} plus arrays for
downstream use.

NOTE: on the ftdavis6 SHORT baselines this runs but is physically ~meaningless
(common-mode ionosphere); it validates the math. Real product is on long baselines.
"""
import numpy as np
import re
from collections import namedtuple, defaultdict
import struct

import os
use_custom_version = os.getenv('USE_CUSTOM_GEORINEX', 'false').lower() == 'true'
if use_custom_version:
    import sys
    sys.path.insert(0, '/sgl/ceph/work/jskeens')
    try: 
        from georinex_custom import load
    except: 
        sys.path.insert(0, '/home/jskeens/oscar_dir/scratch/jskeens')
        from georinex_custom import load
else:
    from georinex import load

# NOTE: NEED TO REMOVE MANY OF THESE
import xarray as xr
import datetime
import itertools
import random
from bisect import bisect_right
import re
from pandas import Timestamp, Index
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from matplotlib import colors
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import TwoSlopeNorm
from scipy.optimize import least_squares
from scipy.interpolate import make_interp_spline
from scipy.stats import linregress, mode
from scipy.sparse import coo_matrix, vstack, csr_matrix, csc_matrix
from scipy.sparse.linalg import lsqr, spsolve,splu
from scipy.optimize import lsq_linear
import scipy.constants as const
from copy import deepcopy
import argparse
import matplotlib.dates as mdates
from typing import List, Dict, Tuple, Optional

from gnsstk import std_vector_string, Position, AntennaStore, AntexData, OceanLoadTides, PoleTides, \
                  AtmLoadTides, SolarSystem, GlobalTropModel, SaasTropModel, NeillTropModel, RinexSatID

from single_diff_tools import import_key_gnss, import_data_vlbi, import_data_vlbi_farfield, import_data_vlbi_ngs, import_data_vlbi_vgosdb, \
                  import_data_vlbi_vda, write_SINEX, datetime64_to_mjd, map_datasets, import_data_nc_sim,\
                  find_common_epochs, BaselineInfo, AntennaInfo, GNSSTKStores, ECEF2ECI, slip_detect_MW, slip_detect_single_freq,\
                  slip_detect_phase_delay, sample_poly_at_interval, trim_amb_Zdom, trim_amb_state, gen_phase_clock_state, adjust_stoch_params, thin_data,\
                  analyze_ls_solution, resolve_float_amb, construct_float_amb, remove_outliers, iterative_remove_outliers, calc_residuals, \
                  calc_jac, plot_time_units, read_src, date_to_common, date_to_mjd, detect_unresolved_amb_vlbi, \
                  detect_unresolved_amb_gnss, set_bounds_phase_clock, union_of_slices, get_residuals, iterative_weight_adjust, iterative_weight_adjust_ls_vce, \
                  iterative_weight_adjust_LS_VCE_full, sample_global_poly_at_interval, gen_key, VMF3Model, \
                  get_obs_weights, vlbi_transform_data, form_double_differences, NavStore, IonexFile

from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
# optional for GIF
try:
    import imageio.v2 as imageio
    HAVE_IMAGEIO = True
except Exception:
    HAVE_IMAGEIO = False

DATA_LEN = 29
_REC = struct.Struct("<5i d")     # 28 numeric bytes; flag is byte 29

Record = namedtuple("Record", "baseline epoch meatyp sat1 sat2 value flag")

# ---- constants -------------------------------------------------------------
A_TEC = 40.3e16            # m^3 s^-2 TECU^-1   (Mannel Eq. 1)
C = 299792458.0

# --- layer 1: PRN range -> system letter (stable; Bernese's own encoding) ---
def system_of(prn):
    if   prn < 100: return 'G'   # GPS
    elif prn < 200: return 'R'   # GLONASS
    elif prn < 300: return 'E'   # Galileo
    elif prn < 400: return 'C'   # BeiDou
    elif prn < 500: return 'J'   # QZSS
    else:           return 'S'   # SBAS

# --- nominal carrier frequencies [Hz] by RINEX band, per system -------------
# value = base frequency; GLONASS L1/L2 are FDMA (handled separately).
FREQ = {
    'G': {1: 1575.42e6, 2: 1227.60e6, 5: 1176.45e6},
    'E': {1: 1575.42e6, 5: 1176.45e6, 7: 1207.140e6, 8: 1191.795e6, 6: 1278.75e6},
    'C': {2: 1561.098e6, 7: 1207.140e6, 6: 1268.520e6,
          1: 1575.42e6, 5: 1176.45e6, 8: 1191.795e6},   # B1I/B2I + B1C/B2a/B2
    'J': {1: 1575.42e6, 2: 1227.60e6, 5: 1176.45e6, 6: 1278.75e6},
    'S': {1: 1575.42e6, 5: 1176.45e6},
}

# --- layer 2: which two bands were differenced, PER SYSTEM (read from the run) ---
# This is the L4 signal pair Bernese used. Set from OBSERV.SEL / the OUT FREQ table.
BAND_PAIR = {'G': (1, 2), 'E': (1, 5), 'C': (2, 7), 'J': (1, 2)}  

def freqs_for_prn(prn, glo_k=None):
    sys = system_of(prn)
    if sys == 'R':                                  # GLONASS FDMA
        if glo_k is None: return None
        return (1602.0e6 + glo_k*0.5625e6, 1246.0e6 + glo_k*0.4375e6)
    b1, b2 = BAND_PAIR[sys]
    f1, f2 = FREQ[sys].get(b1), FREQ[sys].get(b2)
    return None if (f1 is None or f2 is None) else (f1, f2)

def read_res(path, marker=4, endian="<"):
    """Return a flat list of Record for every 29-byte data record, in file order."""
    mfmt = {(4, "<"): "<i", (4, ">"): ">i",
            (8, "<"): "<q", (8, ">"): ">q"}[(marker, endian)]
    msz = marker
    recnum = struct.Struct(mfmt)

    with open(path, "rb") as f:
        raw = f.read()
    out, n, pos = [], len(raw), 0
    while pos + msz <= n:
        lead = recnum.unpack(raw[pos:pos+msz])[0]
        if lead < 0 or pos + msz + lead + msz > n:
            break
        body = raw[pos+msz: pos+msz+lead]
        tail = recnum.unpack(raw[pos+msz+lead: pos+msz+lead+msz])[0]
        if tail != lead:
            break
        if lead == DATA_LEN:
            bl, iepo, mty, s1, s2, val = _REC.unpack(body[:28])
            flag = body[28:29].decode("latin1")
            out.append(Record(bl, iepo, mty, s1, s2, val, flag))
        pos += msz + lead + msz
    return out


def split_baselines(recs):
    """Split by the baseline index carried in each record (field 0). Returns
    {baseline_index: [Record, ...]} in ascending index order."""
    out = defaultdict(list)
    for r in recs:
        out[r.baseline].append(r)
    return dict(sorted(out.items()))


def l4_to_stec(sd_l4_m, prn):
    """Convert an L4 single-difference residual [m] to slant TEC [TECU] for the
    given satellite's frequencies.  L4 = I1 - I2 = I1*(1 - f1^2/f2^2);
    STEC = I1 * f1^2 / A_TEC.  Returns None if frequencies unknown."""
    fr = freqs_for_prn(prn)
    if fr is None:
        return None
    f1, f2 = fr
    ion_fac = 1.0 - (f1*f1)/(f2*f2)      # negative; carries the sign
    i1_m = sd_l4_m / ion_fac             # L1-equivalent ionospheric delay [m]
    return i1_m * (f1*f1) / A_TEC        # slant TEC [TECU]


def solve_epoch(epoch_recs):
    """Undifference one baseline-epoch. epoch_recs: list of Record (same baseline,
    same epoch). Returns dict {sat: sd_l4_m} (zero-mean across sats), or {} if <2
    satellites / degenerate."""
    # collect the ordered satellite set and the DD rows
    sats = []
    seen = {}
    for r in epoch_recs:
        for s in (r.sat1, r.sat2):
            if s not in seen:
                seen[s] = len(sats); sats.append(s)
    nsat = len(sats)
    if nsat < 2 or len(epoch_recs) < 1:
        return {}
    ndd = len(epoch_recs)
    D = np.zeros((ndd + 1, nsat))
    rhs = np.zeros(ndd + 1)
    for i, r in enumerate(epoch_recs):
        D[i, seen[r.sat1]] = 1.0
        D[i, seen[r.sat2]] = -1.0
        rhs[i] = r.value
    D[-1, :] = 1.0                        # zero-mean constraint, RHS already 0
    sd, *_ = np.linalg.lstsq(D, rhs, rcond=None)
    return {sats[j]: sd[j] for j in range(nsat)}

def baseline_stations(out_path):
    """Map baseline index -> (station1, station2) from the GPSEST OUT
    observation-file table."""
    out = {}
    with open(out_path) as f:
        lines = f.readlines()
    # find the table header, then read rows until a blank/non-data line
    start = None
    for i, ln in enumerate(lines):
        if re.search(r'\bFILE\b.*\bSTATION 1\b.*\bSTATION 2\b', ln):
            start = i + 1
            break
    if start is None:
        raise RuntimeError("observation-file table not found in " + out_path)
    for ln in lines[start:]:
        if not ln.strip():
            break
        # columns are fixed-width; station fields are 16 chars wide and may
        # contain a name plus a DOMES number -> take the first token of each.
        m = re.match(r'\s*(\d+)\s+\S+\s+\S+\s+(.{16})(.{16})', ln)
        if not m:
            # table ended (hit a non-row line)
            break
        idx = int(m.group(1))
        sta1 = m.group(2).split()[0]      # name only, drop DOMES
        sta2 = m.group(3).split()[0]
        out[idx] = (sta1, sta2)
    return out

def process(path):
    recs = read_res(path)
    bsl = split_baselines(recs)
    # results[baseline][epoch][sat] = (sd_l4_m, sd_stec_TECU_or_None)
    results = {}
    for bid, rs in bsl.items():
        by_epoch = defaultdict(list)
        for r in rs:
            by_epoch[r.epoch].append(r)
        ep_out = {}
        for ep, er in by_epoch.items():
            sd = solve_epoch(er)
            ep_out[ep] = {s: (v, l4_to_stec(v, s)) for s, v in sd.items()}
        results[bid] = ep_out
    return results


_DOMES = re.compile(r"^\d{5}[A-Z]\d{3}$")     # e.g. 40442M017
 
def baseline_stations(out_path):
    """Parse the GPSEST OUT observation-file table -> {baseline_index:(sta1,sta2)}.
    The FILE number is the .RES baseline index; STATION 1/2 are the endpoints.
    Token-based (not fixed-width): each row is
        idx 'P' 'L4' STA1 [DOMES1] STA2 [DOMES2] SESS date ...
    so we read STA1, skip an optional DOMES, read STA2, skip an optional DOMES.
    VERIFY the printed mapping against your network (the GPS-only baselines from
    read_res should match the short legs here)."""
    with open(out_path) as f:
        lines = f.read().splitlines()
    start = None
    for i, ln in enumerate(lines):
        if re.search(r"\bFILE\b", ln) and "STATION 1" in ln and "STATION 2" in ln:
            start = i + 1
            break
    if start is None:
        raise RuntimeError("observation-file table not found in " + out_path)
    out = {}
    for ln in lines[start:]:
        if not ln.strip():
            if out:
                break          # blank line after data rows -> table ended
            continue           # blank/separator line between header and rows
        if set(ln.strip()) <= {"-"}:
            continue           # dashes rule line
        toks = ln.split()
        if not toks[0].isdigit():
            break
        idx = int(toks[0])
        rest = toks[3:]                        # drop idx, 'P', freq code
        sta1 = rest[0]; p = 1
        if p < len(rest) and _DOMES.match(rest[p]):
            p += 1
        sta2 = rest[p]; p += 1
        out[idx] = (sta1, sta2)
    return out

def _components(stations, edges):
    """Union-find on the SD graph. Returns list of (component_stations_set,
    component_edges_list). A satellite seen at only some stations yields a
    forest, so components matter: ZD values are determined only WITHIN a
    component (a between-station SD across components is not constrained by the
    data and must not be formed)."""
    parent = {s: s for s in stations}
 
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
 
    for a, b, _ in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
 
    comp_st = defaultdict(set)
    comp_ed = defaultdict(list)
    for s in stations:
        comp_st[find(s)].add(s)
    for a, b, v in edges:
        comp_ed[find(a)].append((a, b, v))
    return [(comp_st[r], comp_ed.get(r, [])) for r in comp_st]
 
def sd_to_zd(results, bsl_stations):
    """results: {baseline:{epoch:{sat:(sd_l4_m,stec)}}}  from process()
       bsl_stations: {baseline:(sta1,sta2)}              from baseline_stations()
       SIGN CONVENTION: the per-baseline SD value is taken as ZD(sta1)-ZD(sta2),
       i.e. STATION 1 minus STATION 2 in OUT order. Verify with check_consistency().
 
       Returns zd[epoch][sat] = {station:(zd_l4_m, comp_id)}, zero-mean per
       connected component. Use single_difference() to form SD for any pair."""
    # invert to epoch -> sat -> [(sta1, sta2, sd_l4_m), ...]
    by_es = defaultdict(lambda: defaultdict(list))
    missing = set()
    for bid, eps in results.items():
        if bid not in bsl_stations:
            missing.add(bid); continue
        s1, s2 = bsl_stations[bid]
        for ep, sats in eps.items():
            for sat, (sd_l4, _stec) in sats.items():
                by_es[ep][sat].append((s1, s2, sd_l4))
    if missing:
        print(f"sd_to_zd: WARNING baselines with no station mapping (skipped): {sorted(missing)}")
 
    zd = {}
    for ep, satmap in by_es.items():
        zd[ep] = {}
        for sat, edges in satmap.items():
            stations = {x for a, b, _ in edges for x in (a, b)}
            sat_out = {}
            comp_id = 0
            for comp_stations, comp_edges in _components(stations, edges):
                cs = sorted(comp_stations)
                idx = {s: i for i, s in enumerate(cs)}
                n = len(cs)
                A = np.zeros((len(comp_edges) + 1, n))
                rhs = np.zeros(len(comp_edges) + 1)
                for i, (a, b, v) in enumerate(comp_edges):
                    A[i, idx[a]] = 1.0
                    A[i, idx[b]] = -1.0
                    rhs[i] = v
                A[-1, :] = 1.0                 # zero-mean over component, RHS 0
                sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
                for s in cs:
                    sat_out[s] = (sol[idx[s]], comp_id)
                comp_id += 1
            zd[ep][sat] = sat_out
    return zd
 
def single_difference(zd, staA, staB, to_tec=False):
    """SD(A-B,s) = ZD(A,s) - ZD(B,s), only where both stations are present AND in
    the same connected component (else the difference isn't data-determined).
    Returns ({epoch:{sat: value}}, n_skipped). value is L4 metres unless
    to_tec=True, in which case it is slant-TEC TECU (per-satellite frequencies;
    GLONASS needs GLONASS_K populated in dd_to_sd, else those sats give None)."""
    out = defaultdict(dict)
    skipped = 0
    for ep, satmap in zd.items():
        for sat, stamap in satmap.items():
            if staA in stamap and staB in stamap:
                (va, ca), (vb, cb) = stamap[staA], stamap[staB]
                if ca != cb:
                    skipped += 1
                    continue
                sd = va - vb
                out[ep][sat] = l4_to_stec(sd, sat) if to_tec else sd
    return dict(out), skipped
 
def check_consistency(results, zd, bsl_stations, tol=1e-6):
    """For each baseline that is a direct edge, compare its original SD (from
    process) against ZD(sta1)-ZD(sta2). They must agree (same-component sats) if
    the zero-mean solves are consistent and the SD sign convention is right.
    Prints max |difference| and the sign relationship per baseline. A consistent
    NEGATIVE relationship means the sta1/sta2 sign convention is flipped."""
    for bid, (s1, s2) in sorted(bsl_stations.items()):
        if bid not in results:
            continue
        diffs, signs = [], []
        for ep, sats in results[bid].items():
            zep = zd.get(ep, {})
            for sat, (sd_l4, _) in sats.items():
                stamap = zep.get(sat, {})
                if s1 in stamap and s2 in stamap and stamap[s1][1] == stamap[s2][1]:
                    zd_sd = stamap[s1][0] - stamap[s2][0]
                    diffs.append(abs(zd_sd - sd_l4))
                    if abs(sd_l4) > 1e-4:
                        signs.append(np.sign(zd_sd / sd_l4))
        if diffs:
            md = max(diffs)
            sgn = np.mean(signs) if signs else float("nan")
            verdict = "OK" if md < tol else ("SIGN FLIP" if sgn < 0 else "MISMATCH")
            print(f"  baseline {bid} ({s1}-{s2}): max|Δ|={md:.2e}  mean sign={sgn:+.1f}  [{verdict}]")

def session_epoch0_dt(out_path):
    """Return (epoch0 datetime64[ns], dt seconds) from the OUT obs-file table."""
    with open(out_path) as f:
        lines = f.read().splitlines()
    for ln in lines:
        m = re.search(r"(\d{2})-(\d{2})-(\d{2})\s+(\d+):(\d+):(\d+)\s+\d+\s+(\d+)", ln)
        if m:
            yy, mo, dd, hh, mi, ss, dt = map(int, m.groups())
            t0 = np.datetime64(f"20{yy:02d}-{mo:02d}-{dd:02d}T{hh:02d}:{mi:02d}:{ss:02d}", "ns")
            return t0, dt
    raise RuntimeError("could not parse session start/DT from " + out_path)

def epoch_to_dt64(epoch_int, t0, dt):
    """Bernese epoch counter (1-based, on the dt grid) -> datetime64[ns]."""
    return t0 + np.timedelta64(int(round((epoch_int - 1) * dt)), "s").astype("timedelta64[ns]")

class BaselineObj:
    __slots__ = ("baseline_name", "position1", "position2", "sd_arr")
    def __init__(self, name, p1, p2, sd_arr):
        self.baseline_name, self.position1, self.position2, self.sd_arr = name, p1, p2, sd_arr

def baseline_obj_direct(results, bsl_stations, A, B, rxpos_by_name):
    """Build a BaselineObj straight from the directly-estimated baseline A-B in
    the .RES, bypassing the ZD field entirely. Returns None if A-B is not a
    direct baseline in this run."""
    # find the field0 index whose station pair is {A,B}
    target = None
    sign = 1.0
    for bid, (s1, s2) in bsl_stations.items():
        if {s1, s2} == {A, B}:
            target = bid
            sign = 1.0 if (s1, s2) == (A, B) else -1.0   # orient to A - B
            break
    if target is None:
        breakpoint()
        return None          # not a direct baseline; would need reconstruction
    if target not in results:
        print(f"[baseline_obj_direct] {A}-{B}: target bid {target} in bsl_stations "
              f"but NOT in results. results keys={sorted(results)}, "
              f"bsl_stations keys={sorted(bsl_stations)}")
        return None 
    ep_map = results[target]            # {epoch:{sat:(sd_l4_m, sd_stec)}}
    epochs = sorted(ep_map)
    sats   = sorted({s for ep in ep_map.values() for s in ep})
    sidx   = {s: j for j, s in enumerate(sats)}
    M = np.full((len(epochs), len(sats)), np.nan)
    for i, e in enumerate(epochs):
        for s, (_l4, stec) in ep_map[e].items():
            if stec is not None:
                M[i, sidx[s]] = sign * stec      # TEC; sign orients to A - B
    times = np.array(epochs, dtype="datetime64[ns]")
    da = xr.DataArray(M, coords={"time": times, "sv": sats}, dims=["time", "sv"])
    return BaselineObj(f"{A}-{B}", rxpos_by_name[A], rxpos_by_name[B], da)

def _sd_to_xarray(sd_dict):
    """sd_dict: {datetime64: {sat: value}} (from single_difference on the merged,
    time-keyed ZD field) -> (time, sv) DataArray of SD-STEC."""
    tkeys = sorted(sd_dict)
    sats  = sorted({s for ep in sd_dict.values() for s in ep})
    times = np.array(tkeys, dtype='datetime64[ns]')
    M = np.full((len(tkeys), len(sats)), np.nan)
    sidx = {s: j for j, s in enumerate(sats)}
    for i, e in enumerate(tkeys):
        for s, v in sd_dict[e].items():
            if v is not None:                      # GLONASS w/o channel -> None
                M[i, sidx[s]] = v
    return xr.DataArray(M, coords={'time': times, 'sv': sats}, dims=['time', 'sv'])

def bernese_prn_to_satid(sv):
    """Bernese PRN integer -> (gnsstk system char, satellite number)."""
    sv = int(sv)
    if   sv < 100:  return 'G', sv          # GPS
    elif sv < 200:  return 'R', sv - 100    # GLONASS slot
    elif sv < 300:  return 'E', sv - 200    # Galileo
    elif sv < 400:  return 'J', sv - 300    # QZSS
    elif sv < 500:  return 'C', sv - 400    # BeiDou
    else:           return 'S', sv - 500    # SBAS

def _scalar(coord):
    # numpy scalar, preserves dtype: datetime64 stays datetime64 (not int)
    return np.asarray(coord.values).ravel()[0]

def stack_xr(obs_list, var_name, stack_dim='sv', align_dim='time'):
    if stack_dim == align_dim:
        raise ValueError(f"stack_dim must differ from align_dim (both '{stack_dim}')")

    # Union index for the alignment dimension
    align_union = (
        xr.concat([da[align_dim] for da in obs_list], dim=align_dim)
        .to_index().unique().sort_values()
    )

    buckets = {}  # label -> accumulated Dataset

    for i, da in enumerate(obs_list):
        # --- derive the stack label without changing it ---
        if stack_dim in da.coords and da.coords[stack_dim].size == 1:
            label = _scalar(da.coords[stack_dim])
            #label = da.coords[stack_dim].item()
            # squeeze if it's a size-1 dim as well
            if stack_dim in da.dims and da.sizes[stack_dim] == 1:
                da = da.squeeze(stack_dim)
        elif stack_dim in da.coords and align_dim in da.coords[stack_dim].dims:
            # assume const over align_dim; take the first
            label = _scalar(da.coords[stack_dim].isel({align_dim: 0}))
            #label = da.coords[stack_dim].isel({align_dim: 0}).item()
        else:
            label = f"{stack_dim}_{i}"

        # sanity: alignment axis should be unique per slice
        idx = da[align_dim].to_index()
        if not idx.is_unique:
            raise ValueError(
                f"Element {i}: duplicate '{align_dim}' labels not allowed for alignment."
            )

        # drop stack_dim coord (we'll set it via concat labels)
        if stack_dim in da.coords:
            da = da.reset_coords(stack_dim, drop=True)

        ds = da.to_dataset(name=var_name).reindex({align_dim: align_union})

        # --- accumulate by label: merge into a single row for each label ---
        if label in buckets:
            buckets[label] = buckets[label].combine_first(ds)
        else:
            buckets[label] = ds

    # Preserve insertion order of labels
    labels = list(buckets.keys())
    dsets = [buckets[l] for l in labels]

    # Concatenate along the stacking dim with your labels exactly as given
    return xr.concat(dsets, dim=xr.IndexVariable(stack_dim, labels))

def _split_bl(name):
    a, b = str(name).split('-')
    return a, b

def _get_site_objects(baseline_name):
    # pull the baseline object to grab both site positions
    bo = next(bo for bo in baseline_objs if str(bo.baseline_name) == str(baseline_name))
    r1 = np.array(bo.position1, float); rx1 = Position(bo.position1)
    r2 = np.array(bo.position2, float); rx2 = Position(bo.position2)
    lat1, lon1 = rec_latlon(r1)
    lat2, lon2 = rec_latlon(r2)
    return (r1, rx1, lat1, lon1), (r2, rx2, lat2, lon2)

def rec_latlon(rx_ecef):
    lon = np.arctan2(rx_ecef[1], rx_ecef[0])
    lat = np.arctan2(rx_ecef[2], np.hypot(rx_ecef[0], rx_ecef[1]))
    return lat, lon

def load_day_results(res_path, out_path):
    """Per-day pieces WITHOUT undifferencing: raw process() results, the
    baseline->station map, and the day's epoch0/dt. No sd_to_zd call."""
    results = process(res_path)
    bsl     = baseline_stations(out_path)
    t0, dt  = session_epoch0_dt(out_path)
    return results, bsl, (t0, dt)

def combine_days_results(day_specs):
    """Merge per-day results to absolute time, WITHOUT building the ZD field.
    Returns merged results keyed by datetime64, and the union baseline map.
    results_abs[bid][datetime64][sat] = (sd_l4_m, sd_stec)"""
    from collections import defaultdict
    merged = defaultdict(dict)
    ref_bsl = None
    for res_path, out_path in day_specs:
        results, bsl, (t0, dt) = load_day_results(res_path, out_path)
        if ref_bsl is None:
            ref_bsl = dict(bsl)
        elif bsl != ref_bsl:
            print(f"WARNING: baseline map differs in {out_path}; keying on names.")
            ref_bsl.update(bsl)            # union, so all baselines are known
        for bid, eps in results.items():
            for e, satmap in eps.items():
                merged[bid][epoch_to_dt64(e, t0, dt)] = satmap
    return dict(merged), ref_bsl

def build_zd(results_abs, bsl):
    """Lazily build the ZD field ONLY when reconstruction is actually needed.
    Re-keys results_abs (already absolute-time) for sd_to_zd, which expects
    {bid:{epoch_key:{sat:...}}} -- the datetime keys work fine as epoch keys."""
    return sd_to_zd(results_abs, bsl)

_SION_FILE_COL = 0
_SION_SAT_COL  = 1
_SION_ZEN_COL  = 2
_SION_STEC_COL = 5
_SION_SIG_COL  = 6
_SION_MJD_COL  = 11
 
_MJD0 = np.datetime64("1858-11-17T00:00:00", "ns")
 
def parse_file_baseline_map(path):
    """Read the 'FILE TYP FREQ. STATION 1 STATION 2 ...' table -> {file_idx:(A,B)}.
    Station tokens may include a dome number (e.g. 'MGO3 40442M017'); we take the
    4-char station code only."""
    fmap = {}
    in_tbl = False
    hdr = re.compile(r'^\s*FILE\s+TYP\s+FREQ\.\s+STATION\s+1\s+STATION\s+2')
    # FILE  TYP  FREQ  STATION1[ dome]  STATION2[ dome]  SESS(4-digit) ...
    # station code = 4 alphanumerics starting with a letter; dome (optional)
    # is digits+letters which we skip. Anchor on the 4-digit SESS at the end.
    row = re.compile(
        r'^\s*(\d+)\s+\w+\s+\S+\s+'
        r'([A-Za-z]\w{3})(?:\s+\w+)?\s+'
        r'([A-Za-z]\w{3})(?:\s+\w+)?\s+\d{4}\b')
    with open(path) as f:
        for ln in f:
            if hdr.search(ln):
                in_tbl = True
                continue
            if in_tbl:
                m = row.match(ln)
                if m:
                    fmap[int(m.group(1))] = (m.group(2).upper(), m.group(3).upper())
                elif ln.strip() and not ln.startswith('---') and fmap:
                    # table ended (a non-row, non-divider line after we've seen rows)
                    break
    return fmap
 
 
def _read_diff_records(path):
    """Yield (file_idx, sat, mjd, stec, sigma) for every #SION record."""
    with open(path) as f:
        for ln in f:
            if "#SION" not in ln:
                continue
            tok = ln.split()
            if tok[-1] != "#SION":
                continue
            try:
                fil = int(tok[_SION_FILE_COL])
                sat = int(tok[_SION_SAT_COL])
                stec = float(tok[_SION_STEC_COL])
                sig = float(tok[_SION_SIG_COL])
                mjd = float(tok[_SION_MJD_COL])
            except (ValueError, IndexError):
                continue
            yield fil, sat, mjd, stec, sig
 
 
def _clean_baseline(recs, arc_gap_s, nsig, win_s, min_pts, sigma_ceiling,
                    scale_floor):
    """recs: iterable of (sat, mjd, stec, sigma) for ONE baseline.
    Returns {sat: {dt64: stec}} after per-arc iterative outlier removal +
    sigma ceiling (same cleaning as the per-station path)."""
    bysat = {}
    for sat, mjd, stec, sig in recs:
        bysat.setdefault(sat, []).append((mjd, stec, sig))
    out = {}
    n_in = n_out = 0
    for sat, rs in bysat.items():
        rs.sort()
        mjd = np.array([r[0] for r in rs])
        stec = np.array([r[1] for r in rs])
        sig = np.array([r[2] for r in rs])
        n_in += len(rs)
        if sigma_ceiling is not None:
            ok = sig <= sigma_ceiling
            mjd, stec = mjd[ok], stec[ok]
        if len(mjd) == 0:
            continue
        t_s = mjd * 86400.0
        gaps = np.where(np.diff(t_s) > arc_gap_s)[0]
        bounds = np.concatenate(([0], gaps + 1, [len(t_s)]))
        kept = {}
        for a in range(len(bounds) - 1):
            sl = slice(bounds[a], bounds[a + 1])
            ta, ya = t_s[sl], -stec[sl]
            if len(ta) == 0:
                continue
            m = _iter_outlier_mask(ta, ya, nsig, win_s, min_pts, scale_floor)
            for tt, yy in zip(ta[m], ya[m]):
                kept[_mjd_to_dt64(tt / 86400.0)] = yy
        if kept:
            out[sat] = kept
            n_out += len(kept)
    print(f"      cleaned: {n_out}/{n_in} records kept "
          f"({100.0*n_out/max(n_in,1):.1f}%)")
    return out
 
 
def _sd_to_xarray(sd):
    """{sat:{dt64:stec}} -> (time, sv) DataArray of SD-STEC [TECU]."""
    sats = sorted(sd)
    times = sorted({t for d in sd.values() for t in d})
    tidx = {t: i for i, t in enumerate(times)}
    M = np.full((len(times), len(sats)), np.nan)
    for j, s in enumerate(sats):
        for t, v in sd[s].items():
            M[tidx[t], j] = v
    return xr.DataArray(
        M, coords={"time": np.array(times, dtype="datetime64[ns]"), "sv": sats},
        dims=["time", "sv"])
 
 
def build_diff_sip_baseline_objs(BaselineObj, fin_file, antenna_names,
                                 rxpos_by_name, baselines,
                                 arc_gap_s=600.0, nsig=5.0, win_s=900.0,
                                 min_pts=5, sigma_ceiling=0.5, scale_floor=0.3):
    """Read a single concatenated FIN.OUT of differential (baseline) SIPs and
    return the same list of BaselineObj as the other paths -- one per TARGET
    baseline (the (i,j) pairs in `baselines`).
 
    The file-index -> baseline mapping is read from the .OUT header; only files
    whose (A,B) matches a requested target baseline are kept.
    """
    fmap = parse_file_baseline_map(fin_file)
    if not fmap:
        raise ValueError(f"could not parse FILE->baseline table from {fin_file}")
    print(f"  [diff-sip] file->baseline map: "
          + ", ".join(f"{k}:{v[0]}-{v[1]}" for k, v in sorted(fmap.items())))
 
    # group records by file index
    recs_by_file = {}
    for fil, sat, mjd, stec, sig in _read_diff_records(fin_file):
        recs_by_file.setdefault(fil, []).append((sat, mjd, stec, sig))
 
    # target baselines as unordered station-pair -> (i,j) for orientation
    want = {}
    for i, j in baselines:
        want[frozenset((antenna_names[i], antenna_names[j]))] = (antenna_names[i],
                                                                 antenna_names[j])
 
    objs = []
    for fil, (A, B) in sorted(fmap.items()):
        key = frozenset((A, B))
        if key not in want:
            continue                       # not a target baseline (e.g. MGO4-MGO5)
        if fil not in recs_by_file:
            print(f"  [diff-sip] file {fil} ({A}-{B}): no records, skip")
            continue
        Aout, Bout = want[key]             # orientation as requested in `baselines`
        print(f"  [diff-sip] file {fil} -> baseline {Aout}-{Bout}:")
        sd = _clean_baseline(recs_by_file[fil], arc_gap_s, nsig, win_s,
                             min_pts, sigma_ceiling, scale_floor)
        if not sd:
            continue
        da = _sd_to_xarray(sd)
        # orient to requested A-B: the file stores stec as (STATION1 - STATION2)
        # = (A - B) per the header. If the request wants (B - A), negate.
        if (Aout, Bout) != (A, B):
            da = -da
        bo = BaselineObj(f"{Aout}-{Bout}", rxpos_by_name[Aout],
                         rxpos_by_name[Bout], da)
        n_t = bo.sd_arr.sizes.get("time", 0)
        n_s = bo.sd_arr.sizes.get("sv", 0)
        print(f"             {n_t} epochs x {n_s} sats")
        objs.append(bo)
    return objs
 
def _mjd_to_dt64(mjd):
    secs = int(round(mjd * 86400.0))
    return _MJD0 + np.timedelta64(secs, "s").astype("timedelta64[ns]")
  
def _read_sion_records(path, keep_file_slot=2):
    """Yield (sat, mjd, stec, sigma, zen) for #SION records in the requested
    file slot only.  GPSEST writes TWO file slots per station per epoch: slot 1
    is a spurious near-zero-datum duplicate, slot 2 carries the physical STEC.
    keep_file_slot=2 selects the real series; set None to keep all (debug)."""
    with open(path) as f:
        for ln in f:
            if "#SION" not in ln:
                continue
            tok = ln.split()
            if tok[-1] != "#SION":
                continue
            try:
                fil = int(tok[_SION_FILE_COL])
                sat = int(tok[_SION_SAT_COL])
                zen = float(tok[_SION_ZEN_COL])
                stec = float(tok[_SION_STEC_COL])
                sig = float(tok[_SION_SIG_COL])
                mjd = float(tok[_SION_MJD_COL])
            except (ValueError, IndexError):
                continue
            if keep_file_slot is not None and fil != keep_file_slot:
                continue
            yield sat, mjd, stec, sig, zen
 
 
def _robust_arc_model(t_s, y, win_s):
    """For each point, the median of points within +/- win_s/2 seconds.
    t_s sorted ascending."""
    model = np.empty_like(y)
    half = win_s / 2.0
    lo = 0
    hi = 0
    n = len(t_s)
    for i in range(n):
        while lo < n and t_s[lo] < t_s[i] - half:
            lo += 1
        while hi < n and t_s[hi] <= t_s[i] + half:
            hi += 1
        model[i] = np.median(y[lo:hi])
    return model
 
 
def _iter_outlier_mask(t_s, y, nsig, win_s, min_pts, scale_floor):
    keep = np.ones(len(y), dtype=bool)
    while True:
        if keep.sum() < min_pts:
            break
        tt = t_s[keep]
        yy = y[keep]
        model = _robust_arc_model(tt, yy, win_s)
        resid = yy - model
        mad = np.median(np.abs(resid - np.median(resid)))
        scale = 1.4826 * mad
        # Floor the scale at a physical noise level. A flexible moving-median
        # model can track the data so closely that MAD -> ~0.01 TECU, which
        # makes even 5-sigma reject normal point-to-point scatter. Real
        # outliers are TECU-sized, well above the floor.
        scale = max(scale, scale_floor)
        if scale <= 0:
            break
        bad_local = np.abs(resid) > nsig * scale
        if not bad_local.any():
            break
        idx = np.where(keep)[0][bad_local]
        keep[idx] = False
    return keep
 
 
def _clean_station(records, arc_gap_s, nsig, win_s, min_pts, sigma_ceiling, scale_floor):
    bysat = {}
    for sat, mjd, stec, sig, zen in records:
        bysat.setdefault(sat, []).append((mjd, stec, sig))
 
    out = {}
    n_in = n_out = 0
    for sat, recs in bysat.items():
        recs.sort()
        mjd = np.array([r[0] for r in recs])
        stec = np.array([r[1] for r in recs])
        sig = np.array([r[2] for r in recs])
        n_in += len(recs)
 
        if sigma_ceiling is not None:
            ok = sig <= sigma_ceiling
            mjd, stec, sig = mjd[ok], stec[ok], sig[ok]
        if len(mjd) == 0:
            continue
 
        t_s = mjd * 86400.0
        gaps = np.where(np.diff(t_s) > arc_gap_s)[0]
        bounds = np.concatenate(([0], gaps + 1, [len(t_s)]))
 
        kept = {}
        for a in range(len(bounds) - 1):
            sl = slice(bounds[a], bounds[a + 1])
            ta, ya = t_s[sl], stec[sl]
            if len(ta) == 0:
                continue
            m = _iter_outlier_mask(ta, ya, nsig, win_s, min_pts, scale_floor)
            for tt, yy in zip(ta[m], ya[m]):
                kept[_mjd_to_dt64(tt / 86400.0)] = yy
        if kept:
            out[sat] = kept
            n_out += len(kept)
    print(f"      cleaned: {n_out}/{n_in} records kept "
          f"({100.0*n_out/max(n_in,1):.1f}%)")
    return out
 
 
def read_station_sip(path, keep_file_slot=2, arc_gap_s=600.0, nsig=5.0,
                     win_s=900.0, min_pts=5, sigma_ceiling=1.0,
                     scale_floor=0.3):
    """Read one FIN_<STA>.OUT -> {sat:{dt64:stec}}.
 
    keep_file_slot: GPSEST emits two file slots per station; slot 2 is the
                    physical STEC, slot 1 a near-zero-datum duplicate. Keep 2.
                    (Selecting the right slot is what removes the 'low band' and
                    the epoch-to-epoch jumping; without it the two slots collide
                    on (sat,epoch) and interleave.)
    nsig          : light backstop outlier cut (5 sigma) on the single clean
                    series; with the duplicate slot removed this only trims a
                    few genuine spikes.
    other args    : as before (arc split, model window, sigma ceiling).
    """
    recs = list(_read_sion_records(path, keep_file_slot=keep_file_slot))
    return _clean_station(recs, arc_gap_s, nsig, win_s, min_pts, sigma_ceiling,
                          scale_floor)
 
 
def station_sip_to_xarray(sip):
    sats = sorted(sip)
    times = sorted({t for d in sip.values() for t in d})
    tidx = {t: i for i, t in enumerate(times)}
    M = np.full((len(times), len(sats)), np.nan)
    for j, s in enumerate(sats):
        for t, v in sip[s].items():
            M[tidx[t], j] = v
    return xr.DataArray(
        M,
        coords={"time": np.array(times, dtype="datetime64[ns]"), "sv": sats},
        dims=["time", "sv"],
    )
 
 
def _station_from_path(path):
    base = os.path.basename(path)
    base = re.sub(r"\.OUT$", "", base, flags=re.IGNORECASE)
    for tok in reversed(base.split("_")):
        if re.fullmatch(r"[A-Za-z0-9]{4}", tok):
            return tok.upper()
    return base.upper()
 
 
def load_sip_stations(sip_files, **clean_kw):
    out = {}
    for p in sip_files:
        sta = _station_from_path(p)
        print(f"  [sip] {sta}:")
        out[sta] = station_sip_to_xarray(read_station_sip(p, **clean_kw))
    return out
 
 
def baseline_obj_from_sip(BaselineObj, sip_by_station, A, B, rxpos_by_name):
    if A not in sip_by_station or B not in sip_by_station:
        return None
    daA = sip_by_station[A]
    daB = sip_by_station[B]
    daA2, daB2 = xr.align(daA, daB, join="inner")
    sd = daB2 - daA2
    return BaselineObj(f"{A}-{B}", rxpos_by_name[A], rxpos_by_name[B], sd)
 
 
def build_sip_baseline_objs(BaselineObj, sip_files, antenna_names,
                            rxpos_by_name, baselines, **clean_kw):
    sip_by_station = load_sip_stations(sip_files, **clean_kw)
    objs = []
    for i, j in baselines:
        A, B = antenna_names[i], antenna_names[j]
        bo = baseline_obj_from_sip(BaselineObj, sip_by_station, A, B,
                                   rxpos_by_name)
        if bo is None:
            print(f"  [sip] baseline {A}-{B}: missing station file, skipped")
            continue
        n_t = bo.sd_arr.sizes.get("time", 0)
        n_s = bo.sd_arr.sizes.get("sv", 0)
        print(f"  [sip] baseline {A}-{B}: {n_t} epochs x {n_s} sats")
        objs.append(bo)
    return objs

def plot_sip_diagnostic(sip_files, baseline_objs, out_prefix="sip_diag"):
    """Full SIP diagnostic, no per-call parameters:
      * one figure of RAW absolute STEC per station (all PRNs) from sip_files
      * one figure of single-difference STEC per baseline (all PRNs) from
        baseline_objs (the same objects the pipeline is about to write).
    Raw per-station arcs should be smooth if the random-walk solve is sound;
    spikes appearing only in the baseline figure indicate a differencing /
    time-matching problem."""
    # ---- per-station raw absolute STEC ----
    sip = {}
    for p in sip_files:
        sta = _station_from_path(p)
        sip[sta] = station_sip_to_xarray(read_station_sip(p))
        print(f"[sip-diag] {sta}: {sip[sta].sizes.get('time',0)} epochs, "
              f"{sip[sta].sizes.get('sv',0)} sats")

    n = len(sip)
    fig, axes = plt.subplots(n, 1, figsize=(15, 4.5*max(n,1)), squeeze=False)
    for k, (sta, da) in enumerate(sip.items()):
        ax = axes[k][0]
        for sv in da.sv.values:
            s = da.sel(sv=sv)
            ax.plot(s.time.values, s.values, ".", ms=1.5, label=str(int(sv)))
        ax.set_title(f"{sta} : raw absolute STEC per satellite")
        ax.set_ylabel("STEC [TECU]"); ax.grid(alpha=0.3)
        ax.legend(title="PRN", fontsize=6, ncol=2, loc="center left",
                  bbox_to_anchor=(1.0, 0.5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    axes[-1][0].set_xlabel("time (UTC)")
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(f"{out_prefix}_stations.png", dpi=130)
    print(f"[sip-diag] wrote {out_prefix}_stations.png")
    plt.close(fig)

    # ---- per-baseline single difference ----
    m = len(baseline_objs)
    if m:
        fig, axes = plt.subplots(m, 1, figsize=(15, 4.5*m), squeeze=False)
        for k, bo in enumerate(baseline_objs):
            ax = axes[k][0]; da = bo.sd_arr
            for sv in da.sv.values:
                s = da.sel(sv=sv)
                ax.plot(s.time.values, s.values, ".", ms=1.5, label=str(int(sv)))
            ax.set_title(f"{bo.baseline_name} : single-difference STEC per satellite")
            ax.set_ylabel("dSTEC [TECU]"); ax.grid(alpha=0.3)
            ax.legend(title="PRN", fontsize=6, ncol=2, loc="center left",
                      bbox_to_anchor=(1.0, 0.5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        axes[-1][0].set_xlabel("time (UTC)")
        fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(f"{out_prefix}_baselines.png", dpi=130)
        print(f"[sip-diag] wrote {out_prefix}_baselines.png")
        plt.close(fig)

def add_args_to_parser(parser_in):
    """Add arguments to parser."""
    parser.add_argument("--res_file", action="append", type=str, metavar="FILE", help = "The Bernese L4 residuals to use in generating the iono_comp object")
    parser.add_argument("--out_file", action="append", type=str, metavar="FILE", help = "The Bernese OUT file produced while running the L4R pipeline (FIN_YYYYMMDD_001.OUT)")
    parser.add_argument("--sip_file", action="append", type=str, metavar="FILE",
                        help="Concatenated FIN_<STA>.OUT with GPSEST #SION stochastic-"
                             "ionosphere records (one file per station). When given, "
                             "STEC comes from SIPs instead of L4 residuals.")
    parser.add_argument("--iono_file_out", default='iono_out.nc',
                         help="NetCDF Ionosphere correction dataset filename/path"
                         )
    parser.add_argument("-e", dest="eph_files", action="append", type=str, nargs="+", help = 'Ephemeris file. Add files for day before and after experiment too.')
    parser.add_argument("--rxpos", action="append", type=str, nargs="+", help="Receiver position  as 'X Y Z' (m)")
    parser.add_argument("--antenna_name", 
                        dest="antenna_names",
                        help="Name of antennas in RINEX files. Order of file input. Should match loading rinex files.", 
                        default=[], 
                        action='append')
    parser.add_argument("--ionex_file", default=None, action="append", type=str, metavar="FILE", nargs="+",
                         help="GIM for bulk ionosphere component.",
                         )
    parser.add_argument("--plot_STEC", action='store_true', default=False,
                         help="Produce plot showing the differential STEC measurements on the baselines",
                         )
    parser.add_argument("--plot_sip_diag", action="store_true", default=False,
                        help="Diagnostic: plot raw per-station absolute STEC and "
                             "every baseline's single difference, then exit.")
    parser.add_argument("--diff_sip_file", type=str, default=None, metavar="FILE",
                        help="Single concatenated FIN.OUT of DIFFERENTIAL (baseline) "
                             "SIPs (L4R-analogue estimate). File index = baseline, "
                             "value = SD-dSTEC directly.")

TECU_COEFF = 40.308193*1e16
def correct_iono(antenna_handles, nav_store, baseline_objs, ionex_store, iono_file_out, plot_STEC, sip_run):
    """
    Take the single-source data and produce a differential position estimate via least-squares adjustment
    """
    stec_baseline_arr = []
    EL_MASK = 5.0   # degrees; mask if low at EITHER station (SD geometry-amplified)
    for bidx, baseline_obj in enumerate(baseline_objs):
        ct_arr = date_to_common(baseline_obj.sd_arr.time.values)
        stec_epochs = []
        for idx, time in enumerate(baseline_obj.sd_arr.time.values):
            stec_epoch = baseline_obj.sd_arr.isel(time=idx).dropna(dim='sv', how='any')
            if stec_epoch.sizes.get('sv', 0) == 0:
                continue
            rxpos_1 = Position(baseline_obj.position1)
            rxpos_2 = Position(baseline_obj.position2)

            keep_sv, stec1_arr, stec2_arr = [], [], []
            for sv in stec_epoch.sv.values:
                sysc, num = bernese_prn_to_satid(sv)
                RSID = RinexSatID(f"{sysc}{num:02d}")
                sat_xvt = nav_store.get_xvt(RSID, ct_arr[idx])
                if sat_xvt is None:
                    raise ValueError(f'no satellite records for {sv} at {time}')
                sat_x_list = [sat_xvt.x[0], sat_xvt.x[1], sat_xvt.x[2]]
                elev1 = rxpos_1.elevationGeodetic(Position(sat_xvt.x))
                elev2 = rxpos_2.elevationGeodetic(Position(sat_xvt.x))
                if min(elev1, elev2) < EL_MASK:        # <-- the mask, on BOTH ends
                    continue
                ts = (time - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
                time_dt = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).replace(tzinfo=None)
                stec1 = ionex_store.calc_stec_full(time_dt, baseline_obj.position1, sat_x_list, elev1)
                stec2 = ionex_store.calc_stec_full(time_dt, baseline_obj.position2, sat_x_list, elev2)
                keep_sv.append(sv); stec1_arr.append(stec1); stec2_arr.append(stec2)

            if not keep_sv:
                continue
            # restrict the residual to the surviving (masked) satellites, THEN add I_bar
            stec_epoch = stec_epoch.sel(sv=keep_sv)
            gim_sd = xr.DataArray(np.array(stec1_arr) - np.array(stec2_arr),
                                  coords={'sv': keep_sv}, dims=['sv'])
            if sip_run:
                # absolute = SIP stochastic residual + per-satellite GIM dSTEC
                I_bar = gim_sd
            else:
                # L4R path: per-epoch SD mean datum (scalar)
                I_bar = float(gim_sd.mean())
            stec_epoch = (stec_epoch + I_bar).expand_dims(time=[time])
            stec_epochs.append(stec_epoch)

        stec_baseline = stack_xr(stec_epochs, 'STEC', 'time', 'sv')
        stec_baseline_arr.append(stec_baseline.expand_dims(baseline=[baseline_obj.baseline_name]))

    stec_full = xr.concat(stec_baseline_arr, dim='baseline', join='outer')

    # Convert the integer sv coordinate to RINEX satellite IDs (e.g. 5 -> 'G05')
    # for the on-disk product.
    def _sv_to_rsid(sv):
        sysc, num = bernese_prn_to_satid(int(sv))
        return f"{sysc}{num:02d}"
    rsid_coord = [_sv_to_rsid(s) for s in stec_full.sv.values]

    stec_full = stec_full.assign_coords(sv=rsid_coord)

    stec_full.to_netcdf(iono_file_out)
    if plot_STEC:
        make_gif = HAVE_IMAGEIO 
        for bl in stec_full.baseline.values:
            ds_bl = stec_full.sel(baseline=bl).dropna(dim='time', how='all')
            ant1, ant2 = _split_bl(bl)
            (r1, rx1, lat1, lon1), (r2, rx2, lat2, lon2) = _get_site_objects(bl)
            stec_frames_this_bl = []
            times = ds_bl.time.values
            ct_arr = date_to_common(times)

            obs_full = np.array(ds_bl['STEC'].values[~np.isnan(ds_bl['STEC'].values)])

            # --- per-satellite STEC time series (static, before the GIF frames) ---

            fig_ts, ax_ts = plt.subplots(figsize=(12, 5), dpi=300, constrained_layout=True)
            t_dt = ds_bl.time.values.astype('datetime64[ns]')   # x-axis as datetime64

            n_sv = ds_bl.sizes.get('sv', 0)
            cmap = plt.get_cmap('turbo', max(n_sv, 1))
            for k, sv in enumerate(ds_bl.sv.values):
                y = ds_bl['STEC'].sel(sv=sv).values
                if np.all(np.isnan(y)):
                    continue
                ax_ts.plot(t_dt, y, '-', lw=0.8, color=cmap(k), label=str(sv))

            ax_ts.set_title(f'{ant1}-{ant2} : single-difference STEC per satellite')
            ax_ts.set_xlabel('time (UTC)')
            ax_ts.set_ylabel('dSTEC [TECU]')
            ax_ts.grid(True, which='both', alpha=0.3)

            # 3-hour major ticks
            ax_ts.xaxis.set_major_locator(mdates.HourLocator(interval=3))
            ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax_ts.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
            fig_ts.autofmt_xdate(rotation=30)

            # legend outside if there are many satellites
            if n_sv:
                ax_ts.legend(ncol=2, fontsize=7, loc='center left',
                             bbox_to_anchor=(1.005, 0.5), title='PRN', framealpha=0.9)

            ts_png = f"{bl}_STEC_timeseries.png"
            fig_ts.savefig(ts_png, bbox_inches='tight')
            plt.close(fig_ts)
            print(f"Saved STEC time series: {ts_png}")
            # --- end per-satellite STEC time series ---

            F1 = 1575.42e6
            TECU_COEFF = 40.308193e16          # 40.3e16, same as producing script
            EPOCH0 = np.datetime64('1970-01-01T00:00:00')

            fig_d, ax_d = plt.subplots(figsize=(12, 5), dpi=100, constrained_layout=True)
            t_dt = ds_bl.time.values.astype('datetime64[ns]')
            ct_arr = date_to_common(t_dt)
            n_sv = ds_bl.sizes.get('sv', 0)
            cmap = plt.get_cmap('turbo', max(n_sv, 1))

            for k, sv in enumerate(ds_bl.sv.values):
                obs = ds_bl['STEC'].sel(sv=sv).values         # observed SD-dSTEC [TECU]
                if np.all(np.isnan(obs)):
                    continue
                model = np.full(obs.shape, np.nan)
                RSID = RinexSatID(sv)
                for i, t in enumerate(t_dt):
                    if np.isnan(obs[i]):
                        continue
                    xvt = nav_store.get_xvt(RSID, ct_arr[i])
                    if xvt is None:
                        continue
                    sat_x_list = [xvt.x[0], xvt.x[1], xvt.x[2]]
                    satp = Position(xvt.x)
                    el1 = rx1.elevationGeodetic(satp)
                    el2 = rx2.elevationGeodetic(satp)
                    if min(el1, el2) < 10.0:                  # elevation mask -> drop the spikes
                        continue
                    ts = (t - EPOCH0) / np.timedelta64(1, 's')
                    tdt = datetime.datetime.utcfromtimestamp(float(ts))
                    d1 = ionex_store.calc_iono_correction(tdt, F1, r1, sat_x_list, el1)  # m
                    d2 = ionex_store.calc_iono_correction(tdt, F1, r2, sat_x_list, el2)  # m
                    model[i] = (d1 - d2) * F1**2 / TECU_COEFF   # SD-dSTEC model [TECU], sta1-sta2
                diff = obs - model
                ax_d.plot(t_dt, diff, '-', lw=0.8, color=cmap(k), label=str(sv))

            ax_d.axhline(0.0, color='k', lw=0.6, alpha=0.5)
            ax_d.set_title(f'{ant1}-{ant2} : observed dSTEC - IONEX model dSTEC')
            ax_d.set_xlabel('time (UTC)')
            ax_d.set_ylabel('dSTEC residual [TECU]')
            ax_d.grid(True, which='both', alpha=0.3)
            ax_d.xaxis.set_major_locator(mdates.HourLocator(interval=3))
            ax_d.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax_d.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
            fig_d.autofmt_xdate(rotation=30)
            if n_sv:
                ax_d.legend(ncol=2, fontsize=7, loc='center left',
                            bbox_to_anchor=(1.005, 0.5), title='PRN', framealpha=0.9)
            d_png = f"{bl}_dSTEC_minus_IONEX.png"
            fig_d.savefig(d_png, bbox_inches='tight')
            plt.close(fig_d)
            print(f"Saved obs-minus-model: {d_png}")

            # --- two-panel: observed dSTEC (top) vs GIM model dSTEC (bottom) ---
            fig_vg, (ax_o, ax_m) = plt.subplots(
                2, 1, figsize=(12, 9), dpi=100, sharex=True, constrained_layout=True)

            for k, sv in enumerate(ds_bl.sv.values):
                obs = ds_bl['STEC'].sel(sv=sv).values        # observed SD-dSTEC [TECU]
                if np.all(np.isnan(obs)):
                    continue
                model = np.full(obs.shape, np.nan)
                RSID = RinexSatID(sv)
                for i, t in enumerate(t_dt):
                    if np.isnan(obs[i]):
                        continue
                    xvt = nav_store.get_xvt(RSID, ct_arr[i])
                    if xvt is None:
                        continue
                    sat_x_list = [xvt.x[0], xvt.x[1], xvt.x[2]]
                    satp = Position(xvt.x)
                    el1 = rx1.elevationGeodetic(satp)
                    el2 = rx2.elevationGeodetic(satp)
                    if min(el1, el2) < 10.0:                  # same mask as residual
                        continue
                    ts = (t - EPOCH0) / np.timedelta64(1, 's')
                    tdt = datetime.datetime.utcfromtimestamp(float(ts))
                    d1 = ionex_store.calc_iono_correction(tdt, F1, r1, sat_x_list, el1)
                    d2 = ionex_store.calc_iono_correction(tdt, F1, r2, sat_x_list, el2)
                    model[i] = (d1 - d2) * F1**2 / TECU_COEFF  # SD-dSTEC model [TECU], sta1-sta2
                # mask observed to the same epochs the model is defined on, so the
                # two panels are directly comparable point-for-point
                obs_m = np.where(np.isnan(model), np.nan, obs)
                ax_o.plot(t_dt, obs_m, '-', lw=0.8, color=cmap(k), label=str(sv))
                ax_m.plot(t_dt, model, '-', lw=0.8, color=cmap(k))

            ax_o.axhline(0.0, color='k', lw=0.6, alpha=0.5)
            ax_m.axhline(0.0, color='k', lw=0.6, alpha=0.5)
            ax_o.set_title(f'{ant1}-{ant2} : observed dSTEC (SIP)')
            ax_m.set_title(f'{ant1}-{ant2} : GIM model dSTEC')
            ax_o.set_ylabel('dSTEC [TECU]')
            ax_m.set_ylabel('dSTEC [TECU]')
            ax_m.set_xlabel('time (UTC)')
            for a in (ax_o, ax_m):
                a.grid(True, which='both', alpha=0.3)
                a.xaxis.set_major_locator(mdates.HourLocator(interval=3))
                a.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                a.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
            fig_vg.autofmt_xdate(rotation=30)
            if n_sv:
                ax_o.legend(ncol=2, fontsize=7, loc='center left',
                            bbox_to_anchor=(1.005, 0.5), title='PRN', framealpha=0.9)
            vg_png = f"{bl}_dSTEC_vs_GIM.png"
            fig_vg.savefig(vg_png, bbox_inches='tight')
            plt.close(fig_vg)
            print(f"Saved obs-vs-model two-panel: {vg_png}")

            #vabs = max(np.percentile(np.abs(obs_full), 95.0), 1e-3) 
            #norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=+vabs)
            #for it, t in enumerate(times):
            #    ds_epoch = ds_bl.sel(time=t).dropna(dim='sv', how='any')
            #    if ds_epoch.sizes.get('sv', 0) == 0:
            #        continue

            #    fig = plt.figure(figsize=(11, 4.5), dpi=100, constrained_layout=True)
            #    gs  = fig.add_gridspec(nrows=1, ncols=3,
            #                                   width_ratios=[1.0, 1.0, 0.025],
            #                                                          wspace=0.05)
            #    axL  = fig.add_subplot(gs[0, 0], projection='polar')
            #    axR  = fig.add_subplot(gs[0, 1], projection='polar')   
            #    caxR = fig.add_subplot(gs[0, 2])                       
            #
            #    # Collect residuals & sky positions (degrees for plotting)
            #    az1_pts, el1_pts = [], []
            #    az2_pts, el2_pts = [], []
            #    obs_vals = []
            #    for sv in ds_epoch.sv.values:
            #        RSID = RinexSatID(str(sv))
            #        xvt = nav_store.get_xvt(RSID, ct_arr[it])
            #        if xvt is None:
            #            continue
            #
            #        e1 = rx1.elevationGeodetic(Position(xvt.x))     # deg
            #        a1z = rx1.azimuthGeodetic(Position(xvt.x))      # deg
            #        e2 = rx2.elevationGeodetic(Position(xvt.x))     # deg
            #        a2z = rx2.azimuthGeodetic(Position(xvt.x))      # deg
            #
            #        obs = float(ds_epoch['STEC'].sel(sv=sv).values)
            #
            #        az1_pts.append(a1z); el1_pts.append(e1);
            #        az2_pts.append(a2z); el2_pts.append(e2);
            #        obs_vals.append(obs)
            #
            #    if obs_vals:
            #        axL.set_theta_zero_location('N')
            #        axL.set_theta_direction(-1)
            #        axL.set_rlim(90, 0); 
            #        axL.set_title(f'{ant1} sky plot', loc='left')
            #        axL.set_aspect(1.0, adjustable='box', anchor='C') 
            #        theta1 = np.deg2rad(np.asarray(az1_pts))
            #        r1 = np.asarray(el1_pts)
            #        scL = axL.scatter(theta1, r1, c=obs_vals, cmap='coolwarm', norm=norm, s=20, alpha=0.9)
            #
            #        axR.set_theta_zero_location('N')
            #        axR.set_theta_direction(-1)
            #        axR.set_rlim(90, 0); 
            #        axR.set_title(f'{ant2} sky plot', loc='left')
            #        axR.set_aspect(1.0, adjustable='box', anchor='C') 
            #        theta2 = np.deg2rad(np.asarray(az2_pts))
            #        r2 = np.asarray(el2_pts)
            #        scR = axR.scatter(theta2, r2, c=obs_vals, cmap='coolwarm', norm=norm, s=20, alpha=0.9)
            #        cbR = fig.colorbar(scR, cax=caxR)
            #        cbR.set_label('dSTEC [TECU]')
            #
            #    frame_png = f"{bl}_dSTEC_frame_{it:05d}.png"
            #    fig.savefig(frame_png, dpi=100)
            #    plt.close(fig)
            #    stec_frames_this_bl.append(frame_png)
            #
            #if make_gif and stec_frames_this_bl:
            #    gif_path = f"{bl}_dSTEC_field.gif"
            #    with imageio.get_writer(gif_path, mode='I', duration=0.15, loop=0) as writer:
            #        for fp in stec_frames_this_bl:
            #            writer.append_data(imageio.imread(fp))
            #    print(f"Saved GIF: {gif_path}")
            #else:
            #    print(f"Saved {len(stec_frames_this_bl)} PNG STEC frames  for {bl}")

    return 


def sip_sign_test(sip_files, antenna_names, rxpos_by_name,
                  nav_store, ionex_store, bernese_prn_to_satid,
                  RinexSatID, Position, date_to_common,
                  prn=(5,), el_mask=5.0, out_prefix="sip_sign"):
    # per-station SIP: {station: {sat: {dt64: stec}}}
    sip = {}
    for p in sip_files:
        sta = _station_from_path(p)
        sip[sta] = read_station_sip(p)        # default cleaning (slot 2, etc.)
 
    EPOCH0 = np.datetime64('1970-01-01T00:00:00')
 
    for sta in sip:
        if sta not in rxpos_by_name:
            print(f"[sign] {sta}: no position, skip"); continue
        pos = rxpos_by_name[sta]
        rxpos = Position(pos) if not isinstance(pos, Position) else pos
 
        fig, axes = plt.subplots(len(prn), 1, figsize=(14, 4.2*len(prn)),
                                 squeeze=False)
        for pi, sv in enumerate(prn):
            ax = axes[pi][0]
            if sv not in sip[sta]:
                ax.set_title(f"{sta} PRN{sv}: not present"); continue
            # ordered SIP series for this sat
            items = sorted(sip[sta][sv].items())
            times = np.array([t for t, _ in items], dtype='datetime64[ns]')
            svals = np.array([v for _, v in items], float)
            ct = date_to_common(times)
 
            gim = np.full(svals.shape, np.nan)
            sysc, num = bernese_prn_to_satid(sv)
            RSID = RinexSatID(f"{sysc}{num:02d}")
            for i, t in enumerate(times):
                xvt = nav_store.get_xvt(RSID, ct[i])
                if xvt is None:
                    continue
                satp = Position(xvt.x)
                el = rxpos.elevationGeodetic(satp)
                if el < el_mask:
                    continue
                ts = (t - EPOCH0) / np.timedelta64(1, 's')
                tdt = datetime.datetime.utcfromtimestamp(float(ts))
                gim[i] = ionex_store.calc_stec_full(
                    tdt, pos, [xvt.x[0], xvt.x[1], xvt.x[2]], el)  # TECU slant
 
            plus = gim + svals
            minus = gim - svals
            ax.plot(times, gim,  '-', lw=1.0, color='0.5', label='GIM slant')
            ax.plot(times, plus, '-', lw=0.9, color='tab:blue', label='GIM + SIP')
            ax.plot(times, minus,'-', lw=0.9, color='tab:red',  label='GIM - SIP')
            ax.axhline(0.0, color='k', lw=0.6, alpha=0.5)
            # quick numeric discriminator: count negatives + roughness
            for lbl, arr in (('GIM+SIP', plus), ('GIM-SIP', minus)):
                good = arr[~np.isnan(arr)]
                if good.size:
                    nneg = int((good < 0).sum())
                    rough = float(np.nanmean(np.abs(np.diff(good))))
                    print(f"[sign] {sta} PRN{sv} {lbl}: "
                          f"min={good.min():.1f} negs={nneg} rough={rough:.3f}")
            ax.set_title(f"{sta} PRN{sv}: per-station absolute STEC reconstruction")
            ax.set_ylabel("STEC [TECU]")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc='best')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        axes[-1][0].set_xlabel("time (UTC)")
        fig.autofmt_xdate()
        fig.tight_layout()
        fn = f"{out_prefix}_{sta}.png"
        fig.savefig(fn, dpi=130)
        plt.close(fig)
        print(f"[sign] wrote {fn}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args_to_parser(parser)
    args = parser.parse_args()
        
    antenna_names = [antenna_name for antenna_name in args.antenna_names]


    # Pair up per-day (res, out) files. CHANGE add_args_to_parser so --res_file
    # and --out_file use action='append'; pass one pair per day in matching order.
    if args.sip_file:
        day_specs = []                       # not used in SIP mode
    else:
        res_files = args.res_file if isinstance(args.res_file, (list, tuple)) else [args.res_file]
        out_files = args.out_file if isinstance(args.out_file, (list, tuple)) else [args.out_file]
        if len(res_files) != len(out_files):
            raise InsufficientDataError("Need one --out_file per --res_file (matched by order).")
        day_specs = list(zip(res_files, out_files))
        print(f"combining {len(day_specs)} day(s)")

    if args.rxpos is not None:
        rxpos_all = []
        for rxpos_arg in args.rxpos:
            rxpos = [float(pos_comp) for pos_comp in rxpos_arg[0].split()]
            rxpos_all.append(rxpos)

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

    baselines = list(itertools.combinations(range(len(antenna_names)), 2))
    # have to use std_vector_string b/c typemap for list of strings to std_vec<std_string> isnt working
    ant_names_cpp = std_vector_string(antenna_names)

    # initialize the antenna handles
    antenna_handles = []
    for antenna_idx, antenna_name in enumerate(antenna_names):
        antenna_position = rxpos_all[antenna_idx]
        antenna_handle = AntennaInfo(antenna_name, antenna_position, '', 0, False) 
        antenna_handles.append(antenna_handle)

    rxpos_by_name = {name: rxpos_all[i] for i, name in enumerate(antenna_names)}

    if args.diff_sip_file:
        print(f"DIFFERENTIAL SIP MODE: {args.diff_sip_file}")
        baseline_objs = build_diff_sip_baseline_objs(
            BaselineObj, args.diff_sip_file, antenna_names, rxpos_by_name, baselines)
        sip_run = True     # already baseline-differenced -> L4R-style correction
    elif args.sip_file:
        sip_run = True
        # ---- SIP path: STEC from GPSEST #SION output, single-differenced ----
        print(f"SIP MODE: {len(args.sip_file)} station file(s)")
        baseline_objs = build_sip_baseline_objs(
            BaselineObj, args.sip_file, antenna_names, rxpos_by_name, baselines)
    else:
        sip_run = False
        # ---- original L4-residual path (unchanged) ----
        results_abs, bsl = combine_days_results(day_specs)   # NO zd computed
        zd = None                                            # lazy
        baseline_objs = []
        for i, j in baselines:
            A, B = antenna_names[i], antenna_names[j]
            bo = baseline_obj_direct(results_abs, bsl, A, B, rxpos_by_name)
            if bo is None:
                print('ZERO DIFFERENCING DATA')
                if zd is None:
                    zd = build_zd(results_abs, bsl) # built once, on first miss
                sd_dict, _ = single_difference(zd, A, B, to_tec=True)
                if not sd_dict:
                    continue
                bo = BaselineObj(f"{A}-{B}", rxpos_all[i], rxpos_all[j], _sd_to_xarray(sd_dict))
            baseline_objs.append(bo)

    if args.plot_sip_diag:
        plot_sip_diagnostic(args.sip_file or [], baseline_objs,
                            out_prefix=args.iono_file_out.replace(".nc", "_sipdiag"))

    # build the IONEX ionosphere store
    if args.ionex_file is not None:
        ionex_files = [item for sublist in args.ionex_file for item in sublist]
        ionex_store = IonexFile.read(ionex_files)
    else:
        ionex_store = None

    #sip_sign_test(args.sip_file, antenna_names, rxpos_by_name,
    #              nav_store, ionex_store, bernese_prn_to_satid,
    #              RinexSatID, Position, date_to_common,
    #              prn=[5], el_mask=5.0)

    correct_iono(antenna_handles, nav_store, baseline_objs, ionex_store, args.iono_file_out, args.plot_STEC, sip_run)

