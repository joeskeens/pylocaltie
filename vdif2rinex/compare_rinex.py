#!/usr/bin/python
"""
==============================================================================
  This file is part of the VDIF2RINEX software package.

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
  University of Texas at Austin.

  Copyright 2026, The Board of Regents of The University of Texas System
==============================================================================
"""
import os
import argparse
import xarray as xr
import re
import numpy as np

use_custom_version = os.getenv('USE_CUSTOM_GEORINEX', 'false').lower() == 'true'
if use_custom_version:
    import sys
    #sys.path.insert(0, '/sgl/ceph/work/jskeens')
    sys.path.insert(0, '/home/jskeens/oscar_dir/scratch/jskeens')
    sys.path.insert(0, '/trashcan/scratch/jskeens')
    from georinex_custom import load
else:
    from georinex import load
from matplotlib import pyplot as plt

def add_args_to_parser(parser_in):
    """Add arguments to parser."""
    parser.add_argument("--rinex_file_1", required=True,
                         help="Filepath of rinex file 1 in comparison"
                         )
    parser.add_argument("--rinex_file_2", required=True,
                         help="Filepath of rinex file 2 in comparison"
                         )

CANON_GROUPS = {
    # GPS L1C: treat D and L as equivalent for differencing
    ("G", "L1"): {"1S", "1C", "1L"},
    ("G", "L2"): {"2S", "2L"},
    ("G", "L5"): {"5I", "5Q"},

    # Galileo E1: treat B and C as equivalent
    ("E", "E1"): {"1B", "1C"},
    ("E", "E5"): {"5I", "5Q"},

    # BeiDou B1C: treat D and P as equivalent
    ("C", "B1C"): {"1D", "1P"},
    ("C", "B5"): {"5D", "5P"},
}


def _sv_system(sv: str) -> str:
    # "G28" -> "G", "E34" -> "E", "C19" -> "C"
    return str(sv)[0]


def _parse_obs_var(var_name: str):
    """
    Parse e.g. "C1D" -> ("C", "1D").
    Returns (obs_letter, suffix) or None.
    """
    if len(var_name) < 2:
        return None
    obs_letter = var_name[0]
    suffix = var_name[1:]
    if obs_letter not in ("C", "L", "D", "S"):
        return None
    return obs_letter, suffix


def _build_canonical_lookup(ds: xr.Dataset, sv_system_letter: str):
    """
    For a given dataset and SV system (G/E/C), build mapping:
      canonical_key = (obs_letter, canon_band) -> actual var name in ds

    If multiple candidates exist, picks the first encountered (deterministic sort).
    """
    # Collect candidates by canonical key
    canon_to_vars = {}

    for var in sorted(ds.data_vars):
        parsed = _parse_obs_var(var)
        if parsed is None:
            continue
        obs_letter, suffix = parsed

        # Find which canonical group (if any) this suffix belongs to for this system.
        canon_band = None
        for (sys_letter, band), allowed_suffixes in CANON_GROUPS.items():
            if sys_letter == sv_system_letter and suffix in allowed_suffixes:
                canon_band = band
                break

        if canon_band is None:
            # Not aliased; treat as its own "band" using suffix literally.
            # This keeps everything else working unchanged.
            canon_band = suffix

        canon_key = (obs_letter, canon_band)
        canon_to_vars.setdefault(canon_key, []).append(var)

    # Choose one var per canonical key (prefer deterministic choice)
    canon_lookup = {k: v[0] for k, v in canon_to_vars.items()}
    return canon_lookup


def _paired_finite(da1: xr.DataArray, da2: xr.DataArray):
    finite_mask = xr.ufuncs.isfinite(da1) & xr.ufuncs.isfinite(da2)
    da1c = da1.where(finite_mask, drop=True)
    da2c = da2.where(finite_mask, drop=True)
    return da1c["time"].values, da1c.values, da2c.values


def _plot_two_panel_series(time_values, y1, y2, label1, label2, ylabel_top, ylabel_bottom, title, out_path):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(11, 7), sharex=True)
    ax_top, ax_bot = axes

    ax_top.plot(time_values, y1, label=label1)
    ax_top.plot(time_values, y2, label=label2)
    ax_top.set_ylabel(ylabel_top)
    ax_top.set_title(title)
    ax_top.legend(loc="best")
    ax_top.grid(True, alpha=0.3)

    ax_bot.plot(time_values, y1 - y2)
    ax_bot.set_ylabel(ylabel_bottom)
    ax_bot.set_xlabel("time")
    ax_bot.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_single_series(time_values, y1, y2, label1, label2, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(time_values, y1, label=label1)
    ax.plot(time_values, y2, label=label2)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("time")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_common_sv_signal_types(
    rinex_data_1: xr.Dataset,
    rinex_data_2: xr.Dataset,
    label_1: str = "rinex_file_1",
    label_2: str = "rinex_file_2",
    out_dir: str = ".",
):
    # Align on common coordinates first (time, sv). Variables handled via canonical lookup later.
    ds1a, ds2a = xr.align(rinex_data_1, rinex_data_2, join="inner")

    # Identify common SVs (after align, these are common by construction)
    common_svs = [str(sv) for sv in ds1a["sv"].values]
    if not common_svs:
        raise ValueError("No common SVs after coordinate alignment (join='inner').")

    os.makedirs(out_dir, exist_ok=True)

    for sv in common_svs:
        sys_letter = _sv_system(sv)

        ds1_sv = ds1a.sel(sv=sv)
        ds2_sv = ds2a.sel(sv=sv)

        ds1_sv = ds1_sv.drop_vars(
                    [v for v in ds1_sv.data_vars if ds1_sv[v].isnull().all()]
                    )
        ds2_sv = ds2_sv.drop_vars(
                    [v for v in ds2_sv.data_vars if ds2_sv[v].isnull().all()]
                    )

        # Build canonical lookups for this system
        canon1 = _build_canonical_lookup(ds1_sv, sys_letter)
        canon2 = _build_canonical_lookup(ds2_sv, sys_letter)

        # Common canonical keys between files
        common_canon_keys = sorted(set(canon1) & set(canon2))

        # Only consider keys for which we have at least one finite sample in both
        def _has_any_finite(varname, ds):
            return bool(xr.ufuncs.isfinite(ds[varname]).any().item())

        usable_keys = []
        for key in common_canon_keys:
            v1 = canon1[key]
            v2 = canon2[key]
            if _has_any_finite(v1, ds1_sv) and _has_any_finite(v2, ds2_sv):
                usable_keys.append(key)

        # Group by canonical band so we can do (C,L,D,S) plots per band
        bands = sorted({band for (obs, band) in usable_keys})

        for band in bands:
            # Resolve per-observable variable names for this canonical band
            def _get(obs_letter):
                return canon1.get((obs_letter, band)), canon2.get((obs_letter, band))

            # ---- Pseudorange
            v1, v2 = _get("C")
            if v1 and v2:
                t, y1, y2 = _paired_finite(ds1_sv[v1], ds2_sv[v2])
                if t.size > 0:
                    out_path = os.path.join(out_dir, f"{sv}_{band}_C_{v1}_vs_{v2}_{label_1}_{label_2}.png")
                    _plot_two_panel_series(
                        t, y1, y2,
                        f"{label_1} ({v1})", f"{label_2} ({v2})",
                        ylabel_top="Pseudorange",
                        ylabel_bottom=f"{label_1}-{label_2}",
                        title=f"{sv} {band}: Pseudorange (top) and difference (bottom)",
                        out_path=out_path,
                    )

            # ---- ADR
            v1, v2 = _get("L")
            if v1 and v2:
                c = 299792458
                if v1[:2] == 'L1':
                    wavelength = c/1575.42e6
                elif v1[:2] == 'L2':
                    wavelength = c/1227.60e6
                elif v1[:2] == 'L5':
                    wavelength = c/1176.45e6
                elif v1[:3] == 'L7Q':
                    wavelength = c/1207.14e6
                elif v1[:3] == 'L6I':
                    wavelength = c/1268.52e6
                elif v1[:3] == 'L6C':
                    wavelength = c/1278.75e6
                t, y1, y2 = _paired_finite(ds1_sv[v1], ds2_sv[v2])
                delta = np.rint(y2-y1)
                y1 += delta
                if t.size > 0:
                    out_path = os.path.join(out_dir, f"{sv}_{band}_L_{v1}_vs_{v2}_{label_1}_{label_2}.png")
                    _plot_two_panel_series(
                        t, y1, y2,
                        f"{label_1} ({v1})", f"{label_2} ({v2})",
                        ylabel_top="ADR (cycles)",
                        ylabel_bottom=f"{label_1}-{label_2}",
                        title=f"{sv} {band}: ADR (top) and difference (bottom)",
                        out_path=out_path,
                    )

            # ---- Doppler
            v1, v2 = _get("D")
            if v1 and v2:
                t, y1, y2 = _paired_finite(ds1_sv[v1], ds2_sv[v2])
                if t.size > 0:
                    out_path = os.path.join(out_dir, f"{sv}_{band}_D_{v1}_vs_{v2}_{label_1}_{label_2}.png")
                    _plot_two_panel_series(
                        t, y1, y2,
                        f"{label_1} ({v1})", f"{label_2} ({v2})",
                        ylabel_top="Doppler shift (Hz)",
                        ylabel_bottom=f"{label_1}-{label_2}",
                        title=f"{sv} {band}: Doppler shift (top) and difference (bottom)",
                        out_path=out_path,
                    )

            # ---- SNR
            v1, v2 = _get("S")
            if v1 and v2:
                t, y1, y2 = _paired_finite(ds1_sv[v1], ds2_sv[v2])
                if t.size > 0:
                    out_path = os.path.join(out_dir, f"{sv}_{band}_S_{v1}_vs_{v2}_{label_1}_{label_2}.png")
                    _plot_single_series(
                        t, y1, y2,
                        f"{label_1} ({v1})", f"{label_2} ({v2})",
                        ylabel="C/N0 (dB-Hz)",
                        title=f"{sv} {band}: C/N0 (dB-Hz)",
                        out_path=out_path,
                    )

if __name__ == '__main__':    
    ### Parse command-line options
    parser = argparse.ArgumentParser()
    add_args_to_parser(parser)
    args = parser.parse_args()

    rinex_data_1 = load(args.rinex_file_1)
    rinex_data_2 = load(args.rinex_file_2)
    plot_common_sv_signal_types(
        rinex_data_1,
        rinex_data_2,
        label_1=getattr(rinex_data_1, "filename", "rinex_file_1"),
        label_2=getattr(rinex_data_2, "filename", "rinex_file_2"),
        out_dir=".",
    )

