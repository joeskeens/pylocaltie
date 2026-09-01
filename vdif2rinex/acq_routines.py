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
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List
from numba import njit
from math import floor
from datetime import datetime, timedelta, timezone

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# soft_syms: float32 array, +/-1 (or larger magnitudes), shape (N,)
# Convention below: +1 == bit 0, -1 == bit 1 (standard "soft bit" where sign=bit).

SYNC_BITS = np.array([0,1,0,1,1,0,0,0,0,0], dtype=np.int8)
SYNC_PM   = np.where(SYNC_BITS == 0, 1.0, -1.0).astype(np.float32)  # +1/-1

def find_sync(soft, search_len=600):
    """Return (start_index, polarity) where polarity is +1 or -1."""
    win = sliding_window_view(soft[:search_len + 10], 10)
    corr = win @ SYNC_PM
    idx = int(np.argmax(np.abs(corr)))
    pol = 1.0 if corr[idx] >= 0 else -1.0
    return idx, pol

#def deinterleave(sym240):
#    # Written by columns (30 cols x 8 rows), read by rows
#    M = sym240.reshape(30, 8).T      # fill columns -> transpose
#    return M.reshape(-1)

def deinterleave(sym240):
    M = sym240.reshape(8, 30)       # was (30, 8).T
    return M.T.reshape(-1)

def viterbi_k7(soft240):
    """Rate-1/2, K=7, G1=0o171, G2=0o133 inverted. Zero-start, zero-end."""
    K, N_STATES = 7, 64
    G1, G2 = 0o171, 0o133
    # Precompute output symbols (+/-1) for each (state, input)
    out = np.zeros((N_STATES, 2, 2), dtype=np.float32)
    nxt = np.zeros((N_STATES, 2), dtype=np.int32)
    for s in range(N_STATES):
        for u in (0, 1):
            reg = (u << (K-1)) | s
            b1 = bin(reg & G1).count("1") & 1
            b2 = (bin(reg & G2).count("1") & 1) ^ 1  # G2 inverted
            # bit 0 -> +1, bit 1 -> -1
            out[s, u, 0] = 1 - 2*b1
            out[s, u, 1] = 1 - 2*b2
            nxt[s, u] = (reg >> 1) & (N_STATES - 1)

    n_steps = len(soft240) // 2
    INF = 1e18
    pm = np.full(N_STATES, INF, dtype=np.float64); pm[0] = 0.0
    tb = np.zeros((n_steps, N_STATES), dtype=np.int8)
    prev = np.zeros((n_steps, N_STATES), dtype=np.int32)

    for t in range(n_steps):
        r = soft240[2*t:2*t+2]
        new_pm = np.full(N_STATES, INF)
        for s in range(N_STATES):
            if pm[s] >= INF: continue
            for u in (0, 1):
                ns = nxt[s, u]
                # negative correlation = metric to minimize
                m = pm[s] - (out[s,u,0]*r[0] + out[s,u,1]*r[1])
                if m < new_pm[ns]:
                    new_pm[ns] = m
                    tb[t, ns] = u
                    prev[t, ns] = s
        pm = new_pm

    # Traceback from state 0 (encoder flushed)
    s = 0
    bits = np.zeros(n_steps, dtype=np.int8)
    for t in range(n_steps-1, -1, -1):
        bits[t] = tb[t, s]
        s = prev[t, s]
    return bits  # 120 bits

def decode_e1b_with_time(soft, t_sym):
    soft = soft.astype(np.float32)
    start, pol = find_sync(soft)
    soft_a = pol * soft[start:]
    t_a    = t_sym[start:]

    PAGE = 250
    n_full = len(soft_a) // PAGE
    parts, part_times = [], []
    for k in range(n_full):
        i0 = k * PAGE
        block = soft_a[i0:i0+PAGE]
        if np.sign(block[:10]) @ SYNC_PM < 6:     # weak sync -> bail
            break
        parts.append(viterbi_k7(deinterleave(block[10:])))
        part_times.append(t_a[i0])

    # --- pairing: 2x 120-bit parts -> one 240-bit page ---
    words, word_times = [], []
    i = 1 if (parts and parts[0][0] == 1) else 0   # skip orphan odd at start
    while i + 1 < len(parts):
        if parts[i][0] == 0 and parts[i+1][0] == 1:
            words.append(np.concatenate([parts[i], parts[i+1]]))
            word_times.append(part_times[i])
            i += 2
        else:
            i += 1                                 # misaligned, resync
    return np.array(words, dtype=np.int8), np.array(word_times)

GST_EPOCH_UTC = datetime(1999, 8, 22, 0, 0, 0, tzinfo=timezone.utc)
LEAP_DEFAULT  = 18   # GST - UTC, valid post-2017-01-01; overridden by WT6 dt
def _u(bits):
    """Unsigned int from iterable of 0/1."""
    return int("".join(str(int(x)) for x in bits), 2)
 
def _s(bits):
    """Signed two's-complement int from iterable of 0/1."""
    v = _u(bits)
    return v - (1 << len(bits)) if int(bits[0]) else v
 
def decode_inav_page(page_bits):
    """
    Decode one Galileo I/NAV nominal page.
 
    Parameters
    ----------
    page_bits : array-like of 0/1, length 240
        [even(120) | odd(120)], framing bits included.
 
    Returns
    -------
    dict with always-present keys {'word_type'} plus the fields carried by
    that word type. For unknown / reserved / alert pages, only 'word_type'
    (possibly None) and 'alert_page' flags are set.
    """
    b = np.asarray(page_bits, dtype=np.int8).ravel()
    if b.size < 240:
        raise ValueError(f"page must be 240 bits, got {b.size}")
    b = b[:240]
 
    alert = bool(b[1]) or bool(b[121])
    if alert:
        return {"word_type": None, "alert_page": True}
 
    # 128-bit data word = first 112 bits from even half + first 16 bits from odd half
    dw = np.concatenate([b[2:114], b[122:138]])
    wt = _u(dw[0:6])
 
    out = {"word_type": wt, "alert_page": False}
 
    if wt == 0:
        # I/NAV spare word — direct GST
        out["time_flag"] = _u(dw[6:8])          # 2 = synced to GST
        out["WN"]        = _u(dw[96:108])
        out["TOW"]       = _u(dw[108:128])
 
    elif wt == 1:
        # Ephemeris 1/4
        out["IODnav"] = _u(dw[6:16])
        out["t_0e"]   = _u(dw[16:30]) * 60       # s of week, 60 s LSB
 
    elif wt in (2, 3):
        # Ephemeris 2/4, 3/4 — only IODnav is time-relevant here
        out["IODnav"] = _u(dw[6:16])
 
    elif wt == 4:
        # Ephemeris 4/4 (SV clock)
        out["IODnav"] = _u(dw[6:16])
        out["SVID"]   = _u(dw[16:22])
        out["t_0c"]   = _u(dw[54:68]) * 60       # clock ref, s of week
 
    elif wt == 5:
        # Iono / BGD / signal health + GST
        out["WN"]  = _u(dw[73:85])
        out["TOW"] = _u(dw[85:105])
 
    elif wt == 6:
        # GST-UTC conversion
        out["A0"]     = _s(dw[6:38])  * 2**-30
        out["A1"]     = _s(dw[38:62]) * 2**-50
        out["dtLS"]   = _s(dw[62:70])
        out["t_0t"]   = _u(dw[70:78]) * 3600
        out["WN_0t"]  = _u(dw[78:86])             # WN mod 256
        out["WN_LSF"] = _u(dw[86:94])             # WN mod 256
        out["DN"]     = _u(dw[94:97])
        out["dtLSF"]  = _s(dw[97:105])
        out["TOW"]    = _u(dw[105:125])
 
    # 7-10 = almanac, 16 = reduced CED — not decoded here
    return out
 
 
# ---------- convenience: batch decode + absolute-time attachment ----------
def gst_to_utc(wn, tow, leap=LEAP_DEFAULT):
    return GST_EPOCH_UTC + timedelta(weeks=int(wn), seconds=float(tow) - leap)
 
def decode_nav_list(nav_data, attach_utc=True):
    """
    Decode a list of I/NAV pages and return a list of parameter dicts.
 
    If attach_utc=True, a pass is made over the results to:
      * read dtLS from any WT 6 page and use it as the leap-second count,
      * pick the absolute WN from any WT 0 or WT 5 page,
      * add 'utc' to every dict whose time field can be anchored
        (WT 0/5 directly; WT 1/4/6 via the inferred WN; WT 6 via TOW).
    """
    results = [decode_inav_page(p) for p in nav_data]
 
    if not attach_utc:
        return results
 
    # Infer absolute WN and leap seconds from the set
    absolute_wn = None
    leap = LEAP_DEFAULT
    for r in results:
        if r.get("word_type") in (0, 5) and "WN" in r:
            absolute_wn = r["WN"]
        if r.get("word_type") == 6 and "dtLS" in r:
            leap = r["dtLS"]
 
    for r in results:
        wt = r.get("word_type")
        if wt is None:
            continue
        wn  = r.get("WN", absolute_wn)
        tow = r.get("TOW")
        ## For ephemeris pages (1,4) fall back to t_0e / t_0c as "the time this page refers to"
        #if tow is None:
        #    tow = r.get("t_0e", r.get("t_0c"))
        if wn is not None and tow is not None:
            r["utc"] = gst_to_utc(wn, tow, leap=leap)
 
    return results
 
_RELEVANT_KEYS = {
    0: ["time_flag", "WN", "TOW"],
    1: ["IODnav", "t_0e"],
    2: ["IODnav"],
    3: ["IODnav"],
    4: ["IODnav", "SVID", "t_0c"],
    5: ["WN", "TOW"],
    6: ["A0", "A1", "dtLS", "t_0t", "WN_0t", "WN_LSF", "DN", "dtLSF", "TOW"],
}
_WT_NAME = {
    0: "Spare/Time",
    1: "Ephemeris 1/4",
    2: "Ephemeris 2/4",
    3: "Ephemeris 3/4",
    4: "Ephemeris 4/4 (SV clock)",
    5: "Iono/BGD/Health + GST",
    6: "GST-UTC conversion",
}

def make_analytic_blocks_from_real(yhist, N_k, num_accum):
    """
    Convert real yhist into analytic (complex) blocks using FFT-Hilbert.
    Returns X of shape (num_accum, N_k), complex.
    """
    yhist = np.asarray(yhist)
    needed = num_accum * N_k
    if yhist.shape[0] < needed:
        raise ValueError(f"yhist must have at least {needed} samples, got {yhist.shape[0]}")

    Xr = yhist[:needed].reshape(num_accum, N_k).astype(np.float64, copy=False)
    Xf = np.fft.fft(Xr, axis=1)

    h = np.zeros(N_k, dtype=np.float64)
    h[0] = 1.0
    if N_k % 2 == 0:
        h[N_k // 2] = 1.0
        h[1:N_k // 2] = 2.0
    else:
        h[1:(N_k + 1) // 2] = 2.0

    return np.fft.ifft(Xf * h[None, :], axis=1)

def analytic_signal_fft_1d(x: np.ndarray) -> np.ndarray:
    """
    Analytic signal via FFT-Hilbert, 1D.
    x: real-valued array (N,)
    returns: complex analytic signal (N,)
    """
    x = np.asarray(x)
    #x = x.astype(np.float64, copy=False)
    x = x.astype(np.float64)
    N = x.size

    X = np.fft.fft(x)

    h = np.zeros(N, dtype=np.float64)
    h[0] = 1.0
    if N % 2 == 0:
        h[N // 2] = 1.0
        h[1:N // 2] = 2.0
    else:
        h[1:(N + 1) // 2] = 2.0

    return np.fft.ifft(X * h)

#@njit(cache=True)
#def oversample_prn_code_old(code_chips: np.ndarray,
#                       sample_rate: float,
#                       chip_rate: float,
#                       code_length_samples: float | int,
#                       start_chip_phase=0.0,
#                       code_rate=0.0
#                       ) -> np.ndarray:
#    """
#    Oversample a PRN code (one full period in chips) to sample-rate using ZOH.
#    
#    code_chips: length Nchips array, values e.g. +/-1
#    sample_rate: fs [Hz]
#    chip_rate: fc [chips/s]
#    code_length_samples: number of samples to generate for one full code period
#                        (can be float; will be rounded to int)
#    start_chip_phase: optional fractional chip offset (in chips) at sample 0
#                      e.g. 0.0 means aligned to chip boundary.
#    code_rate: the drift of the code alignment (chips/sec)
#    """
#    code_chips = np.asarray(code_chips)
#    Nchips = code_chips.size
#
#    Ns = int(np.round(code_length_samples))
#
#    # chip time of each sample in "chips" (can be fractional)
#    chip_time = (np.arange(Ns) * ((chip_rate+code_rate) / sample_rate)) + start_chip_phase 
#
#    # map sample -> chip index
#    chip_idx = np.floor(chip_time).astype(np.int64) % Nchips
#    return code_chips[chip_idx].astype(np.float32)

@njit(cache=True)
def oversample_prn_code(code_chips, sample_rate, chip_rate,
                        code_length_samples, start_chip_phase=0.0,
                        code_rate=0.0):
    """
    Oversample a PRN code (one full period in chips) to sample-rate using ZOH.
    
    code_chips: length Nchips array, values e.g. +/-1
    sample_rate: fs [Hz]
    chip_rate: fc [chips/s]
    code_length_samples: number of samples to generate for one full code period
                        (can be float; will be rounded to int)
    start_chip_phase: optional fractional chip offset (in chips) at sample 0
                      e.g. 0.0 means aligned to chip boundary.
    code_rate: the drift of the code alignment (chips/sec)
    """
    Nchips = code_chips.shape[0]
    Ns = int(np.round(code_length_samples))
    out = np.empty(Ns, dtype=np.float32)
    rate = (chip_rate + code_rate) / sample_rate  
    for n in range(Ns):
        chip_phase = n * rate + start_chip_phase
        idx = int(np.floor(chip_phase)) % Nchips
        out[n] = code_chips[idx]
    return out

#def oversample_prn_code_check(code_chips: np.ndarray,
#                        sample_rate: float,
#                        chip_rate: float,
#                        code_length_samples: float,
#                        start_chip_phase: float = 0.0,
#                        code_rate: float = 0.0) -> np.ndarray:
#    """
#    Numba-safe oversample PRN code:
#      - ZOH sampling of chips on the sample grid
#      - then apply a 4-tap fractional-delay filter in the *sample domain*
#        with circular indexing (no concatenate).
#
#    Notes:
#      - Fractional delay mu is treated as constant over the block.
#      - code_rate is handled in the chip_time mapping (ZOH stage).
#    """
#    Nchips = code_chips.size
#    Ns = int(np.round(code_length_samples))
#
#    # Requested chip phase -> samples
#    s_phase = start_chip_phase * (sample_rate / chip_rate)
#    s_int = floor(s_phase)
#    mu = s_phase - s_int  # in [0,1) typically (can be 0)
#
#    # Remove fractional-sample part from chip phase for ZOH stage
#    start_chip_phase_int = start_chip_phase - mu * (chip_rate / sample_rate)
#
#    # --- ZOH replica on sample grid ---
#    x = np.empty(Ns, dtype=np.float32)
#    step = chip_rate / sample_rate
#    drift = code_rate / sample_rate  # chips per sample
#
#    # chip_time[n] = n*step + start_chip_phase_int + code_rate*n/fs
#    #            = start_chip_phase_int + n*(step + drift)
#    inc = step + drift
#    chip_t = start_chip_phase_int
#
#    for n in range(Ns):
#        i0 = floor(chip_t)
#        idx = i0 % Nchips
#        x[n] = code_chips[idx]
#        chip_t += inc
#
#    # If no fractional delay, return ZOH replica
#    if mu == 0.0:
#        return x
#
#    # --- 4-tap Lagrange coefficients for x[n+mu] from {x[n-1], x[n], x[n+1], x[n+2]} ---
#    # Same formulas as before, but scalar.
#    h_m1 = -mu * (1.0 - mu) * (2.0 - mu) / 6.0
#    h_0  =  (1.0 + mu) * (1.0 - mu) * (2.0 - mu) / 2.0
#    h_1  =  (1.0 + mu) * mu * (2.0 - mu) / 2.0
#    h_2  = -(1.0 + mu) * mu * (1.0 - mu) / 6.0
#
#    y = np.empty(Ns, dtype=np.float32)
#
#    for n in range(Ns):
#        xm1 = x[n - 1] if n > 0 else x[Ns - 1]
#        x0  = x[n]
#        xp1 = x[n + 1] if n + 1 < Ns else x[0]
#        xp2 = x[n + 2] if n + 2 < Ns else (x[1] if Ns > 1 else x[0])
#
#        y[n] = (h_m1 * xm1 + h_0 * x0 + h_1 * xp1 + h_2 * xp2)
#
#    return y

def shift_code_phase(code_chips, sample_rate, chip_rate, start_chip_phase):
    """
    Circularly shift x by shift_samples (can be fractional) using FFT.
    Positive shift moves x forward: y[n] = x[(n+shift) mod N].
    code_chips: length Nchips array, float values (non-ZoH)
    sample_rate: fs [Hz]
    chip_rate: fc [chips/s]
    start_chip_phase: fractional chip offset (in chips) at sample 0
                      e.g., 0.0 means aligned to chip boundary.
    """
    shift_samples = start_chip_phase*sample_rate/chip_rate
    N = code_chips.size
    k = np.fft.fftfreq(N)  # cycles/sample
    X_fft = np.fft.fft(code_chips)
    phase = np.exp(1j * 2*np.pi * k * shift_samples)
    return np.fft.ifft(X_fft * phase)

def quad_peak_1d_mag2(z, j):
    """Return (delta, y_peak) from a 3-point quadratic fit to |z|^2 around index j."""
    N = z.size
    jm = (j - 1) % N
    jp = (j + 1) % N

    y_m = np.abs(z[jm])**2
    y_0 = np.abs(z[j ])**2
    y_p = np.abs(z[jp])**2

    delta = 0.5 * (y_m - y_p) / (y_m - 2.0*y_0 + y_p) # quadratic fit 
    delta = float(np.clip(delta, -0.5, 0.5)) # keep it sane

    y_peak = y_0 - 0.25 * (y_m - y_p) * delta
    return delta, float(y_peak)

def acq_prn_fft(prn_code, X, T_a, f_IF, N_k, num_accum, f_acq, samples_per_chip, decimation, f_max, m0=0):
    """
    Find code phase and Doppler frequency
      - For each frequency hypothesis, pick code phase index j_k from the FIRST accumulation block
        via argmax(|zk|^2).
      - Then accumulate Z_k = sum_n |zk[n, j_k]|^2 across all accum blocks.

    Parameters
    ----------
    prn_code : array_like (complex or real), shape (N_k,)
        Local PRN code sequence over one code period (length N_k).
    X : array_like, shape (num_accum, N_k)
        analytic signal to search for PRN signal
    T_a : float
        Coherent integration time (seconds) for each FFT block.
    f_IF : float
        IF / bandpass center (Hz) added to each hypothesized Doppler.
    N_k : int
        FFT length / samples per coherent block.
    num_accum : int
        Number of coherent blocks to accumulate noncoherently.
    f_acq : float
        Initial coarse acquisition frequency (Hz). Used as center for fine search.
        Returned updated with best fine frequency. 
    samples_per_chip: int
        Number of samples in one chip of the spreading code
    decimation: int
        Frequency step decimation (10 for fine search, 1 for coarse search)
    f_max: float
        +/- limit of Doppler frequency search 

    Returns
    -------
    f_acq_best : float
        Best fine frequency hypothesis (Hz) (offset, not including f_IF).
    t_acq : float
        Best code phase time (seconds).
    C_N0 : float
        Carrier-to-noise ratio, dB-Hz
    """
    dt_step = T_a / N_k
    df_step = 1.0 / (T_a * decimation)
    f_vec = np.arange(-f_max, f_max + df_step/2.0, df_step) + f_acq

    C_k = np.asarray(prn_code)

    # PRN FFT once
    if C_k.ndim == 1:
        if C_k.shape[0] != N_k:
            raise ValueError(f"prn_code must have length N_k={N_k}, got {C_k.shape[0]}")
        C_r = np.fft.fft(C_k)
        n_code = 1
    elif C_k.ndim == 2:
        # ndim > 2: non-coherent integration of a >1 ms length code
        # note that m0 must be set to the correct 1 ms interval by, for example, an outer loop
        n_code = C_k.shape[0]
        if C_k.shape[1] != N_k:
            raise ValueError(f"C_k must have shape (n_code,{N_k}), got {C_k.shape}")
        C_r = np.fft.fft(C_k, axis=1)
        b = np.arange(num_accum)  # block indices
        slices_idx = (m0 + b) % n_code # (num_accum,)

    # Time vector for mixing
    t_vec = np.arange(N_k) * (T_a / N_k)

    Z_max = -np.inf
    t_acq = 0.0
    f_acq_best = float(f_acq)

    for idx, f_hyp in enumerate(f_vec):
        f_i = f_hyp + f_IF
        mixer = np.exp(-1j * 2.0 * np.pi * f_i * t_vec)          # (N_k,)
        X_mixed = X * mixer[None, :]                              # (num_accum, N_k)
        X_r = np.fft.fft(X_mixed, axis=1)                         # (num_accum, N_k)

        if n_code == 1:
            # As before
            ZK = np.fft.ifft(X_r * np.conj(C_r)[None, :], axis=1)  # (B, N_k)
        else:
            ZK = np.fft.ifft(X_r * np.conj(C_r[slices_idx, :]), axis=1)  # (B, N_k)

        P_map = np.sum(np.abs(ZK)**2, axis=0)   # (N_k,)
        j_k = int(np.argmax(P_map))
        Z_k = float(P_map[j_k])
        if Z_k > Z_max:
            Z_max = Z_k
            t_acq = -j_k * dt_step
            j_peak = j_k
            f_acq_best = float(f_hyp)
            Z_k_best = ZK.copy()

    # do sub-sample interpolation for winning hypothesis,
    # refine code phase estimate with weighted mean
    deltas_global = []
    weights = []
    Zmax_refined = 0.0
    for b in range(num_accum):
        z = Z_k_best[b]
        j_b = int(np.argmax(np.abs(z)**2))
        d_b, ypk = quad_peak_1d_mag2(z, j_b)
        x_b = (j_b + d_b) - j_peak
        deltas_global.append(x_b)
        weights.append(ypk)          # reasonable weight; or 1.0
        Zmax_refined += ypk
    deltas_global = np.asarray(deltas_global, float)
    weights = np.asarray(weights, float)
    delta_hat_global = float(np.sum(weights * deltas_global) / np.sum(weights))
    t_acq_refined = -(j_peak + delta_hat_global) * dt_step
    t_acq = t_acq_refined
    Z_max = Zmax_refined

    idx = np.arange(N_k)
    dist = np.minimum((idx - j_peak) % N_k, (j_peak - idx) % N_k)
    guard_chips = 2        # start with 2–5 chips
    guard = guard_chips * samples_per_chip
    keep = dist > guard
    P = np.abs(Z_k_best[:,keep])**2
    sigma_S2 = np.median(P) / np.log(2.0)
    Z_0 = num_accum*sigma_S2 
    C_N0 = 10 * np.log10((Z_max-Z_0)/(Z_0*T_a))

    return f_acq_best, t_acq, C_N0

def acq_aided(prn_code, X, T_a, f_IF, N_k, num_accum, f_acq, samples_per_chip, m0=0, plot=False):
    """
    Find code phase and Doppler frequency
      - For each frequency hypothesis, pick code phase index j_k from the FIRST accumulation block
        via argmax(|zk|^2).
      - Then accumulate Z_k = sum_n |zk[n, j_k]|^2 across all accum blocks.

    Parameters
    ----------
    prn_code : array_like (complex or real), shape (N_k,)
        Local PRN code sequence over one code period (length N_k).
    X : array_like, shape (num_accum, N_k)
        analytic signal to search for PRN signal
    T_a : float
        Coherent integration time (seconds) for each FFT block.
    f_IF : float
        IF / bandpass center (Hz) added to each hypothesized Doppler.
    N_k : int
        FFT length / samples per coherent block.
    num_accum : int
        Number of coherent blocks to accumulate noncoherently.
    f_acq : float
        Initial coarse acquisition frequency (Hz). Used as center for fine search.
        Returned updated with best fine frequency. 
    samples_per_chip: int
        Number of samples in one chip of the spreading code
    decimation: int
        Frequency step decimation (10 for fine search, 1 for coarse search)
    f_max: float
        +/- limit of Doppler frequency search 

    Returns
    -------
    t_acq : float
        Best code phase time (seconds).
    C_N0 : float
        Carrier-to-noise ratio, dB-Hz
    """
    dt_step = T_a / N_k

    C_k = np.asarray(prn_code)

    # PRN FFT once
    if C_k.ndim == 1:
        if C_k.shape[0] != N_k:
            raise ValueError(f"prn_code must have length N_k={N_k}, got {C_k.shape[0]}")
        C_r = np.fft.fft(C_k)
        n_code = 1
    elif C_k.ndim == 2:
        # ndim > 2: non-coherent integration of a >1 ms length code
        # note that m0 must be set to the correct 1 ms interval by, for example, an outer loop
        n_code = C_k.shape[0]
        if C_k.shape[1] != N_k:
            raise ValueError(f"C_k must have shape (n_code,{N_k}), got {C_k.shape}") 
        C_r = np.fft.fft(C_k, axis=1)
        b = np.arange(num_accum)  # block indices
        slices_idx = (m0 + b) % n_code # (num_accum,)

    # Time vector for mixing
    t_vec = np.arange(N_k) * (T_a / N_k)

    Z_max = -np.inf
    t_acq = 0.0
    f_acq_best = float(f_acq)

    f_i = f_acq + f_IF
    mixer = np.exp(-1j * 2.0 * np.pi * f_i * t_vec)          # (N_k,)
    X_mixed = X * mixer[None, :]                              # (num_accum, N_k)
    X_r = np.fft.fft(X_mixed, axis=1)                         # (num_accum, N_k)

    if n_code == 1:
        # As before
        ZK = np.fft.ifft(X_r * np.conj(C_r)[None, :], axis=1)  # (B, N_k)
    else:
        ZK = np.fft.ifft(X_r * np.conj(C_r[slices_idx, :]), axis=1)  # (B, N_k)

    P_map = np.sum(np.abs(ZK)**2, axis=0)   # (N_k,)
    j_peak = int(np.argmax(P_map))
    Z_max = float(P_map[j_peak])
    Z_k_best = ZK

    # do sub-sample interpolation,
    # refine code phase estimate with weighted mean
    deltas_global = []
    weights = []
    Zmax_refined = 0.0
    for b in range(num_accum):
        z = Z_k_best[b]
        j_b = int(np.argmax(np.abs(z)**2))
        d_b, ypk = quad_peak_1d_mag2(z, j_b)
        x_b = (j_b + d_b) - j_peak
        deltas_global.append(x_b)
        weights.append(ypk)          # reasonable weight; or 1.0
        Zmax_refined += ypk
    deltas_global = np.asarray(deltas_global, float)
    idxs_good = np.abs(deltas_global)<1
    if np.sum(idxs_good)>2:
        deltas_global = deltas_global[idxs_good] # exclude outliers
        weights = np.asarray(weights, float)[idxs_good]
        delta_hat_global = float(np.sum(weights * deltas_global) / np.sum(weights))
    else:
        delta_hat_global=0
    t_acq_refined = -(j_peak + delta_hat_global) * dt_step
    t_acq = t_acq_refined
    Z_max = Zmax_refined

    idx = np.arange(N_k)
    dist = np.minimum((idx - j_peak) % N_k, (j_peak - idx) % N_k)
    guard_chips = 2        # start with 2–5 chips
    guard = guard_chips * samples_per_chip
    keep = dist > guard
    P = np.abs(Z_k_best[:,keep])**2
    sigma_S2 = np.median(P) / np.log(2.0)
    Z_0 = num_accum*sigma_S2 
    C_N0 = 10 * np.log10((Z_max-Z_0)/(Z_0*T_a))
   
    plot = False
    if plot: 
        import matplotlib.pyplot as plt
        # Code phase axis in milliseconds.
        # Bin j corresponds to delay -j * dt_step; wrap to (-T_a/2, T_a/2] for a centered plot,
        # or just use 0..T_a. Here we use 0..T_a in ms for simplicity.
        tau_ms = np.arange(N_k) * dt_step * 1e3  # milliseconds
        # Per-bin C/N0, masking bins where P <= Z_0 (pure noise fluctuations below the mean)
        excess = P_map - Z_0
        CN0_axis = np.full_like(P_map, np.nan, dtype=float)
        pos = excess > 0
        CN0_axis[pos] = 10.0 * np.log10(excess[pos] / (Z_0 * T_a))
        
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(tau_ms, CN0_axis, lw=0.8)
        ax.axvline((-t_acq) * 1e3, color='r', ls='--', lw=1,
                   label=f'peak @ {(-t_acq)*1e3:.4f} ms, {C_N0:.1f} dB-Hz')
        ax.set_xlabel('Code phase (ms)')
        ax.set_ylabel(r'C/N$_0$ (dB-Hz)')
        ax.set_title('CAF slice along code-phase axis')
        ax.set_ylim(-10, max(C_N0 + 5, 40))   # noise floor ~0, headroom above peak
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        fig.tight_layout()
        fig.savefig('caf_pr_axis.png', dpi=150)
        plt.close(fig)


    return t_acq, C_N0

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

def compute_Gd_numeric_coherent(X_blocks, t_blk, C_k, sample_rate, chip_rate, code_period, f_IF, t_0, t_dot, phi_0, f_D, f_D_dot, eml, N_k,
    num_blocks=5, # number of blocks to use in estimating Gd
    delta_eps=0.25,          # perturbation in "code phase" units (chips if your t_0 is chips)
    w_floor_frac=0.05,       # gate low-SNR blocks
    eps_den=1e-12
):
    """
    Numerically estimate discriminator gain G_d for the coherent dot-product DLL discriminator:
        d = Re{(E-L) P*} / |P|^2

    using symmetric perturbations epsilon = +/- delta_eps around the current prompt code phase.

    Returns
    -------
    Gd : float
        Estimated slope dd/depsilon near lock (units: 1/(code phase unit))
    info : dict
        Diagnostics (optional)
    """
    # Time vector per sample for each block (your convention)
    # dt is absolute time since t0 for phase model
    n = np.arange(N_k)
    dt_inblock = n * (code_period / N_k)

    d_plus = np.empty(num_blocks, dtype=float)
    d_minus = np.empty(num_blocks, dtype=float)
    w = np.empty(num_blocks, dtype=float)
    P_arr_p = []
    E_arr_p = []
    L_arr_p = []
    P_arr_m = []
    E_arr_m = []
    L_arr_m = []

    for block in range(num_blocks):
        # Prompt code phase at middle of block interval (your convention)
        code_phase_prompt = t_0 + t_dot * t_blk[block]

        # Absolute time for samples in this block
        dt = block * code_period + dt_inblock

        phase = phi_0 + 2*np.pi * ((f_IF + f_D) * dt + f_D_dot * dt**2)
        mixer = np.exp(-1j * phase)
        X_block = X_blocks[block, :] * mixer
        samples_per_chip = sample_rate/chip_rate
        code_phase_samples = code_phase_prompt * samples_per_chip
        sample_boundary = int(np.floor(code_phase_samples))

        # Helper: compute d at a given prompt phase offset
        def _d_at(code_phase_prompt_offset):
            # Build codes at prompt/early/late
            code_frac_prompt = code_phase_prompt_offset
            C_p = shift_code_phase(C_k, sample_rate, chip_rate, code_frac_prompt)

            code_frac_early = code_phase_prompt_offset - eml/2
            C_e = shift_code_phase(C_k, sample_rate, chip_rate, code_frac_early)

            code_frac_late  = code_phase_prompt_offset + eml/2
            C_l = shift_code_phase(C_k, sample_rate, chip_rate, code_frac_late)

            #P = np.vdot(C_p, X_block)
            #E = np.vdot(C_e, X_block)
            #L = np.vdot(C_l, X_block)
            P_1 = np.vdot(C_p[:sample_boundary], X_block[:sample_boundary])
            P_2 = np.vdot(C_p[sample_boundary:], X_block[sample_boundary:])
            E_1 = np.vdot(C_e[:sample_boundary], X_block[:sample_boundary])
            E_2 = np.vdot(C_e[sample_boundary:], X_block[sample_boundary:])
            L_1 = np.vdot(C_l[:sample_boundary], X_block[:sample_boundary])
            L_2 = np.vdot(C_l[sample_boundary:], X_block[sample_boundary:])
            return P_1, P_2, E_1, E_2, L_1, L_2 

        # Compute d at +/- delta_eps perturbations
        #d_p, P2_p = _d_at(code_phase_prompt + delta_eps)
        #d_m, P2_m = _d_at(code_phase_prompt - delta_eps)
        P_1_p, P_2_p, E_1_p, E_2_p, L_1_p, L_2_p = _d_at(code_phase_prompt + delta_eps)
        P_1_m, P_2_m, E_1_m, E_2_m, L_1_m, L_2_m = _d_at(code_phase_prompt - delta_eps)
        P_arr_p.append(P_1_p)
        P_arr_p.append(P_2_p)
        E_arr_p.append(E_1_p)
        E_arr_p.append(E_2_p)
        L_arr_p.append(L_1_p)
        L_arr_p.append(L_2_p)

        P_arr_m.append(P_1_m)
        P_arr_m.append(P_2_m)
        E_arr_m.append(E_1_m)
        E_arr_m.append(E_2_m)
        L_arr_m.append(L_1_m)
        L_arr_m.append(L_2_m)

        # Weight: use prompt power (average of the two)
        #w[block] = 0.5 * (P2_p + P2_m)
   
    P_arr_p = pair_sum(np.array(P_arr_p))
    E_arr_p = pair_sum(np.array(E_arr_p))
    L_arr_p = pair_sum(np.array(L_arr_p))
    P_arr_m = pair_sum(np.array(P_arr_m))
    E_arr_m = pair_sum(np.array(E_arr_m))
    L_arr_m = pair_sum(np.array(L_arr_m))
    
    P2_p = (P_arr_p.real*P_arr_p.real + P_arr_p.imag*P_arr_p.imag)
    d_plus = np.real((E_arr_p - L_arr_p) * np.conjugate(P_arr_p)) / (P2_p + eps_den)

    P2_m = (P_arr_m.real*P_arr_m.real + P_arr_m.imag*P_arr_m.imag)
    d_minus = np.real((E_arr_m - L_arr_m) * np.conjugate(P_arr_m)) / (P2_m + eps_den)
    w = 0.5 * (P2_p + P2_m)
    # Gate out weak blocks 
    w_med = np.median(w)
    keep = w > (w_floor_frac * w_med)

    # Finite-difference slope per block
    # d(+δ) - d(-δ) ≈ 2δ * Gd
    slope_k = np.abs(d_plus[keep] - d_minus[keep]) / (2.0 * delta_eps)

    # Robust combine: weighted median (very stable)
    # or weighted mean; I'll do weighted mean with clipping:
    wk = w[keep]
    # clip extreme slope outliers
    s_med = np.median(slope_k)
    mad = np.median(np.abs(slope_k - s_med)) + 1e-15
    clip = 6.0 * mad
    good = np.abs(slope_k - s_med) < clip

    slope_k = slope_k[good]
    wk = wk[good]

    Gd = float(np.sum(wk * slope_k) / np.sum(wk)) 
    return Gd


def compute_Gd_numeric_noncoherent(X_blocks, t_blk, prn_code, code_length_samples, sample_rate, chip_rate, code_period, f_IF, t_0, t_dot, phi_0, f_D, f_D_dot, eml, N_k,
    num_blocks=5, # number of blocks to use in estimating Gd
    delta_eps=0.05,          # perturbation in "code phase" units (chips if your t_0 is chips)
    w_floor_frac=0.05,       # gate low-SNR blocks
    eps_den=1e-12
):
    """
    Numerically estimate discriminator gain G_d for the coherent dot-product DLL discriminator:
        d = Re{(E-L) P*} / |P|^2

    using symmetric perturbations epsilon = +/- delta_eps around the current prompt code phase.

    Returns
    -------
    Gd : float
        Estimated slope dd/depsilon near lock (units: 1/(code phase unit))
    info : dict
        Diagnostics (optional)
    """
    # Time vector per sample for each block (your convention)
    # dt is absolute time since t0 for phase model
    n = np.arange(N_k)
    dt_inblock = n * (code_period / N_k)

    d_plus = np.empty(num_blocks, dtype=float)
    d_minus = np.empty(num_blocks, dtype=float)
    w = np.empty(num_blocks, dtype=float)

    for block in range(num_blocks):
        # Prompt code phase at middle of block interval (your convention)
        code_phase_prompt = t_0 + t_dot * t_blk[block]

        # Absolute time for samples in this block
        dt = block * code_period + dt_inblock

        phase = phi_0 + 2*np.pi * ((f_IF + f_D) * dt + 0.5*f_D_dot * dt**2)
        mixer = np.exp(-1j * phase)
        X_block = X_blocks[block, :] * mixer

        # Helper: compute d at a given prompt phase offset
        def _d_at(code_phase_prompt_offset):
            # Build codes at prompt/early/late
            code_frac_prompt = code_phase_prompt_offset
            C_p = oversample_prn_code(prn_code, sample_rate, chip_rate, code_length_samples, code_frac_prompt)

            code_frac_early = code_phase_prompt_offset - eml/2
            C_e = oversample_prn_code(prn_code, sample_rate, chip_rate, code_length_samples, code_frac_early)

            code_frac_late  = code_phase_prompt_offset + eml/2
            C_l = oversample_prn_code(prn_code, sample_rate, chip_rate, code_length_samples, code_frac_late)

            P = np.vdot(C_p, X_block)
            E = np.vdot(C_e, X_block)
            L = np.vdot(C_l, X_block)

            P2 = (P.real*P.real + P.imag*P.imag)
            # coherent dot-product discriminator
            E_abs = np.abs(E)
            L_abs = np.abs(L)
            d = (E_abs - L_abs) / (E_abs + L_abs)
            return d, P2

        # Compute d at +/- delta_eps perturbations
        d_p, P2_p = _d_at(code_phase_prompt + delta_eps)
        d_m, P2_m = _d_at(code_phase_prompt - delta_eps)

        d_plus[block] = d_p
        d_minus[block] = d_m

        # Weight: use prompt power (average of the two)
        w[block] = 0.5 * (P2_p + P2_m)

    # Gate out weak blocks 
    w_med = np.median(w)
    keep = w > (w_floor_frac * w_med)

    # Finite-difference slope per block
    # d(+δ) - d(-δ) ≈ 2δ * Gd
    slope_k = (d_plus[keep] - d_minus[keep]) / (2.0 * delta_eps)

    # Robust combine: weighted median (very stable)
    # or weighted mean; I'll do weighted mean with clipping:
    wk = w[keep]
    # clip extreme slope outliers
    s_med = np.median(slope_k)
    mad = np.median(np.abs(slope_k - s_med)) + 1e-15
    clip = 6.0 * mad
    good = np.abs(slope_k - s_med) < clip

    slope_k = slope_k[good]
    wk = wk[good]

    Gd = float(np.sum(wk * slope_k) / np.sum(wk))

    return Gd

@dataclass(frozen=True)
class S1Params:
    poly_octal: str   # 4 octal digits -> 12-bit mask, MSB is degree 11, LSB is degree 0
    init_octal: str   # 4 octal digits -> 12 bits MSB..LSB, drop MSB (should be 0) => 11-stage state

def _octal_to_int(o: str) -> int:
    return int(o, 8)

def _octal_init_to_state11(init_octal_4digits: str) -> List[int]:
    """
    Parse 4-octal-digit init into 12 bits MSB..LSB.
    Drop the MSB bit (ICD says it's 0). Remaining 11 bits are n11..n1 (MSB..LSB).
    Return state as [n1, n2, ..., n11] so state[-1] is output (n11).
    """
    x = _octal_to_int(init_octal_4digits)
    bits12 = [(x >> (11 - i)) & 1 for i in range(12)]  # MSB..LSB
    bits11 = bits12[1:]                                 # drop MSB (leading 0)
    n11_to_n1 = bits11                                  # MSB..LSB
    return list(reversed(n11_to_n1))                    # [n1..n11]

def _poly_octal_to_taps(poly_octal_4digits: str) -> List[int]:
    """
    Convert 4-octal-digit polynomial mask to tap degrees.
    Convention: 12 bits MSB..LSB correspond to degrees 11..0.
    Return degrees in 1..11 (exclude degree 0 constant term).
    """
    mask = _octal_to_int(poly_octal_4digits)
    bits12 = [(mask >> (11 - i)) & 1 for i in range(12)]  # MSB..LSB
    degs = [11 - i for i, b in enumerate(bits12) if b == 1]  # degrees 11..0
    return [d for d in degs if d != 0]  # exclude constant term

def l1cp_overlay_s1(prn: int, n_chips: int = 1800) -> np.ndarray:
    """
    GPS L1C-P overlay S1 code (chips in {+1, -1} GNSS convention: 0->+1, 1->-1).
    """
    params = L1CO_S1_PARAMS[prn]
    taps = _poly_octal_to_taps(params.poly_octal)
    state = _octal_init_to_state11(params.init_octal)  # [n1..n11], output n11 is state[10]

    out01: List[int] = []
    for _ in range(n_chips):
        out_bit = state[10]
        out01.append(out_bit)

        fb = 0
        for d in taps:
            # degree d corresponds to stage d (n_d) -> index d-1 in [n1..n11]
            if 1 <= d <= 11:
                fb ^= state[d - 1]
            else:
                raise ValueError(f"Unexpected tap degree {d}")

        # shift toward output: new bit enters n1
        state = [fb] + state[:-1]

    out01 = np.asarray(out01, dtype=np.uint8)
    chips_pm1 = (1 - 2 * out01).astype(np.int8)  # 0->+1, 1->-1
    return np.array(chips_pm1)

# PRN -> (S1 polynomial, S1 initial state), IS-GPS-800 Table 3.2-3 
L1CO_S1_PARAMS: Dict[int, S1Params] = {
    1:  S1Params("5111", "3266"),
    2:  S1Params("5421", "2040"),
    3:  S1Params("5501", "1527"),
    4:  S1Params("5403", "3307"),
    5:  S1Params("6417", "3756"),
    6:  S1Params("6141", "3026"),
    7:  S1Params("6351", "0562"),
    8:  S1Params("6501", "0420"),
    9:  S1Params("6205", "3415"),
    10: S1Params("6235", "0337"),
    11: S1Params("7751", "0265"),
    12: S1Params("6623", "1230"),
    13: S1Params("6733", "2204"),
    14: S1Params("7627", "1440"),
    15: S1Params("5667", "2412"),
    16: S1Params("5051", "3516"),
    17: S1Params("7665", "2761"),
    18: S1Params("6325", "3750"),
    19: S1Params("4365", "2701"),
    20: S1Params("4745", "1206"),
    21: S1Params("7633", "1544"),
    22: S1Params("6747", "1774"),
    23: S1Params("4475", "0546"),
    24: S1Params("4225", "2213"),
    25: S1Params("7063", "3707"),
    26: S1Params("4423", "2051"),
    27: S1Params("6651", "3650"),
    28: S1Params("4161", "1777"),
    29: S1Params("7237", "3203"),
    30: S1Params("4473", "1762"),
    31: S1Params("5477", "2100"),
    32: S1Params("6163", "0571"),
    33: S1Params("7223", "3710"),
    34: S1Params("6323", "3535"),
    35: S1Params("7125", "3110"),
    36: S1Params("7035", "1426"),
    37: S1Params("4341", "0255"),
    38: S1Params("4353", "0321"),
    39: S1Params("4107", "3124"),
    40: S1Params("5735", "0572"),
    41: S1Params("6741", "1736"),
    42: S1Params("7071", "3306"),
    43: S1Params("4563", "1307"),
    44: S1Params("5755", "3763"),
    45: S1Params("6127", "1604"),
    46: S1Params("4671", "1021"),
    47: S1Params("4511", "2624"),
    48: S1Params("4533", "0406"),
    49: S1Params("5357", "0114"),
    50: S1Params("5607", "0077"),
    51: S1Params("6673", "3477"),
    52: S1Params("6153", "1000"),
    53: S1Params("7565", "3460"),
    54: S1Params("7107", "2607"),
    55: S1Params("6211", "2057"),
    56: S1Params("4321", "3467"),
    57: S1Params("7201", "0706"),
    58: S1Params("4451", "2032"),
    59: S1Params("5411", "1464"),
    60: S1Params("5141", "0520"),
    61: S1Params("7041", "1766"),
    62: S1Params("6637", "3270"),
    63: S1Params("4577", "0341"),
}
def weil_code_from_wp(w, p, N, L) -> np.ndarray:
    """
    Generate a truncated Weil sequence-based code from parameters (w, p).

    This function implements the standard "Legendre -> Weil -> truncation" pattern used by
    several modern GNSS ranging/secondary codes:
      1) Form Legendre sequence χ[n] over the prime modulus N (n = 0..N-1), where
         χ[0]=0, χ[n]=+1 if n is a quadratic residue mod N, and χ[n]=-1 otherwise.
      2) Form Weil sequence W_w[n] = χ[n] * χ[(n + w) mod N]
      3) Truncate (circularly) starting at index p to length L:
         c[k] = W_w[(p + k) mod N], k=0..L-1

    Notes:
      • N must be an odd prime for Legendre symbols to be well-defined as used here.
      • Many ICDs use 1-indexing for p (and sometimes for w). This function assumes:
            - w is already in the ICD's integer domain (typically 0 < w < N)
            - p is 1-indexed by default (one_indexed_p=True); set False if p is 0-indexed.
      • Different specs may define χ[0] differently (0 vs +1) and/or use complements.
        If your verification chips don't match, this is the first knob to check.

    Parameters
    ----------
    w : int
        Weil shift/phase parameter.
    p : int
        Truncation start index.
    N : int
        Prime modulus / base sequence length.
    L : int
        Output code length (chips).

    Returns
    -------
    code : np.ndarray, shape (L,), dtype=np.int8
        The generated code sequence.
    """
    if N < 3 or N % 2 == 0:
        raise ValueError("N must be an odd prime (>=3).")
    if not (0 <= w < N):
        raise ValueError(f"w must satisfy 0 <= w < N (got w={w}, N={N}).")

    # Convert p to 0-indexed start
    p0 = (p - 1) 
    p0 %= N

    # Build Legendre sequence χ[n] for n=0..N-1:
    # χ[0]=0; for n>0: χ[n]=+1 if n is a quadratic residue mod N else -1
    chi = np.empty(N, dtype=np.int8)
    chi[0] = 1

    # Mark quadratic residues mod N
    residues = np.zeros(N, dtype=bool)
    # For prime N, residues are (k^2 mod N) for k=1..(N-1)/2
    for k in range(1, (N // 2) + 1):
        residues[(k * k) % N] = True

    chi[1:] = np.where(residues[1:], 1, -1).astype(np.int8)

    # Weil sequence: W_w[n] = χ[n] * χ[(n+w) mod N]
    chi_shift = np.roll(chi, -w)  # chi[(n+w) mod N] aligns as roll by -w
    W = (chi * chi_shift).astype(np.int8)

    # Truncate (circular) starting at p0 to length L
    idx = (p0 + np.arange(L, dtype=np.int64)) % N

    return W[idx]


def bits_to_octal(bits):
    if len(bits) % 3 != 0:
        raise ValueError("Bit length must be a multiple of 3 for octal conversion")

    bits = bits.reshape(-1, 3)
    octal = (bits[:,0] << 2) | (bits[:,1] << 1) | bits[:,2]
    return octal


def cs_hex_to_pm1(hex_str: str, n_bits: int) -> np.ndarray:
    """
    Convert a Galileo CS (CS25, CS100, etc.) given as hex into GNSS ±1 chips.

    Assumes:
      - Hex is MSB-first
      - 0 -> +1, 1 -> -1  (GNSS convention)

    Parameters
    ----------
    hex_str : str
        Hexadecimal string from ICD (no 0x required)
    n_bits : int
        Number of bits in the CS code (e.g. 25 or 100)

    Returns
    -------
    chips_pm1 : np.ndarray of shape (n_bits,), dtype=int8
    """
    s = hex_str.strip().lower().replace("0x", "").replace(" ", "")

    if len(s) * 4 < n_bits:
        raise ValueError("Hex string too short for requested bit length")

    # Pad to full bytes
    if len(s) % 2 == 1:
        s = "0" + s          # pad one nibble (4 bits)
        pad_bits = 4
    else:
        pad_bits = 0

    # Hex -> bytes -> bits (MSB first)
    b = bytes.fromhex(s)
    bits01 = np.unpackbits(
        np.frombuffer(b, dtype=np.uint8),
        bitorder="big"
    )

    # Remove pad bits and trim to n_bits
    bits01 = bits01[pad_bits:pad_bits + n_bits]

    # GNSS mapping: 0 -> +1, 1 -> -1
    chips_pm1 = (1 - 2 * bits01).astype(np.int8)

    return chips_pm1


def _parity(x: int) -> int:
    return x.bit_count() & 1

def _build_trellis(polys=(0o171, 0o133), K=7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build trellis for rate-1/2 convolutional code with constraint length K.

    State is (K-1)-bit shift register. Input bit u in {0,1}.
    Next state: ((state << 1) | u) & ((1<<(K-1)) - 1)
    Output bits: parity(regword & g1), parity(regword & g2)
      where regword = (state << 1) | u  (K bits, with u as LSB)
    """
    m = K - 1
    n_states = 1 << m
    next_state = np.zeros((n_states, 2), dtype=np.int16)
    out_bits = np.zeros((n_states, 2, 2), dtype=np.uint8)  # [s,u] -> [b0,b1]
    g1, g2 = polys

    for s in range(n_states):
        for u in (0, 1):
            regword = (s << 1) | u
            b0 = _parity(regword & g1)
            b1 = _parity(regword & g2)
            ns = ((s << 1) | u) & (n_states - 1)
            next_state[s, u] = ns
            out_bits[s, u, 0] = b0
            out_bits[s, u, 1] = b1

    return next_state, out_bits

def viterbi_metric_hard(r01: np.ndarray, polys=(0o171, 0o133), K=7) -> float:
    """
    Compute minimal Viterbi path metric for a hard-decision received stream r01 (0/1),
    assumed to be rate-1/2 (pairs of bits per trellis step).

    Metric: Hamming distance between received pair and expected pair.

    Unknown starting state handled by initializing all states with 0 metric.
    """
    r01 = np.asarray(r01, dtype=np.uint8).reshape(-1)
    if r01.size % 2 != 0:
        raise ValueError("r01 length must be even (rate-1/2: 2 bits per step).")

    next_state, out_bits = _build_trellis(polys=polys, K=K)
    m = K - 1
    n_states = 1 << m
    n_steps = r01.size // 2

    # Initialize: unknown start -> all states equally likely (metric 0)
    pm = np.zeros(n_states, dtype=np.float64)
    pm_new = np.empty_like(pm)

    for t in range(n_steps):
        r0 = r01[2*t]
        r1 = r01[2*t + 1]

        pm_new.fill(np.inf)
        # brute-force over states and inputs (minimal and clear, fast enough for small windows)
        for s in range(n_states):
            base = pm[s]
            # u=0
            ns = next_state[s, 0]
            b0, b1 = out_bits[s, 0]
            d = (b0 ^ r0) + (b1 ^ r1)
            if base + d < pm_new[ns]:
                pm_new[ns] = base + d
            # u=1
            ns = next_state[s, 1]
            b0, b1 = out_bits[s, 1]
            d = (b0 ^ r0) + (b1 ^ r1)
            if base + d < pm_new[ns]:
                pm_new[ns] = base + d

        pm, pm_new = pm_new, pm

    return float(pm.min())

def pick_polarity_viterbi_hard(r01: np.ndarray, **kwargs) -> int:
    """
    Return +1 if r01 is the correct polarity, -1 if inverted polarity fits better.
    """
    r01 = np.asarray(r01, dtype=np.uint8).reshape(-1)
    m1 = viterbi_metric_hard(r01, **kwargs)
    m2 = viterbi_metric_hard(1 - r01, **kwargs)  # invert bits for opposite polarity
    return +1 if m1 <= m2 else -1
